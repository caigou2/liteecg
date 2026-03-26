# ptb_dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.signal import butter, sosfiltfilt, find_peaks
import random
from typing import Optional, Tuple, List, Dict
import wfdb
from utils import simple_baseline_remove, SEED

class PTB_LDCNN_Dataset(Dataset):
    """
    PTB 数据集（按记录/病人划分），默认采样率 360Hz，段时长 2.0s -> segment_len=720。
    构造参数支持直接传入 segment_len（优先）或用 segment_sec 与 fs 计算。
    """
    def __init__(self,
                 root: str,
                 split: str,
                 segment_sec: Optional[float] = 2.0,
                 fs: int = 360,
                 segment_len: Optional[int] = None,
                 lead_priority: Tuple[str, ...] = ("II", "MLII", "V5", "V2", "I"),
                 normalize: str = 'zscore',
                 class_limit: Optional[int] = None,
                 cache_path: Optional[str] = None,
                 seed: int = SEED):
        assert wfdb is not None, "wfdb is required. Please `pip install wfdb` and have PTB files locally."
        self.root = os.path.abspath(root)
        self.split = split.lower()
        self.fs = fs
        self.lead_priority = [lp.upper() for lp in lead_priority]
        self.normalize = normalize
        self.class_limit = class_limit
        self.cache_path = cache_path
        self.seed = seed

        # segment_len 优先，其次用 segment_sec * fs 计算
        if segment_len is not None:
            self.segment_len = int(segment_len)
        else:
            assert segment_sec is not None, "Must provide segment_len or segment_sec"
            self.segment_len = int(round(segment_sec * fs))

        # 尝试加载缓存
        if cache_path and os.path.exists(cache_path):
            print(f"[PTB_LDCNN] Loading cache: {cache_path}", flush=True)
            cache = np.load(cache_path, allow_pickle=True)
            self.X, self.y = cache['X'], cache['y']
            return

        # 扫描目录，按病人分组（使用完整 record path，不使用 relpath）
        patient_to_records: Dict[str, List[str]] = {}
        for subdir, _, files in os.walk(self.root):
            for f in files:
                if f.endswith('.dat'):
                    record_base = os.path.join(subdir, f[:-4])  # full path without extension
                    p_id = os.path.basename(subdir)
                    patient_to_records.setdefault(p_id, []).append(record_base)

        p_ids = sorted(list(patient_to_records.keys()))
        random.seed(self.seed)
        random.shuffle(p_ids)
        num_p = len(p_ids)
        if self.split == 'train':
            sel_p = p_ids[:int(0.7 * num_p)]
        elif self.split == 'val':
            sel_p = p_ids[int(0.7 * num_p):int(0.85 * num_p)]
        elif self.split == 'test':
            sel_p = p_ids[int(0.85 * num_p):]
        else:
            raise ValueError(f"Unknown split: {split}")

        record_list: List[str] = []
        for p in sel_p:
            record_list.extend(patient_to_records[p])

        print(f"[PTB_LDCNN] {self.split}: {len(sel_p)} patients, {len(record_list)} records, segment_len={self.segment_len}", flush=True)

        X_l: List[np.ndarray] = []
        y_l: List[int] = []
        fail_count = 0

        pbar = tqdm(record_list, desc=f"Processing {self.split}")
        for rec_path in pbar:
            try:
                sig, info = wfdb.rdsamp(rec_path)
            except Exception as e:
                # 读取失败直接跳过
                fail_count += 1
                continue

            names = [n.upper() for n in info['sig_name']]
            used_idx = 0
            for target in self.lead_priority:
                if target in names:
                    used_idx = names.index(target)
                    break

            ecg = sig[:, used_idx].astype(np.float32)
            # 基线移除（utils 提供）
            try:
                ecg = simple_baseline_remove(ecg, fs=self.fs)
            except Exception:
                # 若基线移除失败，保留原始信号
                pass

            # 标签解析（尽量鲁棒）
            comments = " ".join(info.get('comments', [])).upper()
            folder_name = os.path.basename(os.path.dirname(rec_path)).upper()
            if ('HEALTH' in comments) or ('NORMAL' in comments) or ('N' in comments and 'NOT' not in comments) or ('NORMAL' in folder_name) or ('HEALTH' in folder_name):
                label = 0
            else:
                label = 1

            # R 峰检测
            peaks = self._detect(ecg)
            if len(peaks) == 0:
                fail_count += 1
                continue

            L = self.segment_len
            half = L // 2
            for r in peaks:
                start = int(r - half)
                end = start + L
                if start < 0 or end > len(ecg):
                    pad_left = max(0, -start)
                    pad_right = max(0, end - len(ecg))
                    seg = ecg[max(0, start):min(len(ecg), end)]
                    if pad_left > 0 or pad_right > 0:
                        seg = np.pad(seg, (pad_left, pad_right), mode='reflect')
                else:
                    seg = ecg[start:end]

                if len(seg) != L:
                    continue

                if self.normalize == 'zscore':
                    m = seg.mean()
                    sd = seg.std() + 1e-6
                    seg = (seg - m) / sd
                elif self.normalize == 'minmax':
                    mn, mx = seg.min(), seg.max()
                    seg = (seg - mn) / (mx - mn + 1e-6)
                    seg = seg * 2 - 1

                X_l.append(seg.astype(np.float32))
                y_l.append(label)

        if fail_count > 0:
            print(f"[PTB_LDCNN] Warning: {fail_count} records failed or had no peaks.", flush=True)

        if not X_l:
            print("[PTB_LDCNN] Error: no heartbeat segments extracted.", flush=True)
            self.X = np.zeros((0, self.segment_len), dtype=np.float32)
            self.y = np.zeros((0,), dtype=np.int64)
        else:
            self.X = np.stack(X_l).astype(np.float32)
            self.y = np.array(y_l).astype(np.int64)

            # 训练集类平衡限制
            if self.split == 'train' and self.class_limit:
                self._balance(self.class_limit)

        print(f"[PTB_LDCNN] {self.split} built, total beats: {len(self.y)}, X.shape={self.X.shape}", flush=True)
        if self.cache_path and len(self.y) > 0:
            np.savez_compressed(self.cache_path, X=self.X, y=self.y)
            print(f"[PTB_LDCNN] Saved cache to {self.cache_path}", flush=True)

    def _detect(self, sig: np.ndarray) -> np.ndarray:
        """改进后的 R 峰检测，使用 sosfiltfilt 做带通（更稳健）"""
        # 1. 5-30 Hz 带通
        sos = butter(3, [5, 30], btype='bandpass', fs=self.fs, output='sos')
        try:
            f = sosfiltfilt(sos, sig)
        except Exception:
            # 若 sosfiltfilt 出错，回退到 ba 形式的 filtfilt
            b, a = butter(3, [5, 30], btype='bandpass', fs=self.fs, output='ba')
            from scipy.signal import filtfilt
            f = filtfilt(b, a, sig)

        # 2. 能量增强
        energy = f ** 2

        # 3. 动态阈值
        height_thr = np.percentile(energy, 95) * 0.3

        # 4. 寻峰，min distance 根据 fs 动态设定（这里 0.4s）
        min_distance = max(1, int(0.4 * self.fs))
        peaks, _ = find_peaks(energy, distance=min_distance, height=height_thr)
        return peaks

    def _balance(self, limit: int):
        i0 = np.where(self.y == 0)[0]
        i1 = np.where(self.y == 1)[0]
        if len(i0) == 0 or len(i1) == 0:
            return
        cnt = min(len(i0), len(i1), limit)
        np.random.seed(self.seed)
        s0 = np.random.choice(i0, cnt, replace=False)
        s1 = np.random.choice(i1, cnt, replace=False)
        idx = np.concatenate([s0, s1])
        np.random.shuffle(idx)
        self.X = self.X[idx]
        self.y = self.y[idx]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        y = self.y[i]
        x = torch.from_numpy(x).unsqueeze(0).float()  # [1, T]
        y = torch.tensor(y).long()
        return x, y