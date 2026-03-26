#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG Lightweight Anomaly Detector - Stratified Dataset
------------------------------------------
Sample-level stratified dataset splitting
"""

from ast import main
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from utils import symbol_to_class, simple_baseline_remove
import wfdb
class StratifiedECGSegments(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 segment_sec: float = 2.0,
                 fs: int = 360,
                 prefer_leads: Tuple[str,...] = ("MLII","V1","V2","V5","II"),
                 normalize: str = 'zscore',
                 baseline_rm: bool = True,
                 class_limit: Optional[int] = None,
                 cache_path: Optional[str] = None,
                 random_state: int = 42):
        """
        root: path to MIT-BIH directory
        split: 'train'|'val'|'test'
        class_limit: cap per-class sample count to mitigate imbalance
        cache_path: .npz path for caching preprocessed segments
        random_state: random seed for train/val/test split
        """
        assert wfdb is not None, "wfdb is required. Please `pip install wfdb` and have MIT-BIH files locally."
        self.root = root
        self.segment_sec = segment_sec
        self.fs = fs
        self.segment_len = int(round(segment_sec * fs))
        self.prefer_leads = prefer_leads
        self.normalize = normalize
        self.baseline_rm = baseline_rm
        self.class_limit = class_limit
        self.split = split
        self.random_state = random_state

        self.X: np.ndarray
        self.y: np.ndarray
       
        if cache_path and os.path.exists(cache_path):
            print(f"[StratifiedECGSegments] Loading cache: {cache_path}")
            cache = np.load(cache_path, allow_pickle=True)
            self.X = cache['X']
            self.y = cache['y']
            return

        # 1. Collect all samples
        X_list, y_list = [], []
        per_class_counter: Dict[int,int] = {}

        # Get all record files
        all_records = [f[:-4] for f in os.listdir(self.root) if f.endswith('.dat')]
        all_records = sorted(list(set(all_records)))

        pbar = tqdm(all_records, desc=f"Loading all records")
        for rid in pbar:
            record_path = os.path.join(self.root, rid)
            try:
                sig, info = wfdb.rdsamp(record_path)
                ann = wfdb.rdann(record_path, 'atr')
            except Exception as e:
                print(f"[WARN] skip record {rid}: {e}")
                continue

            ch_names = [c.upper() for c in info['sig_name']]
            # choose preferred lead
            ch_idx = None
            for lead in self.prefer_leads:
                if lead.upper() in ch_names:
                    ch_idx = ch_names.index(lead.upper())
                    break
            if ch_idx is None:
                ch_idx = 0  # fallback

            ecg = sig[:, ch_idx].astype(np.float32)
            if self.baseline_rm:
                ecg = simple_baseline_remove(ecg, fs=self.fs, win_ms=200)

            rpos = ann.sample
            syms = ann.symbol
            for s, sym in zip(rpos, syms):
                cls = symbol_to_class(sym)
                if cls is None:
                    continue

                L = self.segment_len
                half = L // 2
                start = int(s - half)
                end = start + L
                if start < 0 or end > len(ecg):
                    # pad reflect to keep length
                    pad_left = max(0, -start)
                    pad_right = max(0, end - len(ecg))
                    seg = ecg[max(0,start):min(len(ecg),end)]
                    if pad_left>0 or pad_right>0:
                        seg = np.pad(seg, (pad_left,pad_right), mode='reflect')
                else:
                    seg = ecg[start:end]
                if len(seg) != L:
                    continue

                # normalize per segment
                if self.normalize == 'zscore':
                    m = seg.mean()
                    sd = seg.std() + 1e-6
                    seg = (seg - m) / sd
                elif self.normalize == 'minmax':
                    mn, mx = seg.min(), seg.max()
                    seg = (seg - mn) / (mx - mn + 1e-6)
                    seg = seg * 2 - 1

                X_list.append(seg.astype(np.float32))
                y_list.append(cls)

        # 2. Convert to numpy arrays
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)

        # 3. Sample-level stratified splitting
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=self.random_state
        )

        # 4. Select corresponding data based on split
        if self.split == 'train':
            self.X, self.y = X_train, y_train
            if self.class_limit is not None:
                # Apply sampling limit for each class
                mask = np.zeros(len(self.y), dtype=bool)
                for cls in np.unique(self.y):
                    cls_idx = np.where(self.y == cls)[0]
                    if len(cls_idx) > self.class_limit:
                        selected = np.random.RandomState(self.random_state).choice(
                            cls_idx, self.class_limit, replace=False
                        )
                        mask[selected] = True
                    else:
                        mask[cls_idx] = True
                self.X = self.X[mask]
                self.y = self.y[mask]
        elif self.split == 'val':
            self.X, self.y = X_val, y_val
        else:  # test
            self.X, self.y = X_test, y_test

        print(f"[StratifiedECGSegments] Built {split}: X={self.X.shape}, y distrib={dict(zip(*np.unique(self.y, return_counts=True)))}")

        if cache_path:
            np.savez_compressed(cache_path, X=self.X, y=self.y)
            print(f"[StratifiedECGSegments] Saved cache to {cache_path}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        x = torch.from_numpy(x).unsqueeze(0)  # [1, T]
        y = torch.tensor(y).long()
        return x, y

