#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG Lightweight Anomaly Detector - Utils
----------------------------------------
数据预处理、符号映射、文件操作和其他实用函数

新手提示：
- 本文件包含了数据处理的核心工具函数
- symbol_to_class函数用于将MIT-BIH符号映射到4类标签
- simple_baseline_remove函数用于移除ECG信号的基线漂移
- 其他函数用于数据集分割和模型参数计算
"""

import os
import re
import math
import random
import numpy as np
import pywt
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from scipy import signal
# 设置随机种子
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

############################################
#               数据工具                    #
############################################
# AAMI标准的心跳分类
# AAMI标准的心跳分类 (5分类: N, S, V, F, Q) 
AAMI_N = set(['N', 'L', 'R', 'e', 'j'])  # 正常心跳及其变体 (Normal)
AAMI_SVEB = set(['A', 'a', 'J', 'S'])    # 室上性异位搏动 (SVEB)
AAMI_VEB = set(['V', 'E'])               # 室性异位搏动 (VEB)
AAMI_F = set(['F'])                      # 融合搏动 (Fusion of ventricular and normal beat)
AAMI_Q = set(['/', 'f', 'Q'])            # 未知/起搏心跳 (Paced, fusion of paced and normal, unclassified)

# 类别映射字典 (用于将分类映射为数字索引，方便传入 PyTorch 的 Loss 函数)
SYM2CLS = {
    0: 'N',      # 正常搏动
    1: 'SVEB',   # 室上性异位搏动
    2: 'VEB',    # 室性异位搏动
    3: 'F',      # 融合搏动
    4: 'Q',      # 未知/起搏搏动
}

# 完整的符号转换函数 (对应你前一段代码中的 symbol_to_class)
def symbol_to_class(sym: str) -> int:
    """
    将 MIT-BIH 的心跳标注符号转换为 AAMI 标准的 5 分类索引 (0-4)。
    如果符号不是心跳标注（如节律变化标记 '~', '|', '+' 等），则返回 None。
    """
    if sym in AAMI_N:
        return 0
    elif sym in AAMI_SVEB:
        return 1
    elif sym in AAMI_VEB:
        return 2
    elif sym in AAMI_F:
        return 3
    elif sym in AAMI_Q:
        return 4
    else:
        # 过滤掉非心跳的系统标记或噪声标记
        return None
############################################
#           2. PTB 专用工具                #
############################################
# PTB 类别映射 (根据 LDCNN 2024 论文)
PTB_SYM2CLS = {
    0: 'Healthy Control',
    1: 'Abnormal'
}

def get_ptb_label(comments: List[str]) -> int:
    """
    通过解析 PTB .hea 文件的 comments 字段获取标签。
    0: Healthy control, 1: Abnormal (Myocardial infarction 等)
    """
    diag_str = " ".join(comments).lower()
    if "healthy control" in diag_str:
        return 0
    return 1 # 其他所有情况根据论文归类为 Abnormal


def list_ptb_records(root: str) -> List[str]:
    """
    列出 PTB 数据集中的记录。
    PTB 的结构通常是 root/patientXXX/recordYYY.dat，需递归查找。
    """
    recs = []
    # 遍历所有 patient 文件夹
    for subdir, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(".dat"):
                # 获取相对路径，如 'patient001/s0014lrem'
                rel_path = os.path.join(os.path.basename(subdir), file[:-4])
                # 确保 .hea 文件也存在
                if os.path.exists(os.path.join(subdir, file[:-4] + '.hea')):
                    recs.append(rel_path)
    return sorted(recs)


############################################
#           3. 信号处理工具                #
############################################

def wavelet_denoising(x: np.ndarray, wavelet: str = 'db6', level: int = 3) -> np.ndarray:
    """
    小波阈值去噪 (LDCNN 论文核心预处理步骤阶段 i-a)。
    """
    coeffs = pywt.wavedec(x, wavelet, mode="per")
    # 计算阈值
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    # 对高频系数应用软阈值
    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
    y = pywt.waverec(coeffs, wavelet, mode='per')
    # 确保长度一致
    return y[:len(x)]


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """计算移动平均"""
    if w <= 1: return x
    return np.convolve(x, np.ones(w)/w, mode='same')


def simple_baseline_remove(x: np.ndarray, fs: int, win_ms: int = 200) -> np.ndarray:
    """移除基线漂移（LiteECG 默认方法）。"""
    w = max(1, int(round(win_ms * fs / 1000.0)))
    baseline = moving_average(x, w)
    return x - baseline


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Z-Score 标准化 (LDCNN 阶段 i-b)。"""
    m = x.mean()
    s = x.std() + 1e-7
    return (x - m) / s
# def moving_average(x: np.ndarray, w: int) -> np.ndarray:
#     """计算移动平均"""
#     if w <= 1:
#         return x
#     c = np.convolve(x, np.ones(w)/w, mode='same')
#     return c


# def simple_baseline_remove(x: np.ndarray, fs: int, win_ms: int = 200) -> np.ndarray:
#     """通过减去移动平均来移除基线漂移（类高通滤波效果）。"""
#     w = max(1, int(round(win_ms * fs / 1000.0)))
#     baseline = moving_average(x, w)
#     y = x - baseline
#     return y


def list_records(root: str) -> List[str]:
    """列出MIT-BIH数据集中的所有记录文件"""
    recs = []
    for fname in os.listdir(root):
        if fname.endswith('.dat'):
            rid = fname[:-4]
            if os.path.exists(os.path.join(root, rid + '.hea')):
                recs.append(rid)
    recs = sorted(recs)
    return recs


def default_split(records: List[str], ratios=(0.7, 0.15, 0.15)) -> Tuple[List[str], List[str], List[str]]:
    """默认数据集分割"""
    # 按记录ID确定性地分割
    r = sorted(records)
    n = len(r)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = r[:n_train]
    val = r[n_train:n_train+n_val]
    test = r[n_train+n_val:]
    return train, val, test


def detect_r_peaks(sig, fs=1000):
    """
    针对 PTB 数据集的 R 峰检测 (轻量级 Pan-Tompkins 简化版)
    """
    # 1. 带通滤波 (5-15Hz)
    nyq = 0.5 * fs
    b, a = signal.butter(1, [5/nyq, 15/nyq], btype='band')
    filt = signal.filtfilt(b, a, sig)
    
    # 2. 微分与平方
    diff = np.diff(filt)
    squared = diff ** 2
    
    # 3. 滑动窗口积分
    win = int(0.12 * fs)
    integrated = np.convolve(squared, np.ones(win)/win, mode='same')
    
    # 4. 寻峰
    # PTB 通常心率在 60-100，设置最小距离为 0.5s
    peaks, _ = signal.find_peaks(integrated, distance=int(0.5 * fs), 
                                 height=np.mean(integrated) * 1.5)
    return peaks
def stratified_split(y: List[int], records: List[str], ratios=(0.7, 0.15, 0.15)) -> Tuple[List[str], List[str], List[str]]:
    """分层采样以确保一致的类比例"""
    y = np.array(y)
    records = np.array(records)
    train_val_ratio = ratios[0] + ratios[1]
    train_val_idx, test_idx = train_test_split(np.arange(len(y)), test_size=ratios[2], stratify=y, random_state=SEED)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=ratios[1]/train_val_ratio, stratify=y[train_val_idx], random_state=SEED)
    return records[train_idx].tolist(), records[val_idx].tolist(), records[test_idx].tolist()


def parse_rec_list(rec_str: Optional[str]) -> Optional[List[str]]:
    """解析记录列表字符串"""
    if not rec_str:
        return None
    parts = [s.strip() for s in rec_str.split(',') if s.strip()]
    return parts if parts else None


def count_params(model) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
