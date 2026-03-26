#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG Lightweight Anomaly Detector - Utils
----------------------------------------
Data preprocessing, symbol mapping, file operations and other utility functions
"""

import os
import re
import math
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split

# Set random seed
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

############################################
#               Data Utils                 #
############################################
AAMI_N = set(list("NLR") + ['e','j'])
AAMI_SVEB = set(['A','a','J','S'])
AAMI_VEB = set(['V','E'])

SYM2CLS = {
    0: 'N',
    1: 'SVEB',
    2: 'VEB',
    3: 'Other',
}


def symbol_to_class(sym: str) -> Optional[int]:
    """Map MIT-BIH beat symbol to 4-class label.
    Returns None for symbols we choose to ignore.
    """
    s = sym.strip()
    if s in AAMI_N:
        return 0
    if s in AAMI_SVEB:
        return 1
    if s in AAMI_VEB:
        return 2
    # We include paced 'P', fusion 'F', and others as 'Other'
    # Common symbols: '/', 'f', 'Q', 'F', 'P', 'x', etc.
    return 3


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Calculate moving average"""
    if w <= 1:
        return x
    c = np.convolve(x, np.ones(w)/w, mode='same')
    return c


def simple_baseline_remove(x: np.ndarray, fs: int, win_ms: int = 200) -> np.ndarray:
    """High-pass like baseline wander removal by subtracting MA."""
    w = max(1, int(round(win_ms * fs / 1000.0)))
    baseline = moving_average(x, w)
    y = x - baseline
    return y


def list_records(root: str) -> List[str]:
    """List all record files in MIT-BIH dataset"""
    recs = []
    for fname in os.listdir(root):
        if fname.endswith('.dat'):
            rid = fname[:-4]
            if os.path.exists(os.path.join(root, rid + '.hea')):
                recs.append(rid)
    recs = sorted(recs)
    return recs


def default_split(records: List[str], ratios=(0.7, 0.15, 0.15)) -> Tuple[List[str], List[str], List[str]]:
    """Default dataset split"""
    # Split by record id deterministically
    r = sorted(records)
    n = len(r)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = r[:n_train]
    val = r[n_train:n_train+n_val]
    test = r[n_train+n_val:]
    return train, val, test


def stratified_split(y: List[int], records: List[str], ratios=(0.7, 0.15, 0.15)) -> Tuple[List[str], List[str], List[str]]:
    """Stratified sampling to ensure consistent class ratios"""
    y = np.array(y)
    records = np.array(records)
    train_val_ratio = ratios[0] + ratios[1]
    train_val_idx, test_idx = train_test_split(np.arange(len(y)), test_size=ratios[2], stratify=y, random_state=SEED)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=ratios[1]/train_val_ratio, stratify=y[train_val_idx], random_state=SEED)
    return records[train_idx].tolist(), records[val_idx].tolist(), records[test_idx].tolist()


def parse_rec_list(rec_str: Optional[str]) -> Optional[List[str]]:
    """Parse record list string"""
    if not rec_str:
        return None
    parts = [s.strip() for s in rec_str.split(',') if s.strip()]
    return parts if parts else None


def count_params(model) -> int:
    """Calculate model parameter count"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
