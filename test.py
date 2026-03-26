import os
import json
from collections import Counter
import pandas as pd
from stratified_dataset import StratifiedECGSegments
# 如果您有非 stratified 版本，请确保也导入 ECGSegments
# from your_dataset_file import ECGSegments

from torch.utils.data import DataLoader
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import ECGSegments 

SEED = 1337

SYM2CLS = {
    0: 'N',      # 正常心跳
    1: 'SVEB',   # 室上性异位心跳
    2: 'VEB',    # 室性异位心跳
    3: 'Other',  # 其他心跳
}

def make_loaders(root: str,
                 segment_sec: float = 2.0,
                 fs: int = 360,
                 batch_size: int = 256,
                 num_workers: int = 2,
                 cache: bool = False,
                 class_limit: Optional[int] = None,
                 stratified: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Create data loaders"""
    
    seglen = int(round(segment_sec * fs))
    
    if stratified:
        cache_train = os.path.join(root, f"cache_stratified_train_{seglen}.npz") if cache else None
        cache_val   = os.path.join(root, f"cache_stratified_val_{seglen}.npz") if cache else None
        cache_test  = os.path.join(root, f"cache_stratified_test_{seglen}.npz") if cache else None
    else:
        cache_train = os.path.join(root, f"cache_train_{seglen}.npz") if cache else None
        cache_val   = os.path.join(root, f"cache_val_{seglen}.npz") if cache else None
        cache_test  = os.path.join(root, f"cache_test_{seglen}.npz") if cache else None
  
    if stratified:
        ds_train = StratifiedECGSegments(root, 'train', segment_sec, fs,
                                          class_limit=class_limit,
                                          cache_path=cache_train,
                                          random_state=SEED)
        ds_val   = StratifiedECGSegments(root, 'val', segment_sec, fs,
                                          class_limit=None,
                                          cache_path=cache_val,
                                          random_state=SEED)
        ds_test  = StratifiedECGSegments(root, 'test', segment_sec, fs,
                                          class_limit=None,
                                          cache_path=cache_test,
                                          random_state=SEED)
    else:
        ds_train = ECGSegments(root, 'train', segment_sec, fs,
                               class_limit=class_limit,
                               cache_path=cache_train,
                               random_state=SEED)
        ds_val   = ECGSegments(root, 'val', segment_sec, fs,
                               class_limit=None,
                               cache_path=cache_val,
                               random_state=SEED)
        ds_test  = ECGSegments(root, 'test', segment_sec, fs,
                               class_limit=None,
                               cache_path=cache_test,
                               random_state=SEED)
    
    print(f"Dataset size: Train={len(ds_train)}, Val={len(ds_val)}, Test={len(ds_test)}")
    
    if os.name == 'nt':
        num_workers = 0
    
    generator = torch.Generator()
    generator.manual_seed(SEED)
    
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              generator=generator)
    val_loader   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    
    num_classes = 4
    return train_loader, val_loader, test_loader, num_classes


# 调用 loader
train_loader, val_loader, test_loader, num_classes = make_loaders(
    root='./data/mitbih',
    cache=True
)

def get_class_distribution(loader, num_classes=4, class_map=None):
    """统计 DataLoader 中各个类别的样本数量，并按类别名称输出"""
    all_labels = []
    
    for i in range(len(loader.dataset)):
        _, label = loader.dataset[i]
        if hasattr(label, 'item'):
            label = label.item()
        all_labels.append(label)
    
    counts = Counter(all_labels)
    
    if class_map is None:
        class_map = {i: f"Class {i}" for i in range(num_classes)}
    
    return {class_map[i]: counts.get(i, 0) for i in range(num_classes)}


# 统计各数据集
stats = {
    "Train Set": get_class_distribution(train_loader, num_classes=num_classes, class_map=SYM2CLS),
    "Val Set": get_class_distribution(val_loader, num_classes=num_classes, class_map=SYM2CLS),
    "Test Set": get_class_distribution(test_loader, num_classes=num_classes, class_map=SYM2CLS)
}

# 使用 Pandas 打印
df_stats = pd.DataFrame(stats).T
print("\n--- 各数据集样本类别分布 ---")
print(df_stats)

# 可选：打印总计
print("\nTotal samples per dataset:")
print(df_stats.sum(axis=1))