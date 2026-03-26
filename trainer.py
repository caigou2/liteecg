#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG Lightweight Anomaly Detector - Trainer
------------------------------------------
Training and evaluation functions
Used for training ECG anomaly detection models, including data loading, model training, evaluation and result saving.
"""

# Basic library imports
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm  # Progress bar display

# Evaluation metrics library
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from typing import Optional, List, Tuple  # Type annotations

# Command line argument parsing
import argparse

# Project module imports
from utils import parse_rec_list, count_params, SYM2CLS  # Utility functions and constants
from models import LiteECGNet, DeepECGNet, ECGNet, SE_ECGNet, BiRCNN, LDCNN, ResNet  # Model definitions
from stratified_dataset import StratifiedECGSegments  # Stratified dataset
from config import *  # Configuration parameters
from dataset import ECGSegments  # Non-stratified dataset

# Other utility libraries
import datetime  # Timestamp generation
import time  # Timing functionality
import random  # Random number generation
import psutil  # System resource monitoring
from thop import profile  # Model computational analysis

SEED = 1337  # Global random seed


def set_global_seed(seed=SEED):
    """Set global random seed to ensure reproducibility of experimental results
    
    Args:
        seed: Random seed value, defaults to predefined SEED
    """
    # Set random seeds for Python, NumPy and PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs
    
    # Set cuDNN deterministic operations
    torch.backends.cudnn.deterministic = True  # Ensure deterministic algorithms
    torch.backends.cudnn.benchmark = False     # Disable automatic fastest algorithm search
    torch.backends.cudnn.enabled = True        # Keep cuDNN enabled
    
    # Set PyTorch deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set environment variables to enhance determinism
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set Python hash seed
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Set CuBLAS workspace configuration


# Immediately set global seed to ensure reproducibility of entire training process
set_global_seed(SEED)


class FocalLoss(nn.Module):
    """Focal Loss implementation for addressing class imbalance issues
    
    Reference paper: https://arxiv.org/abs/1708.02002
    Mainly used to increase weights for hard-to-classify samples and reduce weights for easy-to-classify samples
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """Initialize Focal Loss
        
        Args:
            alpha: Balance factor, defaults to 1
            gamma: Focusing parameter, defaults to 2
            reduction: Loss aggregation method, 'mean', 'sum' or 'none'
        """
        super().__init__()
        self.alpha = alpha  # Balance factor
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction  # Loss aggregation method
    
    def forward(self, logits, targets):
        """Calculate Focal Loss
        
        Args:
            logits: Model output logits
            targets: True labels
            
        Returns:
            Calculated Focal Loss value
        """
        # Calculate cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        # Calculate prediction probability
        pt = torch.exp(-ce_loss)
        # Calculate Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Return result according to specified aggregation method
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def set_deterministic():
    """Set deterministic training environment to ensure reproducibility of experimental results
    
    This function further ensures the determinism of the training process based on set_global_seed
    Mainly used to reconfirm random seed settings in the main function
    """
    # Set all random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    # Set deterministic operations
    torch.backends.cudnn.deterministic = True  # Ensure deterministic algorithms
    torch.backends.cudnn.benchmark = False     # Disable automatic fastest algorithm search
    
    # Set environment variables
    import os
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
    # Try to set deterministic algorithms (if supported)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    except:
        # If deterministic algorithms are not supported, continue execution
        pass


def get_class_distribution(loader, class_ids=None, class_names=None):
    """
    统计 DataLoader 对应 dataset 中各类别样本数量

    Args:
        loader: DataLoader
        class_ids: 类别编号列表，例如 [0,1,2,3,4]
        class_names: 类别名称列表，例如 ['N','S','V','F','Q']

    Returns:
        dict: {类名: 数量}
    """
    if class_ids is None:
        class_ids = sorted(list(SYM2CLS.keys()))
    if class_names is None:
        class_names = [SYM2CLS[i] for i in class_ids]

    counts = {name: 0 for name in class_names}

    # 直接遍历 dataset，确保统计的是整个数据集而不是一个 batch
    for i in range(len(loader.dataset)):
        _, label = loader.dataset[i]
        if hasattr(label, 'item'):
            label = int(label.item())
        else:
            label = int(label)

        if label in class_ids:
            idx = class_ids.index(label)
            counts[class_names[idx]] += 1

    return counts


def print_class_distribution(dist_dict, title="Class Distribution"):
    """打印类别分布"""
    print(f"\n=== {title} ===")
    for cls_name, cnt in dist_dict.items():
        print(f"{cls_name}: {cnt}")


def make_loaders(root: str,
                 segment_sec: float = 2.0,
                 fs: int = 360,
                 batch_size: int = 256,
                 num_workers: int = 2,
                 cache: bool = False,
                 class_limit: Optional[int] = None,
                 stratified: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Create data loaders
    
    Args:
        root: Root directory path of MIT-BIH dataset
        segment_sec: Duration of each ECG segment (seconds)
        fs: Sampling frequency (Hz)
        batch_size: Batch size
        num_workers: Number of processes for data loading
        cache: Whether to use cache to accelerate data loading
        class_limit: Maximum number of samples per class in training set (for class balancing)
        stratified: Whether to use stratified sampling
        
    Returns:
        Tuple (train_loader, val_loader, test_loader, num_classes), containing training, validation, test data loaders and number of classes
    """
    # Calculate number of sampling points for each ECG segment
    seglen = int(round(segment_sec * fs))
    
    # Set cache file paths
    if stratified:
        cache_train = os.path.join(root, f"cache_stratified_train_{seglen}.npz") if cache else None
        cache_val   = os.path.join(root, f"cache_stratified_val_{seglen}.npz") if cache else None
        cache_test  = os.path.join(root, f"cache_stratified_test_{seglen}.npz") if cache else None
    else:
        cache_train = os.path.join(root, f"cache_train_{seglen}.npz") if cache else None
        cache_val   = os.path.join(root, f"cache_val_{seglen}.npz") if cache else None
        cache_test  = os.path.join(root, f"cache_test_{seglen}.npz") if cache else None
  
    # Create dataset instances
    if stratified:
        ds_train = StratifiedECGSegments(root, 'train', segment_sec, fs, class_limit=class_limit, cache_path=cache_train, random_state=SEED)
        ds_val   = StratifiedECGSegments(root, 'val',   segment_sec, fs, class_limit=None, cache_path=cache_val, random_state=SEED)
        ds_test  = StratifiedECGSegments(root, 'test',  segment_sec, fs, class_limit=None, cache_path=cache_test, random_state=SEED)
    else:
        ds_train = ECGSegments(root, 'train', segment_sec, fs, class_limit=class_limit, cache_path=cache_train, random_state=SEED)
        ds_val   = ECGSegments(root, 'val',   segment_sec, fs, class_limit=None, cache_path=cache_val, random_state=SEED)
        ds_test  = ECGSegments(root, 'test',  segment_sec, fs, class_limit=None, cache_path=cache_test, random_state=SEED)
    
    # Print dataset size information
    print(f"Dataset size: Train={len(ds_train)}, Val={len(ds_val)}, Test={len(ds_test)}")
    
    # Disable multiprocessing on Windows to ensure reproducibility
    if os.name == 'nt':  # Windows
        num_workers = 0
    
    # Create generator to ensure DataLoader reproducibility
    generator = torch.Generator()
    generator.manual_seed(SEED)
    
    # Create data loaders
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True, 
                             generator=generator)
    val_loader   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(ds_test, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    
    # Number of classes: follow SYM2CLS mapping
    num_classes = len(SYM2CLS)
    return train_loader, val_loader, test_loader, num_classes


def train_one_epoch(model, loader, optimizer, device, criterion):
    """Train model for one epoch
    
    Args:
        model: Model to train
        loader: Data loader
        optimizer: Optimizer
        device: Running device ('cpu' or 'cuda')
        criterion: Loss function
        
    Returns:
        Tuple (avg_loss, f1_score), containing average loss and F1 score
    """
    # Set model to training mode
    model.train()
    losses = []  # Record loss for each batch
    y_true, y_pred = [], []  # Record true labels and predicted labels
    
    # Iterate through each batch in the data loader
    for x, y in tqdm(loader, desc='Train', leave=False):
        # Move data to specified device
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)  # Use set_to_none=True to improve performance
        
        # Choose different forward propagation methods based on model type
        # Check if it's an IMBECGNET-related model
        if hasattr(model, '__class__') and 'IMBECGNET' in model.__class__.__name__:
            if 'Improved' in model.__class__.__name__ or 'Enhanced' in model.__class__.__name__:
                # IMBECGNET_Improved and Enhanced use standard interface
                logits = model(x)
                loss = criterion(logits, y)
            else:
                # Original IMBECGNET uses special interface
                outputs = model(x, class_label=None)
                loss = criterion(outputs, y)
                # Extract final logits for evaluation
                if isinstance(outputs, dict):
                    logits = outputs['final_logits']
                else:
                    logits = outputs
        else:
            # Standard model interface
            logits = model(x)
            loss = criterion(logits, y)
        
        # Backpropagation and parameter update
        loss.backward()
        optimizer.step()
        
        # Record loss and prediction results
        losses.append(loss.item())
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
    
    # Calculate average loss and F1 score for the entire epoch
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    f1 = f1_score(
        y_true,
        y_pred,
        average='macro',
        labels=list(sorted(SYM2CLS.keys())),
        zero_division=0
    )  # Use macro average to calculate F1 score
    
    return float(np.mean(losses)), f1


def evaluate(model, loader, device, criterion=None, verbose=False, split_name='Val'):
    """Evaluate model performance (包含单类指标)
    
    Args:
        model: Model to evaluate
        loader: Data loader
        device: Running device ('cpu' or 'cuda')
        criterion: Loss function, optional
        verbose: Whether to print detailed evaluation results
        split_name: Dataset name (for progress bar display)
        
    Returns:
        Tuple (loss, acc, f1, y_true, y_pred, per_class_acc, per_class_f1)
    """
    # Check if data loader is empty
    if len(loader) == 0:
        print(f"Warning: {split_name} dataset is empty!")
        return 0.0, 0.0, 0.0, np.array([]), np.array([]), np.array([]), np.array([])
    
    # Set model to evaluation mode
    model.eval()
    losses = []  # Record loss
    y_true, y_pred = [], []  # Record true labels and predicted labels
    
    # Disable gradient computation to improve performance and reduce memory usage
    with torch.no_grad():
        # Iterate through each batch in the data loader
        for x, y in tqdm(loader, desc=split_name, leave=False):
            # Move data to specified device
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)            
            logits = model(x)
            
            # If loss function is provided, calculate and record loss
            if criterion is not None:
                losses.append(criterion(logits, y).item())
            
            # Record true labels and predicted labels
            y_true.append(y.detach().cpu().numpy())
            y_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
    
    # Calculate evaluation metrics
    if not y_true:  # If no data
        return 0.0, 0.0, 0.0, np.array([]), np.array([]), np.array([]), np.array([])
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # ---------------- 统一按 SYM2CLS 的类别数计算 ----------------
    class_ids = sorted(list(SYM2CLS.keys()))
    num_classes = len(class_ids)

    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    
    # 每个类的准确率（Recall） = TP / (TP + FN)
    # 加上 1e-8 避免分母为 0
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    
    # 获取每个类的 F1 分数
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=class_ids, zero_division=0)
    # -----------------------------------------------------------

    f1 = f1_score(y_true, y_pred, average='macro', labels=class_ids, zero_division=0)  # Macro average F1 score
    acc = (y_true == y_pred).mean()  # Accuracy
    
    # If verbose mode is enabled, print classification report and confusion matrix
    if verbose:
        target_names = [SYM2CLS[i] for i in class_ids]
        print("\n" + classification_report(
            y_true,
            y_pred,
            labels=class_ids,
            target_names=target_names,
            digits=5,
            zero_division=0
        ))
        print("Confusion Matrix:\n", cm)
        
        # 打印单类指标
        print("\nPer-class Metrics:")
        for i, name in enumerate(target_names):
            print(f"  {name}: Accuracy (Recall) = {per_class_acc[i]:.4f}, F1 = {per_class_f1[i]:.4f}")
    
    # Calculate average loss (if any)
    loss = float(np.mean(losses)) if losses else 0.0
    
    return loss, acc, f1, y_true, y_pred, per_class_acc, per_class_f1


def calculate_model_metrics(model, args, sample_input):
    """Calculate various model metrics"""
    metrics = {}
    
    # 1. Model parameter count
    param_count = count_params(model)
    metrics['param_count'] = param_count
    
    # 2. Model file size
    temp_path = 'temp_model.pt'
    torch.save(model.state_dict(), temp_path)
    model_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    metrics['model_size_mb'] = model_size_mb
    
    # 3. Calculate FLOPs
    model_name = model.__class__.__name__.lower()
    if model_name in ['se_ecgnet', 'seecgnet']:
        print('[WARN] SE_ECGNet does not support FLOPs statistics, skipped.')
        metrics['flops'] = None
        metrics['flops_params'] = None
    else:
        try:
            flops, params = profile(model, inputs=(sample_input,), verbose=False)
            metrics['flops'] = flops
            metrics['flops_params'] = params
        except Exception as e:
            print(f"[WARN] Failed to calculate FLOPs: {e}")
            metrics['flops'] = 0
            metrics['flops_params'] = 0
    
    # 4. Inference latency
    model.eval()
    warmup_runs = 10
    test_runs = 100
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)
    
    # Test inference time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(test_runs):
            _ = model(sample_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / test_runs * 1000  # ms
    metrics['inference_latency_ms'] = avg_inference_time
    
    # 5. Memory usage
    process = psutil.Process(os.getpid())
    memory_usage_mb = process.memory_info().rss / (1024 * 1024)
    metrics['memory_usage_mb'] = memory_usage_mb
    
    return metrics


def train_model(model, train_loader, val_loader, test_loader, args):
    """Main function for training model"""
    model.to(args.device)
    print(model)

    # ---------------- 新增：统计训练集和测试集各类别数量 ----------------
    class_ids = sorted(list(SYM2CLS.keys()))
    class_names = [SYM2CLS[i] for i in class_ids]

    train_class_dist = get_class_distribution(train_loader, class_ids=class_ids, class_names=class_names)
    test_class_dist = get_class_distribution(test_loader, class_ids=class_ids, class_names=class_names)

    print_class_distribution(train_class_dist, title="Training Set Class Distribution")
    print_class_distribution(test_class_dist, title="Test Set Class Distribution")
    # -------------------------------------------------------------------

    # Calculate model metrics
    sample_input = torch.randn(1, 1, int(round(args.segment_sec * args.fs))).to(args.device)
    model_metrics = calculate_model_metrics(model, args, sample_input)
    print(f"Trainable params: {model_metrics['param_count']:,}")
    print(f"Model size: {model_metrics['model_size_mb']:.2f} MB")
    if model_metrics['flops'] is None:
        print(f"FLOPs: N/A")
    else:
        print(f"FLOPs: {model_metrics['flops']:.2e}")
    print(f"Inference latency: {model_metrics['inference_latency_ms']:.2f} ms")
    print(f"Memory usage: {model_metrics['memory_usage_mb']:.2f} MB")

    # Criterion selection
    if getattr(args, 'focal', False):
        criterion = FocalLoss()
        criterion_info = {
            "name": "FocalLoss",
            "alpha": criterion.alpha,
            "gamma": criterion.gamma,
            "reduction": criterion.reduction
        }
    else:
        criterion = nn.CrossEntropyLoss()
        criterion_info = {
            "name": "CrossEntropyLoss"
        }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    # ---------------- 新增：记录最佳模型对应的验证集指标 ----------------
    best_epoch = -1
    best_val_loss = None
    best_val_acc = None
    best_val_f1 = None
    best_val_per_class_acc = None
    best_val_per_class_f1 = None
    # -------------------------------------------------------------------

    # Record training metrics
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    epoch_times = []

    # ---------------- 新增：记录每一轮验证集各类别准确率/F1 ----------------
    val_per_class_acc_history = []
    val_per_class_f1_history = []
    # -------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        epoch_start_time = time.time()
        
        tr_loss, tr_f1 = train_one_epoch(model, train_loader, optimizer, args.device, criterion)
        
        # 接收每类指标，但不逐步打印
        val_loss, val_acc, val_f1, _, _, val_per_class_acc, val_per_class_f1 = evaluate(
            model, val_loader, args.device, criterion, verbose=False, split_name='Val'
        )
        
        scheduler.step()
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        # Record metrics
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_f1s.append(tr_f1)
        val_f1s.append(val_f1)

        # ---------------- 新增：保存每一轮的各类别准确率/F1 ----------------
        val_per_class_acc_history.append({
            class_names[i]: float(val_per_class_acc[i]) for i in range(len(class_ids))
        })
        val_per_class_f1_history.append({
            class_names[i]: float(val_per_class_f1[i]) for i in range(len(class_ids))
        })
        # -------------------------------------------------------------------

        print(f"Train: loss={tr_loss:.4f}, F1={tr_f1:.4f} | Val: loss={val_loss:.4f}, acc={val_acc:.4f}, F1={val_f1:.4f} | Time: {epoch_duration:.2f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0

            # ---------------- 新增：记录最佳轮次的验证集指标 ----------------
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_per_class_acc = {
                class_names[i]: float(val_per_class_acc[i]) for i in range(len(class_ids))
            }
            best_val_per_class_f1 = {
                class_names[i]: float(val_per_class_f1[i]) for i in range(len(class_ids))
            }
            # ----------------------------------------------------------------
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(args.device) for k, v in best_state.items()})

    print("\n=== Final Evaluation on Test ===")
    # 接收测试集的单类指标
    test_loss, test_acc, test_f1, y_true, y_pred, test_per_class_acc, test_per_class_f1 = evaluate(
        model, test_loader, args.device, criterion=None, verbose=True, split_name='Test'
    )
    print(f"Test: acc={test_acc:.4f}, F1={test_f1:.4f}")

    # Save artifacts
    out_dir = os.path.abspath(os.path.join(args.root, '../records'))
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = f"{args.model}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
    torch.save(model.state_dict(), os.path.join(out_dir, f'ecg_lightweight_{model_tag}.pt'))
    
    # Calculate average epoch time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    
    # ---------------- 新增：将测试集单类指标转为字典以便存入 JSON ----------------
    target_names = [SYM2CLS[i] for i in class_ids]
    per_class_acc_dict = {name: float(test_per_class_acc[i]) for i, name in enumerate(target_names)}
    per_class_f1_dict = {name: float(test_per_class_f1[i]) for i, name in enumerate(target_names)}
    # ----------------------------------------------------------------------

    # ---------------- 新增：保存“最佳模型”对应的指标信息 ----------------
    best_model_info = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "best_val_acc": float(best_val_acc) if best_val_acc is not None else None,
        "best_val_f1": float(best_val_f1) if best_val_f1 is not None else None,
        "best_val_per_class_acc": best_val_per_class_acc,
        "best_val_per_class_f1": best_val_per_class_f1,
        "loss_function": criterion_info
    }
    # -------------------------------------------------------------------

    with open(os.path.join(out_dir, f'eval_report_{model_tag}.json'), 'w') as f:
        json.dump({
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'test_per_class_acc': per_class_acc_dict,  # 写入单类准确率
            'test_per_class_f1': per_class_f1_dict,    # 写入单类 F1
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=class_ids).tolist(),
            'class_distribution': {
                'train': train_class_dist,  # 训练集每类数量
                'test': test_class_dist     # 测试集每类数量
            },
            'best_model': best_model_info,
            'model_metrics': model_metrics,
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_f1s': train_f1s,
                'val_f1s': val_f1s,
                'val_per_class_acc_history': val_per_class_acc_history,
                'val_per_class_f1_history': val_per_class_f1_history,
                'epoch_times': epoch_times,
                'avg_epoch_time': avg_epoch_time
            }
        }, f, indent=2)

    print(f"Saved model and report to {out_dir} (tag: {model_tag})")
    print(f"Average epoch time: {avg_epoch_time:.2f}s")


if __name__ == '__main__':
    # First set deterministic training
    set_deterministic()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=DATA_ROOT, help='Path to MIT-BIH directory with .dat/.hea/.atr files')
    parser.add_argument('--train-recs', type=str, default=None, help='Comma-separated record ids for train (e.g., 101,106,...)')
    parser.add_argument('--val-recs', type=str, default=None, help='Comma-separated record ids for val')
    parser.add_argument('--test-recs', type=str, default=None, help='Comma-separated record ids for test')

    parser.add_argument('--segment-sec', type=float, default=2.0)
    parser.add_argument('--fs', type=int, default=360)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--cache', action='store_true', default=False, help='Cache preprocessed segments to .npz files')
    parser.add_argument('--class-limit', type=int, default=None, help='Optional per-class cap in training set (e.g., 8000)')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience (epochs)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--focal', action='store_true', help='Use Focal Loss')
    parser.add_argument('--no-stratified', action='store_true', help='Do not use stratified sampling')
    parser.add_argument('--model', type=str, default='liteecgnet', choices=['liteecgnet', 'deepecgnet', 'ecgnet', 'se_ecgnet', 'resnet', 'bircnn', 'ldcnn'], help='Select model')

    args = parser.parse_args()
    set_deterministic()
   
    train_loader, val_loader, test_loader, num_classes = make_loaders(
        root=args.root,
        segment_sec=args.segment_sec,
        fs=args.fs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache=args.cache,
        class_limit=args.class_limit,
        stratified=not args.no_stratified,
    )
    if args.model == 'ecgnet':
        model = ECGNet(config=ECGNET_CONFIG)
    elif args.model == 'se_ecgnet':
        model = SE_ECGNet(config=SE_ECGNET_CONFIG)
    elif args.model == 'bircnn':
        model = BiRCNN(config=BIRCNN_CONFIG)
    elif args.model == 'resnet':
        model = ResNet(config=RESNET_CONFIG)
    elif args.model == 'liteecgnet':
        model = LiteECGNet(config=LITEECGNET_CONFIG)
    elif args.model == 'deepecgnet':
        model = DeepECGNet(config=DEEPECGNET_CONFIG)
    elif args.model == 'ldcnn':
        input_len = int(round(args.segment_sec * args.fs))
        model = LDCNN(config=LDCNN_CONFIG, input_len=input_len)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    train_model(model, train_loader, val_loader, test_loader, args)