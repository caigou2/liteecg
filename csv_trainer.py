#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG Lightweight Anomaly Detector - Trainer
------------------------------------------
训练和评估函数
用于训练ECG异常检测模型，包括数据加载、模型训练、评估和结果保存。

新手提示：
- 这是项目的主要训练脚本，负责整个模型训练流程
- 支持多种模型架构和训练配置
- 包含详细的评估指标计算
"""

# 基础库导入
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm  # 进度条显示
import pandas as pd

# 评估指标库
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from typing import Optional, List, Tuple  # 类型注解

# 命令行参数解析
import argparse

# 项目模块导入
from utils import parse_rec_list, count_params, PTB_SYM2CLS  # 工具函数和常量
from models import LiteECGNet, DeepECGNet, ECGNet, SE_ECGNet, BiRCNN, LDCNN, ResNet ,PTBLDCNN  # 模型定义
from stratified_dataset import StratifiedECGSegments  # 分层数据集
from PTB_config import *  # 配置参数
from dataset import ECGSegments  # 非分层数据集
# 其他工具库
import datetime  # 时间戳生成
import time  # 计时功能
import random  # 随机数生成
import psutil  # 系统资源监控
from thop import profile  # 模型计算分析

SEED = 1337  # 全局随机种子

def set_global_seed(seed=SEED):
    """设置全局随机种子以确保实验结果的可重复性
    
    Args:
        seed: 随机种子值，默认为预定义的SEED
    """
    # 为Python、NumPy和PyTorch设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子
    
    # 设置cuDNN确定性操作
    torch.backends.cudnn.deterministic = True  # 确保使用确定性算法
    torch.backends.cudnn.benchmark = False     # 禁用自动最快算法搜索
    torch.backends.cudnn.enabled = True        # 保持cuDNN启用
    
    # 设置PyTorch确定性算法
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # 设置环境变量以增强确定性
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 设置CuBLAS工作区配置

# 立即设置全局种子以确保整个训练过程的可重复性
set_global_seed(SEED)

class FocalLoss(nn.Module):
    """Focal Loss实现，用于解决类别不平衡问题
    
    参考论文: https://arxiv.org/abs/1708.02002
    主要用于增加难分类样本的权重，减少易分类样本的权重
    
    新手提示：
    - 当训练数据中不同类别的样本数量差异较大时，使用Focal Loss可以提高模型性能
    - 通过gamma参数控制对难分类样本的关注程度
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """初始化Focal Loss
        
        Args:
            alpha: 平衡因子，默认为1
            gamma: 聚焦参数，默认为2
            reduction: 损失聚合方法，'mean', 'sum' 或 'none'
        """
        super().__init__()
        self.alpha = alpha  # 平衡因子
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction  # 损失聚合方法
    
    def forward(self, logits, targets):
        """计算Focal Loss
        
        Args:
            logits: 模型输出的logits
            targets: 真实标签
            
        Returns:
            计算得到的Focal Loss值
        """
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        # 计算预测概率
        pt = torch.exp(-ce_loss)
        # 计算Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # 根据指定的聚合方法返回结果
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def set_deterministic():
    """设置确定性训练环境以确保实验结果的可重复性
    
    该函数在set_global_seed的基础上进一步确保训练过程的确定性
    主要用于在主函数中重新确认随机种子设置
    """
    # 设置所有随机种子
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    # 设置确定性操作
    torch.backends.cudnn.deterministic = True  # 确保使用确定性算法
    torch.backends.cudnn.benchmark = False     # 禁用自动最快算法搜索
    
    # 设置环境变量
    import os
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
    # 尝试设置确定性算法（如果支持）
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    except:
        # 如果不支持确定性算法，继续执行
        pass


# --- 配套的 Dataset 类 ---
class PTBDataset(Dataset):
    """用于封装 PTB CSV 数据的 PyTorch Dataset"""
    def __init__(self, data, labels):
        # 转换形状为 (Batch, Channel, Length) -> (N, 1, 187)
        self.x = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# --- 修改后的 make_loaders 函数 ---
def make_loaders(root: str,
                 batch_size: int = 128,
                 num_workers: int = 2,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 class_limit: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    针对 PTB 数据集（CSV）创建数据加载器
    
    Args:
        root: 包含 ptbdb_normal.csv 和 ptbdb_abnormal.csv 的目录路径
        batch_size: 批处理大小
        num_workers: 数据加载线程数
        test_size: 测试集占比
        val_size: 从训练集中进一步划分出的验证集占比
        class_limit: 每类样本最大限制（用于手动平衡，可选）
        
    Returns:
        Tuple (train_loader, val_loader, test_loader, num_classes)
    """
    # 1. 加载 CSV 文件
    normal_path = os.path.join(root, 'ptbdb_normal.csv')
    abnormal_path = os.path.join(root, 'ptbdb_abnormal.csv')
    
    df_n = pd.read_csv(normal_path, header=None)
    df_a = pd.read_csv(abnormal_path, header=None)
    
    # 2. 如果设置了样本限制（class_limit），进行采样
    if class_limit is not None:
        df_n = df_n.sample(n=min(len(df_n), class_limit), random_state=SEED)
        df_a = df_a.sample(n=min(len(df_a), class_limit), random_state=SEED)
    
    df = pd.concat([df_n, df_a], axis=0).reset_index(drop=True)
    
    # 3. 提取特征和标签
    X = df.iloc[:, :187].values
    Y = df.iloc[:, 187].values.astype(np.int8)
    
    # 4. 第一次划分：划分出测试集 (Test Set)
    # 使用 stratify=Y 确保正常/异常比例在各集合中一致
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=SEED, stratify=Y
    )
    
    # 5. 第二次划分：从 Train_Val 中划分出验证集 (Val Set)
    # 计算相对于原始总量的有效比例
    actual_val_scale = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=actual_val_scale, random_state=SEED, stratify=y_train_val
    )
    
    # 6. 实例化 Dataset 对象
    ds_train = PTBDataset(x_train, y_train)
    ds_val   = PTBDataset(x_val, y_val)
    ds_test  = PTBDataset(x_test, y_test)
    
    print(f"PTB Dataset loaded from {root}")
    print(f"Size: Train={len(ds_train)}, Val={len(ds_val)}, Test={len(ds_test)}")
    
    # 7. 环境适配 (Windows 下多进程处理)
    if os.name == 'nt':
        num_workers = 0
        
    # 8. 创建确定性生成器
    generator = torch.Generator()
    generator.manual_seed(SEED)
    
    # 9. 创建 DataLoader
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True, generator=generator)
    val_loader   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(ds_test, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    
    # PTB 是二分类任务
    num_classes = 2
    
    return train_loader, val_loader, test_loader, num_classes


# def train_one_epoch(model, loader, optimizer, device, criterion):
#     """训练模型一个epoch
    
#     Args:
#         model: 要训练的模型
#         loader: 数据加载器
#         optimizer: 优化器
#         device: 运行设备（'cpu'或'cuda'）
#         criterion: 损失函数
        
#     Returns:
#         元组 (avg_loss, f1_score)，包含平均损失和F1分数
#     """
#     # 设置模型为训练模式
#     model.train()
#     losses = []  # 记录每个批次的损失
#     y_true, y_pred = [], []  # 记录真实标签和预测标签
    
#     # 遍历数据加载器中的每个批次
#     for x, y in tqdm(loader, desc='Train', leave=False):
#         # 将数据移动到指定设备
#         x = x.to(device, non_blocking=True)
#         y = y.to(device, non_blocking=True)
        
#         # 清零梯度
#         optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True提高性能
        
#         # 根据模型类型选择不同的前向传播方法
#         # 检查是否是IMBECGNET相关模型
#         if hasattr(model, '__class__') and 'IMBECGNET' in model.__class__.__name__:
#             if 'Improved' in model.__class__.__name__ or 'Enhanced' in model.__class__.__name__:
#                 # IMBECGNET_Improved和Enhanced使用标准接口
#                 logits = model(x)
#                 loss = criterion(logits, y)
#             else:
#                 # 原始IMBECGNET使用特殊接口
#                 outputs = model(x, class_label=None)
#                 loss = criterion(outputs, y)
#                 # 提取最终的logits用于评估
#                 if isinstance(outputs, dict):
#                     logits = outputs['final_logits']
#                 else:
#                     logits = outputs
#         else:
#             # 标准模型接口
#             logits = model(x)
#             loss = criterion(logits, y)
        
#         # 反向传播和参数更新
#         loss.backward()
#         optimizer.step()
        
#         # 记录损失和预测结果
#         losses.append(loss.item())
#         y_true.append(y.detach().cpu().numpy())
#         y_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
    
#     # 计算整个epoch的平均损失和F1分数
#     y_true = np.concatenate(y_true)
#     y_pred = np.concatenate(y_pred)
#     f1 = f1_score(y_true, y_pred, average='macro')  # 使用macro平均计算F1分数
    
#     return float(np.mean(losses)), f1
def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    losses = []
    y_true, y_pred = [], []

    is_bce = isinstance(criterion, nn.BCEWithLogitsLoss)

    for x, y in tqdm(loader, desc='Train', leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        # prepare target based on loss
        if is_bce:
            # BCEWithLogitsLoss expects float targets (shape Nx1 if logits Nx1)
            # convert y to float and match shape
            # logits may be (N,1) or (N,) depending on model; we standardize to (N,1)
            if logits.dim() == 2 and logits.size(1) == 1:
                y_loss = y.float().view(-1, 1)
            else:
                # if logits is (N,) reduce to shape (N,)
                y_loss = y.float()
        else:
            # CrossEntropyLoss expects (N,C) and targets (N,) of dtype long
            y_loss = y.long()

        loss = criterion(logits, y_loss)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Get predictions: handle BCE vs CE
        if is_bce:
            # sigmoid + threshold 0.5
            if logits.dim() == 2 and logits.size(1) == 1:
                probs = torch.sigmoid(logits.squeeze(1))
            else:
                probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().cpu().numpy()
        else:
            preds = logits.argmax(dim=1).detach().cpu().numpy()

        y_true.append(y.detach().cpu().numpy())
        y_pred.append(preds)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    return float(np.mean(losses)), f1
def evaluate(model, loader, device, criterion=None, verbose=False, split_name='Val'):
    if len(loader) == 0:
        print(f"Warning: {split_name} dataset is empty!")
        return 0.0, 0.0, 0.0, np.array([]), np.array([])

    model.eval()
    losses = []
    y_true, y_pred = [], []

    is_bce = isinstance(criterion, nn.BCEWithLogitsLoss) if criterion is not None else False

    with torch.no_grad():
        for x, y in tqdm(loader, desc=split_name, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)

            if criterion is not None:
                if is_bce:
                    # prepare y for BCE
                    if logits.dim() == 2 and logits.size(1) == 1:
                        y_loss = y.float().view(-1, 1)
                    else:
                        y_loss = y.float()
                else:
                    y_loss = y.long()
                losses.append(criterion(logits, y_loss).item())

            # preds
            if is_bce:
                if logits.dim() == 2 and logits.size(1) == 1:
                    probs = torch.sigmoid(logits.squeeze(1))
                else:
                    probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().cpu().numpy()
            else:
                preds = logits.argmax(dim=1).detach().cpu().numpy()

            y_true.append(y.detach().cpu().numpy())
            y_pred.append(preds)

    if not y_true:
        return 0.0, 0.0, 0.0, np.array([]), np.array([])

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = (y_true == y_pred).mean()

    if verbose:
        print(classification_report(y_true, y_pred, target_names=[PTB_SYM2CLS[i] for i in range(2)], digits=4))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    loss = float(np.mean(losses)) if losses else 0.0
    return loss, acc, f1, y_true, y_pred
# def evaluate(model, loader, device, criterion=None, verbose=False, split_name='Val'):
#     """评估模型性能
    
#     Args:
#         model: 要评估的模型
#         loader: 数据加载器
#         device: 运行设备（'cpu'或'cuda'）
#         criterion: 损失函数，可选
#         verbose: 是否打印详细的评估结果
#         split_name: 数据集名称（用于进度条显示）
        
#     Returns:
#         元组 (loss, acc, f1, y_true, y_pred)，包含损失、准确率、F1分数、真实标签和预测标签
#     """
#     # 检查数据加载器是否为空
#     if len(loader) == 0:
#         print(f"Warning: {split_name} dataset is empty!")
#         return 0.0, 0.0, 0.0, np.array([]), np.array([])
    
#     # 设置模型为评估模式
#     model.eval()
#     losses = []  # 记录损失
#     y_true, y_pred = [], []  # 记录真实标签和预测标签
    
#     # 禁用梯度计算以提高性能并减少内存使用
#     with torch.no_grad():
#         # 遍历数据加载器中的每个批次
#         for x, y in tqdm(loader, desc=split_name, leave=False):
#             # 将数据移动到指定设备
#             x = x.to(device, non_blocking=True)
#             y = y.to(device, non_blocking=True)            
#             logits = model(x)
            
#             # 如果提供了损失函数，计算并记录损失
#             if criterion is not None:
#                 losses.append(criterion(logits, y).item())
            
#             # 记录真实标签和预测标签
#             y_true.append(y.detach().cpu().numpy())
#             y_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
    
#     # 计算评估指标
#     if not y_true:  # 如果没有数据
#         return 0.0, 0.0, 0.0, np.array([]), np.array([])
    
#     y_true = np.concatenate(y_true)
#     y_pred = np.concatenate(y_pred)
#     f1 = f1_score(y_true, y_pred, average='macro')  # Macro平均F1分数
#     acc = (y_true == y_pred).mean()  # 准确率
    
#     # 如果启用了详细模式，打印分类报告和混淆矩阵
#     if verbose:
#         print(classification_report(y_true, y_pred, target_names=[PTB_SYM2CLS[i] for i in range(2)], digits=4))
#         print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    
#     # 计算平均损失（如果有）
#     loss = float(np.mean(losses)) if losses else 0.0
    
#     return loss, acc, f1, y_true, y_pred


def calculate_model_metrics(model, args, sample_input):
    """计算各种模型指标"""
    metrics = {}
    
    # 1. 模型参数数量
    param_count = count_params(model)
    metrics['param_count'] = param_count
    
    # 2. 模型文件大小
    temp_path = 'temp_model.pt'
    torch.save(model.state_dict(), temp_path)
    model_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    metrics['model_size_mb'] = model_size_mb
    
    # 3. 计算FLOPs
    model_name = model.__class__.__name__.lower()
    if model_name in ['se_ecgnet', 'seecgnet']:
        print('[WARN] SE_ECGNet不支持FLOPs统计，已跳过。')
        metrics['flops'] = None
        metrics['flops_params'] = None
    else:
        try:
            flops, params = profile(model, inputs=(sample_input,), verbose=False)
            metrics['flops'] = flops
            metrics['flops_params'] = params
        except Exception as e:
            print(f"[WARN] 计算FLOPs失败: {e}")
            metrics['flops'] = 0
            metrics['flops_params'] = 0
    
    # 4. 推理延迟
    model.eval()
    warmup_runs = 10
    test_runs = 100
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)
    
    # 测试推理时间
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(test_runs):
            _ = model(sample_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / test_runs * 1000  # ms
    metrics['inference_latency_ms'] = avg_inference_time
    
    # 5. 内存使用
    process = psutil.Process(os.getpid())
    memory_usage_mb = process.memory_info().rss / (1024 * 1024)
    metrics['memory_usage_mb'] = memory_usage_mb
    
    return metrics

def train_model(model, train_loader, val_loader, test_loader, args):
    """训练模型的主函数"""
    model.to(args.device)
    print(model)
    
    # 计算模型指标
     # 1. 自动获取输入长度 (从加载器中取第一个样本的形状)
    # 这样即使以后数据集变成 200 或 300 点，代码也不用改
    try:
        example_data, _ = next(iter(train_loader))
        sample_input = example_data[:1].to(args.device) # 形状是 [Batch, 1, 187]，取索引2
    except:
         # fallback if loader empty
        print("Warning: cannot get sample from train_loader for metric computation:", e)
        # try to obtain input_len from model config if available
        input_len = getattr(model, 'segment_len', None) or getattr(model, 'input_len', None) or 187
        sample_input = torch.randn(1, 1, int(input_len)).to(args.device)
     # If model may lazily create layers (e.g., lazy classifier), run one forward pass
    # BEFORE creating optimizer so optimizer will include these parameters.
    model.eval()
    with torch.no_grad():
        try:
            _ = model(sample_input)
        except Exception as e:
            # If forward fails here, print helpful debug info and re-raise
            print("Error during initial forward with sample_input for metric computation:", e)
            raise
    # 2. 计算模型指标（参数量、FLOPs、推理延迟）
   # sample_input = torch.randn(1, 1, input_len).to(args.device)
    # sample_input = torch.randn(1, 1, int(round(args.segment_sec * args.fs))).to(args.device)
    model_metrics = calculate_model_metrics(model, args, sample_input)
    print(f"Trainable params: {model_metrics['param_count']:,}")
    print(f"Model size: {model_metrics['model_size_mb']:.2f} MB")
    if model_metrics['flops'] is None:
        print(f"FLOPs: N/A")
    else:
        print(f"FLOPs: {model_metrics['flops']:.2e}")
    print(f"Inference latency: {model_metrics['inference_latency_ms']:.2f} ms")
    print(f"Memory usage: {model_metrics['memory_usage_mb']:.2f} MB")

    # 选择损失函数
    # if getattr(args, 'focal', False):
    #     criterion = FocalLoss()
    # else:
    #     criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 选择损失函数：如果用户开启focal则优先用FocalLoss
    if getattr(args, 'focal', False):
        criterion = FocalLoss()
    else:
        # infer model output dim: 如果模型有 fc_out 属性，则读取 out_features
        out_dim = None
        if hasattr(model, 'fc_out') and hasattr(model.fc_out, 'out_features'):
            out_dim = model.fc_out.out_features
        # fallback: 若模型最后输出一维，认为是二元 single-logit -> BCE
        # 若 out_dim == 1 -> BCEWithLogitsLoss, 否则 CrossEntropyLoss
        if out_dim == 1:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    
    # 记录训练指标
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    epoch_times = []

    for epoch in range(1, args.epochs+1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        epoch_start_time = time.time()
        
        tr_loss, tr_f1 = train_one_epoch(model, train_loader, optimizer, args.device, criterion)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, args.device, criterion, verbose=False, split_name='Val')
        scheduler.step()
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        # 记录指标
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_f1s.append(tr_f1)
        val_f1s.append(val_f1)
        
        print(f"Train: loss={tr_loss:.4f}, F1={tr_f1:.4f} | Val: loss={val_loss:.4f}, acc={val_acc:.4f}, F1={val_f1:.4f} | Time: {epoch_duration:.2f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict({k:v.to(args.device) for k,v in best_state.items()})

    print("\n=== Final Evaluation on Test ===")
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, args.device, criterion=None, verbose=True, split_name='Test')
    print(f"Test: acc={test_acc:.4f}, F1={test_f1:.4f}")

    # 保存结果
    out_dir = os.path.abspath(os.path.join(args.root, '../records'))
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = f"{args.model}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
    torch.save(model.state_dict(), os.path.join(out_dir, f'ecg_lightweight_{model_tag}.pt'))
    
    # 计算平均epoch时间
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    
    with open(os.path.join(out_dir, f'eval_report_{model_tag}.json'), 'w') as f:
        json.dump({
            'test_acc': float(test_acc),
            'test_f1': float(test_f1),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'model_metrics': model_metrics,
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_f1s': train_f1s,
                'val_f1s': val_f1s,
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
      # 路径与数据相关
    parser.add_argument('--root', type=str, default='data/ptb/', help='Path to PTB CSV files')
    parser.add_argument('--class-limit', type=int, default=None, help='Optional per-class cap')
    
    # 训练超参数
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    
    # 硬件与策略
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--focal', action='store_true', help='Use Focal Loss to handle imbalance')
    
    # 模型选择
    parser.add_argument('--model', type=str, default='liteecgnet', 
                        choices=['liteecgnet','deepecgnet','ecgnet', 'se_ecgnet','resnet', 'bircnn', 'ldcnn'])

    args = parser.parse_args()
    set_deterministic()
   
    train_loader, val_loader, test_loader, num_classes = make_loaders(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.2,
        val_size=0.1,
        class_limit=args.class_limit,
    )
    try:
        example_data, _ = next(iter(train_loader))
        input_len = int(example_data.shape[2])  # e.g. 187
    except Exception as e:
        print("Warning: cannot fetch batch from train_loader to infer input length:", e)
    input_len = LITEECGNET_CONFIG.get('segment_len', 187)
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
        input_len = 187
        model = PTBLDCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    train_model(model, train_loader, val_loader, test_loader, args)