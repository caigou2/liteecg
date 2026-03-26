import torch
import os
import torch.nn as nn
from models import ECGNet1  # 确保 models.py 在当前目录
from config import ECGNET_CONFIG

def generate_clean_pt(save_path="weights/ecgnet2.pt"):
    """
    生成一个只包含模型权重的干净 .pt 文件
    """
    # 1. 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 2. 初始化模型
    # 确保 config 和训练时一致
    print(">>> 正在初始化干净的模型架构...")
    model = ECGNet1(config=ECGNET_CONFIG)
    
    # 3. 将模型设为 eval 模式（确保 BN/Dropout 状态固定）
    model.eval()

    # 4. 提取 state_dict
    # 只提取神经网络层的权重和偏置，不添加任何 total_ops, total_params
    clean_state_dict = model.state_dict()

    # 5. 直接保存权重字典 (这是最推荐、最轻量的保存方式)
    print(f">>> 正在保存纯净权限到: {save_path}")
    torch.save(clean_state_dict, save_path)
    
    print("-" * 50)
    print("生成成功！这是一个标准的 PyTorch 权重文件。")
    print(f"文件大小: {os.path.getsize(save_path)/1024/1024:.2f} MB")
    print("-" * 50)

if __name__ == "__main__":
    generate_clean_pt()