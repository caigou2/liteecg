import torch
import torch.nn as nn
import os
# 导入你的模型类
from LiteECGNet1 import LiteECGNet

# 1. 定义配置（必须与你训练 liteecgnet1.pt 时完全一致）
# 根据报错，你的模型权重文件里 classifier 的 shape 是 [4, 128]，所以这里必须设为 4
config = {
    'num_classes': 4,      # 修正为 4 分类
    'fs': 360,             # 采样率
    'segment_len': 720,    # 输入信号长度
    'base_channels': 32    # 基础通道数
}

# 2. 实例化模型
model = LiteECGNet(config)

# 3. 加载权重并过滤冗余键值 (thop 产生的 total_ops 等)
pt_model_path = "liteecgnet1.pt" 
print(f"正在加载权重文件: {pt_model_path}...")

# 加载原始权重字典
state_dict = torch.load(pt_model_path, map_location='cpu')

# 重要：过滤掉所有包含 'total_ops' 或 'total_params' 的键
# 这些键是之前运行 FLOPs 统计工具时产生的，不是模型结构的一部分
clean_state_dict = {
    k: v for k, v in state_dict.items() 
    if "total_ops" not in k and "total_params" not in k
}

# 使用严格模式加载。如果结构代码改对了且权重过滤了，这里应该完美匹配。
try:
    model.load_state_dict(clean_state_dict, strict=True)
    print("权重加载成功！所有参数完全匹配。")
except RuntimeError as e:
    print(f"严格加载失败，尝试非严格模式... 错误预览: {str(e)[:200]}...")
    model.load_state_dict(clean_state_dict, strict=False)

# 4. 设置为评估模式
model.eval()

# 5. 创建虚拟输入 (B=1, C=1, T=720)
dummy_input = torch.randn(1, 1, config['segment_len'])

# 6. 执行导出
# 注意：确保 output_dir 文件夹存在
output_dir = "./onnx_models"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

onnx_file_path = os.path.join(output_dir, "liteecgnet1.onnx")

print(f"开始转换模型为 {onnx_file_path}...")

torch.onnx.export(
    model,                      
    dummy_input,                
    onnx_file_path,             
    export_params=True,         
    opset_version=11,           # 保持 Opset 11 以确保最强 TFLite 兼容性
    do_constant_folding=True,   
    input_names=['input'],      
    output_names=['output'],    
    # 如果后续需要支持变化的 Batch Size，可启用以下代码
    # dynamic_axes={
    #     'input': {0: 'batch_size'},
    #     'output': {0: 'batch_size'}
    # }
)

print(f"转换成功！文件已保存至: {onnx_file_path}")