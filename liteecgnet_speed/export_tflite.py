import torch
import ai_edge_torch
from models import ECGNet1

# 1. 加载模型
model = ECGNet1()
model.load_state_dict(torch.load("weights/ecgnet1.pt", map_location="cpu"), strict=False)
model.eval()

# 2. 定义静态输入
sample_input = torch.randn(1, 1, 720)

# 3. 核心：直接转换为 TFLite
# ai_edge_torch 能自动处理 LSTM 的 Rank 和批次序问题
print(">>> 正在使用 ai_edge_torch 转换模型...")
edge_model = ai_edge_torch.convert(model, (sample_input,))
edge_model.export("ecgnet1.tflite")
print(">>> 转换成功: ecgnet1.tflite")