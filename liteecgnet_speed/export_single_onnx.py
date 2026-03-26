import torch
from models import ECGNet1

model = ECGNet1()
# 注意：一定要用 map_location='cpu' 并在加载后 model.eval()
model.load_state_dict(torch.load("weights/ecgnet2.pt", map_location='cpu'), strict=False)
model.eval()

dummy_input = torch.randn(1, 1, 720)
torch.onnx.export(
    model, 
    dummy_input, 
    "ecgnet2_static.onnx",
    opset_version=13, # opset 13 配合 batch_first 是最稳的
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)