import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LDCNN_CONFIG # 假设您依然使用原有的 config 结构

class LDCNN_PTB(nn.Module):
    """
    LDCNN model architecture specifically for PTB dataset 
    as described in Table 9.
    """
    def __init__(self, config=LDCNN_CONFIG, input_len=187):
        super(LDCNN_PTB, self).__init__()
        
        # 获取类别数，PTB通常是二分类，图中显示输出为1
        self.num_classes = config.get('num_classes', 1)

        # --- Block 1 ---
        # Conv1: Input 1x187 -> Output 16x183 (Kernel 5, Valid)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=0)
        # Conv2: Input 16x183 -> Output 16x179 (Kernel 5, Valid)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5, padding=0)
        # MaxPool 1: 16x179 -> 16x89 (Pool size 2, calculated stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.1)

        # --- Block 2 ---
        # Conv3: 16x89 -> 32x87 (Kernel 3, Valid)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, padding=0)
        # Conv4: 32x87 -> 32x85 (Kernel 3, Valid)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, padding=0)
        # MaxPool 2: 32x85 -> 32x42
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.1)

        # --- Block 3 ---
        # Conv5: 32x42 -> 32x40 (Kernel 3, Valid)
        self.conv5 = nn.Conv1d(32, 32, kernel_size=3, padding=0)
        # Conv6: 32x40 -> 32x38 (Kernel 3, Valid)
        self.conv6 = nn.Conv1d(32, 32, kernel_size=3, padding=0)
        # MaxPool 3: 32x38 -> 32x19
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.1)

        # --- Block 4 ---
        # Conv7: 32x19 -> 256x17 (Kernel 3, Valid)
        self.conv7 = nn.Conv1d(32, 256, kernel_size=3, padding=0)
        # Conv8: 256x17 -> 256x15 (Kernel 3, Valid)
        self.conv8 = nn.Conv1d(256, 256, kernel_size=3, padding=0)
        
        # --- Global Pooling & Dense Layers ---
        # GlobalMaxPool1D: 256x15 -> 256
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 64)
        
        # 根据图中 Output shape 为 1 来设定
        self.classifier = nn.Linear(64, self.num_classes)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        
        # Global Pool & Flatten
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout4(x)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output
        x = self.classifier(x)
        
        # # 如果是二分类且输出维度为1，通常使用 Sigmoid
        # if self.num_classes == 1:
        #     x = self.sigmoid(x)
            
        return x

# 测试模型结构
if __name__ == "__main__":
    # 模拟 PTB 数据集的输入形状 (Batch, Channel, Length)
    model = LDCNN_PTB()
    test_input = torch.randn(8, 1, 187)
    output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}") # 预期应该是 [8, 1]