import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LDCNN_CONFIG

class LDCNN(nn.Module):
    """LDCNN: 5-layer convolution + pooling + fully connected, adapted for MIT-BIH classification"""
    def __init__(self, config=LDCNN_CONFIG, input_len=None):
        super().__init__()
        num_classes = config['num_classes']
        if input_len is None:
            input_len = config['input_len']
            
        self.conv1 = nn.Conv1d(1, 16, kernel_size=13, padding=6)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, padding=7)
        self.pool2 = nn.AvgPool1d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=17, padding=8)
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=19, padding=9)
        self.pool4 = nn.AvgPool1d(kernel_size=3, stride=2)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=21, padding=10)
        self.pool5 = nn.AvgPool1d(kernel_size=3, stride=2)
        
        # Calculate flatten dimensions
        dummy = torch.zeros(1, 1, input_len)
        with torch.no_grad():
            x = self.pool1(F.relu(self.conv1(dummy)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            x = self.pool4(F.relu(self.conv4(x)))
            x = self.pool5(F.relu(self.conv5(x)))
            flat_dim = x.shape[1] * x.shape[2]
            
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(flat_dim, 35)
        self.fc2 = nn.Linear(35, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
