import torch
import torch.nn as nn
import torch.nn.functional as F

class PTBLDCNN(nn.Module):
    """
    PTB LDCNN model based on the table you provided.

    Architecture summary (from table):
      - Input: (1 x 187)
      - Conv1: 16 filters, k=5, valid -> 16 x 183
      - Conv2: 16 filters, k=5, valid -> 16 x 179
      - MaxPool1: k=2, s=2 -> 16 x 89
      - Dropout1: p=0.1
      - Conv3: 32 filters, k=3, valid -> 32 x 87
      - Conv4: 32 filters, k=3, valid -> 32 x 85
      - MaxPool2: k=2, s=2 -> 32 x 42
      - Dropout2: p=0.1
      - Conv5: 32 filters, k=3 -> 32 x 40
      - Conv6: 32 filters, k=3 -> 32 x 38
      - MaxPool3: k=2, s=2 -> 32 x 19
      - Dropout3: p=0.1
      - Conv7: 256 filters, k=3 -> 256 x 17
      - Conv8: 256 filters, k=3 -> 256 x 15
      - GlobalMaxPool -> 256
      - Dropout4: p=0.2
      - Dense1: 256 -> 64 (ReLU)
      - Dense2: 64 -> 64 (ReLU)
      - Output: 64 -> out_dim (out_dim = 1 for single-logit / binary, or out_dim = num_classes)
    """
    def __init__(self, num_classes: int = 1):
        """
        Args:
            num_classes: If 1 -> single-logit binary (use BCEWithLogitsLoss).
                         If 2 -> 2 logits (CrossEntropyLoss).
                         If >2 -> multiclass (CrossEntropyLoss).
        """
        super().__init__()
        # Convolutional blocks (valid padding)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=0)   # 187 -> 183
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5, padding=0)  # 183 -> 179
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)        # 179 -> 89
        self.drop1 = nn.Dropout(p=0.1)

        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, padding=0)  # 89 -> 87
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, padding=0)  # 87 -> 85
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)        # 85 -> 42
        self.drop2 = nn.Dropout(p=0.1)

        self.conv5 = nn.Conv1d(32, 32, kernel_size=3, padding=0)  # 42 -> 40
        self.conv6 = nn.Conv1d(32, 32, kernel_size=3, padding=0)  # 40 -> 38
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)        # 38 -> 19
        self.drop3 = nn.Dropout(p=0.1)

        self.conv7 = nn.Conv1d(32, 256, kernel_size=3, padding=0) # 19 -> 17
        self.conv8 = nn.Conv1d(256, 256, kernel_size=3, padding=0)# 17 -> 15
        # global max-pool -> (batch, 256)
        self.drop4 = nn.Dropout(p=0.2)

        # Dense head
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 64)

        # Output layer configuration
        # - if num_classes == 1 -> output_dim = 1 (single logit, use BCEWithLogitsLoss)
        # - if num_classes == 2 -> output_dim = 2 (two logits, use CrossEntropyLoss)
        # - if num_classes > 2 -> output_dim = num_classes
        if num_classes == 1:
            out_dim = 1
        else:
            out_dim = int(num_classes)
        self.fc_out = nn.Linear(64, out_dim)

        # small initialization (optional)
        self._init_weights()

    def _init_weights(self):
        # xavier init for conv and linear layers
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (batch, 1, 187)
        returns logits (no sigmoid/softmax)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.drop3(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        # Global max pool over time dimension
        x = torch.max(x, dim=2).values  # shape: (batch, 256)
        x = self.drop4(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc_out(x)

        return out