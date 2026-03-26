import torch
import torch.nn as nn
from config import ECGNET_CONFIG

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsample=None):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        if self.downsample is not None:
            out = self.maxpool(out)
            identity = self.downsample(x)
        out += identity
        return out

class ECGNet1(nn.Module):
    def __init__(self, config=ECGNET_CONFIG):
        super(ECGNet1, self).__init__()
        struct = config['struct']
        in_channels = config['in_channels'] # 默认为 8
        fixed_kernel_size = config['fixed_kernel_size']
        num_classes = config['num_classes']
        self.planes = 16
        
        self.parallel_conv = nn.ModuleList()
        for kernel_size in struct:
            sep_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=kernel_size,
                               stride=1, padding=0, bias=False)
            self.parallel_conv.append(sep_conv)
            
        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=fixed_kernel_size,
                               stride=2, padding=2, bias=False)
        self.block = self._make_layer(kernel_size=fixed_kernel_size, stride=1, padding=8)
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=8, stride=8, padding=2)
        
        # 修改点 1: 启用 batch_first=True
        # 输入期望: [Batch, Time, Channel]
        self.rnn = nn.LSTM(input_size=8, hidden_size=40, num_layers=1, bidirectional=False, batch_first=True)
        
        # 自动计算维度（仅在初始化时运行）
        dummy_input = torch.randn(1, 8, 720) 
        with torch.no_grad():
            cnn_features = self._forward_cnn(dummy_input)
            rnn_features = self._forward_rnn(dummy_input)
            fc_input_dim = cnn_features.shape[1] + rnn_features.shape[1]
        
        self.fc = nn.Linear(in_features=fc_input_dim, out_features=num_classes)

    def _make_layer(self, kernel_size, stride, blocks=15, padding=0):
        layers = []
        downsample = None
        base_width = self.planes
        for i in range(blocks):
            if (i + 1) % 4 == 0:
                downsample = nn.Sequential(
                    nn.Conv1d(in_channels=self.planes, out_channels=self.planes + base_width, kernel_size=1,
                               stride=1, padding=0, bias=False),
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                )
                layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes + base_width, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsample=downsample))
                self.planes += base_width
            elif (i + 1) % 2 == 0:
                downsample = nn.Sequential(
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                )
                layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsample=downsample))
            else:
                downsample = None
                layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsample=downsample))
        return nn.Sequential(*layers)

    def _forward_cnn(self, x):
        out_sep = []
        # 使用 self.parallel_conv 的长度以保证静态
        for i in range(len(self.parallel_conv)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)
        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)
        # 修改点 2: 使用 flatten(1) 固定 Batch 维度之外的所有维度
        out = torch.flatten(out, start_dim=1)
        return out
    
    def _forward_rnn(self, x):
        # 修改点 3: 适配 batch_first=True
        # 输入 x: [B, 8, T] -> 转置为 [B, T, 8]
        x = x.permute(0, 2, 1)
        
        # 显式重置参数（可选，但在某些环境下更稳定）
        self.rnn.flatten_parameters()
        
        # 执行 LSTM
        # rnn_out: [B, T, Hidden], (h_n, c_n): [1, B, Hidden]
        rnn_out, (rnn_h, rnn_c) = self.rnn(x)
        
        # 返回最后一个 hidden state [B, 40]
        return rnn_h[0]

    def forward(self, x):
        # 修改点 4: 静态 Cat 逻辑，强制将 1 通道的 ECG 转换为 8 通道
        # 这种方式导出的 ONNX 节点关系更简单稳定
        if x.shape[1] == 1:
            x = torch.cat([x, x, x, x, x, x, x, x], dim=1)
        
        cnn_features = self._forward_cnn(x)
        rnn_features = self._forward_rnn(x)
        
        # 特征拼接 [B, CNN_DIM + 40]
        new_out = torch.cat([cnn_features, rnn_features], dim=1)
        result = self.fc(new_out)
        return result