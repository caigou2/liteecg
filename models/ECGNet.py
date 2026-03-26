import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # 主路径计算
        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.downsample is not None:
            # 健壮性检查：只有当输入长度 > 1 时才执行池化
            if x.shape[-1] > 1:
                out = self.maxpool(out)
                identity = self.downsample(x)
            else:
                # 信号已经缩短到1，跳过所有的池化/下采样层
                # 仅执行卷积（用于匹配通道数），跳过池化
                if isinstance(self.downsample, nn.Sequential):
                    for layer in self.downsample:
                        if not isinstance(layer, (nn.MaxPool1d, nn.AvgPool1d)):
                            identity = layer(identity)
                elif not isinstance(self.downsample, (nn.MaxPool1d, nn.AvgPool1d)):
                    identity = self.downsample(x)
                # 如果 downsample 本身就是池化层且长度为1，则 identity 保持不变
        
        # 最后的对齐保护：确保 identity 和 out 维度完全一致
        if out.shape[-1] != identity.shape[-1]:
            diff = out.shape[-1] - identity.shape[-1]
            if diff > 0:
                identity = F.pad(identity, (0, diff))
            else:
                out = F.pad(out, (0, -diff))

        out += identity
        return out

class ECGNet(nn.Module):
    def __init__(self, config):
        super(ECGNet, self).__init__()
        struct = config['struct']
        in_channels = config['in_channels']
        fixed_kernel_size = config['fixed_kernel_size']
        num_classes = config['num_classes']
        self.planes = 16
        
        concat_planes = self.planes * len(struct)
        
        self.parallel_conv = nn.ModuleList()
        for kernel_size in struct:
            sep_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.planes, 
                                 kernel_size=kernel_size, stride=1, 
                                 padding=kernel_size // 2, bias=False)
            self.parallel_conv.append(sep_conv)
            
        self.bn1 = nn.BatchNorm1d(num_features=concat_planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=concat_planes, out_channels=self.planes, 
                               kernel_size=fixed_kernel_size, stride=2, padding=2, bias=False)
        
        self.block = self._make_layer(kernel_size=fixed_kernel_size, stride=1, padding=8)
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.rnn = nn.LSTM(input_size=8, hidden_size=40, num_layers=1, bidirectional=False)
        
        # 初始化时动态计算维度，加入 eval() 模式保护
        self.eval() 
        with torch.no_grad():
            # 使用较长的 dummy_input (512) 确保初始化时维度充足，避免计算过程中过早降为0
            dummy_input = torch.zeros(1, 8, 512)
            cnn_f = self._forward_cnn(dummy_input)
            rnn_f = self._forward_rnn(dummy_input)
            fc_input_dim = cnn_f.shape[1] + rnn_f.shape[1]
            print(f"ECGNet initialization - FC input dim: {fc_input_dim}")
        self.train() 
        
        self.fc = nn.Linear(in_features=fc_input_dim, out_features=num_classes)

    def _make_layer(self, kernel_size, stride, blocks=15, padding=0):
        layers = []
        base_width = self.planes
        for i in range(blocks):
            if (i + 1) % 4 == 0:
                # 增加通道并下采样
                downsample = nn.Sequential(
                    nn.Conv1d(self.planes, self.planes + base_width, kernel_size=1, stride=1, bias=False),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                )
                layers.append(ResBlock(self.planes, self.planes + base_width, kernel_size, stride, padding, downsample))
                self.planes += base_width
            elif (i + 1) % 2 == 0:
                # 仅下采样
                downsample = nn.Sequential(nn.MaxPool1d(kernel_size=2, stride=2))
                layers.append(ResBlock(self.planes, self.planes, kernel_size, stride, padding, downsample))
            else:
                layers.append(ResBlock(self.planes, self.planes, kernel_size, stride, padding, None))
        return nn.Sequential(*layers)

    def _forward_cnn(self, x):
        out_sep = []
        for conv in self.parallel_conv:
            out_sep.append(conv(x))
        
        min_t = min([s.size(2) for s in out_sep])
        out_sep = [s[:, :, :min_t] for s in out_sep]
        out = torch.cat(out_sep, dim=1) 
        
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.avgpool(out)
        return out.view(out.size(0), -1)
    
    def _forward_rnn(self, x):
        # x shape: [B, 8, T] -> [T, B, 8]
        rnn_out, (rnn_h, rnn_c) = self.rnn(x.permute(2, 0, 1))
        return rnn_h[-1, :, :]

    def forward(self, x):
        # 1. 适配输入通道
        if x.shape[1] != 8:
            x = x.repeat(1, 8, 1)
        
        # 2. 动态 Padding 技术：确保最短长度为 187 (PTB标准)
        # 如果信号非常短，Padding 到 256 以保证卷积池化正常
        min_len = 256
        if x.shape[2] < min_len:
            x = F.pad(x, (0, min_len - x.shape[2]))
        
        # 3. 计算特征
        cnn_features = self._forward_cnn(x)
        rnn_features = self._forward_rnn(x)
        
        # 拼接并分类
        return self.fc(torch.cat([cnn_features, rnn_features], dim=1))