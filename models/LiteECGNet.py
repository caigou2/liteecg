import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import LITEECGNET_CONFIG

# ==========================================
# 1. 频率特征提取层 (SincConv1d)
# ==========================================
class SincConv1d(nn.Module):
    """Sinc-based 1D convolution (learnable band-pass)."""
    def __init__(self, out_channels: int, kernel_size: int, sample_rate: int = 360,
                 min_low_hz: float = 0.5, min_band_hz: float = 1.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = torch.linspace(0, 40, out_channels) + min_low_hz
        band_hz = torch.ones(out_channels) * (40 - min_low_hz)
        self.low_hz_ = nn.Parameter(low_hz)
        self.band_hz_ = nn.Parameter(band_hz)

        n = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float()
        self.register_buffer('n', n)
        self.window_ = torch.hamming_window(kernel_size, periodic=False)

    def forward(self, x):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        band = self.min_band_hz + torch.abs(self.band_hz_)
        high = torch.clamp(low + band, max=self.sample_rate / 2 - 1.0)
        f1 = low / self.sample_rate
        f2 = high / self.sample_rate

        n = self.n.to(x.device)
        window = self.window_.to(x.device)
        
        filters = []
        for i in range(self.out_channels):
            f1i, f2i = f1[i], f2[i]
            h = 2 * f2i * torch.sinc(2 * f2i * n) - 2 * f1i * torch.sinc(2 * f1i * n)
            h = h * window
            filters.append(h)
        h = torch.stack(filters).unsqueeze(1)
        return F.conv1d(x, h, stride=1, padding=self.kernel_size // 2, bias=None)

# ==========================================
# 2. 注意力与轻量化组件
# ==========================================
class SE_Lite(nn.Module):
    """Lightweight Squeeze-and-Excitation module"""
    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels // r, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // r, channels, kernel_size=1)

    def forward(self, x):
        s = x.mean(dim=-1, keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class TAG(nn.Module):
    """Temporal Attention Gate"""
    def __init__(self, k: int = 9):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=True)

    def forward(self, x):
        a = x.mean(dim=1, keepdim=True)
        a = torch.sigmoid(self.conv(a))
        return x * a

class CrossChannelAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.fc = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        s = x.mean(dim=-1, keepdim=True)
        s = torch.sigmoid(self.fc(s))
        return x * s

class TemporalAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.attn = nn.Conv1d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        attn_weights = F.softmax(self.attn(x), dim=-1)
        return x * attn_weights

# ==========================================
# 3. 多尺度空洞卷积块 (核心改进)
# ==========================================
class MDS_DSC_Block(nn.Module):
    def __init__(self, cin: int, cout: int, expand: int = 2, k: int = 5, stride: int = 1, dilation_base: int = 1):
        super().__init__()
        hidden = cin * expand
        self.pw_expand = nn.Conv1d(cin, hidden, kernel_size=1, bias=False)

        # 动态计算 Padding 确保不同空洞率下的输出长度一致，防止报错
        pad1 = ((k - 1) * dilation_base) // 2
        pad2 = ((k - 1) * (dilation_base * 2)) // 2

        self.dw1 = nn.Conv1d(hidden, hidden, kernel_size=k, stride=stride, padding=pad1,
                             dilation=dilation_base, groups=hidden, bias=False)
        self.dw2 = nn.Conv1d(hidden, hidden, kernel_size=k, stride=stride, padding=pad2,
                             dilation=dilation_base * 2, groups=hidden, bias=False)
        
        self.bn_dw1 = nn.BatchNorm1d(hidden)
        self.bn_dw2 = nn.BatchNorm1d(hidden)

        self.pw_fuse = nn.Conv1d(hidden * 2, hidden, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm1d(hidden)
        self.pw_proj = nn.Conv1d(hidden, cout, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm1d(cout)
        self.se = SE_Lite(cout, r=8)

        self.use_skip = (stride == 1 and cin == cout)
        if not self.use_skip:
            self.skip = nn.Conv1d(cin, cout, kernel_size=1, stride=stride, bias=False)
            self.bn_skip = nn.BatchNorm1d(cout)

    def forward(self, x):
        z = F.silu(self.pw_expand(x))
        u1 = F.silu(self.bn_dw1(self.dw1(z)))
        u2 = F.silu(self.bn_dw2(self.dw2(z)))
        u = torch.cat([u1, u2], dim=1)
        u = F.silu(self.bn_fuse(self.pw_fuse(u)))
        v = self.bn_proj(self.pw_proj(u))
        v = self.se(v)
        if self.use_skip:
            return v + x
        else:
            return v + self.bn_skip(self.skip(x))

# ==========================================
# 4. LiteECGNet 主网络
# ==========================================
class LiteECGNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_classes = config['num_classes']
        fs = config['fs']
        segment_len = config['segment_len']
        base_channels = config['base_channels']
        
        # Stem
        self.sinc = SincConv1d(out_channels=base_channels, kernel_size=31, sample_rate=fs)
        self.conv7 = nn.Conv1d(1, base_channels*2, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn7 = nn.BatchNorm1d(base_channels*2)
        self.pw32 = nn.Conv1d(base_channels*3, base_channels, kernel_size=1, bias=False)
        self.bn32 = nn.BatchNorm1d(base_channels)

        # 注册位置编码 (Sinusoidal Positional Encoding)
        pe_len = segment_len // 2
        pe = self._get_sinusoidal_encoding(pe_len, base_channels)
        self.register_buffer('pe', pe)

        # Stages: 逐层增加 Dilation 以扩大感受野，捕捉 RR 间期
        self.stage1 = MDS_DSC_Block(base_channels, base_channels*2, expand=2, k=5, stride=2, dilation_base=1)
        self.stage2 = MDS_DSC_Block(base_channels*2, base_channels*3, expand=2, k=7, stride=2, dilation_base=2)
        self.stage3 = MDS_DSC_Block(base_channels*3, base_channels*4, expand=2, k=9, stride=1, dilation_base=4)

        self.tag = TAG(k=9)
        self.cross_channel_attn = CrossChannelAttention(in_channels=base_channels*4)
        self.temporal_attn = TemporalAttention(in_channels=base_channels*4)
        
        self.dropout = nn.Dropout(p=0.1)
        self.num_flat_features = self._get_flatten_size(segment_len)
        self.classifier = nn.Linear(self.num_flat_features, num_classes)

    def _get_sinusoidal_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.transpose(0, 1).unsqueeze(0) # [1, d_model, seq_len]

    def _get_flatten_size(self, segment_len):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, segment_len)
            out = self.forward_features(dummy)
            return out.numel() // out.size(0)

    def forward_features(self, x):
        s1 = self.sinc(x)
        s1 = F.avg_pool1d(s1, kernel_size=2, stride=2)
        s2 = F.silu(self.bn7(self.conv7(x)))
        
        # 对齐拼接
        if s1.size(2) < s2.size(2):
            s1 = F.pad(s1, (0, s2.size(2) - s1.size(2)))
        elif s2.size(2) < s1.size(2):
            s2 = F.pad(s2, (0, s1.size(2) - s2.size(2)))
            
        s = torch.cat([s1, s2], dim=1)
        s = F.silu(self.bn32(self.pw32(s)))

        # 注入位置编码
        s = s + self.pe[:, :, :s.size(2)]

        z = self.stage3(self.stage2(self.stage1(s)))
        z = self.tag(z)
        z = self.cross_channel_attn(z)
        z = self.temporal_attn(z)
        return z

    def forward(self, x):
        z = self.forward_features(x)
        z_flatten = z.view(z.size(0), -1)
        feat = self.dropout(z_flatten)
        return self.classifier(feat)