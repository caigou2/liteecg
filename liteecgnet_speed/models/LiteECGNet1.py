import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x):  # x: [B,1,T]
        low = self.min_low_hz + torch.abs(self.low_hz_)
        band = self.min_band_hz + torch.abs(self.band_hz_)
        high = torch.clamp(low + band, max=self.sample_rate / 2 - 1.0)
        f1 = low / self.sample_rate
        f2 = high / self.sample_rate

        n = self.n.to(x.device)
        window = self.window_.to(x.device)
        
        # 优化：使用向量化操作代替 for 循环，生成更干净的 ONNX 图
        f1 = f1.view(-1, 1, 1)
        f2 = f2.view(-1, 1, 1)
        n = n.view(1, 1, -1)
        
        h = 2 * f2 * torch.sinc(2 * f2 * n) - 2 * f1 * torch.sinc(2 * f1 * n)
        h = h * window.view(1, 1, -1)
        
        # h: [Cout, 1, K]
        return F.conv1d(x, h, stride=1, padding=self.kernel_size // 2, bias=None)


class SE_Lite(nn.Module):
    """Lightweight Squeeze-and-Excitation module"""
    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels // r, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // r, channels, kernel_size=1)

    def forward(self, x):  # [B,C,T]
        s = x.mean(dim=-1, keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class MDS_DSC_Block(nn.Module):
    """
    Multi-scale Depthwise Separable Conv Block (1D)
    已修复 TFLite 不支持 Stride > 1 且 Dilation > 1 的问题
    """
    def __init__(self, cin: int, cout: int, expand: int = 2, k: int = 5, stride: int = 1):
        super().__init__()
        hidden = cin * expand
        self.stride = stride
        self.pw_expand = nn.Conv1d(cin, hidden, kernel_size=1, bias=False)

        # dw1: dilation=1, 保持原始 stride
        pad1 = ((k - 1) // 2) * 1
        self.dw1 = nn.Conv1d(hidden, hidden, kernel_size=k, stride=stride, padding=pad1,
                              dilation=1, groups=hidden, bias=False)
        self.bn_dw1 = nn.BatchNorm1d(hidden)

        # dw2: dilation=2, TFLite 不支持 stride > 1
        # 解决方案：强制让卷积 stride=1，然后单独用池化做下采样
        pad2 = ((k - 1) // 2) * 2
        self.dw2 = nn.Conv1d(hidden, hidden, kernel_size=k, stride=1, padding=pad2,
                              dilation=2, groups=hidden, bias=False)
        self.bn_dw2 = nn.BatchNorm1d(hidden)
        
        # 如果步长 > 1，为 dw2 后续单独设置下采样
        if self.stride > 1:
            # 这里的 AvgPool1d kernel 设为 1，stride 设为 2，纯粹为了丢弃像素降采样
            self.dw2_downsample = nn.AvgPool1d(kernel_size=1, stride=stride)
        else:
            self.dw2_downsample = nn.Identity()

        self.pw_fuse = nn.Conv1d(hidden*2, hidden, kernel_size=1, bias=False)
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
        
        # u1 路径：卷积自带 stride
        u1 = F.silu(self.bn_dw1(self.dw1(z))) 
        
        # u2 路径：卷积 stride=1，手动下采样
        u2 = self.dw2(z)
        u2 = self.dw2_downsample(u2) # 关键修复点
        u2 = F.silu(self.bn_dw2(u2))
        
        # 拼接并在通道上融合
        u = torch.cat([u1, u2], dim=1)
        u = F.silu(self.bn_fuse(self.pw_fuse(u)))
        
        v = self.bn_proj(self.pw_proj(u))
        v = self.se(v)
        
        if self.use_skip:
            out = v + x
        else:
            out = v + self.bn_skip(self.skip(x))
        return out


class CrossChannelAttention(nn.Module):
    """Cross-channel attention mechanism"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.fc = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):  # x: [B, C, T]
        channel_attention = torch.mean(x, dim=-1, keepdim=True)
        channel_attention = torch.sigmoid(self.fc(channel_attention))
        return x * channel_attention


class TAG(nn.Module):
    """Temporal Attention Gate"""
    def __init__(self, k: int = 9):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=True)

    def forward(self, x):  # [B,C,T]
        a = x.mean(dim=1, keepdim=True)
        a = torch.sigmoid(self.conv(a))
        return x * a


class TemporalAttention(nn.Module):
    """Cross-timestep self-attention mechanism"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.attn = nn.Conv1d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        attn_weights = F.softmax(self.attn(x), dim=-1)
        return x * attn_weights


class LiteECGNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_classes = config['num_classes']
        fs = config['fs']
        base_channels = config['base_channels']
        
        self.sinc = SincConv1d(out_channels=base_channels, kernel_size=31, sample_rate=fs)
        self.conv7 = nn.Conv1d(1, base_channels*2, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn7 = nn.BatchNorm1d(base_channels*2)
        
        self.pw32 = nn.Conv1d(base_channels*3, base_channels, kernel_size=1, bias=False)
        self.bn32 = nn.BatchNorm1d(base_channels)
        
        # 所有的 MDS_DSC_Block 内部现在都已修复
        self.stage1 = MDS_DSC_Block(base_channels, base_channels*2, expand=2, k=5, stride=2)
        self.stage2 = MDS_DSC_Block(base_channels*2, base_channels*3, expand=2, k=5, stride=2)
        self.stage3 = MDS_DSC_Block(base_channels*3, base_channels*4, expand=2, k=5, stride=1)
        
        self.tag = TAG(k=9)
        self.dropout = nn.Dropout(p=0.1)
        self.cross_channel_attn = CrossChannelAttention(in_channels=base_channels*4)
        self.temporal_attn = TemporalAttention(in_channels=base_channels*4)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x):
        # Stem
        s1 = self.sinc(x)
        s1 = F.avg_pool1d(s1, kernel_size=2, stride=2)
        s2 = F.silu(self.bn7(self.conv7(x)))
        
        # 对齐长度（防止由于奇偶长度导致的 padding 差异导致无法 concat）
        t1, t2 = s1.size(2), s2.size(2)
        if t1 < t2:
            s1 = F.pad(s1, (0, t2 - t1))
        elif t2 < t1:
            s2 = F.pad(s2, (0, t1 - t2))
            
        s = torch.cat([s1, s2], dim=1)
        s = F.silu(self.bn32(self.pw32(s)))

        # Stages
        z = self.stage1(s)
        z = self.stage2(z)
        z = self.stage3(z)

        # Attention
        z = self.tag(z)
        z = self.cross_channel_attn(z)
        z = self.temporal_attn(z)

        # Head
        z = self.adaptive_pool(z)
        z_flatten = z.view(z.size(0), -1)
        feat = self.dropout(z_flatten)
        logits = self.classifier(feat)
        return logits