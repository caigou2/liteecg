import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LITEECGNET_CONFIG


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
        # build filters
        filters = []
        for i in range(self.out_channels):
            f1i, f2i = f1[i], f2[i]
            h = 2 * f2i * torch.sinc(2 * f2i * n) - 2 * f1i * torch.sinc(2 * f1i * n)
            h = h * window
            filters.append(h)
        h = torch.stack(filters).unsqueeze(1)  # [Cout, 1, K]
        return F.conv1d(x, h, stride=1, padding=self.kernel_size // 2, bias=None)


class SE_Lite(nn.Module):
    """Lightweight Squeeze-and-Excitation module"""
    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels // r, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // r, channels, kernel_size=1)

    def forward(self, x):  # [B,C,T]
        s = x.mean(dim=-1, keepdim=True) # [B, C, 1]
        s = F.silu(self.fc1(s)) # [B, C/r, 1]
        s = torch.sigmoid(self.fc2(s)) # [B, C, 1]
        return x * s


class MDS_DSC_Block(nn.Module):
    """Multi-scale Depthwise Separable Conv Block (1D)"""
    def __init__(self, cin: int, cout: int, expand: int = 2, k: int = 5, stride: int = 1):
        super().__init__()
        hidden = cin * expand
        self.pw_expand = nn.Conv1d(cin, hidden, kernel_size=1, bias=False)

        pad1 = ((k - 1) // 2) * 1
        pad2 = ((k - 1) // 2) * 2
        self.dw1 = nn.Conv1d(hidden, hidden, kernel_size=k, stride=stride, padding=pad1,
                              dilation=1, groups=hidden, bias=False)
        self.dw2 = nn.Conv1d(hidden, hidden, kernel_size=k, stride=stride, padding=pad2,
                              dilation=2, groups=hidden, bias=False)
        self.bn_dw1 = nn.BatchNorm1d(hidden)
        self.bn_dw2 = nn.BatchNorm1d(hidden)

        self.pw_fuse = nn.Conv1d(hidden*2, hidden, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm1d(hidden)
        self.pw_proj = nn.Conv1d(hidden, cout, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm1d(cout)
        self.se = SE_Lite(cout, r=8)

        self.use_skip = (stride == 1 and cin == cout)
        if not self.use_skip:
            self.skip = nn.Conv1d(cin, cout, kernel_size=1, stride=stride, bias=False)
            self.bn_skip = nn.BatchNorm1d(cout)

    def forward(self, x):  # [B,Cin,T]
        z = F.silu(self.pw_expand(x))
        u1 = F.silu(self.bn_dw1(self.dw1(z))) # [B, hidden, T/2], hidden = 2*Cin
        u2 = F.silu(self.bn_dw2(self.dw2(z))) # [B, hidden, T/2], hidden = 2*Cin
        u = torch.cat([u1,u2], dim=1)
        u = F.silu(self.bn_fuse(self.pw_fuse(u))) # [B, hidden, T/2], hidden = 2*Cin
        v = self.bn_proj(self.pw_proj(u)) # [B, cout, T]
        v = self.se(v) # [B, cout, T]
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
        # Calculate weights for each channel
        channel_attention = torch.mean(x, dim=-1, keepdim=True)  # [B, C, 1]
        channel_attention = torch.sigmoid(self.fc(channel_attention))  # [B, C, 1]
        return x * channel_attention  # [B, C, T]


class TAG(nn.Module):
    """Temporal Attention Gate: channel-agnostic time-wise gate with a tiny Conv1d."""
    def __init__(self, k: int = 9):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=True)

    def forward(self, x):  # [B,C,T]
        a = x.mean(dim=1, keepdim=True)  # [B,1,T]
        a = torch.sigmoid(self.conv(a))  # [B,1,T]
        return x * a  # broadcast gate


class TemporalAttention(nn.Module):
    """Cross-timestep self-attention mechanism"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.attn = nn.Conv1d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):  # x: [B, C, T]
        attn_weights = F.softmax(self.attn(x), dim=-1)  # [B, 1, T]
        return x * attn_weights  # Get timestep features through weighting


class LiteECGNet(nn.Module):
    """ECG lightweight anomaly detection network (optimized version with cross-channel attention at the end)"""
    def __init__(self, config):
        super().__init__()
        num_classes = config['num_classes']
        fs = config['fs']
        segment_len = config['segment_len']
        base_channels = config['base_channels']
        self.fs = fs
        self.segment_len = segment_len
        # Stem: SincConv (8) + Conv7/2 -> concat -> PW to 32
        self.sinc = SincConv1d(out_channels=base_channels, kernel_size=31, sample_rate=fs)
        self.conv7 = nn.Conv1d(1, base_channels*2, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn7 = nn.BatchNorm1d(base_channels*2)
        self.pw32 = nn.Conv1d(base_channels*3, base_channels, kernel_size=1, bias=False)
        self.bn32 = nn.BatchNorm1d(base_channels)
        # Stages
        self.stage1 = MDS_DSC_Block(base_channels, base_channels*2, expand=2, k=5, stride=2)  # T/4
    
        self.stage2 = MDS_DSC_Block(base_channels*2, base_channels*3, expand=2, k=5, stride=2)  # T/8
        self.stage3 = MDS_DSC_Block(base_channels*3, base_channels*4, expand=2, k=5, stride=1) # T/8
        self.tag = TAG(k=9)
        self.dropout = nn.Dropout(p=0.1)
        # Add cross-channel attention
        self.cross_channel_attn = CrossChannelAttention(in_channels=base_channels*4)
        # Cross-timestep self-attention
        self.temporal_attn = TemporalAttention(in_channels=base_channels*4)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(base_channels * 4, num_classes)
        #self.classifier = nn.Linear(base_channels*4 * (self.segment_len // 8), num_classes)

    def forward(self, x):  # x: [B,1,T]
        # Initial feature extraction through SincConv and Conv7/2
        s1 = self.sinc(x) # [B,32,T]
        s1 = F.avg_pool1d(s1, kernel_size=2, stride=2) # [B,32,T/2]
        s2 = F.silu(self.bn7(self.conv7(x))) # [B,64,T/2]
        t1 = s1.size(2)
        t2 = s2.size(2)
        if t1 < t2:
        # pad format: (pad_left, pad_right) for 1D padding on last dim
         s1 = F.pad(s1, (0, t2 - t1), mode='constant', value=0)
        elif t2 < t1:
            s2 = F.pad(s2, (0, t1 - t2), mode='constant', value=0)
        s = torch.cat([s1, s2], dim=1)  # [B,96,T/2]

        # Channel fusion through Pointwise convolution
        s = F.silu(self.bn32(self.pw32(s)))  # [B,32,T/2]

        # Multi-scale convolution block processing
        z = self.stage1(s)  # [B,64,T/4]
        z = self.stage2(z)  # [B,96,T/8]
        z = self.stage3(z)  # [B,128,T/8]

        # Timestep weighting through TAG
        z = self.tag(z)     # [B,128,T/8]

        # Cross-channel attention mechanism, focusing on important channels
        z = self.cross_channel_attn(z)  # [B,128,T/8]

        # Cross-timestep self-attention mechanism, enhancing temporal features
        z = self.temporal_attn(z)  # [B,128,T/8]

        z = self.adaptive_pool(z)
        z_flatten = z.view(z.size(0), -1)  # [B, 128*T/8]
        feat = self.dropout(z_flatten)

        # Classification layer
        logits = self.classifier(feat)  # [B, num_classes]
        return logits
