import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import DEEPECGNET_CONFIG

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 7, s: int = 2, p: int = None):
        super().__init__()
        if p is None:
            p = k//2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.residual = (s == 1 and in_ch == out_ch)
    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        if self.residual:
            out = out + x
        return out

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]

class AttentionDenoise(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=dropout)
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        attn_out, _ = self.mha(x, x, x, need_weights=False)
        y = x + self.dropout(self.gamma * attn_out)
        return self.norm(y)

class AttentivePool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.Tanh(),
            nn.Linear(d_model//2, 1)
        )
    def forward(self, x):
        w = self.attn(x)
        w = torch.softmax(w, dim=1)
        pooled = (w * x).sum(dim=1)
        return pooled

class ClassifierHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.pool = AttentivePool(d_model)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_classes)
        )
    def forward(self, x):
        z = self.pool(x)
        logits = self.fc(z)
        return logits

class DeepECGNet(nn.Module):
    def __init__(self, config=DEEPECGNET_CONFIG):
        super().__init__()
        in_ch = config['in_ch']
        base_ch = config['base_ch']
        d_model = config['d_model']
        n_transformer_layers = config['n_transformer_layers']
        nhead = config['nhead']
        dim_ff = config['dim_ff']
        dropout = config['dropout']
        num_classes = config['num_classes']
        
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            ConvBlock(in_ch, base_ch, k=7, s=2),
            ConvBlock(base_ch, base_ch*2, k=5, s=2),
            ConvBlock(base_ch*2, d_model, k=3, s=2),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_transformer_layers)
        self.posenc = PositionalEncoding1D(d_model)
        self.denoise = AttentionDenoise(d_model, nhead=nhead, dropout=dropout)
        self.head = ClassifierHead(d_model, num_classes=num_classes, dropout=dropout)
        
    def forward_features(self, x):
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x = self.posenc(x)
        x = self.transformer(x)
        x = self.denoise(x)
        return x
        
    def forward(self, x):
        feats = self.forward_features(x)
        logits = self.head(feats)
        return logits
