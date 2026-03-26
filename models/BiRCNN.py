import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BIRCNN_CONFIG

class BiRCNN(nn.Module):
    def __init__(self, config=BIRCNN_CONFIG):
        super(BiRCNN, self).__init__()
        num_classes = config['num_classes']
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=128, kernel_size=36, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=36, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=36, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=16, stride=16, padding=0),
            nn.Dropout(p=0.5, inplace=True)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=128, kernel_size=36, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=36, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=36, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=16, stride=16, padding=0),
            nn.Dropout(p=0.5, inplace=True)
        )
        
        # Dynamically calculate CNN output dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 8, 720)  # Default input length
            cnn_output = self.cnn1(dummy_input)  # [1, 128, T']
            cnn_feature_dim = cnn_output.shape[1]  # Feature dimension is channel number 128
        
        self.rnn1 = nn.LSTM(input_size=cnn_feature_dim, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.25)
        self.rnn2 = nn.LSTM(input_size=cnn_feature_dim, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.25)
        self.rnn3 = nn.LSTM(input_size=11, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.25)
        self.fc = nn.Linear(in_features=256 * 3, out_features=num_classes)
    
    def forward(self, x, HRV=None):
        # x: [B, 1, T], adapted for dataset.py
        if x.shape[1] != 8:
            x = x.repeat(1, 8, 1)
        
        batch_size = x.shape[0]
        
        # Process CNN part - batch processing instead of loops
        out_1 = self.cnn1(x)  # [B, 128, T']
        out_2 = self.cnn2(x)  # [B, 128, T']
        
        # Reshape to RNN input format [seq_len, batch, feature]
        out_1 = out_1.permute(2, 0, 1)  # [T', B, 128]
        out_2 = out_2.permute(2, 0, 1)  # [T', B, 128]
        
        # Pass through RNN
        out_rnn, (out_h, out_c) = self.rnn1(out_1)
        out_1 = torch.cat([out_h[-1, :, :], out_h[-2, :, :]], dim=1)  # [B, 256]
        
        out_rnn, (out_h, out_c) = self.rnn2(out_2)
        out_2 = torch.cat([out_h[-1, :, :], out_h[-2, :, :]], dim=1)  # [B, 256]
        
        if HRV is not None:
            HRV = HRV.permute(2, 0, 1)
            out_rnn, (out_h, out_c) = self.rnn3(HRV)
            out_3 = torch.cat([out_h[-1, :, :], out_h[-2, :, :]], dim=1)
        else:
            out_3 = torch.zeros_like(out_1)  # [B, 256]
        
        out = torch.cat([out_1, out_2, out_3], dim=1)  # [B, 768]
        out = self.fc(out)
        return out



if __name__ == '__main__':
    '''
    According to the original paper, continuous Lb segments of heartbeats are input to the network
    Each heartbeat segment is split based on R-peak and fixed to a certain length
    Here I assume splitting [8, 5000] into 6 heartbeat segments (I don't know the actual number, this is an assumption, can be changed later)
    General splitting process: Split several unequal-length heartbeats based on R-peak, normalize each heartbeat to 721 using the g(x) function from the original paper, remove first and last 180, becomes 361
    This 361 should be consistent with the original paper, mainly about how many heartbeat segments 5000 is split into
    I don't know how to calculate HRV either...
    '''
    input = torch.randn(16, 6, 8, 361)
    HRV = torch.randn(16, 11, 2)
    BiRCNN = BiRCNN()
    output = BiRCNN(input, HRV)
    # print(output.shape)