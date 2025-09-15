# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CHANNELS, CNN_OUT_CHANNELS, LSTM_HIDDEN, LSTM_LAYERS, ATTN_LAYERS, ATTN_HEADS, DROPOUT, BLANK_IDX

class SmallCNN(nn.Module):
    """
    CNN encoder: reduces height dimension and produces a sequence along width.
    We design conv layers to downsample height to 1 (or near 1), then treat width as time.
    """
    def __init__(self, in_channels=1, out_channels=CNN_OUT_CHANNELS):
        super().__init__()
        layers = []
        ch = in_channels
        cfg = [64, 128, 256, 256, 512, out_channels]
        for i, nc in enumerate(cfg):
            layers.append(nn.Conv2d(ch, nc, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(nc))
            layers.append(nn.ReLU(inplace=True))
            # apply pooling on alternate layers to reduce H and W
            if i in [0,1,3,4]:
                layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
            ch = nc
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, H, W]
        features = self.cnn(x)  # [B, C', H', W']
        return features

class StackedBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout)

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.rnn(x)
        return out  # [B, T, 2*hidden]

class AttentionStack(nn.Module):
    """
    Stacked Multi-Head Self-Attention layers applied to sequence outputs from BiLSTM.
    This enhances per-timestep features before classification.
    """
    def __init__(self, embed_dim, num_layers=ATTN_LAYERS, num_heads=ATTN_HEADS, dropout=DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            norm1 = nn.LayerNorm(embed_dim)
            ff = nn.Sequential(
                nn.Linear(embed_dim, embed_dim*4),
                nn.ReLU(),
                nn.Linear(embed_dim*4, embed_dim),
            )
            norm2 = nn.LayerNorm(embed_dim)
            self.layers.append(nn.ModuleDict({"attn": attn, "norm1": norm1, "ff": ff, "norm2": norm2}))

    def forward(self, x):
        # x: [B, T, E]
        for l in self.layers:
            attn_out, _ = l["attn"](x, x, x, need_weights=False)
            x = l["norm1"](x + attn_out)
            ff_out = l["ff"](x)
            x = l["norm2"](x + ff_out)
        return x

class CRNNWithAttention(nn.Module):
    def __init__(self, num_classes, in_channels=1, cnn_out_channels=CNN_OUT_CHANNELS, lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS, attn_layers=ATTN_LAYERS, attn_heads=ATTN_HEADS, dropout=DROPOUT):
        super().__init__()
        self.cnn = SmallCNN(in_channels, cnn_out_channels)
        # We'll compute feature dimension after collapsing height
        # After CNN, feature shape: [B, C', H', W'] -> we will collapse H'
        self.lstm_in = cnn_out_channels  # after pooling, we will combine height into channels
        self.bi_lstm = StackedBiLSTM(self.lstm_in, hidden_size=lstm_hidden, num_layers=lstm_layers, dropout=dropout)
        self.attn_stack = AttentionStack(embed_dim=2*lstm_hidden, num_layers=attn_layers, num_heads=attn_heads, dropout=dropout)
        self.classifier = nn.Linear(2*lstm_hidden, num_classes)  # outputs logits per timestep

    def forward(self, x):
        # x: [B, C, H, W]
        feats = self.cnn(x)  # [B, C', H', W']
        B, Cc, Hc, Wc = feats.size()
        # collapse height dimension by squeezing or pooling
        # average pool across height -> [B, C', W']
        if Hc > 1:
            feats = feats.mean(dim=2)  # [B, C', W']
        else:
            feats = feats.squeeze(2)
        # transpose to sequence: [B, T=Wc, F=C']
        seq = feats.permute(0, 2, 1)
        # pass through stacked BiLSTM -> [B, T, 2*hidden]
        rnn_out = self.bi_lstm(seq)
        # pass through attention stack
        attn_out = self.attn_stack(rnn_out)
        logits = self.classifier(attn_out)  # [B, T, num_classes]
        # For CTC loss in PyTorch, expected shape is (T, N, C)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs  # [B, T, C]
