# Implementation of proposed generic 1d CNN
import torch
import torch.nn as nn

class CNN1DGeneric(nn.Module):
    def __init__(self, output_dim=1, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        if x.ndim == 5 and x.shape[1] == 1:
            x = x.squeeze(1)

        if x.ndim == 4:
            b, c, h, w = x.shape
            x = x.reshape(b, c, h * w)   # (B, 3, 65536)
        elif x.ndim == 2:
            x = x.unsqueeze(1)           # legacy support
        elif x.ndim != 3:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        print(x.shape)
        x = self.features(x)
        x = self.regressor(x)
        return x