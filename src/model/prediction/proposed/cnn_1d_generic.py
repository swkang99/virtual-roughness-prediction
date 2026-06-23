# Implementation of proposed generic 1d CNN
import torch
import torch.nn as nn

class CNN1DGeneric(nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout=0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
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
        # x: (batch, input_dim)
        x = x.unsqueeze(1)         # (batch, 1, input_dim)
        x = self.features(x)       # (batch, 128, 1)
        x = self.regressor(x)      # (batch, 1)
        return x