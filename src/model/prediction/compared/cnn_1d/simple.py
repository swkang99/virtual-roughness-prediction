# Implementation of proposed simple 1d CNN
import torch
import torch.nn as nn


class CNN1DSimple(nn.Module):
    def __init__(self, output_dim=1, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=9, stride=1, padding=4),
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

    def forward(self, texture_image, height_map, normal_map):
        inputs = [texture_image, height_map, normal_map]
        processed = []

        for x in inputs:
            if x.ndim == 2:
                x = x.unsqueeze(1)   # legacy support: (B, L) -> (B, 1, L)

            elif x.ndim == 3:
                # (B, H, W) -> (B, 1, H, W)
                x = x.unsqueeze(1)

            elif x.ndim == 5 and x.shape[1] == 1:
                x = x.squeeze(1)

            elif x.ndim != 4:
                raise ValueError(f"Unsupported input shape: {x.shape}")

            processed.append(x)

        x = torch.cat(processed, dim=1)   # (B, 3, H, W)

        if x.ndim == 4:
            b, c, h, w = x.shape
            x = x.reshape(b, c, h * w)    # (B, 3, H*W)
        elif x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"Unsupported concatenated shape: {x.shape}")

        x = self.features(x)
        x = self.regressor(x)
        return x