# Code for 1D CNN proposed from below 
# Taye, G.T., Hwang, HJ. & Lim, K.M. 
# Application of a convolutional neural network for predicting the occurrence of ventricular tachyarrhythmia using heart rate variability features. 
# Sci Rep 10, 6769 (2020). 
# https://doi.org/10.1038/s41598-020-63566-8

import torch
import torch.nn as nn


class CNN1DScirep(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=102),
            nn.BatchNorm1d(3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=3, out_channels=10, kernel_size=24),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=11),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=9),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, feature_dim)
            flat_dim = self.features(dummy).flatten(1).shape[1]

        out_dim = 1

        self.regressor = nn.Sequential(
            nn.Linear(flat_dim, 22),
            nn.ReLU(),
            nn.Linear(22, 22),
            nn.ReLU(),
            nn.Linear(22, out_dim),
        )

    def forward(self, x):
        # x: [B, L] from FeatureDataset
        if x.dim() == 2:
            x = x.unsqueeze(1)   # [B, 1, L]

        x = self.features(x)
        x = self.flatten(x)
        x = self.regressor(x)
        return x