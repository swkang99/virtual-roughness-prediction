# Code for 1D CNN proposed from below 
# Hassan, W., Joolee, J.B. & Jeon, S. 
# Establishing haptic texture attribute space and predicting haptic attributes from image features using 1D-CNN. 
# Sci Rep 13, 11684 (2023). 
# https://doi.org/10.1038/s41598-023-38929-6

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D4HA(nn.Module):
    def __init__(self, input_feature_dim=3955):
        super(CNN1D4HA, self).__init__()

        seq_len_after_pools = max(1, input_feature_dim // 4)

        # =========================
        # Narrow path (kernel=3)
        # =========================
        self.conv1_narrow = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_narrow   = nn.BatchNorm1d(32)
        self.mp1_narrow   = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2_narrow = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_narrow   = nn.BatchNorm1d(64)

        self.conv3_narrow = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3_narrow   = nn.BatchNorm1d(128)

        self.conv4_narrow = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_narrow   = nn.BatchNorm1d(128)

        self.conv5_narrow = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5_narrow   = nn.BatchNorm1d(256)
        self.mp2_narrow   = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1_narrow = nn.Linear(256 * seq_len_after_pools, 100)
        self.fc2_narrow = nn.Linear(100, 50)

        # =========================
        # Wide path (kernel=5)
        # =========================
        self.conv1_wide = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1_wide   = nn.BatchNorm1d(32)
        self.mp1_wide   = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2_wide = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2_wide   = nn.BatchNorm1d(64)

        self.conv3_wide = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3_wide   = nn.BatchNorm1d(128)

        self.conv4_wide = nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2)
        self.bn4_wide   = nn.BatchNorm1d(128)

        self.conv5_wide = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn5_wide   = nn.BatchNorm1d(256)
        self.mp2_wide   = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1_wide = nn.Linear(256 * seq_len_after_pools, 100)
        self.fc2_wide = nn.Linear(100, 50)

        # =========================
        # Combined FC + output
        # =========================
        self.fc_combined  = nn.Linear(50 + 50, 100)
        # self.output_layer = nn.Linear(100, 1)   # roughness 한 축만 예측
        self.output_layer = nn.Linear(100, 4)   # HA 4개 모두 예측

    def forward(self, x):
        # x: (batch, input_feature_dim) -> (batch, channels=1, L)
        x = x.unsqueeze(1)

        # ----- narrow path -----
        x_n = F.relu(self.conv1_narrow(x))
        x_n = self.bn1_narrow(x_n)

        x_n = self.mp1_narrow(x_n)

        x_n = F.relu(self.conv2_narrow(x_n))
        x_n = self.bn2_narrow(x_n)
        
        x_n = F.relu(self.conv3_narrow(x_n))
        x_n = self.bn3_narrow(x_n)
        
        x_n = F.relu(self.conv4_narrow(x_n))
        x_n = self.bn4_narrow(x_n)

        x_n = F.relu(self.conv5_narrow(x_n))
        x_n = self.bn5_narrow(x_n)

        x_n = self.mp2_narrow(x_n)

        x_n = self.flatten(x_n)
        x_n = F.relu(self.fc1_narrow(x_n))
        x_n = F.relu(self.fc2_narrow(x_n))     # (batch, 50)

        # ----- wide path -----
        x_w = F.relu(self.conv1_wide(x))
        x_w = self.bn1_wide(x_w)

        x_w = self.mp1_wide(x_w)

        x_w = F.relu(self.conv2_wide(x_w))
        x_w = self.bn2_wide(x_w)

        x_w = F.relu(self.conv3_wide(x_w))
        x_w = self.bn3_wide(x_w)

        x_w = F.relu(self.conv4_wide(x_w))
        x_w = self.bn4_wide(x_w)
        
        x_w = F.relu(self.conv5_wide(x_w))
        x_w = self.bn5_wide(x_w)

        x_w = self.mp2_wide(x_w)

        x_w = self.flatten(x_w)
        x_w = F.relu(self.fc1_wide(x_w))
        x_w = F.relu(self.fc2_wide(x_w))       # (batch, 50)

        # ----- combine -----
        x_c = torch.cat((x_n, x_w), dim=1)     # (batch, 100)
        x_c = F.relu(self.fc_combined(x_c))    # (batch, 100)

        # 회귀 + Sigmoid
        out = self.output_layer(x_c)           # (batch, 1)
        out = torch.sigmoid(out)               # 논문: Sigmoid activation

        return out