import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScale1DCNN(nn.Module):
    def __init__(self, input_feature_dim=3955):
        super(MultiScale1DCNN, self).__init__()

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

        self.conv4_narrow = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4_narrow   = nn.BatchNorm1d(256)
        self.mp2_narrow   = nn.MaxPool1d(kernel_size=2, stride=2)

        seq_len_after_pools = max(1, input_feature_dim // 4)
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

        self.conv4_wide = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn4_wide   = nn.BatchNorm1d(256)
        self.mp2_wide   = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1_wide = nn.Linear(256 * seq_len_after_pools, 100)
        self.fc2_wide = nn.Linear(100, 50)

        # =========================
        # Combined FC + output
        # =========================
        self.fc_combined  = nn.Linear(50 + 50, 100)
        self.output_layer = nn.Linear(100, 1)   # roughness 한 축만 예측

    def forward(self, x):
        # x: (batch, input_feature_dim) -> (batch, 1, L)
        x = x.unsqueeze(1)

        # ----- narrow path -----
        x_n = self.conv1_narrow(x)
        # x_n = self.bn1_narrow(x_n)
        x_n = F.relu(x_n)
        x_n = self.mp1_narrow(x_n)

        x_n = self.conv2_narrow(x_n)
        # x_n = self.bn2_narrow(x_n)
        x_n = F.relu(x_n)

        x_n = self.conv3_narrow(x_n)
        # x_n = self.bn3_narrow(x_n)
        x_n = F.relu(x_n)

        x_n = self.conv4_narrow(x_n)
        # x_n = self.bn4_narrow(x_n)
        x_n = F.relu(x_n)
        x_n = self.mp2_narrow(x_n)

        x_n = x_n.view(x_n.size(0), -1)
        x_n = F.relu(self.fc1_narrow(x_n))
        x_n = F.relu(self.fc2_narrow(x_n))     # (batch, 50)

        # ----- wide path -----
        x_w = self.conv1_wide(x)
        # x_w = self.bn1_wide(x_w)
        x_w = F.relu(x_w)
        x_w = self.mp1_wide(x_w)

        x_w = self.conv2_wide(x_w)
        # x_w = self.bn2_wide(x_w)
        x_w = F.relu(x_w)

        x_w = self.conv3_wide(x_w)
        # x_w = self.bn3_wide(x_w)
        x_w = F.relu(x_w)

        x_w = self.conv4_wide(x_w)
        # x_w = self.bn4_wide(x_w)
        x_w = F.relu(x_w)
        x_w = self.mp2_wide(x_w)

        x_w = x_w.view(x_w.size(0), -1)
        x_w = F.relu(self.fc1_wide(x_w))
        x_w = F.relu(self.fc2_wide(x_w))       # (batch, 50)

        # ----- combine -----
        x_c = torch.cat((x_n, x_w), dim=1)     # (batch, 100)
        x_c = F.relu(self.fc_combined(x_c))    # (batch, 100)

        # 회귀 + Sigmoid
        out = self.output_layer(x_c)           # (batch, 1)
        out = torch.sigmoid(out)               # 논문: Sigmoid activation

        return out