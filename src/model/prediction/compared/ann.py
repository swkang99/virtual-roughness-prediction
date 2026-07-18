# Code for Artificial neural network

import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, 100)

        self.fc3 = nn.Linear(100, 1)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x