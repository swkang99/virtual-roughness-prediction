# Code for Artificial neural network

import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self, conf, input_dim):
        super().__init__()
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3020, 200)
        self.fc2 = nn.Linear(200, 100)

        if conf['dataset_output'] == 'roughness':
            self.fc3 = nn.Linear(100, 1)  
        elif conf['dataset_output'] == 'four_HAs':
            self.fc3 = nn.Linear(100, 4)  
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x