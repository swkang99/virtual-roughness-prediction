import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_one_fold(model, dataset, device, epochs=200, batch_size=8, lr=1e-3, weight_decay=1e-4):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            if pred.dim() > y.dim():
                pred = pred.squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.detach().item())

    return model

def evaluate_one_fold(model, dataset, device, y_min, y_max):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=False)
    
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            pred_norm = model(x)
            if pred_norm.dim() > y.dim():
                pred_norm = pred_norm.squeeze(-1)
            
            pred_norm = pred_norm.detach().cpu().numpy()
            y_norm = y.detach().cpu().numpy()
            
            pred_raw = pred_norm * (y_max - y_min + 1e-8) + y_min
            gt_raw = y_norm * (y_max - y_min + 1e-8) + y_min
    
    return pred_raw, gt_raw  # (4,) or (1,)