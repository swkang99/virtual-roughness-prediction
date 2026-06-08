import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_one_fold(model, dataset, device, epochs=200, batch_size=8, lr=1e-3, weight_decay=1e-4):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in tqdm(range(epochs), desc="Train One Fold"):
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

def evaluate_one_fold(model, dataset, device, y_min, y_max, batch_size=1):
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model.eval()
    preds_raw = []
    gts_raw = []

    y_min = np.asarray(y_min, dtype=np.float32)
    y_max = np.asarray(y_max, dtype=np.float32)

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            pred_norm = model(x)
            if pred_norm.dim() > y.dim():
                pred_norm = pred_norm.squeeze(-1)

            pred_norm = pred_norm.detach().cpu().numpy()
            y_norm = y.detach().cpu().numpy()

            pred_raw = pred_norm * (y_max - y_min + 1e-8) + y_min
            gt_raw = y_norm * (y_max - y_min + 1e-8) + y_min

            preds_raw.append(pred_raw)
            gts_raw.append(gt_raw)

    preds_raw = np.concatenate(preds_raw, axis=0)
    gts_raw = np.concatenate(gts_raw, axis=0)

    return preds_raw, gts_raw