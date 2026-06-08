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

def evaluate_one_fold(model, X_test, device, y_min, y_max):
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        pred_norm = model(x_test_tensor).cpu().numpy()
    pred_raw = pred_norm[0] * (y_max - y_min + 1e-8) + y_min

    return pred_raw