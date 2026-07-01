import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.dataset import dataset_to_numpy
from tqdm import tqdm
from src.model.prediction.proposed.gated_mlp import GatedFusionRegressor
from src.model.prediction.proposed.gated_mlp_v2 import GatedFusionRegressorV2
import numpy as np

def is_gated_mlp(model):
    return isinstance(model, GatedFusionRegressor) or isinstance(model, GatedFusionRegressorV2)

def is_torch_model(model):
    return isinstance(model, torch.nn.Module)

def prepare_batch_by_model(batch, model, device):
    """
    SeparatedDataset 기준:
        batch = (texture, height, normal, y)

    FeatureDataset 기준:
        batch = (x, y)
    """
    if is_gated_mlp(model):
        texture, height, normal, y = batch
        texture = texture.to(device).float()
        height = height.to(device).float()
        normal = normal.to(device).float()
        y = y.to(device).float()
        return (texture, height, normal), y

    else:
        if len(batch) == 4:
            texture, height, normal, y = batch
            x = torch.cat([texture, height, normal], dim=1).to(device).float()
            y = y.to(device).float()
            return x, y

        elif len(batch) == 2:
            x, y = batch
            x = x.to(device).float()
            y = y.to(device).float()
            return x, y

        else:
            raise ValueError(f"Unsupported batch format with length {len(batch)}")


def forward_by_model(model, inputs):
    if is_gated_mlp(model):
        texture, height, normal = inputs
        return model(texture, height, normal)
    else:
        return model(inputs)


def train_one_fold(model, dataset, device, epochs, batch_size, lr, weight_decay):
    if is_torch_model(model):
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        model.to(device)
        model.train()

        for epoch in tqdm(range(epochs), desc="Train One Fold"):
            epoch_losses = []

            for batch in train_loader:
                inputs, y = prepare_batch_by_model(batch, model, device)

                optimizer.zero_grad()
                pred = forward_by_model(model, inputs)

                if pred.dim() > y.dim():
                    pred = pred.squeeze(-1)

                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.detach().item())

        return model

    else:
        X_train, y_train = dataset_to_numpy(dataset)
        model.fit(X_train, y_train)
        return model

def evaluate_one_fold(model, dataset, device, y_min, y_max, batch_size=32):
    if is_torch_model(model):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

        model.to(device)
        model.eval()

        pred_norm_list = []
        y_norm_list = []

        with torch.no_grad():
            for batch in loader:
                inputs, y = prepare_batch_by_model(batch, model, device)

                pred = forward_by_model(model, inputs)

                if pred.dim() > y.dim():
                    pred = pred.squeeze(-1)

                pred_norm_list.append(pred.detach().cpu().numpy())
                y_norm_list.append(y.detach().cpu().numpy())

        pred_norm = np.concatenate(pred_norm_list, axis=0)
        y_norm = np.concatenate(y_norm_list, axis=0)

    else:
        X_test, y_norm = dataset_to_numpy(dataset)
        pred_norm = model.predict(X_test)

        if pred_norm.ndim == 1:
            pred_norm = pred_norm.reshape(-1, 1)
        if y_norm.ndim == 1:
            y_norm = y_norm.reshape(-1, 1)

    y_min = y_min.item()
    y_max = y_max.item()
    pred_raw = pred_norm * (y_max - y_min + 1e-8) + y_min
    gt_raw = y_norm * (y_max - y_min + 1e-8) + y_min

    return pred_raw, gt_raw

def main():
    # prepare dataset (train, val)
    # load model
    # evaluate
    # metric
    # save model
    pass

if __name__ == '__main__':
    main()    