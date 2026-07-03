import copy
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

from src.data.dataframe import build_split_dataframes
from src.data.dataset import NormalizedSubset
from src.data.factory import build_base_dataset
from src.model.factory import create_model
from src.utils.metrics import metrics


def is_torch_model(model):
    return isinstance(model, torch.nn.Module)


def get_target_col(conf):
    if conf["dataset_output"] == "four_HAs":
        return "haptic_attribute"
    elif conf["dataset_output"] == "roughness":
        return "roughness"
    else:
        raise ValueError(f"Unsupported dataset_output: {conf['dataset_output']}")


def prepare_batch(batch, conf, device):
    model_name = conf["model"]

    if len(batch) == 2:
        x, y = batch
        return x.to(device).float(), y.to(device).float()

    if len(batch) == 4:
        texture, height, normal, y = batch

        texture = texture.to(device).float()
        height = height.to(device).float()
        normal = normal.to(device).float()
        y = y.to(device).float()

        if model_name in {"gated_mlp", "transformer"}:
            return (texture, height, normal), y

        x = torch.cat([texture, height, normal], dim=1)
        return x, y

    raise ValueError(f"Unsupported batch format with length {len(batch)}")


def forward_model(model, inputs, conf):
    if conf["model"] in {"gated_mlp", "transformer"}:
        texture, height, normal = inputs
        return model(texture, height, normal)

    return model(inputs)


def train_one_epoch(model, loader, optimizer, criterion, conf, device):
    model.train()

    losses = []

    for batch in loader:
        inputs, y = prepare_batch(batch, conf, device)

        optimizer.zero_grad()

        pred = forward_model(model, inputs, conf)

        if pred.dim() > y.dim():
            pred = pred.squeeze(-1)

        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        losses.append(loss.detach().item())

    return float(np.mean(losses))


def evaluate(model, loader, conf, device, y_min, y_max):
    model.eval()

    pred_norm_list = []
    y_norm_list = []

    with torch.no_grad():
        for batch in loader:
            inputs, y = prepare_batch(batch, conf, device)

            pred = forward_model(model, inputs, conf)

            if pred.dim() > y.dim():
                pred = pred.squeeze(-1)

            pred_norm_list.append(pred.detach().cpu().numpy())
            y_norm_list.append(y.detach().cpu().numpy())

    pred_norm = np.concatenate(pred_norm_list, axis=0)
    y_norm = np.concatenate(y_norm_list, axis=0)

    if pred_norm.ndim == 1:
        pred_norm = pred_norm.reshape(-1, 1)

    if y_norm.ndim == 1:
        y_norm = y_norm.reshape(-1, 1)

    pred_raw = pred_norm * (y_max - y_min + 1e-8) + y_min
    gt_raw = y_norm * (y_max - y_min + 1e-8) + y_min

    mae_per_output = mean_absolute_error(
        gt_raw,
        pred_raw,
        multioutput="raw_values",
    )

    rmse_per_output = np.sqrt(
        np.mean((gt_raw - pred_raw) ** 2, axis=0)
    )

    return {
        "predictions": pred_raw,
        "ground_truths": gt_raw,
        "mae_per_output": mae_per_output,
        "rmse_per_output": rmse_per_output,
        "mean_mae": float(np.mean(mae_per_output)),
        "mean_rmse": float(np.mean(rmse_per_output)),
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    conf,
    epoch,
    input_dim,
    y_min,
    y_max,
    val_result,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": int(epoch),
        "model": conf["model"],
        "dataset_input": conf["dataset_input"],
        "dataset_output": conf["dataset_output"],
        "input_dim": input_dim,
        "y_min": y_min.tolist(),
        "y_max": y_max.tolist(),
        "val_mean_mae": float(val_result["mean_mae"]),
        "val_mean_rmse": float(val_result["mean_rmse"]),
        "val_mae_per_output": val_result["mae_per_output"].tolist(),
        "val_rmse_per_output": val_result["rmse_per_output"].tolist(),
        "model_state_dict": {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "conf": copy.deepcopy(conf),
    }

    torch.save(checkpoint, path)


def save_metrics(conf, tag, val_result, val_image_ids):
    conf_for_metrics = copy.deepcopy(conf)
    conf_for_metrics["train_tag"] = tag

    metrics(
        conf_for_metrics,
        mae_per_output=val_result["mae_per_output"],
        rmse_per_output=val_result["rmse_per_output"],
        predictions=val_result["predictions"],
        ground_truths=val_result["ground_truths"],
        test_image_ids=val_image_ids,
    )


def train_valid(conf, model_builder):
    epochs = int(conf["epochs"])
    batch_size = int(conf["batch_size"])
    lr = float(conf["learning_rate"])
    weight_decay = float(conf["weight_decay"])
    num_workers = int(conf.get("num_workers", 0))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    train_df, val_df, _ = build_split_dataframes(conf)

    target_col = get_target_col(conf)

    train_base_dataset, train_targets, input_dim = build_base_dataset(
        conf,
        train_df,
        target_col,
        device,
    )

    val_base_dataset, _, _ = build_base_dataset(
        conf,
        val_df,
        target_col,
        device,
    )

    train_targets = np.asarray(train_targets, dtype=np.float32)

    if train_targets.ndim == 1:
        train_targets = train_targets.reshape(-1, 1)

    y_min = train_targets.min(axis=0)
    y_max = train_targets.max(axis=0)

    train_dataset = NormalizedSubset(
        train_base_dataset,
        np.arange(len(train_base_dataset)),
        y_min,
        y_max,
    )

    val_dataset = NormalizedSubset(
        val_base_dataset,
        np.arange(len(val_base_dataset)),
        y_min,
        y_max,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    model = model_builder(conf, input_dim=input_dim, device=device)

    if not is_torch_model(model):
        raise TypeError(
            f"train_model.py는 torch.nn.Module 모델 기준입니다. "
            f"현재 모델 타입: {type(model)}"
        )

    model = model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_tag = str(conf.get("train_tag", conf["model"]))
    save_dir = Path("checkpoints") / train_tag
    save_dir.mkdir(parents=True, exist_ok=True)

    best_mae_path = save_dir / "best_val_mae.pt"
    best_rmse_path = save_dir / "best_val_rmse.pt"
    last_path = save_dir / "last.pt"

    best_val_mae = float("inf")
    best_val_rmse = float("inf")

    best_mae_result = None
    best_rmse_result = None

    val_image_ids = [
        Path(p).stem
        for p in val_df["texture_path"].tolist()
    ]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(val_dataset)}")
    print(f"Save directory: {save_dir}")

    for epoch in tqdm(range(1, epochs + 1), desc="Train/Valid"):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            conf=conf,
            device=device,
        )

        val_result = evaluate(
            model=model,
            loader=val_loader,
            conf=conf,
            device=device,
            y_min=y_min,
            y_max=y_max,
        )

        val_mae = val_result["mean_mae"]
        val_rmse = val_result["mean_rmse"]

        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"train_loss={train_loss:.6f} | "
            f"val_MAE={val_mae:.6f} | "
            f"val_RMSE={val_rmse:.6f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_mae_result = copy.deepcopy(val_result)

            save_checkpoint(
                path=best_mae_path,
                model=model,
                optimizer=optimizer,
                conf=conf,
                epoch=epoch,
                input_dim=input_dim,
                y_min=y_min,
                y_max=y_max,
                val_result=val_result,
            )

            print(f"  -> Saved best MAE model: {best_mae_path}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_rmse_result = copy.deepcopy(val_result)

            save_checkpoint(
                path=best_rmse_path,
                model=model,
                optimizer=optimizer,
                conf=conf,
                epoch=epoch,
                input_dim=input_dim,
                y_min=y_min,
                y_max=y_max,
                val_result=val_result,
            )

            print(f"  -> Saved best RMSE model: {best_rmse_path}")

    save_checkpoint(
        path=last_path,
        model=model,
        optimizer=optimizer,
        conf=conf,
        epoch=epochs,
        input_dim=input_dim,
        y_min=y_min,
        y_max=y_max,
        val_result=val_result,
    )

    if best_mae_result is not None:
        save_metrics(
            conf=conf,
            tag=f"{train_tag}_best_val_mae",
            val_result=best_mae_result,
            val_image_ids=val_image_ids,
        )

    if best_rmse_result is not None:
        save_metrics(
            conf=conf,
            tag=f"{train_tag}_best_val_rmse",
            val_result=best_rmse_result,
            val_image_ids=val_image_ids,
        )

    print("\nTraining finished.")
    print(f"Best validation MAE : {best_val_mae:.6f}")
    print(f"Best validation RMSE: {best_val_rmse:.6f}")
    print(f"Best MAE model path : {best_mae_path}")
    print(f"Best RMSE model path: {best_rmse_path}")
    print(f"Last model path     : {last_path}")


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    train_valid(conf, create_model)


if __name__ == "__main__":
    main()