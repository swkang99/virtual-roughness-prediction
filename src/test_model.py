import copy
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

from src.data.dataframe import build_dataframe_from_file
from src.data.dataset import NormalizedSubset
from src.data.factory import build_base_dataset
from src.model.factory import create_model
from src.utils.metrics import metrics


def get_target_col(conf):
    if conf["dataset_output"] == "four_HAs":
        return "haptic_attribute"
    elif conf["dataset_output"] == "roughness":
        return "roughness"
    else:
        raise ValueError(f"Unsupported dataset_output: {conf['dataset_output']}")


def find_label_file(split_path, candidates):
    split_path = Path(split_path)

    for name in candidates:
        path = split_path / name
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No label csv found in {split_path}. "
        f"Checked: {candidates}"
    )


def load_checkpoint(path, device):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


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


def evaluate(model, loader, conf, device, y_min, y_max):
    model.eval()

    pred_norm_list = []
    y_norm_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Test batches", unit="batch"):
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


def save_test_metrics(conf, tag, test_result, test_image_ids):
    conf_for_metrics = copy.deepcopy(conf)
    conf_for_metrics["train_tag"] = tag

    metrics(
        conf_for_metrics,
        mae_per_output=test_result["mae_per_output"],
        rmse_per_output=test_result["rmse_per_output"],
        predictions=test_result["predictions"],
        ground_truths=test_result["ground_truths"],
        test_image_ids=test_image_ids,
    )


def build_runtime_conf(config_conf, checkpoint):
    ckpt_conf = checkpoint.get("conf", None)

    if isinstance(ckpt_conf, dict):
        conf = copy.deepcopy(ckpt_conf)
    else:
        conf = copy.deepcopy(config_conf)

    # 현재 config.yaml의 경로 설정은 우선 반영
    for key in [
        "data_test_path",
        "data_patch_path",
        "train_tag",
        "batch_size",
        "num_workers",
    ]:
        if key in config_conf:
            conf[key] = config_conf[key]

    return conf


def test_model(config_conf):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    train_tag = str(config_conf.get("train_tag", config_conf["model"]))
    checkpoint_root = Path(config_conf.get("checkpoint_root", "checkpoints"))
    checkpoint_name = config_conf.get("test_checkpoint_name", "best_val_rmse.pt")

    checkpoint_path = checkpoint_root / train_tag / checkpoint_name

    print(f"Checkpoint path: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, device)

    conf = build_runtime_conf(config_conf, checkpoint)

    target_col = get_target_col(conf)

    test_path = Path(conf["data_test_path"])
    patch_path = test_path / Path(conf["data_patch_path"])
    label_path = find_label_file(test_path, ["test.csv"])

    print(f"Test patch path : {patch_path}")
    print(f"Test label path : {label_path}")

    test_df = build_dataframe_from_file(
        conf,
        texture_path=patch_path,
        label_path=label_path,
        header=0,
    )

    print(f"Test samples: {len(test_df)}")

    test_base_dataset, _, input_dim_from_data = build_base_dataset(
        conf,
        test_df,
        target_col,
        device,
    )

    input_dim = checkpoint.get("input_dim", input_dim_from_data)

    y_min = np.asarray(checkpoint["y_min"], dtype=np.float32)
    y_max = np.asarray(checkpoint["y_max"], dtype=np.float32)

    test_dataset = NormalizedSubset(
        test_base_dataset,
        np.arange(len(test_base_dataset)),
        y_min,
        y_max,
    )

    batch_size = int(conf.get("test_batch_size", conf.get("batch_size", 32)))
    num_workers = int(conf.get("num_workers", 0))

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    model = create_model(
        conf,
        input_dim=input_dim,
        device=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_result = evaluate(
        model=model,
        loader=test_loader,
        conf=conf,
        device=device,
        y_min=y_min,
        y_max=y_max,
    )

    test_image_ids = [
        Path(p).stem
        for p in test_df["texture_path"].tolist()
    ]

    result_tag = f"{train_tag}_test_{Path(checkpoint_name).stem}"

    save_test_metrics(
        conf=conf,
        tag=result_tag,
        test_result=test_result,
        test_image_ids=test_image_ids,
    )

    print("\n=== Test Results ===")
    print(f"Test MAE : {test_result['mean_mae']:.6f}")
    print(f"Test RMSE: {test_result['mean_rmse']:.6f}")
    print(f"Result tag: {result_tag}")


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config_conf = yaml.safe_load(f)

    test_model(config_conf)


if __name__ == "__main__":
    main()