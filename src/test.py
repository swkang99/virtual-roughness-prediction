from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from model.prediction.model import MultiBackBoneRegressor
from data.texture_maps import (
    load_grayscale_image,
    extract_height_map,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Inference on MOESM test textures using height-only transformer")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "MOESM",
        help="Path to MOESM data directory",
    )

    parser.add_argument(
        "--test-image-dir",
        type=Path,
        default=None,
        help="Path to test texture image directory. Default: <data-dir>/test/MOESM1",
    )

    parser.add_argument(
        "--test-ids",
        type=Path,
        default=None,
        help="Path to test_ids.csv. Default: <data-dir>/test_ids.csv",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to ParticipantData.csv. Default: <data-dir>/ParticipantData.csv",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "height_transformer" / "best_model.pth",
        help="Path to trained height-transformer checkpoint",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=560,
        help="Input image size. Must match training image_size.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers",
    )

    parser.add_argument(
        "--gt-col",
        type=int,
        default=1,
        help=(
            "Ground-truth column index in ParticipantData.csv. "
            "Default 1 matches the current dataset.py behavior for multi-column CSV."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "test_results" / "height_transformer" / "test_predictions.csv",
        help="Output CSV path",
    )

    return parser.parse_args()


def read_test_ids(test_ids_path: Path):
    if not test_ids_path.exists():
        raise FileNotFoundError(f"test_ids.csv not found: {test_ids_path}")

    df = pd.read_csv(test_ids_path)

    if "id" in df.columns:
        ids = df["id"].astype(str).tolist()
    else:
        df = pd.read_csv(test_ids_path, header=None)
        ids = df.iloc[:, 0].astype(str).tolist()

    return [str(int(x)) if str(x).isdigit() else str(x) for x in ids]


def load_gt_map(labels_path: Path, gt_col: int):
    if not labels_path.exists():
        raise FileNotFoundError(f"ParticipantData.csv not found: {labels_path}")

    labels_df = pd.read_csv(labels_path, header=None)

    if gt_col < 0 or gt_col >= labels_df.shape[1]:
        raise IndexError(
            f"Invalid gt_col={gt_col}. "
            f"{labels_path} has {labels_df.shape[1]} columns."
        )

    gt_map = {}
    for i in range(len(labels_df)):
        sid = str(i + 1)
        gt_map[sid] = float(labels_df.iloc[i, gt_col])

    return gt_map


def find_texture_path(image_dir: Path, sid: str):
    exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

    candidate_stems = [sid]

    try:
        sid_int = int(sid)
        candidate_stems.extend([
            str(sid_int),
            f"{sid_int:03d}",
            f"{sid_int:04d}",
            f"{sid_int:05d}",
        ])
    except ValueError:
        pass

    candidate_stems = list(dict.fromkeys(candidate_stems))

    for stem in candidate_stems:
        for ext in exts:
            path = image_dir / f"{stem}{ext}"
            if path.exists():
                return path

    raise FileNotFoundError(f"Could not find texture image for id={sid} in {image_dir}")


def make_height_from_texture(
    texture_path: Path,
    blur_ksize=5,
    invert=False,
):
    """
    train.py에서 사용한 height map과 같은 방식으로 texture image에서 height image 생성.
    저장하지 않고 메모리상 PIL image로만 반환.
    """
    gray_img = load_grayscale_image(texture_path)

    height_map = extract_height_map(
        gray_img,
        blur_ksize=blur_ksize,
        invert=invert,
        normalize_output=True,
    )

    height_uint8 = (np.clip(height_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    height_img = Image.fromarray(height_uint8, mode="L")

    return height_img


class HeightOnlyTestDataset(Dataset):
    def __init__(self, test_ids, image_dir, gt_map, transform):
        self.test_ids = test_ids
        self.image_dir = Path(image_dir)
        self.gt_map = gt_map
        self.transform = transform

    def __len__(self):
        return len(self.test_ids)

    def __getitem__(self, idx):
        sid = self.test_ids[idx]

        texture_path = find_texture_path(self.image_dir, sid)

        if sid not in self.gt_map:
            raise KeyError(f"GT value for id={sid} was not found in ParticipantData.csv")

        height_img = make_height_from_texture(texture_path)

        if self.transform is not None:
            height_tensor = self.transform(height_img)
        else:
            height_tensor = transforms.ToTensor()(height_img)

        target = torch.tensor(self.gt_map[sid], dtype=torch.float32)

        return height_tensor, target, sid, str(texture_path)


def load_checkpoint(model, checkpoint_path: Path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        state_dict = torch.load(
            str(checkpoint_path),
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(str(checkpoint_path), map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {checkpoint_path}")


def compute_metrics(preds, targets):
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    errors = preds - targets

    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def main():
    args = parse_args()

    data_dir = args.data_dir
    test_image_dir = args.test_image_dir or (data_dir / "test" / "MOESM1")
    test_ids_path = args.test_ids or (data_dir / "test_ids.csv")
    labels_path = args.labels or (data_dir / "ParticipantData.csv")

    if not test_image_dir.exists():
        raise FileNotFoundError(f"Test image directory not found: {test_image_dir}")

    test_ids = read_test_ids(test_ids_path)
    gt_map = load_gt_map(labels_path, gt_col=args.gt_col)

    print(f"Test image directory: {test_image_dir}")
    print(f"Number of test ids: {len(test_ids)}")
    print(f"GT CSV: {labels_path}")
    print(f"GT column index: {args.gt_col}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image size: {args.image_size}")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    dataset = HeightOnlyTestDataset(
        test_ids=test_ids,
        image_dir=test_image_dir,
        gt_map=gt_map,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    model = MultiBackBoneRegressor(
        model_name=None,
        image_size=args.image_size,
        embed_dim=64,#128,
        num_heads=1,#4,
        depth=1,#4,
        mlp_ratio=4.0,
        dropout=0.1,
        bounded_output=False,
        output_scale=100.0,
    )

    load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    preds = []
    targets = []
    ids = []
    texture_paths = []

    with torch.no_grad():
        for height, target, sid, texture_path in tqdm(loader, desc="Inference", unit="batch"):
            height = height.to(device)
            target = target.to(device).view(-1, 1)

            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    output = model(height)
            else:
                output = model(height)

            output = output.detach().cpu().view(-1).numpy()
            target = target.detach().cpu().view(-1).numpy()

            preds.extend(output.tolist())
            targets.extend(target.tolist())
            ids.extend(list(sid))
            texture_paths.extend(list(texture_path))

    metrics = compute_metrics(preds, targets)

    print("\nTest Results")
    print(f"MSE : {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE : {metrics['mae']:.6f}")
    print(f"R2  : {metrics['r2']:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    preds_arr = np.asarray(preds)
    targets_arr = np.asarray(targets)

    result_df = pd.DataFrame({
        "id": ids,
        "target": targets_arr,
        "prediction": preds_arr,
        "absolute_error": np.abs(preds_arr - targets_arr),
        "squared_error": (preds_arr - targets_arr) ** 2,
        "texture_path": texture_paths,
    })

    result_df.to_csv(args.output, index=False)

    metrics_path = args.output.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nSaved predictions to: {args.output}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()