from pathlib import Path
import argparse
import yaml
import time
import json
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm

from model.prediction.model import MultiBackBoneRegressor
from data.dataset import build_original_dataframe


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


class HeightOnlyRegressionDataset(Dataset):
    """
    Height image 하나만 입력으로 사용하는 regression dataset.

    DataFrame에는 최소한 다음 column이 있어야 함:
        height_path
        roughness
    """

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        if "height_path" not in self.df.columns:
            raise KeyError("DataFrame must contain 'height_path' column.")

        if "roughness" not in self.df.columns:
            raise KeyError("DataFrame must contain 'roughness' column.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        height_path = self.df.iloc[idx]["height_path"]
        roughness = self.df.iloc[idx]["roughness"]

        # height map은 grayscale 1채널로 로드
        height_img = Image.open(height_path).convert("L")

        if self.transform:
            height_img = self.transform(height_img)

        target = torch.tensor(float(roughness), dtype=torch.float32)

        return height_img, target


class Trainer:
    def __init__(
        self,
        model,
        df_train,
        df_valid,
        device,
        run_name,
        transform,
        batch_size=8,
        num_epochs=100,
        num_workers=0,
        lr=1e-4,
    ):
        self.model = model
        self.df_train = df_train
        self.df_valid = df_valid
        self.device = device
        self.run_name = run_name
        self.transform = transform
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.lr = lr

        self.criterion = torch.nn.MSELoss()
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

        # 새 Transformer 모델은 전체를 학습해야 함
        for p in self.model.parameters():
            p.requires_grad = True

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

    def prepare_dataloaders(self):
        train_ds = HeightOnlyRegressionDataset(self.df_train, transform=self.transform)
        valid_ds = HeightOnlyRegressionDataset(self.df_valid, transform=self.transform)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        return train_loader, valid_loader

    def train(self):
        best_val_loss = float("inf")

        ckpt_dir = SCRIPT_DIR / "checkpoints" / self.run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        log_path = ckpt_dir / "training_log.csv"

        if not log_path.exists():
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "lr", "timestamp"])

        train_loader, valid_loader = self.prepare_dataloaders()

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            epoch_start = time.time()

            for height_imgs, labels in tqdm(
                train_loader,
                desc=f"Train {epoch + 1}/{self.num_epochs}",
                unit="batch",
            ):
                height_imgs = height_imgs.to(self.device)
                labels = labels.to(self.device).view(-1, 1)

                self.optimizer.zero_grad(set_to_none=True)

                if self.device.type == "cuda":
                    with autocast(device_type="cuda"):
                        outputs = self.model(height_imgs)
                        loss = self.criterion(outputs, labels)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(height_imgs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.detach().item()

            avg_train_loss = total_loss / max(1, len(train_loader))

            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for height_imgs, targets in tqdm(
                    valid_loader,
                    desc=f"Val {epoch + 1}/{self.num_epochs}",
                    unit="batch",
                    leave=False,
                ):
                    height_imgs = height_imgs.to(self.device)
                    targets = targets.to(self.device).view(-1, 1)

                    if self.device.type == "cuda":
                        with autocast(device_type="cuda"):
                            outputs = self.model(height_imgs)
                            loss = self.criterion(outputs, targets)
                    else:
                        outputs = self.model(height_imgs)
                        loss = self.criterion(outputs, targets)

                    val_loss += loss.item()

            avg_val_loss = val_loss / max(1, len(valid_loader))
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Validation Loss: {avg_val_loss:.4f} - "
                f"time: {epoch_time:.1f}s - "
                f"lr: {lr:.2e}"
            )

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    f"{avg_train_loss:.6f}",
                    f"{avg_val_loss:.6f}",
                    f"{lr:.8f}",
                    time.time(),
                ])

            status = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": lr,
                "timestamp": time.time(),
            }

            with open(SCRIPT_DIR / "current_status.json", "w") as f:
                json.dump(status, f, indent=4)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

                best_path_pth = ckpt_dir / "best_model.pth"
                best_path_pt = ckpt_dir / "best.pt"

                torch.save(self.model.state_dict(), str(best_path_pth))
                torch.save(self.model.state_dict(), str(best_path_pt))

                print(f"Saved best model: {best_path_pth}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train height-only transformer roughness regressor"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to config YAML",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of epochs",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input image size",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader num_workers",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="height_transformer",
        help="Checkpoint/log directory name",
    )

    return parser.parse_args()


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"Warning: config file not found: {config_path}")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    args = parse_args()

    config = load_config(args.config)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    df_train, df_valid, df_test = build_original_dataframe()

    print(
        f"Dataset sizes: "
        f"train={len(df_train)}, valid={len(df_valid)}, test={len(df_test)}"
    )

    if len(df_train) == 0:
        raise RuntimeError("Train dataset is empty.")

    if len(df_valid) == 0:
        raise RuntimeError("Valid dataset is empty.")

    cfg_batch_size = args.batch_size or config.get("batch_size", 8)
    cfg_num_epochs = args.num_epochs or config.get("num_epochs", 100)
    cfg_image_size = args.image_size or config.get("image_size", 448)
    cfg_num_workers = (
        args.num_workers
        if args.num_workers is not None
        else config.get("num_workers", 0)
    )
    cfg_lr = args.lr or config.get("lr", 1e-4)

    transform = transforms.Compose([
        transforms.Resize((cfg_image_size, cfg_image_size)),
        transforms.ToTensor(),
    ])

    model = MultiBackBoneRegressor(
        model_name=None,
        image_size=cfg_image_size,
        embed_dim=64,#128,
        num_heads=1,#4,
        depth=1,#4,
        mlp_ratio=4.0,
        dropout=0.1,
        bounded_output=False,
        output_scale=100.0,
    )

    trainer = Trainer(
        model=model,
        df_train=df_train,
        df_valid=df_valid,
        device=device,
        run_name=args.run_name,
        transform=transform,
        batch_size=cfg_batch_size,
        num_epochs=cfg_num_epochs,
        num_workers=cfg_num_workers,
        lr=cfg_lr,
    )

    trainer.train()

    try:
        import subprocess
        import sys

        ck_dir = SCRIPT_DIR / "checkpoints" / args.run_name
        log_file = ck_dir / "training_log.csv"

        if log_file.exists():
            plot_script = SCRIPT_DIR / "utils" / "plot.py"

            if plot_script.exists():
                print("Generating training plots...")
                subprocess.run(
                    [
                        sys.executable,
                        str(plot_script),
                        "--config",
                        str(args.config),
                        "--extractor",
                        args.run_name,
                    ],
                    check=False,
                )
                print("Plot generation finished.")
            else:
                print(f"Plot script not found: {plot_script}")
        else:
            print(f"No training log found at {log_file}, skipping plot generation.")

    except Exception as e:
        print(f"Plot generation failed: {e}")


if __name__ == "__main__":
    main()