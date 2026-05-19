from pathlib import Path
import argparse
import yaml
import torch
from torchvision import transforms

from src.model.prediction.model import MultiBackBoneRegressor
from src.data.dataset import build_dataframe, CachedFeatureDataset
from src.model.feature.extract import FeatureCacheManager

import os
import time
import json
import csv
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.amp import autocast, GradScaler
from tqdm import tqdm

config_path = Path(__file__).resolve().parent.parent / "config.yaml"

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

class Trainer:
    def __init__(self, model, df_train, df_valid, device, feature_extractor, cache_root, transform, batch_size=8, num_epochs=100, num_workers=0):
        self.model = model
        self.df_train = df_train
        self.df_valid = df_valid
        self.device = device
        self.feature_extractor = feature_extractor
        self.cache_root = Path(cache_root)
        self.transform = transform
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers

        self.scaler = GradScaler()
        self.criterion = torch.nn.MSELoss()
        # freeze backbones by default
        for name, p in self.model.named_parameters():
            if not name.startswith('regressor'):
                p.requires_grad = False
        self.model.to(self.device)
        try:
            self.model.backbone_texture.eval()
            self.model.backbone_normal.eval()
            self.model.backbone_height.eval()
        except Exception:
            pass

        self.optimizer = torch.optim.AdamW(self.model.regressor.parameters(), lr=1e-4)

    def prepare_cached_dataloaders(self):
        train_ds = CachedFeatureDataset(self.df_train, self.cache_root / self.feature_extractor, 'train')
        valid_ds = CachedFeatureDataset(self.df_valid, self.cache_root / self.feature_extractor, 'valid')
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=(self.device.type=='cuda'))
        valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=(self.device.type=='cuda'))
        return train_loader, valid_loader

    def train(self):
        best_val_loss = float('inf')
        ckpt_dir = Path('checkpoints') / self.feature_extractor
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_path = ckpt_dir / f'training_log.csv'
        if not log_path.exists():
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr', 'timestamp'])

        train_loader, valid_loader = self.prepare_cached_dataloaders()

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            epoch_start = time.time()

            for batch in tqdm(train_loader, desc=f"Train {epoch+1}/{self.num_epochs}", unit='batch'):
                self.optimizer.zero_grad()
                feats, labels = batch
                feats = feats.to(self.device)
                labels = labels.to(self.device).view(-1,1)

                if self.device.type == 'cuda':
                    with autocast(self.device.type):
                        outputs = self.model.regressor(feats)
                        loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model.regressor(feats)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.detach().item()

            avg_train_loss = total_loss / max(1, len(train_loader))

            # validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(valid_loader, desc=f"Val {epoch+1}/{self.num_epochs}", unit='batch', leave=False):
                    feats, targets = batch
                    feats = feats.to(self.device)
                    targets = targets.to(self.device).view(-1,1)
                    outputs = self.model.regressor(feats)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()

            avg_val_loss = val_loss / max(1, len(valid_loader))
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0
            print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f} - time: {epoch_time:.1f}s - lr: {lr:.2e}")

            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}", f"{lr:.6f}", time.time()])

            status = {'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'lr': lr, 'timestamp': time.time()}
            with open('current_status.json', 'w') as f:
                json.dump(status, f)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # save best model in checkpoint directory
                best_path_pth = Path('checkpoints') / self.feature_extractor / 'best_model.pth'
                best_path_pt = Path('checkpoints') / self.feature_extractor / 'best.pt'
                torch.save(self.model.state_dict(), str(best_path_pth))
                # also save a copy as best.pt for convenience
                torch.save(self.model.state_dict(), str(best_path_pt))


def parse_args():
    parser = argparse.ArgumentParser(description='Train regressor with optional feature cache handling')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML')
    parser.add_argument('--skip-cache', dest='skip_cache', action='store_true', help='Skip cache generation; require cache already exists')
    parser.add_argument('--force-cpu-cache', dest='force_cpu_cache', action='store_true', help='Force CPU extraction when generating cache')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--num-epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--feature-extractor', type=str, help='Feature extractor model name (overrides config)')
    parser.add_argument('--image-size', type=int, help='Image resize size (square)')
    parser.add_argument('--num-workers', type=int, help='DataLoader num_workers')
    return parser.parse_args()


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    args = parse_args()

    # dataframes
    df_train, df_valid, df_test = build_dataframe()
    print(f"Dataset sizes: train={len(df_train)}, valid={len(df_valid)}, test={len(df_test)}")

    # load config
    config_path = Path(args.config)
    config = {}
    if config_path.exists():
        if yaml is None:
            raise SystemExit('PyYAML is required to load config.yaml. Install with: pip install pyyaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

    # defaults and overrides
    cfg_feature_extractor = args.feature_extractor or config.get('feature_extractor')
    cfg_batch_size = args.batch_size or config.get('batch_size')
    cfg_num_epochs = args.num_epochs or config.get('num_epochs')
    cfg_image_size = args.image_size or config.get('image_size')
    cfg_num_workers = args.num_workers if args.num_workers is not None else config.get('num_workers', 0)
    # extraction config
    extraction_cfg = config.get('extraction', {}) if config else {}
    cfg_extraction_batch = extraction_cfg.get('batch_size', 8)
    cfg_extraction_workers = extraction_cfg.get('num_workers', cfg_num_workers)
    cfg_extraction_use_mp = extraction_cfg.get('use_multiprocessing', False)

    # rebuild transform with configured image size
    transform = transforms.Compose([
        transforms.Resize((cfg_image_size, cfg_image_size)),
        transforms.ToTensor()
    ])

    model = MultiBackBoneRegressor(cfg_feature_extractor)

    cache_root = Path(config.get('cache_root', 'feature_cache'))
    cache_mgr = FeatureCacheManager(model, transform, cache_root, device)
    trainer = Trainer(model, df_train, df_valid, device, cfg_feature_extractor, cache_root, transform, batch_size=cfg_batch_size, num_epochs=cfg_num_epochs, num_workers=cfg_num_workers)

    # Ensure cache exists (compute if missing) with respect to flags
    if args.skip_cache:
        if not cache_mgr.cache_exists(cfg_feature_extractor):
            raise SystemExit('Cache missing but --skip-cache specified. Aborting.')
    else:
        if not cache_mgr.cache_exists(cfg_feature_extractor):
            print('Computing feature cache (this runs backbone once per image)...')
            cache_mgr.compute_and_cache(
                df_train, cfg_feature_extractor, 'train',
                force_cpu=args.force_cpu_cache,
                batch_size=cfg_extraction_batch,
                num_workers=cfg_extraction_workers,
                use_multiprocessing=cfg_extraction_use_mp
            )
            cache_mgr.compute_and_cache(
                df_valid, cfg_feature_extractor, 'valid',
                force_cpu=args.force_cpu_cache,
                batch_size=cfg_extraction_batch,
                num_workers=cfg_extraction_workers,
                use_multiprocessing=cfg_extraction_use_mp
            )

    # start training (uses cached features)
    trainer.train()

    # after training, try to generate plots by running plot.py in a subprocess
    try:
        import subprocess, sys

        ck_dir = Path('checkpoints') / cfg_feature_extractor
        log_file = ck_dir / 'training_log.csv'
        if log_file.exists():
            print('Generating training plots...')
            plot_script = Path(__file__).parent / 'plot.py'
            # call plot.py as a separate process to avoid any matplotlib backend issues
            subprocess.run([sys.executable, str(plot_script), '--config', str(config_path), '--extractor', cfg_feature_extractor], check=False)
            print('Plot generation finished (if plot.py succeeded).')
        else:
            print(f'No training log found at {log_file}, skipping plot generation.')
    except Exception as e:
        print(f'Plot generation failed: {e}')


if __name__ == '__main__':
    main()