import os
import time
import json
import csv
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.amp import autocast, GradScaler
from tqdm import tqdm


class CachedFeatureDataset(Dataset):
    def __init__(self, df, cache_root, split_name):
        self.df = df.reset_index(drop=True)
        self.split_cache = Path(cache_root) / split_name
        self.ids = [str(int(Path(p).stem)) for p in self.df['texture_path'].tolist()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        feat_path = self.split_cache / f"{sid}.pt"
        feat = torch.load(str(feat_path))
        target = torch.tensor(self.df.iloc[idx]['roughness'], dtype=torch.float32)
        return feat, target


class FeatureCacheManager:
    def __init__(self, model, transform, cache_root: Path, device: torch.device):
        self.model = model
        self.transform = transform
        self.cache_root = Path(cache_root)
        self.device = device

    def cache_exists(self, extractor_name):
        d = self.cache_root / extractor_name
        return d.exists() and any(d.iterdir())

    def compute_and_cache(self, df, extractor_name, split_name, force_cpu=False, batch_size=8, num_workers=None, use_multiprocessing=False):
        split_cache = self.cache_root / extractor_name / split_name
        split_cache.mkdir(parents=True, exist_ok=True)

        # build an index of rows to process (skip existing)
        rows_to_process = []
        ids = []
        for idx, row in df.iterrows():
            sid = str(int(Path(row['texture_path']).stem))
            out_path = split_cache / f"{sid}.pt"
            if out_path.exists():
                continue
            rows_to_process.append(row)
            ids.append(sid)

        if not rows_to_process:
            return

        # ImageTripletDataset for batched loading
        class ImageTripletDataset(Dataset):
            def __init__(self, rows, transform):
                self.rows = rows
                self.transform = transform

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, idx):
                row = self.rows[idx]
                tex = Image.open(row['texture_path']).convert('RGB')
                nor = Image.open(row['normal_path']).convert('RGB')
                hei = Image.open(row['height_path']).convert('RGB')
                return self.transform(tex), self.transform(nor), self.transform(hei), str(int(Path(row['texture_path']).stem))

        def _run_extraction(device_for_extr, batch_size=8, num_workers=0, use_autocast=True):
            ds = ImageTripletDataset(rows_to_process, self.transform)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device_for_extr.type=='cuda'))

            # move backbones to extraction device
            try:
                self.model.backbone_texture.to(device_for_extr).eval()
                self.model.backbone_normal.to(device_for_extr).eval()
                self.model.backbone_height.to(device_for_extr).eval()
            except Exception:
                pass

            for batch in tqdm(loader, desc=f'cache:{split_name}'):
                tex_batch, nor_batch, hei_batch, batch_ids = batch
                tex_batch = tex_batch.to(device_for_extr)
                nor_batch = nor_batch.to(device_for_extr)
                hei_batch = hei_batch.to(device_for_extr)

                with torch.no_grad():
                    if use_autocast and device_for_extr.type == 'cuda':
                        from torch.cuda.amp import autocast
                        with autocast():
                            feat_tex = self.model.backbone_texture(tex_batch)
                            feat_nor = self.model.backbone_normal(nor_batch)
                            feat_hei = self.model.backbone_height(hei_batch)
                    else:
                        feat_tex = self.model.backbone_texture(tex_batch)
                        feat_nor = self.model.backbone_normal(nor_batch)
                        feat_hei = self.model.backbone_height(hei_batch)

                feats = torch.cat([feat_tex, feat_nor, feat_hei], dim=1)  # [B, N]
                feats_cpu = feats.cpu()

                # atomic save each sample (tmp -> move)
                for i, sid in enumerate(batch_ids):
                    out_path = split_cache / f"{sid}.pt"
                    if out_path.exists():
                        continue
                    tmp_path = split_cache / f".{sid}.pt.tmp"
                    torch.save(feats_cpu[i], str(tmp_path))
                    try:
                        tmp_path.replace(out_path)
                    except Exception:
                        # fallback to os.replace
                        os.replace(str(tmp_path), str(out_path))

        # attempt GPU extraction with retries on OOM (halve batch size on OOM)
        if force_cpu:
            _run_extraction(torch.device('cpu'), batch_size=1, num_workers=0, use_autocast=False)
            return

        # default params; can be passed in
        initial_batch = batch_size or 8
        if num_workers is None:
            num_workers = min(4, max(0, os.cpu_count() - 1 or 0))

        try:
            _run_extraction(self.device, batch_size=initial_batch, num_workers=num_workers, use_autocast=True)
        except RuntimeError as e:
            msg = str(e).lower()
            if 'outofmemory' in msg or 'cuda' in msg:
                print('CUDA OOM during batched extraction: retrying with smaller batch sizes, then fallback to CPU if necessary.')
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                # progressively reduce batch size
                bs = initial_batch
                success = False
                while bs >= 1:
                    try:
                        _run_extraction(self.device, batch_size=bs, num_workers=num_workers, use_autocast=True)
                        success = True
                        break
                    except RuntimeError as e2:
                        if 'outofmemory' in str(e2).lower():
                            bs = bs // 2
                            print(f'Retrying with batch size {bs}...')
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            continue
                        else:
                            raise

                if not success:
                    print('All GPU retries failed; falling back to CPU extraction.')
                    _run_extraction(torch.device('cpu'), batch_size=1, num_workers=0, use_autocast=False)
            else:
                raise


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
