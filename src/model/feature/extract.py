import argparse
from pathlib import Path
import yaml
import torch
import pandas as pd
from torchvision import transforms
from src.model.prediction.model import MultiBackBoneRegressor
from src.model.feature.extract import FeatureCacheManager
from src.data.dataset import build_dataframe
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from tqdm import tqdm


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




def run_extraction(config_path='config.yaml', splits=('train','valid'), force_cpu=False):
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise SystemExit(f'Config not found: {cfg_path}')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    feature_extractor = config.get('feature_extractor')
    if not feature_extractor:
        raise SystemExit('feature_extractor not set in config.yaml')

    cache_root = Path(config.get('cache_root', 'feature_cache'))
    image_size = config.get('image_size', 560)
    extraction_cfg = config.get('extraction', {})
    batch_size = extraction_cfg.get('batch_size', 8)
    num_workers = extraction_cfg.get('num_workers', 2)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # build transform and model
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    model = MultiBackBoneRegressor(feature_extractor)
    cache_mgr = FeatureCacheManager(model, transform, cache_root, device)

    # load dataframes to iterate
    df_train, df_valid, df_test = build_dataframe()
    mapping = {'train': df_train, 'valid': df_valid, 'full': pd.concat([df_train, df_valid], ignore_index=True)}

    for split in splits:
        if split not in mapping:
            print(f'Skipping unknown split: {split}')
            continue
        df = mapping[split]
        split_cache = cache_root / feature_extractor / split
        if split_cache.exists() and any(split_cache.iterdir()):
            print(f'Cache for {split} already exists at {split_cache}, skipping')
            continue
        print(f'Generating cache for {split}...')
        cache_mgr.compute_and_cache(df, feature_extractor, split, force_cpu=force_cpu, batch_size=batch_size, num_workers=num_workers)


def main():
    parser = argparse.ArgumentParser(description='Extract and cache features per config')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--splits', type=str, default='train,valid', help='Comma separated splits to extract (train,valid)')
    parser.add_argument('--force-cpu', action='store_true')
    parser.add_argument('--only-missing', dest='only_missing', action='store_true', help='Only extract missing splits (default)')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='Overwrite existing cache for the requested splits')
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    # determine behavior: default is only_missing unless --overwrite specified
    if args.overwrite:
        # remove existing cache for splits before extraction
        cfg_path = Path(args.config)
        with open(cfg_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        feature_extractor = config.get('feature_extractor')
        cache_root = Path(config.get('cache_root', 'feature_cache'))
        for split in splits:
            dir_to_rm = cache_root / feature_extractor / split
            if dir_to_rm.exists():
                import shutil
                shutil.rmtree(dir_to_rm)
    run_extraction(args.config, splits=splits, force_cpu=args.force_cpu)


if __name__ == '__main__':
    main()
