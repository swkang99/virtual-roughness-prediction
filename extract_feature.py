import argparse
from pathlib import Path
import yaml
import torch
from torchvision import transforms
from model import MultiBackBoneRegressor
from engine import FeatureCacheManager
import data


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
    df_train, df_valid, df_test = data.build_dataframe()
    mapping = {'train': df_train, 'valid': df_valid}

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
