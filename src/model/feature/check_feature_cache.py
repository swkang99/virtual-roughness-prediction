from pathlib import Path
import torch
import yaml
from src.model.feature.extract import run_extraction
import argparse


def inspect_and_maybe_extract(config_path='config.yaml', splits=('train',)):
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise SystemExit(f'Config not found: {cfg_path}')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    feature_extractor = config.get('feature_extractor')
    if not feature_extractor:
        raise SystemExit('feature_extractor not set in config.yaml')

    cache_root = Path(config.get('cache_root', 'feature_cache'))
    results = {}
    for split in splits:
        split_dir = cache_root / feature_extractor / split
        if split_dir.exists() and any(split_dir.iterdir()):
            pts = sorted(list(split_dir.glob('*.pt')))
            if pts:
                p = pts[0]
                feat = torch.load(str(p))
                results[split] = {'exists': True, 'example': {'path': str(p), 'shape': tuple(feat.shape), 'dtype': str(feat.dtype)}}
                continue
        results[split] = {'exists': False}

    # If any split missing, run extraction for those splits
    missing = [s for s, info in results.items() if not info['exists']]
    if missing:
        print(f'Missing cache for splits: {missing}. Running extraction...')
        run_extraction(config_path, splits=missing, force_cpu=False)
        # re-inspect after extraction
        return inspect_and_maybe_extract(config_path, splits)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--splits', default='train,valid')
    args = parser.parse_args()
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    res = inspect_and_maybe_extract(args.config, splits=tuple(splits))
    for s, info in res.items():
        if info['exists']:
            ex = info['example']
            print(f"Split {s}: example feature {ex['path']} shape={ex['shape']} dtype={ex['dtype']}")


if __name__ == '__main__':
    main()
