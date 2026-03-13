from pathlib import Path
import argparse
try:
    import yaml
except Exception:
    yaml = None
import torch
from torchvision import transforms

from model import MultiBackBoneRegressor
import data
from engine import FeatureCacheManager, Trainer

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
    df_train, df_valid, df_test = data.build_dataframe()
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