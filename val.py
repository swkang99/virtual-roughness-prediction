from pathlib import Path
import numpy as np
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import csv
from model import MultiBackBoneRegressor
import data
from engine import CachedFeatureDataset
import argparse
try:
    import yaml
except Exception:
    yaml = None

def eval_test_set(model, test_loader, criterion, device, tol=0.5, out_dir=None, feature_extractor=None):
    if feature_extractor is None:
        raise ValueError('feature_extractor must be provided')
    model_path = Path(f'checkpoints/{feature_extractor}') / 'best_model.pth'
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()
    
    test_loss = 0.0
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for batch in test_loader:
            # Support cached-feature batches: (feat_tensor, targets)
            if isinstance(batch, (list, tuple)) and torch.is_tensor(batch[0]):
                feats, targets = batch
                feats = feats.to(device)
                targets = targets.to(device).view(-1, 1)
                outputs = model.regressor(feats)
            else:
                (texture, normal, height), targets = batch
                # 입력 및 타겟 GPU 이동
                texture = texture.to(device)
                normal = normal.to(device)
                height = height.to(device)
                targets = targets.to(device).view(-1, 1)
                outputs = model(texture, normal, height)

            loss = criterion(outputs, targets)
            test_loss += loss.item()
            # 값 저장 (배치 전체 저장)
            predictions.append(outputs.cpu().numpy())
            ground_truths.append(targets.cpu().numpy())

    # (배치, 4) 형태를 (N, 4)로 합침
    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)
    
    # now only Roughness (first/only column)
    # predictions and ground_truths are shapes (N, 1) or (N,)
    preds = predictions.reshape(-1)
    gts = ground_truths.reshape(-1)

    mae = mean_absolute_error(gts, preds)
    r2 = r2_score(gts, preds)
    # RMSE
    rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))

    print(f"\nTest Results (Roughness): MAE={mae:.4f} | R2={r2:.4f} | RMSE={rmse:.4f} | Avg Loss: {test_loss / len(test_loader):.4f}")

    # ensure output directory exists
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # write CSV summary of metrics (single target)
    csv_path = out_dir / 'val_metrics.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'roughness'])
        writer.writerow(['MAE', f"{mae:.6f}"])
        writer.writerow(['R2', f"{r2:.6f}"])
        writer.writerow(['RMSE', f"{rmse:.6f}"])

    # write human-readable summary
    summary_path = out_dir / 'val_summary.txt'
    with summary_path.open('w', encoding='utf-8') as f:
        f.write('Validation summary\n')
        f.write('==================\n')
        f.write(f"Items evaluated: {preds.shape[0]}\n")
        f.write(f"Average Loss: {test_loss / len(test_loader):.6f}\n")
        f.write(f"MAE (Roughness): {mae:.6f}\n")
        f.write(f"R2 (Roughness): {r2:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")

    # Roughness scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(gts, preds, alpha=0.5)
    plt.xlabel('Ground Truth (Roughness)')
    plt.ylabel('Prediction')
    plt.title('Roughness')
    mn, mx = min(gts.min(), preds.min()), max(gts.max(), preds.max())
    plt.plot([mn, mx], [mn, mx], 'r--', lw=1)
    plt.tight_layout()
    plt.savefig(str(out_dir / 'results_roughness.png'))
    plt.close()

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='Validate model using images or cached features')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML')
    args = parser.parse_args()

    # load config
    config = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        if yaml is None:
            raise SystemExit('PyYAML is required to load config.yaml. Install with: pip install pyyaml')
        with open(cfg_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

    cfg_feature_extractor = config.get('feature_extractor', 'eva_giant_patch14_560.m30m_ft_in22k_in1k')
    cfg_batch_size = config.get('batch_size', 32)
    cfg_image_size = config.get('image_size', 560)
    cfg_num_workers = config.get('num_workers', 0)
    cfg_cache_root = Path(config.get('cache_root', 'feature_cache'))

    # build model and transform
    model = MultiBackBoneRegressor(cfg_feature_extractor).to(device)
    transform_local = transforms.Compose([
        transforms.Resize((cfg_image_size, cfg_image_size)),
        transforms.ToTensor()
    ])

    df_train, df_valid, df_test = data.build_dataframe()
    # prefer cached features if available
    cache_valid_dir = cfg_cache_root / cfg_feature_extractor / 'valid'
    if cache_valid_dir.exists() and any(cache_valid_dir.iterdir()):
        valid_dataset = CachedFeatureDataset(df_valid, cfg_cache_root / cfg_feature_extractor, 'valid')
        valid_loader = DataLoader(valid_dataset, batch_size=cfg_batch_size, shuffle=False, num_workers=cfg_num_workers, pin_memory=(device.type=='cuda'))
    else:
        valid_dataset = data.CustomRegressionDataset(df_valid, transform_local)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg_batch_size, shuffle=False, num_workers=cfg_num_workers, pin_memory=(device.type=='cuda'))

    print(f"Dataset sizes: train={len(df_train)}, valid={len(df_valid)}, test={len(df_test)}")
    eval_test_set(model, valid_loader, nn.MSELoss(), device, out_dir=Path(f'results/{cfg_feature_extractor}'), feature_extractor=cfg_feature_extractor)

if __name__ == '__main__':
    main()