import argparse
from pathlib import Path
import importlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import pandas as pd
import yaml
import csv
import time
from tqdm import tqdm
from src.model.model import MultiBackBoneRegressor
from src.engine.engine import FeatureCacheManager
import src.utils.data as data
from src.engine.engine import CachedFeatureDataset

def loocv_evaluation(config_path='config.yaml', force_cpu=False):
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
    batch_size = config.get('batch_size', 8)
    num_epochs = config.get('num_epochs', 100)
    num_workers = config.get('num_workers', 0)

    device = torch.device('cuda') if torch.cuda.is_available() and not force_cpu else torch.device('cpu')

    # Load full dataset from original source, ignoring any split directories
    full_df = data.build_original_dataframe()
    print(f"Full original dataset size: {len(full_df)}")

    # Ensure features are cached for the full dataset
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    # Load 1D CNN class dynamically because module name starts with a digit
    model_module = importlib.import_module('src.model.1dCNN')
    MultiScale1DCNN = model_module.MultiScale1DCNN

    # Use the MultiBackBoneRegressor only for feature extraction/caching
    feature_extractor_model = MultiBackBoneRegressor(feature_extractor)
    cache_mgr = FeatureCacheManager(feature_extractor_model, transform, cache_root, device)

    # Cache features for full dataset if not exists
    full_cache_dir = cache_root / feature_extractor / 'full'
    avg_cache_time_per_sample = 0.0
    if not full_cache_dir.exists() or not any(full_cache_dir.iterdir()):
        print("Caching features for full dataset...")
        cache_start = time.time()
        cache_mgr.compute_and_cache(full_df, feature_extractor, 'full', force_cpu=force_cpu, batch_size=batch_size, num_workers=num_workers)
        cache_time = time.time() - cache_start
        avg_cache_time_per_sample = cache_time / len(full_df)
        print(f"Total caching time: {cache_time:.4f}s, Avg per sample: {avg_cache_time_per_sample:.6f}s")
    else:
        print("Features already cached, skipping caching time measurement.")

    # Create dataset
    dataset = CachedFeatureDataset(full_df, cache_root / feature_extractor, 'full')
    feature_dim = dataset[0][0].shape[0]

    # LOOCV
    predictions = []
    ground_truths = []
    image_ids = []

    for i in range(len(dataset)):
        print(f"LOOCV iteration {i+1}/{len(dataset)}")

        # Train indices: all except i
        train_indices = list(range(len(dataset)))
        train_indices.remove(i)

        # Test indices: only i
        test_indices = [i]

        # Create subsets
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)

        # DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False, num_workers=0)

        # Model
        model = MultiScale1DCNN(input_feature_dim=feature_dim)
        model.to(device)

        # Optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        # Create log directory
        log_dir = Path(f'experiments/runs/{feature_extractor}/loocv_logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f'iter_{i+1}_log.csv'
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr', 'timestamp', 'predict_time'])

        # Train
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            epoch_start = time.time()
            for feats, targets in tqdm(train_loader, desc=f"Train Iter {i+1} Epoch {epoch+1}/{num_epochs}", unit='batch'):
                feats = feats.to(device)
                targets = targets.to(device).view(-1, 1)

                optimizer.zero_grad()
                outputs = model(feats)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().item()

            avg_train_loss = total_loss / max(1, len(train_loader))
            # Validation on the single left-out sample
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for feats, targets in test_loader:
                    feats = feats.to(device)
                    targets = targets.to(device).view(-1, 1)
                    outputs = model(feats)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            avg_val_loss = val_loss / max(1, len(test_loader))
            model.train()

            epoch_time = time.time() - epoch_start
            lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
            print(f"Iter {i+1} Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - time: {epoch_time:.1f}s - lr: {lr:.2e}")

            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}", f"{lr:.6f}", time.time(), ""])

        # Evaluate on test (single sample) for final prediction
        model.eval()
        pred_start = time.time()
        with torch.no_grad():
            for feats, targets in test_loader:
                feats = feats.to(device)
                targets = targets.to(device).view(-1, 1)
                outputs = model(feats)
                predictions.append(outputs.cpu().numpy().flatten()[0])
                ground_truths.append(targets.cpu().numpy().flatten()[0])
                image_ids.append(dataset.ids[i])
        pred_time = time.time() - pred_start
        total_time = pred_time + avg_cache_time_per_sample
        print(f"Iter {i+1} total execution time: {total_time:.4f}s (pred: {pred_time:.4f}s + cache: {avg_cache_time_per_sample:.6f}s)")
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['final', '', '', '', time.time(), f"{total_time:.6f}"])

    # Compute metrics
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    mae = mean_absolute_error(ground_truths, predictions)
    rmse = root_mean_squared_error(ground_truths, predictions)
    r2 = r2_score(ground_truths, predictions)

    print(f"LOOCV Results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    # Save results
    results_df = pd.DataFrame({
        'image_id': image_ids,
        'ground_truth': ground_truths,
        'prediction': predictions
    })
    
    results_df.to_csv(Path(f'experiments/runs/{feature_extractor}/loocv_results.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description='Perform Leave-One-Out Cross-Validation')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--force-cpu', action='store_true')
    args = parser.parse_args()

    loocv_evaluation(args.config, force_cpu=args.force_cpu)

if __name__ == '__main__':
    main()