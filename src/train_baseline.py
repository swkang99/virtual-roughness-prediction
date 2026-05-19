import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import pandas as pd
import yaml
import time
import json
from tqdm import tqdm

from src.data.dataset import build_original_dataframe
from src.model.feature.glcm import gray_level_co_occurrence_matrix
from src.model.feature.lbp import extract_lbp_feature
from src.model.prediction.cnn_1d import MultiScale1DCNN


def extract_glcm_features(image_array):
    glcm_2d = gray_level_co_occurrence_matrix(image_array)
    return glcm_2d.flatten().astype(np.float32)


def extract_lbp_features(image_array):
    feature_vector, lbp_maps = extract_lbp_feature(image_array, grid=(7, 7))
    return np.asarray(feature_vector, dtype=np.float32)


def extract_resnet50_features(image_tensor, model, device):
    model.eval()
    with torch.no_grad():
        features = model(image_tensor.unsqueeze(0).to(device))
    return features.cpu().numpy().flatten().astype(np.float32)


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_flops_approximate(texture_size, glcm_ops=1e5, lbp_ops=2e5, resnet50_ops=8.2e9, cnn_1d_ops=None):
    total_flops = glcm_ops + lbp_ops + resnet50_ops
    if cnn_1d_ops is not None:
        total_flops += cnn_1d_ops
    return total_flops


def estimate_1dcnn_flops(input_dim=3955):
    estimated_1dcnn_flops = 100e6
    return estimated_1dcnn_flops


class FeatureDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def extract_single_image_features(img_pil, transform, resnet50, device):
    img_np = np.array(img_pil)
    
    glcm_start = time.perf_counter()
    glcm_feat = extract_glcm_features(img_np)
    glcm_time = time.perf_counter() - glcm_start

    lbp_start = time.perf_counter()
    lbp_feat = extract_lbp_features(img_np)
    lbp_time = time.perf_counter() - lbp_start

    resnet_start = time.perf_counter()
    img_tensor = transform(img_pil)
    resnet_feat = extract_resnet50_features(img_tensor, resnet50, device)
    resnet_time = time.perf_counter() - resnet_start

    return np.concatenate([glcm_feat, lbp_feat, resnet_feat]).astype(np.float32), glcm_time, lbp_time, resnet_time

def build_all_features(full_df, transform, resnet50, device):
    all_features = []
    all_targets = []
    image_ids = []

    glcm_times = []
    lbp_times = []
    resnet_times = []

    for _, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Precompute features", unit="sample"):
        texture_path = row['texture_path']
        texture_img = Image.open(texture_path).convert('RGB')
        normal_img = Image.open(row['normal_path']).convert('RGB')
        height_img = Image.open(row['height_path']).convert('RGB')
        gt = float(row['roughness'])
 
        texture_feat, texture_glcm_time, texture_lbp_time, texture_resnet_time = extract_single_image_features(texture_img, transform, resnet50, device)
        # normal_feat, normal_glcm_time, normal_lbp_time, normal_resnet_time  = extract_single_image_features(normal_img, transform, resnet50, device)
        # height_feat, height_glcm_time, height_lbp_time, height_resnet_time  = extract_single_image_features(height_img, transform, resnet50, device)

        # combined_feat = np.concatenate([texture_feat, normal_feat, height_feat]).astype(np.float32)
        combined_feat = texture_feat

        all_features.append(combined_feat)
        all_targets.append(gt)
        image_ids.append(str(int(Path(texture_path).stem)))

        glcm_times.append(texture_glcm_time)
        # glcm_times.append(normal_glcm_time)
        # glcm_times.append(height_glcm_time)
        lbp_times.append(texture_lbp_time)
        # lbp_times.append(normal_lbp_time)
        # lbp_times.append(height_lbp_time)
        resnet_times.append(texture_resnet_time)
        # resnet_times.append(normal_resnet_time)
        # resnet_times.append(height_resnet_time)

    return (
        np.stack(all_features),
        np.array(all_targets, dtype=np.float32),
        image_ids,
        np.array(glcm_times),
        np.array(lbp_times),
        np.array(resnet_times),
    )


def train_one_fold(model, train_features, train_targets, device, epochs=200, batch_size=8, lr=1e-3, weight_decay=1e-4):
    train_dataset = FeatureDataset(train_features, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x).squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.detach().item())

    return model


def baseline_loocv_train(force_cpu=False, epochs=200, batch_size=8, lr=1e-3, weight_decay=1e-4):
    device = torch.device('cuda') if torch.cuda.is_available() and not force_cpu else torch.device('cpu')
    print(f"Device: {device}")

    full_df = build_original_dataframe()
    print(f"Full original dataset size: {len(full_df)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("Loading ResNet50...")
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet50.eval()
    resnet50.to(device)

    for p in resnet50.parameters():
        p.requires_grad = False

    print("Precomputing features for all samples...")
    (
        all_features,
        all_targets,
        image_ids,
        glcm_times,
        lbp_times,
        resnet_times,
    ) = build_all_features(full_df, transform, resnet50, device)

    train_type = 'baseline_train_loocv'
    results_dir = Path(f'experiments/runs/{train_type}')
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    ground_truths = []
    test_image_ids = []

    train_times = []
    infer_times = []
    total_fold_times = []

    input_feature_dim = 3955

    print(f"\nStarting LOOCV training with {len(all_features)} samples...")

    for test_idx in tqdm(range(len(all_features)), desc="LOOCV", unit="fold"):
        fold_start = time.perf_counter()

        train_mask = np.ones(len(all_features), dtype=bool)
        train_mask[test_idx] = False

        X_train = all_features[train_mask]
        X_test = all_features[test_idx:test_idx+1]

        y_train_raw = all_targets[train_mask]
        y_test_raw  = all_targets[test_idx:test_idx+1]

        y_min = y_train_raw.min()
        y_max = y_train_raw.max()

        y_train = (y_train_raw - y_min) / (y_max - y_min + 1e-8)
        y_test = (y_test_raw - y_min) / (y_max - y_min + 1e-8)

        model = MultiScale1DCNN(input_feature_dim=input_feature_dim).to(device)

        train_start = time.perf_counter()
        model = train_one_fold(
            model=model,
            train_features=X_train,
            train_targets=y_train,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
        )
        train_time = time.perf_counter() - train_start

        model.eval()
        infer_start = time.perf_counter()
        with torch.no_grad():
            x_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            pred_norm = model(x_test_tensor).cpu().numpy().flatten()[0]
        infer_time = time.perf_counter() - infer_start
        pred_raw = pred_norm * (y_max - y_min + 1e-8) + y_min
        y_test_raw = y_test_raw[0]
        fold_time = time.perf_counter() - fold_start

        predictions.append(pred_raw)
        ground_truths.append(y_test_raw)
        test_image_ids.append(image_ids[test_idx])

        train_times.append(train_time)
        infer_times.append(infer_time)
        total_fold_times.append(fold_time)

    predictions = np.array(predictions, dtype=np.float32)
    ground_truths = np.array(ground_truths, dtype=np.float32)

    train_times = np.array(train_times)
    infer_times = np.array(infer_times)
    total_fold_times = np.array(total_fold_times)

    mae = mean_absolute_error(ground_truths, predictions)
    rmse = root_mean_squared_error(ground_truths, predictions)
    r2 = r2_score(ground_truths, predictions)

    cnn_1d = MultiScale1DCNN(input_feature_dim=input_feature_dim)
    resnet50_params = count_model_parameters(resnet50)
    cnn_1d_params = count_model_parameters(cnn_1d)
    total_params = resnet50_params + cnn_1d_params

    cnn_1d_flops = estimate_1dcnn_flops(input_feature_dim)
    total_flops = count_flops_approximate(texture_size=448, cnn_1d_ops=cnn_1d_flops)

    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'timing': {
            'feature_extraction_ms': {
                'glcm_avg': float(glcm_times.mean() * 1000),
                'glcm_std': float(glcm_times.std() * 1000),
                'lbp_avg': float(lbp_times.mean() * 1000),
                'lbp_std': float(lbp_times.std() * 1000),
                'resnet_avg': float(resnet_times.mean() * 1000),
                'resnet_std': float(resnet_times.std() * 1000),
            },
            'loocv_ms': {
                'train_avg': float(train_times.mean() * 1000),
                'train_std': float(train_times.std() * 1000),
                'infer_avg': float(infer_times.mean() * 1000),
                'infer_std': float(infer_times.std() * 1000),
                'fold_total_avg': float(total_fold_times.mean() * 1000),
                'fold_total_std': float(total_fold_times.std() * 1000),
            }
        },
        'total_flops': float(total_flops),
        'total_parameters': int(total_params),
        'resnet50_parameters': int(resnet50_params),
        'cnn_1d_parameters': int(cnn_1d_params),
        'num_samples': int(len(predictions)),
        'train_hparams': {
            'epochs': int(epochs),
            'batch_size': int(batch_size),
            'lr': float(lr),
            'weight_decay': float(weight_decay),
        }
    }

    results_df = pd.DataFrame({
        'image_id': test_image_ids,
        'ground_truth': ground_truths,
        'prediction': predictions,
        'abs_error': np.abs(ground_truths - predictions),
        'train_time_sec': train_times,
        'infer_time_sec': infer_times,
        'fold_total_time_sec': total_fold_times,
    })

    results_csv = results_dir / f'{train_type}_results.csv'
    metrics_json = results_dir / f'{train_type}_metrics.json'

    results_df.to_csv(results_csv, index=False)
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== Baseline Model LOOCV Training Results ===")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Train time per fold (avg): {train_times.mean()*1000:.3f} ± {train_times.std()*1000:.3f} ms")
    print(f"Inference time per fold (avg): {infer_times.mean()*1000:.3f} ± {infer_times.std()*1000:.3f} ms")
    print(f"Total fold time (avg): {total_fold_times.mean()*1000:.3f} ± {total_fold_times.std()*1000:.3f} ms")
    print(f"Total FLOPs: {total_flops:.2e}")
    print(f"Total Parameters: {total_params:,}")
    print(f"\nResults saved to {results_dir}")
    print(f"  - CSV: {results_csv}")
    print(f"  - Metrics: {metrics_json}")


def main():
    parser = argparse.ArgumentParser(description='Baseline model LOOCV training with GLCM+LBP+ResNet50+1DCNN')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU execution')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    args = parser.parse_args()

    baseline_loocv_train(
        force_cpu=args.force_cpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


if __name__ == '__main__':
    main()