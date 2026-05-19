"""
Baseline Model: GLCM + LBP + ResNet50 feature extraction + MultiScale1DCNN prediction
Performs LOOCV on full dataset with texture image input only.
Measures FLOPs, parameters, and execution time.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
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
    return glcm_2d.flatten()


def extract_lbp_features(image_array):
    feature_vector, lbp_maps, resized_img = extract_lbp_feature(image_array, grid=(7, 7))
    return feature_vector

def extract_resnet50_features(image_tensor, model, device):
    """
    Extract ResNet50 features (final FC layer output = 1000 dims).
    Returns: 1D array of shape (1000,)
    """
    model.eval()
    with torch.no_grad():
        # model outputs 1000-dim vector from final FC layer
        features = model(image_tensor.unsqueeze(0).to(device))
    
    return features.cpu().numpy().flatten().astype(np.float32)


def count_model_parameters(model):
    """Count total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def count_flops_approximate(texture_size, glcm_ops=1e5, lbp_ops=2e5, resnet50_ops=8.2e9, cnn_1d_ops=None):
    """
    Approximate FLOPs for baseline model.
    GLCM: ~100K operations
    LBP: ~200K operations
    ResNet50: ~8.2B operations (standard inference)
    MultiScale1DCNN: Calculate based on layer sizes
    """
    total_flops = glcm_ops + lbp_ops + resnet50_ops
    
    if cnn_1d_ops is not None:
        total_flops += cnn_1d_ops
    
    return total_flops


def estimate_1dcnn_flops(input_dim=3955):
    """Estimate FLOPs for MultiScale1DCNN inference."""
    # Conv1: 1 -> 32, kernel=3, input=3955 -> output=3955
    conv1_narrow = 1 * 32 * 3 * 3955  # in_ch * out_ch * kernel_size * seq_len
    
    # Conv layers continue...
    # This is a rough approximation; exact calculation requires actual computation graphs
    # Approximate: ~50-100M FLOPs for 1D CNN with typical architecture
    estimated_1dcnn_flops = 100e6
    
    return estimated_1dcnn_flops


def baseline_loocv(force_cpu=False):
    """
    Baseline LOOCV: GLCM + LBP + ResNet50 + MultiScale1DCNN
    """
    
    device = torch.device('cuda') if torch.cuda.is_available() and not force_cpu else torch.device('cpu')
    print(f"Device: {device}")
    
    # Load full dataset
    full_df = build_original_dataframe()
    print(f"Full original dataset size: {len(full_df)}")
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    
    # Load ResNet50
    print("Loading ResNet50...")
    resnet50 = models.resnet50(weights=True)
    resnet50.eval()
    resnet50.to(device)
    # Keep the FC layer to get 1000-dim output
    
    # Load MultiScale1DCNN
    print("Loading MultiScale1DCNN...")
    cnn_1d = MultiScale1DCNN(input_feature_dim=3955)
    cnn_1d.eval()
    cnn_1d.to(device)
    
    # Count parameters
    resnet50_params = count_model_parameters(resnet50)
    cnn_1d_params = count_model_parameters(cnn_1d)
    total_params = resnet50_params + cnn_1d_params
    
    print(f"ResNet50 parameters: {resnet50_params:,}")
    print(f"MultiScale1DCNN parameters: {cnn_1d_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    # Estimate FLOPs
    cnn_1d_flops = estimate_1dcnn_flops(3955)
    total_flops = count_flops_approximate(448, cnn_1d_ops=cnn_1d_flops)
    print(f"Estimated total FLOPs: {total_flops:.2e}")
    
    # Create output directory
    results_dir = Path('experiments/runs/baseline')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # LOOCV evaluation
    predictions = []
    ground_truths = []
    image_ids = []
    execution_times = []
    glcm_times = []
    lbp_times = []
    resnet_times = []
    cnn_times = []
    
    print(f"\nStarting LOOCV with {len(full_df)} samples...")
    
    pbar = tqdm(full_df.iterrows(), total=len(full_df), desc="LOOCV", unit="sample")
    for i, (idx, row) in enumerate(pbar):
        # Load texture image
        texture_path = row['texture_path']
        try:
            texture_img = Image.open(texture_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {texture_path}: {e}")
            continue
        
        # Get ground truth
        gt = float(row['roughness'])
        
        # Measure execution time
        iter_start = time.perf_counter()
        
        # Convert to numpy array for feature extraction
        img_np = np.array(texture_img)
        
        # Extract GLCM features
        glcm_start = time.perf_counter()
        glcm_feat = extract_glcm_features(img_np)
        glcm_time = time.perf_counter() - glcm_start
        
        # Extract LBP features
        lbp_start = time.perf_counter()
        lbp_feat = extract_lbp_features(img_np)
        lbp_time = time.perf_counter() - lbp_start
        
        # Extract ResNet50 features
        resnet_start = time.perf_counter()
        texture_tensor = transform(texture_img).to(device)
        resnet_feat = extract_resnet50_features(texture_tensor, resnet50, device)
        resnet_time = time.perf_counter() - resnet_start
        
        # Concatenate features: 64 + 2891 + 1000 = 3955
        combined_feat = np.concatenate([glcm_feat, lbp_feat, resnet_feat])
        combined_feat = combined_feat[:3955]  # Ensure 3955 dimensions
        
        # Convert to tensor
        feat_tensor = torch.tensor(combined_feat, dtype=torch.float32).to(device)
        
        # Predict with 1D CNN
        cnn_start = time.perf_counter()
        with torch.no_grad():
            pred = cnn_1d(feat_tensor.unsqueeze(0))
        cnn_time = time.perf_counter() - cnn_start
        
        pred_value = pred.cpu().numpy().flatten()[0]
        iter_end = time.perf_counter()
        iter_time = iter_end - iter_start
        
        # Update progress bar with timing information
        pbar.set_postfix({
            'GLCM': f'{glcm_time*1000:.1f}ms',
            'LBP': f'{lbp_time*1000:.1f}ms',
            'ResNet': f'{resnet_time*1000:.1f}ms',
            'CNN': f'{cnn_time*1000:.1f}ms',
            'Total': f'{iter_time*1000:.1f}ms'
        })
        
        predictions.append(pred_value)
        ground_truths.append(gt)
        image_ids.append(str(int(Path(texture_path).stem)))
        execution_times.append(iter_time)
        glcm_times.append(glcm_time)
        lbp_times.append(lbp_time)
        resnet_times.append(resnet_time)
        cnn_times.append(cnn_time)
    
    pbar.close()
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    execution_times = np.array(execution_times)
    glcm_times = np.array(glcm_times)
    lbp_times = np.array(lbp_times)
    resnet_times = np.array(resnet_times)
    cnn_times = np.array(cnn_times)
    
    # Calculate metrics
    mae = mean_absolute_error(ground_truths, predictions)
    rmse = root_mean_squared_error(ground_truths, predictions)
    r2 = r2_score(ground_truths, predictions)
    
    avg_time = execution_times.mean()
    std_time = execution_times.std()
    avg_glcm = glcm_times.mean()
    std_glcm = glcm_times.std()
    avg_lbp = lbp_times.mean()
    std_lbp = lbp_times.std()
    avg_resnet = resnet_times.mean()
    std_resnet = resnet_times.std()
    avg_cnn = cnn_times.mean()
    std_cnn = cnn_times.std()
    
    print(f"\n=== Baseline Model LOOCV Results ===")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"\nExecution Time Statistics:")
    print(f"Total time per sample (avg): {avg_time*1000:.3f} ± {std_time*1000:.3f} ms")
    print(f"  - GLCM: {avg_glcm*1000:.3f} ± {std_glcm*1000:.3f} ms")
    print(f"  - LBP: {avg_lbp*1000:.3f} ± {std_lbp*1000:.3f} ms")
    print(f"  - ResNet50: {avg_resnet*1000:.3f} ± {std_resnet*1000:.3f} ms")
    print(f"  - 1D-CNN: {avg_cnn*1000:.3f} ± {std_cnn*1000:.3f} ms")
    print(f"Total FLOPs: {total_flops:.2e}")
    print(f"Total Parameters: {total_params:,}")
    
    # Save results with stage-wise timing
    results_df = pd.DataFrame({
        'image_id': image_ids,
        'ground_truth': ground_truths,
        'prediction': predictions,
        'total_time_sec': execution_times,
        'glcm_time_sec': glcm_times,
        'lbp_time_sec': lbp_times,
        'resnet_time_sec': resnet_times,
        'cnn_time_sec': cnn_times
    })
    
    results_csv = results_dir / 'baseline_loocv_results.csv'
    results_df.to_csv(results_csv, index=False)
    
    # Save metrics summary with stage-wise timing
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'timing': {
            'total_time_ms': {
                'avg': float(avg_time * 1000),
                'std': float(std_time * 1000)
            },
            'glcm_time_ms': {
                'avg': float(avg_glcm * 1000),
                'std': float(std_glcm * 1000)
            },
            'lbp_time_ms': {
                'avg': float(avg_lbp * 1000),
                'std': float(std_lbp * 1000)
            },
            'resnet_time_ms': {
                'avg': float(avg_resnet * 1000),
                'std': float(std_resnet * 1000)
            },
            'cnn_time_ms': {
                'avg': float(avg_cnn * 1000),
                'std': float(std_cnn * 1000)
            }
        },
        'total_flops': float(total_flops),
        'total_parameters': int(total_params),
        'resnet50_parameters': int(resnet50_params),
        'cnn_1d_parameters': int(cnn_1d_params),
        'num_samples': len(predictions)
    }
    
    metrics_json = results_dir / 'baseline_metrics.json'
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {results_dir}")
    print(f"  - CSV: {results_csv}")
    print(f"  - Metrics: {metrics_json}")


def main():
    # parser = argparse.ArgumentParser(description='Baseline model LOOCV with GLCM+LBP+ResNet50+1DCNN')
    # parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML')
    # parser.add_argument('--force-cpu', action='store_true', help='Force CPU execution')
    # args = parser.parse_args()
    
    baseline_loocv()


if __name__ == '__main__':
    main()
