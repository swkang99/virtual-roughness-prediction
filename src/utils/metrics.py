import json
from pathlib import Path
import numpy as np
import pandas as pd

def metrics(conf, mae_per_output, rmse_per_output, predictions, ground_truths, test_image_ids):
    train_type = conf['train_type']
    results_dir = Path(f'experiments/runs/{train_type}')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Name mapping for each output dimension
    per_output_names = ['rough-smooth', 'flat-bumpy', 'sticky-slippery', 'hard-soft']

    per_output_dict = {}
    for name, m, r in zip(per_output_names, mae_per_output, rmse_per_output):
        per_output_dict[name] = {'mae': float(m), 'rmse': float(r)}

    metrics = {
        'per_output': per_output_dict,
        'num_samples': int(len(predictions)),
        'train_hparams': {
            'epochs': int(conf['epochs']),
            'batch_size': int(conf['batch_size']),
            'lr': float(conf['learning_rate']),
            'weight_decay': float(conf['weight_decay']),
        }
    }

    df_data = {
        'image_id': test_image_ids,
    }
    
    # Add per-output ground_truth and prediction columns
    for i, name in enumerate(per_output_names):
        df_data[f'ground_truth_{name}'] = ground_truths[:, i]
        df_data[f'prediction_{name}'] = predictions[:, i]
        df_data[f'abs_error_{name}'] = np.abs(ground_truths[:, i] - predictions[:, i])

    results_df = pd.DataFrame(df_data)
    results_df.to_csv(results_csv, index=False)
    results_csv = results_dir / f'{train_type}_results.csv'
    metrics_json = results_dir / f'{train_type}_metrics.json'
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== Baseline Model LOOCV Training Results ===")
    print(f"\nResults saved to {results_dir}")
    print(f"  - CSV: {results_csv}")
    print(f"  - Metrics: {metrics_json}")

    # Print per-output MAE and RMSE for each named target
    try:
        for name, m, r in zip(per_output_names, mae_per_output, rmse_per_output):
            print(f"{name} MAE: {m:.4f} | RMSE: {r:.4f}")
    except Exception:
        print("Per-output MAE:", mae_per_output)
        print("Per-output RMSE:", rmse_per_output)