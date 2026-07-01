import argparse
import re
import sys
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

# Ensure repository root is on sys.path when running this script directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.dataframe import build_dataframe
import yaml
config_path = Path(__file__).resolve().parent.parent / "config.yaml"

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def read_training_log(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f'Log not found: {csv_path}')
    epochs = []
    train_loss = []
    val_loss = []
    lr = []
    with csv_path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row.get('epoch', len(epochs)+1)))
            except Exception:
                epochs.append(len(epochs)+1)
            train_loss.append(float(row.get('train_loss', 'nan')))
            val_loss.append(float(row.get('val_loss', 'nan')))
            try:
                lr.append(float(row.get('lr', 'nan')))
            except Exception:
                lr.append(np.nan)
    return np.array(epochs), np.array(train_loss), np.array(val_loss), np.array(lr)


def plot_losses(epochs, train_loss, val_loss, out_png: Path):
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(out_png))
    plt.close()


def plot_lr(epochs, lr_arr, out_png: Path):
    if np.all(np.isnan(lr_arr)):
        return
    plt.figure(figsize=(8,3))
    plt.plot(epochs, lr_arr, label='lr')
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(out_png))
    plt.close()


def parse_iteration_id(csv_path: Path):
    match = re.search(r'iter_(\d+)_log\.csv$', csv_path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f'Could not parse iteration id from {csv_path.name}')


def read_iter_log(csv_path: Path):
    """Read iter log CSV and extract epoch, train_loss, val_loss, predict_time"""
    if not csv_path.exists():
        raise FileNotFoundError(f'Log not found: {csv_path}')
    epochs = []
    train_loss = []
    val_loss = []
    predict_time = None
    with csv_path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_val = row.get('epoch', '').strip()
            # Skip 'final' row for epoch plot, but extract predict_time
            if epoch_val == 'final':
                try:
                    pt_val = row.get('predict_time', '').strip()
                    if pt_val:
                        predict_time = float(pt_val)
                    else:
                        predict_time = np.nan
                except (ValueError, AttributeError):
                    predict_time = np.nan
                continue
            try:
                epochs.append(int(epoch_val))
            except Exception:
                continue
            try:
                train_loss.append(float(row.get('train_loss', 'nan')))
            except Exception:
                train_loss.append(np.nan)
            try:
                val_loss.append(float(row.get('val_loss', 'nan')))
            except Exception:
                val_loss.append(np.nan)
    return np.array(epochs), np.array(train_loss), np.array(val_loss), predict_time


def plot_iter_losses(iteration_ids, train_losses, val_losses, out_png: Path, top_n=3):
    """Plot train and val loss for top-n iterations with lowest final val_loss"""
    # Calculate final val_loss for each iteration and select top-n
    final_val_losses = []
    for vl in val_losses:
        if len(vl) > 0:
            final_val_losses.append(vl[-1])  # Last val_loss value
        else:
            final_val_losses.append(float('inf'))
    
    # Get indices of top-n lowest final val_loss
    sorted_indices = sorted(range(len(final_val_losses)), key=lambda i: final_val_losses[i])[:top_n]
    
    plt.figure(figsize=(10, 6))
    for idx in sorted_indices:
        tl = train_losses[idx]
        vl = val_losses[idx]
        iter_id = iteration_ids[idx] if idx < len(iteration_ids) else idx + 1
        if len(tl) > 0:
            plt.plot(tl, label=f'iter {iter_id} train', alpha=0.7)
            plt.plot(vl, label=f'iter {iter_id} val', alpha=0.7, linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=100, bbox_inches='tight')
    plt.close()


def find_image_path(image_id: str):
    df_train, df_valid = build_dataframe()
    full_df = pd.concat([df_train, df_valid], ignore_index=True)
    for _, row in full_df.iterrows():
        try:
            sid = str(int(Path(row['texture_path']).stem))
        except Exception:
            continue
        if sid == image_id:
            return row['texture_path']
    return None


def plot_iter_predict_times(iteration_ids, predict_times, out_png: Path):
    """Plot total execution time (prediction + caching) for each iteration (in milliseconds)"""
    valid_iters = []
    valid_times_ms = []
    for iter_id, pt in zip(iteration_ids, predict_times):
        if not np.isnan(pt):
            valid_iters.append(iter_id)
            valid_times_ms.append(pt * 1000)  # Convert to milliseconds
    
    if len(valid_times_ms) == 0:
        print('No valid prediction time values found for LOOCV iterations')
        return
    
    plt.figure(figsize=(10, 5))
    plt.bar(valid_iters, valid_times_ms, alpha=0.7)
    plt.xlabel('iteration')
    plt.ylabel('total execution time (ms)')
    plt.title('Total Execution Time per Iteration (Prediction + Caching)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_png))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training_log.csv for a feature_extractor or LOOCV logs')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--results-path', type=str, default='loocv_results.csv', help='Path to loocv_results.csv')
    args = parser.parse_args()

    extractor = args.extractor or config.get('feature_extractor')
    if not extractor:
        # try to infer from checkpoints directory
        ck = Path('checkpoints')
        subs = [p for p in ck.iterdir() if p.is_dir()]
        if len(subs) == 1:
            extractor = subs[0].name
        else:
            raise SystemExit('feature_extractor not specified and could not be inferred; pass --extractor or set in config.yaml')
    
    # Plot training log
    log_path = Path('checkpoints') / extractor / 'training_log.csv'
    if not log_path.exists():
        raise SystemExit(f'Log not found: {log_path}')

    epochs, train_loss, val_loss, lr = read_training_log(log_path)

    out_dir = log_path.parent
    plot_losses(epochs, train_loss, val_loss, out_dir / 'training_loss.png')
    plot_lr(epochs, lr, out_dir / 'training_lr.png')

    print(f'Plots written to: {out_dir / "training_loss.png"} and {out_dir / "training_lr.png"}')


if __name__ == '__main__':
    main()
