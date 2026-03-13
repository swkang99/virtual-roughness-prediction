import argparse
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
try:
    import yaml
except Exception:
    yaml = None


def load_config(path='config.yaml'):
    p = Path(path)
    if not p.exists():
        return {}
    if yaml is None:
        raise SystemExit('PyYAML required to read config.yaml (pip install pyyaml)')
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


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


def main():
    parser = argparse.ArgumentParser(description='Plot training_log.csv for a feature_extractor')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--extractor', type=str, help='Feature extractor name (overrides config)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    extractor = args.extractor or cfg.get('feature_extractor')
    if not extractor:
        # try to infer from checkpoints directory
        ck = Path('checkpoints')
        subs = [p for p in ck.iterdir() if p.is_dir()]
        if len(subs) == 1:
            extractor = subs[0].name
        else:
            raise SystemExit('feature_extractor not specified and could not be inferred; pass --extractor or set in config.yaml')

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
