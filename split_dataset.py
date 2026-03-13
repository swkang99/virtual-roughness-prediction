"""
Split dataset triples (same filename stem across subfolders) into train/valid sets.

Usage:
    python split_dataset.py --data-dir data --train-size 50 --valid-size 50 --seed 42

This will create:
    data/train/<subfolder>/*.png
    data/valid/<subfolder>/*.png
and CSV files:
    data/train_ids.csv
    data/valid_ids.csv

By default files are copied (shutil.copy2). If there are fewer complete triples than requested,
sizes will be adjusted automatically.
"""
from pathlib import Path
import argparse
import random
import shutil
import csv
import sys


def find_subfolders(data_dir: Path):
    # Find subdirectories containing image files, ignoring archive files
    subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
    # Filter out common unwanted dirs like 'train' or 'valid' if they already exist
    subdirs = [p for p in subdirs if p.name.lower() not in ("train", "valid")] 
    # Keep only those that contain at least one file with an image extension
    img_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    good = []
    for d in subdirs:
        for f in d.iterdir():
            if f.suffix.lower() in img_exts and f.is_file():
                good.append(d)
                break
    return sorted(good)


def collect_ids(subfolders):
    sets = []
    for d in subfolders:
        stems = {f.stem for f in d.iterdir() if f.is_file()}
        sets.append(stems)
    # intersection across all
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def copy_split(ids, split_ids, subfolders, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # create subfolders inside out_dir to match original structure
    for sf in subfolders:
        (out_dir / sf.name).mkdir(parents=True, exist_ok=True)
    copied = 0
    for sid in split_ids:
        for sf in subfolders:
            src_candidates = list(sf.glob(f"{sid}.*"))
            if not src_candidates:
                # no file with that stem in this subfolder; skip and warn
                print(f"Warning: missing {sid} in {sf}")
                continue
            # choose the first candidate (preserves original extension)
            src = src_candidates[0]
            dst = out_dir / sf.name / src.name
            shutil.copy2(str(src), str(dst))
            copied += 1
    return copied


def write_csv(path: Path, ids):
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id'])
        for i in ids:
            w.writerow([i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'), help='Path to data directory')
    parser.add_argument('--train-size', type=int, default=50, help='Number of items for train')
    parser.add_argument('--valid-size', type=int, default=50, help='Number of items for valid')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dry-run', action='store_true', help='Do not copy files, just print plan')
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"Error: data dir {data_dir} does not exist", file=sys.stderr)
        sys.exit(2)

    subfolders = find_subfolders(data_dir)
    if not subfolders:
        print(f"Error: no valid subfolders with images found in {data_dir}", file=sys.stderr)
        sys.exit(2)

    print("Found subfolders:")
    for s in subfolders:
        print("  -", s.name)

    ids = collect_ids(subfolders)
    total = len(ids)
    print(f"Found {total} complete items (present in all subfolders)")

    wanted = args.train_size + args.valid_size
    if total < wanted:
        print(f"Warning: requested {wanted} items but only {total} available. Adjusting sizes.")
        # make train/valid roughly half-half while keeping requested ratio if possible
        half = total // 2
        train_size = min(args.train_size, half)
        valid_size = total - train_size
    else:
        train_size = args.train_size
        valid_size = args.valid_size

    random.seed(args.seed)
    ids_copy = ids.copy()
    random.shuffle(ids_copy)
    train_ids = ids_copy[:train_size]
    valid_ids = ids_copy[train_size:train_size+valid_size]

    print(f"Train: {len(train_ids)} items, Valid: {len(valid_ids)} items")

    if args.dry_run:
        print("Dry run: no files will be copied. Example train ids:")
        print(train_ids[:5])
        return

    # perform copy
    train_out = data_dir / 'train'
    valid_out = data_dir / 'valid'
    copied_train = copy_split(ids, train_ids, subfolders, train_out)
    copied_valid = copy_split(ids, valid_ids, subfolders, valid_out)

    write_csv(data_dir / 'train_ids.csv', train_ids)
    write_csv(data_dir / 'valid_ids.csv', valid_ids)

    print(f"Copied: {copied_train} files to {train_out}")
    print(f"Copied: {copied_valid} files to {valid_out}")
    print("Done.")


if __name__ == '__main__':
    main()
