"""
Picks each model's best_epoch from the light CV run under the final recipe + chosen lambda
(run_cv.sh's tune_epochs stage). This is deliberately the ONLY hyperparameter selected per model - see
run_holdout.sh's module docstring for why: epoch count is "when did this architecture converge", not a
recipe design choice, so letting it differ by model doesn't reopen the "same recipe" fairness argument
(strategy doc §7) the way tuning loss/lr/lds/aug per model would.

Reads each run folder's *_meta.json (written by test/cv_experiment.py; 'best_epoch' per repeat x fold)
and reports the median across all of them per model.

    python -m test.pick_epochs output/results_cv --pairs ann:ann_cv5_l1_cos_val_tex_mae ...

Writes <root>/epochs.csv (model,best_epoch,n_runs,min,max) and prints the same as a markdown table.
run_holdout.sh reads epochs.csv directly to override --epochs per model.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd


def best_epochs_of(run_dir):
    metas = [json.load(open(f)) for f in sorted(Path(run_dir).glob('*_meta.json'))]
    return [m['best_epoch'] for m in metas if m.get('best_epoch', -1) >= 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--pairs', nargs='+', required=True, help='model:run_folder_name entries')
    a = ap.parse_args()

    rows = []
    for pair in a.pairs:
        model, folder = pair.split(':', 1)
        run_dir = Path(a.root) / folder
        epochs = best_epochs_of(run_dir)
        if not epochs:
            print(f'[warn] no completed runs with a best_epoch found for {model} in {run_dir}; skipping')
            continue
        median_epoch = int(round(float(np.median(epochs))))
        rows.append(dict(model=model, best_epoch=median_epoch, n_runs=len(epochs),
                         min=min(epochs), max=max(epochs)))

    if not rows:
        raise SystemExit('no model produced a usable best_epoch - nothing written')

    df = pd.DataFrame(rows)
    df.to_csv(Path(a.root) / 'epochs.csv', index=False)

    md = ['## Per-model epoch selection (median best_epoch over the tune_epochs CV runs)', '',
          '| Model | best_epoch (median) | n runs | range |', '|---|---|---|---|']
    for r in rows:
        md.append(f"| {r['model']} | {r['best_epoch']} | {r['n_runs']} | {r['min']}–{r['max']} |")
    print('\n'.join(md))


if __name__ == '__main__':
    main()
