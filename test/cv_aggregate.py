"""
Aggregate results written by test/cv_experiment.py.

    python -m test.cv_aggregate results_cv/<tag> [results_cv/<tag2> ...] [--max_repeats 3]

For each run directory:
  * per repeat, pool the out-of-fold (OOF) predictions of all test textures
    -> patch MAE / RMSE, texture-level MAE, slope (GT->pred), Pearson r, CCC
  * report mean ± SD over repeats  (= "10회 반복 평균 ± 표준편차")
  * also fold-level mean ± SD (split-to-split variability)
  * constant baselines evaluated on the *same* folds:
        median / mean of the training-fold labels
  * train-set fit (texture MAE, slope) to diagnose under-fitting
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd


def tex_metrics(d):
    t = d.groupby('texture').agg(g=('gt', 'first'), p=('pred', 'mean'))
    if t.p.std() < 1e-9:   # constant predictor: correlation undefined
        return dict(patch_MAE=np.abs(d.pred - d['gt']).mean(), patch_RMSE=np.sqrt(((d.pred - d['gt']) ** 2).mean()),
                    tex_MAE=np.abs(t.p - t.g).mean(), slope=0.0, r=np.nan, CCC=0.0)
    slope = np.polyfit(t.g, t.p, 1)[0]
    ccc = 2 * np.cov(t.g, t.p, bias=True)[0, 1] / (t.g.var(ddof=0) + t.p.var(ddof=0) + (t.g.mean() - t.p.mean()) ** 2)
    return dict(patch_MAE=np.abs(d.pred - d['gt']).mean(), patch_RMSE=np.sqrt(((d.pred - d['gt']) ** 2).mean()),
                tex_MAE=np.abs(t.p - t.g).mean(), slope=slope, r=np.corrcoef(t.g, t.p)[0, 1], CCC=ccc)


def ms(x):
    x = np.asarray(x, float)
    return f'{x.mean():.2f} ± {x.std(ddof=1):.2f}' if len(x) > 1 else f'{x.mean():.2f}'


def summarize(run_dir, max_repeats=None):
    run_dir = Path(run_dir)
    P = pd.concat([pd.read_csv(f) for f in sorted(run_dir.glob('*_pred.csv'))], ignore_index=True)
    M = [json.load(open(f)) for f in sorted(run_dir.glob('*_meta.json'))]
    if max_repeats is not None:   # e.g. compare a 3-repeat ablation with repeats 0-2 of the baseline (same folds)
        P = P[P.repeat < max_repeats]; M = [m for m in M if m['repeat'] < max_repeats]
    lab_file = run_dir / 'labels.csv'
    labels = pd.read_csv(lab_file, index_col=0)['gt'] if lab_file.exists() else P.groupby('texture')['gt'].first()
    rows, base_rows = [], []
    for r, d in P.groupby('repeat'):
        rows.append(dict(repeat=r, **tex_metrics(d)))
        # constant baselines on identical folds
        for kind in ['median', 'mean']:
            parts = []
            for m in [m for m in M if m['repeat'] == r]:
                tr = [t for t in labels.index if t not in m['test_textures'] and t not in m['val_textures']]
                c = getattr(labels.loc[tr], kind)()
                q = d[d.fold == m['fold']].copy(); q['pred'] = c; parts.append(q)
            base_rows.append(dict(repeat=r, kind=kind, **tex_metrics(pd.concat(parts))))
    R = pd.DataFrame(rows); B = pd.DataFrame(base_rows)
    F = pd.DataFrame([dict(repeat=m['repeat'], fold=m['fold'], best_epoch=m['best_epoch'], minutes=m['minutes'],
                           **{f'test_{k}': v for k, v in m['test'].items()},
                           **{f'train_{k}': v for k, v in m['train'].items()}) for m in M])
    n_rep, n_fold = R.repeat.nunique(), F.fold.nunique()
    lines = [f'## {run_dir.name}  ({n_rep} repeats × {n_fold} folds, {P.texture.nunique()} test textures per repeat)', '',
             '| metric | model (mean ± SD over repeats) | median baseline | mean baseline |', '|---|---|---|---|']
    for k in ['patch_MAE', 'patch_RMSE', 'tex_MAE', 'slope', 'r', 'CCC']:
        lines.append(f'| {k} | {ms(R[k])} | {ms(B[B.kind=="median"][k])} | {ms(B[B.kind=="mean"][k])} |')
    lines += ['', f'fold-level test patch MAE: {ms(F.test_patch_mae)} (split-to-split variability)',
              f'train fit: texture MAE {ms(F.train_tex_mae)}, slope {ms(F.train_tex_slope)}  '
              f'(slope << 1 on *training* textures = under-fitting / range compression)',
              f'best epoch: median {F.best_epoch.median():.0f} (range {F.best_epoch.min()}–{F.best_epoch.max()}), '
              f'{F.minutes.mean():.1f} min/run', '']
    # per-texture OOF error averaged over repeats (for failure analysis)
    T = P.groupby(['repeat', 'texture']).agg(gt=('gt', 'first'), pred=('pred', 'mean')).reset_index()
    T = T.groupby('texture').agg(gt=('gt', 'first'), pred_mean=('pred', 'mean'), pred_sd_over_repeats=('pred', 'std'))
    T['abs_err'] = (T.pred_mean - T['gt']).abs()
    T.sort_values('abs_err').to_csv(run_dir / 'oof_texture_summary.csv')
    R.to_csv(run_dir / 'repeat_metrics.csv', index=False); F.to_csv(run_dir / 'fold_metrics.csv', index=False)
    B.to_csv(run_dir / 'baseline_metrics.csv', index=False)
    text = '\n'.join(lines)
    (run_dir / 'summary.md').write_text(text, encoding='utf-8')
    return text


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('dirs', nargs='+'); ap.add_argument('--max_repeats', type=int, default=None)
    a = ap.parse_args()
    for d in a.dirs:
        if list(Path(d).glob('*_pred.csv')):
            print(summarize(d, a.max_repeats))
