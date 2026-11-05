"""
Aggregate test/holdout_eval.py runs into the paper's headline comparison table (PPT 261109
slide 5): Model | Texture MAE | CCC | Runtime (ms), plus the significance tests (strategy doc §7):
    primary (uncorrected):   Ours vs Simple 1D-CNN              - Wilcoxon signed-rank, n=8
    secondary (Holm-corrected): Ours vs {LR, SVR, ANN, Hassan 1D-CNN, Scirep 1D-CNN}
Both run on paired per-texture absolute error (|pred-gt|) over the 8 Tier A holdout textures.
Runtime numbers are not computed here (they come from test/model_running_time.py, reused unchanged
per the strategy doc - training recipe changed, inference graph did not); pass them via --runtime.

    python -m test.holdout_compare output/results_holdout
    python -m test.holdout_compare output/results_holdout --runtime output/results_holdout/runtime.csv
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DISPLAY = {'lr': 'LR', 'svr': 'SVR', 'ann': 'ANN', 'cnn_1d_wassem': 'Hassan 1D-CNN',
           'cnn_1d_scirep': 'Scirep 1D-CNN', 'cnn_1d_simple': 'Simple 1D-CNN', 'transformer': 'Transformer (ours)'}
ORDER = ['lr', 'svr', 'ann', 'cnn_1d_wassem', 'cnn_1d_scirep', 'cnn_1d_simple', 'transformer']
PRIMARY_REF = 'transformer'
PRIMARY_VS = 'cnn_1d_simple'
SECONDARY_VS = ['lr', 'svr', 'ann', 'cnn_1d_wassem', 'cnn_1d_scirep']


def tex_metrics(P):
    t = P.groupby('texture').agg(g=('gt', 'first'), p=('pred', 'mean'))
    ccc = 2 * np.cov(t.g, t.p, bias=True)[0, 1] / (t.g.var(ddof=0) + t.p.var(ddof=0) + (t.g.mean() - t.p.mean()) ** 2)
    return dict(tex_MAE=float(np.abs(t.p - t.g).mean()), CCC=float(ccc))


def texture_abs_err(P):
    t = P.groupby('texture').agg(g=('gt', 'first'), p=('pred', 'mean'))
    return (t.p - t.g).abs()


def load_runs(root, dirs=None):
    runs = {}
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir() or (dirs is not None and d.name not in dirs):
            continue
        pred = d / 'pred.csv'
        if not pred.exists():
            continue
        args = json.load(open(d / 'args.json'))
        key = args['model']
        if key in runs:
            print(f'[warn] multiple holdout runs for model={key}; keeping {runs[key]["dir"].name}, '
                  f'ignoring {d.name} (pass --runs to disambiguate)')
            continue
        runs[key] = dict(dir=d, P=pd.read_csv(pred))
    return runs


def holm_correct(pvals):
    """Holm-Bonferroni step-down correction. NaN p-values pass through unchanged."""
    pv = np.asarray(pvals, float)
    holm = np.full_like(pv, np.nan)
    idx = np.where(~np.isnan(pv))[0]
    order = idx[np.argsort(pv[idx])]
    run_max = 0.0
    for i, j in enumerate(order):
        run_max = max(run_max, min(1.0, (len(order) - i) * pv[j]))
        holm[j] = run_max
    return holm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default='output/results_holdout')
    ap.add_argument('--runs', nargs='+', default=None, help='explicit run-folder names')
    ap.add_argument('--runtime', default=None, help='csv with columns model,runtime_ms,runtime_sd')
    a = ap.parse_args()

    runs = load_runs(a.root, dirs=a.runs)
    if not runs:
        raise SystemExit(f'no holdout runs (pred.csv) found under {a.root}')

    rt = pd.read_csv(a.runtime, index_col='model') if a.runtime else None
    models = [m for m in ORDER if m in runs] + [m for m in runs if m not in ORDER]

    T = {m: tex_metrics(v['P']) for m, v in runs.items()}
    md = [f'## Holdout test (Tier A, n=8 textures), single run per model', '',
          '| Model | Texture MAE | CCC | Runtime (ms) |', '|---' * 4 + '|']
    for m in models:
        rtxt = (f"{rt.loc[m, 'runtime_ms']:.2f} ± {rt.loc[m, 'runtime_sd']:.2f}"
                if rt is not None and m in rt.index else '–')
        md.append(f"| {DISPLAY.get(m, m)} | {T[m]['tex_MAE']:.2f} | {T[m]['CCC']:.3f} | {rtxt} |")

    sig = []
    if PRIMARY_REF in runs:
        ref_err = texture_abs_err(runs[PRIMARY_REF]['P'])
        sig = ['', f'## Significance ({DISPLAY[PRIMARY_REF]} vs baselines, Wilcoxon signed-rank on '
                   f'per-texture |pred-gt|, n={len(ref_err)})', '',
               '| Comparison | mean Δ|err| (model − ours) | p | p (Holm) |', '|---' * 4 + '|']

        def wilcoxon_vs(key):
            o_err = texture_abs_err(runs[key]['P']).loc[ref_err.index]
            d = o_err - ref_err
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else np.nan
            return d.mean(), p

        if PRIMARY_VS in runs:
            dmean, p = wilcoxon_vs(PRIMARY_VS)
            sig.append(f'| vs {DISPLAY[PRIMARY_VS]} (primary, uncorrected) | {dmean:+.3f} | {p:.4f} | – |')
        else:
            print(f'[warn] primary comparison model {PRIMARY_VS!r} not found among runs; skipping')

        sec_keys = [m for m in SECONDARY_VS if m in runs]
        missing = [m for m in SECONDARY_VS if m not in runs]
        if missing:
            print(f'[warn] secondary comparison models not found, skipped: {missing}')
        results = [(m, *wilcoxon_vs(m)) for m in sec_keys]
        holm_p = holm_correct([p for _, _, p in results])
        for (m, dmean, p), hp in zip(results, holm_p):
            sig.append(f'| vs {DISPLAY.get(m, m)} (secondary, Holm–Bonferroni n={len(sec_keys)}) | '
                       f'{dmean:+.3f} | {p:.4f} | {hp:.4f} |')
    else:
        print(f'[warn] reference model {PRIMARY_REF!r} not among runs; skipping significance tests')

    text = '\n'.join(md + sig)
    (Path(a.root) / 'HOLDOUT_SUMMARY.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
