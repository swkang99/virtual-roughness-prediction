"""
Compare several models run with test/cv_experiment.py on identical folds.

    python -m test.cv_compare output/results_cv --protocol cv    --ref transformer
    python -m test.cv_compare output/results_cv --protocol fixed --ref transformer

Because make_folds() depends only on (labels, seed), every model sees exactly the
same train/val/test textures in repeat r -> paired comparisons are valid.

Outputs (in <results_dir>/compare_<protocol>/):
  table.md / table.tex     mean ± SD over repeats for each model (+ constant median baseline)
  paired_vs_<ref>.md       per-repeat Δ(MAE, slope, CCC) vs reference, wins, Wilcoxon signed-rank tests (+ Holm)

    --runs  dirA dirB ...   compare exactly these run folders (any setting), e.g. final recipe vs paper baselines
    --ref   model name or run-folder name
    --name  output folder name (default compare_<protocol>, or compare_<protocol>_ablation with --include_ablations)
  per_repeat.csv, per_texture.csv
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from test.cv_aggregate import tex_metrics

DISPLAY = {'lr': 'LR', 'svr': 'SVR', 'ann': 'ANN', 'cnn_1d_simple': 'Simple 1D-CNN',
           'cnn_1d_wassem': 'Hassan 1D-CNN', 'cnn_1d_scirep': 'Scirep 1D-CNN',
           'transformer': 'Transformer (ours)', 'gated_mlp': 'Gated MLP', 'gated_mlp_v2': 'Gated MLP v2'}
ORDER = ['lr', 'svr', 'ann', 'cnn_1d_wassem', 'cnn_1d_scirep', 'cnn_1d_simple', 'transformer']
METRICS = [('patch_MAE', 'MAE (patch)', 'min'), ('patch_RMSE', 'RMSE (patch)', 'min'),
           ('tex_MAE', 'MAE (texture)', 'min'), ('slope', 'Slope', 'one'),
           ('r', 'Pearson r', 'max'), ('CCC', 'CCC', 'max')]


def load_runs(root, protocol, only_paper_setting=True, dirs=None, feature_mode='original'):
    runs = {}
    for d in sorted(Path(root).iterdir()):
        if dirs is not None and d.name not in dirs:
            continue
        a = d / 'args.json'
        if not a.exists() or not list(d.glob('*_pred.csv')):
            continue
        args = json.load(open(a))
        if args['protocol'] != protocol:
            continue
        fm = args.get('feature_mode') or ('legacy' if args['model'] not in ('transformer', 'cnn_1d_simple') else None)
        paper = (args['loss'] == 'mse' and not args['lds'] and not args['aug']
                 and not args['cosine'] and args['select'] == 'val_rmse' and fm in (None, feature_mode))
        key = args['model'] if paper else f"{args['model']}:{d.name}"
        if only_paper_setting and not paper and dirs is None:
            continue
        P = pd.concat([pd.read_csv(f) for f in sorted(d.glob('*_pred.csv'))], ignore_index=True)
        M = [json.load(open(f)) for f in sorted(d.glob('*_meta.json'))]
        runs[key] = dict(dir=d, P=P, M=M, labels=pd.read_csv(d / 'labels.csv', index_col=0)['gt'])
    return runs


def median_baseline(run):
    rows = []
    for r, d in run['P'].groupby('repeat'):
        parts = []
        for m in [m for m in run['M'] if m['repeat'] == r]:
            tr = [t for t in run['labels'].index if t not in m['test_textures'] and t not in m['val_textures']]
            q = d[d.fold == m['fold']].copy(); q['pred'] = run['labels'].loc[tr].median(); parts.append(q)
        rows.append(dict(repeat=r, **tex_metrics(pd.concat(parts))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default='output/results_cv')
    ap.add_argument('--protocol', choices=['cv', 'fixed'], default='cv')
    ap.add_argument('--ref', default='transformer')
    ap.add_argument('--include_ablations', action='store_true', help='also list non-paper settings')
    ap.add_argument('--runs', nargs='+', default=None, help='explicit list of run folder names to compare')
    ap.add_argument('--name', default=None, help='output folder name inside root')
    ap.add_argument('--feature_mode', default='original', help='which feature-extraction mode counts as the baseline setting')
    a = ap.parse_args()

    runs = load_runs(a.root, a.protocol, only_paper_setting=not a.include_ablations, dirs=a.runs, feature_mode=a.feature_mode)
    if not runs:
        raise SystemExit(f'no {a.protocol} runs found in {a.root}')
    # --ref may be a model name ('transformer') or a run-folder name
    if a.ref not in runs:
        hit = [k for k, v in runs.items() if v['dir'].name == a.ref]
        if hit:
            a.ref = hit[0]
    name_out = a.name or (f'compare_{a.protocol}_ablation' if a.include_ablations else f'compare_{a.protocol}')
    out = Path(a.root) / name_out; out.mkdir(exist_ok=True)

    # common repeats across all models (paired design)
    common = sorted(set.intersection(*[set(v['P'].repeat.unique()) for v in runs.values()]))
    per_rep, per_tex = [], []
    for k, v in runs.items():
        P = v['P'][v['P'].repeat.isin(common)]
        for r, d in P.groupby('repeat'):
            per_rep.append(dict(model=k, repeat=r, **tex_metrics(d)))
        t = P.groupby(['repeat', 'texture']).agg(gt=('gt', 'first'), pred=('pred', 'mean')).reset_index()
        t = t.groupby('texture').agg(gt=('gt', 'first'), pred=('pred', 'mean'))
        t['abs_err'] = (t.pred - t['gt']).abs(); t['model'] = k
        per_tex.append(t.reset_index())
    any_run = next(iter(runs.values()))
    any_run = dict(any_run, P=any_run['P'][any_run['P'].repeat.isin(common)])
    base = median_baseline(any_run).assign(model='median_baseline')
    R = pd.concat([pd.DataFrame(per_rep), base], ignore_index=True)
    TT = pd.concat(per_tex, ignore_index=True)
    R.to_csv(out / 'per_repeat.csv', index=False); TT.to_csv(out / 'per_texture.csv', index=False)

    models = [m for m in ORDER if m in runs] + [m for m in runs if m not in ORDER] + ['median_baseline']
    name = lambda m: 'Median of train labels' if m == 'median_baseline' else DISPLAY.get(m.split(':')[0], m) + (f" [{m.split(':')[1]}]" if ':' in m else '')
    agg = R.groupby('model')[[c for c, _, _ in METRICS]].agg(['mean', 'std'])

    best = {}
    for c, _, rule in METRICS:
        vals = agg[(c, 'mean')].drop('median_baseline', errors='ignore')
        best[c] = (vals.idxmin() if rule == 'min' else vals.idxmax() if rule == 'max' else (vals - 1).abs().idxmin())

    n_rep = len(common); n_fold = len({m['fold'] for m in any_run['M']})
    head = f'{a.protocol.upper()}: {n_rep} repeats × {n_fold} fold(s); mean ± SD over repeats; identical folds for all models'
    md = [f'## {head}', '', '| Model | ' + ' | '.join(l for _, l, _ in METRICS) + ' |', '|---' * (len(METRICS) + 1) + '|']
    tex = [r'\begin{table}[t]', r'\centering', r'\caption{' + head.replace('×', r'$\times$').replace('±', r'$\pm$') + '}',
           r'\label{tab:cv_compare_' + a.protocol + '}', r'\resizebox{\linewidth}{!}{%',
           r'\begin{tabular}{l' + 'c' * len(METRICS) + '}', r'\toprule',
           'Model & ' + ' & '.join(l for _, l, _ in METRICS) + r' \\', r'\midrule']
    for m in models:
        cells_md, cells_tex = [], []
        for c, _, _ in METRICS:
            mu, sd = agg.loc[m, (c, 'mean')], agg.loc[m, (c, 'std')]
            if np.isnan(mu):
                cells_md.append('–'); cells_tex.append('--'); continue
            s = f'{mu:.2f} ± {sd:.2f}' if n_rep > 1 else f'{mu:.2f}'
            st = f'{mu:.2f} $\\pm$ {sd:.2f}' if n_rep > 1 else f'{mu:.2f}'
            if best[c] == m:
                s, st = f'**{s}**', r'\textbf{' + st + '}'
            cells_md.append(s); cells_tex.append(st)
        if m == 'median_baseline':
            tex.append(r'\midrule')
        md.append(f'| {name(m)} | ' + ' | '.join(cells_md) + ' |')
        tex.append(f'{name(m)} & ' + ' & '.join(cells_tex) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    md += ['', 'Slope: regression slope of texture-mean prediction on ground truth (1 = no range compression). '
               'CCC: concordance correlation coefficient. Bold = best model (baseline excluded).']
    (out / 'table.md').write_text('\n'.join(md), encoding='utf-8')
    (out / 'table.tex').write_text('\n'.join(tex), encoding='utf-8')

    # paired comparisons vs reference: MAE (lower better), slope & CCC (higher better)
    if a.ref in runs:
        n_tex = int((TT.model == a.ref).sum())
        PAIR = [('patch_MAE', 'ΔMAE', -1), ('slope', 'Δslope', +1), ('CCC', 'ΔCCC', +1)]
        hdr = ' | '.join(f'{l} (ref − model), mean ± SD | ref better | p | p Holm' for _, l, _ in PAIR)
        pm = [f'## Paired comparison vs {name(a.ref)} ({a.protocol}, {n_rep} repeats, {n_tex} test textures)', '',
              f'| Model | {hdr} | ΔMAE per texture | p (textures) |', '|---' * (1 + 4 * len(PAIR) + 2) + '|']
        others = [m for m in models if m != a.ref]
        cells = {m: [] for m in others}
        for c, _, sign in PAIR:
            ref_r = R[R.model == a.ref].set_index('repeat')[c]
            ps, rows = [], {}
            for m in others:
                d = ref_r - R[R.model == m].set_index('repeat')[c].loc[ref_r.index]   # ref − model
                good = d * sign                                                     # >0 : reference is better
                ok = n_rep >= 5 and np.any(d != 0) and not (m == 'median_baseline' and c != 'patch_MAE')
                p = stats.wilcoxon(d).pvalue if ok else np.nan
                rows[m] = (d, good, p); ps.append(p)
            # Holm correction across the compared models (per metric)
            pv = np.array(ps, float); holm = np.full_like(pv, np.nan); idx = np.where(~np.isnan(pv))[0]
            order = idx[np.argsort(pv[idx])]; run_max = 0.0
            for i, j in enumerate(order):
                run_max = max(run_max, min(1.0, (len(order) - i) * pv[j])); holm[j] = run_max
            for m, h in zip(others, holm):
                d, good, p = rows[m]
                cells[m].append(f'{d.mean():+.3f} ± {d.std(ddof=1) if n_rep > 1 else 0:.3f} | '
                                f'{int((good > 0).sum())}/{n_rep} | {p:.4f} | {h:.4f}')
        ref_t = TT[TT.model == a.ref].set_index('texture').abs_err
        for m in others:
            if m == 'median_baseline':
                dt, p_t = np.nan, np.nan
            else:
                ot = TT[TT.model == m].set_index('texture').abs_err.loc[ref_t.index]
                dt = (ot - ref_t).mean(); p_t = stats.wilcoxon(ot - ref_t).pvalue
            pm.append(f'| {name(m)} | ' + ' | '.join(cells[m]) + f' | {dt:+.2f} | {p_t:.4g} |')
        pm += ['', 'Each Δ is reference minus model per repeat; "ref better" counts repeats where the reference has lower MAE '
                   '/ higher slope / higher CCC. Wilcoxon signed-rank over repeats (10 repeats: smallest two-sided p = 0.002); '
                   'Holm = Holm–Bonferroni across the listed models for that metric. Texture-level column: per-texture '
                   f'absolute error averaged over repeats, model minus ref (> 0: ref better), {n_tex} textures.']
        safe = a.ref.replace(':', '__')                                 # ':' is not allowed in Windows file names
        (out / f'paired_vs_{safe}.md').write_text('\n'.join(pm), encoding='utf-8')
        print('\n'.join(md)); print(); print('\n'.join(pm))
    else:
        print('\n'.join(md))


if __name__ == '__main__':
    main()
