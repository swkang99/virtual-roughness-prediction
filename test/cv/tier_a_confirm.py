"""
Confirmatory Tier A evaluation (strategy doc claude/whc2027_apt_relabeling_strategy.md, §18).

Combines the two roles §18.1 splits the old single-model-single-evaluation design into:

  Role 1 (PRIMARY STATISTICAL EVIDENCE): for each model, its single final, fully-recipe-locked CV run
      (test/cv/cv_experiment.py --eval_tier_a), whose 5*repeats fold models' Tier A predictions are
      reused - never retrained - as the repeated-evaluation sample. This script pools those
      rep{R}_fold{F}_pred_tierA.csv files, reports per-repeat texture MAE/slope/r/CCC (mean ± SD over
      repeats, same methodology as the C0/C4/E1-E8 screening in §16/§17), and runs the primary/secondary
      confirmatory hypothesis tests on a per-texture point estimate (every fold model's Tier A
      prediction for that texture, averaged over ALL repeats and folds).

  Role 2 (deployment / User Study model, reported ALONGSIDE for consistency only - NOT primary
      statistical evidence, see §18.1 point 1): each model's existing test/holdout/holdout_eval.py
      single full-pool run (one model trained once on all 98 Tier B + Tier A' textures, evaluated once).

Hypothesis tests (§18.2 - replaces the old Holm-Bonferroni design in test/holdout/holdout_compare.py):
  primary (pre-registered, uncorrected): Ours (transformer) vs Simple 1D-CNN - Wilcoxon signed-rank,
      n=8 textures, on the repeated-evaluation per-texture point estimate's |pred-gt|.
  secondary: Ours vs {LR, SVR, ANN, Hassan 1D-CNN, Scirep 1D-CNN} - Friedman omnibus test across ALL 7
      models (ranks of per-texture |pred-gt|, lower = better), Nemenyi post-hoc (Demšar 2006) only if
      the omnibus is significant. Both a chi2-approximation p-value and a Monte Carlo permutation
      p-value are reported; §18.2 recommends the permutation one given n=8 is too small for the chi2
      approximation to be trustworthy.

Usage (from repository root), once every model has its final-locked --eval_tier_a CV run AND its
test/holdout/holdout_eval.py run (role 2):
    python -m test.cv.tier_a_confirm output/results_cv --holdout_root output/results_holdout

Each positional/--cv_runs directory is expected to be one test/cv/cv_experiment.py --eval_tier_a run
folder (as under output/results_cv/<tag>/, same as test/cv/cv_aggregate.py's input) - this script finds
them itself by scanning `cv_root` for args.json files with eval_tier_a=true and the matching
rep*_fold*_pred_tierA.csv files, keyed by model name, exactly as test/holdout/holdout_compare.py does
for holdout runs.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from test.holdout.holdout_compare import (
    DISPLAY, ORDER, PRIMARY_REF, PRIMARY_VS, SECONDARY_VS, load_runs as load_holdout_runs,
)

# Nemenyi critical values q_alpha (alpha=0.05, two-sided), Demsar 2006 "Statistical Comparisons of
# Classifiers over Multiple Data Sets", Table 5(b) - indexed by k = number of models/treatments compared.
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

ALL_SEVEN = [PRIMARY_REF, PRIMARY_VS] + SECONDARY_VS   # the 7 models the paper's confirmatory tests cover


def load_cv_tier_a_runs(root, dirs=None):
    """Scan `root` for test/cv/cv_experiment.py --eval_tier_a run folders (rep{R}_fold{F}_pred_tierA.csv
    present, args.json has eval_tier_a=true), keyed by model name - mirrors
    test/holdout/holdout_compare.py's load_runs() for the CV side."""
    runs = {}
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir() or (dirs is not None and d.name not in dirs):
            continue
        files = sorted(d.glob('*_pred_tierA.csv'))
        args_path = d / 'args.json'
        if not files or not args_path.exists():
            continue
        args = json.load(open(args_path))
        if not args.get('eval_tier_a'):
            continue
        key = args['model']
        if key in runs:
            print(f'[warn] multiple --eval_tier_a CV runs for model={key}; keeping {runs[key]["dir"].name}, '
                  f'ignoring {d.name} (pass --cv_runs to disambiguate)')
            continue
        runs[key] = dict(dir=d, P=pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
    return runs


def repeated_eval_summary(P):
    """Per-repeat texture-level metrics, mean ± SD over repeats - same methodology as
    test/cv/cv_aggregate.py / the C0/C4/E1-E8 screening analysis (§16/§17), applied here to the pooled
    Tier A predictions instead of the Tier B/A' OOF predictions. Every fold model in a repeat predicts
    on ALL 8 Tier A textures (Tier A is never folded - it is held out of training entirely), so "pooling
    a repeat" here means averaging that repeat's k fold models' predictions per texture, i.e. a small
    k-model ensemble per repeat; the R repeats then give R such (correlated, not fully independent, since
    they share the same train pool and recipe) ensemble estimates to report mean ± SD over."""
    rows = []
    for r, d in P.groupby('repeat'):
        t = d.groupby('texture').agg(g=('gt', 'first'), p=('pred', 'mean'))
        has_var = t.p.std() > 1e-9
        slope = np.polyfit(t.g, t.p, 1)[0] if has_var else 0.0
        ccc = (2 * np.cov(t.g, t.p, bias=True)[0, 1] /
               (t.g.var(ddof=0) + t.p.var(ddof=0) + (t.g.mean() - t.p.mean()) ** 2))
        rows.append(dict(repeat=r, patch_MAE=np.abs(d.pred - d['gt']).mean(), tex_MAE=np.abs(t.p - t.g).mean(),
                          slope=slope, r=(np.corrcoef(t.g, t.p)[0, 1] if has_var else np.nan), CCC=ccc))
    return pd.DataFrame(rows)


def repeated_texture_point_estimate(P):
    """Collapse ALL repeats and folds into one point estimate per texture (mean prediction over every
    fold model that predicted on it) - used for the primary/secondary confirmatory hypothesis tests,
    which need exactly one paired value per texture per model (n=8)."""
    return P.groupby('texture').agg(gt=('gt', 'first'), pred=('pred', 'mean'))


def texture_abs_err(t):
    return (t.pred - t['gt']).abs()


def ms(x):
    x = np.asarray(pd.Series(x).dropna(), float)
    return f'{x.mean():.2f} ± {x.std(ddof=1):.2f}' if len(x) > 1 else (f'{x.mean():.2f}' if len(x) else 'nan')


def friedman_stat(X):
    """X: (N textures, k models) of per-texture |pred-gt| (lower = better). Friedman chi2 statistic from
    the average-rank formula (pandas rank(method='average') handles ties the same way everywhere else
    in this script)."""
    N, k = X.shape
    R = pd.DataFrame(X).rank(axis=1, method='average').values.sum(axis=0)
    return (12 * N) / (k * (k + 1)) * np.sum(R ** 2) - 3 * N * (k + 1)


def friedman_permutation_pvalue(X, n_perm, seed):
    """Monte Carlo permutation p-value for the Friedman test: independently permute each block's (each
    texture's) model labels, recompute the statistic, and see how often the permuted statistic is >= the
    observed one. Preferred over the chi2 approximation at n=8 textures (§18.2) - exact enumeration
    (k! per block) is intractable for k=7, so this uses n_perm Monte Carlo draws instead."""
    N, k = X.shape
    rng = np.random.default_rng(seed)
    obs = friedman_stat(X)
    count = 0
    Xp = np.empty_like(X)
    for _ in range(n_perm):
        for i in range(N):
            Xp[i] = X[i, rng.permutation(k)]
        if friedman_stat(Xp) >= obs - 1e-9:
            count += 1
    return (count + 1) / (n_perm + 1)   # add-one smoothing: never report p=0 from a finite sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cv_root', nargs='?', default='output/results_cv',
                    help='directory containing one --eval_tier_a run folder per model (same layout as '
                         'test/cv/cv_aggregate.py)')
    ap.add_argument('--cv_runs', nargs='+', default=None, help='explicit run-folder names under cv_root')
    ap.add_argument('--holdout_root', default='output/results_holdout',
                    help='role 2 (deployment/User Study model) run folders - test/holdout/holdout_eval.py')
    ap.add_argument('--holdout_runs', nargs='+', default=None, help='explicit run-folder names under holdout_root')
    ap.add_argument('--n_perm', type=int, default=20000, help='Monte Carlo permutations for the Friedman p-value')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    cv_runs = load_cv_tier_a_runs(a.cv_root, dirs=a.cv_runs)
    if not cv_runs:
        raise SystemExit(f'no --eval_tier_a CV runs (rep*_fold*_pred_tierA.csv + args.json eval_tier_a=true) '
                         f'found under {a.cv_root}')
    holdout_runs = load_holdout_runs(a.holdout_root, dirs=a.holdout_runs) if Path(a.holdout_root).exists() else {}
    if not holdout_runs:
        print(f'[warn] no role-2 holdout runs found under {a.holdout_root} - '
              f'Role-2 columns below will show "-"')

    models = [m for m in ORDER if m in cv_runs] + [m for m in cv_runs if m not in ORDER]

    md = ['# Tier A confirmatory evaluation (strategy doc §18)', '',
          '## Role 1 (primary statistical evidence) - CV-fold-reuse repeated evaluation, '
          'vs. Role 2 (deployment/User Study model, reported for consistency only)', '',
          '| Model | repeats | Role-1 texture MAE | slope | r | CCC | Role-2 texture MAE | Role-2 CCC |',
          '|---' * 8 + '|']
    role1_point = {}
    for m in models:
        P = cv_runs[m]['P']
        rep = repeated_eval_summary(P)
        role1_point[m] = repeated_texture_point_estimate(P)
        role2 = holdout_runs.get(m)
        if role2 is not None:
            t2 = role2['P'].groupby('texture').agg(g=('gt', 'first'), p=('pred', 'mean'))
            ccc2 = (2 * np.cov(t2.g, t2.p, bias=True)[0, 1] /
                    (t2.g.var(ddof=0) + t2.p.var(ddof=0) + (t2.g.mean() - t2.p.mean()) ** 2))
            r2_mae, r2_ccc = f'{np.abs(t2.p - t2.g).mean():.2f}', f'{ccc2:.3f}'
        else:
            r2_mae, r2_ccc = '–', '–'
        md.append(f"| {DISPLAY.get(m, m)} | {rep.repeat.nunique()} | {ms(rep.tex_MAE)} | {ms(rep.slope)} | "
                  f"{ms(rep.r)} | {ms(rep.CCC)} | {r2_mae} | {r2_ccc} |")

    # ---------------------------------------------------------------- primary hypothesis
    md += ['', '## Primary hypothesis (pre-registered, uncorrected)', '']
    if PRIMARY_REF in role1_point and PRIMARY_VS in role1_point:
        ref_err = texture_abs_err(role1_point[PRIMARY_REF])
        vs_err = texture_abs_err(role1_point[PRIMARY_VS]).loc[ref_err.index]
        d = vs_err - ref_err
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else np.nan
        md.append(f"{DISPLAY[PRIMARY_REF]} vs {DISPLAY[PRIMARY_VS]}: mean delta|err| (baseline - ours) = "
                  f"{d.mean():+.3f}, Wilcoxon signed-rank p = {p:.4f} "
                  f"(n={len(ref_err)} textures, repeated-evaluation point estimate)")
    else:
        missing = [m for m in (PRIMARY_REF, PRIMARY_VS) if m not in role1_point]
        md.append(f'[skip] missing from cv_runs: {missing}')

    # ---------------------------------------------------------------- secondary hypothesis
    md += ['', '## Secondary hypothesis - Friedman omnibus + Nemenyi post-hoc (Demsar 2006)', '']
    present = [m for m in ALL_SEVEN if m in role1_point]
    missing = [m for m in ALL_SEVEN if m not in role1_point]
    if missing:
        md.append(f'[warn] missing from cv_runs, excluded from Friedman: {[DISPLAY.get(m, m) for m in missing]}')
        md.append('')
    if len(present) >= 3:
        err_table = pd.DataFrame({m: texture_abs_err(role1_point[m]) for m in present}).dropna()
        X = err_table.values
        N, k = X.shape
        avg_rank = pd.DataFrame(X, columns=present).rank(axis=1, method='average').mean(axis=0).sort_values()
        chi2, p_chi2 = stats.friedmanchisquare(*[err_table[m].values for m in present])
        p_perm = friedman_permutation_pvalue(X, a.n_perm, a.seed)
        md.append(f'Friedman statistic = {chi2:.3f} (k={k} models, N={N} textures); '
                  f'p (chi2 approximation) = {p_chi2:.4f}; '
                  f'**p (Monte Carlo permutation, {a.n_perm} draws, preferred at this N) = {p_perm:.4f}**')
        md.append('')
        md.append('Average ranks (1 = best / lowest |pred-gt|):')
        for m, rk in avg_rank.items():
            md.append(f'  - {DISPLAY.get(m, m)}: {rk:.2f}')
        md.append('')
        if p_perm < 0.05:
            q = NEMENYI_Q05.get(k)
            if q is None:
                md.append(f'[warn] no tabulated Nemenyi q_0.05 for k={k} models; skipping post-hoc CD')
            else:
                cd = q * np.sqrt(k * (k + 1) / (6 * N))
                md.append(f'Omnibus significant -> Nemenyi post-hoc (alpha=0.05, CD = {cd:.3f}):')
                ref_rank = avg_rank.get(PRIMARY_REF)
                if ref_rank is None:
                    md.append(f'[warn] {PRIMARY_REF!r} not among present models; no post-hoc comparisons vs Ours')
                else:
                    for m in present:
                        if m == PRIMARY_REF:
                            continue
                        diff = abs(avg_rank[m] - ref_rank)
                        sig = '**significant**' if diff > cd else 'not significant'
                        md.append(f'  - {DISPLAY[PRIMARY_REF]} vs {DISPLAY.get(m, m)}: '
                                  f'|rank diff| = {diff:.3f} -> {sig}')
        else:
            md.append('Omnibus NOT significant -> no post-hoc comparisons performed '
                      '(Demsar 2006 gating rule: Nemenyi is only run after a significant Friedman omnibus).')
    else:
        md.append(f'[skip] fewer than 3 models with repeated-evaluation data (have '
                  f'{[DISPLAY.get(m, m) for m in present]})')

    text = '\n'.join(md)
    Path(a.cv_root, 'TIER_A_CONFIRM.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
