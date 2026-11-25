"""
Pre-confirmatory CV diagnostics (strategy doc claude/whc2027_apt_relabeling_strategy.md, §10.3).

Run AFTER every recipe (Ours = C0, each baseline = its own original-paper recipe - §12) is 100% locked
and the final CV (`run/run_cv.sh confirm` = test/cv/cv_experiment.py --eval_tier_a, repeats=10) has
finished for every model, but BEFORE test/cv/tier_a_confirm.py's confirmatory Tier A numbers are read or
acted on in any way. That ordering is the whole point of §10.3: catch a problem in a view that CANNOT
leak Tier A information (CV repeat-to-repeat stability, best_epoch selection stability, and this CV's
agreement with the separately-trained Role-2 holdout model - none of these ever look at Tier A) before
the one-shot confirmatory Tier A result is examined.

Most of §10.1/§10.2's nine original contingency scenarios were specifically about side effects of the
auxiliary-loss mechanism (PK sampler / aggregation+variance loss / ranking loss); since §17 discarded
that mechanism entirely (Ours' final recipe is C0, no auxiliary losses), those scenarios no longer apply.
What is still live and checked here:
  1. Repeat-to-repeat stability: does this recipe's mean OOF texture MAE over its first --tune_repeats
     (default 2, matching run/run_cv.sh's tune_epochs stage) repeats differ a lot from the mean over all
     10 repeats? A big swing means a light few-repeat CV would have been an unreliable signal for this
     model/recipe (relevant mainly as a retrospective check on tune_epochs's epoch selection - it has no
     bearing on the Tier A result itself, which was never based on a light-CV signal).
  2. best_epoch selection stability: coefficient of variation of best_epoch across all 5*repeats fold
     models (precedent: ANN's best_epoch ranged 1-145 across folds - see §8).
  3. CV (this run's Tier B/A' out-of-fold result) vs. Role-2 holdout (single full-pool model, already
     evaluated on Tier A) agreement in how the 7 models rank against each other (Spearman rho). A large
     disagreement is exactly the "CV<->holdout inconsistency" already seen once historically (§8: the
     Transformer ranked 1st in CV but 5th on the Tier A holdout). It is not something this script can fix
     or should try to - the two evaluations measure different things (Tier B/A' OOF vs. Tier A) by
     design and are not expected to match exactly - it is only worth knowing about in advance.

This script never reads *_pred_tierA.csv (the Role-1 Tier A sample) or test/cv/tier_a_confirm.py's
output - only the ordinary *_pred.csv (Tier B/A' OOF) and the Role-2 holdout's pred.csv.

Usage (from repository root), once run/run_cv.sh confirm and run/run_holdout.sh have both finished for
every model:
    python -m test.cv.pre_confirm_diagnostics output/results_cv --holdout_root output/results_holdout
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from test.cv.cv_aggregate import tex_metrics, complete_repeats
from test.holdout.holdout_compare import DISPLAY, ORDER, load_runs as load_holdout_runs


def load_cv_runs(root, dirs=None):
    """Find every model's locked-recipe CV run (args.json + *_pred.csv present), keyed by model name."""
    runs = {}
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir() or (dirs is not None and d.name not in dirs):
            continue
        if not list(d.glob('*_pred.csv')):
            continue
        args_path = d / 'args.json'
        if not args_path.exists():
            continue
        args = json.load(open(args_path))
        key = args['model']
        if key in runs:
            print(f'[warn] multiple CV runs for model={key}; keeping {runs[key]["dir"].name}, '
                  f'ignoring {d.name} (pass --cv_runs to disambiguate)')
            continue
        runs[key] = dict(dir=d, args=args)
    return runs


def cv_tex_mae(run_dir, max_repeats=None):
    """Mean +/- SD of texture MAE over complete repeats (optionally capped at max_repeats), pooling all
    folds of each repeat first - same methodology as test/cv/cv_aggregate.py. Returns
    ((mean, sd) or (None, None), per-repeat tex_MAE array or None, list of best_epoch values (sklearn
    models contribute none - cv_experiment.py records best_epoch=-1 for them))."""
    d = Path(run_dir)
    P = pd.concat([pd.read_csv(f) for f in sorted(d.glob('*_pred.csv'))], ignore_index=True)
    M = [json.load(open(f)) for f in sorted(d.glob('*_meta.json'))]
    P, M = complete_repeats(d, P, M)
    if not M:
        return (None, None), None, []
    if max_repeats is not None:
        P = P[P.repeat < max_repeats]
        M = [m for m in M if m['repeat'] < max_repeats]
    if P.empty:
        return (None, None), None, []
    tex_mae = np.array([tex_metrics(P[P.repeat == r])['tex_MAE'] for r in sorted(P.repeat.unique())])
    epochs = [m['best_epoch'] for m in M if m['best_epoch'] >= 0]
    sd = float(tex_mae.std(ddof=1)) if len(tex_mae) > 1 else 0.0
    return (float(tex_mae.mean()), sd), tex_mae, epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cv_root', nargs='?', default='output/results_cv')
    ap.add_argument('--cv_runs', nargs='+', default=None, help='explicit run-folder names under cv_root')
    ap.add_argument('--holdout_root', default='output/results_holdout')
    ap.add_argument('--holdout_runs', nargs='+', default=None)
    ap.add_argument('--tune_repeats', type=int, default=2,
                    help='repeats used by run/run_cv.sh tune_epochs (the "light CV" signal to check against)')
    ap.add_argument('--epoch_cv_warn', type=float, default=0.5,
                    help='flag a model if std(best_epoch)/mean(best_epoch) exceeds this')
    ap.add_argument('--repeat_swing_warn', type=float, default=0.15,
                    help='flag a model if the relative gap between the --tune_repeats-mean and the '
                         'all-repeats-mean texture MAE exceeds this fraction')
    a = ap.parse_args()

    cv_runs = load_cv_runs(a.cv_root, dirs=a.cv_runs)
    if not cv_runs:
        raise SystemExit(f'no CV runs (*_pred.csv + args.json) found under {a.cv_root}')
    holdout_runs = load_holdout_runs(a.holdout_root, dirs=a.holdout_runs) if Path(a.holdout_root).exists() else {}

    models = [m for m in ORDER if m in cv_runs] + [m for m in cv_runs if m not in ORDER]
    md = ['# Pre-confirmatory CV diagnostics (strategy doc §10.3)', '',
          '## 1-2. Repeat-to-repeat stability + best_epoch selection stability', '',
          f'| Model | texMAE (first {a.tune_repeats} reps) | texMAE (all reps) | relative swing | '
          'best_epoch mean ± SD (n folds) | CV of epoch |', '|---' * 6 + '|']
    warnings = []
    cv_full_mae = {}
    for m in models:
        d = cv_runs[m]['dir']
        (meanN, sdN), tex_mae_all, epochs = cv_tex_mae(d, max_repeats=None)
        if meanN is None:
            md.append(f'| {DISPLAY.get(m, m)} | no complete repeat yet | - | - | - | - |')
            continue
        cv_full_mae[m] = meanN
        (mean2, _), _, _ = cv_tex_mae(d, max_repeats=a.tune_repeats)
        ref = mean2 if mean2 is not None else meanN
        swing = abs(ref - meanN) / meanN if meanN else np.nan
        if epochs:
            ep = np.array(epochs, float)
            ep_mean = ep.mean(); ep_sd = ep.std(ddof=1) if len(ep) > 1 else 0.0
            ep_cv = ep_sd / ep_mean if ep_mean else np.nan
            ep_txt, cv_txt = f'{ep_mean:.1f} ± {ep_sd:.1f} (n={len(ep)})', f'{ep_cv:.2f}'
            if np.isfinite(ep_cv) and ep_cv > a.epoch_cv_warn:
                warnings.append(f'[WARN] {DISPLAY.get(m, m)}: best_epoch coefficient of variation {ep_cv:.2f} '
                                f'> {a.epoch_cv_warn} - unstable epoch selection (precedent: ANN, §8)')
        else:
            ep_txt, cv_txt = 'n/a (no-epoch model)', '-'
        mean2_txt = f'{mean2:.2f}' if mean2 is not None else 'n/a (< tune_repeats complete)'
        swing_txt = f'{swing * 100:.1f}%' if np.isfinite(swing) else 'n/a'
        if np.isfinite(swing) and swing > a.repeat_swing_warn:
            warnings.append(f'[WARN] {DISPLAY.get(m, m)}: {a.tune_repeats}-repeat vs {len(tex_mae_all)}-repeat '
                            f'texture MAE differs by {swing * 100:.1f}% (> {a.repeat_swing_warn * 100:.0f}%) - '
                            'the light CV signal would have been unreliable for this model/recipe')
        md.append(f'| {DISPLAY.get(m, m)} | {mean2_txt} | {meanN:.2f} ± {sdN:.2f} | {swing_txt} | {ep_txt} | {cv_txt} |')

    md += ['', "## 3. CV (Tier B/A' OOF, this locked-recipe run) vs. Role-2 holdout (Tier A, single full-pool model)", '']
    if holdout_runs:
        hold_mae = {}
        for m in models:
            if m in holdout_runs:
                t = holdout_runs[m]['P'].groupby('texture').agg(g=('gt', 'first'), p=('pred', 'mean'))
                hold_mae[m] = float(np.abs(t.p - t.g).mean())
        cv_rank = pd.Series(cv_full_mae).rank()
        hold_rank = pd.Series(hold_mae).rank()
        md += ["| Model | CV texMAE (Tier B/A') | Holdout texMAE (Tier A) | CV rank | Holdout rank |", '|---' * 5 + '|']
        for m in models:
            cvv = f'{cv_full_mae[m]:.2f}' if m in cv_full_mae else '-'
            hv = f'{hold_mae[m]:.2f}' if m in hold_mae else '-'
            cr = f'{cv_rank[m]:.0f}' if m in cv_rank.index else '-'
            hr = f'{hold_rank[m]:.0f}' if m in hold_rank.index else '-'
            md.append(f'| {DISPLAY.get(m, m)} | {cvv} | {hv} | {cr} | {hr} |')
        common = [m for m in models if m in cv_rank.index and m in hold_rank.index]
        if len(common) >= 3:
            rho = pd.Series({m: cv_rank[m] for m in common}).corr(
                pd.Series({m: hold_rank[m] for m in common}), method='spearman')
            md += ['', f'Spearman rank correlation (CV ranking vs. Holdout ranking) = {rho:.3f}']
            if rho < 0.5:
                warnings.append(f'[WARN] CV vs. Holdout rank correlation is low (rho={rho:.3f}) - the two '
                                'evaluations disagree substantially on model ordering (cf. §8 precedent: '
                                'Transformer ranked 1st in CV but 5th on the Tier A holdout)')
        else:
            md += ['', f'[skip] fewer than 3 models present in both CV and holdout (have {common})']
    else:
        md.append(f'[skip] no holdout runs found under {a.holdout_root} (run run/run_holdout.sh first)')

    md += ['', '## Verdict', '']
    if warnings:
        md += [f'**{len(warnings)} warning(s) - review before reading the Tier A confirmatory numbers '
              '(test/cv/tier_a_confirm.py):**', ''] + warnings
    else:
        md.append('No warnings raised by these checks. This does not guarantee the Tier A confirmatory '
                  'result will look good - only that nothing in this battery of sanity checks looks off.')

    text = '\n'.join(md)
    Path(a.cv_root, 'PRE_CONFIRM_DIAGNOSTICS.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
