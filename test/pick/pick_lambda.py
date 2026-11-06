"""
Auto-picks the lambda_ratio from run/run_cv.sh's lambda_sweep stage (PPT 261109_Real-Time_Roughness_
Prediction_in_VR.pptx slides 4/8/9; see claude/whc2027_apt_relabeling_strategy.md). lambda_sweep runs
the proposed model only, 5-fold CV x 2 repeats, for each --lambda_ratio in LAMBDA_GRID ({8, 16, 27} by
default - the 90% CI around the analytically-derived point estimate 16.3), then test.cv.cv_compare
aggregates all of them into one compare_cv_lambda_sweep/per_repeat.csv.

Selection rule: the lambda with the lowest mean texture-level MAE (tex_MAE) over the shared CV repeats
- the same "lower is better" criterion test.cv.cv_compare itself uses to bold the best model in its
comparison tables, and the same quantity --select val_tex_mae optimizes for during training. This is a
light 1-D check of whether the analytically-derived ratio sits in a flat region or a clear local
optimum exists (strategy doc), not a second round of model comparison.

    python -m test.pick.pick_lambda output/results_cv/compare_cv_lambda_sweep/per_repeat.csv \
        --pairs 8:transformer:tag8 16:transformer:tag16 27:transformer:tag27

Prints a markdown table + the chosen value (run/run_cv.sh pipes this to <out>/LAMBDA_SELECTED.md via tee,
same pattern as test/pick/pick_epochs.py + EPOCHS_SELECTED.md) and writes <out>/lambda.txt itself (just the
chosen lambda, one line - for `LAMBDA=$(cat output/results_cv/lambda.txt)` in a wrapper script). <out>
defaults to the per_repeat.csv's own parent directory's parent (i.e. output/results_cv, not
.../compare_cv_lambda_sweep), so lambda.txt sits next to epochs.csv (test/pick/pick_epochs.py's output) for
run/run_cv.sh's tune_epochs stage and run/run_holdout.sh to both pick up directly.
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('per_repeat_csv', help='compare_cv_lambda_sweep/per_repeat.csv from test.cv.cv_compare')
    ap.add_argument('--pairs', nargs='+', required=True,
                     help='lambda:model_column_value entries, e.g. 8:transformer:tag8 (the model column may '
                          'itself contain ":", so only the FIRST ":" splits lambda from the rest)')
    ap.add_argument('--out', default=None, help='output dir (default: per_repeat_csv\'s grandparent dir)')
    a = ap.parse_args()

    df = pd.read_csv(a.per_repeat_csv)
    out = Path(a.out) if a.out else Path(a.per_repeat_csv).resolve().parent.parent
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for pair in a.pairs:
        lam, model_key = pair.split(':', 1)
        sub = df[df.model == model_key]
        if sub.empty:
            print(f'[warn] no rows for lambda={lam} (model column {model_key!r}) in {a.per_repeat_csv}; skipping')
            continue
        rows.append(dict(lam=float(lam), lam_str=lam, mean_tex_MAE=sub.tex_MAE.mean(),
                          sd_tex_MAE=sub.tex_MAE.std(ddof=1) if len(sub) > 1 else 0.0, n_repeats=len(sub)))

    if not rows:
        raise SystemExit('no lambda produced a usable tex_MAE - nothing written')

    res = pd.DataFrame(rows).sort_values('mean_tex_MAE').reset_index(drop=True)
    best = res.iloc[0]

    md = ['## Lambda selection (lowest mean texture MAE over the lambda_sweep CV repeats)', '',
          '| lambda | mean tex_MAE | SD | n repeats |', '|---|---|---|---|']
    for _, r in res.iterrows():
        mark = ' **<- selected**' if r.lam_str == best.lam_str else ''
        md.append(f"| {r.lam_str} | {r.mean_tex_MAE:.3f} | {r.sd_tex_MAE:.3f} | {int(r.n_repeats)}{mark} |")
    md.append('')
    md.append(f'Selected lambda_ratio = **{best.lam_str}** (written to {out / "lambda.txt"}).')
    print('\n'.join(md))

    # Only lambda.txt is written here; the markdown above is captured by the caller (run/run_cv.sh pipes
    # stdout to $OUT/LAMBDA_SELECTED.md via tee, same pattern as test/pick/pick_epochs.py + EPOCHS_SELECTED.md) -
    # writing that file here too would race with tee's own write to the same path.
    (out / 'lambda.txt').write_text(f'{best.lam_str}\n', encoding='utf-8')


if __name__ == '__main__':
    main()
