"""
Stage 2 holdout evaluation (see claude/whc2027_apt_relabeling_strategy.md, PPT 261109_Real-Time_Roughness_
Prediction_in_VR.pptx slide 5): train ONE model ONCE on the full Tier B + Tier A' pool (98 textures, no
internal CV/validation split), then evaluate it ONCE on the Tier A holdout (8 native APT textures, never
seen in training). This produces the paper's headline "Tier 1" numbers (Texture MAE, CCC per model).

This is stage 2 of a two-stage protocol - stage 1 (test/cv_experiment.py's lambda_sweep, see run_cv.sh)
picks --lambda_ratio via a light 5-fold CV check on the proposed model only; that value is then reused
here, unchanged, for every model ("동일 레시피" - strategy doc §7). There is no validation split in this
stage, so none of cv_experiment.py's --select / --patience early-stopping machinery applies: PyTorch
models train for the full --epochs (default 300; --cosine anneals the LR to ~0 by the end) and the
FINAL epoch's weights are used, not a "best" checkpoint.

Usage (from repository root), typically via run_holdout.sh which loops over every model:
    python -m test.holdout_eval --model transformer --loss l1 --cosine --lr 3e-4 --aug --lambda_ratio 16
"""
import argparse, json, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataframe import build_tier_dataframe
from src.data.dataset import NormalizedSubset, dataset_to_numpy
from src.data.factory import MODEL_DATASET_TYPE
from src.model.factory import create_model
from src.trainer import prepare_batch_by_model, forward_by_model, is_torch_model
from src.model.feature.feature_extractor import GLCM_SLICE
from test.cv_experiment import (
    ColumnMinMax, TierWeighted, build_train_pool_dataframes, build_concat_base, base_features,
    set_seed, predict, metrics, make_criterion, dihedral_aug, lds_weights,
)


def build_tier_a_dataframe(conf, data_root=None):
    """Tier A (8 native APT textures, string ids 'A1'..'A8') - never used in build_train_pool_dataframes,
    loaded separately here as the single holdout evaluation set. Unlike Tier B / Tier A', ids are NOT cast
    to int (see claude/whc2027_apt_relabeling_strategy.md)."""
    tier_root = Path(data_root) if data_root else Path(conf.get('data_tier_root', 'data/trustworthy'))
    tier_dir = tier_root / 'Tier A'
    df = build_tier_dataframe(tier_dir, tier_dir / 'Tier A.csv', tier_name='A')
    df['image_id'] = df['texture_path'].map(lambda p: Path(p).stem)
    df['patch_idx'] = df['image_id'].map(lambda s: int(s.split('_')[-1]))
    return df.reset_index(drop=True)


def train_full(model, train_ds, args, device, weight_tr=None):
    """Train for the full --epochs with no validation split and no early stopping (see module docstring).
    Returns the model with its FINAL-epoch weights (there is no "best" checkpoint to pick here)."""
    crit = make_criterion(args.loss)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.cosine:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda e: (e + 1) / max(1, args.warmup) if e < args.warmup
            else 0.5 * (1 + math.cos(math.pi * (e - args.warmup) / max(1, args.epochs - args.warmup))))
    else:
        sched = None

    if weight_tr is None:
        weight_tr = np.ones(len(train_ds), dtype=np.float32)
    train_ds = TierWeighted(train_ds, weight_tr)

    if args.lds:
        y_tr = train_ds._y if hasattr(train_ds, '_y') else np.array([train_ds[i][-2].item() for i in range(len(train_ds))])
        w_lds = lds_weights(y_tr)
        loader = DataLoader(train_ds, batch_size=args.batch_size,
                            sampler=WeightedRandomSampler(torch.tensor(w_lds, dtype=torch.double), len(w_lds), replacement=True))
    else:
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda')
    history = []
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); run_loss = 0.0
        for batch in loader:
            *core, w = batch
            inputs, y = prepare_batch_by_model(core, model, device)
            w = w.to(device)
            if args.aug and isinstance(inputs, tuple) and len(inputs) == 3 and inputs[0].dim() == 4:
                inputs = dihedral_aug(*inputs)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
                pred = forward_by_model(model, inputs)
            pred = pred.float().view_as(y)
            loss = crit(pred, y, w)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run_loss += loss.item() * len(y)
        if sched: sched.step()
        history.append(dict(epoch=epoch, train_loss=run_loss / len(train_ds), sec=time.time() - t0))
        if args.verbose:
            print(f'  ep {epoch:3d} loss {history[-1]["train_loss"]:.4f}', flush=True)
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='transformer')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--weight_decay', type=float, default=None)
    ap.add_argument('--loss', choices=['mse', 'huber', 'l1', 'ccc', 'mse_ccc'], default='l1')
    ap.add_argument('--lds', action='store_true')
    ap.add_argument('--aug', action='store_true', help='image models only (transformer, cnn_1d_simple)')
    ap.add_argument('--cosine', action='store_true')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--amp', action='store_true', help='mixed precision on CUDA')
    ap.add_argument('--lambda_ratio', type=float, default=16.0,
                    help="lambda_A':lambda_B tier loss-weight ratio - same value for every model (see "
                         "run_cv.sh's lambda_sweep stage, which picks this on the proposed model only)")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--out', default='output/results_holdout', help='output/ is git-ignored')
    ap.add_argument('--data_root', default=None, help='override data_tier_root, e.g. data/trustworthy')
    ap.add_argument('--feature_cache_root', default=None)
    ap.add_argument('--feature_mode', choices=['legacy', 'original'], default=None)
    ap.add_argument('--print_tag', action='store_true')
    ap.add_argument('--max_patches', type=int, default=None, help='smoke test only')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    conf = yaml.safe_load(open('config.yaml', encoding='utf-8'))
    conf['model'] = args.model
    if args.feature_cache_root:
        conf['feature_cache_root'] = args.feature_cache_root
    if args.feature_mode:
        conf['feature_mode'] = args.feature_mode
    is_feature_model = MODEL_DATASET_TYPE.get(args.model) == 'feature'
    args.feature_mode = conf.get('feature_mode', 'original') if is_feature_model else None
    if is_feature_model:
        conf['feature_mode'] = args.feature_mode
    args.epochs = args.epochs or int(conf.get('epochs', 300))
    args.batch_size = args.batch_size or int(conf.get('batch_size', 56))
    args.lr = args.lr or float(conf.get('learning_rate', 1e-3))
    args.weight_decay = args.weight_decay if args.weight_decay is not None else float(conf.get('weight_decay', 1e-4))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tag = args.tag or (f'{args.model}_holdout_{args.loss}' + ('_lds' if args.lds else '') + ('_aug' if args.aug else '') +
                       ('_cos' if args.cosine else '') +
                       (f'_feat{args.feature_mode}' if args.feature_mode not in (None, 'legacy') else '') +
                       (f'_lr{args.lambda_ratio:g}' if args.lambda_ratio != 16.0 else ''))
    if args.print_tag:
        print(tag); return
    out = Path(args.out) / tag; out.mkdir(parents=True, exist_ok=True)
    if (out / 'pred.csv').exists():
        print(f'[skip] {tag} already done'); return
    json.dump(vars(args), open(out / 'args.json', 'w'), indent=1)

    set_seed(args.seed)
    random.seed(args.seed)

    dfs = build_train_pool_dataframes(conf, args.data_root)
    base, all_df, input_dim, keep = build_concat_base(conf, dfs, device, args.max_patches)
    all_df['weight'] = np.where(all_df['tier'] == "A'", args.lambda_ratio, 1.0)  # L = lambda_B*L_B + lambda_A'*L_A'
    all_df.groupby('texture').agg(gt=('roughness', 'first'), tier=('tier', 'first')).to_csv(out / 'train_labels.csv')

    df_a = build_tier_a_dataframe(conf, args.data_root)
    base_a, all_df_a, input_dim_a, keep_a = build_concat_base(conf, {'A': df_a}, device, args.max_patches)

    print(f'[data] train pool: {all_df.texture.nunique()} textures / {len(keep)} patches | '
          f'holdout (Tier A): {all_df_a.texture.nunique()} textures / {len(keep_a)} patches | '
          f'device={device} | tag={tag} | lambda_ratio={args.lambda_ratio:g}')

    y_tr = all_df.roughness.values[keep]
    w_tr = all_df.weight.values[keep]
    y_min, y_max = float(y_tr.min()), float(y_tr.max())   # normalisation from the train pool only

    ds_tr = NormalizedSubset(base, keep, y_min, y_max)
    ds_tr._y = (y_tr - y_min) / (y_max - y_min + 1e-8)
    ds_te = NormalizedSubset(base_a, keep_a, y_min, y_max)

    if args.feature_mode == 'original':                  # GLCM min-max fitted on the train pool only
        cols = GLCM_SLICE['original']
        F_tr = base_features(base)[keep][:, cols]
        lo, hi = F_tr.min(0), F_tr.max(0)
        ds_tr, ds_te = ColumnMinMax(ds_tr, cols, lo, hi), ColumnMinMax(ds_te, cols, lo, hi)
        ds_tr._y = (y_tr - y_min) / (y_max - y_min + 1e-8)

    t0 = time.time()
    model = create_model(conf, input_dim=input_dim, device=device)
    if is_torch_model(model):
        model, hist = train_full(model, ds_tr, args, device, weight_tr=w_tr)
        pd.DataFrame(hist).to_csv(out / 'train_history.csv', index=False)
        p, g = predict(model, ds_te, device, args.batch_size, args.amp)
        ptr, gtr = predict(model, ds_tr, device, args.batch_size, args.amp)
    else:                                                  # sklearn models (lr, svr)
        X, yy = dataset_to_numpy(ds_tr); model.fit(X, yy, sample_weight=w_tr)
        Xt, g = dataset_to_numpy(ds_te); p = np.asarray(model.predict(Xt)).ravel()
        ptr, gtr = np.asarray(model.predict(X)).ravel(), yy
    den = lambda v: v * (y_max - y_min + 1e-8) + y_min
    p, g, ptr, gtr = den(p), den(g), den(ptr), den(gtr)

    pred_df = pd.DataFrame({'image_id': all_df_a.image_id.values[keep_a], 'texture': all_df_a.texture.values[keep_a],
                            'tier': all_df_a.tier.values[keep_a], 'gt': g, 'pred': p})
    pred_df.to_csv(out / 'pred.csv', index=False)
    lsfg = ([round(float(v), 4) for v in model.get_lsfg_weights().cpu()]
            if hasattr(model, 'get_lsfg_weights') else None)
    meta = dict(minutes=(time.time() - t0) / 60, lsfg_weights=lsfg,
                n_train_tex=int(all_df.texture.nunique()), n_test_tex=int(all_df_a.texture.nunique()),
                test=metrics(p, g, all_df_a.texture.values[keep_a]),
                train=metrics(ptr, gtr, all_df.texture.values[keep]))
    json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
    print(f'[{tag}] test texMAE={meta["test"]["tex_mae"]:.2f} test CCC={meta["test"]["tex_ccc"]:.3f} | '
          f'train texMAE={meta["train"]["tex_mae"]:.2f} slope={meta["train"]["tex_slope"]:.2f} '
          f'({meta["minutes"]:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
