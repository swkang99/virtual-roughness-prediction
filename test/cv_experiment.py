"""
Texture-level repeated K-fold cross-validation for roughness prediction.

Why texture-level (grouped) folds?
    The 64 patches of one texture share the same label and look alike.
    If patches of one texture appear in both train and test, the model can
    memorise texture identity and the score is inflated (data leakage).
    Every fold therefore holds out *whole textures*.

Protocol (default: --protocol cv --k 5 --repeats 10)
    for repeat r in 0..R-1 (seed = base_seed + r):
        StratifiedGroupKFold(K) over the 100 textures, stratified by label quintile
        for each outer fold k:
            test  = textures of fold k                       (100/K textures)
            val   = --n_val textures from the rest (stratified, inner split)
            train = remaining textures                       (K=5, n_val=10 -> 70 textures, same as paper)
            train with early stopping on val, evaluate best checkpoint on test
    -> out-of-fold (OOF) predictions for all 100 textures per repeat
    -> report mean ± SD over repeats (aggregate with test/cv_aggregate.py)

--protocol fixed : the original train/val/test split, repeated with R seeds
                   (mean ± SD over seeds, comparable with the current paper table)

Options to address under-fitting / range compression (all OFF by default = paper setting):
    --loss {mse,huber,ccc,mse_ccc}   CCC = concordance correlation coefficient loss
    --lds                            label-distribution-smoothing re-weighting
    --select {val_rmse,val_tex_mae,val_ccc}
    --aug                            exact dihedral (flip / 90° rotation) augmentation
    --cosine --warmup N              learning-rate schedule
    --patience N                     early stopping

Usage (from repository root):
    python -m test.cv_experiment --model transformer --protocol cv --k 5 --repeats 10 --patience 40 --amp
    python -m test.cv_experiment --model transformer --protocol fixed --repeats 10 --patience 40 --amp
"""
import argparse, copy, json, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from src.data.dataframe import build_dataframe_from_file
from src.data.dataset import NormalizedSubset, dataset_to_numpy
from src.data.factory import build_base_dataset, MODEL_DATASET_TYPE
from src.model.factory import create_model
from src.trainer import prepare_batch_by_model, forward_by_model, is_torch_model


# ----------------------------------------------------------------------------- utils
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def texture_id(path):
    return int(Path(path).stem.split('_')[0])


def build_all_dataframes(conf, data_root=None):
    """Build per-split dataframes exactly as Trainer.build_split_dataframe does."""
    dfs = {}
    for split in ['train', 'val', 'test']:
        base = Path(data_root) / split if data_root else Path(conf[f'data_{split}_path'])
        df = build_dataframe_from_file(
            texture_path=base / conf['data_patch_path'],
            label_path=base / f'{split}.csv',
            header=0,
        )
        df['orig_split'] = split
        df['texture'] = df['texture_path'].map(texture_id)
        df['image_id'] = df['texture_path'].map(lambda p: Path(p).stem)
        df['patch_idx'] = df['image_id'].map(lambda s: int(s.split('_')[-1]))
        dfs[split] = df.reset_index(drop=True)
    return dfs


def build_concat_base(conf, dfs, device, max_patches=None):
    """
    Build one base dataset over all 100 textures.

    Feature-based models: the feature cache is resolved *per original split*
    (data/features/{train,val,test}/features.npz). Building a cache from a mixed
    dataframe would make src.data.factory delete the existing cache, so we never do that.
    """
    parts, frames = [], []
    for split in ['train', 'val', 'test']:
        df = dfs[split]
        base, _, input_dim = build_base_dataset(conf, df, device)
        parts.append(base); frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    base = ConcatDataset(parts)
    keep = np.arange(len(all_df))
    if max_patches is not None:                       # only for smoke tests
        keep = np.where(all_df['patch_idx'].values < max_patches)[0]
    return base, all_df, input_dim, keep


# ----------------------------------------------------------------------------- folds
def make_folds(all_df, protocol, k, n_val, seed):
    """Return list of (train_tex, val_tex, test_tex) texture-id arrays."""
    tex = all_df.groupby('texture').agg(y=('roughness', 'first'), split=('orig_split', 'first')).reset_index()
    if protocol == 'fixed':
        return [(tex[tex.split == 'train'].texture.values,
                 tex[tex.split == 'val'].texture.values,
                 tex[tex.split == 'test'].texture.values)]

    bins = pd.qcut(tex.y, 5, labels=False).values
    outer = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for fold_i, (rest_idx, test_idx) in enumerate(outer.split(tex, bins, groups=tex.texture)):
        rest = tex.iloc[rest_idx].reset_index(drop=True)
        n_inner = max(2, int(round(len(rest) / n_val)))
        inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed * 100 + fold_i)
        rbins = pd.qcut(rest.y, 5, labels=False).values
        tr_idx, va_idx = next(inner.split(rest, rbins, groups=rest.texture))
        folds.append((rest.texture.values[tr_idx], rest.texture.values[va_idx], tex.texture.values[test_idx]))
    return folds


# ----------------------------------------------------------------------------- losses
def ccc_loss(pred, y):
    """1 - concordance correlation coefficient (penalises shrinkage of the output range)."""
    pm, ym = pred.mean(), y.mean()
    pv, yv = pred.var(unbiased=False), y.var(unbiased=False)
    cov = ((pred - pm) * (y - ym)).mean()
    return 1 - 2 * cov / (pv + yv + (pm - ym) ** 2 + 1e-8)


def make_criterion(name):
    mse, huber = nn.MSELoss(reduction='none'), nn.HuberLoss(reduction='none', delta=0.1)

    def crit(pred, y, w):
        if name in ('mse', 'mse_ccc'):
            base = (mse(pred, y).view(len(y), -1).mean(1) * w).sum() / w.sum()
        elif name == 'huber':
            base = (huber(pred, y).view(len(y), -1).mean(1) * w).sum() / w.sum()
        elif name == 'ccc':
            return ccc_loss(pred.view(-1), y.view(-1))
        if name == 'mse_ccc':
            return base + 0.5 * ccc_loss(pred.view(-1), y.view(-1))
        return base
    return crit


def lds_weights(y_norm, bins=50, sigma=2.0):
    """Label Distribution Smoothing (Yang et al., ICML 2021): w ∝ 1/sqrt(smoothed density)."""
    hist, edges = np.histogram(y_norm, bins=bins, range=(0, 1))
    k = np.arange(-3 * int(sigma), 3 * int(sigma) + 1)
    kern = np.exp(-k ** 2 / (2 * sigma ** 2)); kern /= kern.sum()
    smooth = np.convolve(hist, kern, mode='same')
    idx = np.clip(np.digitize(y_norm, edges[1:-1]), 0, bins - 1)
    w = 1 / np.sqrt(np.maximum(smooth[idx], 1e-6))
    return w / w.mean()


# ----------------------------------------------------------------------------- augmentation
def dihedral_aug(texture, height, normal):
    """
    Exact flip / transpose augmentation for (texture, height, normal) batches.
    Normal map channels are encoded as (n+1)/2, n=(nx,ny,nz) with nx∝-d/dx, ny∝-d/dy.
        horizontal flip -> nx := -nx   (channel0 -> 1 - channel0)
        vertical   flip -> ny := -ny   (channel1 -> 1 - channel1)
        transpose       -> swap nx, ny
    Their combinations generate all 8 dihedral transforms (incl. 90° rotations).
    """
    if random.random() < 0.5:
        texture, height, normal = texture.flip(-1), height.flip(-1), normal.flip(-1)
        normal = torch.cat([1 - normal[:, 0:1], normal[:, 1:]], 1)
    if random.random() < 0.5:
        texture, height, normal = texture.flip(-2), height.flip(-2), normal.flip(-2)
        normal = torch.cat([normal[:, 0:1], 1 - normal[:, 1:2], normal[:, 2:]], 1)
    if random.random() < 0.5:
        texture, height, normal = [t.transpose(-1, -2) for t in (texture, height, normal)]
        normal = normal[:, [1, 0, 2]]
    return texture, height, normal


# ----------------------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(model, dataset, device, batch_size, amp):
    model.eval()
    P, Y = [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        inputs, y = prepare_batch_by_model(batch, model, device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
            p = forward_by_model(model, inputs)
        P.append(p.float().view(len(y), -1).cpu().numpy()); Y.append(y.view(len(y), -1).cpu().numpy())
    return np.concatenate(P).ravel(), np.concatenate(Y).ravel()


def metrics(pred, gt, tex):
    d = pd.DataFrame({'p': pred, 'g': gt, 't': tex})
    t = d.groupby('t').mean()
    slope = np.polyfit(t.g, t.p, 1)[0] if len(t) > 1 else np.nan
    r = np.corrcoef(t.g, t.p)[0, 1] if len(t) > 2 else np.nan
    ccc = 2 * np.cov(t.g, t.p, bias=True)[0, 1] / (t.g.var(ddof=0) + t.p.var(ddof=0) + (t.g.mean() - t.p.mean()) ** 2)
    return dict(patch_mae=float(np.abs(d.p - d.g).mean()),
                patch_rmse=float(np.sqrt(((d.p - d.g) ** 2).mean())),
                tex_mae=float(np.abs(t.p - t.g).mean()), tex_slope=float(slope),
                tex_r=float(r), tex_ccc=float(ccc))


def train_torch(model, train_ds, val_ds, val_tex, args, device, y_min, y_max):
    crit = make_criterion(args.loss)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs
    if args.cosine:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda e: (e + 1) / max(1, args.warmup) if e < args.warmup
            else 0.5 * (1 + math.cos(math.pi * (e - args.warmup) / max(1, total - args.warmup))))
    else:
        sched = None

    if args.lds:
        y_tr = np.array([train_ds[i][-1].item() for i in range(len(train_ds))]) if not hasattr(train_ds, '_y') else train_ds._y
        w = lds_weights(y_tr)
        loader = DataLoader(train_ds, batch_size=args.batch_size,
                            sampler=WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(w), replacement=True))
    else:
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda')
    best, best_state, best_epoch, bad, history = None, None, -1, 0, []
    for epoch in range(total):
        model.train(); t0 = time.time(); run_loss = 0.0
        for batch in loader:
            inputs, y = prepare_batch_by_model(batch, model, device)
            if args.aug and len(inputs) == 3:
                inputs = dihedral_aug(*inputs)
            w = torch.ones(len(y), device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
                pred = forward_by_model(model, inputs)
            pred = pred.float().view_as(y)
            loss = crit(pred, y, w)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run_loss += loss.item() * len(y)
        if sched: sched.step()

        pv, gv = predict(model, val_ds, device, args.batch_size, args.amp)
        pv = pv * (y_max - y_min + 1e-8) + y_min; gv = gv * (y_max - y_min + 1e-8) + y_min
        m = metrics(pv, gv, val_tex)
        score = {'val_rmse': m['patch_rmse'], 'val_tex_mae': m['tex_mae'], 'val_ccc': -m['tex_ccc']}[args.select]
        history.append(dict(epoch=epoch, train_loss=run_loss / len(train_ds), sec=time.time() - t0, **{f'val_{k}': v for k, v in m.items()}))
        if best is None or score < best:
            best, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if args.verbose:
            print(f'  ep {epoch:3d} loss {history[-1]["train_loss"]:.4f} val_rmse {m["patch_rmse"]:.2f} '
                  f'val_texMAE {m["tex_mae"]:.2f} slope {m["tex_slope"]:.2f} {"*" if bad == 0 else ""}', flush=True)
        if args.patience and bad >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='transformer')
    ap.add_argument('--protocol', choices=['cv', 'fixed'], default='cv')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--n_val', type=int, default=10, help='validation textures inside each outer fold')
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--start_repeat', type=int, default=0, help='resume / split work across GPUs')
    ap.add_argument('--base_seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--weight_decay', type=float, default=None)
    ap.add_argument('--patience', type=int, default=0, help='0 = no early stopping (paper setting)')
    ap.add_argument('--loss', choices=['mse', 'huber', 'ccc', 'mse_ccc'], default='mse')
    ap.add_argument('--lds', action='store_true')
    ap.add_argument('--select', choices=['val_rmse', 'val_tex_mae', 'val_ccc'], default='val_rmse')
    ap.add_argument('--aug', action='store_true')
    ap.add_argument('--cosine', action='store_true')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--amp', action='store_true', help='mixed precision on CUDA')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--out', default='results_cv')
    ap.add_argument('--data_root', default=None, help='e.g. data  (if config paths differ)')
    ap.add_argument('--max_patches', type=int, default=None, help='smoke test only')
    ap.add_argument('--max_folds', type=int, default=None, help='smoke test only')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    conf = yaml.safe_load(open('config.yaml', encoding='utf-8'))
    conf['model'] = args.model
    args.epochs = args.epochs or int(conf.get('epochs', 300))
    args.batch_size = args.batch_size or int(conf.get('batch_size', 56))
    args.lr = args.lr or float(conf.get('learning_rate', 1e-3))
    args.weight_decay = args.weight_decay if args.weight_decay is not None else float(conf.get('weight_decay', 1e-4))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tag = args.tag or (f'{args.model}_{args.protocol}' + (f'{args.k}' if args.protocol == 'cv' else '') +
                       f'_{args.loss}' + ('_lds' if args.lds else '') + ('_aug' if args.aug else '') +
                       ('_cos' if args.cosine else '') + f'_{args.select}')
    out = Path(args.out) / tag; out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / 'args.json', 'w'), indent=1)

    dfs = build_all_dataframes(conf, args.data_root)
    base, all_df, input_dim, keep = build_concat_base(conf, dfs, device, args.max_patches)
    all_df.groupby('texture').roughness.first().rename('gt').to_csv(out / 'labels.csv')
    print(f'[data] {all_df.texture.nunique()} textures, {len(keep)} patches used | device={device} | tag={tag}')

    for r in range(args.start_repeat, args.repeats):
        seed = args.base_seed + r
        folds = make_folds(all_df, args.protocol, args.k, args.n_val, seed)
        for f, (tr_t, va_t, te_t) in enumerate(folds[:args.max_folds]):
            run_name = f'rep{r:02d}_fold{f}'
            if (out / f'{run_name}_pred.csv').exists():
                print(f'[skip] {run_name} already done'); continue
            set_seed(seed * 100 + f)
            sel = lambda ts: keep[np.isin(all_df.texture.values[keep], ts)]
            i_tr, i_va, i_te = sel(tr_t), sel(va_t), sel(te_t)
            y_tr = all_df.roughness.values[i_tr]
            y_min, y_max = float(y_tr.min()), float(y_tr.max())       # normalisation from train fold only
            ds_tr = NormalizedSubset(base, i_tr, y_min, y_max)
            ds_tr._y = (y_tr - y_min) / (y_max - y_min + 1e-8)
            ds_va = NormalizedSubset(base, i_va, y_min, y_max)
            ds_te = NormalizedSubset(base, i_te, y_min, y_max)

            t0 = time.time()
            model = create_model(conf, input_dim=input_dim, device=device)
            if is_torch_model(model):
                model, best_epoch, hist = train_torch(model, ds_tr, ds_va, all_df.texture.values[i_va],
                                                      args, device, y_min, y_max)
                pd.DataFrame(hist).to_csv(out / f'{run_name}_history.csv', index=False)
                p, g = predict(model, ds_te, device, args.batch_size, args.amp)
                ptr, gtr = predict(model, ds_tr, device, args.batch_size, args.amp)
            else:                                                     # sklearn models (lr, svr)
                X, yy = dataset_to_numpy(ds_tr); model.fit(X, yy); best_epoch = -1
                Xt, g = dataset_to_numpy(ds_te); p = np.asarray(model.predict(Xt)).ravel()
                ptr = np.asarray(model.predict(X)).ravel(); gtr = yy
            den = lambda v: v * (y_max - y_min + 1e-8) + y_min
            p, g, ptr, gtr = den(p), den(g), den(ptr), den(gtr)

            pred_df = pd.DataFrame({'image_id': all_df.image_id.values[i_te], 'texture': all_df.texture.values[i_te],
                                    'orig_split': all_df.orig_split.values[i_te], 'gt': g, 'pred': p,
                                    'repeat': r, 'fold': f})
            pred_df.to_csv(out / f'{run_name}_pred.csv', index=False)
            meta = dict(repeat=r, fold=f, seed=seed, best_epoch=int(best_epoch), minutes=(time.time() - t0) / 60,
                        n_train_tex=len(tr_t), n_val_tex=len(va_t), n_test_tex=len(te_t),
                        test=metrics(p, g, all_df.texture.values[i_te]),
                        train=metrics(ptr, gtr, all_df.texture.values[i_tr]),
                        test_textures=[int(t) for t in te_t], val_textures=[int(t) for t in va_t])
            json.dump(meta, open(out / f'{run_name}_meta.json', 'w'), indent=1)
            print(f'[{run_name}] best_epoch={best_epoch} test texMAE={meta["test"]["tex_mae"]:.2f} '
                  f'patchMAE={meta["test"]["patch_mae"]:.2f} slope={meta["test"]["tex_slope"]:.2f} | '
                  f'train texMAE={meta["train"]["tex_mae"]:.2f} slope={meta["train"]["tex_slope"]:.2f} '
                  f'({meta["minutes"]:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
