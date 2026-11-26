"""
Deployment-model extractor for the "role 2" checkpoints test/holdout/holdout_eval.py saves
(strategy doc §18.1 role 2: the single full-pool model trained per architecture on Tier B + Tier A',
evaluated once on Tier A, and the one actually used for User Study #1-1/#2 - see run/run_holdout.sh).

holdout_eval.py saves, for every model it trains, one of:
    output/results_holdout/<tag>/model.pt    (torch models: ann, cnn_1d_scirep, cnn_1d_wassem,
                                               cnn_1d_simple, transformer)
    output/results_holdout/<tag>/model.pkl   (sklearn models: lr, svr - no state_dict)
alongside everything needed to reconstruct the exact trained model and reproduce its input/output
scaling at inference time: `model` (architecture name), `input_dim`, `feature_mode`, `y_min`/`y_max`
(train-pool target normalization - REQUIRED to turn a raw model output back into an APT-scale
roughness value), and for feature_mode='original' models, `glcm_cols`/`glcm_lo`/`glcm_hi` (the GLCM
feature min-max fit needed to reproduce identical input scaling at inference - see
src/model/feature/feature_extractor.py).

This script was rewritten 2026-11-26 to match that checkpoint format (the old version hardcoded a
pre-Tier checkpoint layout under experiments/checkpoints/..., stale train_y_min/train_y_max
constants, and only supported cnn_1d_simple/transformer - see git history for the old file).

What this does, per run found under --run_root:
    - torch models  -> rebuilt via src.model.factory.create_model() + state_dict, then torch.onnx
                       export (dynamic batch dim), plus a `<model>_meta.json` deployment-metadata file.
    - sklearn models (lr, svr) -> ONNX export needs skl2onnx, which this repo does not depend on, so
                       these are copied through as `<model>.pkl` (already a complete, loadable
                       estimator) with the same `<model>_meta.json` alongside, instead.

Usage (from repository root):
    # scan output/results_holdout/ and export every run that has a checkpoint
    python -m src.model.export

    # only specific architectures, explicit run_root/out/opset
    python -m src.model.export --models transformer,cnn_1d_wassem \
        --run_root output/results_holdout --out output/exports --opset 15
"""
import argparse
import json
import shutil
from pathlib import Path

import torch

from src.model.factory import create_model
from src.trainer import is_torch_model

# "maps" models take a (texture_image, height_map, normal_map) triplet of 256x256 patches (see
# src/data/factory.py MODEL_DATASET_TYPE, src/data/dataset.py PatchDataset); every other model in the
# 7-model roster takes one flat feature vector (src/data/dataset.py PatchFeatureDataset /
# src/model/feature/feature_extractor.py). gated_mlp / gated_mlp_v2 are not part of the roster
# run/run_holdout.sh trains, so they are intentionally left out here.
MAPS_MODELS = {'transformer', 'cnn_1d_simple'}
PATCH_SIZE = 256


def _strip_prefixes(state_dict):
    """Undo nn.DataParallel's 'module.' / torch.compile's '_orig_mod.' key prefixes, if present."""
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        new_key = new_key.replace('_orig_mod.', '') if new_key.startswith('_orig_mod.') else new_key
        cleaned[new_key] = value
    return cleaned


def make_dummy_inputs(model_name: str, input_dim, device: torch.device):
    if model_name in MAPS_MODELS:
        texture_image = torch.randn(1, 1, PATCH_SIZE, PATCH_SIZE, dtype=torch.float32, device=device)
        height_map = torch.randn(1, 1, PATCH_SIZE, PATCH_SIZE, dtype=torch.float32, device=device)
        normal_map = torch.randn(1, 3, PATCH_SIZE, PATCH_SIZE, dtype=torch.float32, device=device)
        return (texture_image, height_map, normal_map), ['texture_image', 'height_map', 'normal_map']
    if input_dim is None:
        raise ValueError(f"{model_name}: checkpoint has no input_dim recorded - can't build a dummy "
                         f"feature-vector input for ONNX export.")
    x = torch.randn(1, int(input_dim), dtype=torch.float32, device=device)
    return (x,), ['features']


def _build_meta(ckpt: dict, input_names, opset=None, onnx_path=None, pkl_path=None):
    """Deployment metadata: the fields a downstream extractor needs beyond the weights themselves -
    see module docstring. Mirrors (and is sourced directly from) holdout_eval.py's own deploy_meta."""
    meta = dict(
        model=ckpt['model'], input_dim=ckpt.get('input_dim'), feature_mode=ckpt.get('feature_mode'),
        y_min=ckpt.get('y_min'), y_max=ckpt.get('y_max'),
        glcm_cols=ckpt.get('glcm_cols'), glcm_lo=ckpt.get('glcm_lo'), glcm_hi=ckpt.get('glcm_hi'),
        input_names=input_names,
        output_names=['roughness'],
        output_description="raw model output is on the fold's [0,1]-normalized target scale - "
                           "denormalize with: roughness = raw * (y_max - y_min) + y_min",
    )
    if ckpt['model'] in MAPS_MODELS:
        meta['input_shapes'] = {name: [1, (3 if name == 'normal_map' else 1), PATCH_SIZE, PATCH_SIZE]
                               for name in input_names}
    else:
        meta['input_shapes'] = {'features': [1, ckpt.get('input_dim')]}
    if opset is not None:
        meta['onnx_opset'] = opset
    if onnx_path is not None:
        meta['onnx_file'] = onnx_path.name
    if pkl_path is not None:
        meta['pkl_file'] = pkl_path.name
    return meta


def _write_meta(meta_path: Path, meta: dict):
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _export_torch(run_dir: Path, pt_path: Path, out_root: Path, opset: int):
    device = torch.device('cpu')
    # weights_only=False: this checkpoint is a plain dict of our own deploy_meta (str/int/float/list/
    # None) + a tensor state_dict, not just tensors - torch >= 2.6 defaults weights_only=True, which can
    # refuse to unpickle that mix depending on version. We wrote this file ourselves, so it's trusted.
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    model_name = ckpt['model']
    out_dir = out_root / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = create_model({'model': model_name}, input_dim=ckpt.get('input_dim'), device=device)
    if not is_torch_model(model):
        raise TypeError(f"{run_dir.name}: checkpoint says model={model_name!r} but create_model() "
                        f"returned a non-torch object ({type(model)}) - registry/checkpoint mismatch.")
    model.load_state_dict(_strip_prefixes(ckpt['model_state_dict']), strict=True)
    model.to(device).eval()

    inputs, input_names = make_dummy_inputs(model_name, ckpt.get('input_dim'), device)
    with torch.no_grad():
        output = model(*inputs)
        if isinstance(output, (tuple, list)):
            raise ValueError(f"{run_dir.name}: ONNX export expects a single tensor output, "
                             f"got {type(output)}.")
        print(f'[{run_dir.name}] PyTorch forward OK. Output shape: {tuple(output.shape)}')

    onnx_path = out_dir / f'{model_name}.onnx'
    dynamic_axes = {name: {0: 'batch_size'} for name in input_names}
    dynamic_axes['roughness'] = {0: 'batch_size'}
    torch.onnx.export(
        model, inputs, str(onnx_path),
        export_params=True, opset_version=opset, do_constant_folding=True,
        input_names=input_names, output_names=['roughness'], dynamic_axes=dynamic_axes,
    )

    meta = _build_meta(ckpt, input_names, opset=opset, onnx_path=onnx_path)
    _write_meta(out_dir / f'{model_name}_meta.json', meta)
    print(f'[{run_dir.name}] exported ONNX -> {onnx_path}')
    return onnx_path


def _export_sklearn(run_dir: Path, pkl_path: Path, out_root: Path):
    import joblib
    ckpt = joblib.load(pkl_path)
    model_name = ckpt['model']
    out_dir = out_root / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ONNX export needs skl2onnx (not a repo dependency) - pass the fitted estimator through as-is
    # instead; it is already a complete, loadable sklearn object (see holdout_eval.py's 'estimator' key).
    out_pkl = out_dir / f'{model_name}.pkl'
    shutil.copyfile(pkl_path, out_pkl)

    meta = _build_meta(ckpt, input_names=['features'], pkl_path=out_pkl)
    _write_meta(out_dir / f'{model_name}_meta.json', meta)
    print(f'[{run_dir.name}] no ONNX for sklearn model {model_name!r} (needs skl2onnx) - '
         f'copied -> {out_pkl}')
    return out_pkl


def export_run(run_dir: Path, out_root: Path, opset: int):
    pt_path, pkl_path = run_dir / 'model.pt', run_dir / 'model.pkl'
    if pt_path.exists():
        return _export_torch(run_dir, pt_path, out_root, opset)
    if pkl_path.exists():
        return _export_sklearn(run_dir, pkl_path, out_root)
    return None


def find_runs(run_root: Path):
    """Every immediate subdirectory of run_root holding a model.pt or model.pkl (one per
    holdout_eval.py tag - see that module's tag() string for what the folder name encodes)."""
    if not run_root.is_dir():
        return []
    return sorted(d for d in run_root.iterdir()
                 if d.is_dir() and ((d / 'model.pt').exists() or (d / 'model.pkl').exists()))


def main():
    ap = argparse.ArgumentParser(description='Export role-2 holdout checkpoints for deployment '
                                            '(torch -> ONNX, sklearn -> .pkl passthrough)')
    ap.add_argument('--run_root', default='output/results_holdout',
                    help='directory of per-tag subfolders, as written by test/holdout/holdout_eval.py')
    ap.add_argument('--out', default='output/exports', help='output/ is git-ignored')
    ap.add_argument('--opset', type=int, default=15, help='ONNX opset version (torch models only)')
    ap.add_argument('--models', default=None,
                    help="comma-separated architecture names to restrict to (e.g. "
                         "'transformer,cnn_1d_wassem'); default: every run found under --run_root")
    args = ap.parse_args()

    run_root, out_root = Path(args.run_root), Path(args.out)
    wanted = set(args.models.split(',')) if args.models else None

    runs = find_runs(run_root)
    if not runs:
        raise SystemExit(f'No model.pt/model.pkl found under {run_root} - has run/run_holdout.sh been '
                         f're-run since the checkpoint-saving code was added (strategy doc §18.1)?')

    exported, skipped = 0, 0
    for run_dir in runs:
        # peek at the architecture name without fully loading, to honor --models cheaply
        if wanted is not None:
            probe_pt, probe_pkl = run_dir / 'model.pt', run_dir / 'model.pkl'
            if probe_pt.exists():
                model_name = torch.load(probe_pt, map_location='cpu', weights_only=False)['model']
            else:
                import joblib
                model_name = joblib.load(probe_pkl)['model']
            if model_name not in wanted:
                skipped += 1
                continue
        export_run(run_dir, out_root, args.opset)
        exported += 1

    print(f'[done] exported {exported} model(s)' + (f', skipped {skipped} (--models filter)' if skipped else ''))


if __name__ == '__main__':
    main()
