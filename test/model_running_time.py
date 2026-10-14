"""
Single-patch inference time of every compared model (batch size 1), following the VR use case:

    finger touches texture T at (u, v) -> crop the 256x256 patch -> predict roughness

* Image models (Transformer, Simple 1D-CNN): the height / normal maps are texture assets that
  already exist in Unity, so only "patch -> tensor" and the forward pass are timed.
* Feature models (LR, SVR, ANN, Hassan 1D-CNN, Scirep 1D-CNN): their input features depend on the
  touched patch, so GLCM + LBP + ResNet-50 extraction is timed on every call, plus the forward pass.

Reported per model: mean, SD, median and 95th percentile (ms) over `speed_test_num_samples` runs after
`warmup_iters` warm-up runs. Results are also written to output/runtime/.

Usage (repo root):
    python -m test.model_running_time [--feature_mode legacy|matlab] [--device cuda|cpu] [--runs 1000]
Weights do not affect run time, so untrained models are used; LR / SVR are fitted on the cached
training features (SVR prediction time grows with its number of support vectors).
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from src.model.factory import create_model
from src.model.feature.feature_extractor import FeatureExtractor

FEATURE_MODELS = ["lr", "svr", "ann", "cnn_1d_wassem", "cnn_1d_scirep"]
MAP_MODELS = ["cnn_1d_simple", "transformer"]
DISPLAY = {"lr": "LR", "svr": "SVR", "ann": "ANN", "cnn_1d_wassem": "Hassan 1D-CNN",
           "cnn_1d_scirep": "Scirep 1D-CNN", "cnn_1d_simple": "Simple 1D-CNN", "transformer": "Transformer (ours)"}


# ----------------------------------------------------------------------------- helpers
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _stats(ms):
    ms = np.asarray(ms, dtype=np.float64)
    return {"mean": ms.mean(), "sd": ms.std(ddof=1), "median": np.median(ms), "p95": np.percentile(ms, 95)}


def _load_patch(texture_path, height_path, normal_path):
    t = np.array(Image.open(texture_path).convert("L"), dtype=np.uint8)
    rgb = np.array(Image.open(texture_path).convert("RGB"), dtype=np.uint8)
    h = np.array(Image.open(height_path).convert("L"), dtype=np.uint8)
    n = np.array(Image.open(normal_path).convert("RGB"), dtype=np.uint8)
    return t, h, n, rgb


def _build_model(name, conf, input_dim, device):
    c = dict(conf); c["model"] = name
    m = create_model(c, input_dim=input_dim, device=device)
    if isinstance(m, torch.nn.Module):
        m.eval()
    return m


def _cached_train_features(conf, feature_mode):
    root = Path(conf.get("feature_cache_root", "data/features"))
    if feature_mode != "legacy":
        root = root / feature_mode
    cache = root / "train" / "features.npz"
    return np.load(cache, allow_pickle=False)["features"] if cache.exists() else None


def _fit_sklearn(model, input_dim, conf, feature_mode):
    """Fit LR / SVR on the cached training features (realistic number of support vectors)."""
    root = Path(conf.get("feature_cache_root", "data/features"))
    if feature_mode != "legacy":
        root = root / feature_mode
    cache = root / "train" / "features.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        X, y = d["features"], d["targets"].ravel()
        src = str(cache)
    else:
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(4480, input_dim)).astype(np.float32), rng.uniform(10, 95, 4480)
        src = "random features (cache not found - SVR time is NOT representative)"
    model.fit(X, y)
    info = {"fit_data": src}
    if hasattr(model, "support_"):
        info["n_support_vectors"] = int(len(model.support_))
    return info


# ----------------------------------------------------------------------------- timing
def time_feature_models(models, extractor, patch_np, runs, warmup, device, glcm_lo=None, glcm_hi=None):
    """Feature extraction is shared by all feature models:
    each call = (grey conversion) + GLCM + LBP + ResNet-50 (+ GLCM min-max in 'original' mode) + forward."""
    from src.model.feature.feature_extractor import GLCM_SLICE
    rec = {k: [] for k in models}
    parts = {"gray": [], "glcm": [], "lbp": [], "resnet50": []}
    original = extractor.mode == "original"
    cols = GLCM_SLICE[extractor.mode]
    rng = None if glcm_lo is None else np.where(glcm_hi - glcm_lo > 0, glcm_hi - glcm_lo, 1.0)

    def extract():
        _sync(device); t0 = time.perf_counter()
        img = extractor._gray(patch_np); tg = time.perf_counter()          # RGB -> grey (original mode)
        g = extractor.extract_glcm_features(img); t1 = time.perf_counter()
        l = extractor.extract_lbp_features(img); t2 = time.perf_counter()
        r = extractor.extract_resnet50_features(img); _sync(device); t3 = time.perf_counter()
        x = extractor.concat(g, l, r)
        if original and rng is not None:
            x[cols] = (x[cols] - glcm_lo) / rng
        t4 = time.perf_counter()
        parts["gray"].append((tg - t0) * 1e3)
        return x, ((t1 - tg) * 1e3, (t2 - t1) * 1e3, (t3 - t2 + t4 - t3) * 1e3)

    def forward(name, m, x):
        _sync(device); t0 = time.perf_counter()
        if isinstance(m, torch.nn.Module):
            m(torch.from_numpy(x).to(device).unsqueeze(0))
        else:
            m.predict(x[None, :])
        _sync(device)
        return (time.perf_counter() - t0) * 1e3

    with torch.inference_mode():
        for _ in tqdm(range(warmup), desc="feature models warm-up", leave=False):
            x, _ = extract()
            for name, m in models.items():
                forward(name, m, x)
        for _ in tqdm(range(runs), desc="feature models", leave=False):
            x, (g, l, r) = extract()
            parts["glcm"].append(g); parts["lbp"].append(l); parts["resnet50"].append(r)
            for name, m in models.items():
                rec[name].append(forward(name, m, x))

    parts["gray"] = parts["gray"][-runs:]                                  # drop warm-up entries
    feat_total = np.asarray(parts["gray"]) + np.add(np.add(parts["glcm"], parts["lbp"]), parts["resnet50"])
    out = {}
    for name in models:
        fwd = np.asarray(rec[name])
        out[name] = {"feature_ms": _stats(feat_total), "gray_ms": _stats(parts["gray"]), "glcm_ms": _stats(parts["glcm"]), "lbp_ms": _stats(parts["lbp"]),
                     "resnet50_ms": _stats(parts["resnet50"]), "forward_ms": _stats(fwd), "total_ms": _stats(feat_total + fwd)}
    return out


def time_map_model(model, t_np, h_np, n_np, runs, warmup, device, desc):
    """patch arrays -> tensors on device (input_ms) + forward pass (forward_ms)."""
    to_tensor = transforms.ToTensor()

    def once():
        _sync(device); t0 = time.perf_counter()
        t = to_tensor(t_np).unsqueeze(0).to(device)
        h = to_tensor(h_np).unsqueeze(0).to(device)
        n = to_tensor(n_np).unsqueeze(0).to(device)
        _sync(device); t1 = time.perf_counter()
        model(t, h, n)
        _sync(device); t2 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3

    with torch.inference_mode():
        for _ in tqdm(range(warmup), desc=f"{desc} warm-up", leave=False):
            once()
        inp, fwd = zip(*[once() for _ in tqdm(range(runs), desc=desc, leave=False)])
    inp, fwd = np.asarray(inp), np.asarray(fwd)
    return {"input_ms": _stats(inp), "forward_ms": _stats(fwd), "total_ms": _stats(inp + fwd)}


# ----------------------------------------------------------------------------- main
def run(conf, feature_mode, device, runs, warmup):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    tp = Path(conf.get("single_texture_path", "data/test/patches/2_patch_0000.png"))
    hp = Path(conf.get("single_height_path", "data/test/ps_maps/2_patch_0000_height_map_gray.png"))
    npth = Path(conf.get("single_normal_path", "data/test/ps_maps/2_patch_0000_normal_map_rgb.png"))
    t_np, h_np, n_np, rgb_np = _load_patch(tp, hp, npth)

    env = {"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "cpu": platform.processor() or platform.machine(), "torch": torch.__version__,
           "torch_threads": torch.get_num_threads(), "batch_size": 1, "patch": list(t_np.shape),
           "runs": runs, "warmup": warmup, "feature_mode": feature_mode, "texture_path": str(tp)}
    print(json.dumps(env, indent=1))

    extractor = FeatureExtractor(device, mode=feature_mode)
    patch_in = rgb_np if feature_mode == "original" else t_np      # original: MATLAB rgb2gray is part of the pipeline
    dim = int(extractor.extract_array_features(patch_in).shape[0])
    feat_models, fit_info = {}, {}
    for name in FEATURE_MODELS:
        m = _build_model(name, conf, dim, device)
        if not isinstance(m, torch.nn.Module):
            fit_info[name] = _fit_sklearn(m, dim, conf, feature_mode)
        feat_models[name] = m

    lo = hi = None
    if feature_mode == "original":                                  # GLCM min-max statistics (training features)
        Xc = _cached_train_features(conf, feature_mode)
        from src.model.feature.feature_extractor import GLCM_SLICE
        G = Xc[:, GLCM_SLICE["original"]] if Xc is not None else np.random.default_rng(0).random((10, 64))
        lo, hi = G.min(0), G.max(0)
    results = time_feature_models(feat_models, extractor, patch_in, runs, warmup, device, lo, hi)
    for name in MAP_MODELS:
        results[name] = time_map_model(_build_model(name, conf, None, device), t_np, h_np, n_np, runs, warmup, device, name)
    return env, fit_info, results


def report(env, fit_info, results, out_dir):
    rows = []
    for name in FEATURE_MODELS + MAP_MODELS:
        r = results[name]
        row = {"model": DISPLAY[name]}
        for k in ["gray_ms", "glcm_ms", "lbp_ms", "resnet50_ms", "feature_ms", "input_ms", "forward_ms", "total_ms"]:
            if k in r:
                row[k.replace("_ms", "")] = r[k]["mean"]
        row["total_sd"], row["total_median"], row["total_p95"] = (r["total_ms"][s] for s in ("sd", "median", "p95"))
        rows.append(row)
    df = pd.DataFrame(rows)

    print("\n========== Single-patch inference time (ms, batch 1) ==========")
    print(df.round(3).to_string(index=False))
    print("total = feature extraction (grey+GLCM+LBP+ResNet-50) + forward for feature models; "
          "patch->tensor + forward for image models (maps are pre-existing Unity assets).")
    if fit_info:
        print("sklearn fit:", json.dumps(fit_info))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"runtime_{env['device'].split(':')[0]}_{env['feature_mode']}"
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    json.dump({"env": env, "sklearn_fit": fit_info, "results": results}, open(out_dir / f"{stem}.json", "w"),
              indent=1, default=float)
    print(f"saved: {out_dir / (stem + '.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_mode", choices=["legacy", "original"], default=None)
    ap.add_argument("--feature_cache_root", default=None, help="where the training features are cached (LR/SVR fit, GLCM min-max)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--out", default="output/runtime")
    a = ap.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    feature_mode = a.feature_mode or conf.get("feature_mode", "original")
    if a.feature_cache_root:
        conf["feature_cache_root"] = a.feature_cache_root
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    runs = a.runs or int(conf.get("speed_test_num_samples", 1000))
    warmup = a.warmup or int(conf.get("warmup_iters", 20))

    env, fit_info, results = run(conf, feature_mode, device, runs, warmup)
    report(env, fit_info, results, Path(a.out))


if __name__ == "__main__":
    main()
