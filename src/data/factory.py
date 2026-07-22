import numpy as np
from pathlib import Path
from src.data.dataset import PatchDataset, PatchFeatureDataset
from src.model.feature.feature_extractor import FeatureExtractor


MODEL_DATASET_TYPE = {
    "lr"           : "feature",
    "svr"          : "feature",
    "ann"          : "feature",
    "cnn_1d_scirep": "feature",
    "cnn_1d_wassem": "feature",
    "cnn_1d_simple": "maps",
    "transformer"  : "maps",
    "gated_mlp"    : "feature",
    "gated_mlp_v2" : "feature",
}


def _infer_split_name(full_df):
    """
    Infer split name from texture_path.

    Expected examples:
        .../train/...
        .../val/...
        .../test/...
    """
    if len(full_df) == 0 or "texture_path" not in full_df.columns:
        return "unknown"

    path = Path(str(full_df.iloc[0]["texture_path"]))
    parts = [p.lower() for p in path.parts]

    if "train" in parts:
        return "train"

    if "val" in parts or "valid" in parts:
        return "val"

    if "test" in parts:
        return "test"

    return "unknown"


def _get_feature_cache_root(conf):
    """
    Default:
        data/features

    Optional config.yaml override:
        feature_cache_root: data/features
    """
    if conf is not None and "feature_cache_root" in conf:
        return Path(conf["feature_cache_root"])

    # factory.py is located at data/factory.py
    # parent == data/
    return Path(__file__).resolve().parent / "features"


def _get_feature_cache_path(conf, full_df):
    split_name = _infer_split_name(full_df)
    cache_root = _get_feature_cache_root(conf)
    return cache_root / split_name / "features.npz"


def _get_texture_paths(full_df):
    """
    Store paths as a unicode string array, not an object array.
    This avoids:
        Object arrays cannot be loaded when allow_pickle=False
    """
    return np.asarray(
        full_df["texture_path"].map(str).tolist(),
        dtype=np.str_,
    )


def _get_targets(full_df):
    return full_df["roughness"].to_numpy(dtype=np.float32).reshape(-1, 1)


def _remove_cache(cache_path, reason):
    if cache_path.exists():
        print(f"[Feature cache] Removing old cache: {cache_path}")
        print(f"[Feature cache] Reason: {reason}")
        cache_path.unlink()


def _load_feature_cache(cache_path, full_df):
    """
    Return:
        (features, targets) if cache is valid.
        None if cache is missing, broken, or mismatched.

    If the cache exists but does not match the current dataset,
    it is deleted so that new features can be generated and saved.
    """
    if not cache_path.exists():
        return None

    try:
        data = np.load(cache_path, allow_pickle=False)

        required_keys = {"features", "targets", "texture_paths"}
        if not required_keys.issubset(set(data.files)):
            _remove_cache(cache_path, "missing required keys")
            return None

        cached_paths = data["texture_paths"].astype(np.str_)
        current_paths = _get_texture_paths(full_df)

        if len(cached_paths) != len(current_paths):
            _remove_cache(
                cache_path,
                f"sample count mismatch: cache={len(cached_paths)}, current={len(current_paths)}",
            )
            return None

        if not np.array_equal(cached_paths, current_paths):
            _remove_cache(cache_path, "texture_path order or values mismatch")
            return None

        cached_targets = data["targets"].astype(np.float32)
        current_targets = _get_targets(full_df)

        if cached_targets.shape != current_targets.shape:
            _remove_cache(
                cache_path,
                f"target shape mismatch: cache={cached_targets.shape}, current={current_targets.shape}",
            )
            return None

        if not np.allclose(cached_targets, current_targets, atol=1e-6):
            _remove_cache(cache_path, "target values mismatch")
            return None

        features = data["features"].astype(np.float32)

        if features.shape[0] != len(current_paths):
            _remove_cache(
                cache_path,
                f"feature sample count mismatch: features={features.shape[0]}, current={len(current_paths)}",
            )
            return None

        print(f"[Feature cache] Loaded: {cache_path}")
        return features, cached_targets

    except Exception as e:
        _remove_cache(cache_path, f"failed to load cache: {e}")
        return None


def _save_feature_cache(cache_path, full_df, features, targets):
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        cache_path,
        features=features.astype(np.float32),
        targets=targets.astype(np.float32),
        texture_paths=_get_texture_paths(full_df),
    )

    print(f"[Feature cache] Saved: {cache_path}")


def build_feature_base_dataset(full_df, device, conf=None):
    """
    Build feature-based dataset.

    If a valid cache exists:
        load data/features/{train,val,test}/features.npz

    If the cache is missing or does not match current data:
        remove old cache
        extract features again
        save new cache
    """
    if conf is None:
        conf = {}

    cache_path = _get_feature_cache_path(conf, full_df)
    force_recompute = bool(conf.get("force_recompute_features", False))

    if force_recompute and cache_path.exists():
        _remove_cache(cache_path, "force_recompute_features=True")

    if not force_recompute:
        cached = _load_feature_cache(cache_path, full_df)

        if cached is not None:
            full_features, full_targets = cached
            input_dim = full_features.shape[1]
            return PatchFeatureDataset(full_features, full_targets), full_targets, input_dim

    feature_extractor = FeatureExtractor(device)
    full_features, full_targets = feature_extractor.precompute_features_and_targets(full_df)

    input_dim = full_features.shape[1]

    _save_feature_cache(cache_path, full_df, full_features, full_targets)

    return PatchFeatureDataset(full_features, full_targets), full_targets, input_dim


def build_maps_base_dataset(full_df):
    base_dataset = PatchDataset(full_df)
    full_targets = full_df["roughness"].to_numpy(dtype=np.float32).reshape(-1, 1)
    return base_dataset, full_targets, None


def build_base_dataset(conf, full_df, device):
    dataset_type = MODEL_DATASET_TYPE[conf["model"]]

    if dataset_type == "feature":
        return build_feature_base_dataset(full_df, device, conf)

    elif dataset_type == "maps":
        return build_maps_base_dataset(full_df)

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")