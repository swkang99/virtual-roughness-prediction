import numpy as np
from scipy.stats import skew, kurtosis
from PIL import Image

def compute_basic_stats(arr, prefix=""):
    arr = np.asarray(arr, dtype=np.float64).ravel()

    stats = {
        f"{prefix}mean": float(np.mean(arr)),
        f"{prefix}std": float(np.std(arr)),
        f"{prefix}min": float(np.min(arr)),
        f"{prefix}max": float(np.max(arr)),
        f"{prefix}median": float(np.median(arr)),
        f"{prefix}range": float(np.max(arr) - np.min(arr)),
        f"{prefix}skewness": float(skew(arr, bias=False)),
        f"{prefix}kurtosis": float(kurtosis(arr, bias=False)),
        f"{prefix}p10": float(np.percentile(arr, 10)),
        f"{prefix}p25": float(np.percentile(arr, 25)),
        f"{prefix}p75": float(np.percentile(arr, 75)),
        f"{prefix}p90": float(np.percentile(arr, 90)),
        f"{prefix}energy": float(np.mean(arr ** 2)),
    }
    return stats


def load_height_map(path, normalize=True):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)

    if normalize:
        arr = arr / 255.0

    return arr


def load_normal_map(path, normalize=True):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)

    if normalize:
        arr = arr / 255.0

    return arr


def extract_height_features(height_map):
    return compute_basic_stats(height_map, prefix="height_")


def extract_normal_features(normal_map):
    features = {}

    # channel-wise statistics
    features.update(compute_basic_stats(normal_map[:, :, 0], prefix="normal_r_"))
    features.update(compute_basic_stats(normal_map[:, :, 1], prefix="normal_g_"))
    features.update(compute_basic_stats(normal_map[:, :, 2], prefix="normal_b_"))

    # magnitude statistics
    magnitude = np.linalg.norm(normal_map, axis=2)
    features.update(compute_basic_stats(magnitude, prefix="normal_mag_"))

    return features

STAT_NAMES = [
    "mean", "std", "min", "max", "median", "range",
    "skewness", "kurtosis", "p10", "p25", "p75", "p90", "energy"
]

HEIGHT_KEYS = [f"height_{s}" for s in STAT_NAMES]

NORMAL_KEYS = []
for prefix in ["normal_r_", "normal_g_", "normal_b_", "normal_mag_"]:
    NORMAL_KEYS.extend([prefix + s for s in STAT_NAMES])


def dict_to_ordered_vector(feature_dict, keys):
    return np.asarray([feature_dict[k] for k in keys], dtype=np.float32)


if __name__ == "__main__":
    height_path = "height.png"
    normal_path = "normal.png"

    height_map = load_height_map(height_path, normalize=True)
    normal_map = load_normal_map(normal_path, normalize=True)

    height_features = extract_height_features(height_map)
    normal_features = extract_normal_features(normal_map)

    all_features = {**height_features, **normal_features}

    for k, v in all_features.items():
        print(f"{k}: {v:.6f}")