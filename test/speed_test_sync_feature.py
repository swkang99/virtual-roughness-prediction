import time
import yaml
import torch
from pathlib import Path
import numpy as np

from src.data.factory import MODEL_DATASET_TYPE
from src.model.factory import create_model
from src.model.feature.feature_extractor import FeatureExtractor
from src.data.texture_maps import load_grayscale_image, extract_height_map, extract_normal_map_rgb

from PIL import Image
from torchvision import transforms


def _sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_single_image_pipeline(conf, texture_path: Path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset_type = MODEL_DATASET_TYPE[conf["model"]]

    # Create model and optional feature_extractor
    feature_extractor = None
    if dataset_type != "original":
        feature_extractor = FeatureExtractor(device)

    # Prepare transforms
    pil_to_tensor = transforms.Compose([transforms.ToTensor()])

    # Map extraction (measure)
    fm_start = time.perf_counter()
    gray = load_grayscale_image(texture_path)
    height_map = extract_height_map(
        gray,
        blur_ksize=int(conf.get("blur_ksize", 5)),
        invert=False,
        normalize_output=True,
    )
    normal_rgb = extract_normal_map_rgb(
        height_map,
        strength=float(conf.get("normal_strength", 4.0)),
        invert_y=False,
    )
    _sync_if_cuda(device)
    fm_end = time.perf_counter()

    # Feature extraction (measure) and compute input_dim for model
    _sync_if_cuda(device)
    f_start = time.perf_counter()

    input_dim = None
    # Precompute feature tensors/vectors for forward (reuse later)

    if dataset_type == "feature":
        t_feat = feature_extractor.extract_single_image_features(str(texture_path))
        n_feat = feature_extractor.extract_single_image_features(str(texture_path))
        h_feat = feature_extractor.extract_single_image_features(str(texture_path))
        input_dim = int(np.asarray(t_feat).shape[0] + np.asarray(n_feat).shape[0] + np.asarray(h_feat).shape[0])
        combined = np.concatenate([t_feat, n_feat, h_feat]).astype(np.float32)
        x = torch.tensor(combined, dtype=torch.float32, device=device).unsqueeze(0)

    else:
        input_dim = None

    _sync_if_cuda(device)
    f_end = time.perf_counter()

    # instantiate model (now that input_dim is known)
    model = create_model(conf, input_dim=input_dim, device=device)
    model.eval()

    # Forward
    _sync_if_cuda(device)
    fw_start = time.perf_counter()
    with torch.inference_mode():
        if dataset_type == "feature":
            pred = model(x)
        else:
            # original: prepare tensors and forward
            img = Image.open(texture_path).convert("L")
            t_img = pil_to_tensor(img).unsqueeze(0).to(device)
            h_img = Image.fromarray((height_map * 255.0).astype(np.uint8)).convert("L")
            n_img = Image.fromarray(normal_rgb).convert("RGB")
            h_t = pil_to_tensor(h_img).unsqueeze(0).to(device)
            n_t = pil_to_tensor(n_img.convert("L")).unsqueeze(0).to(device)
            pred = model(t_img, h_t, n_t) if hasattr(model, "forward") else model(t_img)
    _sync_if_cuda(device)
    fw_end = time.perf_counter()

    # Postprocess
    post_start = time.perf_counter()
    if isinstance(pred, torch.Tensor):
        out = pred.detach().cpu().numpy()
    else:
        out = np.asarray(pred)
    post_end = time.perf_counter()

    map_ms = (fm_end - fm_start) * 1000.0
    feature_ms = (f_end - f_start) * 1000.0
    forward_ms = (fw_end - fw_start) * 1000.0
    post_ms = (post_end - post_start) * 1000.0
    total_ms = (post_end - f_start) * 1000.0

    return {
        "map_ms": map_ms,
        "feature_ms": feature_ms,
        "forward_ms": forward_ms,
        "post_ms": post_ms,
        "total_ms": total_ms,
        "output": out,
    }


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    single = Path(conf.get("single_texture_path", "data/split/test/patches/2_patch_0000.png"))
    res = run_single_image_pipeline(conf, single)

    print("\n=== Single Image Pipeline Timing ===")
    print(f"Map extraction      : {res['map_ms']:.3f} ms")
    print(f"Feature extraction  : {res['feature_ms']:.3f} ms")
    print(f"Model forward       : {res['forward_ms']:.3f} ms")
    print(f"Postprocess         : {res['post_ms']:.3f} ms")
    print(f"Total               : {res['total_ms']:.3f} ms")


if __name__ == "__main__":
    main()