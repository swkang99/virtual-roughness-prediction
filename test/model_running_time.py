import time
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from src.model.factory import create_model
from src.model.feature.feature_extractor import FeatureExtractor


TARGET_MODELS = ["cnn_1d_4ha", "cnn_1d_generic", "transformer"]


def _sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _mean_ms(times):
    return float(np.mean(times)) if len(times) > 0 else 0.0


def _load_nparrays_from_base_images(texture_path, height_path, normal_path):
    t_img = Image.open(texture_path).convert("L")
    t_np = np.array(t_img, dtype=np.uint8)

    h_img = Image.open(height_path).convert("L")
    h_np = np.array(h_img, dtype=np.uint8)

    n_img = Image.open(normal_path).convert("RGB")
    n_np = np.array(n_img, dtype=np.uint8)

    return t_np, h_np, n_np


def _build_model(model_name, conf, input_dim, device):
    model_conf = dict(conf)
    model_conf["model"] = model_name
    model = create_model(model_conf, input_dim=input_dim, device=device)
    model.eval()
    return model


def _time_cnn_1d_4ha(model, feature_extractor, texture_np, num_runs, warmup, device):
    glcm_times = []
    lbp_times = []
    resnet_times = []
    forward_times = []

    with torch.inference_mode():
        for _ in tqdm(range(warmup), desc="cnn_1d_4ha warmup", leave=False):
            glcm_feat = feature_extractor.extract_glcm_features(texture_np)
            lbp_feat = feature_extractor.extract_lbp_features(texture_np)
            resnet_feat = feature_extractor.extract_resnet50_features(texture_np)

            combined = np.concatenate([
                np.asarray(glcm_feat, dtype=np.float32),
                np.asarray(lbp_feat, dtype=np.float32),
                np.asarray(resnet_feat, dtype=np.float32),
            ], axis=0).astype(np.float32)

            x = torch.from_numpy(combined).to(device).unsqueeze(0)
            _ = model(x)

    _sync_if_cuda(device)

    with torch.inference_mode():
        for _ in tqdm(range(num_runs), desc="cnn_1d_4ha measure", leave=False):
            _sync_if_cuda(device)
            t0 = time.perf_counter()
            glcm_feat = feature_extractor.extract_glcm_features(texture_np)
            _sync_if_cuda(device)
            t1 = time.perf_counter()

            lbp_feat = feature_extractor.extract_lbp_features(texture_np)
            _sync_if_cuda(device)
            t2 = time.perf_counter()

            resnet_feat = feature_extractor.extract_resnet50_features(texture_np)
            _sync_if_cuda(device)
            t3 = time.perf_counter()

            combined = np.concatenate([
                np.asarray(glcm_feat, dtype=np.float32),
                np.asarray(lbp_feat, dtype=np.float32),
                np.asarray(resnet_feat, dtype=np.float32),
            ], axis=0).astype(np.float32)

            x = torch.from_numpy(combined).to(device).unsqueeze(0)

            _sync_if_cuda(device)
            t4 = time.perf_counter()
            _ = model(x)
            _sync_if_cuda(device)
            t5 = time.perf_counter()

            glcm_times.append((t1 - t0) * 1000.0)
            lbp_times.append((t2 - t1) * 1000.0)
            resnet_times.append((t3 - t2) * 1000.0)
            forward_times.append((t5 - t4) * 1000.0)

    return {
        "glcm_ms": _mean_ms(glcm_times),
        "lbp_ms": _mean_ms(lbp_times),
        "resnet50_ms": _mean_ms(resnet_times),
        "feature_total_ms": _mean_ms(
            [g + l + r for g, l, r in zip(glcm_times, lbp_times, resnet_times)]
        ),
        "forward_ms": _mean_ms(forward_times),
        "total_ms": _mean_ms(
            [g + l + r + f for g, l, r, f in zip(glcm_times, lbp_times, resnet_times, forward_times)]
        ),
    }


def _time_end_to_end_model(model, texture_tensor, height_tensor, normal_tensor, num_runs, warmup, device, desc):
    forward_times = []

    with torch.inference_mode():
        for _ in tqdm(range(warmup), desc=f"{desc} warmup", leave=False):
            _ = model(texture_tensor, height_tensor, normal_tensor)

    _sync_if_cuda(device)

    with torch.inference_mode():
        for _ in tqdm(range(num_runs), desc=f"{desc} measure", leave=False):
            _sync_if_cuda(device)
            t0 = time.perf_counter()
            _ = model(texture_tensor, height_tensor, normal_tensor)
            _sync_if_cuda(device)
            t1 = time.perf_counter()
            forward_times.append((t1 - t0) * 1000.0)

    return {
        "forward_ms": _mean_ms(forward_times),
    }


def run_all_single_image_benchmarks(conf):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    texture_path = Path(conf.get("single_texture_path", "data/split/test/patches/2_patch_0000.png"))
    height_path = Path(conf.get("single_height_path", "data/split/test/output_maps/2_patch_0000_height_map_gray.png"))
    normal_path = Path(conf.get("single_normal_path", "data/split/test/output_maps/2_patch_0000_normal_map_rgb.png"))
    num_runs = int(conf.get("speed_test_num_samples", 1000))
    warmup = int(conf.get("warmup_iters", 20))

    print(f"Texture path: {texture_path}")
    print(f"Height path : {height_path}")
    print(f"Normal path : {normal_path}")
    print("Batch size: 1")
    print(f"Warmup iterations: {warmup}")
    print(f"Measured runs: {num_runs}")

    texture_np, height_np, normal_np = _load_nparrays_from_base_images(
        texture_path, height_path, normal_path
    )

    feature_extractor = FeatureExtractor(device)
    results = {}

    print("\nPreparing cnn_1d_4ha feature dimensions...")
    glcm_feat = feature_extractor.extract_glcm_features(texture_np)
    lbp_feat = feature_extractor.extract_lbp_features(texture_np)
    resnet_feat = feature_extractor.extract_resnet50_features(texture_np)

    input_dim_4ha = int(
        np.asarray(glcm_feat).shape[0]
        + np.asarray(lbp_feat).shape[0]
        + np.asarray(resnet_feat).shape[0]
    )

    model_4ha = _build_model("cnn_1d_4ha", conf, input_dim=input_dim_4ha, device=device)
    results["cnn_1d_4ha"] = _time_cnn_1d_4ha(
        model=model_4ha,
        feature_extractor=feature_extractor,
        texture_np=texture_np,
        num_runs=num_runs,
        warmup=warmup,
        device=device,
    )

    to_tensor = transforms.ToTensor()
    texture_tensor = to_tensor(texture_np).unsqueeze(0).to(device)
    height_tensor = to_tensor(height_np).unsqueeze(0).to(device)
    normal_tensor = to_tensor(normal_np).unsqueeze(0).to(device)

    model_generic = _build_model("cnn_1d_generic", conf, input_dim=None, device=device)
    results["cnn_1d_generic"] = _time_end_to_end_model(
        model=model_generic,
        texture_tensor=texture_tensor,
        height_tensor=height_tensor,
        normal_tensor=normal_tensor,
        num_runs=num_runs,
        warmup=warmup,
        device=device,
        desc="cnn_1d_generic",
    )

    model_transformer = _build_model("transformer", conf, input_dim=None, device=device)
    results["transformer"] = _time_end_to_end_model(
        model=model_transformer,
        texture_tensor=texture_tensor,
        height_tensor=height_tensor,
        normal_tensor=normal_tensor,
        num_runs=num_runs,
        warmup=warmup,
        device=device,
        desc="transformer",
    )

    return results


def print_results(results):
    print("\n========== Single-image inference benchmark (mean over configured runs) ==========")

    r = results["cnn_1d_4ha"]
    print("\n[cnn_1d_4ha]")
    print(f"GLCM feature extraction   : {r['glcm_ms']:.4f} ms")
    print(f"LBP feature extraction    : {r['lbp_ms']:.4f} ms")
    print(f"ResNet50 feature extract. : {r['resnet50_ms']:.4f} ms")
    print(f"Feature extraction total  : {r['feature_total_ms']:.4f} ms")
    print(f"Model forward             : {r['forward_ms']:.4f} ms")
    print(f"Total                     : {r['total_ms']:.4f} ms")

    r = results["cnn_1d_generic"]
    print("\n[cnn_1d_generic]")
    print(f"Model forward             : {r['forward_ms']:.4f} ms")

    r = results["transformer"]
    print("\n[transformer]")
    print(f"Model forward             : {r['forward_ms']:.4f} ms")

    print("\n===============================================================================")


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    results = run_all_single_image_benchmarks(conf)
    print_results(results)


if __name__ == "__main__":
    main()