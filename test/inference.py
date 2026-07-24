import argparse
import csv
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.model.factory import create_model
from src.data.texture_maps import process_texture


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CKPT_EXTS = {".pt", ".pth", ".ckpt"}


def find_images(input_dir: Path, recursive: bool = False):
    if recursive:
        image_paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    else:
        image_paths = [p for p in input_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS]
    return sorted(image_paths)


def resolve_checkpoint_path(checkpoint_arg: str) -> Path:
    ckpt_path = Path(checkpoint_arg)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    if ckpt_path.is_file():
        return ckpt_path

    candidates = [p for p in ckpt_path.rglob("*") if p.is_file() and p.suffix.lower() in CKPT_EXTS]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in directory: {ckpt_path}")

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def extract_model_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint_obj:
                return checkpoint_obj[key]
    return checkpoint_obj


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_model(model_name: str, conf: dict, checkpoint_path: Path, device: torch.device):
    model_conf = dict(conf)
    model_conf["model"] = model_name

    input_dim = conf.get("input_dim", 5)
    model = create_model(model_conf, input_dim=input_dim, device=device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_model_state_dict(checkpoint)
    state_dict = strip_module_prefix(state_dict)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def get_target_size(conf: dict):
    size = conf.get("input_size", None)

    if size is None:
        return None

    if isinstance(size, int):
        return (size, size)

    if isinstance(size, (list, tuple)) and len(size) == 2:
        return (int(size[0]), int(size[1]))

    raise ValueError(f"Invalid input_size in config.yaml: {size}")


def get_target_range(conf: dict):
    if "train_y_min" not in conf or "train_y_max" not in conf:
        raise KeyError("config.yaml must contain 'train_y_min' and 'train_y_max' for denormalization.")

    y_min = float(conf["train_y_min"])
    y_max = float(conf["train_y_max"])

    if y_max <= y_min:
        raise ValueError(f"Invalid target range: train_y_min={y_min}, train_y_max={y_max}")

    return y_min, y_max


def denormalize_prediction(y_norm: float, conf: dict) -> float:
    y_min, y_max = get_target_range(conf)
    return y_norm * (y_max - y_min) + y_min


def maybe_resize_image(img: Image.Image, target_size, is_normal_map=False):
    if target_size is None:
        return img, False, img.size

    target_w, target_h = target_size
    current_w, current_h = img.size

    if (current_w, current_h) == (target_w, target_h):
        return img, False, img.size

    resample = Image.BILINEAR if is_normal_map else Image.BICUBIC
    resized = img.resize((target_w, target_h), resample=resample)
    return resized, True, resized.size


def load_input_tensors(texture_path: Path, height_path: Path, normal_path: Path, device: torch.device, conf: dict):
    to_tensor = transforms.ToTensor()
    target_size = get_target_size(conf)

    texture_img = Image.open(texture_path).convert("L")
    height_img = Image.open(height_path).convert("L")
    normal_img = Image.open(normal_path).convert("RGB")

    texture_img, texture_resized, final_size = maybe_resize_image(
        texture_img, target_size, is_normal_map=False
    )
    height_img, height_resized, _ = maybe_resize_image(
        height_img, target_size, is_normal_map=False
    )
    normal_img, normal_resized, _ = maybe_resize_image(
        normal_img, target_size, is_normal_map=True
    )

    texture_tensor = to_tensor(texture_img).unsqueeze(0).to(device)
    height_tensor = to_tensor(height_img).unsqueeze(0).to(device)
    normal_tensor = to_tensor(normal_img).unsqueeze(0).to(device)

    resized = texture_resized or height_resized or normal_resized
    return texture_tensor, height_tensor, normal_tensor, resized, final_size


def predict_one(model, texture_path: Path, conf: dict, device: torch.device):
    height_path, normal_path = process_texture(
        texture_path=texture_path,
        output_dir=conf.get("output_dir_name", "output_maps"),
        save_texture_maps=bool(conf.get("save_texture_maps", False)),
        blur_ksize=int(conf.get("blur_ksize", 5)),
        strength=float(conf.get("normal_strength", 4.0)),
        invert=bool(conf.get("invert_height", False)),
        invert_y=bool(conf.get("invert_normal_y", False)),
    )

    texture_tensor, height_tensor, normal_tensor, resized, final_size = load_input_tensors(
        texture_path=texture_path,
        height_path=Path(height_path),
        normal_path=Path(normal_path),
        device=device,
        conf=conf,
    )

    with torch.inference_mode():
        pred = model(texture_tensor, height_tensor, normal_tensor)

    pred_norm = float(pred.reshape(-1)[0].detach().cpu().item())
    pred_denorm = denormalize_prediction(pred_norm, conf)

    return pred_norm, pred_denorm, str(height_path), str(normal_path), resized, final_size


def print_results(results):
    print("\n========================= Inference Results =========================")
    print(
        f"{'No.':>4} | {'Filename':<30} | {'Pred(norm)':>11} | {'Roughness':>12} | {'Resized':>7} | {'Input Size':>10}"
    )
    print("-" * 96)

    for i, row in enumerate(results, start=1):
        resized_str = "Yes" if row["resized"] else "No"
        print(
            f"{i:>4} | "
            f"{row['filename']:<30} | "
            f"{row['pred_norm']:>11.6f} | "
            f"{row['roughness']:>12.6f} | "
            f"{resized_str:>7} | "
            f"{row['final_size']:>10}"
        )

    print("-" * 96)
    print(f"Total images: {len(results)}")
    print("====================================================================\n")


def save_results_csv(results, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "texture_path",
            "height_path",
            "normal_path",
            "model",
            "checkpoint_path",
            "pred_norm",
            "roughness",
            "resized",
            "final_size",
        ])

        for row in results:
            writer.writerow([
                row["filename"],
                row["texture_path"],
                row["height_path"],
                row["normal_path"],
                row["model"],
                row["checkpoint_path"],
                f"{row['pred_norm']:.6f}",
                f"{row['roughness']:.6f}",
                row["resized"],
                row["final_size"],
            ])


def main():
    parser = argparse.ArgumentParser(description="Run roughness inference on all local images in a directory.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/inference",
        help="Input directory containing texture images",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn_1d_simple", "transformer"],
        help="Model name for inference",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/",
        help="Checkpoint file path or checkpoint directory path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device like "cpu" or "cuda:0"',
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search images in subdirectories",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="output/inference/inference_results.csv",
        help="CSV path to save inference results",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    image_paths = find_images(input_dir, recursive=args.recursive)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    target_size = get_target_size(conf)
    y_min, y_max = get_target_range(conf)

    print(f"Device         : {device}")
    print(f"Model          : {args.model}")
    print(f"Checkpoint     : {checkpoint_path}")
    print(f"Input dir      : {input_dir}")
    print(f"Num files      : {len(image_paths)}")
    print(f"Target size    : {target_size}")
    print(f"train_y_min    : {y_min}")
    print(f"train_y_max    : {y_max}")

    model = load_model(args.model, conf, checkpoint_path, device)
    results = []

    for texture_path in image_paths:
        pred_norm, roughness, height_path, normal_path, resized, final_size = predict_one(
            model=model,
            texture_path=texture_path,
            conf=conf,
            device=device,
        )

        results.append({
            "filename": texture_path.name,
            "texture_path": str(texture_path),
            "height_path": height_path,
            "normal_path": normal_path,
            "model": args.model,
            "checkpoint_path": str(checkpoint_path),
            "pred_norm": pred_norm,
            "roughness": roughness,
            "resized": resized,
            "final_size": f"{final_size[0]}x{final_size[1]}",
        })

    print_results(results)
    save_results_csv(results, Path(args.save_csv))
    print(f"Saved CSV: {args.save_csv}")


if __name__ == "__main__":
    main()