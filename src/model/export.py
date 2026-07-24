# Model export into ONNXimport argparse

import argparse
import json
from pathlib import Path

import torch


CHECKPOINT_PATHS = {
    "cnn_1d_simple": Path("experiments/checkpoints/patch_roughness_cnn_1d_simple_300epoch/best.pt"),
    "transformer": Path("experiments/checkpoints/patch_roughness_transformer_300epoch/best.pt"),
}

EXPORT_PATHS = {
    "cnn_1d_simple": Path("output/exports/onnx/cnn_1d_simple/cnn_1d_simple.onnx"),
    "transformer": Path("output/exports/onnx/transformer/transformer.onnx"),
}

META_PATHS = {
    "cnn_1d_simple": Path("output/exports/onnx/cnn_1d_simple/cnn_1d_simple_meta.json"),
    "transformer": Path("output/exports/onnx/transformer/transformer_meta.json"),
}


def build_model(model_name: str, device: torch.device):
    from src.model.factory import create_model

    conf = {
        "model": model_name
    }

    input_dim = None
    model = create_model(conf=conf, input_dim=input_dim, device=device)

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"{model_name} is not a torch.nn.Module. Got: {type(model)}")

    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        new_key = new_key.replace("_orig_mod.", "") if new_key.startswith("_orig_mod.") else new_key
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def make_dummy_inputs(device: torch.device):
    texture_image = torch.randn(1, 1, 256, 256, dtype=torch.float32, device=device)
    height_map = torch.randn(1, 1, 256, 256, dtype=torch.float32, device=device)
    normal_map = torch.randn(1, 3, 256, 256, dtype=torch.float32, device=device)
    return texture_image, height_map, normal_map


def save_meta(meta_path: Path, model_name: str, y_min: float, y_max: float):
    meta = {
        "model": model_name,
        "input_names": ["texture_image", "height_map", "normal_map"],
        "input_shapes": {
            "texture_image": [1, 1, 256, 256],
            "height_map": [1, 1, 256, 256],
            "normal_map": [1, 3, 256, 256],
        },
        "output_names": ["roughness"],
        "output_description": "normalized roughness prediction",
        "normalization": {
            "target_min": y_min,
            "target_max": y_max,
        },
        "onnx_opset": 15
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def export_onnx(model_name: str, opset: int = 15):
    device = torch.device("cpu")

    ckpt_path = CHECKPOINT_PATHS[model_name]
    export_path = EXPORT_PATHS[model_name]
    meta_path = META_PATHS[model_name]

    export_path.parent.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(model_name, device)
    model = load_checkpoint(model, ckpt_path, device)

    texture_image, height_map, normal_map = make_dummy_inputs(device)

    with torch.no_grad():
        output = model(texture_image, height_map, normal_map)
        if isinstance(output, (tuple, list)):
            raise ValueError("ONNX export expects tensor output; current model returned tuple/list.")
        print(f"[INFO] PyTorch forward success. Output shape: {tuple(output.shape)}")

    torch.onnx.export(
        model,
        (texture_image, height_map, normal_map),
        str(export_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["texture_image", "height_map", "normal_map"],
        output_names=["roughness"],
        dynamic_axes={
            "texture_image": {0: "batch_size"},
            "height_map": {0: "batch_size"},
            "normal_map": {0: "batch_size"},
            "roughness": {0: "batch_size"},
        },
    )

    train_y_min = 13.4860
    train_y_max = 95.5010
    save_meta(meta_path, model_name, train_y_min, train_y_max)

    print(f"[INFO] Exported ONNX model to: {export_path}")
    print(f"[INFO] Saved metadata to: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Export trained model to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn_1d_simple", "transformer"],
        help="Model name to export"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=15,
        help="ONNX opset version"
    )
    args = parser.parse_args()

    export_onnx(args.model, args.opset)


if __name__ == "__main__":
    main()