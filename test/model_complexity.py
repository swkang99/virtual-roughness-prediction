import argparse
import torch
from src.model.factory import create_model


MODEL_RECOMMENDED_INPUTS = {
    "lr": {"input_dim": 3955, "input_res": None},
    "svr": {"input_dim": 3955, "input_res": None},
    "ann": {"input_dim": 3955, "input_res": (1, 3955)},
    "cnn_1d_scirep": {"input_dim": 3955, "input_res": (1, 3955)},
    "cnn_1d_4ha": {"input_dim": 3955, "input_res": (1, 3955)},
    "cnn_1d_simple": {
        "input_dim": None,
        "input_res": ((1, 256, 256), (1, 256, 256), (3, 256, 256)),
    },
    "transformer": {
        "input_dim": None,
        "input_res": ((1, 256, 256), (1, 256, 256), (3, 256, 256)),
    },
    "gated_mlp": {
        "input_dim": {"texture_dim": 1024, "height_dim": 1024, "normal_dim": 1024},
        "input_res": None,
    },
    "gated_mlp_v2": {
        "input_dim": {"texture_dim": 1024, "height_dim": 1024, "normal_dim": 1024},
        "input_res": None,
    },
}


def _wrap_model_for_flops(model_name, model):
    return model


def _default_backend_for_model(model_name):
    if model_name == "transformer":
        return "aten"
    return "pytorch"


def _infer_dtype_and_device(model):
    try:
        p = next(model.parameters())
        return p.dtype, p.device
    except StopIteration:
        return torch.float32, torch.device("cpu")


def _make_tensor(shape, dtype, device):
    return torch.randn((1, *shape), dtype=dtype, device=device)


def _triple_input_constructor_factory(model):
    dtype, device = _infer_dtype_and_device(model)

    def prepare_input(input_res):
        if not isinstance(input_res, (tuple, list)) or len(input_res) != 3:
            raise ValueError(
                "Multi-input input_res must be a 3-tuple like "
                "((1,256,256), (1,256,256), (3,256,256))"
            )

        texture_shape, height_shape, normal_shape = input_res

        texture_image = _make_tensor(texture_shape, dtype, device)
        height_map = _make_tensor(height_shape, dtype, device)
        normal_map = _make_tensor(normal_shape, dtype, device)

        return {
            "texture_image": texture_image,
            "height_map": height_map,
            "normal_map": normal_map,
        }

    return prepare_input


def _input_constructor_for_model(model_name, model):
    if model_name in {"transformer", "cnn_1d_simple"}:
        return _triple_input_constructor_factory(model)
    return None


def measure_model(
    model,
    input_res=None,
    compute_flops=False,
    as_strings=False,
    print_per_layer=False,
    device=None,
    model_name=None,
    backend=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    result = {
        "params": params,
        "flops": None,
        "params_from_tool": None,
        "flops_error": None,
        "backend": backend,
    }

    if compute_flops:
        from ptflops import get_model_complexity_info

        if backend is None:
            backend = _default_backend_for_model(model_name)
            result["backend"] = backend

        try:
            flops_model = _wrap_model_for_flops(model_name, model)
            input_constructor = _input_constructor_for_model(model_name, flops_model)

            if input_constructor is None and input_res is None:
                raise ValueError("input_res must be provided to compute FLOPs")

            flops, params_from_tool = get_model_complexity_info(
                flops_model,
                input_res,
                as_strings=as_strings,
                print_per_layer_stat=print_per_layer,
                input_constructor=input_constructor,
                backend=backend,
            )

            result["flops"] = flops
            result["params_from_tool"] = params_from_tool

        except Exception as e:
            result["flops_error"] = str(e)

    return result


def build_model_from_name(model_name, input_dim=None, device=None):
    conf = {"model": model_name}
    return create_model(conf, input_dim, device=torch.device(device) if device else None)


def get_recommended_inputs(model_name, input_dim=None, input_res=None):
    spec = MODEL_RECOMMENDED_INPUTS.get(model_name, {})
    resolved_input_dim = input_dim if input_dim is not None else spec.get("input_dim")
    resolved_input_res = input_res if input_res is not None else spec.get("input_res")
    return resolved_input_dim, resolved_input_res


def parse_shape_string(s):
    s = s.strip()
    if not s:
        return None
    return tuple(int(x) for x in s.split(",") if x.strip())


def parse_input_res(s, model_name=None):
    if s is None:
        return None

    s = s.strip()

    if ";" in s:
        return tuple(parse_shape_string(part) for part in s.split(";"))

    return parse_shape_string(s)


def format_number(n):
    return f"{int(n):,}"


def main():
    parser = argparse.ArgumentParser(description="Measure model complexity (params and FLOPs/MACs).")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--input_dim", type=int, default=None, help="Input dim for model builder")
    parser.add_argument(
        "--input_res",
        type=str,
        default=None,
        help=(
            'Single input example: "3,224,224" or "3,65536". '
            'Multi-input example: "1,256,256;1,256,256;3,256,256"'
        ),
    )
    parser.add_argument("--compute_flops", action="store_true", help="Compute FLOPs/MACs using ptflops")
    parser.add_argument("--for_paper", action="store_true", help="Print paper-friendly summary")
    parser.add_argument("--print_per_layer", action="store_true", help="Print per-layer ptflops stats")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["pytorch", "aten"],
        help="ptflops backend; default is model-dependent",
    )
    parser.add_argument("--device", type=str, default=None, help='Device like "cpu" or "cuda:0"')
    args = parser.parse_args()

    input_dim, input_res = get_recommended_inputs(
        args.model,
        input_dim=args.input_dim,
        input_res=parse_input_res(args.input_res, model_name=args.model),
    )

    model = build_model_from_name(args.model, input_dim=input_dim, device=args.device)

    if input_dim is not None and args.input_dim is None:
        print(f"Using recommended input_dim for {args.model}: {input_dim}")
    if input_res is not None and args.input_res is None:
        print(f"Using recommended input_res for {args.model}: {input_res}")

    result = measure_model(
        model,
        input_res=input_res,
        compute_flops=args.compute_flops,
        as_strings=False,
        print_per_layer=args.print_per_layer,
        device=args.device,
        model_name=args.model,
        backend=args.backend,
    )

    print(f"Model: {args.model}")
    print(f"Parameters: {format_number(result['params'])}")

    if args.compute_flops:
        if result["flops"] is None:
            print("MACs (ptflops): failed")
            if result["backend"] is not None:
                print(f"Backend: {result['backend']}")
            if result["flops_error"] is not None:
                print(f"Reason: {result['flops_error']}")
        else:
            macs = float(result["flops"])
            gmacs = macs / 1e9
            gflops_approx = gmacs * 2.0

            if result["backend"] is not None:
                print(f"Backend: {result['backend']}")

            if args.for_paper:
                print(f"MACs (ptflops): {format_number(macs)}")
                print(f"GMACs: {gmacs:.3f}")
                print(f"GFLOPs (2x MACs convention): {gflops_approx:.3f}")
            else:
                print(f"MACs (raw): {format_number(macs)}")


if __name__ == "__main__":
    main()