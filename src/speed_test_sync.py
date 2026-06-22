import time
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.dataframe import build_dataframe_from_file
from src.data.factory import build_base_dataset
from src.model.factory import create_model


def run_inference_speed_test(conf):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    full_df = build_dataframe_from_file(conf)

    target_col = (
        "haptic_attribute" if conf["dataset_output"] == "four_HAs"
        else "roughness"
    )

    base_dataset, full_targets, input_dim = build_base_dataset(
        conf, full_df, target_col, device
    )

    batch_size = int(conf.get("test_batch_size", 1))
    warmup_iters = int(conf.get("warmup_iters", 10))

    test_loader = DataLoader(
        base_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = create_model(conf, input_dim=input_dim, device=device)
    model.eval()

    print(f"Number of samples: {len(base_dataset)}")
    print(f"Inference batch size: {batch_size}")
    print(f"Warmup iterations: {warmup_iters}")

    # -------------------------
    # Warmup
    # -------------------------
    with torch.inference_mode():
        for i, batch in enumerate(test_loader):
            if i >= warmup_iters:
                break

            *inputs, y = batch
            inputs = [x.to(device, non_blocking=True) for x in inputs]

            if len(inputs) == 1:
                _ = model(inputs[0])
            else:
                _ = model(*inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # -------------------------
    # Synchronous GPU inference timing
    # -------------------------
    total_time = 0.0
    total_samples = 0
    total_batches = 0

    with torch.inference_mode():
        for batch in test_loader:
            *inputs, y = batch
            inputs = [x.to(device, non_blocking=True) for x in inputs]

            # 입력 GPU 전송이 끝난 뒤부터 순수 forward 시간 측정
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            if len(inputs) == 1:
                pred = model(inputs[0])
            else:
                pred = model(*inputs)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            elapsed = end - start
            current_batch_size = inputs[0].size(0)

            total_time += elapsed
            total_samples += current_batch_size
            total_batches += 1

    avg_time_per_batch = total_time / total_batches
    avg_time_per_sample = total_time / total_samples
    fps = total_samples / total_time

    print("\n========== Inference Speed ==========")
    print(f"Total inference time     : {total_time:.6f} sec")
    print(f"Total samples            : {total_samples}")
    print(f"Total batches            : {total_batches}")
    print(f"Avg time / batch         : {avg_time_per_batch * 1000:.4f} ms")
    print(f"Avg time / sample        : {avg_time_per_sample * 1000:.4f} ms")
    print(f"Throughput               : {fps:.2f} samples/sec")
    print("=====================================")


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    run_inference_speed_test(conf)


if __name__ == "__main__":
    main()