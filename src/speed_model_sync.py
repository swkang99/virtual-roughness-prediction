import time
import math
import yaml
import torch

from src.model.factory import create_model


def run_inference_speed_test(conf):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    batch_size = int(conf.get("test_batch_size", 1))
    warmup_iters = int(conf.get("warmup_iters", 10))

    # config.yaml의 train_tag 출력
    train_tag = conf.get("train_tag", "unknown")

    # 외부 dataset을 쓰지 않으므로 속도 측정용 샘플 개수를 config에서 가져오거나 기본값 사용
    num_samples = int(conf.get("speed_test_num_samples", 100))
    total_batches = math.ceil(num_samples / batch_size)

    # create_model에서 input_dim 인자가 필요하므로 더미 값 사용
    # TransformerRegressor 계열에서는 보통 input_dim이 직접 사용되지 않음
    input_dim = int(conf.get("input_dim", 5))

    model = create_model(conf, input_dim=input_dim, device=device)
    model.eval()

    # -------------------------
    # Dummy input images
    # -------------------------
    # texture image : [B, 1, 256, 256]
    # height map    : [B, 1, 256, 256]
    # normal map    : [B, 3, 256, 256]
    dummy_texture = torch.rand(batch_size, 1, 256, 256, device=device)
    dummy_height = torch.rand(batch_size, 1, 256, 256, device=device)
    dummy_normal = torch.rand(batch_size, 3, 256, 256, device=device)

    print(f"Train tag: {train_tag}")
    print(f"Number of dummy samples: {num_samples}")
    print(f"Inference batch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Dummy texture shape: {tuple(dummy_texture.shape)}")
    print(f"Dummy height shape : {tuple(dummy_height.shape)}")
    print(f"Dummy normal shape : {tuple(dummy_normal.shape)}")

    # -------------------------
    # Warmup
    # -------------------------
    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = model(dummy_texture, dummy_height, dummy_normal)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # -------------------------
    # Synchronous GPU inference timing
    # -------------------------
    total_time = 0.0
    measured_samples = 0
    measured_batches = 0

    with torch.inference_mode():
        for batch_idx in range(total_batches):
            remaining = num_samples - measured_samples
            current_batch_size = min(batch_size, remaining)

            # 마지막 batch가 batch_size보다 작을 수 있으므로 slicing
            texture = dummy_texture[:current_batch_size]
            height = dummy_height[:current_batch_size]
            normal = dummy_normal[:current_batch_size]

            # 입력은 이미 GPU 위에 있으므로, 여기서는 순수 forward 시간만 측정
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            pred = model(texture, height, normal)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            elapsed = end - start

            total_time += elapsed
            measured_samples += current_batch_size
            measured_batches += 1

    avg_time_per_batch = total_time / measured_batches
    avg_time_per_sample = total_time / measured_samples
    fps = measured_samples / total_time

    print("\n========== Inference Speed ==========")
    print(f"Train tag                : {train_tag}")
    print(f"Total inference time     : {total_time:.6f} sec")
    print(f"Total samples            : {measured_samples}")
    print(f"Total batches            : {measured_batches}")
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