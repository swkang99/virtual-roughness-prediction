# measure_inference_time.py
from os import times
import time
import torch
from torch import nn
import timm

from data import CustomRegressionDataset   # 실제 경로에 맞게 수정
from data import build_dataframe          # 위에서 작성한 함수

from torchvision import transforms

from model import MultiBackBoneRegressor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def measure_model_time(model, sample, n_warmup=5, n_iter=20):
    """단일 샘플에 대한 평균 forward 시간 측정 (model: nn.Module, sample: (texture, normal, height) 또는 tensor)"""
    model.eval()
    model.to(DEVICE)

    # sample을 디바이스로
    if isinstance(sample, tuple):
        sample = tuple(x.unsqueeze(0).to(DEVICE) for x in sample)  # (C,H,W) -> (1,C,H,W)
    else:
        sample = sample.unsqueeze(0).to(DEVICE)

    # warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            if isinstance(sample, tuple):
                _ = model(*sample)
            else:
                _ = model(sample)

    times = []
    with torch.no_grad():
        for _ in range(n_iter):
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            if isinstance(sample, tuple):
                _ = model(*sample)
            else:
                _ = model(sample)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    if len(times) > n_warmup:
        valid_times = times[n_warmup:]
    else:
        valid_times = times
    avg = sum(valid_times) / len(valid_times)
    print(f"평균 forward 시간 ({n_iter - n_warmup}회, device={DEVICE}): {avg*1000:.3f} ms")


def main():
    model_name = "eva02_large_patch14_448.mim_in22k_ft_in1k"  # 실제 사용하는 timm 모델명으로 변경

    train_df, _ = build_dataframe(base_dir="data")
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    train_dataset = CustomRegressionDataset(train_df, transform=transform)

    (tex_img, nor_img, hei_img), _ = train_dataset[0]

    print("=== MultiBackBoneRegressor ===")
    multi_model = MultiBackBoneRegressor(model_name)
    measure_model_time(multi_model, (tex_img, nor_img, hei_img))


if __name__ == "__main__":
    main()
