from tracemalloc import start

import torch
import torch.nn as nn
import timm
import time

class MultiBackBoneRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.backbone_texture = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone_normal = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone_height = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # 특징 차원 동적 계산
        with torch.no_grad():
            dummy = torch.randn(1, 3, 448, 448)
            self.feat_dim = self.backbone_texture(dummy).shape[1]
        
        self.regressor = nn.Sequential(
            nn.Linear(self.feat_dim * 3, 448),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(448, 1)
        )

    def _time_block(self, fn, *args, **kwargs):
        """GPU면 torch.cuda.Event, 아니면 perf_counter로 ms 단위 시간 반환"""
        if torch.cuda.is_available():
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            out = fn(*args, **kwargs)
            ender.record()
            torch.cuda.synchronize()
            elapsed_ms = starter.elapsed_time(ender)  # float(ms)
        else:
            start = time.perf_counter()
            out = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        return out, elapsed_ms

    def forward(self, texture_img, normal_map, height_map):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_start = time.perf_counter()
        
        texture_feature, t_tex = self._time_block(self.backbone_texture, texture_img)
        print(f"texture feature extraction 시간: {t_tex:.3f} ms")

        normal_feature, t_nor = self._time_block(self.backbone_normal, normal_map)
        print(f"normal feature extraction 시간: {t_nor:.3f} ms")

        height_feature, t_hei = self._time_block(self.backbone_height, height_map)
        print(f"height feature extraction 시간: {t_hei:.3f} ms")

        def reg_fn(x):
            return self.regressor(x)

        concat = torch.cat([texture_feature, normal_feature, height_feature], dim=1)
        out, t_reg = self._time_block(reg_fn, concat)
        print(f"regressor 시간: {t_reg:.3f} ms")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_end = time.perf_counter()
        print(f"total forward 시간: {(total_end - total_start)*1000:.3f} ms")

        return out  # [batch, 1]


class SingleBackBoneRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        # 특징 차원 계산 (예시: resnet18은 512)
        example = torch.randn(1, 3, 256, 256)
        feat_dim = self.backbone(example).shape[1]

        self.regressor = nn.Sequential(
            nn.Linear(feat_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )