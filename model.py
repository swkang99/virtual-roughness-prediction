import torch
import torch.nn as nn
import timm

class MultiBackBoneRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.backbone_texture = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone_normal = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone_height = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # 특징 차원 동적 계산
        with torch.no_grad():
            dummy = torch.randn(1, 3, 256, 256)
            self.feat_dim = self.backbone_texture(dummy).shape[1]
        
        self.regressor = nn.Sequential(
            nn.Linear(self.feat_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)
        )

    def forward(self, texture_img, normal_map, height_map):
        texture_feature = self.backbone_texture(texture_img)
        normal_feature = self.backbone_normal(normal_map)
        height_feature = self.backbone_height(height_map)

        concat = torch.cat([texture_feature, normal_feature, height_feature], dim=1)
        return self.regressor(concat).squeeze(1)  # [batch]


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
            nn.Linear(256, 4)
        )