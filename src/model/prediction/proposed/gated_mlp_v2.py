import torch
import torch.nn as nn

class GatedFusionRegressorV2(nn.Module):
    def __init__(self, input_dim, fusion_dim=128, output_dim=1):
        super().__init__()

        texture_dim = input_dim['texture_dim']
        height_dim = input_dim['height_dim']
        normal_dim = input_dim['normal_dim']

        self.texture_proj = nn.Sequential(
            nn.Linear(texture_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
        self.height_proj = nn.Sequential(
            nn.Linear(height_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
        self.normal_proj = nn.Sequential(
            nn.Linear(normal_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )

        self.gate_network = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 3),
            nn.Sigmoid()
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, output_dim)
        )

    def forward(self, texture_feat, height_feat, normal_feat):
        t = self.texture_proj(texture_feat)
        h = self.height_proj(height_feat)
        n = self.normal_proj(normal_feat)

        concat = torch.cat([t, h, n], dim=1)
        gates = self.gate_network(concat)

        gate_t = gates[:, 0:1]
        gate_h = gates[:, 1:2]
        gate_n = gates[:, 2:3]

        t_gated = t * gate_t + t
        h_gated = h * gate_h + h
        n_gated = n * gate_n + n

        fused = torch.cat([t_gated, h_gated, n_gated], dim=1)
        output = self.fusion_mlp(fused)
        return output