# Implementation of proposed gated fusion MLP
import torch
import torch.nn as nn

class GatedFusionRegressor(nn.Module):
    """
    gated fusion by modality (texture, height, normal)
    """
    def __init__(self, input_dim, fusion_dim=128, output_dim=1):
        super().__init__()
        
        texture_dim = input_dim['texture_dim']
        height_dim = input_dim['height_dim']
        normal_dim = input_dim['normal_dim']
        self.fusion_dim = fusion_dim

        self.texture_proj = nn.Linear(texture_dim, fusion_dim)
        self.height_proj = nn.Linear(height_dim, fusion_dim)
        self.normal_proj = nn.Linear(normal_dim, fusion_dim)
        
        self.gate_network = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 3),
            nn.ReLU(),
            nn.Linear(fusion_dim * 3, fusion_dim * 3),
            nn.Sigmoid()  # 0-1 range
        )
        
        # Fusion MLP
        self.fusion_ml = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, output_dim)
        )
        
    def forward(self, texture_feat, height_feat, normal_feat):
        t = self.texture_proj(texture_feat)
        h = self.height_proj(height_feat)
        n = self.normal_proj(normal_feat)
        
        # Concat for gate
        concat = torch.cat([t, h, n], dim=1)  # (batch, fusion_dim*3)
        
        gates = self.gate_network(concat)  # (batch, fusion_dim*3)
        
        gate_t = gates[:, :self.fusion_dim]
        gate_h = gates[:, self.fusion_dim:2*self.fusion_dim]
        gate_n = gates[:, 2*self.fusion_dim:]
        
        # Gated fusion
        t_gated = t * gate_t
        h_gated = h * gate_h
        n_gated = n * gate_n
        
        fused = torch.cat([t_gated, h_gated, n_gated], dim=1)
        
        # Regression output
        output = self.fusion_ml(fused)
        return output