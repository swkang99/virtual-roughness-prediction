import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerRegressor(nn.Module):
    """
    3-image Transformer-based roughness regressor.

    Input:
        texture_image, height_map, normal_map: [B, 1, H, W] or [B, 3, H, W]

    Output:
        roughness: [B, 1]
    """

    def __init__(
        self,
        image_size=448,
        embed_dim=128,
        num_heads=4,
        depth=4,
        mlp_ratio=4.0,
        dropout=0.1,
        bounded_output=False,
        output_scale=100.0,
    ):
        super().__init__()

        self.image_size = image_size
        self.embed_dim = embed_dim
        self.bounded_output = bounded_output
        self.output_scale = output_scale

        # 1. Shared CNN stem
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # 448 -> 224
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),   # 224 -> 112
            nn.BatchNorm2d(96),
            nn.GELU(),

            nn.Conv2d(96, embed_dim, kernel_size=3, stride=2, padding=1),  # 112 -> 56
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # 2. Patch embedding
        self.patch_embed = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=4,
            stride=4,
            padding=0,
        )

        num_tokens = (image_size // 8 // 4) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(embed_dim)

        # 4. Fusion CNN head
        # 3개 이미지 feature를 concat하므로 입력 채널 = embed_dim * 3
        self.cnn_head = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),

            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
        )

        # 5. Global pooling + regressor
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim // 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _to_height_input(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], but got shape {x.shape}")

        if x.size(1) == 1:
            return x

        if x.size(1) == 3:
            return x.mean(dim=1, keepdim=True)

        raise ValueError(f"Expected channel size 1 or 3, but got {x.size(1)}")

    def _get_pos_embed(self, h, w):
        n = h * w

        if n == self.pos_embed.size(1):
            return self.pos_embed

        old_n = self.pos_embed.size(1)
        old_hw = int(old_n ** 0.5)

        pos = self.pos_embed.transpose(1, 2).reshape(1, self.embed_dim, old_hw, old_hw)
        pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
        pos = pos.flatten(2).transpose(1, 2)

        return pos

    def _encode_one_image(self, x):
        x = self._to_height_input(x)

        x = self.cnn_stem(x)            # [B, D, 56, 56]
        x = self.patch_embed(x)         # [B, D, 14, 14]
        b, c, h, w = x.shape

        tokens = x.flatten(2).transpose(1, 2)   # [B, N, D]
        pos_embed = self._get_pos_embed(h, w)
        tokens = tokens + pos_embed

        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        x = tokens.transpose(1, 2).reshape(b, c, h, w)  # [B, D, h, w]
        return x

    def forward(self, height_img1, height_img2, height_img3):
        x1 = self._encode_one_image(height_img1)   # [B, D, h, w]
        x2 = self._encode_one_image(height_img2)   # [B, D, h, w]
        x3 = self._encode_one_image(height_img3)   # [B, D, h, w]

        # feature fusion
        x = torch.cat([x1, x2, x3], dim=1)         # [B, 3D, h, w]

        x = self.cnn_head(x)                       # [B, D/2, h, w]
        x = self.global_pool(x)                    # [B, D/2, 1, 1]
        out = self.regressor(x)                    # [B, 1]

        if self.bounded_output:
            out = torch.sigmoid(out) * self.output_scale

        return out