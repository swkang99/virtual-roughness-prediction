import torch
import torch.nn as nn
import torch.nn.functional as F


class HeightTransformerRegressor(nn.Module):
    """
    Height-only Transformer-based roughness regressor.

    Input:
        height_img: [B, 1, H, W] or [B, 3, H, W]

    Output:
        roughness: [B, 1]

    Default setting assumes 448x448 input:
        CNN stem output      : [B, D, 56, 56]
        Patch embedding      : [B, D, 14, 14]
        Transformer tokens   : [B, 196, D]
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

        # --------------------------------------------------
        # 1. CNN stem
        # [B, 1, 448, 448] -> [B, D, 56, 56]
        # --------------------------------------------------
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

        # --------------------------------------------------
        # 2. Patch embedding
        # [B, D, 56, 56] -> [B, D, 14, 14]
        # 14 x 14 = 196 tokens
        # --------------------------------------------------
        self.patch_embed = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=4,
            stride=4,
            padding=0,
        )

        # 448 기준 token 개수: 14 x 14 = 196
        num_tokens = (image_size // 8 // 4) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        # --------------------------------------------------
        # 3. Transformer encoder
        # --------------------------------------------------
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

        # --------------------------------------------------
        # 4. CNN aggregation head
        # [B, D, 14, 14] -> [B, D/2, 14, 14]
        # --------------------------------------------------
        self.cnn_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),

            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
        )

        # --------------------------------------------------
        # 5. Global pooling + MLP regressor
        # --------------------------------------------------
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
        """
        입력이 [B, 3, H, W]이면 grayscale처럼 [B, 1, H, W]로 변환.
        입력이 이미 [B, 1, H, W]이면 그대로 사용.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], but got shape {x.shape}")

        if x.size(1) == 1:
            return x

        if x.size(1) == 3:
            return x.mean(dim=1, keepdim=True)

        raise ValueError(f"Expected channel size 1 or 3, but got {x.size(1)}")

    def _get_pos_embed(self, h, w):
        """
        기본 448 입력에서는 pos_embed가 [1, 196, D] 그대로 사용됨.
        입력 크기가 달라져 token grid가 바뀌면 positional embedding을 보간.
        """
        n = h * w

        if n == self.pos_embed.size(1):
            return self.pos_embed

        old_n = self.pos_embed.size(1)
        old_hw = int(old_n ** 0.5)

        pos = self.pos_embed.transpose(1, 2).reshape(1, self.embed_dim, old_hw, old_hw)
        pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
        pos = pos.flatten(2).transpose(1, 2)

        return pos

    def forward(self, height_img, normal_map=None, height_map=None):
        """
        기본 사용:
            out = model(height_img)

        기존 3-input 코드와 호환하고 싶을 때:
            out = model(texture_img, normal_map, height_map)

        이 경우 texture_img와 normal_map은 무시하고 height_map만 사용함.
        """

        # 기존 MultiBackBoneRegressor 호출 방식과의 호환:
        # model(texture_img, normal_map, height_map)
        if height_map is not None:
            x = height_map
        else:
            x = height_img

        x = self._to_height_input(x)

        # CNN stem
        x = self.cnn_stem(x)          # [B, D, 56, 56]

        # Patch embedding
        x = self.patch_embed(x)       # [B, D, 14, 14]
        b, c, h, w = x.shape

        # Flatten to tokens
        tokens = x.flatten(2).transpose(1, 2)   # [B, 196, D]

        # Positional embedding
        pos_embed = self._get_pos_embed(h, w)
        tokens = tokens + pos_embed

        # Transformer
        tokens = self.transformer(tokens)       # [B, 196, D]
        tokens = self.norm(tokens)

        # Reshape tokens back to 2D feature map
        x = tokens.transpose(1, 2).reshape(b, c, h, w)  # [B, D, 14, 14]

        # CNN aggregation
        x = self.cnn_head(x)       # [B, D/2, 14, 14]

        # Global pooling + regression
        x = self.global_pool(x)    # [B, D/2, 1, 1]
        out = self.regressor(x)    # [B, 1]

        # 필요하면 0~100 범위로 강제
        if self.bounded_output:
            out = torch.sigmoid(out) * self.output_scale

        return out


class MultiBackBoneRegressor(HeightTransformerRegressor):
    """
    기존 train.py와의 이름 호환을 위한 wrapper.

    기존 코드:
        model = MultiBackBoneRegressor("resnet50")

    새 모델에서는 "resnet50" 같은 model_name은 사용하지 않음.
    대신 height-only transformer regressor를 생성함.
    """

    def __init__(
        self,
        model_name=None,
        image_size=448,
        embed_dim=128,
        num_heads=4,
        depth=4,
        mlp_ratio=4.0,
        dropout=0.1,
        bounded_output=False,
        output_scale=100.0,
    ):
        super().__init__(
            image_size=image_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            bounded_output=bounded_output,
            output_scale=output_scale,
        )

        self.model_name = model_name


class SingleBackBoneRegressor(HeightTransformerRegressor):
    """
    혹시 기존 코드에서 SingleBackBoneRegressor를 import하는 경우를 위한 호환용 alias.
    """

    def __init__(
        self,
        model_name=None,
        image_size=448,
        embed_dim=128,
        num_heads=4,
        depth=4,
        mlp_ratio=4.0,
        dropout=0.1,
        bounded_output=False,
        output_scale=100.0,
    ):
        super().__init__(
            image_size=image_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            bounded_output=bounded_output,
            output_scale=output_scale,
        )

        self.model_name = model_name