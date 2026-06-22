import torch
import torch.nn as nn


class TransformerRegressor(nn.Module):
    """
    Normal + normal gradient local-window Transformer-based roughness regressor.

    Input:
        height_img1, height_img2, height_img3:
            Existing interface is kept for compatibility.

        Only height_img3 is used.
        height_img3 is assumed to be the normal image.

    Internal input feature:
        normal      : [B, 3, H, W]
        normal_dx   : [B, 3, H, W]
        normal_dy   : [B, 3, H, W]

        concat -> [B, 9, H, W]

    Output:
        roughness: [B, 1]

    Main structure:
        normal feature [B, 9, 448, 448]
        -> 16x16 window partition
        -> each window: 1x1 pixel tokens
        -> 256 tokens per window
        -> local self-attention
        -> window pooling with mean + max + std
        -> [B, D, 28, 28]
        -> CNN head
        -> global pooling with avg + max + std
        -> MLP
        -> [B, 1]
    """

    def __init__(
        self,
        image_size=448,  # kept for compatibility with existing config/model factory
        embed_dim=64,  # 128
        num_heads=4,
        depth=1,  # 4
        mlp_ratio=2.0,  # 4.0
        dropout=0.1,
        bounded_output=False,
        output_scale=100.0,
        window_size=16,  # 32
        subpatch_size=1,  # 2
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.bounded_output = bounded_output
        self.output_scale = output_scale

        self.window_size = window_size
        self.subpatch_size = subpatch_size

        # normal, normal_dx, normal_dy
        # each has 3 channels -> total 9 channels
        self.input_channels = 9

        if image_size % window_size != 0:
            raise ValueError(
                f"image_size must be divisible by window_size, "
                f"but got image_size={image_size}, window_size={window_size}"
            )

        if window_size % subpatch_size != 0:
            raise ValueError(
                f"window_size must be divisible by subpatch_size, "
                f"but got window_size={window_size}, subpatch_size={subpatch_size}"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, "
                f"but got embed_dim={embed_dim}, num_heads={num_heads}"
            )

        self.tokens_per_side = window_size // subpatch_size
        self.tokens_per_window = self.tokens_per_side ** 2

        # For subpatch_size = 1:
        # token_input_dim = 9 * 1 * 1 = 9
        token_input_dim = self.input_channels * subpatch_size * subpatch_size

        self.token_embed = nn.Sequential(
            nn.Linear(token_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.local_pos_embed = nn.Parameter(
            torch.zeros(1, self.tokens_per_window, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(embed_dim)

        # New:
        # window token pooling uses mean + max + std.
        # [B*num_windows, 3D] -> [B*num_windows, D]
        self.window_pool_proj = nn.Linear(embed_dim * 3, embed_dim)

        self.cnn_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),

            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
        )

        # New:
        # global pooling uses avg + max + std.
        # CNN head output channel is embed_dim // 2.
        # Therefore regressor input dim is 3 * (embed_dim // 2).
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear((embed_dim // 2) * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.local_pos_embed, std=0.02)

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

    def _to_normal_input(self, x):
        """
        Convert normal input to unit normal vector [B, 3, H, W].

        Assumption:
            input normal map is in [0, 1].

        Process:
            [0, 1] -> [-1, 1]
            then unit-vector normalization.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], but got shape {x.shape}")

        if x.size(1) == 1:
            # fallback for grayscale normal-like input
            x = x.repeat(1, 3, 1, 1)

        if x.size(1) != 3:
            raise ValueError(f"Expected normal channel size 1 or 3, but got {x.size(1)}")

        x = x.float()

        # Decode normal map from [0, 1] to [-1, 1]
        x = x * 2.0 - 1.0

        # Normalize as unit normal vector
        x = x / (x.norm(dim=1, keepdim=True) + 1e-6)

        return x

    def _normal_gradients(self, normal):
        """
        Compute normal_dx and normal_dy.

        normal:
            [B, 3, H, W]

        normal_dx:
            difference along width direction
            normal[:, :, :, x+1] - normal[:, :, :, x]

        normal_dy:
            difference along height direction
            normal[:, :, y+1, :] - normal[:, :, y, :]

        Output:
            normal_dx, normal_dy: each [B, 3, H, W]
        """
        # dx: horizontal difference
        dx = torch.cat(
            [
                normal[:, :, :, 1:] - normal[:, :, :, :-1],
                torch.zeros_like(normal[:, :, :, :1]),
            ],
            dim=3,
        )

        # dy: vertical difference
        dy = torch.cat(
            [
                normal[:, :, 1:, :] - normal[:, :, :-1, :],
                torch.zeros_like(normal[:, :, :1, :]),
            ],
            dim=2,
        )

        return dx, dy

    def _build_normal_feature_input(self, normal_img):
        """
        Build final input feature from normal image.

        normal_img:
            [B, 3, H, W] or [B, 1, H, W]

        Output:
            [B, 9, H, W]
            = concat(normal, normal_dx, normal_dy)
        """
        normal = self._to_normal_input(normal_img)
        normal_dx, normal_dy = self._normal_gradients(normal)

        x = torch.cat([normal, normal_dx, normal_dy], dim=1)

        return x

    def _partition_windows_to_tokens(self, x):
        b, c, h, w = x.shape

        if c != self.input_channels:
            raise ValueError(
                f"Expected input with {self.input_channels} channels, but got {c}"
            )

        if h % self.window_size != 0 or w % self.window_size != 0:
            raise ValueError(
                f"Input H and W must be divisible by window_size={self.window_size}, "
                f"but got H={h}, W={w}"
            )

        grid_h = h // self.window_size
        grid_w = w // self.window_size

        ws = self.window_size
        sp = self.subpatch_size
        tps = self.tokens_per_side

        # [B, 9, H, W]
        # -> [B, 9, grid_h, 16, grid_w, 16]
        x = x.reshape(b, c, grid_h, ws, grid_w, ws)

        # -> [B, grid_h, grid_w, 9, 16, 16]
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()

        # For subpatch_size = 1:
        # -> [B, grid_h, grid_w, 9, 16, 1, 16, 1]
        x = x.reshape(b, grid_h, grid_w, c, tps, sp, tps, sp)

        # -> [B, grid_h, grid_w, 16, 16, 9, 1, 1]
        x = x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()

        # -> [B, grid_h * grid_w, 256, 9]
        x = x.reshape(
            b,
            grid_h * grid_w,
            self.tokens_per_window,
            c * sp * sp,
        )

        # -> [B, grid_h * grid_w, 256, D]
        x = self.token_embed(x)

        # -> [B * grid_h * grid_w, 256, D]
        x = x.reshape(
            b * grid_h * grid_w,
            self.tokens_per_window,
            self.embed_dim,
        )

        return x, grid_h, grid_w

    def _encode_local_windows(self, x):
        b = x.size(0)

        tokens, grid_h, grid_w = self._partition_windows_to_tokens(x)

        tokens = tokens + self.local_pos_embed

        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # New window pooling:
        # [B * num_windows, tokens_per_window, D]
        # -> mean/max/std each [B * num_windows, D]
        mean_feat = tokens.mean(dim=1)
        max_feat = tokens.max(dim=1).values
        std_feat = tokens.std(dim=1, unbiased=False)

        # -> [B * num_windows, 3D]
        window_features = torch.cat([mean_feat, max_feat, std_feat], dim=1)

        # -> [B * num_windows, D]
        window_features = self.window_pool_proj(window_features)

        # -> [B, num_windows, D]
        window_features = window_features.reshape(
            b,
            grid_h * grid_w,
            self.embed_dim,
        )

        # -> [B, D, grid_h, grid_w]
        feature_map = window_features.transpose(1, 2).reshape(
            b,
            self.embed_dim,
            grid_h,
            grid_w,
        )

        return feature_map

    def _global_pool_features(self, x):
        """
        Global pooling with avg + max + std.

        x:
            [B, C, H, W]

        Output:
            [B, 3C]
        """
        avg_feat = x.mean(dim=(2, 3))
        max_feat = x.amax(dim=(2, 3))
        std_feat = x.std(dim=(2, 3), unbiased=False)

        x = torch.cat([avg_feat, max_feat, std_feat], dim=1)

        return x

    def forward(self, height_img1, height_img2, height_img3):
        # height_img1 and height_img2 are intentionally ignored.
        # Only height_img3 is used as the normal image.
        x = self._build_normal_feature_input(height_img3)

        # [B, 9, 448, 448] -> [B, D, 28, 28]
        x = self._encode_local_windows(x)

        # [B, D, 28, 28] -> [B, D/2, 28, 28]
        x = self.cnn_head(x)

        # [B, D/2, 28, 28] -> [B, 3 * D/2]
        x = self._global_pool_features(x)

        # [B, 3 * D/2] -> [B, 1]
        out = self.regressor(x)

        if self.bounded_output:
            out = torch.sigmoid(out) * self.output_scale

        return out