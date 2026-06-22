import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerRegressor(nn.Module):
    """
    3-image local-window Transformer-based roughness regressor.

    Input:
        height_img1, height_img2, height_img3:
            each [B, 1, H, W] or [B, 3, H, W]

        Internally:
            three inputs are converted to 1 channel each,
            then concatenated into [B, 3, H, W].

    Output:
        roughness: [B, 1]

    Main structure:
        [B, 3, 448, 448]
        -> 32x32 window partition
        -> each window: 2x2 subpatch tokens
        -> 256 tokens per window
        -> local self-attention
        -> window pooling
        -> [B, D, 14, 14]
        -> CNN head
        -> global pooling
        -> MLP
        -> [B, 1]
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
        window_size=32,
        subpatch_size=2,
    ):
        super().__init__()

        self.image_size = image_size
        self.embed_dim = embed_dim
        self.bounded_output = bounded_output
        self.output_scale = output_scale

        self.window_size = window_size
        self.subpatch_size = subpatch_size

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

        # 32x32 window, 2x2 subpatch:
        # tokens_per_window = (32 / 2)^2 = 16^2 = 256
        self.tokens_per_side = window_size // subpatch_size
        self.tokens_per_window = self.tokens_per_side ** 2

        # Each token contains values from 3 images over a 2x2 subpatch:
        # token_dim = 3 * 2 * 2 = 12
        token_input_dim = 3 * subpatch_size * subpatch_size

        # 1. Token embedding
        self.token_embed = nn.Sequential(
            nn.Linear(token_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # 2. Local positional embedding inside each 32x32 window
        self.local_pos_embed = nn.Parameter(
            torch.zeros(1, self.tokens_per_window, embed_dim)
        )

        # 3. Local Transformer encoder
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

        # 4. CNN head over 14x14 window feature map
        self.cnn_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
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

    def _to_height_input(self, x):
        """
        Convert each input image to a single-channel image.

        If x is [B, 1, H, W], keep it.
        If x is [B, 3, H, W], average channels into [B, 1, H, W].
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], but got shape {x.shape}")

        if x.size(1) == 1:
            return x

        if x.size(1) == 3:
            return x.mean(dim=1, keepdim=True)

        raise ValueError(f"Expected channel size 1 or 3, but got {x.size(1)}")

    def _partition_windows_to_tokens(self, x):
        """
        Input:
            x: [B, 3, H, W]

        Output:
            tokens: [B * num_windows, tokens_per_window, embed_dim]
            grid_h, grid_w: number of windows along height and width

        For H = W = 448, window_size = 32, subpatch_size = 2:
            [B, 3, 448, 448]
            -> [B, 196, 256, 12]
            -> [B * 196, 256, D]
        """
        b, c, h, w = x.shape

        if c != 3:
            raise ValueError(f"Expected concatenated input with 3 channels, but got {c}")

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

        # [B, 3, H, W]
        # -> [B, 3, grid_h, 32, grid_w, 32]
        x = x.reshape(b, c, grid_h, ws, grid_w, ws)

        # -> [B, grid_h, grid_w, 3, 32, 32]
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()

        # Split each 32x32 window into 2x2 subpatches.
        # [B, grid_h, grid_w, 3, 32, 32]
        # -> [B, grid_h, grid_w, 3, 16, 2, 16, 2]
        x = x.reshape(b, grid_h, grid_w, c, tps, sp, tps, sp)

        # Move subpatch grid before channel/subpatch pixels.
        # -> [B, grid_h, grid_w, 16, 16, 3, 2, 2]
        x = x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()

        # Flatten:
        # 16 x 16 = 256 tokens per window
        # 3 x 2 x 2 = 12 values per token
        # -> [B, grid_h * grid_w, 256, 12]
        x = x.reshape(
            b,
            grid_h * grid_w,
            self.tokens_per_window,
            c * sp * sp,
        )

        # Token embedding:
        # [B, num_windows, 256, 12]
        # -> [B, num_windows, 256, D]
        x = self.token_embed(x)

        # Merge batch and window dimensions:
        # -> [B * num_windows, 256, D]
        x = x.reshape(
            b * grid_h * grid_w,
            self.tokens_per_window,
            self.embed_dim,
        )

        return x, grid_h, grid_w

    def _encode_local_windows(self, x):
        """
        Input:
            x: [B, 3, H, W]

        Output:
            feature map: [B, D, grid_h, grid_w]
            For 448x448 input: [B, D, 14, 14]
        """
        b = x.size(0)

        tokens, grid_h, grid_w = self._partition_windows_to_tokens(x)

        # Add local positional embedding.
        tokens = tokens + self.local_pos_embed

        # Local self-attention inside each 32x32 window.
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # Window pooling:
        # [B * num_windows, 256, D]
        # -> [B * num_windows, D]
        window_features = tokens.mean(dim=1)

        # Restore windows:
        # [B * num_windows, D]
        # -> [B, grid_h * grid_w, D]
        window_features = window_features.reshape(
            b,
            grid_h * grid_w,
            self.embed_dim,
        )

        # Restore 2D feature map:
        # [B, 196, D]
        # -> [B, D, 14, 14]
        feature_map = window_features.transpose(1, 2).reshape(
            b,
            self.embed_dim,
            grid_h,
            grid_w,
        )

        return feature_map

    def forward(self, height_img1, height_img2, height_img3):
        # Convert each input to [B, 1, H, W]
        x1 = self._to_height_input(height_img1)
        x2 = self._to_height_input(height_img2)
        x3 = self._to_height_input(height_img3)

        # Early fusion:
        # [B, 1, H, W] x 3 -> [B, 3, H, W]
        x = torch.cat([x1, x2, x3], dim=1)

        # Local window Transformer:
        # [B, 3, 448, 448] -> [B, D, 14, 14]
        x = self._encode_local_windows(x)

        # CNN head:
        # [B, D, 14, 14] -> [B, D/2, 14, 14]
        x = self.cnn_head(x)

        # Global pooling:
        # [B, D/2, 14, 14] -> [B, D/2, 1, 1]
        x = self.global_pool(x)

        # Regression:
        # [B, D/2, 1, 1] -> [B, 1]
        out = self.regressor(x)

        if self.bounded_output:
            out = torch.sigmoid(out) * self.output_scale

        return out