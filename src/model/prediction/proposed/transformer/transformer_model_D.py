import torch
import torch.nn as nn


class TransformerRegressor(nn.Module):  # Total trainable parameters = 118,689
    """
    Texture gray + height gray + normal local-window Transformer-based roughness regressor
    with learnable local roughness token.

    Input:
        height_img1, height_img2, height_img3:
            Existing interface is kept for compatibility.

        height_img1 is assumed to be texture image.
        height_img2 is assumed to be height map.
        height_img3 is assumed to be normal map.

    Internal input feature:
        texture_gray : [B, 1, H, W]
        height_gray  : [B, 1, H, W]
        normal       : [B, 3, H, W]

        concat -> [B, 5, H, W]

    Output:
        roughness: [B, 1]

    Main structure:
        feature [B, 5, 256, 256]
        -> 16x16 window partition
        -> each window: 1x1 pixel tokens
        -> 256 pixel tokens per window
        -> prepend learnable local roughness token
        -> 257 tokens per window
        -> local self-attention
        -> take local roughness token output as window descriptor
        -> [B, D, 16, 16]
        -> CNN head
        -> global pooling with avg + max + std
        -> MLP
        -> [B, 1]
    """

    def __init__(
        self,
        image_size=256,
        embed_dim=64,
        num_heads=4,
        depth=1,
        mlp_ratio=2.0,
        dropout=0.1,
        bounded_output=False,
        output_scale=100.0,
        window_size=16,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.bounded_output = bounded_output
        self.output_scale = output_scale
        self.window_size = window_size

        # texture_gray, height_gray, normal_xyz
        # 1 + 1 + 3 = 5 channels
        self.input_channels = 5

        if image_size % window_size != 0:
            raise ValueError(
                f"image_size must be divisible by window_size, "
                f"but got image_size={image_size}, window_size={window_size}"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, "
                f"but got embed_dim={embed_dim}, num_heads={num_heads}"
            )

        # Since each pixel inside a window is one token:
        # For window_size = 16:
        # tokens_per_window = 16 * 16 = 256 pixel tokens
        self.tokens_per_window = window_size ** 2

        # Each pixel token has 5 channels:
        # [texture_gray, height_gray, normal_x, normal_y, normal_z]
        self.token_embed = nn.Sequential(
            nn.Linear(self.input_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Positional embedding for 256 pixel tokens inside each local window.
        self.local_pos_embed = nn.Parameter(
            torch.zeros(1, self.tokens_per_window, embed_dim)
        )

        # Learnable local roughness token.
        # This token is prepended to the 256 pixel tokens in every local window.
        # After local Transformer, this token output becomes the window descriptor.
        self.local_roughness_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
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

        self.cnn_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),

            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
        )

        # Global pooling uses avg + max + std.
        # CNN head output channel is embed_dim // 2.
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
        nn.init.trunc_normal_(self.local_roughness_token, std=0.02)

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

    def _to_grayscale_input(self, x):
        """
        Convert texture or height input to [B, 1, H, W].

        If input has 1 channel, keep it.
        If input has 3 channels, average it to grayscale.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], but got shape {x.shape}")

        x = x.float()

        if x.size(1) == 1:
            return x

        if x.size(1) == 3:
            return x.mean(dim=1, keepdim=True)

        raise ValueError(f"Expected channel size 1 or 3, but got {x.size(1)}")

    def _to_normal_input(self, x):
        """
        Convert normal input to [B, 3, H, W].

        Assumption:
            input normal map is already in [0, 1].

        Important:
            If the input has 3 channels, they are preserved.
            Normal x/y/z channels are not averaged.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], but got shape {x.shape}")

        x = x.float()

        if x.size(1) == 1:
            # fallback for grayscale normal-like input
            x = x.repeat(1, 3, 1, 1)

        if x.size(1) != 3:
            raise ValueError(f"Expected normal channel size 1 or 3, but got {x.size(1)}")

        return x

    def _build_5ch_feature_input(self, texture_img, height_img, normal_img):
        """
        Build final input feature.

        texture_img:
            [B, 3, H, W] or [B, 1, H, W]
            -> grayscale [B, 1, H, W]

        height_img:
            [B, 3, H, W] or [B, 1, H, W]
            -> grayscale [B, 1, H, W]

        normal_img:
            [B, 3, H, W] or [B, 1, H, W]
            -> normal [B, 3, H, W]

        Output:
            [B, 5, H, W]
            = concat(texture_gray, height_gray, normal_xyz)
        """
        texture_gray = self._to_grayscale_input(texture_img)
        height_gray = self._to_grayscale_input(height_img)
        normal = self._to_normal_input(normal_img)

        x = torch.cat([texture_gray, height_gray, normal], dim=1)

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

        # [B, 5, H, W]
        # For 256x256 and window_size=16:
        # -> [B, 5, 16, 16, 16, 16]
        x = x.reshape(b, c, grid_h, ws, grid_w, ws)

        # -> [B, grid_h, grid_w, 16, 16, 5]
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()

        # -> [B, grid_h * grid_w, 256, 5]
        x = x.reshape(
            b,
            grid_h * grid_w,
            self.tokens_per_window,
            c,
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

        # Add positional embedding only to pixel tokens.
        # tokens: [B * num_windows, 256, D]
        tokens = tokens + self.local_pos_embed

        # Prepend learnable local roughness token to every local window.
        # roughness_tokens: [B * num_windows, 1, D]
        roughness_tokens = self.local_roughness_token.expand(
            tokens.size(0),
            -1,
            -1,
        )

        # [B * num_windows, 257, D]
        tokens = torch.cat([roughness_tokens, tokens], dim=1)

        # Local self-attention:
        # roughness token attends to 256 pixel tokens,
        # and its output becomes a learned window descriptor.
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # Take only local roughness token output.
        # [B * num_windows, D]
        window_features = tokens[:, 0]

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
        # height_img1: texture image
        # height_img2: height map
        # height_img3: normal map
        x = self._build_5ch_feature_input(height_img1, height_img2, height_img3)

        # [B, 5, 256, 256]
        # -> local roughness-token-based window features
        # -> [B, D, 16, 16]
        x = self._encode_local_windows(x)

        # [B, D, 16, 16] -> [B, D/2, 16, 16]
        x = self.cnn_head(x)

        # [B, D/2, 16, 16] -> [B, 3 * D/2]
        x = self._global_pool_features(x)

        # [B, 3 * D/2] -> [B, 1]
        out = self.regressor(x)

        if self.bounded_output:
            out = torch.sigmoid(out) * self.output_scale

        return out