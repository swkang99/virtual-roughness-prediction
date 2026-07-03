import torch
import torch.nn as nn


class TransformerRegressor(nn.Module):  # Total trainable parameters ~= 109,057
    """
    Hierarchical Local-Global Roughness Token Transformer.

    Texture gray + height gray + normal local/global Transformer-based roughness regressor
    without mean/max/std pooling.

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
        -> token_embed: Linear 5 -> D
        -> prepend learnable local roughness token
        -> local Transformer Encoder
        -> take local roughness token output as window descriptor
        -> [B, 256 windows, D]
        -> add global window positional embedding
        -> prepend learnable global roughness token
        -> global Window Transformer Encoder
        -> take global roughness token output as image descriptor
        -> MLP
        -> [B, 1]

    Important:
        No mean pooling.
        No max pooling.
        No std pooling.
        Local and global roughness information is aggregated by learnable roughness tokens.
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
        global_depth=1,
        global_mlp_ratio=2.0,
    ):
        super().__init__()

        self.image_size = image_size
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

        self.grid_size = image_size // window_size
        self.num_windows = self.grid_size ** 2

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

        # Positional embedding for pixel tokens inside each local window.
        # Shape:
        #   [1, 256, D]
        self.local_pos_embed = nn.Parameter(
            torch.zeros(1, self.tokens_per_window, embed_dim)
        )

        # Learnable local roughness token.
        # This token is prepended to the pixel tokens inside every local window.
        # After local Transformer, this token becomes the window-level roughness descriptor.
        self.local_roughness_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        local_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.local_transformer = nn.TransformerEncoder(
            local_encoder_layer,
            num_layers=depth,
        )

        self.local_norm = nn.LayerNorm(embed_dim)

        # Positional embedding for window-level descriptors.
        # For image_size=256 and window_size=16:
        #   num_windows = 16 * 16 = 256
        # Shape:
        #   [1, 256, D]
        self.global_window_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_windows, embed_dim)
        )

        # Learnable global roughness token.
        # This token is prepended to the 256 window descriptors.
        # After global Transformer, this token becomes the image-level roughness descriptor.
        self.global_roughness_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        global_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * global_mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.global_transformer = nn.TransformerEncoder(
            global_encoder_layer,
            num_layers=global_depth,
        )

        self.global_norm = nn.LayerNorm(embed_dim)

        # Final regression from global roughness descriptor.
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.local_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.global_window_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.local_roughness_token, std=0.02)
        nn.init.trunc_normal_(self.global_roughness_token, std=0.02)

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
            raise ValueError(
                f"Expected 4D tensor [B, C, H, W], but got shape {x.shape}"
            )

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
            raise ValueError(
                f"Expected 4D tensor [B, C, H, W], but got shape {x.shape}"
            )

        x = x.float()

        if x.size(1) == 1:
            # fallback for grayscale normal-like input
            x = x.repeat(1, 3, 1, 1)

        if x.size(1) != 3:
            raise ValueError(
                f"Expected normal channel size 1 or 3, but got {x.size(1)}"
            )

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

        if (
            texture_gray.shape[-2:] != height_gray.shape[-2:]
            or texture_gray.shape[-2:] != normal.shape[-2:]
        ):
            raise ValueError(
                "All inputs must have the same spatial resolution. "
                f"Got texture={tuple(texture_gray.shape[-2:])}, "
                f"height={tuple(height_gray.shape[-2:])}, "
                f"normal={tuple(normal.shape[-2:])}."
            )

        x = torch.cat([texture_gray, height_gray, normal], dim=1)

        return x

    def _partition_windows_to_tokens(self, x):
        b, c, h, w = x.shape

        if c != self.input_channels:
            raise ValueError(
                f"Expected input with {self.input_channels} channels, but got {c}"
            )

        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"This model expects {self.image_size}x{self.image_size} inputs, "
                f"but got H={h}, W={w}."
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
        """
        Encode each local window using local roughness token.

        Input:
            x: [B, 5, 256, 256]

        Output:
            window_features: [B, 256, D]
        """
        b = x.size(0)

        tokens, grid_h, grid_w = self._partition_windows_to_tokens(x)

        # Add local positional embedding only to pixel tokens.
        # tokens: [B * num_windows, 256, D]
        tokens = tokens + self.local_pos_embed

        # Prepend learnable local roughness token to every local window.
        # local_roughness_tokens: [B * num_windows, 1, D]
        local_roughness_tokens = self.local_roughness_token.expand(
            tokens.size(0),
            -1,
            -1,
        )

        # [B * num_windows, 257, D]
        tokens = torch.cat([local_roughness_tokens, tokens], dim=1)

        # Local intra-window self-attention.
        # The local roughness token attends to 256 pixel tokens.
        tokens = self.local_transformer(tokens)
        tokens = self.local_norm(tokens)

        # Take only local roughness token output.
        # [B * num_windows, D]
        window_features = tokens[:, 0]

        # -> [B, num_windows, D]
        window_features = window_features.reshape(
            b,
            grid_h * grid_w,
            self.embed_dim,
        )

        return window_features, grid_h, grid_w

    def _encode_global_windows(self, window_features, grid_h, grid_w):
        """
        Encode all window-level descriptors using global roughness token.

        Input:
            window_features: [B, 256, D]

        Output:
            global_feature: [B, D]
        """
        b, num_windows, d = window_features.shape

        if d != self.embed_dim:
            raise ValueError(
                f"Expected window feature dim {self.embed_dim}, but got {d}"
            )

        if num_windows != self.num_windows:
            raise ValueError(
                f"Expected {self.num_windows} windows from image_size/window_size, "
                f"but got {num_windows}. "
                f"Check image_size, input resolution, and window_size."
            )

        if num_windows != grid_h * grid_w:
            raise ValueError(
                f"num_windows must equal grid_h * grid_w, "
                f"but got num_windows={num_windows}, grid_h={grid_h}, grid_w={grid_w}"
            )

        # Add window-level positional embedding to window descriptors only.
        # [B, 256, D] + [1, 256, D]
        window_features = window_features + self.global_window_pos_embed

        # Prepend learnable global roughness token.
        # global_roughness_tokens: [B, 1, D]
        global_roughness_tokens = self.global_roughness_token.expand(
            b,
            -1,
            -1,
        )

        # [B, 257, D]
        tokens = torch.cat([global_roughness_tokens, window_features], dim=1)

        # Global inter-window self-attention.
        # The global roughness token attends to 256 window descriptors.
        tokens = self.global_transformer(tokens)
        tokens = self.global_norm(tokens)

        # Take only global roughness token output.
        # [B, D]
        global_feature = tokens[:, 0]

        return global_feature

    def forward(self, height_img1, height_img2, height_img3):
        # height_img1: texture image
        # height_img2: height map
        # height_img3: normal map
        x = self._build_5ch_feature_input(
            height_img1,
            height_img2,
            height_img3,
        )

        # [B, 5, 256, 256]
        # -> [B, 256, D]
        window_features, grid_h, grid_w = self._encode_local_windows(x)

        # [B, 256, D]
        # -> [B, D]
        x = self._encode_global_windows(
            window_features,
            grid_h,
            grid_w,
        )

        # [B, D] -> [B, 1]
        out = self.regressor(x)

        if self.bounded_output:
            out = torch.sigmoid(out) * self.output_scale

        return out