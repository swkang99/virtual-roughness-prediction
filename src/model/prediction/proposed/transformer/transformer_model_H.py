import torch
import torch.nn as nn


class ModalityAwareTokenEmbed(nn.Module):
    """
    Modality-aware token embedding with global learnable gates.

    Input token channel order:
        texture_gray : 1 channel
        height_gray  : 1 channel
        normal_xyz   : 3 channels

    Since each pixel is one token, each raw token has:
        [texture_gray, height_gray, normal_x, normal_y, normal_z]

    Input:
        x: [B, num_windows, tokens_per_window, 5]

    Output:
        x: [B, num_windows, tokens_per_window, D]
    """

    def __init__(
        self,
        embed_dim,
        init_gate_logits=(1.0, 1.0, 1.0),
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.texture_embed = nn.Linear(1, embed_dim)
        self.height_embed = nn.Linear(1, embed_dim)
        self.normal_embed = nn.Linear(3, embed_dim)

        # Global learnable modality gates.
        # Gate order:
        #   [texture, height, normal]
        self.gate_logits = nn.Parameter(
            torch.tensor(init_gate_logits, dtype=torch.float32)
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

        self.last_contrib = None

    def forward(self, x):
        """
        x:
            [B, num_windows, tokens_per_window, 5]

        Channel order:
            texture_gray, height_gray, normal_x, normal_y, normal_z
        """
        texture = x[..., 0:1]
        height = x[..., 1:2]
        normal = x[..., 2:5]

        texture_feat = self.texture_embed(texture)
        height_feat = self.height_embed(height)
        normal_feat = self.normal_embed(normal)

        gates = torch.softmax(self.gate_logits, dim=0)

        # Optional diagnostic values.
        # These values are not used for training loss.
        with torch.no_grad():
            texture_contrib = (gates[0] * texture_feat).norm(dim=-1).mean()
            height_contrib = (gates[1] * height_feat).norm(dim=-1).mean()
            normal_contrib = (gates[2] * normal_feat).norm(dim=-1).mean()

            self.last_contrib = {
                "texture": texture_contrib.detach(),
                "height": height_contrib.detach(),
                "normal": normal_contrib.detach(),
            }

        x = (
            gates[0] * texture_feat
            + gates[1] * height_feat
            + gates[2] * normal_feat
        )

        x = self.norm(x)
        x = self.act(x)

        return x

    def get_gates(self):
        """
        Return current modality gates as probabilities:
            [texture_gate, height_gate, normal_gate]
        """
        return torch.softmax(self.gate_logits.detach(), dim=0)


class TransformerRegressor(nn.Module):  # Total trainable parameters ~= 181,092
    """
    Modality-aware local-global Transformer roughness regressor.

    Texture gray + height gray + normal local-window Transformer-based roughness regressor
    with local Transformer + global Window Transformer + modality-aware input gates.

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
        -> modality-aware token embedding with learnable gates
        -> 256 tokens per window
        -> local self-attention
        -> window pooling with mean + max + std
        -> Linear 192 -> 64
        -> [B, 256 windows, 64]
        -> global window positional embedding
        -> global Window Transformer
        -> [B, 256 windows, 64]
        -> reshape to [B, 64, 16, 16]
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
        global_depth=1,
        global_mlp_ratio=2.0,
        init_gate_logits=(1.0, 1.0, 1.0),
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

        # Since each pixel inside a window is one token.
        # For window_size = 16:
        # tokens_per_window = 16 * 16 = 256
        self.tokens_per_window = window_size ** 2

        # Modality-aware token embedding:
        # texture, height, normal are embedded separately and combined by learnable gates.
        self.token_embed = ModalityAwareTokenEmbed(
            embed_dim=embed_dim,
            init_gate_logits=init_gate_logits,
        )

        # Positional embedding for 256 pixel tokens inside each local window.
        self.local_pos_embed = nn.Parameter(
            torch.zeros(1, self.tokens_per_window, embed_dim)
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

        self.transformer = nn.TransformerEncoder(
            local_encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Window token pooling:
        # mean + max + std
        # [B*num_windows, 3D] -> [B*num_windows, D]
        self.window_pool_proj = nn.Linear(embed_dim * 3, embed_dim)

        # Positional embedding for 256 window descriptors.
        # For image_size=256 and window_size=16:
        # num_windows = 16 * 16 = 256
        self.global_window_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_windows, embed_dim)
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
        nn.init.trunc_normal_(self.global_window_pos_embed, std=0.02)

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

    def get_modality_gates(self):
        """
        Return current modality gates as probabilities:
            [texture_gate, height_gate, normal_gate]
        """
        return self.token_embed.get_gates()

    def get_modality_contrib(self):
        """
        Return last diagnostic contribution values.

        This is only for inspection/debugging.
        It is updated during forward().
        """
        return self.token_embed.last_contrib

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

        # Modality-aware token embedding:
        # [B, grid_h * grid_w, 256, 5]
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

        # Local positional embedding.
        # tokens: [B * num_windows, 256, D]
        tokens = tokens + self.local_pos_embed

        # Local intra-window self-attention.
        # Each window is processed independently.
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # Window pooling:
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

        return window_features, grid_h, grid_w

    def _encode_global_windows(self, window_features, grid_h, grid_w):
        """
        Encode relations among window-level descriptors using global Window Transformer.

        Input:
            window_features: [B, num_windows, D]

        Output:
            feature_map: [B, D, grid_h, grid_w]
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

        # Add window-level positional embedding.
        # [B, 256, D] + [1, 256, D]
        window_features = window_features + self.global_window_pos_embed

        # Global inter-window self-attention.
        # This models relations among the 256 window descriptors.
        window_features = self.global_transformer(window_features)
        window_features = self.global_norm(window_features)

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

    def forward(self, texture_image, height_map, normal_map):
        x = self._build_5ch_feature_input(texture_image, height_map, normal_map)

        # [B, 5, 256, 256]
        # -> modality-aware token embedding
        # -> local window descriptors [B, 256, D]
        window_features, grid_h, grid_w = self._encode_local_windows(x)

        # [B, 256, D]
        # -> global window Transformer
        # -> [B, D, 16, 16]
        x = self._encode_global_windows(window_features, grid_h, grid_w)

        # [B, D, 16, 16] -> [B, D/2, 16, 16]
        x = self.cnn_head(x)

        # [B, D/2, 16, 16] -> [B, 3 * D/2]
        x = self._global_pool_features(x)

        # [B, 3 * D/2] -> [B, 1]
        out = self.regressor(x)

        if self.bounded_output:
            out = torch.sigmoid(out) * self.output_scale

        return out