import torch
import torch.nn as nn


class ModalityAwareTokenEmbed(nn.Module):
    """
    Modality-aware token embedding with global learnable gates.

    Input token channel order:
        texture_gray : 1 channel
        height_gray  : 1 channel
        normal_xyz   : 3 channels

    For subpatch_size = 1:
        input token dim = 5

    For general subpatch_size = s:
        texture token dim = 1 * s * s
        height token dim  = 1 * s * s
        normal token dim  = 3 * s * s
    """

    def __init__(
        self,
        embed_dim,
        subpatch_size=1,
        init_gate_logits=(1.0, 1.0, 1.0),
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.subpatch_size = subpatch_size

        patch_area = subpatch_size * subpatch_size

        self.texture_embed = nn.Linear(1 * patch_area, embed_dim)
        self.height_embed = nn.Linear(1 * patch_area, embed_dim)
        self.normal_embed = nn.Linear(3 * patch_area, embed_dim)

        # Global learnable modality gates:
        # gate order = [texture, height, normal]
        self.gate_logits = nn.Parameter(
            torch.tensor(init_gate_logits, dtype=torch.float32)
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        """
        x:
            [B, num_windows, tokens_per_window, 5 * patch_area]

        Channel order is assumed to be:
            texture_gray, height_gray, normal_x, normal_y, normal_z
        """
        patch_area = self.subpatch_size * self.subpatch_size

        texture = x[..., 0 * patch_area : 1 * patch_area]
        height = x[..., 1 * patch_area : 2 * patch_area]
        normal = x[..., 2 * patch_area : 5 * patch_area]

        texture_feat = self.texture_embed(texture)
        height_feat = self.height_embed(height)
        normal_feat = self.normal_embed(normal)

        gates = torch.softmax(self.gate_logits, dim=0)

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


class TransformerRegressor(nn.Module):
    """
    Texture gray + height gray + normal local-window Transformer-based roughness regressor.

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
        feature [B, 5, 448, 448]
        -> 16x16 window partition
        -> each window: 1x1 pixel tokens
        -> modality-aware token embedding with learnable global gates
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
        image_size=448,
        embed_dim=64,
        num_heads=4,
        depth=1,
        mlp_ratio=2.0,
        dropout=0.1,
        bounded_output=False,
        output_scale=100.0,
        window_size=16,
        subpatch_size=1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.bounded_output = bounded_output
        self.output_scale = output_scale

        self.window_size = window_size
        self.subpatch_size = subpatch_size

        # texture_gray, height_gray, normal_xyz
        # 1 + 1 + 3 = 5 channels
        self.input_channels = 5

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

        self.token_embed = ModalityAwareTokenEmbed(
            embed_dim=embed_dim,
            subpatch_size=subpatch_size,
            init_gate_logits=(1.0, 1.0, 1.0),
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

        # Window token pooling:
        # mean + max + std
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
        sp = self.subpatch_size
        tps = self.tokens_per_side

        # [B, 5, H, W]
        # -> [B, 5, grid_h, 16, grid_w, 16]
        x = x.reshape(b, c, grid_h, ws, grid_w, ws)

        # -> [B, grid_h, grid_w, 5, 16, 16]
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()

        # For subpatch_size = 1:
        # -> [B, grid_h, grid_w, 5, 16, 1, 16, 1]
        x = x.reshape(b, grid_h, grid_w, c, tps, sp, tps, sp)

        # -> [B, grid_h, grid_w, 16, 16, 5, 1, 1]
        x = x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()

        # -> [B, grid_h * grid_w, 256, 5]
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

        # [B, 5, 448, 448] -> [B, D, 28, 28]
        x = self._encode_local_windows(x)

        # [B, D, 28, 28] -> [B, D/2, 28, 28]
        x = self.cnn_head(x)

        # [B, D/2, 28, 28] -> [B, 3 * D/2]
        x = self._global_pool_features(x)

        # [B, 3 * D/2] -> [B, 1]
        out = self.regressor(x)

        if self.bounded_output:
            out = torch.sigmoid(out) * self.output_scale

        '''
        gates = self.token_embed.get_gates()

        print(
            f"texture_gate={gates[0].item():.4f}, "
            f"height_gate={gates[1].item():.4f}, "
            f"normal_gate={gates[2].item():.4f}"
        )

        contrib = self.token_embed.last_contrib

        print(
            f"contrib_texture={contrib['texture'].item():.4f}, "
            f"contrib_height={contrib['height'].item():.4f}, "
            f"contrib_normal={contrib['normal'].item():.4f}"
        )
        '''

        return out