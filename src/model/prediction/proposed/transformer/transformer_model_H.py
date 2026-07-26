import torch
import torch.nn as nn


class LearnableSoftmaxFusionGates(nn.Module):
    """
    Learnable Softmax Fusion Gates (LSFG) for texture, height, and normal inputs.

    Input token channel order:
        texture_gray : 1 channel
        height_gray  : 1 channel
        normal_xyz   : 3 channels

    Since each pixel is one token, each raw pixel token has:
        [texture_gray, height_gray, normal_x, normal_y, normal_z]

    Input:
        pixel_tokens: [B, num_windows, tokens_per_window, 5]

    Output:
        fused_pixel_tokens: [B, num_windows, tokens_per_window, D]
    """

    def __init__(
        self,
        embed_dim,
        init_gates=(1.0, 1.0, 1.0),
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Independent linear projections for texture, height, and normal inputs.
        self.texture_projection = nn.Linear(1, embed_dim)
        self.height_projection = nn.Linear(1, embed_dim)
        self.normal_projection = nn.Linear(3, embed_dim)

        # Learnable gates for LSFG.
        # Gate order:
        #   [texture, height, normal]
        self.fusion_gates = nn.Parameter(
            torch.tensor(init_gates, dtype=torch.float32)
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()

        # Optional diagnostic values.
        # These values are not used for training loss.
        self.last_feature_contribution = None

    def forward(self, pixel_tokens):
        """
        pixel_tokens:
            [B, num_windows, tokens_per_window, 5]

        Channel order:
            texture_gray, height_gray, normal_x, normal_y, normal_z
        """
        texture_token = pixel_tokens[..., 0:1]
        height_token = pixel_tokens[..., 1:2]
        normal_token = pixel_tokens[..., 2:5]

        texture_embed = self.texture_projection(texture_token)
        height_embed = self.height_projection(height_token)
        normal_embed = self.normal_projection(normal_token)

        fusion_weights = torch.softmax(self.fusion_gates, dim=0)

        with torch.no_grad():
            texture_contribution = (
                fusion_weights[0] * texture_embed
            ).norm(dim=-1).mean()
            height_contribution = (
                fusion_weights[1] * height_embed
            ).norm(dim=-1).mean()
            normal_contribution = (
                fusion_weights[2] * normal_embed
            ).norm(dim=-1).mean()

            self.last_feature_contribution = {
                "texture": texture_contribution.detach(),
                "height": height_contribution.detach(),
                "normal": normal_contribution.detach(),
            }

        fused_pixel_tokens = (
            fusion_weights[0] * texture_embed
            + fusion_weights[1] * height_embed
            + fusion_weights[2] * normal_embed
        )

        fused_pixel_tokens = self.norm(fused_pixel_tokens)
        fused_pixel_tokens = self.activation(fused_pixel_tokens)

        return fused_pixel_tokens

    def get_fusion_weights(self):
        """
        Return current LSFG weights as probabilities:
            [texture_weight, height_weight, normal_weight]
        """
        return torch.softmax(self.fusion_gates.detach(), dim=0)


class TransformerRegressor(nn.Module):
    """
    Local-global transformer-based perceived roughness regressor with LSFG.

    Input:
        texture_image, height_map, normal_map

        texture_image:
            1-channel grayscale texture map or 3-channel texture image

        height_map:
            1-channel height map or 3-channel height-like input

        normal_map:
            3-channel normal map

    Internal input feature:
        texture_gray : [B, 1, H, W]
        height_gray  : [B, 1, H, W]
        normal       : [B, 3, H, W]

        concat -> [B, 5, H, W]

    Output:
        perceived_roughness: [B, 1]

    Main structure:
        input maps [B, 5, 256, 256]
        -> Window partitioning
        -> Learnable Softmax Fusion Gates (LSFG)
        -> Local transformer
        -> Local pooling with mean + max + standard deviation
        -> Global transformer
        -> CNN
        -> Global pooling with mean + max + standard deviation
        -> MLP
        -> perceived roughness [B, 1]
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
        init_gates=(1.0, 1.0, 1.0),
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

        self.window_grid_size = image_size // window_size
        self.num_windows = self.window_grid_size ** 2

        # Since each pixel inside a window is one token.
        # For window_size = 16:
        # tokens_per_window = 16 * 16 = 256
        self.tokens_per_window = window_size ** 2

        # Learnable Softmax Fusion Gates (LSFG):
        # texture, height, and normal inputs are projected independently and fused with learnable softmax weights.
        self.LSFG = LearnableSoftmaxFusionGates(
            embed_dim=embed_dim,
            init_gates=init_gates,
        )

        # Pixel-level positional embedding for 256 pixel tokens inside each local window.
        self.pixel_level_pos_embed = nn.Parameter(
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

        self.local_multi_heads_self_attention_transformer = nn.TransformerEncoder(
            local_encoder_layer,
            num_layers=depth,
        )

        self.local_norm = nn.LayerNorm(embed_dim)

        # Window features projection:
        # [B*num_windows, 3D] -> [B*num_windows, D]
        self.window_features_projection = nn.Linear(embed_dim * 3, embed_dim)

        # Window-level positional embedding for 256 window descriptors.
        # For image_size=256 and window_size=16:
        # num_windows = 16 * 16 = 256
        self.window_level_pos_embed = nn.Parameter(
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

        self.global_multi_heads_self_attention_transformer = nn.TransformerEncoder(
            global_encoder_layer,
            num_layers=global_depth,
        )

        self.global_norm = nn.LayerNorm(embed_dim)

        # CNN head for spatial refinement of the window-level feature map.
        """
            Apply the CNN head to refine the spatial feature map.
        
            Input:
                spatial_feature_map: [B, D, 16, 16]
        
            Output:
                spatial_feature_map: [B, D/2, 16, 16]
        """
        self.CNN = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),

            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
        )

        # MLP for final perceived roughness prediction.
        """
            Apply the MLP to predict perceived roughness.

            Input:
                global_feature: [B, 3 * D/2]

            Output:
                roughness: [B, 1]
        """
        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear((embed_dim // 2) * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pixel_level_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.window_level_pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_lsfg_weights(self):
        """
        Return current LSFG weights as probabilities:
            [texture_weight, height_weight, normal_weight]
        """
        return self.LSFG.get_fusion_weights()

    def get_lsfg_feature_contribution(self):
        """
        Return last diagnostic feature contribution values.

        This is only for inspection/debugging.
        It is updated during forward().
        """
        return self.LSFG.last_feature_contribution

    def _to_grayscale_map(self, input_map):
        """
        Convert texture or height input to [B, 1, H, W].

        If input has 1 channel, keep it.
        If input has 3 channels, average it to grayscale.
        """
        if input_map.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor [B, C, H, W], "
                f"but got shape {input_map.shape}"
            )

        input_map = input_map.float()

        if input_map.size(1) == 1:
            return input_map

        if input_map.size(1) == 3:
            return input_map.mean(dim=1, keepdim=True)

        raise ValueError(
            f"Expected channel size 1 or 3, but got {input_map.size(1)}"
        )

    def _to_normal_map(self, input_map):
        """
        Convert normal input to [B, 3, H, W].

        Important:
            If the input has 3 channels, they are preserved.
            Normal x/y/z channels are not averaged.
        """
        if input_map.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor [B, C, H, W], "
                f"but got shape {input_map.shape}"
            )

        input_map = input_map.float()

        if input_map.size(1) == 1:
            # Fallback for grayscale normal-like input.
            input_map = input_map.repeat(1, 3, 1, 1)

        if input_map.size(1) != 3:
            raise ValueError(
                f"Expected normal channel size 1 or 3, "
                f"but got {input_map.size(1)}"
            )

        return input_map

    def build_input_maps(self, texture_image, height_map, normal_map):
        """
        Build the 5-channel input maps.

        texture_image:
            [B, 3, H, W] or [B, 1, H, W]
            -> grayscale texture map [B, 1, H, W]

        height_map:
            [B, 3, H, W] or [B, 1, H, W]
            -> grayscale height map [B, 1, H, W]

        normal_map:
            [B, 3, H, W] or [B, 1, H, W]
            -> normal map [B, 3, H, W]

        Output:
            input_maps: [B, 5, H, W]
            = concat(texture_gray, height_gray, normal_xyz)
        """
        texture_gray = self._to_grayscale_map(texture_image)
        height_gray = self._to_grayscale_map(height_map)
        normal = self._to_normal_map(normal_map)

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

        input_maps = torch.cat([texture_gray, height_gray, normal], dim=1)

        return input_maps

    def window_partitioning(self, input_maps):
        """
        Partition input maps into local windows.

        Input:
            input_maps: [B, 5, 256, 256]

        Output:
            pixel_tokens: [B, num_windows, tokens_per_window, 5]
            grid_h, grid_w
        """
        batch_size, channels, height, width = input_maps.shape

        if channels != self.input_channels:
            raise ValueError(
                f"Expected input with {self.input_channels} channels, "
                f"but got {channels}"
            )

        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"This model expects {self.image_size}x{self.image_size} inputs, "
                f"but got H={height}, W={width}."
            )

        if height % self.window_size != 0 or width % self.window_size != 0:
            raise ValueError(
                f"Input H and W must be divisible by window_size={self.window_size}, "
                f"but got H={height}, W={width}"
            )

        grid_h = height // self.window_size
        grid_w = width // self.window_size
        window_size = self.window_size

        # [B, 5, H, W]
        # For 256x256 and window_size=16:
        # -> [B, 5, 16, 16, 16, 16]
        x = input_maps.reshape(
            batch_size,
            channels,
            grid_h,
            window_size,
            grid_w,
            window_size,
        )

        # -> [B, grid_h, grid_w, 16, 16, 5]
        x = x.permute(
            0, 2, 4, 3, 5, 1
        ).contiguous()

        # -> [B, grid_h * grid_w, 256, 5]
        pixel_tokens = x.reshape(
            batch_size,
            grid_h * grid_w,
            self.tokens_per_window,
            channels,
        )

        return pixel_tokens, grid_h, grid_w

    def learnable_softmax_fusion_gates(self, pixel_tokens):
        """
        Construct fused pixel tokens using Learnable Softmax Fusion Gates.

        Input:
            pixel_tokens: [B, num_windows, tokens_per_window, 5]

        Output:
            fused_pixel_tokens: [B * num_windows, tokens_per_window, D]
        """
        batch_size, num_windows, _, _ = pixel_tokens.shape

        # Learnable Softmax Fusion Gates:
        # [B, num_windows, 256, 5]
        # -> [B, num_windows, 256, D]
        x = self.LSFG(pixel_tokens)

        # -> [B * num_windows, 256, D]
        fused_pixel_tokens = x.reshape(
            batch_size * num_windows,
            self.tokens_per_window,
            self.embed_dim,
        )

        return fused_pixel_tokens

    def local_transformer(self, fused_pixel_tokens):
        """
        Apply the local transformer to pixel tokens within each local window.

        Input:
            fused_pixel_tokens: [B * num_windows, tokens_per_window, D]

        Output:
            local_token_features: [B * num_windows, tokens_per_window, D]
        """
        # Add pixel-level positional embedding.
        # fused_pixel_tokens: [B * num_windows, 256, D]
        fused_pixel_tokens = fused_pixel_tokens + self.pixel_level_pos_embed

        # Local self-attention:
        # each local window is processed independently.
        local_token_features = self.local_multi_heads_self_attention_transformer(
            fused_pixel_tokens
        )
        local_token_features = self.local_norm(local_token_features)

        return local_token_features

    def local_pooling(self, local_token_features):
        """
        Apply local pooling along the token dimension.

        Input:
            local_token_features: [B * num_windows, tokens_per_window, D]

        Output:
            window_features: [B * num_windows, 3D]
        """
        # Local pooling along the token dimension:
        # [B * num_windows, tokens_per_window, D]
        # -> mean/max/std each [B * num_windows, D]
        mean_feature = local_token_features.mean(dim=1)
        max_feature = local_token_features.max(dim=1).values
        std_feature = local_token_features.std(dim=1, unbiased=False)

        # -> [B * num_windows, 3D]
        window_features = torch.cat(
            [mean_feature, max_feature, std_feature],
            dim=1,
        )

        return window_features

    def global_transformer(self, window_descriptors, grid_h, grid_w):
        """
        Apply the global transformer to model relationships among window descriptors.

        Input:
            window_descriptors: [B, num_windows, D]

        Output:
            spatial_feature_map: [B, D, grid_h, grid_w]
        """
        batch_size, num_windows, descriptor_dim = window_descriptors.shape

        if descriptor_dim != self.embed_dim:
            raise ValueError(
                f"Expected window descriptor dim {self.embed_dim}, "
                f"but got {descriptor_dim}"
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
                f"but got num_windows={num_windows}, "
                f"grid_h={grid_h}, grid_w={grid_w}"
            )

        # Add window-level positional embedding.
        # [B, 256, D] + [1, 256, D]
        window_descriptors = window_descriptors + self.window_level_pos_embed

        # Global self-attention among the 256 window descriptors.
        spatial_feature_map = self.global_multi_heads_self_attention_transformer(
            window_descriptors
        )
        spatial_feature_map = self.global_norm(spatial_feature_map)

        # -> [B, D, grid_h, grid_w]
        spatial_feature_map = spatial_feature_map.transpose(1, 2).reshape(
            batch_size,
            self.embed_dim,
            grid_h,
            grid_w,
        )

        return spatial_feature_map

    def global_pooling(self, spatial_feature_map):
        """
        Apply global pooling to the entire spatial feature map.

        Input:
            spatial_feature_map: [B, C, H, W]

        Output:
            global_feature: [B, 3C]
        """
        mean_feature = spatial_feature_map.mean(dim=(2, 3))
        max_feature = spatial_feature_map.amax(dim=(2, 3))
        std_feature = spatial_feature_map.std(dim=(2, 3), unbiased=False)

        global_feature = torch.cat(
            [mean_feature, max_feature, std_feature],
            dim=1,
        )

        return global_feature

    def forward(self, texture_image, height_map, normal_map):
        input_maps = self.build_input_maps(texture_image, height_map, normal_map)
        batch_size = input_maps.size(0)

        # Window partitioning:
        # [B, 5, 256, 256] -> pixel tokens [B, 256, 256, 5]
        pixel_tokens, grid_h, grid_w = self.window_partitioning(input_maps)

        # Learnable Softmax Fusion Gates:
        # [B, 256, 256, 5] -> fused pixel tokens [B * 256, 256, D]
        fused_pixel_tokens = self.learnable_softmax_fusion_gates(pixel_tokens)

        # Local transformer:
        # [B * 256, 256, D] -> local token features [B * 256, 256, D]
        local_token_features = self.local_transformer(fused_pixel_tokens)

        # Local pooling:
        # [B * 256, 256, D] -> window features [B * 256, 3D]
        window_features = self.local_pooling(local_token_features)

        # [B * num_windows, 3D] -> [B * num_windows, D]
        window_descriptors = self.window_features_projection(
            window_features
        )

        # -> [B, num_windows, D]
        window_descriptors = window_descriptors.reshape(
            batch_size,
            grid_h * grid_w,
            self.embed_dim,
        )

        # Global transformer:
        # [B, 256, D] -> spatial feature map [B, D, 16, 16]
        spatial_feature_map = self.global_transformer(
            window_descriptors,
            grid_h,
            grid_w,
        )

        # CNN:
        # [B, D, 16, 16] -> [B, D/2, 16, 16]
        x = self.CNN(spatial_feature_map)

        # Global pooling:
        # [B, D/2, 16, 16] -> [B, 3 * D/2]
        x = self.global_pooling(x)

        # MLP:
        # [B, 3 * D/2] -> [B, 1]
        roughness = self.MLP(x)

        if self.bounded_output:
            roughness = torch.sigmoid(roughness) * self.output_scale

        return roughness
