# Copyright (c) 2026 Weirong Chen
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any
from copy import deepcopy

from src.models.layers import PatchEmbed
from src.models.layers.block import Block
from src.models.layers.rope import RotaryPositionEmbedding2D, RotaryPositionEmbedding4D, PositionGetter
from src.models.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from src.models.layers.mlp import Mlp
from src.models.layers.patch_embed import Token3DEmbedMLP
logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class AggregatorPts3D(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "pts3d", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        num_3d_tokens=512,
        token_dim_3d=1024,
        pos_3d_embed_type=None,
        token_3d_embed_config=None,
        depth_3d_stride=1,
        share_3d_attn=False,
        add_camera_token_3d=False,
        **kwargs
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)
        self.__built_3d_tokens__(num_3d_tokens, token_dim_3d, embed_dim, token_3d_embed_config)

        # Initialize rotary position embedding if frequency > 0
        self.pos_3d_embed_type = pos_3d_embed_type
        if pos_3d_embed_type == "3d":
            self.rope = RotaryPositionEmbedding4D(frequency=rope_freq) if rope_freq > 0 else None
        else:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None

        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.share_3d_attn = share_3d_attn
        if not share_3d_attn:
            self.pts3d_blocks = nn.ModuleList(
                [
                    block_fn(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                    )
                    for _ in range(depth // depth_3d_stride)
                ]
            )
        # self.pts3d_blocks = deepcopy(self.frame_blocks)


        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.depth_3d_stride = depth_3d_stride

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        self.add_camera_token_3d = add_camera_token_3d
        if self.add_camera_token_3d == 'new':
            self.camera_token_3d = nn.Parameter(torch.randn(1, 1, embed_dim))
            nn.init.normal_(self.camera_token_3d, std=1e-6)


        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # 3D tokens
        # self.pts3d_token = nn.Parameter(torch.randn(1, num_3d_tokens, embed_dim))
        # nn.init.normal_(self.pts3d_token, std=1e-6)


        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )

            

    def __built_3d_tokens__(
        self,
        num_3d_tokens,
        token_dim_3d,
        embed_dim,
        token_3d_embed_config=None
    ):
        self.num_3d_tokens = num_3d_tokens
        self.token_dim_3d = token_dim_3d
        self.embed_dim = embed_dim

        if embed_dim == token_dim_3d:
            self.pts3d_token = nn.Parameter(torch.randn(1, num_3d_tokens, embed_dim))
            nn.init.trunc_normal_(self.pts3d_token, std=0.02)
        else:
            # use a MLP to project the 3D tokens to the same dimension as the patch tokens
            self.pts3d_token = nn.Parameter(torch.randn(1, num_3d_tokens, token_dim_3d))
            nn.init.trunc_normal_(self.pts3d_token, std=0.02)
            self.pts3d_token_proj = Mlp(
                in_features=token_dim_3d,
                hidden_features=embed_dim,
                out_features=embed_dim,
                act_layer=nn.GELU,
                drop=0.0,
                bias=False,
            )
        
        # Register gradient hook to monitor and normalize gradients on pts3d_token
        def pts3d_token_grad_hook(grad):
            # Check for NaN/Inf in gradients
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                print(f"[CRITICAL] NaN/Inf detected in pts3d_token gradients!")
                print(f"  NaN count: {torch.isnan(grad).sum().item()}")
                print(f"  Inf count: {torch.isinf(grad).sum().item()}")
                # Replace NaN/Inf with zeros to prevent parameter corruption
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"  [WARNING] Replaced NaN/Inf gradients with zeros")
            
            # Calculate gradient norm and analyze
            grad_norm = torch.norm(grad)
            
            # ADAPTIVE GRADIENT SCALING
            max_norm = 10.0
            
            if grad_norm > max_norm:
                # Normalize gradient to max_norm
                scale_factor = max_norm / (grad_norm + 1e-8)
                grad = grad * scale_factor
            
            return grad
        
        # Gradient hooks disabled - they can break autograd graph and cause NaN
        # Let the training loop handle gradient clipping instead
        
        # self.pts3d_token.register_hook(pts3d_token_grad_hook)
        
        if token_3d_embed_config is not None:
            self.token_3d_embed = eval(token_3d_embed_config.name)(**token_3d_embed_config.params)
        else:
            self.token_3d_embed = None

    
    def _get_3d_tokens(self, cond=None):
        # Check for NaN - if found, abort training (don't try to fix)
        if torch.isnan(self.pts3d_token).any():
            raise ValueError(f"pts3d_token contains NaN! Training should restart from clean checkpoint.")
        
        if self.embed_dim == self.token_dim_3d:
            pts3d_tokens = self.pts3d_token
        else:
            # Check pts3d_token_proj weights for NaN
            if hasattr(self, 'pts3d_token_proj'):
                for name, param in self.pts3d_token_proj.named_parameters():
                    if torch.isnan(param).any():
                        print(f"[CRITICAL] NaN in pts3d_token_proj.{name}")
                        with torch.no_grad():
                            param[torch.isnan(param)] = 0.0
            
            pts3d_tokens = self.pts3d_token_proj(self.pts3d_token)
            pts3d_tokens = pts3d_tokens.view(1, self.num_3d_tokens, self.embed_dim)
            
            # Clamp extreme values that could cause downstream NaN
            pts3d_tokens = torch.clamp(pts3d_tokens, min=-10.0, max=10.0)
        
        if self.token_3d_embed is not None:
            # pts3d_tokens   B, S, C
            # cond:          B, C
            
            # SAFEGUARD: Check cond (class_tokens) for NaN before processing
            if torch.isnan(cond).any():
                print(f"[ERROR] cond (class_tokens) contains NaN before token_3d_embed processing!")
                print(f"  NaN count: {torch.isnan(cond).sum().item()} / {cond.numel()}")
                # Replace NaN with zeros as emergency fallback
                cond = torch.nan_to_num(cond, nan=0.0)
                print(f"  [WARNING] Replaced NaN in cond with zeros")
            
            if self.token_3d_embed.use_cond_embed == 'first':
                cond = cond[:, 0] # use the first frame only
            elif self.token_3d_embed.use_cond_embed == 'mean':
                cond = cond.mean(dim=1) # use the mean of all frames
            
            has_nan_input = torch.isnan(pts3d_tokens).any() or torch.isnan(cond).any()
            if has_nan_input:
                print(f"[ERROR] NaN in inputs BEFORE token_3d_embed!")
                if torch.isnan(pts3d_tokens).any():
                    print(f"  pts3d_tokens NaN count: {torch.isnan(pts3d_tokens).sum().item()}")
                if torch.isnan(cond).any():
                    print(f"  cond NaN count: {torch.isnan(cond).sum().item()}")
            
            # Check token_3d_embed weights for NaN
            for name, param in self.token_3d_embed.named_parameters():
                if torch.isnan(param).any():
                    print(f"[ERROR] NaN in token_3d_embed weight: {name}")
                    print(f"  NaN count: {torch.isnan(param).sum().item()} / {param.numel()}")
            
            # Check for extreme values that might cause NaN in forward
            if (torch.abs(pts3d_tokens) > 1e6).any():
                print(f"[WARNING] Extreme values in pts3d_tokens: max_abs={torch.abs(pts3d_tokens).max().item()}")
            if (torch.abs(cond) > 1e6).any():
                print(f"[WARNING] Extreme values in cond: max_abs={torch.abs(cond).max().item()}")
            
            # Register hook on pts3d_tokens to track gradient BEFORE token_3d_embed
            pts3d_tokens_before_embed = pts3d_tokens.clone().requires_grad_(True)
            
            def pts3d_tokens_pre_embed_hook(grad):
                if torch.norm(grad) > 10.0:
                    print(f"[GRADIENT FLOW] Gradient entering token_3d_embed: norm={torch.norm(grad).item():.2f}")
                return grad
            
            # pts3d_tokens_before_embed.register_hook(pts3d_tokens_pre_embed_hook)
            
            pts3d_tokens = self.token_3d_embed(pts3d_tokens_before_embed, cond=cond)
            
            if torch.isnan(pts3d_tokens).any():
                print(f"[ERROR] NaN in pts3d_tokens AFTER token_3d_embed!")
                print(f"  NaN count: {torch.isnan(pts3d_tokens).sum().item()}")
                print(f"  Output stats: min={pts3d_tokens[~torch.isnan(pts3d_tokens)].min().item() if (~torch.isnan(pts3d_tokens)).any() else 'all NaN'}, max={pts3d_tokens[~torch.isnan(pts3d_tokens)].max().item() if (~torch.isnan(pts3d_tokens)).any() else 'all NaN'}")
                raise ValueError("NaN detected in token_3d_embed output!")


        if self.add_camera_token_3d:
            # add camera token to 3D tokens
            if self.add_camera_token_3d == 'first':
                # add camera token only to the first frame
                camera_token = self.camera_token[:, 0, :].expand(pts3d_tokens.shape[0], -1, -1)
                pts3d_tokens = torch.cat([camera_token, pts3d_tokens], dim=1)
            elif self.add_camera_token_3d == 'new':
                # add a new camera token to the 3D tokens
                camera_token = self.camera_token_3d.expand(pts3d_tokens.shape[0], -1, -1)
                pts3d_tokens = torch.cat([camera_token, pts3d_tokens], dim=1)
            else:
                raise NotImplementedError(f"Unsupported add_camera_token_3d: {self.add_camera_token_3d}. Only 'first' is supported.")


        return pts3d_tokens

    def _get_3d_pos_embed(self, B):
        P_3d = self.num_3d_tokens

        if self.pos_3d_embed_type == "3d":
            if P_3d == 512:
                # 3D grid sample 8
                x = torch.arange(0, 8)
                y = torch.arange(0, 8)
                z = torch.arange(0, 8)
                x, y, z = torch.meshgrid(x, y, z)
                w = torch.ones_like(x)
                # concatenate the 3D coordinates
                pos_3d = torch.stack([x, y, z, w], dim=-1).reshape(1, -1, 4)
                pos_3d = pos_3d.expand(B, -1, -1)
            else:
                raise ValueError(f"Unsupported 3D token size: {P_3d}. Only 512 is supported for 3D grid sampling.")
        elif self.pos_3d_embed_type == "1d":
            pos_3d = torch.zeros(B, P_3d, 2).to(self.pts3d_token.device)
            pos_3d[:, :, 0] = torch.arange(0, P_3d)
        else:
            pos_3d = torch.zeros(B, P_3d, 2).to(self.pts3d_token.device)

        # add camera token position if needed
        if self.add_camera_token_3d:
            # add an empty position for the camera token
            pos_3d = torch.cat([torch.zeros(B, 1, 2).to(pos_3d.device).to(pos_3d.dtype), pos_3d], dim=1)
        # pos_3d.shape = (B, P_3d, 2)

        return pos_3d.long()

    def get_3d_blocks(self, pts3d_idx):
        """
        Returns the 3D blocks for attention processing.
        """
        if self.share_3d_attn:
            return self.frame_blocks[pts3d_idx * self.depth_3d_stride]
        else:
            return self.pts3d_blocks[pts3d_idx]


    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        detach_vit_token: bool = False,
        view_offset: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in
            range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            3d_token_mask (torch.Tensor, optional): Mask for 3D tokens with shape [B, N_token].
            view_offset (torch.Tensor, optional): Per-view additive offset of
                shape ``[B, S, C]`` (``C`` matches patch-token channel dim).
                Added to the patch tokens *after* the optional ``detach_vit_token``
                so the offset's gradients survive the detach (the detach only
                severs gradient flow into the frozen ViT, not into the offset
                source — typically a small learnable embedding owned by the
                wrapping model). Broadcast across patch positions, so every
                patch token of view ``s`` receives ``view_offset[b, s, :]``.
        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            class_tokens = patch_tokens["x_norm_clstoken"]  # B * 1024
            patch_tokens = patch_tokens["x_norm_patchtokens"]


        _, P, C = patch_tokens.shape

        class_tokens = class_tokens.view(B, S, C).detach()

        if detach_vit_token:
            patch_tokens = patch_tokens.detach()

        # Optional per-view additive offset, applied AFTER the ViT detach so
        # the offset source (a small learnable per-view embedding owned by the
        # wrapping model) keeps its gradient path even when the ViT itself is
        # frozen. Shape contract: [B, S, C] → broadcast to [B*S, 1, C] and
        # added to the patch tokens.
        if view_offset is not None:
            if view_offset.shape[:2] != (B, S) or view_offset.shape[-1] != C:
                raise ValueError(
                    f"view_offset shape {tuple(view_offset.shape)} mismatched "
                    f"with patch tokens [B={B}, S={S}, C={C}]"
                )
            offset_flat = view_offset.reshape(B * S, 1, C).to(patch_tokens.dtype)
            patch_tokens = patch_tokens + offset_flat

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        # (B*S, P+5, C)
        
        # Register gradient hook on tokens for comparison
        if tokens.requires_grad:
            def tokens_grad_hook(grad):
                grad_norm = torch.norm(grad)
                if grad_norm > 10.0:
                    print(f"[GRADIENT COMPARISON] tokens (camera+register+patch): norm={grad_norm.item():.2f}")
                return grad
            # tokens.register_hook(tokens_grad_hook)
  
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        if self.pos_3d_embed_type == "3d":
            # append two more zeros to last dim 
            pos_summy = torch.zeros(B * S, pos.shape[1], 2).to(pos.device).to(pos.dtype)
            pos = torch.cat([pos, pos_summy], dim=-1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        pts3d_idx = 0
        output_list = []
        output_3d_list = []

        # pos.shape = (B*S, 5+P, 2) -> indicating the 2D position of each token
        # pts3d_tokens = self.pts3d_token.expand(B, -1, -1)
        pts3d_tokens = self._get_3d_tokens(class_tokens)
        pts3d_tokens = pts3d_tokens.expand(B, -1, -1)
        # add camera token to 3d tokens if needed

        # pos_3d = torch.zeros(B, P_3d, 2).to(images.device).to(pos.dtype)
        P_3d = pts3d_tokens.shape[1]
        pos_3d = self._get_3d_pos_embed(B).to(pos.device)


        for i in range(self.aa_block_num):
            run_3d_attn = False
            if self.depth_3d_stride > 0 and (i+1) % self.depth_3d_stride == 0:
                run_3d_attn = True

            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "pts3d":
                    if run_3d_attn:
                        pts3d_tokens, pts3d_idx, pts3d_intermediates = self._process_pts3d_attention(
                            pts3d_tokens, B, P_3d, C, pts3d_idx, pos=pos_3d
                        )

                elif attn_type == "global":
                    tokens, pts3d_tokens, global_idx, global_intermediates, global_intermediates_3d = self._process_global_attention(
                        tokens, pts3d_tokens, B, S, P, C, global_idx, pos=pos, pos_3d=pos_3d, run_3d_attn=run_3d_attn
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

                # concat pts3d and global intermediates, [B x P_3d x 2C]
                if run_3d_attn:
                    concat_inter_3d = torch.cat([pts3d_intermediates[i], global_intermediates_3d[i]], dim=-1)
                    output_3d_list.append(concat_inter_3d)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        del pts3d_intermediates


        return output_list, output_3d_list, self.patch_start_idx, patch_tokens

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).reshape(B * S, P, C)

        C_pos = pos.shape[2] if pos is not None else 0
        if pos is not None and pos.shape != (B * S, P, C_pos):
            pos = pos.view(B, S, P, C_pos).reshape(B * S, P, C_pos)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_pts3d_attention(self, pts3d_tokens, B, P, C, pts3d_idx, pos=None):
        
        intermediates = []
        
        for _ in range(self.aa_block_size):
            block_3d = self.get_3d_blocks(pts3d_idx)
            pts3d_tokens = block_3d(pts3d_tokens, pos=pos)
            pts3d_idx += 1
            intermediates.append(pts3d_tokens.view(B, P, C))

        return pts3d_tokens, pts3d_idx, intermediates

    def _process_global_attention(self, tokens, pts3d_tokens, B, S, P, C, global_idx, pos=None, pos_3d=None, run_3d_attn=False):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        C_pos = pos.shape[2] if pos is not None else 0
        if pos is not None and pos.shape != (B, S * P, C_pos):
            pos = pos.view(B, S, P, C_pos).view(B, S * P, C_pos)

        # concat
        # tokens = torch.cat([tokens, pts3d_tokens], dim=1)

        intermediates = []
        intermediates_3d = []

        if run_3d_attn:
            P_3d = pts3d_tokens.shape[1]
            tokens_all = torch.cat([tokens, pts3d_tokens], dim=1)
            pos_all = torch.cat([pos, pos_3d], dim=1)
        else:
            tokens_all = tokens
            pos_all = pos

    

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens_all = self.global_blocks[global_idx](tokens_all, pos=pos_all)
            global_idx += 1
            tokens = tokens_all[:, :S * P, :]
            intermediates.append(tokens.view(B, S, P, C))
            if run_3d_attn:
                pts3d_tokens = tokens_all[:, S * P :, :]
                intermediates_3d.append(pts3d_tokens.view(B, P_3d, C))

        return tokens, pts3d_tokens, global_idx, intermediates, intermediates_3d


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
