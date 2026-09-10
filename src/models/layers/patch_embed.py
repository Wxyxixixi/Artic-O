# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union
from .mlp import Mlp

import torch
from torch import Tensor
import torch.nn as nn

def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Token3DEmbedMLP(nn.Module):
    def __init__(
        self, dim, mlp_ratio=1.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop=0., embed_channels=None, use_cond_embed=False, use_input_norm=False,
    ):
        super().__init__()
        
        # Optional input normalization (can disable if initialization handles it)
        self.use_input_norm = use_input_norm
        if use_input_norm:
            self.input_norm = norm_layer(dim)
        
        # Use LayerNorm with eps=1e-6 to prevent division by zero
        self.norm = norm_layer(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            # bias=False,
            bias=True
        )

        self.emb_out_channels = 3 * dim
        if embed_channels is None:
            embed_channels = dim
        self.use_cond_embed = use_cond_embed
        if use_cond_embed:
            self.emb_layer = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(
                        embed_channels,
                        self.emb_out_channels,
                        bias=True,
                    ),
                )
        else:
            self.emb_layer = None
        
        # Register hooks on emb_layer and mlp to monitor for NaN gradients
        if self.emb_layer is not None:
            for module in self.emb_layer.modules():
                if isinstance(module, nn.Linear):
                    module.weight.register_hook(self._make_grad_hook('emb_layer.weight'))
                    if module.bias is not None:
                        module.bias.register_hook(self._make_grad_hook('emb_layer.bias'))
        
        for name, module in self.mlp.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.register_hook(self._make_grad_hook(f'mlp.{name}.weight'))
                if module.bias is not None:
                    module.bias.register_hook(self._make_grad_hook(f'mlp.{name}.bias'))
    
    def _make_grad_hook(self, name):
        """Create a gradient hook that checks for NaN and normalizes extreme values"""
        def hook(grad):
            if grad is None:
                return grad
            
            # Check for NaN/Inf
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                print(f"[CRITICAL] NaN/Inf in Token3DEmbedMLP.{name} gradient!")
                print(f"  NaN: {torch.isnan(grad).sum().item()}, Inf: {torch.isinf(grad).sum().item()}")
                grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
                print(f"  [WARNING] Replaced NaN/Inf with safe values")
            
            # Normalize extreme gradients with adaptive scaling
            grad_norm = torch.norm(grad)
            max_norm = 10.0  # More conservative for Token3DEmbedMLP
            
            return grad
        return hook

    
    def forward(self, x, cond=None):
        if self.use_cond_embed:
            # Apply input normalization if enabled
            if self.use_input_norm:
                x_input = x
                x = self.input_norm(x)
                # SAFEGUARD: If norm produces NaN (e.g., zero variance), use original input
                if torch.isnan(x).any():
                    print(f"[ERROR] input_norm produced NaN! Using original input.")
                    x = x_input
            
            # RISK: emb_layer can produce NaN if weights have NaN or cond is extreme
            cond_embed = self.emb_layer(cond).type(x.dtype)
            # SAFEGUARD: Check emb_layer output
            if torch.isnan(cond_embed).any():
                print(f"[CRITICAL] emb_layer produced NaN! This indicates NaN in emb_layer weights.")
                # Check weights
                for name, param in self.emb_layer.named_parameters():
                    if torch.isnan(param).any():
                        print(f"  NaN in {name}: {torch.isnan(param).sum().item()} values")
                cond_embed = torch.nan_to_num(cond_embed, nan=0.0)
            
            shift, scale, gate = cond_embed.chunk(3, dim=-1)
            # SAFEGUARD: Replace NaN before clamping (clamp doesn't fix NaN!)
            if torch.isnan(shift).any():
                print(f"[ERROR] shift contains NaN! Replacing with zeros.")
                shift = torch.nan_to_num(shift, nan=0.0)
            if torch.isnan(scale).any():
                print(f"[ERROR] scale contains NaN! Replacing with ones.")
                scale = torch.nan_to_num(scale, nan=1.0)
            if torch.isnan(gate).any():
                print(f"[ERROR] gate contains NaN! Replacing with zeros.")
                gate = torch.nan_to_num(gate, nan=0.0)
            
            # RISK: Clamping range for scale (0.01 to 10.0) might be too wide
            # Large scale values can cause overflow in modulation
            # RISK: LayerNorm can produce NaN if variance is zero or input contains NaN
            x_prenorm = x
            normed = self.norm(x)
            # SAFEGUARD: Check norm output
            if torch.isnan(normed).any():
                print(f"[ERROR] LayerNorm produced NaN! Using original input.")
                normed = x_prenorm
            
            # RISK: modulate can overflow if scale and normed are both large
            # modulated = normed * (1 + scale) + shift
            # Worst case: normed=large, scale=5.0 → 6*large
            modulated = modulate(normed, shift, scale)
            # SAFEGUARD: Clamp modulated values to prevent MLP overflow
            if torch.isnan(modulated).any():
                print(f"[ERROR] modulate produced NaN!")
                modulated = torch.nan_to_num(modulated, nan=0.0)
            # Clamp to reasonable range before MLP
            modulated = torch.clamp(modulated, min=-50.0, max=50.0)
            
            # RISK: MLP can amplify extreme values, GELU can produce NaN with extreme inputs
            mlp_out = self.mlp(modulated)
            # SAFEGUARD: Check MLP output
            if torch.isnan(mlp_out).any():
                print(f"[CRITICAL] MLP produced NaN! This indicates NaN in MLP weights or extreme input.")
                # Check MLP weights
                for name, param in self.mlp.named_parameters():
                    if torch.isnan(param).any():
                        print(f"  NaN in mlp.{name}: {torch.isnan(param).sum().item()} values")
                mlp_out = torch.nan_to_num(mlp_out, nan=0.0)
            
            # Clamp MLP output to prevent overflow in gating
            mlp_out = torch.clamp(mlp_out, min=-100.0, max=100.0)
            
            # RISK: gate * mlp_out can produce extreme values
            # Worst case: gate=10.0, mlp_out=100.0 → 1000.0
            gated_mlp = gate.unsqueeze(1) * mlp_out
            # SAFEGUARD: Clamp gated output
            if torch.isnan(gated_mlp).any():
                print(f"[ERROR] Gating produced NaN!")
                gated_mlp = torch.nan_to_num(gated_mlp, nan=0.0)
            gated_mlp = torch.clamp(gated_mlp, min=-100.0, max=100.0)
            
            # RISK: Residual connection x + gated_mlp can overflow if gated_mlp is extreme
            x = x + gated_mlp
            # FINAL SAFEGUARD: Ensure output doesn't contain NaN
            if torch.isnan(x).any():
                print(f"[CRITICAL] Final output contains NaN! Using prenorm input as fallback.")
                x = x_prenorm
        else:
            x = x + self.mlp(self.norm(x))
        return x
        