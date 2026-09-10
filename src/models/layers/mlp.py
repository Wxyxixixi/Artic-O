# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

from torch import Tensor, nn
import torch

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        # DEBUG: Check input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[ERROR] NaN/Inf in MLP input: shape={x.shape}, nan_count={torch.isnan(x).sum()}, inf_count={torch.isinf(x).sum()}")
        
        x = self.fc1(x)
        
        # DEBUG: Check after fc1
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[ERROR] NaN/Inf after fc1: nan={torch.isnan(x).sum()}, inf={torch.isinf(x).sum()}")
            # Check fc1 weights
            if torch.isnan(self.fc1.weight).any():
                print(f"[ERROR] fc1 weights contain NaN!")
        
        x = self.act(x)
        x = self.drop(x)
        
        # DEBUG: Check before fc2 (the problematic layer)
        if torch.isnan(x).any() or (torch.abs(x) > 1e6).any():
            print(f"[ERROR] Before fc2: nan={torch.isnan(x).any()}, max_abs={torch.abs(x).max():.4f}")
        
        x = self.fc2(x)
        
        # DEBUG: Check after fc2
        if torch.isnan(x).any():
            print(f"[ERROR] NaN after fc2 (this is where backward fails)!")
            print(f"  Output: nan_count={torch.isnan(x).sum()}, shape={x.shape}")
            # Check fc2 weights
            if torch.isnan(self.fc2.weight).any():
                print(f"[ERROR] fc2 weights contain NaN!")
            else:
                print(f"  fc2.weight: min={self.fc2.weight.min():.6f}, max={self.fc2.weight.max():.6f}")
        
        x = self.drop(x)
        return x
