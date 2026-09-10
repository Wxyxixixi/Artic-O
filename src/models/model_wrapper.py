# Copyright (c) 2026 Weirong Chen
import torch
from torch import nn


class BatchModelWrapper(nn.Module):
    """Wraps a Nova3r model for use with the ODE solver.

    Calls model._encode() separately, then feeds encoder_data into _decode()
    at each ODE step.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Unwrap DDP once rather than checking hasattr on every ODE step
        self._model = model.module if hasattr(model, 'module') else model

    @torch.no_grad()
    def forward(self, x, t, images, encoder_data=None, **extras):
        if len(t.shape) == 0:
            B = x.shape[0]
            t = t.reshape(-1, 1).expand(B, x.shape[1])

        if encoder_data is None:
            raise ValueError("encoder_data is required. Call model._encode() first and pass the result.")

        output = self._model._decode(tokens=encoder_data['tokens'], images=images, query_points=x, timestep=t)
        return output['pts3d_xyz']
