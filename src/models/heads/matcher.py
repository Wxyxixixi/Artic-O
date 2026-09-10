"""Hungarian matcher for multi-part segmentation (PARTICULATE-style, adapted).

We keep slot 0 = static base (never matched, always class 0) and Hungarian-match
only the K movable GT parts to the P0 movable slots (columns 1..P0). The cost is
the single mask-NLL term from PARTICULATE (matcher.py): the negative sum of the
log-probability that a GT part's points are assigned to a movable slot. After
matching we remap each movable point's target to its matched slot column, so a
plain cross-entropy over the 1+P0 columns supervises segmentation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def hungarian_match_movable(
    seg_logits: torch.Tensor,   # [B, N, 1+P0]  col 0 = base, 1..P0 = movable slots
    part_ids: torch.Tensor,     # [B, N] long   0 = base, 1..K = movable GT parts
    num_movable: torch.Tensor,  # [B] long      K per object
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match movable GT parts to movable slots per object.

    Returns:
        target_col: [B, N] long — per-point CE target column: 0 for base points,
            else ``1 + matched_slot`` for movable points.
        matched_slot: [B, P0] long — ``matched_slot[b, j]`` = movable-slot index
            (0..P0-1) assigned to GT part ``j+1``; -1 for ``j >= K_b``.
    """
    B, N, C = seg_logits.shape
    P0 = C - 1
    device = seg_logits.device
    logp = torch.log_softmax(seg_logits.float(), dim=-1)   # [B, N, 1+P0]

    target_col = torch.zeros((B, N), dtype=torch.long, device=device)
    matched_slot = torch.full((B, P0), -1, dtype=torch.long, device=device)

    for b in range(B):
        K = int(num_movable[b].item())
        if K <= 0:
            continue
        pid = part_ids[b]                                  # [N]
        logp_mov = logp[b, :, 1:]                           # [N, P0]
        # one-hot over movable GT parts 1..K -> [N, K]
        onehot = F.one_hot(pid.clamp(min=0, max=K), num_classes=K + 1).float()[:, 1:]
        cost = -(onehot.t() @ logp_mov)                    # [K, P0]
        # This runs inside autocast; cost may be bf16 which scipy/numpy reject.
        rows, cols = linear_sum_assignment(cost.float().cpu().numpy())
        slot_of_part = np.full(K, -1, dtype=np.int64)
        for r, c in zip(rows, cols):
            slot_of_part[r] = c
            matched_slot[b, r] = int(c)
        slot_of_part_t = torch.from_numpy(slot_of_part).to(device)  # [K]
        # movable points: target column = 1 + matched slot of their GT part
        mov = pid >= 1
        tgt = 1 + slot_of_part_t[(pid[mov] - 1).long()]
        target_col[b, mov] = tgt

    return target_col, matched_slot
