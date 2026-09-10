"""Phase-5 articulation heads.

Reads from the slot-1 (active part) embedding emitted by
:class:`PATSegHead` / :class:`ImageGroundedPATSegHead` and predicts the
two articulation parameters per motion type. Two motion representations
are supported:

* ``plucker`` (default, Phase 5 / 5b / 5b-SE legacy):
    - revolute: 6-D Plücker line ``[l, m]`` regressed directly with L1,
      plus a small ``(l · m)²`` validity penalty and a unit-norm prior
      on ``l``.
    - prismatic: 3-D unit direction + scalar range.

* ``per_point_closest`` (PARTICULATE-style, Phase 5b-SE-pp):
    - revolute: 3-D axis direction (per-part) + per-point ``closest
      point on axis`` (PointAxisDecoder, MLP on ``concat(point_feat,
      slot_active_broadcast)``).
    - prismatic: same 3-D unit direction + scalar range.

  At inference, a 6-D Plücker is reassembled from
  ``(axis_dir, median(closest_pts on predicted active points))`` so all
  downstream LARM-style metrics (axis_angle, axis_origin, Mr, Md) keep
  working unchanged. The motivation is to densify the supervision
  signal for the revolute axis origin: directly regressing the 6-D
  Plücker moment ``m = l × p`` through one slot-level vector is
  bottlenecked by tiny direction errors blowing up ``m``. PARTICULATE's
  per-point closest-point head sidesteps this by giving the model a
  point-level supervision signal and recovering ``m`` analytically at
  test time.

Motion type is **known** at train and inference time (no none / both
samples in the LARM dataset; all samples are exactly one of revolute /
prismatic), so we don't predict it. Both heads still run every forward
— the loss is gated by GT motion type so each head only sees its own
samples' gradients. At inference the wrapper picks the matching head's
output.

Sign convention: from ``src.datasets.articulation``, the dataset stores
ranges as ``[lower − saved_state, upper − saved_state] = [0, hi]`` with
``hi ≥ 0``. The signed direction of the s0→s1 motion is carried by the
axis (Plücker ``l`` for revolute, ``prismatic_axis`` for prismatic) and
the *magnitude* is non-negative. We honor that: both range heads pass
through ``softplus`` to enforce non-negativity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dim: int, out_dim: int, hidden_ratio: int = 4) -> nn.Sequential:
    hidden = dim * hidden_ratio
    return nn.Sequential(
        nn.Linear(dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


def _mlp_in(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class ArticulationHead(nn.Module):
    """Per-part articulation MLPs over the slot-1 (active) embedding,
    with an optional per-point closest-point-on-axis branch.

    Forward signature:
        forward(slot_active, point_features=None) -> dict

    where ``slot_active`` is ``[B, dim]`` and ``point_features`` is
    ``[B, N, dim]`` (only required when
    ``motion_representation == 'per_point_closest'``).

    Returns a dict whose keys depend on the motion representation:

    Common to both:
        - ``revolute_range``: ``[B]`` non-negative scalar (radians).
        - ``prismatic_axis``: ``[B, 3]`` raw 3-vector. Trainer applies
          L1 / cosine vs GT axis; at inference the vector is normalized.
        - ``prismatic_range``: ``[B]`` non-negative scalar.

    ``plucker`` mode also returns:
        - ``revolute_plucker``: ``[B, 6]`` raw 6-vector.

    ``per_point_closest`` mode also returns:
        - ``revolute_axis_dir``: ``[B, 3]`` raw 3-vector (will be unit-
          normalized at inference).
        - ``revolute_closest_pt``: ``[B, N, 3]`` per-point closest-point
          on the revolute axis. Trainer supervises only on revolute
          samples and only on GT-active points.
    """

    def __init__(
        self,
        dim: int,
        hidden_ratio: int = 4,
        motion_representation: str = "plucker",
    ):
        super().__init__()
        if motion_representation not in ("plucker", "per_point_closest"):
            raise ValueError(
                f"motion_representation must be 'plucker' or 'per_point_closest', "
                f"got {motion_representation!r}"
            )
        self.motion_representation = motion_representation

        self.revolute_range = _mlp(dim, 1, hidden_ratio)
        self.prismatic_axis = _mlp(dim, 3, hidden_ratio)
        self.prismatic_range = _mlp(dim, 1, hidden_ratio)

        if motion_representation == "plucker":
            self.revolute_plucker = _mlp(dim, 6, hidden_ratio)
        else:
            self.revolute_axis_dir = _mlp(dim, 3, hidden_ratio)
            # PARTICULATE-style: input is concat(point_feat, slot_active)
            # broadcast over points; output is the closest point on the
            # revolute axis for that point. Hidden width follows the
            # other head MLPs (dim * hidden_ratio).
            self.point_axis_decoder = _mlp_in(
                in_dim=dim * 2,
                out_dim=3,
                hidden=dim * hidden_ratio,
            )

    def forward(
        self,
        slot_active: torch.Tensor,
        point_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        rev_range = F.softplus(self.revolute_range(slot_active))   # [B, 1]
        pris_axis = self.prismatic_axis(slot_active)               # [B, 3]
        pris_range = F.softplus(self.prismatic_range(slot_active)) # [B, 1]

        out: dict[str, torch.Tensor] = {
            "revolute_range": rev_range.squeeze(-1),
            "prismatic_axis": pris_axis,
            "prismatic_range": pris_range.squeeze(-1),
        }

        if self.motion_representation == "plucker":
            out["revolute_plucker"] = self.revolute_plucker(slot_active)
        else:
            out["revolute_axis_dir"] = self.revolute_axis_dir(slot_active)  # [B, 3]
            if point_features is None:
                raise RuntimeError(
                    "ArticulationHead(motion_representation='per_point_closest') "
                    "requires point_features in forward()."
                )
            B, N, D = point_features.shape
            slot_b = slot_active.unsqueeze(1).expand(B, N, D)              # [B, N, D]
            joint_in = torch.cat([point_features, slot_b], dim=-1)         # [B, N, 2D]
            out["revolute_closest_pt"] = self.point_axis_decoder(joint_in) # [B, N, 3]
        return out

    # ------------------------------------------------------------------
    # Multi-part path (P0 movable slots). Per-slot MLPs broadcast over the
    # slot axis; the per-point closest-pt is computed once per point using
    # that point's assigned slot embedding (kept at [B, N, 3] memory).
    # ------------------------------------------------------------------
    def forward_slots(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-slot articulation outputs.

        Args:
            slots: ``[B, P0, D]`` movable-slot embeddings.
        Returns dict of ``[B, P0, ...]`` tensors (revolute_range, prismatic_axis,
        prismatic_range, plus revolute_axis_dir or revolute_plucker).
        """
        rev_range = F.softplus(self.revolute_range(slots)).squeeze(-1)   # [B, P0]
        pris_axis = self.prismatic_axis(slots)                            # [B, P0, 3]
        pris_range = F.softplus(self.prismatic_range(slots)).squeeze(-1)  # [B, P0]
        out: dict[str, torch.Tensor] = {
            "revolute_range": rev_range,
            "prismatic_axis": pris_axis,
            "prismatic_range": pris_range,
        }
        if self.motion_representation == "plucker":
            out["revolute_plucker"] = self.revolute_plucker(slots)        # [B, P0, 6]
        else:
            out["revolute_axis_dir"] = self.revolute_axis_dir(slots)      # [B, P0, 3]
        return out

    def closest_pt(
        self,
        point_features: torch.Tensor,   # [B, N, D]
        slot_emb_per_point: torch.Tensor,  # [B, N, D] — each point's assigned slot embedding
    ) -> torch.Tensor:
        """Per-point closest-point-on-axis, gating each point by its own part slot."""
        joint_in = torch.cat([point_features, slot_emb_per_point], dim=-1)  # [B, N, 2D]
        return self.point_axis_decoder(joint_in)                            # [B, N, 3]


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def articulation_losses(
    pred: dict[str, torch.Tensor],
    *,
    is_revolute: torch.Tensor,
    is_prismatic: torch.Tensor,
    gt_plucker: torch.Tensor,
    gt_revolute_range_hi: torch.Tensor,
    gt_prismatic_axis: torch.Tensor,
    gt_prismatic_range_hi: torch.Tensor,
    plucker_unit_weight: float = 0.1,
    plucker_orth_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Plücker-mode loss terms (legacy Phase 5 / 5b / 5b-SE).

    Inputs (all on the same device, batch dim ``B``):
        pred: output of :class:`ArticulationHead` in ``plucker`` mode.
        is_revolute / is_prismatic: bool ``[B]``.
        gt_plucker: ``[B, 6]`` ``[l, m]`` (camera-frame, post-scale).
        gt_revolute_range_hi: ``[B]`` upper limit in radians (lower=0).
        gt_prismatic_axis: ``[B, 3]`` unit direction.
        gt_prismatic_range_hi: ``[B]`` upper limit in normalized units.

    Returns a dict of scalar losses; absent motion types yield 0 tensors
    on the same device. Caller weights and sums them.
    """
    device = pred["revolute_plucker"].device
    zero = torch.zeros((), device=device, dtype=pred["revolute_plucker"].dtype)
    out: dict[str, torch.Tensor] = {}

    # ---- revolute -------------------------------------------------------
    if is_revolute.any():
        m = is_revolute
        plucker_pred = pred["revolute_plucker"][m]
        out["loss_rev_plucker"] = F.l1_loss(plucker_pred, gt_plucker[m])

        # Plücker validity: ``l · m = 0``. Soft penalty on the dot product.
        l_pred = plucker_pred[..., :3]
        m_pred = plucker_pred[..., 3:6]
        out["loss_rev_orth"] = (
            plucker_orth_weight * (l_pred * m_pred).sum(-1).pow(2).mean()
        )

        # Direction unit-norm prior — encourage ``||l|| = 1``.
        l_norm = l_pred.norm(dim=-1)
        out["loss_rev_unit"] = (
            plucker_unit_weight * F.l1_loss(l_norm, torch.ones_like(l_norm))
        )

        out["loss_rev_range"] = F.l1_loss(
            pred["revolute_range"][m], gt_revolute_range_hi[m]
        )
    else:
        out["loss_rev_plucker"] = zero
        out["loss_rev_orth"] = zero
        out["loss_rev_unit"] = zero
        out["loss_rev_range"] = zero

    # ---- prismatic ------------------------------------------------------
    if is_prismatic.any():
        m = is_prismatic
        axis_pred = pred["prismatic_axis"][m]
        out["loss_pris_axis"] = F.l1_loss(axis_pred, gt_prismatic_axis[m])
        a_norm = axis_pred.norm(dim=-1)
        out["loss_pris_unit"] = (
            plucker_unit_weight * F.l1_loss(a_norm, torch.ones_like(a_norm))
        )
        out["loss_pris_range"] = F.l1_loss(
            pred["prismatic_range"][m], gt_prismatic_range_hi[m]
        )
    else:
        out["loss_pris_axis"] = zero
        out["loss_pris_unit"] = zero
        out["loss_pris_range"] = zero

    return out


def gt_closest_point_on_axis(
    points: torch.Tensor,
    plucker: torch.Tensor,
) -> torch.Tensor:
    """Project each point onto the GT revolute axis defined by ``plucker``.

    ``plucker`` is ``[B, 6]`` ``[l, m]`` with ``l`` unit and
    ``m = l × p_axis`` for any ``p_axis`` on the line.

    The closest point on the line to a query ``q`` is::

        q_proj = origin + dot(q - origin, l) * l

    where ``origin = m × l / ||l||²`` is the closest point on the line
    to the world origin (matches ``plucker_axis_origin`` convention).

    Args:
        points: ``[B, N, 3]`` query points (camera/normalized frame —
            same frame the Plücker is in).
        plucker: ``[B, 6]`` revolute Plücker coordinates.

    Returns:
        ``[B, N, 3]`` closest points on the axis.
    """
    l = plucker[..., :3]                                          # [B, 3]
    m = plucker[..., 3:6]                                         # [B, 3]
    l_norm_sq = (l * l).sum(-1, keepdim=True).clamp_min(1e-12)    # [B, 1]
    l_unit = l / l_norm_sq.sqrt()                                 # [B, 3]
    origin = torch.cross(m, l, dim=-1) / l_norm_sq                # [B, 3]

    origin_b = origin.unsqueeze(1)                                # [B, 1, 3]
    l_unit_b = l_unit.unsqueeze(1)                                # [B, 1, 3]
    rel = points - origin_b                                       # [B, N, 3]
    t = (rel * l_unit_b).sum(-1, keepdim=True)                    # [B, N, 1]
    return origin_b + t * l_unit_b                                # [B, N, 3]


def articulation_losses_per_point_closest(
    pred: dict[str, torch.Tensor],
    *,
    is_revolute: torch.Tensor,
    is_prismatic: torch.Tensor,
    gt_plucker: torch.Tensor,
    gt_revolute_range_hi: torch.Tensor,
    gt_prismatic_axis: torch.Tensor,
    gt_prismatic_range_hi: torch.Tensor,
    points: torch.Tensor,
    active_mask: torch.Tensor,
    plucker_unit_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """PARTICULATE-style (per-point closest-pt) loss terms.

    Inputs (all on same device, batch dim ``B``, point dim ``N``):
        pred: output of :class:`ArticulationHead` in
            ``per_point_closest`` mode. Must contain
            ``revolute_axis_dir`` ``[B, 3]`` and ``revolute_closest_pt``
            ``[B, N, 3]``.
        is_revolute / is_prismatic: bool ``[B]``.
        gt_plucker: ``[B, 6]`` GT Plücker (used to derive per-point GT
            closest-point on the revolute axis).
        gt_revolute_range_hi: ``[B]`` upper limit (radians).
        gt_prismatic_axis: ``[B, 3]`` unit direction.
        gt_prismatic_range_hi: ``[B]`` upper limit.
        points: ``[B, N, 3]`` query points (the same point cloud that
            produced ``revolute_closest_pt``; in the trainer this is
            ``input_pts``).
        active_mask: ``[B, N]`` bool — which points belong to the
            active part. Per-point closest-pt loss is only applied to
            revolute samples on their active points.

    Returns:
        Dict of scalar losses with keys
        ``loss_rev_axis_dir`` (L1 on the 3-D direction),
        ``loss_rev_closest_pt`` (L1 on per-point closest-pt),
        ``loss_rev_unit`` (unit-norm prior on the 3-D direction),
        ``loss_rev_range``,
        ``loss_pris_axis``, ``loss_pris_unit``, ``loss_pris_range``.
    """
    device = pred["revolute_axis_dir"].device
    zero = torch.zeros((), device=device, dtype=pred["revolute_axis_dir"].dtype)
    out: dict[str, torch.Tensor] = {}

    # ---- revolute -------------------------------------------------------
    if is_revolute.any():
        m_rev = is_revolute
        axis_dir_pred = pred["revolute_axis_dir"][m_rev]           # [Br, 3]
        gt_l = gt_plucker[m_rev, :3]                                # [Br, 3]
        out["loss_rev_axis_dir"] = F.l1_loss(axis_dir_pred, gt_l)

        a_norm = axis_dir_pred.norm(dim=-1)
        out["loss_rev_unit"] = (
            plucker_unit_weight * F.l1_loss(a_norm, torch.ones_like(a_norm))
        )

        out["loss_rev_range"] = F.l1_loss(
            pred["revolute_range"][m_rev], gt_revolute_range_hi[m_rev]
        )

        # Per-point closest-pt loss — only revolute samples, only their
        # active points.
        gt_cp = gt_closest_point_on_axis(points, gt_plucker)        # [B, N, 3]
        pred_cp = pred["revolute_closest_pt"]                       # [B, N, 3]
        # Active mask gated to revolute samples only.
        rev_b = m_rev.unsqueeze(-1).expand_as(active_mask)          # [B, N]
        sel = active_mask & rev_b                                   # [B, N]
        if sel.any():
            out["loss_rev_closest_pt"] = F.l1_loss(pred_cp[sel], gt_cp[sel])
        else:
            out["loss_rev_closest_pt"] = zero
    else:
        out["loss_rev_axis_dir"] = zero
        out["loss_rev_unit"] = zero
        out["loss_rev_range"] = zero
        out["loss_rev_closest_pt"] = zero

    # ---- prismatic ------------------------------------------------------
    if is_prismatic.any():
        m_pris = is_prismatic
        axis_pred = pred["prismatic_axis"][m_pris]
        out["loss_pris_axis"] = F.l1_loss(axis_pred, gt_prismatic_axis[m_pris])
        a_norm = axis_pred.norm(dim=-1)
        out["loss_pris_unit"] = (
            plucker_unit_weight * F.l1_loss(a_norm, torch.ones_like(a_norm))
        )
        out["loss_pris_range"] = F.l1_loss(
            pred["prismatic_range"][m_pris], gt_prismatic_range_hi[m_pris]
        )
    else:
        out["loss_pris_axis"] = zero
        out["loss_pris_unit"] = zero
        out["loss_pris_range"] = zero

    return out


def axis_dir_and_point_to_plucker(
    axis_dir: torch.Tensor,
    point_on_axis: torch.Tensor,
) -> torch.Tensor:
    """Recover a 6-D Plücker line from ``(axis_dir, any point on it)``.

    With unit ``l`` and any point ``p`` on the line,
    ``m = l × p`` is the Plücker moment (matches the dataset convention,
    where ``plucker_axis_origin`` returns ``cross(m, l)``).

    Args:
        axis_dir: ``[..., 3]`` (will be unit-normalized).
        point_on_axis: ``[..., 3]`` any point on the line.

    Returns:
        ``[..., 6]`` Plücker line ``[l, m]`` with ``||l|| = 1``.
    """
    l = axis_dir / axis_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    m = torch.cross(l, point_on_axis, dim=-1)
    return torch.cat([l, m], dim=-1)
