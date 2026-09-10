"""Articulation math + rigid-frame helpers used by the Stage-1/Stage-2
datasets. Pure NumPy; no project-internal imports so this stays cheap to use
from debug scripts as well.

The transforms follow the ``joint.json`` schema written by
``src/data_process/process_object.py``:

- ``revolute_plucker[part]`` is ``[l(3), m(3)]`` with ``l`` unit and
  ``m = l x p`` where ``p`` lies on the axis.
- ``revolute_range[part]`` is ``[lower, upper]`` in radians, relative to
  ``saved_state = 0``.
- ``prismatic_axis[part]`` is a unit direction and ``prismatic_range[part]``
  is ``[lower, upper]`` already scaled by the bbox-normalization factor (so
  prismatic distances are in normalized-frame units).
- ``is_part_revolute[part]`` / ``is_part_prismatic[part]`` select the
  parametrization; exactly one is true per moving part.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Per-state active-part transform
# ---------------------------------------------------------------------------


def plucker_to_4x4(plucker: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation by ``angle`` around the Plücker line ``(l, m)``."""
    plucker = np.asarray(plucker, dtype=np.float64).reshape(6)
    l = plucker[:3]
    m = plucker[3:6]
    l = l / (np.linalg.norm(l) + 1e-12)
    point = np.cross(m, l)
    K = np.array([[0.0, -l[2], l[1]],
                  [l[2], 0.0, -l[0]],
                  [-l[1], l[0], 0.0]], dtype=np.float64)
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = point - R @ point
    return T


def prismatic_to_4x4(axis: np.ndarray, distance: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    T = np.eye(4)
    T[:3, 3] = float(distance) * axis
    return T


# ---------------------------------------------------------------------------
# Articulation eval metrics — LARM-style + a signed direction metric.
#
# Mirrors ``third_party/larm_exp/metrics/eval.py:404-584`` (in NumPy on
# host arrays so we can use them from val without GPU plumbing). The
# four metrics:
#
#   * ``axis_angle`` — angle between unit-direction axes, line-style
#     (uses ``abs(dot)``, so axis flips don't penalize). Radians.
#   * ``axis_origin`` — shortest distance between the two axes, treated
#     as 3D lines. Revolute only; prismatic returns 0 (no fixed origin
#     point). Same units as the input frame.
#   * ``Mr`` — motion-range distance, ``|pred_range - gt_range|``.
#     Radians for revolute, normalized units for prismatic.
#   * ``Md`` — motion-direction difference, *signed* angle between unit
#     direction vectors. ``acos(dot)`` without ``abs``. Catches axis
#     flips the line-angle metric hides.
#
# All four are returned as raw floats; the trainer also reports
# success rate at user-configurable thresholds (defaults from the LARM
# paper, per the user spec: 0.25 rad axis_angle, 0.15 axis_origin,
# 0.3 Mr, 0.3 Md).
# ---------------------------------------------------------------------------


def _safe_unit_np(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)[:3]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def axis_angle_line(pred_dir: np.ndarray, gt_dir: np.ndarray) -> float:
    """Line angle (axis-flip invariant) in radians."""
    a = _safe_unit_np(pred_dir)
    b = _safe_unit_np(gt_dir)
    return float(np.arccos(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0)))


def axis_angle_signed(pred_dir: np.ndarray, gt_dir: np.ndarray) -> float:
    """Signed direction angle in radians — uses ``acos(dot)`` without ``abs``."""
    a = _safe_unit_np(pred_dir)
    b = _safe_unit_np(gt_dir)
    return float(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0)))


def line_line_distance(
    p0: np.ndarray, d0: np.ndarray,
    p1: np.ndarray, d1: np.ndarray,
) -> float:
    """Shortest distance between two lines defined by ``(point, direction)``.

    Mirrors LARM's ``_line_line_distance`` (eval.py:419). Falls back to
    perpendicular-distance-from-p1 when the lines are parallel.
    """
    d0u = _safe_unit_np(d0)
    d1u = _safe_unit_np(d1)
    w0 = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    c = np.cross(d0u, d1u)
    denom = float(np.linalg.norm(c))
    if denom < 1e-9:
        proj = w0 - float(np.dot(w0, d0u)) * d0u
        return float(np.linalg.norm(proj))
    return float(abs(float(np.dot(w0, c))) / denom)


def plucker_axis_origin(plucker: np.ndarray) -> np.ndarray:
    """Closest point on the Plücker line to the origin, ``m × l / ‖l‖²``.

    Matches the convention in :func:`plucker_to_4x4` (``point = cross(m, l)``
    when ``l`` is unit).
    """
    plucker = np.asarray(plucker, dtype=np.float64).reshape(6)
    l = plucker[:3]
    m = plucker[3:6]
    n = float(np.linalg.norm(l))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)
    l_unit = l / n
    return np.cross(m, l_unit)


def articulation_metrics_revolute(
    pred_plucker: np.ndarray,
    gt_plucker: np.ndarray,
    pred_range: float,
    gt_range: float,
) -> dict:
    """Four metrics for a single revolute joint sample (NumPy)."""
    pred_l = pred_plucker[:3]
    gt_l = gt_plucker[:3]
    pred_p = plucker_axis_origin(pred_plucker)
    gt_p = plucker_axis_origin(gt_plucker)
    return {
        "axis_angle": axis_angle_line(pred_l, gt_l),       # rad, line-style
        "axis_origin": line_line_distance(gt_p, gt_l, pred_p, pred_l),  # frame units
        "Mr": float(abs(float(pred_range) - float(gt_range))),
        "Md": axis_angle_signed(pred_l, gt_l),             # rad, signed
    }


def articulation_metrics_prismatic(
    pred_axis: np.ndarray,
    gt_axis: np.ndarray,
    pred_range: float,
    gt_range: float,
) -> dict:
    """Three metrics for a single prismatic joint sample.

    ``axis_origin`` is omitted (prismatic has no fixed origin point);
    callers should treat it as ``nan`` / skip when aggregating.
    """
    return {
        "axis_angle": axis_angle_line(pred_axis, gt_axis),
        "Mr": float(abs(float(pred_range) - float(gt_range))),
        "Md": axis_angle_signed(pred_axis, gt_axis),
    }


def plucker_to_4x4_torch(plucker: "torch.Tensor", angle: "torch.Tensor") -> "torch.Tensor":
    """Torch port of :func:`plucker_to_4x4` for batched per-sample use.

    ``plucker`` is ``[..., 6]`` (l (3), m (3)); ``angle`` is ``[...]`` in
    radians. Returns ``[..., 4, 4]``. Assumes the caller has already
    handled batching consistently.
    """
    import torch
    plucker = plucker.reshape(*plucker.shape[:-1], 6)
    l = plucker[..., :3]
    m = plucker[..., 3:6]
    l = l / (l.norm(dim=-1, keepdim=True) + 1e-12)
    point = torch.cross(m, l, dim=-1)
    zeros = torch.zeros_like(l[..., 0])
    K = torch.stack([
        torch.stack([zeros, -l[..., 2], l[..., 1]], dim=-1),
        torch.stack([l[..., 2], zeros, -l[..., 0]], dim=-1),
        torch.stack([-l[..., 1], l[..., 0], zeros], dim=-1),
    ], dim=-2)  # [..., 3, 3]
    eye3 = torch.eye(3, device=l.device, dtype=l.dtype).expand(*K.shape)
    sin = torch.sin(angle)[..., None, None]
    cos_complement = (1.0 - torch.cos(angle))[..., None, None]
    R = eye3 + sin * K + cos_complement * (K @ K)
    T = torch.eye(4, device=l.device, dtype=l.dtype).expand(*l.shape[:-1], 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = point - (R @ point.unsqueeze(-1)).squeeze(-1)
    return T


def prismatic_to_4x4_torch(axis: "torch.Tensor", distance: "torch.Tensor") -> "torch.Tensor":
    """Torch port of :func:`prismatic_to_4x4`. Batched."""
    import torch
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-12)
    T = torch.eye(4, device=axis.device, dtype=axis.dtype).expand(
        *axis.shape[:-1], 4, 4
    ).clone()
    T[..., :3, 3] = distance.unsqueeze(-1) * axis
    return T


def active_transform(joint_row: dict, state: float) -> np.ndarray:
    """4x4 transform for the active part at ``state in [0, 1]`` (linearly
    interpolated between ``lower`` and ``upper``). Returns identity for a
    fixed joint (neither revolute nor prismatic)."""
    if bool(joint_row["is_part_revolute"][1]):
        plucker = np.asarray(joint_row["revolute_plucker"][1], dtype=np.float64)
        lo, hi = [float(x) for x in joint_row["revolute_range"][1]]
        angle = lo + float(state) * (hi - lo)
        return plucker_to_4x4(plucker, angle)
    if bool(joint_row["is_part_prismatic"][1]):
        axis = np.asarray(joint_row["prismatic_axis"][1], dtype=np.float64)
        lo, hi = [float(x) for x in joint_row["prismatic_range"][1]]
        d = lo + float(state) * (hi - lo)
        return prismatic_to_4x4(axis, d)
    return np.eye(4)


# ---------------------------------------------------------------------------
# Rigid-frame helpers
# ---------------------------------------------------------------------------


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N, 3) points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return (points @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def transform_plucker(plucker: np.ndarray, T: np.ndarray,
                      uniform_scale: float = 1.0) -> np.ndarray:
    """Transform a Plücker line ``(l, m)`` by a rigid motion and an optional
    uniform scale applied *after* the rigid part (so the moment scales but
    the direction stays unit-length).

    Line-adjoint formula for the ``m = l × p`` convention used by
    ``src.data_process.plucker.axis_point_to_plucker``:
    ``l' = R @ l``, ``m' = s * (R @ m - t × (R @ l))``.
    """
    plucker = np.asarray(plucker, dtype=np.float64).reshape(6)
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    l_new = R @ plucker[:3]
    m_new = R @ plucker[3:6] - np.cross(t, l_new)
    # keep direction unit (scale-invariant), scale the moment by s.
    n = np.linalg.norm(l_new) + 1e-12
    l_new = l_new / n
    m_new = m_new / n
    m_new = float(uniform_scale) * m_new
    return np.concatenate([l_new, m_new]).astype(np.float32)


def transform_direction(direction: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Rotate a unit direction by the rotation block of ``T``."""
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    R = np.asarray(T, dtype=np.float64)[:3, :3]
    out = R @ direction
    return (out / (np.linalg.norm(out) + 1e-12)).astype(np.float32)


def embed_rotation_4x4(R: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return T


# ---------------------------------------------------------------------------
# Random rotations
# ---------------------------------------------------------------------------


def random_so3(rng: np.random.Generator | None = None) -> np.ndarray:
    """Haar-uniform random rotation matrix via QR of a Gaussian matrix.

    Mirrors LARM's rotation sampler.
    """
    rng = np.random.default_rng() if rng is None else rng
    M = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.astype(np.float64)


# ---------------------------------------------------------------------------
# Self-check
