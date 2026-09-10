"""Stage-1 dataset for the articulated-object MVP.

Reads a HuggingFace cache built by ``src.datasets.hf_builder`` and returns,
per ``__getitem__``, a normalized full-state point cloud at a random
articulation state ``s_i`` plus the articulation parameters.

Per-sample output (see :meth:`BaseDataset.__getitem__`):

- ``pts``: ``pcd_full`` at ``s_i``, in the first-camera frame, ``median_3``
  normalized. Shape ``[num_points, 3]``.
- ``pts_gt``: same cloud in *world* frame pre-augmentation / pre-normalization.
- ``pts_part_mask``: bool mask over ``pts`` (True where active). Recover the
  per-part slices as ``pts[~pts_part_mask]`` / ``pts[pts_part_mask]`` —
  they are not returned as separate tensors because per-sample row counts
  vary and that breaks PyTorch's default collate.
- ``si``: sampled articulation state in ``[0, 1]``.
- articulation params (Plücker / ranges / flags) also in the final frame.
- ``c2w_first`` / ``w2c_first`` / ``aug_R``: every transform used in the
  chain, so evaluation / denorm can reconstruct the mapping exactly.

The preprocessing pipeline bakes bbox-normalization into the raw cache;
this loader adds the NOVA3R-compatible ``world_to_cam`` +
``normalize_input(mode="median_3")`` pass on top so returned tensors are in
the frame the NOVA3R encoder was pretrained on.
"""
from __future__ import annotations

import json
import random
from typing import Iterable, Sequence

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

from src.datasets.articulation import (
    active_transform,
    apply_transform,
    embed_rotation_4x4,
    random_so3,
    transform_direction,
    transform_plucker,
)


# ---------------------------------------------------------------------------
# Shared transform pipeline used by Stage 1 and Stage 2.
# ---------------------------------------------------------------------------


def _median_norm_factor(pts: np.ndarray, target_median: float = 3.0) -> tuple[float, float]:
    """Mirror of ``src.datasets.utils.normalize_input`` for ``mode="median_3"``
    but evaluated on a single cloud. Returns ``(norm_factor, target_median)``
    so the normalized cloud is ``pts / norm_factor * target_median``.
    """
    dists = np.linalg.norm(pts, axis=-1)
    if dists.size == 0:
        norm_factor = 1.0
    else:
        norm_factor = float(np.median(dists))
    norm_factor = float(np.clip(norm_factor, 0.01, 100.0))
    return norm_factor, float(target_median)


def _sample_indices(n: int, k: int, rng: random.Random) -> np.ndarray:
    """Return ``k`` indices from ``[0, n)`` with replacement when ``n < k``."""
    if n >= k:
        return np.asarray(rng.sample(range(n), k), dtype=np.int64)
    # With-replacement top-up when the pool is smaller than requested.
    base = np.arange(n)
    extra = np.asarray([rng.randrange(n) for _ in range(k - n)], dtype=np.int64)
    return np.concatenate([base, extra])


def _collect_state_mode(mode: str, rng: random.Random) -> float:
    if mode == "uniform":
        return float(rng.random())
    if mode == "discrete":
        return float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]))
    if mode == "endpoints":
        return float(rng.choice([0.0, 1.0]))
    if mode == "zero":
        return 0.0
    if mode == "one":
        return 1.0
    raise ValueError(f"Unknown state_mode {mode!r}")


def build_sample_payload(
    row: dict,
    num_points: int,
    state_mode: str,
    augment: bool,
    target_median: float,
    rng: random.Random,
    np_rng: np.random.Generator,
    sampling: str = "merged",
    part_ratio: float | None = None,
    ref_view_idx: int = 0,
) -> dict:
    """Compose the per-item dict shared between Stage 1 and Stage 2.

    Steps, in order:

    1. Draw ``s_i`` per ``state_mode`` and articulate the active part.
    2. Build the world-frame ``pts`` cloud:
       - ``sampling="merged"`` (default): pick ``num_points`` indices from
         the merged, area-weighted ``pcd_full_raw`` sample baked at build
         time. The accompanying ``pcd_full_part_mask`` says which subset is
         active, so the articulation transform applies only there.
       - ``sampling="per_part"``: resample ``pcd_ctx`` and ``pcd_part_raw``
         independently to ``num_points`` total. By default (``part_ratio``
         is ``None``) the split mirrors the on-disk size ratio
         ``n_ctx_src : n_part_src``; pass ``part_ratio = part / ctx`` to
         override (e.g. ``0.5`` -> 2 ctx : 1 part, ``2.0`` -> 1 ctx : 2 part,
         ``1.0`` -> 50/50).
    3. Optional random SO(3) augment in world frame (returned as ``aug_R``).
    4. ``world_to_cam`` using the (augmented) reference-camera pose
       (``cam_c2w[ref_view_idx]`` — default 0 for Stage 1; Stage 2 passes
       the first picked s0 view so the model's reference frame matches the
       first input image).
    5. ``median_3`` normalization. Uniform scale propagates to the Plücker
       moment, prismatic range, and camera translations.
    """
    # Cameras: prefer the per-state schema (cam_c2w_s0 + cam_c2w_s1, each
    # [N,4,4]) which we concat to [2N,4,4] so frame ordering becomes
    # [s0_views..., s1_views...]. Stage 2 picks per-state and offsets s1
    # indices by N when looking up cam_c2w_final. Legacy single-array
    # `cam_c2w` is still accepted (concat-with-self preserves shape so any
    # legacy index keeps working).
    if row.get("cam_c2w_s0") is not None and row.get("cam_c2w_s1") is not None:
        c2w_s0 = np.asarray(row["cam_c2w_s0"], dtype=np.float64)
        c2w_s1 = np.asarray(row["cam_c2w_s1"], dtype=np.float64)
        cam_c2w = np.concatenate([c2w_s0, c2w_s1], axis=0)  # [2N, 4, 4]
    elif row.get("cam_c2w") is not None:
        cam_c2w = np.asarray(row["cam_c2w"], dtype=np.float64)
    else:
        raise KeyError(
            "row missing camera poses: expected 'cam_c2w_s0'/'cam_c2w_s1' "
            "(new schema) or 'cam_c2w' (legacy)"
        )
    plucker = np.asarray(row["revolute_plucker"][1], dtype=np.float64)
    prismatic_axis = np.asarray(row["prismatic_axis"][1], dtype=np.float64)
    prismatic_range = np.asarray(row["prismatic_range"][1], dtype=np.float32)
    revolute_range = np.asarray(row["revolute_range"][1], dtype=np.float32)
    is_revolute = bool(row["is_part_revolute"][1])
    is_prismatic = bool(row["is_part_prismatic"][1])

    # 1) draw state
    si = _collect_state_mode(state_mode, rng)
    T_act = active_transform(row, si)

    # 2) build world-frame pts + part mask
    if sampling == "merged" and "pcd_full_raw" in row and row.get("pcd_full_raw") is not None:
        pcd_full = np.asarray(row["pcd_full_raw"], dtype=np.float32)
        full_mask = np.asarray(row["pcd_full_part_mask"], dtype=bool)

        n_full = pcd_full.shape[0]
        idx = _sample_indices(n_full, num_points, rng)
        pts_world = pcd_full[idx].astype(np.float32).copy()
        part_mask = full_mask[idx]
        # Apply articulation only to the active subset (in place).
        if part_mask.any():
            pts_world[part_mask] = apply_transform(pts_world[part_mask], T_act)
        n_ctx = int((~part_mask).sum())
    elif sampling in ("merged", "per_part"):
        pcd_ctx = np.asarray(row["pcd_ctx"], dtype=np.float32)
        pcd_part_raw = np.asarray(row["pcd_part_raw"], dtype=np.float32)
        pcd_part_si = apply_transform(pcd_part_raw, T_act)

        n_ctx_src = pcd_ctx.shape[0]
        n_part_src = pcd_part_si.shape[0]
        if part_ratio is None:
            n_total_src = max(n_ctx_src + n_part_src, 1)
            n_ctx = int(round(num_points * n_ctx_src / n_total_src))
        else:
            pr = float(part_ratio)
            if pr < 0.0:
                raise ValueError(f"part_ratio must be >= 0, got {pr}")
            # part_ratio = part / ctx; ctx share = 1 / (part_ratio + 1).
            n_ctx = int(round(num_points / (pr + 1.0)))
        n_ctx = int(np.clip(n_ctx, 1, num_points - 1)) if num_points > 1 else 1
        n_part = num_points - n_ctx

        idx_ctx = _sample_indices(n_ctx_src, n_ctx, rng)
        idx_part = _sample_indices(n_part_src, n_part, rng)
        pts_world = np.concatenate(
            [pcd_ctx[idx_ctx], pcd_part_si[idx_part]], axis=0
        ).astype(np.float32)
        part_mask = np.concatenate([
            np.zeros(n_ctx, dtype=bool),
            np.ones(n_part, dtype=bool),
        ])
    else:
        raise ValueError(f"Unknown sampling mode {sampling!r} (use 'merged' or 'per_part')")

    # 3) optional world-frame SO(3) augment
    if augment:
        R_aug = random_so3(np_rng)
    else:
        R_aug = np.eye(3)
    T_aug = embed_rotation_4x4(R_aug)
    cam_c2w_aug = (T_aug @ cam_c2w).astype(np.float64)  # broadcast over [V,4,4]
    pts_aug = apply_transform(pts_world, T_aug)
    plucker_aug = (
        transform_plucker(plucker, T_aug, uniform_scale=1.0) if is_revolute
        else plucker.astype(np.float32)
    )
    prismatic_axis_aug = (
        transform_direction(prismatic_axis, T_aug) if is_prismatic
        else prismatic_axis.astype(np.float32)
    )

    # 4) world -> reference-camera frame (``ref_view_idx``)
    if not (0 <= ref_view_idx < cam_c2w_aug.shape[0]):
        raise IndexError(
            f"ref_view_idx={ref_view_idx} out of range for "
            f"cam_c2w with {cam_c2w_aug.shape[0]} views"
        )
    c2w_first_aug = cam_c2w_aug[ref_view_idx]
    T_w2c = np.linalg.inv(c2w_first_aug)
    cam_c2w_final = np.einsum("ij,vjk->vik", T_w2c, cam_c2w_aug)

    pts_cam = apply_transform(pts_aug, T_w2c)
    plucker_cam = (
        transform_plucker(plucker_aug, T_w2c, uniform_scale=1.0) if is_revolute
        else plucker_aug
    )
    prismatic_axis_cam = (
        transform_direction(prismatic_axis_aug, T_w2c) if is_prismatic
        else prismatic_axis_aug
    )

    # 5) median_3 normalization
    norm_factor, tm = _median_norm_factor(pts_cam, target_median=target_median)
    s_scale = tm / norm_factor
    pts_final = (pts_cam * s_scale).astype(np.float32)

    cam_c2w_out = cam_c2w_final.copy()
    cam_c2w_out[:, :3, 3] *= s_scale

    # First-cam-frame copy BEFORE the median-3 translation rescale. Lives in
    # the same frame as ``pts_gt`` (= ``pts_cam``, unscaled) and the
    # post-chamfer-aligned predictions used by val_epoch metrics. Downstream
    # snapshot tools (forward_pass + project_color + articulate_to_qpos) use
    # this so cameras and the aligned pred mesh are in the same frame.
    cam_c2w_first_cam = cam_c2w_final.astype(np.float32)

    # Plücker moment scales with distance; line direction is invariant.
    plucker_final = plucker_cam.copy()
    if is_revolute:
        plucker_final[3:6] = plucker_cam[3:6] * s_scale
    prismatic_range_final = prismatic_range.astype(np.float32) * np.float32(s_scale)
    prismatic_axis_final = prismatic_axis_cam.astype(np.float32)

    # Per-part slices are recoverable as ``pts[~pts_part_mask]`` /
    # ``pts[pts_part_mask]`` — we don't return them directly because their
    # row counts vary per sample and that blows up PyTorch's default collate.

    return {
        "pts": torch.from_numpy(pts_final),                        # [N, 3]
        "pts_gt": torch.from_numpy(pts_cam),                     # reference-camera frame, before normalization
        "pts_part_mask": torch.from_numpy(part_mask),

        "pts_norm_factor": torch.tensor(norm_factor, dtype=torch.float32),
        "pts_target_median": torch.tensor(tm, dtype=torch.float32),

        "c2w_first": torch.from_numpy(c2w_first_aug.astype(np.float32)),
        "w2c_first": torch.from_numpy(T_w2c.astype(np.float32)),
        "aug_R": torch.from_numpy(R_aug.astype(np.float32)),

        "si": torch.tensor(si, dtype=torch.float32),
        "joint_type": str(row["joint_type"]),
        "plucker": torch.from_numpy(plucker_final.astype(np.float32)),
        "prismatic_axis": torch.from_numpy(prismatic_axis_final),
        "revolute_range": torch.from_numpy(revolute_range.astype(np.float32)),
        "prismatic_range": torch.from_numpy(prismatic_range_final),
        "is_revolute": bool(is_revolute),
        "is_prismatic": bool(is_prismatic),
        "saved_state": torch.tensor(float(row["saved_state"]), dtype=torch.float32),

        "cam_c2w_final": torch.from_numpy(cam_c2w_out.astype(np.float32)), # camera poses we might need in the future
        "cam_c2w_first_cam": torch.from_numpy(cam_c2w_first_cam),  # same frame as pts_gt (unscaled)
        "ref_view_idx": torch.tensor(int(ref_view_idx), dtype=torch.int64),

        "sample_id": str(row["sample_id"]),
    }


# ---------------------------------------------------------------------------
# Stage-1 dataset
# ---------------------------------------------------------------------------


def _materialize_rows(ds, fields: Sequence[str]) -> list[dict]:
    """Load a restricted view of the HF dataset into RAM as plain dicts.

    We avoid pulling Image features here — Stage 1 doesn't need them, and
    skipping them both saves memory and avoids any lazy-decode work.

    The view is read in one batched call (``ds_view[:]``) and then
    transposed into per-row dicts. Per-index access
    (``[ds_view[i] for i in range(n)]``) scales super-linearly on splits
    with Image(decode=False) columns — each call re-enters HF's batch
    lookup path, so materializing 1000+ rows takes ~30 min instead of ~30 s.
    """
    ds_view = ds.select_columns([c for c in fields if c in ds.column_names])
    ds_view = ds_view.with_format("numpy")
    cols = ds_view.column_names
    batch = ds_view[:]
    n = len(ds_view)
    return [{k: batch[k][i] for k in cols} for i in range(n)]


def _load_obj_id_to_category(split_json: str) -> dict[str, str]:
    """Read a LARM-style ``train/test -> category -> obj_id`` split JSON and
    return a flat ``{obj_id: category}`` map. Same obj_id under different
    splits maps to the same category, so the combined view is unambiguous.
    """
    with open(split_json) as f:
        split = json.load(f)
    mapping: dict[str, str] = {}
    for split_name in ("train", "test"):
        for cat, objs in split.get(split_name, {}).items():
            for obj_id in objs.keys():
                mapping[str(obj_id)] = str(cat)
    return mapping


_BASE_FIELDS = (
    "sample_id", "obj_id", "joint_name", "joint_type",
    "saved_state",
    "is_part_revolute", "is_part_prismatic",
    "revolute_plucker", "revolute_range",
    "prismatic_axis", "prismatic_range",
    "motion_hierarchy",
    "norm_center", "norm_scale",
    "pcd_ctx", "pcd_part_raw",
    "pcd_full_raw", "pcd_full_part_mask",
    # Cameras: new schema (cam_c2w_s0 / cam_c2w_s1) and the legacy
    # single-array schema (cam_c2w). select_columns() drops whichever
    # the on-disk dataset doesn't have, so listing both is safe.
    "cam_c2w", "cam_c2w_s0", "cam_c2w_s1",
    "active_visible_frac_s0", "active_visible_frac_s1",
    "cam_K", "cam_width", "cam_height",
)


class BaseDataset(Dataset):
    """Full-cloud reconstruction dataset at a random articulation state.

    Args:
        hf_path: directory passed to ``datasets.load_from_disk``.
        split: ``"train"`` or ``"test"`` key inside the loaded DatasetDict.
        num_points: target size of the returned ``pts`` cloud.
        sampling: ``"merged"`` (default; uniform density across parts via the
            pre-baked merged sample) or ``"per_part"`` (resample ``pcd_ctx``
            and ``pcd_part_raw`` independently and preserve their on-disk
            ratio — produces per-part density mismatches but keeps both
            counts deterministic).
        part_ratio: only used when ``sampling="per_part"``. ``None`` (default)
            keeps the on-disk ratio. Otherwise interpreted as ``part / ctx``,
            so ``0.5`` gives 2 ctx : 1 part, ``1.0`` gives 50/50, ``2.0``
            gives 1 ctx : 2 part.
        state_mode: ``"uniform"`` (default), ``"discrete"``
            (``{0, 0.25, 0.5, 0.75, 1}``), or ``"endpoints"`` (``{0, 1}``).
        augment: apply a random SO(3) rotation in world frame before
            ``world_to_cam``; rotates the Plücker / prismatic axis / cameras
            consistently. The applied rotation is returned as ``aug_R``.
        target_median: ``median_3`` target scale (default 3.0 to match the
            LARM-era BaseDataset).
        load_in_memory: materialize every row's arrays into a Python list at
            init time so ``__getitem__`` does no Arrow reads.
        repeat: multiply dataset length (handy when sampling many states
            per physical sample per epoch).
        max_samples: if set, restrict to the first ``N`` samples of the
            split (after `datasets.select(range(N))`). Useful for overfit /
            smoke configs where you want to iterate one or a handful of
            samples many times.
        exclude_categories: categories (by LARM-split label, e.g. ``"Oven"``)
            to drop from this split. Matched via ``obj_id`` using the JSON at
            ``split_json``. Samples whose ``obj_id`` is not present in the
            JSON are kept (and get an empty ``category`` string).
        split_json: path to the LARM split JSON (``train/test -> category ->
            obj_id``). Used to tag every sample with ``category`` and to
            resolve ``exclude_categories``. If ``None``, both features are
            disabled and ``category`` is returned as ``""``.
        seed: deterministic state + augment draws when > 0.
    """

    def __init__(
        self,
        hf_path: str,
        split: str = "train",
        num_points: int = 30000,
        sampling: str = "merged",
        part_ratio: float | None = None,
        state_mode: str = "uniform",
        augment: bool = False,
        target_median: float = 3.0,
        load_in_memory: bool = True,
        repeat: int = 1,
        max_samples: int | None = None,
        exclude_categories: Iterable[str] | None = None,
        split_json: str | None = None,
        seed: int = 0,
    ):
        super().__init__()
        self.split = split
        self.num_points = int(num_points)
        self.sampling = str(sampling)
        self.part_ratio = None if part_ratio is None else float(part_ratio)
        self.state_mode = state_mode
        self.augment = bool(augment)
        self.target_median = float(target_median)
        self.repeat = max(1, int(repeat))
        self._seed = int(seed)

        excl = set(str(c) for c in (exclude_categories or []))
        self._obj_id_to_cat: dict[str, str] = (
            _load_obj_id_to_category(split_json) if split_json else {}
        )

        dsd = load_from_disk(hf_path)
        ds = dsd[split] if hasattr(dsd, "__getitem__") else dsd
        if max_samples is not None:
            ds = ds.select(range(min(int(max_samples), len(ds))))

        if excl:
            if not self._obj_id_to_cat:
                raise ValueError(
                    "exclude_categories requires split_json to map obj_id -> "
                    "category, but split_json was not provided."
                )
            obj_ids = ds["obj_id"]
            keep = [i for i, oid in enumerate(obj_ids)
                    if self._obj_id_to_cat.get(str(oid), "") not in excl]
            ds = ds.select(keep)

        self._ds = ds
        if load_in_memory:
            self._rows = _materialize_rows(ds, _BASE_FIELDS)
            self._ds_np = None
        else:
            self._rows = None
            # Project to just the Stage-1 columns + numpy format ONCE so the
            # cold per-item read does no image decode and no per-call format
            # setup. Previously ``_row`` called ``self._ds.with_format("numpy")``
            # on every __getitem__ and read every column — including PIL
            # Image features Stage 1 doesn't use. See
            cols = [c for c in _BASE_FIELDS if c in ds.column_names]
            self._ds_np = ds.select_columns(cols).with_format("numpy")

    def __len__(self) -> int:
        base = len(self._rows) if self._rows is not None else len(self._ds)
        return base * self.repeat

    def _row(self, idx: int) -> dict:
        base = len(self._rows) if self._rows is not None else len(self._ds)
        i = idx % base
        if self._rows is not None:
            return self._rows[i]
        return dict(self._ds_np[i])

    def __getitem__(self, idx: int) -> dict:
        row = self._row(idx)
        # Fresh per-call randomness so that across epochs the model sees
        # different view picks and different point subsamples for the same
        # idx. Using a deterministic ``(seed, idx)`` here turns training into
        # a fixed (idx -> single-instance) map and starves stage-2 of view
        # augmentation, which severely hurts image-to-3D learning.
        rng = random.Random()
        np_rng = np.random.default_rng()
        # Pick a random reference camera per call so the encoder sees a
        # different canonical view of the same object across epochs. v2 rows
        # carry cam_c2w_s0 + cam_c2w_s1 and build_sample_payload concats them
        # ([s0_views..., s1_views...]) so V_total = s0_count + s1_count;
        # legacy v1 rows use a single cam_c2w array. Validation/test stays
        # deterministic on view 0 to keep eval CD comparable across runs.
        if self.split == "train":
            if row.get("cam_c2w_s0") is not None and row.get("cam_c2w_s1") is not None:
                v_total = int(np.asarray(row["cam_c2w_s0"]).shape[0]
                              + np.asarray(row["cam_c2w_s1"]).shape[0])
            else:
                v_total = int(np.asarray(row["cam_c2w"]).shape[0])
            ref_view_idx = rng.randrange(v_total)
        else:
            ref_view_idx = 0
        payload = build_sample_payload(
            row=row,
            num_points=self.num_points,
            state_mode=self.state_mode,
            augment=self.augment and self.split == "train",
            target_median=self.target_median,
            rng=rng,
            np_rng=np_rng,
            sampling=self.sampling,
            part_ratio=self.part_ratio,
            ref_view_idx=ref_view_idx,
        )
        payload["category"] = self._obj_id_to_cat.get(str(row.get("obj_id", "")), "")
        return payload
