"""Stage-2 dataset for the articulated-object MVP.

Same on-disk cache as :class:`src.datasets.base_dataset.BaseDataset` but
additionally returns ``num_views_per_state`` randomly-selected views at
state ``s0`` *and* at state ``s1``, stacked together as ``[2*V, 3, H, W]``
with a matching ``state_tag``. Cameras are mapped into the same final frame
as the point clouds (first-camera-centered + ``median_3`` scaled), so image
conditioning stays consistent with the geometric supervision.
"""
from __future__ import annotations

import io
import json
import random
from typing import Sequence

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image as PILImage
from torch.utils.data import Dataset

from typing import Iterable

from src.datasets.base_dataset import (
    _BASE_FIELDS,
    _load_obj_id_to_category,
    _materialize_rows,
    build_sample_payload,
)


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------


def _decode_image_record(record, resolution: int) -> tuple[np.ndarray, float, float]:
    """Decode an HF ``Image``-feature record into a ``[3, H, W]`` float32
    array in ``[0, 1]``. Returns ``(image, sx, sy)`` where ``sx, sy`` are
    the x/y resize ratios used (so intrinsics can be rescaled).

    RGBA images are flattened onto a white background (matches the
    LARM-era ArticODataset; our renders have transparent backgrounds so
    the composited white bg is cleaner than PIL's default alpha-drop).
    """
    if isinstance(record, dict):
        data = record.get("bytes")
        if data is None:
            path = record.get("path")
            img = PILImage.open(path)
        else:
            img = PILImage.open(io.BytesIO(data))
    elif isinstance(record, PILImage.Image):
        img = record
    elif isinstance(record, np.ndarray):
        # HF's `.with_format("numpy")` path — the Image feature auto-decoded
        # to a uint8 ndarray already.
        arr = record
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        img = PILImage.fromarray(arr.astype(np.uint8),
                                 mode="RGBA" if arr.shape[-1] == 4 else "RGB")
    else:
        raise TypeError(f"Unexpected image record type: {type(record)}")

    W0, H0 = img.size
    if img.mode == "RGBA":
        bg = PILImage.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if resolution and (W0, H0) != (resolution, resolution):
        img = img.resize((resolution, resolution), PILImage.BICUBIC)
        sx = resolution / float(W0)
        sy = resolution / float(H0)
    else:
        sx = sy = 1.0

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # [H,W,3] -> [3,H,W]
    return arr, float(sx), float(sy)


# ---------------------------------------------------------------------------
# Stage-2 dataset
# ---------------------------------------------------------------------------


_ARTIC_O_FIELDS = _BASE_FIELDS + (
    "images_s0", "images_s1",
)


class ArticODataset(Dataset):
    """Image-conditioned reconstruction dataset.

    Returns everything :class:`BaseDataset` does, plus:

    - ``image``: ``[2*V, 3, H, W]`` float32 in ``[0, 1]``; the first ``V``
      rows are random views from ``s0``, the next ``V`` from ``s1``.
    - ``c2w``: ``[2*V, 4, 4]`` mapped into the same first-camera +
      ``median_3`` scaled frame used for ``pts``.
    - ``fxfycxcy``: ``[2*V, 4]`` intrinsics rescaled to the loader resolution.
    - ``state_tag``: ``[2*V]`` int8 (0 for s0 views, 1 for s1 views).

    Args mirror :class:`BaseDataset` with the image additions:

        num_views_per_state: number of random views drawn per state.
        views_source: which states to draw views from. ``"both"`` (default)
            picks ``num_views_per_state`` from each of s0 and s1. ``"s0"``
            picks ``2 * num_views_per_state`` from s0 only; ``"s1"``
            likewise from s1 only. Total view count is preserved.
        anchor_state: which state's first picked view becomes the reference
            frame the model sees first. ``"s0"`` (default) keeps the legacy
            ordering of ``[s0_views..., s1_views...]`` with state_tag
            ``[0,...,1,...]``. ``"s1"`` reorders to ``[s1_views..., s0_views...]``
            with state_tag ``[1,...,0,...]`` so the anchor image (and the
            target reference frame) sits in s1. ``state_mode`` is independent
            and still controls which articulation state the GT cloud lives at.
        resolution: isotropic image size after resize. Set to ``0`` to keep
            the native resolution.
    """

    def __init__(
        self,
        hf_path: str,
        split: str = "train",
        num_points: int = 30000,
        sampling: str = "merged",
        part_ratio: float | None = None,
        num_views_per_state: int = 3,
        views_source: str = "both",
        anchor_state: str = "s0",
        resolution: int = 224,
        state_mode: str = "uniform",
        augment: bool = False,
        target_median: float = 3.0,
        load_in_memory: bool = True,
        repeat: int = 1,
        max_samples: int | None = None,
        exclude_categories: Iterable[str] | None = None,
        split_json: str | None = None,
        min_active_visible_frac: float = 0.0,
        seed: int = 0,
        val_view_picks_path: str | None = None,
        force_two_state_tag: bool = False,
    ):
        super().__init__()
        if views_source not in ("both", "s0", "s1"):
            raise ValueError(
                f"views_source must be one of 'both', 's0', 's1'; got {views_source!r}"
            )
        if anchor_state not in ("s0", "s1"):
            raise ValueError(
                f"anchor_state must be one of 's0', 's1'; got {anchor_state!r}"
            )
        self.split = split
        self.num_points = int(num_points)
        self.sampling = str(sampling)
        self.part_ratio = None if part_ratio is None else float(part_ratio)
        self.num_views_per_state = int(num_views_per_state)
        self.views_source = views_source
        self.anchor_state = anchor_state
        # Ablation knob (Reviewer-1 "prior vs. inter-state difference"): when
        # True, the returned ``state_tag`` is forced to the two-state pattern
        # [0]*k + [1]*k regardless of ``views_source``. Combined with
        # ``views_source=s1`` this feeds one-state pixels while telling the
        # encoder the first k views are s0 — the "fooled into two identical
        # states" probe. No effect (default) when False.
        self.force_two_state_tag = bool(force_two_state_tag)
        self.resolution = int(resolution)
        self.state_mode = state_mode
        self.augment = bool(augment)
        self.target_median = float(target_median)
        self.repeat = max(1, int(repeat))
        # Per-state visibility filter: only views with
        # ``active_visible_frac >= τ`` are eligible to be picked. τ=0.0 is a
        # no-op (matches legacy behavior). When the filtered pool is empty for
        # a state, we fall back to all views for that state — better to train
        # on a stale view than to silently drop the sample mid-epoch.
        self.min_active_visible_frac = float(min_active_visible_frac)
        self._seed = int(seed)

        # Pre-computed deterministic per-sample view picks. When set, val
        # ``__getitem__`` uses these indices instead of randomly sampling
        # views — eliminates run-to-run val noise (~3pp axis_angle drift)
        # and matches LARM's KMeans++-seeded view-selection strategy on
        # our renders. See ``scripts/build_val_view_picks.py``. The JSON
        # is sample_id → {"s0": [i0, i1, i2], "s1": [j0, j1, j2]}.
        # ``num_views_per_state`` must match the stored pick counts.
        # Only consulted when ``self.split != "train"``; train always
        # uses fresh random for view augmentation.
        self._val_view_picks: dict[str, dict[str, list[int]]] | None = None
        if val_view_picks_path:
            with open(val_view_picks_path) as f:
                payload = json.load(f)
            self._val_view_picks = payload.get("picks", payload)

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

        # Disable Image auto-decode so the formatter returns raw
        # {bytes, path} dicts instead of PIL-decoding each 512×512 RGBA PNG
        # into a ~1 MB uint8 ndarray. With 64 views/row that keeps
        # materialization at ~14 MB/row instead of ~64 MB/row.
        #
        # We mutate ``_info.features`` in place rather than calling
        # ``cast_column(HFImage(decode=False))``. cast_column rewrites the
        # column through pyarrow, and the combined image buffer exceeds the
        # ``binary`` type's int32 offset limit (~2 GB) on splits this big —
        # it raised ``ArrowInvalid: offset overflow while concatenating
        # arrays``. The in-place flag change is a metadata-only toggle: no
        # column rewrite, no transient memory spike, no overflow.
        #
        # Memory footprint at load_in_memory=True on the full train split
        # (1097 rows, 4-rank DDP): ~15 GB steady-state per rank (~60 GB
        # total). No 4× init-time spike from the cast rewrite either.
        for c in ("images_s0", "images_s1"):
            if c in ds._info.features:
                ds._info.features[c].feature.decode = False

        self._ds = ds
        if load_in_memory:
            self._rows = _materialize_rows(ds, _ARTIC_O_FIELDS)
            self._ds_np = None
        else:
            self._rows = None
            cols = [c for c in _ARTIC_O_FIELDS if c in ds.column_names]
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

    def _eligible_indices(self, n: int, visibility) -> list[int]:
        """Indices in ``[0, n)`` whose ``active_visible_frac >= τ``.

        Falls back to all ``[0, n)`` when the threshold is 0, the visibility
        array is missing, or no view passes the threshold (rare: small active
        part invisible from every camera).
        """
        if n <= 0:
            return []
        if self.min_active_visible_frac <= 0.0 or visibility is None:
            return list(range(n))
        v = np.asarray(visibility, dtype=np.float32).reshape(-1)
        if v.size != n:
            return list(range(n))
        eligible = [i for i in range(n) if v[i] >= self.min_active_visible_frac]
        return eligible if eligible else list(range(n))

    def _pick_view_indices(
        self,
        images_s0: Sequence,
        images_s1: Sequence,
        rng: random.Random,
        visible_s0=None,
        visible_s1=None,
    ) -> tuple[list[int], list[int]]:
        """Draw view indices for s0 and s1 according to ``views_source``.

        - ``"both"``: ``num_views_per_state`` from each state (legacy).
        - ``"s0"``: ``2 * num_views_per_state`` from s0, none from s1.
        - ``"s1"``: ``2 * num_views_per_state`` from s1, none from s0.

        Picks happen *before* the payload is built so that the first picked
        index becomes the reference frame passed to
        :func:`build_sample_payload`. This guarantees the first image fed to
        VGGT lives at identity c2w and supervision is in the same frame.

        When ``min_active_visible_frac > 0`` the candidate pool is restricted
        to indices whose per-view ``active_visible_frac`` clears the
        threshold; the actual *count* drawn (k or 2k) is unchanged.
        """
        k = self.num_views_per_state
        pool_s0 = self._eligible_indices(len(images_s0), visible_s0)
        pool_s1 = self._eligible_indices(len(images_s1), visible_s1)

        def _draw(pool: list[int], count: int) -> list[int]:
            if count <= 0 or len(pool) == 0:
                return []
            if len(pool) >= count:
                return rng.sample(pool, count)
            # With-replacement when the eligible pool is smaller than the
            # required count.
            return [rng.choice(pool) for _ in range(count)]

        if self.views_source == "both":
            return _draw(pool_s0, k), _draw(pool_s1, k)
        if self.views_source == "s0":
            return _draw(pool_s0, 2 * k), []
        # views_source == "s1"
        return [], _draw(pool_s1, 2 * k)

    def _decode_picked(
        self,
        images_s0: Sequence,
        images_s1: Sequence,
        idx_s0: Sequence[int],
        idx_s1: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        imgs, sxs, sys = [], [], []
        for i in idx_s0:
            a, sx, sy = _decode_image_record(images_s0[i], self.resolution)
            imgs.append(a); sxs.append(sx); sys.append(sy)
        for i in idx_s1:
            a, sx, sy = _decode_image_record(images_s1[i], self.resolution)
            imgs.append(a); sxs.append(sx); sys.append(sy)

        image = np.stack(imgs, axis=0).astype(np.float32)          # [2V,3,H,W]
        picks = np.asarray(list(idx_s0) + list(idx_s1), dtype=np.int64)
        tags = np.asarray([0] * len(idx_s0) + [1] * len(idx_s1), dtype=np.int8)
        sxs_arr = np.asarray(sxs, dtype=np.float32)
        sys_arr = np.asarray(sys, dtype=np.float32)
        return image, picks, tags, sxs_arr, sys_arr

    def __getitem__(self, idx: int) -> dict:
        row = self._row(idx)
        # Fresh per-call randomness so that across epochs the model sees
        # different view picks and different point subsamples for the same
        # idx. Using a deterministic ``(seed, idx)`` here turns training into
        # a fixed (idx -> single-instance) map and starves stage-2 of view
        # augmentation, which severely hurts image-to-3D learning.
        rng = random.Random()
        np_rng = np.random.default_rng()

        # Pick views first: the first s0 pick is the reference frame so that
        # the first image the model sees sits at identity c2w. This must
        # happen BEFORE build_sample_payload so it can place the world→cam
        # transform at the right view.
        # Val path: if precomputed deterministic picks are configured AND
        # we're not on the train split, look up by sample_id and use those
        # indices instead of random sampling.
        precomputed_picks = None
        if (
            self._val_view_picks is not None
            and self.split != "train"
        ):
            sid = str(row.get("sample_id", "") or "")
            if sid and sid in self._val_view_picks:
                precomputed_picks = self._val_view_picks[sid]

        if precomputed_picks is not None:
            idx_s0 = list(int(i) for i in precomputed_picks.get("s0", []))
            idx_s1 = list(int(i) for i in precomputed_picks.get("s1", []))
            # Sanity: pick counts must agree with config so downstream
            # tensor shapes don't surprise. If they disagree the JSON was
            # built with a different ``num_views_per_state``.
            k = self.num_views_per_state
            expect = {
                "both": (k, k),
                "s0":   (2 * k, 0),
                "s1":   (0, 2 * k),
            }[self.views_source]
            if (len(idx_s0), len(idx_s1)) != expect:
                raise ValueError(
                    f"val_view_picks for sample_id={sid!r} has "
                    f"({len(idx_s0)}, {len(idx_s1)}) picks, but "
                    f"views_source={self.views_source!r} + "
                    f"num_views_per_state={k} expects {expect}"
                )
        else:
            idx_s0, idx_s1 = self._pick_view_indices(
                row["images_s0"], row["images_s1"], rng,
                visible_s0=row.get("active_visible_frac_s0"),
                visible_s1=row.get("active_visible_frac_s1"),
            )
        # Reference view (anchor): the first picked image of the anchor state.
        # Per-state cameras concat to [2*n_s0, 4, 4] in [s0..., s1...] order,
        # so s1 picks must be offset by n_s0 to address the right rows.
        n_s0 = int(len(row["images_s0"]))
        if self.anchor_state == "s1" and len(idx_s1) > 0:
            ref_view_idx = int(idx_s1[0]) + n_s0
        elif len(idx_s0) > 0:
            ref_view_idx = int(idx_s0[0])
        elif len(idx_s1) > 0:
            ref_view_idx = int(idx_s1[0]) + n_s0
        else:
            raise RuntimeError(f"no views picked for sample idx={idx}")

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

        image_np, picks, tags, sxs, sys = self._decode_picked(
            row["images_s0"], row["images_s1"], idx_s0, idx_s1,
        )

        # _decode_picked stacks images as [s0..., s1...] with state_tag = [0..., 1...].
        # When anchor_state == "s1", we want the first images the model sees to be
        # from s1 (where the active part is most extended), so reorder to
        # [s1..., s0...]. The reorder permutation is also applied to picks_global
        # below so c2w_picked stays row-aligned with images.
        if self.anchor_state == "s1" and len(idx_s1) > 0 and len(idx_s0) > 0:
            n0_pick = len(idx_s0)
            n1_pick = len(idx_s1)
            view_perm = np.concatenate([
                np.arange(n0_pick, n0_pick + n1_pick, dtype=np.int64),
                np.arange(0, n0_pick, dtype=np.int64),
            ])
            image_np = image_np[view_perm]
            picks = picks[view_perm]
            tags = tags[view_perm]
            sxs = sxs[view_perm]
            sys = sys[view_perm]

        # Pick the right rows out of cam_c2w_final. With per-state cameras
        # the array is [s0_0..s0_{N-1}, s1_0..s1_{N-1}]; s1 picks must be
        # offset by N. With the legacy single-array schema the shape is
        # [N, 4, 4] (concat-with-self never happened) and no offset is
        # needed — detect by comparing total length to len(images_s0).
        c2w_final = payload["cam_c2w_final"].numpy()
        per_state_layout = (c2w_final.shape[0] == 2 * n_s0)
        if per_state_layout:
            s0_global = np.asarray(idx_s0, dtype=np.int64)
            s1_global = np.asarray(idx_s1, dtype=np.int64) + n_s0
            if self.anchor_state == "s1":
                picks_global = np.concatenate([s1_global, s0_global])
            else:
                picks_global = np.concatenate([s0_global, s1_global])
        else:
            picks_global = picks
        c2w_picked = c2w_final[picks_global]
        c2w_first_cam_full = payload["cam_c2w_first_cam"].numpy()
        c2w_first_cam_picked = c2w_first_cam_full[picks_global]

        # Ablation: overwrite the state tag with the two-state pattern while
        # keeping whatever pixels were picked (see __init__). Total view count
        # is always 2*k, so [0]*k + [1]*k is well-defined.
        if self.force_two_state_tag:
            k = self.num_views_per_state
            tags = np.asarray([0] * k + [1] * k, dtype=np.int8)

        K = np.asarray(row["cam_K"], dtype=np.float32)
        fx0, fy0, cx0, cy0 = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        fxfycxcy = np.stack([
            fx0 * sxs, fy0 * sys, cx0 * sxs, cy0 * sys,
        ], axis=-1).astype(np.float32)

        payload.update({
            "image": torch.from_numpy(image_np),
            "c2w": torch.from_numpy(c2w_picked.astype(np.float32)),
            "c2w_first_cam": torch.from_numpy(c2w_first_cam_picked.astype(np.float32)),
            "fxfycxcy": torch.from_numpy(fxfycxcy),
            "state_tag": torch.from_numpy(tags),
            "view_indices": torch.from_numpy(picks),
        })
        payload["category"] = self._obj_id_to_cat.get(str(row.get("obj_id", "")), "")
        return payload
