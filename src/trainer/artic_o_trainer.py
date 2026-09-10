"""Stage-2 articulated trainer.

Inherits from :class:`ReconTrainer` and augments per-step training with a
second forward at ``t = 1`` whose query points are the GT point cloud.
Per-point features captured at the FM decoder's ``linear_out`` are passed
to the seg head, and the seg loss is added to the standard FM loss.

Phase 3 supervises the seg head with binary cross-entropy on
``pts_part_mask``. Phase 4 will add dice; Phase 5 will add articulation
parameter heads + losses.

Validation does a single dataloader pass per epoch: encode each batch
once, run the ODE solver for geometry (CD/F1) AND a single ``t=1`` decode
for seg (acc/IoU), all sharing the same encoder output. Both sets of
metrics are logged together under the same overall/ood/per_class
grouping, and per-step PLY dumps (geometry + seg) land in the same
``step_<N>/`` directory.
"""
from __future__ import annotations

import gc
import json
import logging
import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points
from tqdm import tqdm

from src.datasets.articulation import (
    articulation_metrics_prismatic,
    articulation_metrics_revolute,
    plucker_to_4x4_torch,
    prismatic_to_4x4_torch,
)
from src.datasets.utils import (
    scale_shift_alignment_pointcloud,
)
from src.flow_matching.solver import ODESolver
from src.models.heads.matcher import hungarian_match_movable
from src.models.heads.articulation import (
    articulation_losses,
    articulation_losses_per_point_closest,
    axis_dir_and_point_to_plucker,
)
from src.models.model_wrapper import BatchModelWrapper
from src.trainer.recon_trainer import ReconTrainer

logger = logging.getLogger(__name__)


def _per_part_filter_indices(
    xyz: np.ndarray, seg: np.ndarray, *,
    nb_neighbors: int, std_ratio: float,
) -> np.ndarray:
    """Statistical outlier filter applied SEPARATELY to ctx (seg==0) and active
    (seg==1). Returns sorted indices of points to keep. Mirrors
    ``src/viz/forward_pass.py::_per_part_filter_indices`` so val-time and
    snapshot-time post-processing stay aligned.
    """
    import open3d as o3d
    keep: list[int] = []
    for label in (0, 1):
        idx = np.where(seg == label)[0]
        if len(idx) == 0:
            continue
        if len(idx) < int(nb_neighbors) + 1:
            keep.extend(idx.tolist())
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[idx].astype(np.float64))
        _, kept = pcd.remove_statistical_outlier(
            nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio),
        )
        keep.extend(idx[np.asarray(kept, dtype=np.int64)].tolist())
    if not keep:
        return np.arange(xyz.shape[0], dtype=np.int64)
    return np.asarray(sorted(set(keep)), dtype=np.int64)


def _subsample_seg_ratio_indices(
    seg: np.ndarray, n_target: int, rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n_target`` indices from ``seg`` keeping the ctx/active count
    ratio of the input. Returns absolute indices into ``seg``."""
    n_full = int(seg.shape[0])
    if n_full <= n_target:
        return np.arange(n_full, dtype=np.int64)
    n_ctx = int((seg == 0).sum())
    n_act = int((seg == 1).sum())
    total = max(n_ctx + n_act, 1)
    n_ctx_t = int(round(n_target * n_ctx / total))
    n_act_t = n_target - n_ctx_t
    ctx_idx = np.where(seg == 0)[0]
    act_idx = np.where(seg == 1)[0]
    n_ctx_t = min(n_ctx_t, len(ctx_idx))
    n_act_t = min(n_act_t, len(act_idx))
    parts = []
    if n_ctx_t > 0:
        parts.append(rng.choice(ctx_idx, n_ctx_t, replace=False))
    if n_act_t > 0:
        parts.append(rng.choice(act_idx, n_act_t, replace=False))
    return np.concatenate(parts).astype(np.int64) if parts else np.empty((0,), dtype=np.int64)


def _binary_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """IoU(pred, gt) over a single binary mask. Returns NaN if both are empty."""
    inter = (pred_mask & gt_mask).float().sum()
    union = (pred_mask | gt_mask).float().sum()
    if union.item() == 0:
        return torch.tensor(float("nan"), device=pred_mask.device)
    return inter / union


def _binary_seg_scores(pred_lbl: torch.Tensor, gt_lbl: torch.Tensor) -> dict:
    """Macro per-sample binary (static=0 / movable=1) IoU-based scores.

    Defined for every sample that has a GT movable part (union non-empty),
    so the bucket ``nanmean`` denominators stay consistent across metrics.
    Movable precision/recall/F1 are NOT returned here — they are aggregated
    **micro** (pooled tp/fp/fn) at the bucket level to avoid per-sample NaN
    / survivorship bias (a sample that predicts no movable point has an
    undefined precision that would otherwise silently drop from the mean).
    """
    pred_mov = pred_lbl == 1
    gt_mov = gt_lbl == 1
    iou_mov = float(_binary_iou(pred_mov, gt_mov))
    iou_sta = float(_binary_iou(~pred_mov, ~gt_mov))
    defined = [v for v in (iou_mov, iou_sta) if not math.isnan(v)]
    miou = float(sum(defined) / len(defined)) if defined else float("nan")
    acc = float((pred_lbl == gt_lbl).float().mean())
    return {
        "seg_iou_movable": iou_mov,
        "seg_iou_static": iou_sta,
        "seg_miou": miou,
        "seg_accuracy": acc,
    }


def _binary_seg_counts(pred_lbl: torch.Tensor, gt_lbl: torch.Tensor) -> tuple:
    """Movable-class (tp, fp, fn) counts for micro precision/recall/F1."""
    pred_mov = pred_lbl == 1
    gt_mov = gt_lbl == 1
    tp = float((pred_mov & gt_mov).sum())
    fp = float((pred_mov & ~gt_mov).sum())
    fn = float((~pred_mov & gt_mov).sum())
    return tp, fp, fn


class ArticOTrainer(ReconTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        loss_cfg = self.config.get("articulated_loss", {}) or {}
        # Weight applied to seg CE loss when summed with the FM loss:
        #   loss = loss_fm + seg_weight * loss_seg + articulation_loss_terms
        self.seg_loss_weight = float(loss_cfg.get("seg", 1.0))

        # Phase-5 articulation loss weights. Each axis / range head's loss
        # is gated by GT motion type at sample level (see ``articulation_losses``).
        self.art_rev_plucker_w = float(loss_cfg.get("rev_plucker", 1.0))
        self.art_rev_range_w = float(loss_cfg.get("rev_range", 1.0))
        self.art_pris_axis_w = float(loss_cfg.get("pris_axis", 1.0))
        self.art_pris_range_w = float(loss_cfg.get("pris_range", 1.0))
        self.art_unit_w = float(loss_cfg.get("axis_unit", 1.0))    # multiplies the unit-norm priors
        self.art_orth_w = float(loss_cfg.get("plucker_orth", 1.0))
        # PARTICULATE-style ``per_point_closest`` revolute representation:
        # 3-D axis direction + per-point closest-point-on-axis. Default
        # weights match ``rev_plucker`` / ``rev_axis_dir`` semantics.
        self.art_rev_axis_dir_w = float(loss_cfg.get("rev_axis_dir", 1.0))
        self.art_rev_closest_pt_w = float(loss_cfg.get("rev_closest_pt", 1.0))
        # Multi-part (rebuttal): >2 seg classes, Hungarian-matched per-part
        # supervision. Skips the LARM-bucketed single-part val_epoch.
        self.multipart = bool(self.config.get("multipart", False))
        self.mp_dice_w = float(loss_cfg.get("dice", 1.0))
        # Per-state evaluation grid (Phase-5 val). Defaults match the user's
        # 5-point sweep: t ∈ {0, 0.25, 0.5, 0.75, 1}.
        self.eval_states = list(self.config.get("eval_states", [0.0, 0.25, 0.5, 0.75, 1.0]))
        # LARM-style metric thresholds (success-rate gates) — defaults from
        # the LARM paper (axis_angle in radians, axis_origin / Mr / Md):
        thr = self.config.get("articulation_metric_thresholds", {}) or {}
        self.metric_thr_axis_angle = float(thr.get("axis_angle", 0.25))
        self.metric_thr_axis_origin = float(thr.get("axis_origin", 0.15))
        self.metric_thr_Mr = float(thr.get("Mr", 0.3))
        self.metric_thr_Md = float(thr.get("Md", 0.3))

        # LARM cross-comparison: load the per-sample success record from
        # ``larm_exp/inference_work/inference_status.json`` so val can
        # report metrics on the SAME subset LARM's published table
        # covers. Two LARM evaluation variants exist on disk under
        # ``third_party/larm_exp/inference_work/``:
        #
        #   ``eval_results_hf`` (n=304, joint=255)
        #     LARM's mesh-based eval. Drops 8 samples whose part-mesh
        #     extraction failed (the eval needs both base + part
        #     meshes; samples with ``tsdf_part=False`` get pruned).
        #     The 8 missing: 4 degenerate TSDFs that still hallucinated
        #     a URDF (47817_j1, 48379_j0/1, 102316_j0) plus 4 partial-
        #     mesh failures (7128_j1, 101564_j10, 102389_j2, 102389_j7).
        #     Of the 304 evaluated, 255 have a valid URDF (strict
        #     4-stage criterion).
        #
        #   ``eval_results_hf_pcd`` (n=312, joint=259)
        #     LARM's point-cloud-based eval. Includes all 312 samples;
        #     of those, 259 have a valid URDF (urdf-only criterion;
        #     keeps the 4 degenerate-TSDF samples as joint-success).
        #
        # ``larm_eval_variant`` selects which of LARM's published rows
        # we line up with. Both variants derive from the same
        # ``larm_eval.json`` per-sample status:
        #   * hf:     evaluated = ``inference & tsdf_base & tsdf_part`` (304),
        #             with_joint = ``inference & tsdf_base & tsdf_part & urdf`` (255).
        #   * hf_pcd: evaluated = all (312),
        #             with_joint = ``urdf`` (259).
        # Default ``hf`` (the LARM mesh-based eval, mirrored in
        # ``docs/larm_results.txt``). Samples not in the variant's
        # ``evaluated`` set are skipped from every Val-* bucket so
        # ``success_rate_all`` denominators equal LARM's "ALL N"
        # column exactly.
        self.larm_success_ids: set[str] | None = None
        self.larm_fail_ids: set[str] | None = None
        self.larm_evaluated_ids: set[str] | None = None
        self.larm_eval_variant: str = "hf"
        larm_status_path = self.config.get(
            "larm_status_json_path",
            "docs/larm_eval.json",
        )
        # Backward-compat shim: if a config still sets ``larm_status_strict``,
        # interpret strict=True as variant="hf" (closest semantic match —
        # both pin to the strict 4-stage subset of 255 LARM-success
        # samples). Explicit ``larm_eval_variant`` always wins.
        if "larm_eval_variant" in self.config:
            variant = str(self.config["larm_eval_variant"]).lower()
        elif "larm_status_strict" in self.config:
            # Backward-compat shim. ``strict=False`` (the historical
            # urdf-only criterion) maps to hf_pcd; ``strict=True`` maps
            # to hf (4-stage success ⊂ inf+tsdf+tsdf eval set).
            variant = "hf" if self.config["larm_status_strict"] else "hf_pcd"
        else:
            variant = "hf"
        if variant not in ("hf", "hf_pcd"):
            logger.warning(
                f"[LARM-compare] unknown larm_eval_variant={variant!r}; "
                f"falling back to 'hf'"
            )
            variant = "hf"
        self.larm_eval_variant = variant

        if larm_status_path and os.path.exists(larm_status_path):
            try:
                with open(larm_status_path) as f:
                    status = json.load(f)
                evaluated: set[str] = set()
                success: set[str] = set()
                fail: set[str] = set()
                for entry in status.get("samples", []):
                    name = str(entry.get("name", ""))
                    if not name:
                        continue
                    if variant == "hf":
                        is_eval = all(
                            bool(entry.get(k, False))
                            for k in ("inference", "tsdf_base", "tsdf_part")
                        )
                        is_succ = is_eval and bool(entry.get("urdf", False))
                    else:  # hf_pcd
                        is_eval = True
                        is_succ = bool(entry.get("urdf", False))
                    if not is_eval:
                        continue
                    evaluated.add(name)
                    if is_succ:
                        success.add(name)
                    else:
                        fail.add(name)
                self.larm_evaluated_ids = evaluated
                self.larm_success_ids = success
                self.larm_fail_ids = fail
                logger.info(
                    f"[LARM-compare] loaded inference_status from {larm_status_path} "
                    f"(variant={variant}): evaluated={len(evaluated)}, "
                    f"success={len(success)}, fail={len(fail)}"
                )
            except Exception as e:
                logger.warning(f"[LARM-compare] failed to load {larm_status_path}: {e}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def _multipart_seg_art_loss(self, seg_out, part_ids, mp_gt, num_movable, input_pts):
        """Matched multi-part loss: Hungarian-matched seg CE + per-part
        articulation (per_point_closest), reusing the single-part loss helper
        via pseudo-batch flattening (one element per valid (object, part)).

        Returns (loss_seg, loss_art, target_col).
        """
        seg_logits = seg_out["seg_logits"]              # [B, N, 1+P0]
        B, N, C = seg_logits.shape
        P0 = C - 1
        device = seg_logits.device

        # Matching + per-point closest-pt were computed inside model.forward
        # (so DDP tracks point_axis_decoder). Read them here.
        target_col = seg_out["mp_target_col"]           # [B, N]
        matched_slot = seg_out["mp_matched_slot"]       # [B, P0]
        loss_seg_ce = F.cross_entropy(seg_logits.transpose(1, 2), target_col)
        # Multi-class soft dice over the 1+P0 columns, averaged over the classes
        # that actually appear in GT — counteracts the heavy base-vs-parts
        # imbalance that otherwise collapses seg to all-base.
        probs = torch.softmax(seg_logits.float(), dim=-1)          # [B, N, 1+P0]
        tgt_oh = F.one_hot(target_col, num_classes=C).float()      # [B, N, 1+P0]
        inter = (probs * tgt_oh).sum(dim=1)                        # [B, 1+P0]
        denom = probs.sum(dim=1) + tgt_oh.sum(dim=1)               # [B, 1+P0]
        dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)           # [B, 1+P0]
        present = tgt_oh.sum(dim=1) > 0                            # [B, 1+P0]
        loss_dice = (dice * present).sum() / present.sum().clamp_min(1)
        loss_seg = loss_seg_ce + self.mp_dice_w * loss_dice

        art = seg_out["articulation"]                   # per-slot dict incl closest_pt
        closest_pt = art["revolute_closest_pt"]         # [B, N, 3]
        ms_safe = matched_slot.clamp(min=0)             # [B, P0]

        # Pseudo-batch: one element per valid (object, part).
        valid_flat = mp_gt["valid"].reshape(-1)                     # [B*P0]
        idx_b = torch.arange(B, device=device).repeat_interleave(P0)[valid_flat]
        idx_j = torch.arange(P0, device=device).repeat(B)[valid_flat]
        if idx_b.numel() == 0:
            zero = torch.zeros((), device=device, dtype=loss_seg.dtype)
            return loss_seg, zero, target_col
        ms_flat = ms_safe.reshape(-1)[valid_flat]                  # [P_tot]

        def gather_slot(t):
            tb = t.reshape(B, P0, -1)                              # [B, P0, F]
            return tb[idx_b, ms_flat]                              # [P_tot, F]

        pred_pb = {
            "revolute_axis_dir": gather_slot(art["revolute_axis_dir"]),   # [P,3]
            "revolute_range": gather_slot(art["revolute_range"]).squeeze(-1),
            "prismatic_axis": gather_slot(art["prismatic_axis"]),         # [P,3]
            "prismatic_range": gather_slot(art["prismatic_range"]).squeeze(-1),
            "revolute_closest_pt": closest_pt[idx_b],                     # [P,N,3]
        }
        points_pb = input_pts[idx_b]                                       # [P,N,3]
        active_mask_pb = part_ids[idx_b] == (idx_j + 1).unsqueeze(1)       # [P,N]
        art_losses = articulation_losses_per_point_closest(
            pred_pb,
            is_revolute=mp_gt["is_revolute"].reshape(B * P0)[valid_flat],
            is_prismatic=mp_gt["is_prismatic"].reshape(B * P0)[valid_flat],
            gt_plucker=mp_gt["plucker"].reshape(B * P0, 6)[valid_flat],
            gt_revolute_range_hi=mp_gt["revolute_range_hi"].reshape(B * P0)[valid_flat],
            gt_prismatic_axis=mp_gt["prismatic_axis"].reshape(B * P0, 3)[valid_flat],
            gt_prismatic_range_hi=mp_gt["prismatic_range_hi"].reshape(B * P0)[valid_flat],
            points=points_pb,
            active_mask=active_mask_pb,
            plucker_unit_weight=self.art_unit_w,
        )
        loss_art = (
            self.art_rev_axis_dir_w * art_losses["loss_rev_axis_dir"]
            + self.art_rev_closest_pt_w * art_losses["loss_rev_closest_pt"]
            + art_losses["loss_rev_unit"]
            + self.art_rev_range_w * art_losses["loss_rev_range"]
            + self.art_pris_axis_w * art_losses["loss_pris_axis"]
            + art_losses["loss_pris_unit"]
            + self.art_pris_range_w * art_losses["loss_pris_range"]
        )
        return loss_seg, loss_art, target_col

    def train_epoch(
        self,
        epoch: int,
        train_bar: tqdm = None,
    ):
        gc.collect()
        self.model.train(True)

        # Gradient accumulation. Lets a memory-bound config (e.g. res-518, where
        # batch 8 OOMs) reach the same *effective* batch as the tuned recipe on
        # fewer GPUs: global = batch_size x n_gpu x accum_steps. Default 1 is an
        # exact no-op, so existing configs are unaffected.
        accum_steps = max(1, int(self.train_cfg.get("grad_accum_steps", 1)))
        # Logging reads grad_norm on every micro-batch, but it is only computed
        # on stepping ones; seed it so a non-stepping iteration reports the last
        # real value instead of raising NameError.
        grad_norm = float("nan")

        for data_iter_step, data in enumerate(self.train_dataloader):
            if data_iter_step % accum_steps == 0:
                self.optimizer.zero_grad()

            images = data["image"].to(self.device, non_blocking=True)
            input_pts = data["pts"].to(self.device, non_blocking=True)
            part_mask = data["pts_part_mask"].to(self.device, non_blocking=True)
            state_tag = data.get("state_tag", None)
            if state_tag is not None:
                state_tag = state_tag.to(self.device, non_blocking=True)

            # Phase-5 articulation GT (only used if model has the head).
            inner = self.model.module if hasattr(self.model, "module") else self.model
            has_art_head = getattr(inner, "articulation_head", None) is not None
            mp_gt = None
            num_movable = None
            if has_art_head and self.multipart:
                num_movable = data["num_movable"].to(self.device, non_blocking=True).long()
                mp_gt = {
                    "plucker": data["mp_plucker"].to(self.device, non_blocking=True),
                    "prismatic_axis": data["mp_prismatic_axis"].to(self.device, non_blocking=True),
                    "revolute_range_hi": data["mp_revolute_range_hi"].to(self.device, non_blocking=True),
                    "prismatic_range_hi": data["mp_prismatic_range_hi"].to(self.device, non_blocking=True),
                    "is_revolute": data["mp_is_revolute"].to(self.device, non_blocking=True).bool(),
                    "is_prismatic": data["mp_is_prismatic"].to(self.device, non_blocking=True).bool(),
                    "valid": data["mp_valid"].to(self.device, non_blocking=True).bool(),
                }
            elif has_art_head:
                gt_plucker = data["plucker"].to(self.device, non_blocking=True)         # [B, 6]
                gt_pris_axis = data["prismatic_axis"].to(self.device, non_blocking=True) # [B, 3]
                gt_rev_range = data["revolute_range"].to(self.device, non_blocking=True) # [B, 2]
                gt_pris_range = data["prismatic_range"].to(self.device, non_blocking=True) # [B, 2]
                is_rev = data["is_revolute"].to(self.device, non_blocking=True).bool()
                is_pris = data["is_prismatic"].to(self.device, non_blocking=True).bool()
                gt_rev_range_hi = gt_rev_range[:, 1]
                gt_pris_range_hi = gt_pris_range[:, 1]

            B, N = input_pts.shape[:2]

            # ----------------------------------------------------------
            # Single combined pass: FM loss + seg/articulation supervision
            # both ride on the same forward through the FM decoder.
            #
            # Why: the previous two-pass setup trained the seg/articulation
            # heads exclusively on FM-decoder features at (query=GT, t=1),
            # but at deployment val the heads see (query=ODE-pred, t=1) —
            # a different per-point distribution. By moving the seg head
            # onto pass 1's ``x_t = path(noise, GT, t)`` features, the
            # heads see the full noise schedule including the high-t
            # regime that matches inference. ``t1_anchor_frac`` (default
            # 0.0) optionally forces a fraction of samples to t=1.0 to
            # preserve a clean-query anchor when sampling random t alone
            # rarely lands at exactly 1.0.
            #
            # Particles are 1-1 indexed by ``path.sample`` so seg labels
            # (``part_mask``) remain valid for x_t. For per-point
            # closest-pt the GT projection uses ``input_pts`` (the GT
            # particle position) rather than x_t — a stable target across
            # t that matches what inference (with x ≈ GT) expects.
            # ----------------------------------------------------------
            t1_anchor_frac = float(self.train_cfg.get("t1_anchor_frac", 0.0))
            # ``two_pass_seg_art``: decouple FM and seg/art training.
            #   * Pass 1 (FM): t_fm sampled freely from skewed schedule
            #     (no t=1 anchor) — FM gets its full noise schedule.
            #   * Pass 2 (seg/art): t_sa sampled with the t1 anchor
            #     (controlled by ``t1_anchor_frac``) — seg/art mixes
            #     clean inference-matched queries (anchor at t=1) with
            #     noisy queries (random t).
            # Two passes through the FM decoder; encoder is run once and
            # ``encoder_data`` is shared. Adds ~15-20% step time vs the
            # merged-pass design.
            #
            # ``t1_anchor_fm_mask``: in merged-pass mode only — when true,
            # drop anchored samples from FM loss to avoid the degenerate
            # t=1 FM gradient. Logically subsumed by ``two_pass_seg_art``
            # but kept as a cheaper alternative for the merged-pass path.
            two_pass = bool(self.train_cfg.get("two_pass_seg_art", False))
            t1_anchor_fm_mask = bool(
                self.train_cfg.get("t1_anchor_fm_mask", False)
            )

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Encoder runs once; ``encoder_data`` is shared between
                # passes. We reach into the unwrapped module for
                # ``_encode`` because DDP only wraps ``forward``; the
                # gradient sync still happens at ``backward`` once the
                # loss touches the encoder parameters via the shared
                # tokens below.
                encoder_data = inner._encode(
                    images=images, pointmaps=None, state_tag=state_tag,
                )

                if two_pass:
                    # ---- Pass 1: FM only, t_fm random skewed (no anchor) --
                    if self.train_cfg.skewed_timesteps:
                        t_fm = self.skewed_timestep_sample(B, device=self.device)
                    else:
                        t_fm = torch.rand(B, device=self.device)
                    noise_fm = torch.rand_like(input_pts) * 2.0 - 1.0
                    path_fm = self.path.sample(t=t_fm, x_0=noise_fm, x_1=input_pts)
                    x_t_fm = path_fm.x_t
                    u_t_fm = path_fm.dx_t

                    fm_out = self.model(
                        images=images,
                        query_points=x_t_fm,
                        timestep=t_fm.unsqueeze(1),
                        run_seg=False,
                        encoder_data=encoder_data,
                    )
                    v_predict = fm_out["pts3d_xyz"]
                    loss_fm = torch.pow(v_predict - u_t_fm, 2).mean()

                    # ---- Pass 2: seg/art at t_sa ----------------------
                    # Two composable knobs control the t_sa distribution:
                    #   * ``t_sa_min`` (default 0.0): when > 0, the base
                    #     distribution is linearly remapped from [0, 1] into
                    #     [t_sa_min, 1.0]. Skew shape (heavy bias toward 1)
                    #     is preserved.
                    #   * ``t1_anchor_frac`` (default 0.0): a fraction of
                    #     samples are deterministically pinned to t=1.0
                    #     (the clean-GT-query anchor).
                    # Setting both to 0 gives the original skewed schedule.
                    t_sa_min = float(self.train_cfg.get("t_sa_min", 0.0))
                    if self.train_cfg.skewed_timesteps:
                        t_sa = self.skewed_timestep_sample(B, device=self.device)
                    else:
                        t_sa = torch.rand(B, device=self.device)
                    if t_sa_min > 0.0:
                        t_sa = t_sa_min + t_sa * (1.0 - t_sa_min)
                    anchor_mask = torch.zeros(B, device=self.device, dtype=torch.bool)
                    if t1_anchor_frac > 0.0:
                        anchor_mask = torch.rand(B, device=self.device) < t1_anchor_frac
                        t_sa = torch.where(anchor_mask, torch.ones_like(t_sa), t_sa)
                    noise_sa = torch.rand_like(input_pts) * 2.0 - 1.0
                    path_sa = self.path.sample(t=t_sa, x_0=noise_sa, x_1=input_pts)
                    x_t_sa = path_sa.x_t

                    seg_out = self.model(
                        images=images,
                        query_points=x_t_sa,
                        timestep=t_sa.unsqueeze(1),
                        run_seg=True,
                        encoder_data=encoder_data,
                        mp_part_ids=part_mask.long() if self.multipart else None,
                        mp_num_movable=num_movable if self.multipart else None,
                    )
                else:
                    # ---- Merged-pass (existing xt recipe) --------------
                    if self.train_cfg.skewed_timesteps:
                        t = self.skewed_timestep_sample(B, device=self.device)
                    else:
                        t = torch.rand(B, device=self.device)
                    anchor_mask = torch.zeros(B, device=self.device, dtype=torch.bool)
                    if t1_anchor_frac > 0.0:
                        anchor_mask = torch.rand(B, device=self.device) < t1_anchor_frac
                        t = torch.where(anchor_mask, torch.ones_like(t), t)
                    noise = torch.rand_like(input_pts) * 2.0 - 1.0
                    path_sample = self.path.sample(t=t, x_0=noise, x_1=input_pts)
                    x_t = path_sample.x_t
                    u_t = path_sample.dx_t
                    t_unsqueezed = t.unsqueeze(1)

                    seg_out = self.model(
                        images=images,
                        query_points=x_t,
                        timestep=t_unsqueezed,
                        run_seg=True,
                        encoder_data=encoder_data,
                        mp_part_ids=part_mask.long() if self.multipart else None,
                        mp_num_movable=num_movable if self.multipart else None,
                    )
                    v_predict = seg_out["pts3d_xyz"]
                    if t1_anchor_fm_mask and bool(anchor_mask.any()):
                        fm_keep = ~anchor_mask
                        if bool(fm_keep.any()):
                            loss_fm = torch.pow(
                                (v_predict - u_t)[fm_keep], 2
                            ).mean()
                        else:
                            loss_fm = torch.zeros(
                                (), device=self.device, dtype=v_predict.dtype,
                            )
                    else:
                        loss_fm = torch.pow(v_predict - u_t, 2).mean()

                if self.multipart:
                    seg_logits = seg_out["seg_logits"]   # [B, N, 1+P0]
                    seg_target = part_mask.long()         # [B, N] multiclass part ids
                    loss_seg, loss_art, mp_target_col = self._multipart_seg_art_loss(
                        seg_out, seg_target, mp_gt, num_movable, input_pts.float(),
                    )
                    art_losses = {}
                else:
                    seg_logits = seg_out["seg_logits"]  # [B, N, num_classes]
                    seg_target = part_mask.long()        # [B, N]
                    # CE expects class dim at position 1: [B, C, N].
                    loss_seg = F.cross_entropy(
                        seg_logits.transpose(1, 2), seg_target
                    )

                    # Phase-5 articulation losses (gated by GT motion type).
                    # Two paths matched to the head's motion_representation:
                    #   - 'plucker'           : legacy 6-D Plücker direct regression.
                    #   - 'per_point_closest' : 3-D axis + per-point closest-pt on axis.
                    art_losses: dict[str, torch.Tensor] = {}
                    loss_art = torch.zeros((), device=self.device, dtype=loss_seg.dtype)
                    motion_repr = (
                        inner.articulation_head.motion_representation
                        if has_art_head else "plucker"
                    )
                    if has_art_head and ("articulation" in seg_out):
                        if motion_repr == "plucker":
                            art_losses = articulation_losses(
                                seg_out["articulation"],
                                is_revolute=is_rev,
                                is_prismatic=is_pris,
                                gt_plucker=gt_plucker,
                                gt_revolute_range_hi=gt_rev_range_hi,
                                gt_prismatic_axis=gt_pris_axis,
                                gt_prismatic_range_hi=gt_pris_range_hi,
                                plucker_unit_weight=self.art_unit_w,
                                plucker_orth_weight=self.art_orth_w,
                            )
                            loss_art = (
                                self.art_rev_plucker_w * art_losses["loss_rev_plucker"]
                                + art_losses["loss_rev_orth"]      # already weighted internally
                                + art_losses["loss_rev_unit"]      # already weighted internally
                                + self.art_rev_range_w * art_losses["loss_rev_range"]
                                + self.art_pris_axis_w * art_losses["loss_pris_axis"]
                                + art_losses["loss_pris_unit"]     # already weighted internally
                                + self.art_pris_range_w * art_losses["loss_pris_range"]
                            )
                        else:
                            art_losses = articulation_losses_per_point_closest(
                                seg_out["articulation"],
                                is_revolute=is_rev,
                                is_prismatic=is_pris,
                                gt_plucker=gt_plucker,
                                gt_revolute_range_hi=gt_rev_range_hi,
                                gt_prismatic_axis=gt_pris_axis,
                                gt_prismatic_range_hi=gt_pris_range_hi,
                                points=input_pts.float(),
                                active_mask=(part_mask == 1),
                                plucker_unit_weight=self.art_unit_w,
                            )
                            loss_art = (
                                self.art_rev_axis_dir_w * art_losses["loss_rev_axis_dir"]
                                + self.art_rev_closest_pt_w * art_losses["loss_rev_closest_pt"]
                                + art_losses["loss_rev_unit"]      # already weighted internally
                                + self.art_rev_range_w * art_losses["loss_rev_range"]
                                + self.art_pris_axis_w * art_losses["loss_pris_axis"]
                                + art_losses["loss_pris_unit"]     # already weighted internally
                                + self.art_pris_range_w * art_losses["loss_pris_range"]
                            )

            # Diagnostics outside autocast (fp32 metrics).
            with torch.no_grad():
                seg_pred = seg_logits.argmax(dim=-1)            # [B, N]
                if self.multipart:
                    gt = mp_target_col                          # matched slot columns
                    seg_acc = (seg_pred == gt).float().mean()
                    ctx_mask = (gt == 0)
                    act_mask = (gt >= 1)
                    ctx_acc = (
                        (seg_pred[ctx_mask] == 0).float().mean()
                        if ctx_mask.any()
                        else torch.tensor(float("nan"), device=self.device)
                    )
                    act_acc = (
                        (seg_pred[act_mask] == gt[act_mask]).float().mean()
                        if act_mask.any()
                        else torch.tensor(float("nan"), device=self.device)
                    )
                    iou_act = _binary_iou(seg_pred >= 1, gt >= 1)
                    iou_ctx = _binary_iou(seg_pred == 0, gt == 0)
                    act_frac = act_mask.float().mean()
                else:
                    gt = seg_target
                    seg_acc = (seg_pred == gt).float().mean()
                    ctx_mask = (gt == 0)
                    act_mask = (gt == 1)
                    ctx_acc = (
                        (seg_pred[ctx_mask] == 0).float().mean()
                        if ctx_mask.any()
                        else torch.tensor(float("nan"), device=self.device)
                    )
                    act_acc = (
                        (seg_pred[act_mask] == 1).float().mean()
                        if act_mask.any()
                        else torch.tensor(float("nan"), device=self.device)
                    )
                    iou_act = _binary_iou(seg_pred == 1, gt == 1)
                    iou_ctx = _binary_iou(seg_pred == 0, gt == 0)
                    act_frac = act_mask.float().mean()

            loss = (
                loss_fm
                + self.seg_loss_weight * loss_seg
                + loss_art
            )
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                raise ValueError(f"Loss is {loss_value}, stopping training")

            # Scale so the accumulated gradient equals the mean over the whole
            # effective batch, matching what a single large batch would give.
            (loss / accum_steps).backward()
            is_step = ((data_iter_step + 1) % accum_steps == 0)
            if is_step:
                max_norm = self.config.trainer.optim.get("grad_clip", 1.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm
                )
                self.optimizer.step()

            lr = self.optimizer.param_groups[0]["lr"]

            # tqdm: short summary only — full breakdown lives in logger.info.
            if train_bar is not None and data_iter_step % self.train_cfg.logging_interval == 0:
                train_bar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    lr=f"{lr:.2e}",
                    refresh=False,
                )
                train_bar.update(self.train_cfg.logging_interval)

            if data_iter_step % self.train_cfg.logging_interval == 0:
                grad_norm_v = (
                    grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                )
                art_str = ""
                if has_art_head and art_losses:
                    rev_axis_key = (
                        "loss_rev_plucker" if motion_repr == "plucker"
                        else "loss_rev_axis_dir"
                    )
                    rev_axis_lbl = "pl" if motion_repr == "plucker" else "ad"
                    art_str = (
                        f"  art={loss_art.item():.4f}"
                        f"  {rev_axis_lbl}={art_losses[rev_axis_key].item():.3f}"
                        f"  rr={art_losses['loss_rev_range'].item():.3f}"
                        f"  pa={art_losses['loss_pris_axis'].item():.3f}"
                        f"  pr={art_losses['loss_pris_range'].item():.3f}"
                    )
                    if motion_repr == "per_point_closest":
                        art_str += f"  cp={art_losses['loss_rev_closest_pt'].item():.3f}"
                logger.info(
                    f"Epoch {epoch} [{data_iter_step}/{len(self.train_dataloader)}]: "
                    f"loss={loss_value:.6f}  fm={loss_fm.item():.6f}  "
                    f"seg={loss_seg.item():.6f}{art_str}  "
                    f"acc={seg_acc.item():.4f}  "
                    f"acc_ctx={ctx_acc.item():.4f}  acc_act={act_acc.item():.4f}  "
                    f"iou_ctx={iou_ctx.item():.4f}  iou_act={iou_act.item():.4f}  "
                    f"act_frac={act_frac.item():.4f}  "
                    f"grad_norm={grad_norm_v:.4f}  lr={lr:.2e}"
                )

            if self.wandb_run is not None and data_iter_step % self.train_cfg.logging_interval == 0:
                log = {
                    "train/loss": loss_value,
                    "train/loss_fm": loss_fm.item(),
                    "train/loss_seg": loss_seg.item(),
                    "train/seg_acc": seg_acc.item(),
                    "train/seg_acc_ctx": ctx_acc.item(),
                    "train/seg_acc_act": act_acc.item(),
                    "train/seg_iou_ctx": iou_ctx.item(),
                    "train/seg_iou_act": iou_act.item(),
                    "train/seg_act_frac": act_frac.item(),
                    "train/grad_norm": (
                        grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                    ),
                    "train/lr": lr,
                    "train/epoch": epoch,
                    "train/step": self.train_step,
                }
                if has_art_head and art_losses:
                    log["train/loss_art"] = loss_art.item()
                    for k, v in art_losses.items():
                        log[f"train/{k}"] = v.item()
                self.wandb_run.log(log)

            # One scheduler tick per *optimizer* step, so warmup_steps and the
            # cosine horizon keep their meaning when accumulating.
            if is_step:
                self.train_step += 1
                self.scheduler.step()

            del images, input_pts, part_mask, data

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Validation — multi-part (rebuttal)
    # ------------------------------------------------------------------

    # >=16 visually distinct colors; index = GT part id - 1. Part color
    # encodes PART identity (all articulated states of a part share it),
    # unlike the single-part STATE_PALETTE where color encodes qpos state.
    _MP_PART_PALETTE = [
        (0.90, 0.20, 0.20), (0.20, 0.50, 0.95), (0.25, 0.80, 0.35),
        (0.95, 0.65, 0.12), (0.65, 0.30, 0.85), (0.10, 0.75, 0.80),
        (0.95, 0.40, 0.70), (0.55, 0.55, 0.20), (0.30, 0.30, 0.90),
        (0.85, 0.45, 0.20), (0.20, 0.70, 0.55), (0.70, 0.20, 0.45),
        (0.50, 0.75, 0.15), (0.40, 0.45, 0.75), (0.80, 0.75, 0.25),
        (0.15, 0.55, 0.35),
    ]
    _MP_BASE_COLOR = (0.70, 0.70, 0.70)

    # Per-STATE palette for the per-joint eval dump. MUST match
    # scripts/larm/multipart_eval.py:_STATE_PALETTE (q = 0,0.25,0.5,0.75,1.0)
    # and _CTX_COLOR so multipart_eval --layout ours buckets our clouds by the
    # same colors it uses for the single-part baseline.
    _MP_STATE_PALETTE = [
        (0.20, 0.40, 0.95),  # 0.00 blue
        (0.20, 0.80, 0.85),  # 0.25 cyan
        (0.30, 0.85, 0.30),  # 0.50 green
        (0.95, 0.70, 0.10),  # 0.75 orange
        (0.90, 0.20, 0.20),  # 1.00 red
    ]
    _MP_CTX_COLOR = (0.70, 0.70, 0.70)

    def _val_epoch_multipart(self):
        """Multi-part validation.

        Seg + per-part joint metrics run on the GT rest cloud for every
        object (cheap: no ODE). Whole-object reconstruction CD/F1 and a
        per-part-colored merged PLY dump run only for the first
        ``save_pcd_interval`` objects on rank 0 (the ODE solve is the
        expensive part). Everything is scored in the median-3 normalized
        ref-cam frame; joint metrics are pushed to camera units by
        ``/s_scale`` (both pred and GT live in the normalized frame here,
        because seg is queried directly on the GT rest cloud rather than a
        reconstructed cloud — so no alignment scale/shift is involved).
        """
        import open3d as o3d

        self.model.eval()
        device = torch.device(self.device)
        is_rank0 = self.runner_info.rank == 0
        inner = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        step_size = self.config.get("fm_step_size", 0.04)
        method = self.config.get("fm_sampling", "euler")
        num_steps = round(1.0 / step_size)
        T = torch.linspace(0, 1, num_steps + 1, device=device)
        num_queries = self.config.get("val_num_queries", 8192)
        fs_thres = self.config.get("val_fs_thres", 0.05)
        align_cfg = self.config.get("alignment", None)
        align_kwargs = dict(align_cfg) if align_cfg is not None else {}
        save_pcd_interval = self.train_cfg.get("save_pcd_interval", 10)
        # When true, reconstruct EVERY object on EVERY rank and emit per-joint
        # STATE_PALETTE PLYs (<obj>_joint_<k>_merged_{pred,gt}.ply) consumable by
        # scripts/larm/multipart_eval.py --layout ours. This is the apples-to-
        # apples headline eval vs the offline single-part assembler baseline.
        mp_eval_dump = bool(self.config.get("mp_eval_dump", False))

        wrapper = BatchModelWrapper(model=self.model)
        solver = ODESolver(velocity_model=wrapper)

        timestamp = getattr(self.runner_info, "timestamp", "")
        pcd_dir = os.path.join(
            self.runner_info.work_dir, "val_pcd", timestamp,
            f"step_{self.val_step}",
        )
        if is_rank0:
            os.makedirs(pcd_dir, exist_ok=True)

        # Per-rank accumulators (plain floats → all_gather_object friendly).
        acc = {
            "seg_movable_iou": [],   # per object
            "seg_miou": [],          # per object (base + matched parts)
            "joint_axis_succ": [],   # per part (0/1)
            "joint_range_succ": [],  # per part (0/1)
            "geom_cd": [],           # per object (recon, rank0 subset)
            "geom_f1": [],           # per object (recon, rank0 subset)
        }

        def _iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
            inter = (pred_mask & gt_mask).float().sum()
            union = (pred_mask | gt_mask).float().sum()
            if union.item() == 0:
                return float("nan")
            return float((inter / union).item())

        val_bar = tqdm(
            self.val_dataloader, desc=f"Val-MP (step {self.val_step})",
            unit="batch", disable=not is_rank0, dynamic_ncols=True,
        )

        for val_idx, data in enumerate(val_bar):
            images = data["image"].to(device, non_blocking=True)
            pts = data["pts"].to(device, non_blocking=True)                  # [1,N,3] normalized
            part_mask = data["pts_part_mask"].to(device, non_blocking=True)  # [1,N]
            state_tag = data.get("state_tag", None)
            if state_tag is not None:
                state_tag = state_tag.to(device, non_blocking=True)
            num_movable = data["num_movable"].to(device, non_blocking=True).long()  # [1]

            part_mask0 = part_mask[0].long()
            K = int(num_movable[0].item())
            s_scale_b = float(data["s_scale"][0].item())
            sid = str(data.get("sample_id", [f"{val_idx:06d}"])[0]) \
                if isinstance(data.get("sample_id"), (list, tuple)) \
                else str(data.get("sample_id", f"{val_idx:06d}"))

            mp_valid0 = data["mp_valid"][0].bool()
            mp_is_rev0 = data["mp_is_revolute"][0].bool()
            mp_is_pris0 = data["mp_is_prismatic"][0].bool()
            mp_plucker0 = data["mp_plucker"][0].float().to(device)           # [P0,6]
            mp_pris_axis0 = data["mp_prismatic_axis"][0].float().to(device)  # [P0,3]
            mp_rev_hi0 = data["mp_revolute_range_hi"][0].float().to(device)  # [P0]
            mp_pris_hi0 = data["mp_prismatic_range_hi"][0].float().to(device)  # [P0]
            P0 = mp_valid0.shape[0]

            t_ones = torch.ones((1, 1), device=device)

            # ---- Encode once + seg on GT rest cloud (with GT part ids so the
            # model returns per-point revolute_closest_pt). --------------
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    enc = inner._encode(
                        images=images, pointmaps=None, test=True,
                        state_tag=state_tag,
                    )
                    seg_gt = self.model(
                        images=images, query_points=pts, timestep=t_ones,
                        run_seg=True, encoder_data=enc,
                        mp_part_ids=part_mask.long(), mp_num_movable=num_movable,
                    )

            seg_logits = seg_gt["seg_logits"].float()            # [1,N,1+P0]
            pred_col = seg_logits.argmax(dim=-1)[0]              # [N] col 0=base, c=slot c-1
            _, matched_slot = hungarian_match_movable(
                seg_logits, part_mask.long(), num_movable,
            )
            matched_slot0 = matched_slot[0]                      # [P0]

            art = seg_gt["articulation"]
            rev_axis_dir = art["revolute_axis_dir"].float()      # [1,P0,3]
            rev_range = art["revolute_range"].float()            # [1,P0]
            pris_axis = art["prismatic_axis"].float()            # [1,P0,3]
            pris_range = art["prismatic_range"].float()          # [1,P0]
            closest_pt = art["revolute_closest_pt"].float()[0]   # [N,3]

            # ---- (2) Seg metrics ------------------------------------------
            acc["seg_movable_iou"].append(_iou(pred_col >= 1, part_mask0 >= 1))
            ious = [_iou(pred_col == 0, part_mask0 == 0)]
            for j in range(K):
                s = int(matched_slot0[j].item())
                if s < 0:
                    continue
                ious.append(_iou(pred_col == (s + 1), part_mask0 == (j + 1)))
            ious = [v for v in ious if not math.isnan(v)]
            if ious:
                acc["seg_miou"].append(float(np.mean(ious)))

            # ---- (3) Per-part joint metrics + stash pred joints for dump --
            pred_joints: dict[int, dict] = {}
            for j in range(P0):
                if not bool(mp_valid0[j].item()):
                    continue
                s = int(matched_slot0[j].item())
                if s < 0:
                    continue
                is_rev_j = bool(mp_is_rev0[j].item())
                is_pris_j = bool(mp_is_pris0[j].item())
                if is_rev_j:
                    axis_dir = rev_axis_dir[0, s]
                    sel = (pred_col == (s + 1))
                    if sel.any():
                        cp = closest_pt[sel]
                    else:
                        gt_sel = (part_mask0 == (j + 1))
                        cp = closest_pt[gt_sel] if gt_sel.any() else closest_pt
                    med_pt = cp.median(dim=0).values
                    pred_pl = axis_dir_and_point_to_plucker(axis_dir, med_pt)  # [6] unit l
                    pred_range = float(rev_range[0, s].item())
                    pred_joints[j] = {
                        "type": "revolute", "plucker": pred_pl.detach(),
                        "range": pred_range,
                    }
                    pred_pl_cam = pred_pl.detach().cpu().numpy().astype(np.float64)
                    pred_pl_cam[3:6] = pred_pl_cam[3:6] / s_scale_b
                    gt_pl_cam = mp_plucker0[j].detach().cpu().numpy().astype(np.float64)
                    gt_pl_cam[3:6] = gt_pl_cam[3:6] / s_scale_b
                    m = articulation_metrics_revolute(
                        pred_pl_cam, gt_pl_cam, pred_range,
                        float(mp_rev_hi0[j].item()),
                    )
                elif is_pris_j:
                    axis = pris_axis[0, s]
                    pred_range = float(pris_range[0, s].item())
                    pred_joints[j] = {
                        "type": "prismatic", "axis": axis.detach(),
                        "range": pred_range,
                    }
                    pred_axis_np = axis.detach().cpu().numpy().astype(np.float64)
                    gt_axis_np = mp_pris_axis0[j].detach().cpu().numpy().astype(np.float64)
                    m = articulation_metrics_prismatic(
                        pred_axis_np, gt_axis_np,
                        pred_range / s_scale_b,
                        float(mp_pris_hi0[j].item()) / s_scale_b,
                    )
                else:
                    continue
                acc["joint_axis_succ"].append(
                    1.0 if m["axis_angle"] < self.metric_thr_axis_angle else 0.0
                )
                acc["joint_range_succ"].append(
                    1.0 if m["Mr"] < self.metric_thr_Mr else 0.0
                )

            # ---- (4) Reconstruction CD/F1 + per-part-colored merged PLY ---
            # Expensive (ODE); rank-0 only, first ``save_pcd_interval`` objs.
            do_viz = is_rank0 and val_idx < save_pcd_interval
            if do_viz or mp_eval_dump:
                # Optional test-time best-of-N recon selection. The FM decoder is
                # frozen while the encoder trains, so a few (object, x_init) pairs
                # collapse to a degenerate blob; run seg on such a blob and the head
                # predicts all-static, so the object silently produces no part PLYs
                # (a pure eval artifact -- seg on GT geometry is fine). We therefore
                # run the ODE ``mp_eval_n_noise`` times with different x_init seeds,
                # segment EACH sample independently (no concat -> no dilution of the
                # per-point seg), and keep the single sample with the largest movable
                # fraction: a collapsed blob has few movable points and is rejected.
                # n_noise=1 reproduces the single-seed path exactly.
                mp_n_noise = max(1, int(self.config.get("mp_eval_n_noise", 1)))
                base_seed = int(self.config.get("seed", 0)) * 1009 + val_idx
                best_mov = -1.0
                pred_cloud = None
                seg_recon = None
                pred_col_recon = None
                with torch.inference_mode():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        for k in range(mp_n_noise):
                            gen = torch.Generator(device=device).manual_seed(
                                (base_seed + k * 7919) % (2**31 - 1)
                            )
                            x_init = torch.rand(
                                1, num_queries, 3, device=device, generator=gen,
                            ) * 2.0 - 1.0
                            pred_k = solver.sample(
                                x_init=x_init, time_grid=T, method=method,
                                step_size=step_size, return_intermediates=False,
                                images=images, token_mask=None, encoder_data=enc,
                                pointmaps=None,
                            ).float()
                            seg_k = self.model(
                                images=images, query_points=pred_k,
                                timestep=t_ones, run_seg=True, encoder_data=enc,
                            )
                            col_k = seg_k["seg_logits"].float().argmax(dim=-1)[0]
                            mov = float((col_k >= 1).float().mean())
                            if mov > best_mov:
                                best_mov = mov
                                pred_cloud = pred_k
                                seg_recon = seg_k
                                pred_col_recon = col_k
                # Launder out of inference-mode so downstream chamfer / matmul
                # produce normal tensors.
                pred_cloud = pred_cloud.clone()

                gt_cam = data["pts_gt"].to(device).float()           # [1,N,3] camera
                gt_valid = torch.ones(gt_cam.shape[:2], dtype=torch.bool)
                _, _, a_scale, a_shift = scale_shift_alignment_pointcloud(
                    pred_cloud, gt_cam, gt_valid,
                    return_transform=True, **align_kwargs,
                )
                pred_cam = a_scale[0] * pred_cloud + a_shift[0]       # [1,Nq,3]
                d_tup, _ = chamfer_distance(
                    pred_cam, gt_cam, batch_reduction=None,
                    point_reduction=None, norm=2,
                )
                d_p, d_g = d_tup
                d_p = torch.sqrt(d_p); d_g = torch.sqrt(d_g)
                cd = float(((d_p.mean(dim=1) + d_g.mean(dim=1)) / 2.0).item())
                prec = (d_p < fs_thres).float().mean(dim=1)
                rec = (d_g < fs_thres).float().mean(dim=1)
                f1 = float((2.0 * prec * rec / (prec + rec + 1e-8)).item())
                acc["geom_cd"].append(cd)
                acc["geom_f1"].append(f1)

                gt_joints = self._mp_gt_joints(
                    mp_valid0, mp_is_rev0, mp_is_pris0,
                    mp_plucker0, mp_rev_hi0, mp_pris_axis0, mp_pris_hi0,
                )

                # ---- Merged PLYs (normalized frame, per-part colored) ----
                if do_viz:
                    self._dump_mp_merged_ply(
                        o3d, os.path.join(pcd_dir, f"{sid}_mp_merged_pred.ply"),
                        pred_cloud[0], pred_col_recon, matched_slot0, pred_joints,
                        side="pred",
                    )
                    self._dump_mp_merged_ply(
                        o3d, os.path.join(pcd_dir, f"{sid}_mp_merged_gt.ply"),
                        pts[0], part_mask0, matched_slot0, gt_joints, side="gt",
                    )

                # ---- Per-joint STATE_PALETTE PLYs for multipart_eval ------
                if mp_eval_dump:
                    # Per-part joints computed ON the reconstructed cloud so the
                    # pred parts + their joints share one frame; parts are walked
                    # in normalized frame then mapped to camera by (a_scale,
                    # a_shift). GT parts (pts) walk in normalized frame then /
                    # s_scale — matching val_epoch's frame handling and the
                    # baseline's camera-frame per-joint PLYs.
                    feats_r = seg_recon["point_features"].float()       # [1,Nq,D]
                    slots_r = seg_recon["slots_movable"].float()        # [1,P0,D]
                    art_r = seg_recon["articulation"]
                    slot_idx_r = (pred_col_recon - 1).clamp(min=0)      # [Nq]
                    Dd = feats_r.shape[-1]
                    with torch.inference_mode():
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            slot_emb_r = torch.gather(
                                slots_r, 1,
                                slot_idx_r.view(1, -1, 1).expand(1, slot_idx_r.shape[0], Dd),
                            )
                            cp_r = inner.articulation_head.closest_pt(
                                feats_r, slot_emb_r,
                            ).float()[0]                                # [Nq,3]
                    cp_r = cp_r.clone()
                    rev_axis_r = art_r["revolute_axis_dir"].float()
                    rev_range_r = art_r["revolute_range"].float()
                    pris_axis_r = art_r["prismatic_axis"].float()
                    pris_range_r = art_r["prismatic_range"].float()
                    # Correspondence vs membership -- the load-bearing split.
                    # NN-transferred GT labels (``recon_part_lbl``) are used ONLY
                    # to decide which recon slot is "GT part j"; they never select
                    # the points that make up a part. Membership comes from the
                    # model's own ``pred_col_recon``, so the assembled prediction
                    # carries this model's segmentation error -- matching how the
                    # single-part baseline assembles (``pred_seg_for_pred_cloud``)
                    # and how LARM's base/part meshes are split.
                    #
                    # The matching must be Hungarian, not a per-part mode() vote:
                    # mode() is not injective, so two GT parts collapsed into one
                    # slot would both select the SAME point set and the assembly
                    # would duplicate that geometry under two different joints.
                    # ``linear_sum_assignment`` is injective, so the selected
                    # masks partition the cloud.
                    from pytorch3d.ops import knn_points
                    pred_cam_pts = (a_scale[0] * pred_cloud[0] + a_shift[0]).float()
                    _, nn_idx, _ = knn_points(
                        pred_cam_pts[None], gt_cam.float(), K=1,
                    )
                    recon_part_lbl = part_mask0[nn_idx[0, :, 0]]        # [Nq] 0..K
                    _, matched_slot_recon = hungarian_match_movable(
                        seg_recon["seg_logits"].float(),
                        recon_part_lbl[None].long(), num_movable,
                    )
                    matched_slot_recon0 = matched_slot_recon[0]         # [P0]
                    # Diagnostic only: restores the GT-labelled assembly this eval
                    # used before, to measure what oracle segmentation was worth.
                    use_gt_labels = bool(
                        self.config.get("mp_eval_gt_part_labels", False)
                    )
                    if use_gt_labels:
                        seg_col_pred = recon_part_lbl
                        slot_map_pred = torch.arange(
                            P0, device=pred_col_recon.device,
                        )
                    else:
                        seg_col_pred = pred_col_recon
                        slot_map_pred = matched_slot_recon0
                    pred_joints_recon: dict[int, dict] = {}
                    n_parts_emitted = 0
                    for j in range(P0):
                        if not bool(mp_valid0[j].item()):
                            continue
                        s = int(slot_map_pred[j].item())
                        if s < 0:
                            continue
                        part_sel = (seg_col_pred == (s + 1))
                        # Honest drop: the model found no points for this part.
                        if not part_sel.any():
                            continue
                        n_parts_emitted += 1
                        # Slot to read articulation from. In the prediction-only
                        # path membership IS slot ``s``, so ``s`` is already right.
                        # Under GT labelling the part was selected by a GT id, which
                        # is not a slot index, so recover the recon's own dominant
                        # slot exactly as the pre-patch code did -- otherwise this
                        # diagnostic reads an arbitrary slot's joint and stops being
                        # a faithful replay of the old behaviour.
                        s_art = s
                        if use_gt_labels:
                            cols = pred_col_recon[part_sel]
                            cols = cols[cols >= 1]
                            if cols.numel() > 0:
                                s_art = int(torch.mode(cols).values.item()) - 1
                            else:
                                s_art = int(matched_slot0[j].item())
                                if s_art < 0:
                                    continue
                        if bool(mp_is_rev0[j].item()):
                            cp = cp_r[part_sel]
                            med = cp.median(dim=0).values
                            pl = axis_dir_and_point_to_plucker(rev_axis_r[0, s_art], med)
                            pred_joints_recon[j] = {
                                "type": "revolute", "plucker": pl.detach(),
                                "range": float(rev_range_r[0, s_art].item()),
                            }
                        elif bool(mp_is_pris0[j].item()):
                            pred_joints_recon[j] = {
                                "type": "prismatic",
                                "axis": pris_axis_r[0, s_art].detach(),
                                "range": float(pris_range_r[0, s_art].item()),
                            }
                    logger.info(
                        f"[MP-dump] {sid}: K_gt={K} parts_emitted={n_parts_emitted} "
                        f"gt_labels={use_gt_labels} n_noise={mp_n_noise}"
                    )
                    self._dump_mp_perjoint_plys(
                        o3d, pcd_dir, sid, pred_cloud[0], seg_col_pred,
                        slot_map_pred, pred_joints_recon, side="pred",
                        a_scale=a_scale[0], a_shift=a_shift[0], s_scale=s_scale_b,
                    )
                    self._dump_mp_perjoint_plys(
                        o3d, pcd_dir, sid, pts[0], part_mask0,
                        matched_slot0, gt_joints, side="gt",
                        a_scale=None, a_shift=None, s_scale=s_scale_b,
                    )

            del images, pts, part_mask, data

        # ---- Aggregate across ranks -----------------------------------
        is_dist = dist.is_available() and dist.is_initialized()
        if is_dist:
            world_size = dist.get_world_size()
            gathered: list[dict | None] = [None] * world_size
            dist.all_gather_object(gathered, acc)
        else:
            gathered = [acc]

        merged = {k: [] for k in acc}
        for rd in gathered:
            if not rd:
                continue
            for k, v in rd.items():
                merged[k].extend(v)

        def _mean(vals):
            arr = np.asarray(vals, dtype=np.float64)
            arr = arr[~np.isnan(arr)]
            return float(arr.mean()) if arr.size else float("nan")

        out = {
            "mp/seg/movable_iou": _mean(merged["seg_movable_iou"]),
            "mp/seg/miou": _mean(merged["seg_miou"]),
            "mp/joint/axis_succ": _mean(merged["joint_axis_succ"]),
            "mp/joint/range_succ": _mean(merged["joint_range_succ"]),
            "mp/geom/cd": _mean(merged["geom_cd"]),
            "mp/geom/f1": _mean(merged["geom_f1"]),
        }
        n_obj = len(merged["seg_movable_iou"])
        n_parts = len(merged["joint_axis_succ"])
        n_geom = len(merged["geom_cd"])

        if is_rank0:
            logger.info(
                f"[Val-MP/seg] movable_iou={out['mp/seg/movable_iou']:.4f} "
                f"miou={out['mp/seg/miou']:.4f} n={n_obj}"
            )
            logger.info(
                f"[Val-MP/joint] "
                f"axis_angle_succ@{self.metric_thr_axis_angle}="
                f"{out['mp/joint/axis_succ']:.4f} "
                f"range_succ@{self.metric_thr_Mr}={out['mp/joint/range_succ']:.4f} "
                f"n_parts={n_parts}"
            )
            logger.info(
                f"[Val-MP/geom] CD={out['mp/geom/cd']:.6f} "
                f"F1@{fs_thres}={out['mp/geom/f1']:.4f} n={n_geom}"
            )
            if self.wandb_run is not None:
                self.wandb_run.log({**out, "val/step": float(self.val_step)})

        self.val_step += 1
        del wrapper, solver
        torch.cuda.empty_cache()
        self.model.train()
        return out if is_rank0 else {}

    # --- multi-part dump helpers ---------------------------------------

    def _mp_gt_joints(
        self, mp_valid0, mp_is_rev0, mp_is_pris0,
        mp_plucker0, mp_rev_hi0, mp_pris_axis0, mp_pris_hi0,
    ) -> dict[int, dict]:
        """Build the per-part GT joint dict (normalized frame) keyed by GT
        part index ``j`` (part id ``j+1``), mirroring the pred-joint schema."""
        out: dict[int, dict] = {}
        for j in range(mp_valid0.shape[0]):
            if not bool(mp_valid0[j].item()):
                continue
            if bool(mp_is_rev0[j].item()):
                out[j] = {
                    "type": "revolute", "plucker": mp_plucker0[j],
                    "range": float(mp_rev_hi0[j].item()),
                }
            elif bool(mp_is_pris0[j].item()):
                out[j] = {
                    "type": "prismatic", "axis": mp_pris_axis0[j],
                    "range": float(mp_pris_hi0[j].item()),
                }
        return out

    def _dump_mp_merged_ply(
        self, o3d, path, cloud, seg_col, matched_slot0, joints, side,
    ):
        """Write one per-part-colored merged PLY.

        ``cloud`` [N,3] and ``seg_col`` [N] are in the normalized frame.
        For ``side=="pred"`` ``seg_col`` is the argmax column (0=base,
        c=slot c-1) and parts are keyed by matched slot; for ``side=="gt"``
        ``seg_col`` is ``part_mask`` (0=base, j+1=part j). Each matched/valid
        part is articulated through ``self.eval_states`` by its (pred/gt)
        joint and painted with a single per-part color across all states.
        Base + unassigned points are gray, drawn once at rest.
        """
        cloud = cloud.float()
        xyz_layers: list[np.ndarray] = []
        rgb_layers: list[np.ndarray] = []
        assigned = torch.zeros(cloud.shape[0], dtype=torch.bool, device=cloud.device)

        for j, joint in joints.items():
            if side == "pred":
                s = int(matched_slot0[j].item())
                if s < 0:
                    continue
                mask = (seg_col == (s + 1))
            else:
                mask = (seg_col == (j + 1))
            if not mask.any():
                continue
            assigned |= mask
            part_pts = cloud[mask]                                   # [Nj,3]
            color = self._MP_PART_PALETTE[j % len(self._MP_PART_PALETTE)]
            for t_state in self.eval_states:
                # Multipart pts_rest is the raw mesh (angle 0 == URDF state 0),
                # unlike single-part val whose cloud sits at s1 (state 1). So the
                # walk to state t is angle = range_hi * t (delta = t), which lands
                # at the same absolute angle (range_hi * t from raw) that the
                # single-part baseline reaches via its s1 cloud + delta=(t-1).
                delta = float(t_state)
                if joint["type"] == "revolute":
                    Tm = plucker_to_4x4_torch(
                        joint["plucker"].float(),
                        torch.tensor(joint["range"] * delta, device=cloud.device),
                    )
                else:
                    axis = joint["axis"].float()
                    axis = axis / (axis.norm() + 1e-12)
                    Tm = prismatic_to_4x4_torch(
                        axis,
                        torch.tensor(joint["range"] * delta, device=cloud.device),
                    )
                pts_t = part_pts @ Tm[:3, :3].T + Tm[:3, 3]
                xyz_layers.append(pts_t.detach().cpu().numpy())
                rgb_layers.append(
                    np.tile(np.asarray(color, dtype=np.float32),
                            (pts_t.shape[0], 1))
                )

        base_mask = ~assigned
        if base_mask.any():
            base_pts = cloud[base_mask].detach().cpu().numpy()
            xyz_layers.append(base_pts)
            rgb_layers.append(
                np.tile(np.asarray(self._MP_BASE_COLOR, dtype=np.float32),
                        (base_pts.shape[0], 1))
            )

        if not xyz_layers:
            return
        xyz = np.concatenate(xyz_layers, axis=0)
        rgb = np.concatenate(rgb_layers, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
        o3d.io.write_point_cloud(path, pcd)

    def _dump_mp_perjoint_plys(
        self, o3d, out_dir, obj_id, cloud, seg_col, matched_slot0, joints, side,
        a_scale, a_shift, s_scale,
    ):
        """Write per-joint STATE_PALETTE merged PLYs for multipart_eval.

        One file per valid part ``j``: ``<obj>_joint_<j>_merged_<side>.ply``.
        Layout matches ``scripts/larm/multipart_eval.py`` (--layout ours):
        ``ctx`` (gray, everything except part j at rest) + part j articulated
        through ``self.eval_states`` colored by ``_MP_STATE_PALETTE``. Walk is
        done in the normalized frame (delta = t; see ``_dump_mp_merged_ply``),
        then mapped to camera units — pred by ``a_scale*x + a_shift`` (the
        chamfer alignment), gt by ``x / s_scale`` — so both live in the same
        camera frame as the s0.36 baseline's per-joint PLYs.
        """
        cloud = cloud.float()
        dev = cloud.device

        def _to_cam(x):  # [M,3] normalized -> camera
            if side == "pred":
                return a_scale * x + a_shift
            return x / s_scale

        for j, joint in joints.items():
            if side == "pred":
                s = int(matched_slot0[j].item())
                if s < 0:
                    continue
                active_mask = (seg_col == (s + 1))
            else:
                active_mask = (seg_col == (j + 1))
            if not active_mask.any():
                continue
            ctx_pts = cloud[~active_mask]        # normalized, at rest
            part_pts = cloud[active_mask]        # normalized, at rest (raw)

            xyz_layers: list[np.ndarray] = []
            rgb_layers: list[np.ndarray] = []
            ctx_cam = _to_cam(ctx_pts).detach().cpu().numpy()
            xyz_layers.append(ctx_cam)
            rgb_layers.append(
                np.tile(np.asarray(self._MP_CTX_COLOR, dtype=np.float32),
                        (ctx_cam.shape[0], 1))
            )
            for i, t_state in enumerate(self.eval_states):
                delta = float(t_state)
                if joint["type"] == "revolute":
                    Tm = plucker_to_4x4_torch(
                        joint["plucker"].float(),
                        torch.tensor(joint["range"] * delta, device=dev),
                    )
                else:
                    axis = joint["axis"].float()
                    axis = axis / (axis.norm() + 1e-12)
                    Tm = prismatic_to_4x4_torch(
                        axis, torch.tensor(joint["range"] * delta, device=dev),
                    )
                pts_t = part_pts @ Tm[:3, :3].T + Tm[:3, 3]   # normalized
                pts_t_cam = _to_cam(pts_t).detach().cpu().numpy()
                color = self._MP_STATE_PALETTE[i % len(self._MP_STATE_PALETTE)]
                xyz_layers.append(pts_t_cam)
                rgb_layers.append(
                    np.tile(np.asarray(color, dtype=np.float32),
                            (pts_t_cam.shape[0], 1))
                )

            xyz = np.concatenate(xyz_layers, axis=0)
            rgb = np.concatenate(rgb_layers, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
            o3d.io.write_point_cloud(
                os.path.join(out_dir, f"{obj_id}_joint_{j}_merged_{side}.ply"),
                pcd,
            )

    # ------------------------------------------------------------------
    # Validation — merged geometry + seg pass
    # ------------------------------------------------------------------

    def val_epoch(self):
        """Single-pass validation: per batch, encode once and reuse the
        latent for both the ODE-based geometry sample and the ``t=1`` seg
        decode. The image encoder is the dominant cost, so this halves
        per-batch val compute compared to running geometry val and seg
        val as two independent dataloader iterations.

        Implementation notes:

        - ``BatchModelWrapper`` calls ``_decode`` once per ODE step using
          the cached ``encoder_data``, so the solver path doesn't re-run
          the encoder.
        - The seg pass goes through ``model(..., encoder_data=enc,
          run_seg=True)`` so the captured-features hook on
          ``pts3d_head.linear_out`` fires and the seg head sees the
          last-block per-point features.
        - Seg-pass metrics are accumulated as plain Python dicts of
          floats (per-category counts) and gathered via
          ``all_gather_object`` because the set of categories seen on
          each rank can differ; an ``all_reduce`` over a missing key
          would hang the collective.
        - Geometry-pass metrics use the same per-sample tensor +
          ``all_reduce`` pattern the parent uses (no per-rank category
          mismatch since we materialize the global category union from
          the seg-side ``all_gather_object`` first... actually we
          replicate the parent's ``all_gather_object`` on the cat set
          to keep both paths aligned).
        """
        # Multi-part (rebuttal) uses a multiclass part_mask + per-part joint
        # tensors the LARM-bucketed single-part val path can't consume; eval is
        # done offline via scripts/eval_multipart_forward.py + multipart_eval.py.
        if getattr(self, "multipart", False):
            return self._val_epoch_multipart()
        self.model.eval()
        device = torch.device(self.device)
        is_rank0 = self.runner_info.rank == 0

        # Geometry / FM-sampling config (mirror ReconTrainer.val_epoch).
        step_size = self.config.get("fm_step_size", 0.04)
        method = self.config.get("fm_sampling", "euler")
        num_steps = round(1.0 / step_size)
        T = torch.linspace(0, 1, num_steps + 1, device=device)
        num_queries = self.config.get("val_num_queries", 8192)
        # Final per-sample point budget after merge + per-part outlier filter.
        # Defaults to num_queries (back-compat). Set this when you want to
        # decouple the FM-ODE x_init size from the scoring budget — e.g. run
        # the ODE at the training-time 8192 and still report metrics on a
        # 30000-pt cloud built from N noise-sample concatenation.
        subsample_target = int(self.config.get("val_subsample_target", num_queries))
        fs_thres = self.config.get("val_fs_thres", 0.05)
        align_cfg = self.config.get("alignment", None)
        align_kwargs = dict(align_cfg) if align_cfg is not None else {}

        # Noise-ensemble + per-part outlier postprocess (off by default).
        # When n_noise>1 we run the ODE solver N times with different x_init
        # seeds, concat the predictions, then (optionally) per-part filter
        # ctx/active separately and subsample back to ``num_queries`` so the
        # downstream metrics see a budget identical to the N=1 baseline. See
        # ``src/viz/forward_pass.py`` — same recipe, but invoked at val-loop
        # scale instead of one snapshot.
        n_noise = int(self.config.get("val_num_noise_samples", 1))
        do_postproc = bool(self.config.get("val_postprocess", False))
        pp_psnb = int(self.config.get("val_pp_per_sample_nb", 20))
        pp_psstd = float(self.config.get("val_pp_per_sample_std", 1.5))
        pp_gnb = int(self.config.get("val_pp_global_nb", 20))
        pp_gstd = float(self.config.get("val_pp_global_std", 1.5))
        # Complement seg metric: also query the seg head directly on GT
        # geometry (exact, no NN transfer) to isolate seg-head quality from
        # reconstruction error. On by default; costs one extra forward/sample.
        seg_eval_on_gt = bool(self.config.get("val_seg_eval_on_gt", True))
        if self.runner_info.rank == 0:
            logger.info(
                f"[val_epoch] n_noise={n_noise} postprocess={do_postproc} "
                f"per_sample=(nb={pp_psnb}, std={pp_psstd}) "
                f"global=(nb={pp_gnb}, std={pp_gstd}) num_queries={num_queries} "
                f"subsample_target={subsample_target} "
                f"seg_eval_on_gt={seg_eval_on_gt}"
            )

        wrapper = BatchModelWrapper(model=self.model)
        solver = ODESolver(velocity_model=wrapper)

        # ---------- Per-sid PCD save (rank-0, incremental) -----------
        # Old: accumulate everything into all_pred/all_gt/all_images and a
        # flat art_viz_buffer, then walk + write at end of val_epoch.
        # New: per-sid 5-state buffer; flush to disk and pop as soon as a
        # sid hits len(self.eval_states) entries. Input-image PNGs and
        # per-sample PLYs are written immediately inside the loop, so CPU
        # memory stays flat and IO overlaps with the next iter's GPU work.
        from collections import defaultdict
        import open3d as o3d  # local import keeps non-rank-0 ranks happy
        save_pcd_interval = self.train_cfg.get("save_pcd_interval", 10)
        art_viz_by_sid: dict[str, list] = defaultdict(list)
        # Per-sample (cd, f1) per state — for offline ranking + visualization
        # selection. Flat: sid -> {"category", "motion_type", "states": {sk -> {cd, f1}}}.
        per_sample_metrics: dict[str, dict] = {}
        n_eval_states = len(self.eval_states)
        merged_count = 0  # merged-state PLYs written this val_epoch
        STATE_PALETTE = [
            (0.20, 0.40, 0.95),  # state 0.00 — blue
            (0.20, 0.80, 0.85),  # state 0.25 — cyan
            (0.30, 0.85, 0.30),  # state 0.50 — green
            (0.95, 0.70, 0.10),  # state 0.75 — orange
            (0.90, 0.20, 0.20),  # state 1.00 — red
        ]
        CTX_COLOR = (0.70, 0.70, 0.70)

        def _build_merged_xyz_rgb(entries: list, side: str):
            """Stitch (ctx once + per-state active) for one sid into
            (xyz, rgb) ready for open3d. Returns None on empty input."""
            entries_sorted = sorted(entries, key=lambda e: float(e[0]))
            xyz_layers: list[np.ndarray] = []
            rgb_layers: list[np.ndarray] = []
            ctx_added = False
            for idx, (sk, pred_pts, pred_seg, gt_pts_t, gt_seg) in enumerate(entries_sorted):
                pts = pred_pts if side == "pred" else gt_pts_t
                seg = pred_seg if side == "pred" else gt_seg
                if pts is None or seg is None:
                    continue
                pts_np = pts.numpy() if torch.is_tensor(pts) else np.asarray(pts)
                seg_np = seg.reshape(-1).numpy() if torch.is_tensor(seg) else np.asarray(seg).reshape(-1)
                pts_np = pts_np.reshape(-1, 3)
                if not ctx_added:
                    ctx_mask = (seg_np == 0)
                    if ctx_mask.any():
                        ctx = pts_np[ctx_mask]
                        xyz_layers.append(ctx)
                        rgb_layers.append(np.tile(np.asarray(CTX_COLOR, dtype=np.float32), (ctx.shape[0], 1)))
                        ctx_added = True
                act_mask = (seg_np == 1)
                if act_mask.any():
                    act = pts_np[act_mask]
                    color = STATE_PALETTE[idx % len(STATE_PALETTE)]
                    xyz_layers.append(act)
                    rgb_layers.append(np.tile(np.asarray(color, dtype=np.float32), (act.shape[0], 1)))
            if not xyz_layers:
                return None
            return np.concatenate(xyz_layers, axis=0), np.concatenate(rgb_layers, axis=0)

        def _flush_sid(sid: str) -> int:
            """Build + write merged PLYs (pred + gt) for one sid, then pop.
            Returns number of PLYs written (0, 1, or 2)."""
            nonlocal merged_count
            entries = art_viz_by_sid.pop(sid, [])
            if not entries:
                return 0
            n = 0
            for side in ("pred", "gt"):
                m = _build_merged_xyz_rgb(entries, side)
                if m is None:
                    continue
                xyz, rgb = m
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
                o3d.io.write_point_cloud(
                    os.path.join(pcd_dir, f"{sid}_merged_{side}.ply"), pcd,
                )
                n += 1
            merged_count += n
            return n

        # ---------- LARM-comparable flat-bucket accumulator ------------
        # Each bucket is a per-rank local view; cross-rank gather + merge
        # happens after the dataloader pass. Bucket key namespace:
        #
        #   "oven"                           — held-out OOD class
        #   "geometry"                       — non-Oven (any LARM bucket)
        #   "articulation"                   — non-Oven  ∩ LARM-success
        #   f"class:{cat}-geom"              — per-class slice of "geometry"
        #   f"class:{cat}-art"               — per-class slice of "articulation"
        #
        # Bucket payload:
        #   {
        #     "geom_state_counts": {state_key: {cd_sum, f1_sum, n}},
        #     "art_metrics":       {metric_name: [per-sample values]},
        #     "n_total":           int,    # bucket population (success_rate_all denom)
        #     "n_with_joint":      int,    # samples with motion (success_rate_with_joint denom)
        #   }
        # Plain Python so DDP ``all_gather_object`` doesn't care about
        # per-rank schema mismatch (a class only seen on one rank still
        # gathers cleanly).
        val_buckets: dict[str, dict] = {}

        def _ensure_bucket(key: str) -> dict:
            b = val_buckets.get(key)
            if b is None:
                b = {
                    "geom_state_counts": {},
                    "geom_state_counts_active": {},
                    "geom_state_counts_static": {},
                    # Segmentation-isolated part reconstruction: the movable /
                    # static split of the *prediction* is chosen by GT labels
                    # NN-transferred onto the reconstruction (not the seg head),
                    # so these numbers exclude segmentation error. ``_pure`` also
                    # removes joint error (scored at rest, Chamfer is invariant to
                    # a common rigid joint transform); ``_pj`` keeps the predicted
                    # joint (per state).
                    "geom_active_gtseg_pure": {},
                    "geom_static_gtseg_pure": {},
                    "geom_active_gtseg_pj": {},
                    "art_metrics": {},
                    "seg_metrics": {},
                    "seg_counts": {},
                    "n_total": 0,
                    "n_with_joint": 0,
                }
                val_buckets[key] = b
            return b

        def _bump_count(keys: list[str], with_joint: bool) -> None:
            for k in keys:
                b = _ensure_bucket(k)
                b["n_total"] += 1
                if with_joint:
                    b["n_with_joint"] += 1

        def _bump_geom(keys: list[str], state_key: str,
                       cd: float, f1: float) -> None:
            for k in keys:
                b = _ensure_bucket(k)
                s = b["geom_state_counts"].setdefault(
                    state_key, {"cd_sum": 0.0, "f1_sum": 0.0, "n": 0},
                )
                s["cd_sum"] += cd
                s["f1_sum"] += f1
                s["n"] += 1

        def _bump_art_metric(keys: list[str], mname: str, val: float) -> None:
            for k in keys:
                b = _ensure_bucket(k)
                b["art_metrics"].setdefault(mname, []).append(val)

        def _bump_geom_field(field: str, keys: list[str], state_key: str,
                             cd: float, f1: float) -> None:
            """Like ``_bump_geom`` but into a named per-part geom field
            (``geom_state_counts_active`` / ``_static``)."""
            for k in keys:
                b = _ensure_bucket(k)
                s = b[field].setdefault(
                    state_key, {"cd_sum": 0.0, "f1_sum": 0.0, "n": 0},
                )
                s["cd_sum"] += cd
                s["f1_sum"] += f1
                s["n"] += 1

        def _part_cd_f1(p_cloud, g_cloud):
            """Symmetric CD + F1@fs_thres for two ``[1, N, 3]`` clouds.
            Returns ``(cd, f1)`` or ``None`` if either cloud is empty."""
            if p_cloud.shape[1] == 0 or g_cloud.shape[1] == 0:
                return None
            dtp, _ = chamfer_distance(
                p_cloud, g_cloud, batch_reduction=None,
                point_reduction=None, norm=2,
            )
            dpp, dgg = dtp
            dpp = torch.sqrt(dpp)
            dgg = torch.sqrt(dgg)
            cd = ((dpp.mean(dim=1) + dgg.mean(dim=1)) / 2.0).item()
            pp = (dpp < fs_thres).float().mean(dim=1)
            rr = (dgg < fs_thres).float().mean(dim=1)
            f1 = (2.0 * pp * rr / (pp + rr + 1e-8)).item()
            return cd, f1

        def _bump_seg_metric(keys: list[str], mname: str, val: float) -> None:
            for k in keys:
                b = _ensure_bucket(k)
                b["seg_metrics"].setdefault(mname, []).append(val)

        def _bump_seg_counts(keys: list[str], variant: str,
                             tp: float, fp: float, fn: float) -> None:
            for k in keys:
                b = _ensure_bucket(k)
                c = b["seg_counts"].setdefault(
                    variant, {"tp": 0.0, "fp": 0.0, "fn": 0.0},
                )
                c["tp"] += tp
                c["fp"] += fp
                c["fn"] += fn

        def _state_key(t: float) -> str:
            return f"{t:.2f}"

        def _larm_bucket_for(sid: str) -> str | None:
            """Look up which LARM bucket this sample belongs to. Returns
            None if LARM status isn't loaded or the sample id isn't in
            either set (e.g., a custom test split LARM didn't run on)."""
            if self.larm_success_ids is None or self.larm_fail_ids is None:
                return None
            if sid in self.larm_success_ids:
                return "larm_success"
            if sid in self.larm_fail_ids:
                return "larm_fail"
            return None

        ood_set = self._ood_categories()  # e.g. {"Oven"} when held out

        def _resolve_buckets(cat: str, larm_bucket: str | None
                             ) -> tuple[list[str], list[str]]:
            """Return (geom_buckets, art_buckets) for a sample.

            * Pure-eval mode (no ``larm_status`` loaded, so
              ``larm_evaluated_ids is None``): there is no LARM-comparable
              set to line up with, so every sample gets per-category
              geometry AND articulation unconditionally. This is the mode
              used for the R1 OOD-generalization eval on categories LARM
              never ran (Dishwasher / WashingMachine / Box / Suitcase /
              Toilet) — each category emits its own ``Val-{cat}-Geometry``
              and ``Val-{cat}-Articulation`` rows.
            * Oven samples go solely into ``oven`` (regardless of LARM bucket
              — Oven is the OOD experiment, not the LARM-comparable set).
            * Non-Oven samples always go into ``geometry`` + ``class:{cat}-geom``
              (geom rows). They additionally go into ``articulation`` +
              ``class:{cat}-art`` only when LARM ran the sample successfully —
              matching the user's spec that articulation metrics are over
              the LARM-success subset only.
            """
            if self.larm_evaluated_ids is None:
                geom_keys = ["geometry"]
                art_keys = ["articulation"]
                if cat:
                    geom_keys.append(f"class:{cat}-geom")
                    art_keys.append(f"class:{cat}-art")
                return geom_keys, art_keys
            if cat in ood_set:
                return ["oven"], ["oven"]
            geom_keys = ["geometry"]
            if cat:
                geom_keys.append(f"class:{cat}-geom")
            art_keys: list[str] = []
            if larm_bucket == "larm_success":
                art_keys.append("articulation")
                if cat:
                    art_keys.append(f"class:{cat}-art")
            return geom_keys, art_keys

        # Phase-5 per-state articulated-cloud viz buffer (rank 0 only).
        # ``val_step`` increments at the end; capture for the dir name now
        # so geometry PCDs and seg PLYs from this pass cluster together.
        step_for_dir = self.val_step
        timestamp = getattr(self.runner_info, "timestamp", "")
        pcd_dir = os.path.join(
            self.runner_info.work_dir, "val_pcd", timestamp,
            f"step_{step_for_dir}",
        )
        if is_rank0:
            os.makedirs(pcd_dir, exist_ok=True)

        val_bar = tqdm(
            self.val_dataloader,
            desc=f"Val (step {step_for_dir})",
            unit="batch",
            disable=not is_rank0,
            dynamic_ncols=True,
        )

        inner = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        for val_idx, data in enumerate(val_bar):
            images = data["image"].to(self.device, non_blocking=True)
            gt_pts = data["pts_gt"].to(self.device, non_blocking=True)
            input_pts = data["pts"].to(self.device, non_blocking=True)
            part_mask = data["pts_part_mask"].to(self.device, non_blocking=True)
            state_tag = data.get("state_tag", None)
            if state_tag is not None:
                state_tag = state_tag.to(self.device, non_blocking=True)
            cats: list[str] = list(data.get("category", []))
            sample_ids: list[str] = list(data.get("sample_id", []))

            input_images = images
            B = gt_pts.shape[0]

            # The articulation_head is required for this trainer — it
            # gives us both the joint-parameter prediction and the seg
            # mask we use to identify the predicted active part.
            has_art = getattr(inner, "articulation_head", None) is not None

            # ----- N-noise ensemble + per-part outlier postprocess ------
            # When n_noise>1, run the ODE+seg N times with different x_init
            # seeds, concat all predictions into one big cloud per sample,
            # optionally per-part filter, then subsample back to
            # ``num_queries`` so the alignment + per-state metrics see the
            # same point budget as the N=1 baseline. Per-point quantities
            # (seg, revolute_closest_pt) are sliced with the same indices
            # to stay in lockstep; global articulation scalars are mean-
            # aggregated across noise samples.
            preds_list: list[torch.Tensor] = []
            seg_logits_list: list[torch.Tensor] = []
            art_list: list[dict] = []
            t_ones = torch.ones((B, 1), device=device)
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    encoder_data = inner._encode(
                        images=images, pointmaps=None, test=True,
                        state_tag=state_tag,
                    )
                    for k in range(n_noise):
                        # Per-(epoch, val_idx, k) seed so different ranks /
                        # repeats don't all draw the same noise. Mod 2**31 to
                        # stay inside int32.
                        seed_k = (
                            (self.config.get("seed", 0) * 1_000_003)
                            + (val_idx * 1009) + (k * 17)
                        ) % (2**31 - 1)
                        gen = torch.Generator(device=device).manual_seed(int(seed_k))
                        x_init = torch.rand(
                            B, num_queries, 3, device=device, generator=gen,
                        ) * 2.0 - 1.0
                        pred_k = solver.sample(
                            x_init=x_init,
                            time_grid=T,
                            method=method,
                            step_size=step_size,
                            return_intermediates=False,
                            images=images,
                            token_mask=None,
                            encoder_data=encoder_data,
                            pointmaps=None,
                        ).float()
                        preds_list.append(pred_k.detach())
                        if has_art:
                            seg_out_k = self.model(
                                images=images,
                                query_points=pred_k,
                                timestep=t_ones,
                                run_seg=True,
                                encoder_data=encoder_data,
                            )
                            seg_logits_list.append(seg_out_k["seg_logits"].float().detach())
                            art_list.append({
                                kk: (vv.detach().float() if torch.is_tensor(vv) else vv)
                                for kk, vv in seg_out_k["articulation"].items()
                            })

            # ---------- Per-noise-sample per-part outlier filter -------
            # For each k: split (pred_k, seg_k) into ctx/active by the
            # per-noise seg labels and filter each part separately. Keep
            # the survivors and slice every per-point quantity (seg
            # logits, revolute_closest_pt) by the same indices so the
            # k-th cloud stays internally consistent before merging. This
            # mirrors ``src/viz/forward_pass.py``'s recipe — filter each
            # noise sample first, THEN concat, THEN one global per-part
            # filter on the merged cloud after alignment.
            #
            # NOTE: per-sample filtering produces variable point counts
            # per (b, k), so we currently require batch_size == 1 in val
            # when this path is on (the existing val_dataloader already
            # uses B=1 — ``eval.py`` hardcodes ``batch_size=1``).
            if has_art and do_postproc:
                if B != 1:
                    raise RuntimeError(
                        f"val_postprocess expects val batch_size=1, got B={B}. "
                        "Either disable val_postprocess or set val_dataloader.batch_size=1."
                    )
                new_preds: list[torch.Tensor] = []
                new_seg_logits: list[torch.Tensor] = []
                new_arts: list[dict] = []
                for k in range(n_noise):
                    pts_k_np = preds_list[k][0].float().cpu().numpy()
                    seg_k_np = (
                        seg_logits_list[k][0].argmax(dim=-1).cpu().numpy().astype(np.int64)
                    )
                    keep_idx = _per_part_filter_indices(
                        pts_k_np, seg_k_np,
                        nb_neighbors=pp_psnb, std_ratio=pp_psstd,
                    )
                    keep_t = torch.from_numpy(keep_idx).long().to(device)
                    new_preds.append(preds_list[k].index_select(1, keep_t))
                    new_seg_logits.append(seg_logits_list[k].index_select(1, keep_t))
                    art_k = dict(art_list[k])
                    if art_k.get("revolute_closest_pt") is not None:
                        art_k["revolute_closest_pt"] = (
                            art_k["revolute_closest_pt"].index_select(1, keep_t)
                        )
                    new_arts.append(art_k)
                preds_list = new_preds
                seg_logits_list = new_seg_logits
                art_list = new_arts

            # ---------- Concat across noise samples --------------------
            pred_concat = torch.cat(preds_list, dim=1)  # [B, N*nq, 3]

            # Aggregate articulation. Global keys → mean across N (axis_dir
            # / pris_axis renormalize). Per-point keys → concat along dim=1.
            articulation_pred: dict = {}
            seg_logits_concat: torch.Tensor | None = None
            if has_art:
                seg_logits_concat = torch.cat(seg_logits_list, dim=1)  # [B, N*nq, 2]
                def _mean_unit(stack):
                    m = stack.mean(dim=0)
                    return m / (m.norm(dim=-1, keepdim=True).clamp_min(1e-8))
                def _stack_or_none(key):
                    vals = [a.get(key) for a in art_list]
                    if any(v is None for v in vals):
                        return None
                    return torch.stack(vals, dim=0)
                ax_stack = _stack_or_none("revolute_axis_dir")
                pl_stack = _stack_or_none("revolute_plucker")
                rr_stack = _stack_or_none("revolute_range")
                pa_stack = _stack_or_none("prismatic_axis")
                pr_stack = _stack_or_none("prismatic_range")
                cp_present = all(
                    a.get("revolute_closest_pt") is not None for a in art_list
                )
                articulation_pred = {
                    "revolute_axis_dir": _mean_unit(ax_stack) if ax_stack is not None else None,
                    "revolute_plucker": pl_stack.mean(dim=0) if pl_stack is not None else None,
                    "revolute_range": rr_stack.mean(dim=0) if rr_stack is not None else None,
                    "prismatic_axis": _mean_unit(pa_stack) if pa_stack is not None else None,
                    "prismatic_range": pr_stack.mean(dim=0) if pr_stack is not None else None,
                    "revolute_closest_pt": (
                        torch.cat([a["revolute_closest_pt"] for a in art_list], dim=1)
                        if cp_present else None
                    ),
                }

            # ---------- GT pts subsample (unchanged) -------------------
            if gt_pts.shape[1] > num_queries:
                idx = torch.randperm(gt_pts.shape[1], device=device)[:num_queries]
                gt_pts = gt_pts[:, idx, :].float()
            else:
                gt_pts = gt_pts.float()

            # ---------- Alignment fit on the FULL concat cloud ---------
            # Alignment is one (scale, shift) per-batch-entry regardless of
            # point count, so fitting on the dense concat gives a more
            # robust chamfer optimum than fitting on a single noise sample.
            gt_valid = torch.ones((gt_pts.shape[:2]), dtype=torch.bool)
            _, _, align_scale, align_shift = scale_shift_alignment_pointcloud(
                pred_concat, gt_pts, gt_valid, return_transform=True, **align_kwargs,
            )

            # ---------- Global per-part filter + subsample to num_queries
            # Per-noise-sample filter already ran above (when do_postproc).
            # Here we (a) optionally apply ONE global per-part filter on
            # the merged cloud — catches residual cross-sample outliers
            # that survived per-sample filtering — and (b) subsample the
            # merged cloud back to ``num_queries`` so the downstream
            # per-state CD/F1 path operates on the same budget as the
            # N=1 baseline. Per-point articulation outputs (seg logits,
            # revolute_closest_pt) are sliced with the same indices to
            # stay in lockstep.
            if has_art and (do_postproc or n_noise > 1):
                pred_seg_concat = seg_logits_concat.argmax(dim=-1).long()
                pred_kept = []
                seg_kept = []
                cp_present_local = (
                    articulation_pred.get("revolute_closest_pt") is not None
                )
                cp_kept: list[torch.Tensor] = [] if cp_present_local else []
                rng_master = np.random.default_rng(
                    int(self.config.get("seed", 0)) * 100003 + val_idx,
                )
                for b in range(B):
                    pts_b = pred_concat[b].float().cpu().numpy()
                    seg_b = pred_seg_concat[b].cpu().numpy().astype(np.int64)
                    if do_postproc:
                        keep_abs = _per_part_filter_indices(
                            pts_b, seg_b, nb_neighbors=pp_gnb, std_ratio=pp_gstd,
                        )
                    else:
                        keep_abs = np.arange(pts_b.shape[0], dtype=np.int64)
                    seg_after = seg_b[keep_abs]
                    if seg_after.shape[0] > subsample_target:
                        sub_idx_local = _subsample_seg_ratio_indices(
                            seg_after, subsample_target, rng_master,
                        )
                    else:
                        sub_idx_local = np.arange(seg_after.shape[0], dtype=np.int64)
                    abs_idx = keep_abs[sub_idx_local]
                    abs_idx_t = torch.from_numpy(abs_idx).long().to(device)
                    pred_kept.append(pred_concat[b].index_select(0, abs_idx_t))
                    seg_kept.append(pred_seg_concat[b].index_select(0, abs_idx_t))
                    if cp_present_local:
                        cp_b = articulation_pred["revolute_closest_pt"][b]
                        cp_kept.append(cp_b.index_select(0, abs_idx_t))
                # All B entries can have different surviving N when
                # do_postproc=true; B=1 in val so a stack is fine.
                pred_s1_unaligned = torch.stack(pred_kept, dim=0)
                pred_seg_for_pred_cloud = torch.stack(seg_kept, dim=0)
                if cp_present_local:
                    articulation_pred["revolute_closest_pt"] = torch.stack(cp_kept, dim=0)
            else:
                # N=1 path. ``pred_concat`` IS the single noise cloud; just
                # forward seg outputs through.
                pred_s1_unaligned = pred_concat
                if has_art:
                    pred_seg_for_pred_cloud = (
                        seg_logits_concat.argmax(dim=-1).long()
                    )

            if has_art:
                gt_plucker = data["plucker"].to(device).float()
                gt_pris_axis = data["prismatic_axis"].to(device).float()
                gt_rev_range_hi = data["revolute_range"][:, 1].to(device).float()
                gt_pris_range_hi = data["prismatic_range"][:, 1].to(device).float()
                is_rev = data["is_revolute"].to(device).bool()
                is_pris = data["is_prismatic"].to(device).bool()

                # Compose a 6-D Plücker for the LARM-style joint metrics.
                # ``plucker`` mode emits it directly; ``per_point_closest``
                # mode reassembles it from axis-direction + median per-
                # point closest-pt over predicted active points
                # (PARTICULATE). When the predicted active mask is empty
                # we fall back to the median over all pred points so the
                # metric still has *some* origin.
                motion_repr_eval = inner.articulation_head.motion_representation
                if motion_repr_eval == "plucker":
                    pred_plucker = articulation_pred["revolute_plucker"].float()
                else:
                    pred_axis_dir = articulation_pred["revolute_axis_dir"].float()
                    pred_closest_pt = articulation_pred["revolute_closest_pt"].float()
                    B_b = pred_axis_dir.shape[0]
                    pred_plucker = torch.zeros(
                        (B_b, 6), device=device, dtype=torch.float32,
                    )
                    pred_active_for_assembly = (pred_seg_for_pred_cloud == 1)
                    for b_i in range(B_b):
                        sel = pred_active_for_assembly[b_i]
                        if sel.any():
                            pts_b = pred_closest_pt[b_i][sel]
                        else:
                            pts_b = pred_closest_pt[b_i]
                        med_pt = pts_b.median(dim=0).values
                        pred_plucker[b_i] = axis_dir_and_point_to_plucker(
                            pred_axis_dir[b_i], med_pt,
                        )
                pred_rev_range = articulation_pred["revolute_range"].float()
                pred_pris_axis = articulation_pred["prismatic_axis"].float()
                pred_pris_range = articulation_pred["prismatic_range"].float()

                input_pts_f = input_pts.float()
                pred_s1_f = pred_s1_unaligned.float()
                # Dataset normalization factor: ``pts_final = pts_cam * s_scale``,
                # where ``s_scale = target_median / norm_factor``. Used to push
                # GT from median-3 frame back to camera frame.
                s_scale_per_sample = (
                    data["pts_target_median"].to(device).float()
                    / data["pts_norm_factor"].to(device).float()
                )  # [B]

                # Complement seg: run the seg head on the GT input cloud
                # (== FM target frame, index-aligned to ``part_mask``) →
                # exact per-point label with no NN transfer. Isolates
                # seg-head quality from reconstruction error.
                seg_on_gt_pred = None
                if seg_eval_on_gt:
                    with torch.inference_mode():
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            seg_gt_out = self.model(
                                images=images,
                                query_points=input_pts_f,
                                timestep=t_ones,
                                run_seg=True,
                                encoder_data=encoder_data,
                            )
                    seg_on_gt_pred = (
                        seg_gt_out["seg_logits"].argmax(dim=-1).long().clone()
                    )

                # Pred side uses the s1 scale-shift alignment fitted
                # above (pred_s1 in median-3 ↔ gt_pts in camera). Same
                # (scale, shift) is shared across all states so qpos≠1
                # rows surface articulation/geometry error cleanly.

                for b in range(B):
                    motion_type = (
                        "revolute" if bool(is_rev[b].item())
                        else ("prismatic" if bool(is_pris[b].item()) else "none")
                    )
                    sid_b = (
                        sample_ids[b] if b < len(sample_ids)
                        else f"{val_idx:06d}_{b}"
                    )
                    larm_bucket = _larm_bucket_for(str(sid_b))
                    cat_b = (cats[b] if b < len(cats) else None) or ""

                    # Skip samples LARM didn't evaluate under the
                    # selected variant — they have no LARM number to
                    # line up with, so contributing them would skew the
                    # ALL-N denom out of LARM-comparable territory.
                    if (self.larm_evaluated_ids is not None
                            and str(sid_b) not in self.larm_evaluated_ids):
                        continue

                    geom_keys, art_keys = _resolve_buckets(cat_b, larm_bucket)
                    has_motion = motion_type != "none"
                    bucket_keys_for_count = list(set(geom_keys) | set(art_keys))
                    _bump_count(bucket_keys_for_count, with_joint=has_motion)

                    # LARM-style joint metrics — only on samples with
                    # motion and only when the sample qualifies for an
                    # articulation bucket (i.e., LARM ran successfully or
                    # this is the Oven OOD experiment). Compute on
                    # camera-frame quantities so units match LARM.
                    if has_motion and art_keys:
                        a_scale_b_f = float(align_scale[b].item())
                        a_shift_b_f = (
                            align_shift[b].squeeze(0).detach().cpu().numpy().astype(np.float64)
                        )
                        s_scale_b_f = float(s_scale_per_sample[b].item())
                        if motion_type == "revolute":
                            pred_pl_norm = pred_plucker[b].detach().cpu().numpy().astype(np.float64)
                            pred_pl_cam = pred_pl_norm.copy()
                            pred_pl_cam[3:6] = a_scale_b_f * pred_pl_norm[3:6] + np.cross(
                                pred_pl_norm[:3], a_shift_b_f,
                            )
                            gt_pl_norm = gt_plucker[b].detach().cpu().numpy().astype(np.float64)
                            gt_pl_cam = gt_pl_norm.copy()
                            gt_pl_cam[3:6] = gt_pl_norm[3:6] / s_scale_b_f
                            m_dict = articulation_metrics_revolute(
                                pred_pl_cam, gt_pl_cam,
                                float(pred_rev_range[b].item()),
                                float(gt_rev_range_hi[b].item()),
                            )
                        else:
                            pred_axis_np = pred_pris_axis[b].detach().cpu().numpy().astype(np.float64)
                            gt_axis_np = gt_pris_axis[b].detach().cpu().numpy().astype(np.float64)
                            pred_range_cam = float(pred_pris_range[b].item()) * a_scale_b_f
                            gt_range_cam = float(gt_pris_range_hi[b].item()) / s_scale_b_f
                            m_dict = articulation_metrics_prismatic(
                                pred_axis_np, gt_axis_np,
                                pred_range_cam, gt_range_cam,
                            )
                        # Pris samples have no axis_origin field; LARM
                        # injects 0.0 so they trivially pass that
                        # threshold and the combined "all" mean stays
                        # apples-to-apples with LARM's table.
                        if motion_type == "prismatic" and "axis_origin" not in m_dict:
                            m_dict["axis_origin"] = 0.0
                        for mname, mval in m_dict.items():
                            _bump_art_metric(art_keys, mname, float(mval))

                    # ---- Per-state geometry CD / F1 ------------------
                    # GT cloud at qpos t: articulate active part by GT
                    # joint when motion ∈ {revolute, prismatic}; else
                    # static (transform = identity).
                    # Pred cloud at qpos t: articulate by pred joint
                    # ONLY when (larm_bucket == "larm_success" ∧ motion);
                    # otherwise hold pred at s1 unchanged. This is the
                    # LARM "frozen pred when joint estimation failed"
                    # rule — see ``larm_exp/metrics/eval.py:198–199``.
                    # The decision is purely on LARM-success / motion;
                    # it is NOT gated by ``art_keys`` so that Oven samples
                    # LARM failed on still take the static-s1 path
                    # (LARM bucket carries through, even though the
                    # articulation-bucket dispatcher routes Oven to its
                    # own ``oven`` key).
                    pred_articulated = (
                        larm_bucket == "larm_success" and has_motion
                    )
                    gt_articulated = has_motion

                    if gt_articulated:
                        gt_active_mask = (part_mask[b] == 1)
                        gt_ctx_gt = input_pts_f[b][~gt_active_mask]
                        gt_active_s1 = input_pts_f[b][gt_active_mask]
                    if pred_articulated:
                        pred_active_mask = (pred_seg_for_pred_cloud[b] == 1)
                        pred_ctx_pred = pred_s1_f[b][~pred_active_mask]
                        pred_active_s1 = pred_s1_f[b][pred_active_mask]
                        if pred_active_s1.shape[0] == 0:
                            # Empty active prediction → fall back to the
                            # static-s1 path (LARM-style frozen pred).
                            pred_articulated = False

                    a_scale_b = align_scale[b]    # [1, 1]
                    a_shift_b = align_shift[b]    # [1, 3]
                    s_scale_b = s_scale_per_sample[b]

                    # Pre-build the static-s1 pred cloud in camera frame
                    # (used for every qpos when pred_articulated=False).
                    pred_static_cam = (
                        a_scale_b * pred_s1_f[b].unsqueeze(0) + a_shift_b
                    )

                    # ---- Part-segmentation metrics (static, at s1) -------
                    # Over the articulated subset (has a movable part).
                    #   * end-to-end: label the predicted cloud, NN-transfer
                    #     onto GT points, score vs part_mask (the deployed
                    #     "key output").
                    #   * seg-on-GT (optional): seg head queried directly on
                    #     GT geometry — exact, index-aligned, no NN.
                    gt_has_movable = (
                        gt_articulated and bool((part_mask[b] == 1).any())
                    )
                    if gt_has_movable:
                        gt_lbl_b = part_mask[b].long()
                        gt_cam_seg = (input_pts_f[b] / s_scale_b).unsqueeze(0)
                        nn = knn_points(gt_cam_seg, pred_static_cam, K=1)
                        nn_idx = nn.idx[0, :, 0]
                        pred_lbl_on_gt = pred_seg_for_pred_cloud[b].index_select(
                            0, nn_idx,
                        )
                        seg_scores = _binary_seg_scores(pred_lbl_on_gt, gt_lbl_b)
                        for mname, mval in seg_scores.items():
                            _bump_seg_metric(geom_keys, mname, mval)
                        tp, fp, fn = _binary_seg_counts(pred_lbl_on_gt, gt_lbl_b)
                        _bump_seg_counts(geom_keys, "e2e", tp, fp, fn)
                        seg_scores_gt = None
                        if seg_on_gt_pred is not None:
                            seg_scores_gt = _binary_seg_scores(
                                seg_on_gt_pred[b], gt_lbl_b,
                            )
                            for mname, mval in seg_scores_gt.items():
                                _bump_seg_metric(
                                    geom_keys, f"{mname}_ongt", mval,
                                )
                            tpg, fpg, fng = _binary_seg_counts(
                                seg_on_gt_pred[b], gt_lbl_b,
                            )
                            _bump_seg_counts(geom_keys, "ongt", tpg, fpg, fng)
                        rec_seg = per_sample_metrics.setdefault(str(sid_b), {
                            "category": cat_b,
                            "motion_type": motion_type,
                            "larm_bucket": larm_bucket,
                            "states": {},
                        })
                        rec_seg["seg"] = seg_scores
                        if seg_scores_gt is not None:
                            rec_seg["seg_ongt"] = seg_scores_gt

                        # ---- Segmentation-ISOLATED part reconstruction -------
                        # Choose the prediction's movable/static split by GT
                        # labels NN-transferred onto the reconstruction (the
                        # movable region is defined by GT geometry, NOT the seg
                        # head), so the number excludes segmentation error. This
                        # is the same NN label transfer used by the multipart
                        # eval and by PARTICULATE's part matching.
                        nn_p2g = knn_points(pred_static_cam, gt_cam_seg, K=1)
                        pred_gtlbl = part_mask[b].long().index_select(
                            0, nn_p2g.idx[0, :, 0],
                        )                                        # [M] 0/1 by GT
                        pred_mov_gtseg_s1 = pred_s1_f[b][pred_gtlbl == 1]
                        pred_sta_gtseg_s1 = pred_s1_f[b][pred_gtlbl != 1]
                        # (b) PURE geometry: scored at rest (s1). Chamfer is
                        # invariant to a common rigid joint transform, so this
                        # equals the GT-joint-articulated number and removes BOTH
                        # segmentation and joint error -> pure movable-part
                        # reconstruction quality.
                        _mvb = _part_cd_f1(
                            a_scale_b * pred_mov_gtseg_s1.unsqueeze(0) + a_shift_b,
                            (gt_active_s1 / s_scale_b).unsqueeze(0),
                        )
                        if _mvb is not None:
                            _bump_geom_field(
                                "geom_active_gtseg_pure", geom_keys, "1.00",
                                _mvb[0], _mvb[1],
                            )
                        _svb = _part_cd_f1(
                            a_scale_b * pred_sta_gtseg_s1.unsqueeze(0) + a_shift_b,
                            (gt_ctx_gt / s_scale_b).unsqueeze(0),
                        )
                        if _svb is not None:
                            _bump_geom_field(
                                "geom_static_gtseg_pure", geom_keys, "1.00",
                                _svb[0], _svb[1],
                            )

                    for t_state in self.eval_states:
                        delta = float(t_state) - 1.0
                        # ---- GT side ---------------------------------
                        if gt_articulated:
                            if motion_type == "revolute":
                                gt_angle = (gt_rev_range_hi[b] * delta).float()
                                T_gt = plucker_to_4x4_torch(
                                    gt_plucker[b].float(), gt_angle,
                                )
                            else:
                                d_gt = (gt_pris_range_hi[b] * delta).float()
                                T_gt = prismatic_to_4x4_torch(
                                    gt_pris_axis[b].float(), d_gt,
                                )
                            gt_active_t = (
                                gt_active_s1 @ T_gt[:3, :3].T + T_gt[:3, 3]
                            )
                            gt_at_t_norm = torch.cat(
                                [gt_ctx_gt, gt_active_t], dim=0
                            ).unsqueeze(0)
                        else:
                            gt_at_t_norm = input_pts_f[b].unsqueeze(0)
                        gt_at_t = gt_at_t_norm / s_scale_b

                        # ---- Pred side --------------------------------
                        if pred_articulated:
                            if motion_type == "revolute":
                                pred_angle = (pred_rev_range[b] * delta).float()
                                T_pred = plucker_to_4x4_torch(
                                    pred_plucker[b].float(), pred_angle,
                                )
                            else:
                                axis_pred = pred_pris_axis[b].float()
                                axis_pred_unit = axis_pred / (axis_pred.norm() + 1e-12)
                                d_pred = (pred_pris_range[b] * delta).float()
                                T_pred = prismatic_to_4x4_torch(
                                    axis_pred_unit, d_pred,
                                )
                            pred_active_t = (
                                pred_active_s1 @ T_pred[:3, :3].T + T_pred[:3, 3]
                            )
                            pred_at_t_norm = torch.cat(
                                [pred_ctx_pred, pred_active_t], dim=0
                            ).unsqueeze(0)
                            pred_at_t = a_scale_b * pred_at_t_norm + a_shift_b
                        else:
                            pred_at_t = pred_static_cam

                        d_tup, _ = chamfer_distance(
                            pred_at_t, gt_at_t,
                            batch_reduction=None, point_reduction=None, norm=2,
                        )
                        d_p, d_g = d_tup
                        d_p = torch.sqrt(d_p)
                        d_g = torch.sqrt(d_g)
                        cd_t = ((d_p.mean(dim=1) + d_g.mean(dim=1)) / 2.0).item()
                        prec = (d_p < fs_thres).float().mean(dim=1)
                        rec = (d_g < fs_thres).float().mean(dim=1)
                        f1_t = (2.0 * prec * rec / (prec + rec + 1e-8)).item()

                        sk = _state_key(t_state)
                        _bump_geom(geom_keys, sk, cd_t, f1_t)

                        # ---- Per-part CD/F1 (movable vs static) ----------
                        # Only over samples with a GT movable part. Pred is
                        # split by predicted seg; GT by part_mask. The movable
                        # part is articulated by its own joint at this state
                        # (sub-clouds built above); when the pred joint was
                        # frozen (fail), the static-s1 pred cloud is split by
                        # predicted seg instead. The static part is state-
                        # invariant, so it is scored once (at delta==0).
                        if gt_has_movable:
                            gt_active_cam = (
                                gt_active_t / s_scale_b
                            ).unsqueeze(0)
                            if pred_articulated:
                                pred_active_cam = (
                                    a_scale_b * pred_active_t.unsqueeze(0)
                                    + a_shift_b
                                )
                                pred_static_part_cam = (
                                    a_scale_b * pred_ctx_pred.unsqueeze(0)
                                    + a_shift_b
                                )
                            else:
                                pmask = (pred_seg_for_pred_cloud[b] == 1)
                                pred_active_cam = (
                                    a_scale_b * pred_s1_f[b][pmask].unsqueeze(0)
                                    + a_shift_b
                                )
                                pred_static_part_cam = (
                                    a_scale_b
                                    * pred_s1_f[b][~pmask].unsqueeze(0)
                                    + a_shift_b
                                )
                            gt_static_cam = (
                                gt_ctx_gt / s_scale_b
                            ).unsqueeze(0)
                            part_specs = [
                                ("geom_state_counts_active",
                                 pred_active_cam, gt_active_cam),
                            ]
                            if delta == 0.0:
                                part_specs.append(
                                    ("geom_state_counts_static",
                                     pred_static_part_cam, gt_static_cam),
                                )
                            for field, p_cloud, g_cloud in part_specs:
                                if (p_cloud.shape[1] == 0
                                        or g_cloud.shape[1] == 0):
                                    continue
                                dtp, _ = chamfer_distance(
                                    p_cloud, g_cloud,
                                    batch_reduction=None,
                                    point_reduction=None, norm=2,
                                )
                                dpp, dgg = dtp
                                dpp = torch.sqrt(dpp)
                                dgg = torch.sqrt(dgg)
                                cd_pp = (
                                    (dpp.mean(dim=1) + dgg.mean(dim=1)) / 2.0
                                ).item()
                                pp = (dpp < fs_thres).float().mean(dim=1)
                                rr = (dgg < fs_thres).float().mean(dim=1)
                                f1_pp = (
                                    2.0 * pp * rr / (pp + rr + 1e-8)
                                ).item()
                                _bump_geom_field(
                                    field, geom_keys, sk, cd_pp, f1_pp,
                                )

                            # (a) seg-ISOLATED, articulated by the PREDICTED
                            # joint: the GT-seg-selected movable prediction
                            # (pred_mov_gtseg_s1, from the seg block above)
                            # articulated by the pred joint vs GT movable by GT
                            # joint. Removes segmentation error but keeps joint
                            # error, so the gap to (b)_pure isolates articulation.
                            if pred_articulated:
                                pmov_t = (
                                    pred_mov_gtseg_s1 @ T_pred[:3, :3].T
                                    + T_pred[:3, 3]
                                )
                                _mva = _part_cd_f1(
                                    a_scale_b * pmov_t.unsqueeze(0) + a_shift_b,
                                    (gt_active_t / s_scale_b).unsqueeze(0),
                                )
                                if _mva is not None:
                                    _bump_geom_field(
                                        "geom_active_gtseg_pj", geom_keys, sk,
                                        _mva[0], _mva[1],
                                    )

                        # Per-sample (cd, f1) accumulation for offline use.
                        sid_str_b = str(sid_b)
                        rec = per_sample_metrics.setdefault(sid_str_b, {
                            "category": cat_b,
                            "motion_type": motion_type,
                            "larm_bucket": larm_bucket,
                            "states": {},
                        })
                        rec["states"][sk] = {"cd": float(cd_t), "f1": float(f1_t)}

                        # Per-state articulated-cloud viz (rank-0,
                        # interval-gated). Only emit when both sides
                        # have an active subset to draw — otherwise the
                        # merged-state PLY would degenerate to a single
                        # static cloud per state, which is already
                        # captured by the s1 row.
                        if (is_rank0
                            and val_idx % save_pcd_interval == 0
                            and pred_articulated and gt_articulated):
                            pred_seg_viz = torch.cat([
                                torch.zeros(pred_ctx_pred.shape[0], dtype=torch.long, device=device),
                                torch.ones(pred_active_s1.shape[0], dtype=torch.long, device=device),
                            ], dim=0)
                            gt_seg_viz = torch.cat([
                                torch.zeros(gt_ctx_gt.shape[0], dtype=torch.long, device=device),
                                torch.ones(gt_active_s1.shape[0], dtype=torch.long, device=device),
                            ], dim=0)
                            sid_str = str(sid_b)
                            art_viz_by_sid[sid_str].append((
                                sk,
                                pred_at_t.squeeze(0).detach().cpu(),
                                pred_seg_viz.detach().cpu(),
                                gt_at_t.squeeze(0).detach().cpu(),
                                gt_seg_viz.detach().cpu(),
                            ))
                            # Flush this sid as soon as all 5 states arrive.
                            if len(art_viz_by_sid[sid_str]) >= n_eval_states:
                                _flush_sid(sid_str)

            # ---------- Buffer artifacts at intervals (rank 0) ---------
            if is_rank0 and val_idx % save_pcd_interval == 0:
                # Per-iter: write input-image PNGs immediately, no buffer.
                # The s1 pred cloud is no longer dumped here — the merged-
                # state PLY built per-sid (when all 5 states arrive) covers
                # it as the qpos=1.00 layer.
                imgs_to_save = (input_images.detach() * 0.5 + 0.5).clamp(0, 1)
                imgs_to_save = (imgs_to_save * 255.0).to(torch.uint8)
                imgs_to_save = imgs_to_save.permute(0, 1, 3, 4, 2).contiguous().cpu().numpy()
                batch_ids = data.get("sample_id", None)
                if batch_ids is None:
                    batch_ids = [
                        f"{val_idx:06d}_{i}" for i in range(B)
                    ]
                for bi in range(imgs_to_save.shape[0]):
                    name = (
                        str(batch_ids[bi])
                        if bi < len(batch_ids)
                        else f"{val_idx:06d}_{bi}"
                    )
                    for v in range(imgs_to_save.shape[1]):
                        Image.fromarray(imgs_to_save[bi, v]).save(
                            os.path.join(pcd_dir, f"{name}_img_v{v}.png")
                        )

            if val_idx % 20 == 0:
                torch.cuda.empty_cache()

            del encoder_data, preds_list, pred_concat
            del images, gt_pts, input_pts, part_mask, data

        # ================================================================
        # DDP gather — single pass over the flat bucket map.
        # ================================================================
        is_dist = dist.is_available() and dist.is_initialized()
        if is_dist:
            world_size = dist.get_world_size()
            gathered_buckets: list[dict | None] = [None] * world_size
            dist.all_gather_object(gathered_buckets, val_buckets)
            gathered_per_sample: list[dict | None] = [None] * world_size
            dist.all_gather_object(gathered_per_sample, per_sample_metrics)
        else:
            gathered_buckets = [val_buckets]
            gathered_per_sample = [per_sample_metrics]

        # Non-rank-0 ranks are done; rank 0 handles all logging + artifacts.
        if not is_rank0:
            self.val_step += 1
            del wrapper, solver
            torch.cuda.empty_cache()
            self.model.train()
            return None

        # Merge per-rank bucket maps. ``geom_state_counts`` cells sum;
        # ``art_metrics`` lists concatenate; ``n_total`` / ``n_with_joint``
        # sum.
        merged: dict[str, dict] = {}
        for rd in gathered_buckets:
            if not rd:
                continue
            for key, payload in rd.items():
                m = merged.setdefault(key, {
                    "geom_state_counts": {},
                    "geom_state_counts_active": {},
                    "geom_state_counts_static": {},
                    "geom_active_gtseg_pure": {},
                    "geom_static_gtseg_pure": {},
                    "geom_active_gtseg_pj": {},
                    "art_metrics": {},
                    "seg_metrics": {},
                    "seg_counts": {},
                    "n_total": 0,
                    "n_with_joint": 0,
                })
                m["n_total"] += int(payload.get("n_total", 0))
                m["n_with_joint"] += int(payload.get("n_with_joint", 0))
                for field in ("geom_state_counts",
                              "geom_state_counts_active",
                              "geom_state_counts_static",
                              "geom_active_gtseg_pure",
                              "geom_static_gtseg_pure",
                              "geom_active_gtseg_pj"):
                    for sk, counts in payload.get(field, {}).items():
                        s = m[field].setdefault(
                            sk, {"cd_sum": 0.0, "f1_sum": 0.0, "n": 0},
                        )
                        s["cd_sum"] += counts.get("cd_sum", 0.0)
                        s["f1_sum"] += counts.get("f1_sum", 0.0)
                        s["n"] += int(counts.get("n", 0))
                for mname, vals in payload.get("art_metrics", {}).items():
                    m["art_metrics"].setdefault(mname, []).extend(vals)
                for mname, vals in payload.get("seg_metrics", {}).items():
                    m["seg_metrics"].setdefault(mname, []).extend(vals)
                for variant, c in payload.get("seg_counts", {}).items():
                    mc = m["seg_counts"].setdefault(
                        variant, {"tp": 0.0, "fp": 0.0, "fn": 0.0},
                    )
                    mc["tp"] += c.get("tp", 0.0)
                    mc["fp"] += c.get("fp", 0.0)
                    mc["fn"] += c.get("fn", 0.0)

        thresholds = {
            "axis_angle": self.metric_thr_axis_angle,
            "axis_origin": self.metric_thr_axis_origin,
            "Mr": self.metric_thr_Mr,
            "Md": self.metric_thr_Md,
        }
        # Map metric → (rate_key) used in the wandb tag. The user spec
        # uses ``axis`` / ``origin`` / ``Mr`` for the success-rate keys
        # rather than the full metric names.
        rate_key_map = {
            "axis_angle": "axis",
            "axis_origin": "origin",
            "Mr": "Mr",
        }

        def _bucket_geom_summary(payload: dict,
                                 field: str = "geom_state_counts") -> dict:
            """Return per-state CD/F1 means + the cross-state ``mean`` row.

            Output: {"qpos-{t}": {"cd", "f1", "n"}, ..., "qpos-mean": {"cd", "f1", "n_states"}}.
            States are sorted by their float key. Empty payload → empty dict.
            ``field`` selects which counts dict to summarize (whole-object
            ``geom_state_counts`` or per-part ``geom_state_counts_active`` /
            ``_static``).
            """
            out: dict[str, dict[str, float]] = {}
            sum_cd = 0.0
            sum_f1 = 0.0
            cnt_states = 0
            for sk, counts in sorted(
                payload.get(field, {}).items(),
                key=lambda kv: float(kv[0]),
            ):
                n = int(counts.get("n", 0))
                if n == 0:
                    continue
                cd = counts["cd_sum"] / n
                f1 = counts["f1_sum"] / n
                out[f"qpos-{sk}"] = {"cd": cd, "f1": f1, "n": n}
                sum_cd += cd
                sum_f1 += f1
                cnt_states += 1
            if cnt_states > 0:
                out["qpos-mean"] = {
                    "cd": sum_cd / cnt_states,
                    "f1": sum_f1 / cnt_states,
                    "n_states": cnt_states,
                }
            return out

        def _bucket_seg_summary(payload: dict) -> dict:
            """Segmentation summary: macro (per-sample mean) IoU-based
            metrics + micro (pooled tp/fp/fn) movable precision/recall/F1.

            Output: {"{mname}": {"mean", "n"}}. For macro metrics ``n`` is
            the number of scored samples; for micro precision/recall/F1
            (``seg_precision`` / ``seg_recall`` / ``seg_f1`` and their
            ``_ongt`` variants) ``n`` is the pooled movable-point count.
            """
            out: dict[str, dict[str, float]] = {}
            for mname, vals in payload.get("seg_metrics", {}).items():
                arr = np.asarray(vals, dtype=np.float64)
                valid = arr[~np.isnan(arr)]
                if valid.size == 0:
                    continue
                out[mname] = {
                    "mean": float(valid.mean()),
                    "n": int(valid.size),
                }
            for variant, c in payload.get("seg_counts", {}).items():
                tp, fp, fn = c.get("tp", 0.0), c.get("fp", 0.0), c.get("fn", 0.0)
                prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
                rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
                if (not math.isnan(prec) and not math.isnan(rec)
                        and (prec + rec) > 0):
                    f1 = 2.0 * prec * rec / (prec + rec)
                else:
                    f1 = float("nan")
                suffix = "" if variant == "e2e" else "_ongt"
                n_pool = int(tp + fn)  # pooled GT movable points
                out[f"seg_precision{suffix}"] = {"mean": prec, "n": n_pool}
                out[f"seg_recall{suffix}"] = {"mean": rec, "n": n_pool}
                out[f"seg_f1{suffix}"] = {"mean": f1, "n": n_pool}
            return out

        def _bucket_art_summary(payload: dict, n_population: int) -> dict:
            """Return joint-metric means + success-rates.

            Output:
              {
                "{mname}":               {"mean", "n", "thr", "with_joint", "all"}  for mname in (axis_angle, axis_origin, Mr, Md)
              }
            ``with_joint`` denominator = ``n_with_joint`` from the
            payload (samples in this articulation bucket that have
            motion). ``all`` denominator = ``n_population`` — the total
            non-Oven val population for ``Val-Articulation`` /
            ``Val-{class}-Articulation``, or the total Oven population
            for ``Val-Oven``. This makes ``success_rate_all`` row-for-
            row comparable to LARM's "over ALL N, missing counted as
            fail" column even though our articulation bucket excludes
            the LARM-fail samples.
            """
            out: dict[str, dict[str, float]] = {}
            n_with_joint = int(payload.get("n_with_joint", 0))
            for mname in ("axis_angle", "axis_origin", "Mr", "Md"):
                vals = payload.get("art_metrics", {}).get(mname, [])
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=np.float64)
                thr = thresholds.get(mname)
                entry = {"mean": float(arr.mean()), "n": int(arr.size)}
                if thr is not None:
                    pass_mask = arr < thr
                    n_pass = int(pass_mask.sum())
                    entry["thr"] = float(thr)
                    if n_with_joint > 0:
                        entry["with_joint"] = float(n_pass / n_with_joint)
                    if n_population > 0 and mname in rate_key_map:
                        # success_rate_all is only emitted for the LARM
                        # triplet (axis / origin / Mr) per spec. Denom
                        # is the broader val-population count, not the
                        # articulation bucket size.
                        entry["all"] = float(n_pass / n_population)
                out[mname] = entry
            return out

        # ================================================================
        # Tag mapping — bucket key → (top-level wandb tag, kind flag).
        # ``kind`` ∈ {"oven", "geom", "art"} controls which sub-emitter
        # writes the bucket's metrics.
        # ================================================================
        def _tag_for(bucket_key: str) -> tuple[str | None, str]:
            if bucket_key == "oven":
                return "Val-Oven", "oven"
            if bucket_key == "geometry":
                return "Val-Geometry", "geom"
            if bucket_key == "articulation":
                return "Val-Articulation", "art"
            if bucket_key.startswith("class:") and bucket_key.endswith("-geom"):
                cat = bucket_key[len("class:"):-len("-geom")]
                return f"Val-{cat}-Geometry", "geom"
            if bucket_key.startswith("class:") and bucket_key.endswith("-art"):
                cat = bucket_key[len("class:"):-len("-art")]
                return f"Val-{cat}-Articulation", "art"
            return None, ""

        # Order tags: top-level first, then per-class alphabetically.
        # Within per-class, Geometry then Articulation (mirrors the user's
        # ``Val/{class}/Articulation`` + ``Val/{class}/Geometry`` listing).
        def _bucket_sort_key(bk: str) -> tuple[int, str]:
            if bk == "oven":
                return (0, "")
            if bk == "geometry":
                return (1, "")
            if bk == "articulation":
                return (2, "")
            return (3, bk)

        # ================================================================
        # Stdout log + wandb log_dict assembly.
        # ================================================================
        log_dict: dict[str, float] = {}
        if self.wandb_run is not None:
            log_dict["val/step"] = float(self.val_step)

        # Helper: for an art bucket, look up the matching geometry
        # bucket's n_total — this is the "ALL N, missing counted as
        # fail" denominator for ``success_rate_all``. For ``Val-Oven``,
        # the Oven bucket is unified (geom + art), so n_population is
        # its own n_total.
        def _population_for(bucket_key: str) -> int:
            if bucket_key == "oven":
                return int(merged.get("oven", {}).get("n_total", 0))
            if bucket_key == "articulation":
                return int(merged.get("geometry", {}).get("n_total", 0))
            if bucket_key.startswith("class:") and bucket_key.endswith("-art"):
                cat = bucket_key[len("class:"):-len("-art")]
                geom_key = f"class:{cat}-geom"
                return int(merged.get(geom_key, {}).get("n_total", 0))
            return 0

        def _emit(tag: str, bucket_key: str, payload: dict, kind: str) -> None:
            n_total = int(payload.get("n_total", 0))
            n_with_joint = int(payload.get("n_with_joint", 0))
            n_population = _population_for(bucket_key)

            # ---- per-qpos geometry rows (oven + geom) -----------------
            if kind in ("oven", "geom"):
                geom_sum = _bucket_geom_summary(payload)
                for row_key, m in geom_sum.items():
                    if row_key == "qpos-mean":
                        line = (
                            f"[{tag}/{row_key}] CD={m['cd']:.6f}  "
                            f"F1@{fs_thres}={m['f1']:.4f}  "
                            f"n_states={m['n_states']}"
                        )
                    else:
                        line = (
                            f"[{tag}/{row_key}] CD={m['cd']:.6f}  "
                            f"F1@{fs_thres}={m['f1']:.4f}  n={m['n']}"
                        )
                    logger.info(line)
                    log_dict[f"{tag}/{row_key}/cd"] = m["cd"]
                    log_dict[f"{tag}/{row_key}/f1@{fs_thres}"] = m["f1"]
                    if "n" in m:
                        log_dict[f"{tag}/{row_key}/n"] = float(m["n"])

                # ---- per-part reconstruction rows (movable / static) ---
                for part, field in (
                    ("movable", "geom_state_counts_active"),
                    ("static", "geom_state_counts_static"),
                ):
                    part_sum = _bucket_geom_summary(payload, field)
                    for row_key, m in part_sum.items():
                        if row_key == "qpos-mean":
                            logger.info(
                                f"[{tag}/part-{part}/{row_key}] "
                                f"CD={m['cd']:.6f}  F1@{fs_thres}={m['f1']:.4f}  "
                                f"n_states={m['n_states']}"
                            )
                        else:
                            logger.info(
                                f"[{tag}/part-{part}/{row_key}] "
                                f"CD={m['cd']:.6f}  F1@{fs_thres}={m['f1']:.4f}  "
                                f"n={m['n']}"
                            )
                        log_dict[f"{tag}/part-{part}/{row_key}/cd"] = m["cd"]
                        log_dict[
                            f"{tag}/part-{part}/{row_key}/f1@{fs_thres}"
                        ] = m["f1"]

                # ---- segmentation-ISOLATED part-recon rows -------------
                # Movable/static split of the prediction chosen by GT labels
                # (NN-transfer), so these exclude seg-head error. ``pure`` =
                # seg- and joint-isolated (scored at rest); ``predjoint`` =
                # seg-isolated only, articulated by the predicted joint.
                for label, field in (
                    ("movable-gtseg-pure", "geom_active_gtseg_pure"),
                    ("static-gtseg-pure", "geom_static_gtseg_pure"),
                    ("movable-gtseg-predjoint", "geom_active_gtseg_pj"),
                ):
                    iso_sum = _bucket_geom_summary(payload, field)
                    m = iso_sum.get("qpos-mean")
                    if m is None:
                        continue
                    logger.info(
                        f"[{tag}/part-{label}/qpos-mean] "
                        f"CD={m['cd']:.6f}  F1@{fs_thres}={m['f1']:.4f}  "
                        f"n_states={m['n_states']}"
                    )
                    log_dict[f"{tag}/part-{label}/cd"] = m["cd"]
                    log_dict[f"{tag}/part-{label}/f1@{fs_thres}"] = m["f1"]

                # ---- part-segmentation rows ----------------------------
                seg_sum = _bucket_seg_summary(payload)
                for mname, entry in seg_sum.items():
                    logger.info(
                        f"[{tag}/seg/{mname}] mean={entry['mean']:.4f}  "
                        f"n={entry['n']}"
                    )
                    log_dict[f"{tag}/seg/{mname}/mean"] = entry["mean"]

            # ---- joint metrics (oven + art) ---------------------------
            if kind in ("oven", "art"):
                art_sum = _bucket_art_summary(payload, n_population)
                for mname, entry in art_sum.items():
                    parts = [f"mean={entry['mean']:.4f}", f"n_with_joint={entry['n']}"]
                    if "with_joint" in entry:
                        parts.append(
                            f"success_with_joint@{entry['thr']:.3g}="
                            f"{entry['with_joint']:.4f}"
                        )
                    if "all" in entry:
                        parts.append(
                            f"success_all@{entry['thr']:.3g}="
                            f"{entry['all']:.4f}"
                        )
                    logger.info(
                        f"[{tag}/joint/{mname}] " + "  ".join(parts)
                    )
                    log_dict[f"{tag}/joint/{mname}/mean"] = entry["mean"]
                    if "with_joint" in entry:
                        rkey = rate_key_map.get(mname, mname)
                        log_dict[
                            f"{tag}/joint/{rkey}/success_rate_with_joint"
                        ] = entry["with_joint"]
                    if "all" in entry:
                        rkey = rate_key_map.get(mname, mname)
                        log_dict[
                            f"{tag}/joint/{rkey}/success_rate_all"
                        ] = entry["all"]

            # Bucket-level population counters — useful for sanity-checking
            # n_with_joint / n_total against LARM's table.
            log_dict[f"{tag}/n_total"] = float(n_total)
            log_dict[f"{tag}/n_with_joint"] = float(n_with_joint)

        for bucket_key in sorted(merged.keys(), key=_bucket_sort_key):
            tag, kind = _tag_for(bucket_key)
            if tag is None:
                continue
            payload = merged[bucket_key]
            if payload.get("n_total", 0) == 0:
                continue
            _emit(tag, bucket_key, payload, kind)

        if self.wandb_run is not None:
            self.wandb_run.log(log_dict)

        # ================================================================
        # Artifact save: input-image PNGs + per-sid merged-state PLYs are
        # both written incrementally inside the val loop (rank 0 only).
        # Here we just flush stragglers — sids that didn't reach all 5
        # states (shouldn't happen with the current pipeline since
        # pred_articulated/gt_articulated are stable across the per-state
        # loop, but kept for behavior parity with the old buffered path).
        # ================================================================
        if is_rank0 and art_viz_by_sid:
            for sid in list(art_viz_by_sid.keys()):
                _flush_sid(sid)

        # Write per-sample CD/F1 JSON (rank 0). Useful for offline ranking +
        # visualization selection. Computes per-sample qpos-mean F1/CD from
        # the per-state values for convenient sorting.
        if is_rank0:
            merged_per_sample: dict[str, dict] = {}
            for rd in gathered_per_sample:
                if not rd:
                    continue
                for sid, payload in rd.items():
                    if sid not in merged_per_sample:
                        merged_per_sample[sid] = payload
                    else:
                        merged_per_sample[sid]["states"].update(payload.get("states", {}))
            for sid, rec in merged_per_sample.items():
                states = rec.get("states", {})
                if not states:
                    rec["qpos_mean"] = {"cd": None, "f1": None, "n_states": 0}
                    continue
                f1s = [v["f1"] for v in states.values()]
                cds = [v["cd"] for v in states.values()]
                rec["qpos_mean"] = {
                    "cd": float(np.mean(cds)),
                    "f1": float(np.mean(f1s)),
                    "n_states": len(states),
                }
            per_sample_path = os.path.join(
                self.runner_info.work_dir, f"per_sample_metrics_{timestamp}.json",
            )
            with open(per_sample_path, "w") as f:
                json.dump(merged_per_sample, f, indent=2)
            logger.info(
                f"[Val] per-sample CD/F1 for {len(merged_per_sample)} samples -> "
                f"{per_sample_path}"
            )
        if merged_count:
            logger.info(
                f"[Val] {merged_count} merged-state PLYs (ctx + 5 state-colored "
                f"active layers) saved -> {pcd_dir}"
            )

        self.val_step += 1
        del wrapper, solver
        torch.cuda.empty_cache()
        self.model.train()
        return None
