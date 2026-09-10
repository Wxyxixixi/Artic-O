"""Reconstruct one object and pose it at a given articulation state.

Takes one sample from the test split, runs the model forward once, and writes
a point cloud per requested state. Reports a timing breakdown.

    python scripts/infer_one.py \
        --ckpt-path ./checkpoints/artic_o_s0_36.pth \
        --sample 48379_joint_0 --state 1.0 0.5 0.0 \
        --out-dir ./work_dirs/infer

What the model predicts, and what it does not:

* Predicted: the complete point cloud at state s1, the movable-part
  segmentation, and the joint's axis direction, origin and range.
* Read from the sample, NOT predicted: whether the joint is revolute or
  prismatic, and whether it moves at all. ``val_epoch`` makes the same
  choice (``artic_o_trainer.py:2321-2356``), so this matches how the
  paper's numbers were produced.

Two things this deliberately omits, both of which need ground truth and so
have no place in inference:

* The chamfer ``(scale, shift)`` alignment to GT that ``val_epoch`` fits
  before scoring. Output here is the *unaligned* prediction, so it will not
  sit on top of a GT cloud.
* The per-part outlier filtering that the alignment path applies.

State convention (same as evaluation, and the usual surprise): ``delta =
state - 1.0``. **s1 is the anchor: --state 1.0 is identity**, and
--state 0.0 walks the movable part back by the full predicted range.

Frame: dataset-normalized by default (what the network emits). ``--frame
camera`` divides by the per-sample ``s_scale``. The dataset world frame is
not reachable without the per-sample camera metadata, which this release
does not ship.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate

# Python puts this script's own directory on sys.path, not the repo root, so
# ``src`` is not importable without help. Resolve it from __file__ so the
# script works from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.configs import build_dataset, build_model, load_config
from src.datasets.articulation import plucker_to_4x4_torch, prismatic_to_4x4_torch
from src.datasets.utils import save_pointcloud, save_segmented_pointcloud
from src.flow_matching.solver import ODESolver
from src.models.heads.articulation import axis_dir_and_point_to_plucker
from src.models.model_wrapper import BatchModelWrapper

TIMINGS: dict[str, float] = {}


@contextmanager
def timed(label: str, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if device.type == "cuda":
        torch.cuda.synchronize()
    TIMINGS[label] = time.perf_counter() - t0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./src/configs/artic_o.yaml")
    ap.add_argument("--ckpt-path", "--ckpt_path", dest="ckpt_path", required=True)
    ap.add_argument("--sample", default=None,
                    help="sample id (e.g. 48379_joint_0); defaults to --index")
    ap.add_argument("--index", type=int, default=0,
                    help="dataset index, used when --sample is omitted")
    ap.add_argument("--state", type=float, nargs="+", default=[1.0, 0.5, 0.0],
                    help="articulation states; 1.0 is the anchor (identity)")
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir", default="./work_dirs/infer")
    ap.add_argument("--num-queries", "--num_queries", dest="num_queries", type=int,
                    default=None, help="ODE query points (default: config val_num_queries)")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    with timed("build_model", device):
        model = build_model(cfg).to(device).eval()

    with timed("load_checkpoint", device):
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"WARNING: {len(missing)} missing / {len(unexpected)} unexpected keys "
              f"— the prediction below is not trustworthy")

    with timed("open_dataset", device):
        ds = build_dataset(cfg.val_dataloader.dataset)

    idx = args.index
    if args.sample is not None:
        # Pull the id column straight off the Arrow table — going through
        # __getitem__ would decode 6 images and 30k points per sample.
        ids = [str(x) for x in ds._ds.select_columns(["sample_id"])["sample_id"]]
        if args.sample not in ids:
            raise SystemExit(f"sample {args.sample!r} not in the split "
                             f"({len(ids)} samples; e.g. {ids[:3]})")
        idx = ids.index(args.sample)
    data = default_collate([ds[idx]])
    sample_id = str(data["sample_id"][0])

    images = data["image"].to(device)
    state_tag = data["state_tag"].to(device) if "state_tag" in data else None
    # s_scale = target_median / norm_factor  (artic_o_trainer.py:2097-2100)
    s_scale = float(data["pts_target_median"][0]) / float(data["pts_norm_factor"][0])
    # Joint type is read from the sample, not predicted
    # (artic_o_trainer.py:2127-2130).
    motion_type = ("revolute" if bool(data["is_revolute"][0])
                   else "prismatic" if bool(data["is_prismatic"][0]) else "none")

    step_size = cfg.get("fm_step_size", 0.04)
    method = cfg.get("fm_sampling", "euler")
    num_queries = args.num_queries or int(cfg.get("val_num_queries", 8192))
    T = torch.linspace(0, 1, round(1.0 / step_size) + 1, device=device)

    inner = model.module if hasattr(model, "module") else model
    solver = ODESolver(velocity_model=BatchModelWrapper(model=model))
    t_ones = torch.ones((1, 1), device=device)

    def forward_once():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc = inner._encode(images=images, pointmaps=None, test=True,
                                state_tag=state_tag)
            x_init = torch.rand(1, num_queries, 3, device=device) * 2.0 - 1.0
            pred = solver.sample(
                x_init=x_init, time_grid=T, method=method, step_size=step_size,
                return_intermediates=False, images=images, token_mask=None,
                encoder_data=enc, pointmaps=None,
            ).float()
            out = model(images=images, query_points=pred, timestep=t_ones,
                        run_seg=True, encoder_data=enc)
        return pred, out

    if device.type == "cuda":                       # discard launch-queue noise
        forward_once()

    with timed("forward_total", device):
        pred, seg_out = forward_once()

    pred_s1 = pred[0].float()                                   # [N, 3]
    seg = seg_out["seg_logits"].float().argmax(dim=-1).long()[0]  # [N]
    art = {k: (v.float() if torch.is_tensor(v) else v)
           for k, v in seg_out["articulation"].items()}
    active = (seg == 1)

    # Reassemble the 6-D Plucker exactly as val_epoch does
    # (artic_o_trainer.py:2068-2088).
    if inner.articulation_head.motion_representation == "plucker":
        plucker = art["revolute_plucker"][0]
    else:
        cp = art["revolute_closest_pt"][0]
        pts_for_origin = cp[active] if active.any() else cp
        plucker = axis_dir_and_point_to_plucker(
            art["revolute_axis_dir"][0], pts_for_origin.median(dim=0).values,
        )

    rev_range = float(art["revolute_range"][0])
    pris_range = float(art["prismatic_range"][0])
    has_motion = motion_type in ("revolute", "prismatic")

    summary = {
        "sample_id": sample_id, "motion_type": motion_type,
        "n_points": int(pred_s1.shape[0]),
        "active_fraction": float(active.float().mean()),
        "predicted_revolute_range_rad": rev_range,
        "predicted_prismatic_range": pris_range,
        "frame": "dataset-normalized", "s_scale": s_scale,
        "num_queries": num_queries, "states": {},
    }

    with timed("articulate_and_write", device):
        for st in args.state:
            delta = float(st) - 1.0          # s1 is the anchor
            if has_motion and active.any():
                if motion_type == "revolute":
                    Tm = plucker_to_4x4_torch(plucker, torch.tensor(
                        rev_range * delta, device=device))
                else:
                    axis = art["prismatic_axis"][0]
                    Tm = prismatic_to_4x4_torch(
                        axis / (axis.norm() + 1e-12),
                        torch.tensor(pris_range * delta, device=device))
                moved = pred_s1.clone()
                moved[active] = pred_s1[active] @ Tm[:3, :3].T + Tm[:3, 3]
            else:
                moved = pred_s1

            tag = f"{st:.2f}".replace(".", "p")
            base = os.path.join(args.out_dir, f"{sample_id}_state{tag}")
            save_pointcloud(moved, f"{base}.ply")
            save_segmented_pointcloud(moved, seg, f"{base}_seg.ply")
            summary["states"][f"{st:.2f}"] = os.path.basename(base) + ".ply"

    with open(os.path.join(args.out_dir, f"{sample_id}_summary.json"), "w") as f:
        json.dump({**summary, "timings_s": TIMINGS}, f, indent=2)

    print(f"\nsample {sample_id}  motion={motion_type}  "
          f"points={summary['n_points']}  active={summary['active_fraction']:.3f}")
    if motion_type == "prismatic":
        print(f"predicted prismatic range: {pris_range:.4f} (normalized units)")
    elif motion_type == "revolute":
        print(f"predicted revolute range: {rev_range:.4f} rad "
              f"({rev_range * 180 / 3.14159265:.1f} deg)")
    else:
        print("sample has no motion; every state is identical")
    print(f"\nwrote {len(args.state)} state(s) to {args.out_dir}/")
    print(f"\ntiming (num_queries={num_queries}, {round(1.0/step_size)} {method} steps)")
    for k, v in TIMINGS.items():
        print(f"  {k:24} {v*1000:9.1f} ms")
    print(f"  {'-'*24} {'-'*9}")
    print(f"  {'forward_total':24} {TIMINGS['forward_total']*1000:9.1f} ms  <- inference")


if __name__ == "__main__":
    main()
