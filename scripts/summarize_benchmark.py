"""Aggregate a benchmark run into the bucket the paper reports.

``eval.py`` prints ``Val-Geometry`` over every sample LARM evaluated (n=285
excluding Oven, or 304 including it). The paper's headline table is a
different bucket: the n=255 subset LARM *succeeded* on. This reads the
per-sample dump and reports that subset so the two are comparable.

Usage:
    python scripts/summarize_benchmark.py work_dirs/release_check_<timestamp>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

EXPECTED = {"cd": 0.01661, "f1": 0.9578}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="evaluation run directory")
    args = ap.parse_args()

    hits = glob.glob(os.path.join(args.run_dir, "per_sample_metrics_*.json"))
    if not hits:
        raise SystemExit(f"no per_sample_metrics_*.json under {args.run_dir}")
    per_sample = json.load(open(max(hits, key=os.path.getmtime)))

    cds, f1s = [], []
    for rec in per_sample.values():
        if rec.get("larm_bucket") != "larm_success":
            continue
        states = rec.get("states") or {}
        if not states:
            continue
        cds.append(statistics.fmean(s["cd"] for s in states.values()))
        f1s.append(statistics.fmean(s["f1"] for s in states.values()))

    cd, f1 = statistics.fmean(cds), statistics.fmean(f1s)
    print(f"\nval-larm-success-all  (n={len(cds)}, expected 255)\n")
    print(f"  {'metric':10} {'this run':>10} {'paper':>10} {'delta':>10}")
    print(f"  {'CD':10} {cd:10.5f} {EXPECTED['cd']:10.5f} {cd - EXPECTED['cd']:+10.5f}")
    print(f"  {'F1@0.05':10} {f1:10.4f} {EXPECTED['f1']:10.4f} {f1 - EXPECTED['f1']:+10.4f}")
    ok = abs(cd - EXPECTED["cd"]) < 3e-4
    print(f"\n  CD within the ~3e-4 stochastic band: {'yes' if ok else 'NO'}")
    if not ok:
        print("  A larger gap means an asset or config path was retargeted wrong.")


if __name__ == "__main__":
    main()
