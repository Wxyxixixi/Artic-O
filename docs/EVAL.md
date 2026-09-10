# Reading the evaluation output

`eval.py` calls `ArticOTrainer.val_epoch`
(`src/trainer/artic_o_trainer.py`), which emits several `Val-*` rows.
This note explains how samples are routed into those rows, so the numbers are
interpretable without reading the trainer.

## Bucketing

Every sample is routed by two facts: its category, and whether the LARM baseline
successfully evaluated it. LARM's per-sample status is read from
`docs/larm_eval.json`, shipped with this release.

- **Geometry buckets** (`geometry`, `class:{cat}-geom`) — all non-Oven samples
  that LARM evaluated, whether LARM succeeded or failed on them.
- **Articulation buckets** (`articulation`, `class:{cat}-art`) — only samples
  LARM succeeded on. Joint metrics on samples where the baseline's joint
  estimation failed are not comparable, so they are excluded.
- **Oven** — routed solely to its own `oven` bucket. Oven is held out of
  training (`exclude_categories: [Oven]` in the config) and is the
  out-of-distribution experiment, not part of the LARM-comparable set.
- Samples LARM never evaluated are **skipped entirely** — including them would
  move the denominator out of LARM-comparable territory.

If `docs/larm_eval.json` is missing, `val_epoch` falls back to "pure-eval mode":
every sample contributes to both geometry and articulation per category. That
mode is useful for evaluating on new categories, but its numbers are **not**
the ones in the paper.

## Which row is the paper's table?

Neither `Val-Geometry` nor any single printed row. The paper reports
`val-larm-success-all`: the n=255 samples LARM succeeded on, across all
categories including Oven. `Val-Geometry` is the wider n=285 LARM-*evaluated*
set minus Oven, so its CD is slightly worse — the 49 LARM-fail samples carry
frozen predictions by design.

`scripts/summarize_benchmark.py` aggregates the n=255 bucket from
`per_sample_metrics_<timestamp>.json`, which tags every sample with its
`larm_bucket`. `scripts/verify_benchmark.sh` runs it automatically.

## Geometry — per-state CD / F1

The model predicts at state s1 (fully open). For each queried state
`t ∈ {0, 0.25, 0.5, 0.75, 1.0}`:

- **GT side** — the GT movable part is walked from s1 to `t` using the GT joint;
  context is held at s1.
- **Pred side** — if LARM succeeded on the sample and it has motion, the
  predicted movable part is walked from s1 to `t` using the *predicted* joint.
  Otherwise the prediction is **frozen at s1** for every state. This mirrors
  LARM's own "frozen prediction when joint estimation failed" rule so the two
  methods are scored under identical conditions.

CD and F1@0.05 are computed in camera frame: GT is divided by the per-sample
normalization scale, and the prediction is mapped through the chamfer-fitted
`(scale, shift)` alignment.

## Articulation — joint metrics

Computed only for samples with GT motion that landed in an articulation bucket.
Predicted and GT joint parameters are both mapped into camera frame before
scoring, so units match.

- `axis_angle@0.25` — fraction of samples whose predicted axis direction is
  within threshold.
- `axis_origin@0.15` — fraction whose predicted axis position is within
  threshold. For prismatic joints the axis has no origin, so `0.0` is injected
  to keep the combined mean comparable with LARM's table.
- `Mr@0.3` / `Md@0.3` — motion range / direction accuracy.

## Things that surprise people

- The reference state is **s1**, not s0. There is no separate s0 prediction; the
  frozen-on-fail fallback reuses the same s1 cloud at every queried state.
- At `t = 1.0` the GT transform is identity; at `t = 0.0` the movable part is
  walked back by the full GT range.
- Geometry pools LARM-success and LARM-fail samples (with the frozen-prediction
  fallback for fails); articulation is success-only. The two row families
  therefore have different denominators by design.
- Evaluation is stochastic: the ODE initialization is unseeded and GT is
  subsampled per run. Treat CD differences below ~3e-4 as noise.
