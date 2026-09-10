#!/usr/bin/env bash
# Reproduce the paper's benchmark table.
#
# Requires (see data/README.md and checkpoints/README.md):
#   ./checkpoints/artic_o_s0_36.pth
#   ./data/artic_o_test/            HF dataset split
#   ./data/val_views.json
#   ./data/larm_train_test_list.json
#
# Results land in work_dirs/release_check_<timestamp>/.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${CKPT:-./checkpoints/artic_o_s0_36.pth}
CONFIG=${CONFIG:-./src/configs/artic_o.yaml}
NPROC=${NPROC:-4}
WORK_DIR=${WORK_DIR:-./work_dirs}
RUN_NAME=${RUN_NAME:-release_check}

[ -e "$CKPT" ] || { echo "missing: $CKPT  (see checkpoints/README.md)"; exit 1; }
for p in ./data/artic_o_test ./data/val_views.json ./data/larm_train_test_list.json; do
    [ -e "$p" ] || { echo "missing: $p  (see data/README.md)"; exit 1; }
done

torchrun --nproc_per_node="$NPROC" eval.py \
    --config "$CONFIG" \
    --ckpt-path "$CKPT" \
    --work-dir "$WORK_DIR" \
    --run-name "$RUN_NAME"

# eval.py prints Val-Geometry over every LARM-evaluated sample (n=285 without
# Oven). The paper's headline table is the n=255 LARM-success subset, so
# summarize that from the per-sample dump rather than eyeballing the wrong row.
RUN_DIR=$(ls -1d "$WORK_DIR"/"$RUN_NAME"_* 2>/dev/null | sort | tail -1)
if [ -n "$RUN_DIR" ]; then
    python scripts/summarize_benchmark.py "$RUN_DIR"
    echo
    echo "Full log and per-sample metrics: $RUN_DIR"
else
    echo "warning: could not locate the run directory under $WORK_DIR" >&2
fi

cat <<'EOF'

------------------------------------------------------------------
Articulation metrics are printed above as Val-Articulation/joint/*.
Compare the `success_with_joint` column (denominator = the 248 of the
255 LARM-success samples that have motion):

  axis_angle@0.25    0.9490
  axis_origin@0.15   0.9804
  Mr@0.3             0.9216

Evaluation is stochastic (the ODE x_init is unseeded and GT is
subsampled), so ~1pp of run-to-run movement on the joint rates and
below ~3e-4 on CD is noise.
------------------------------------------------------------------
EOF
