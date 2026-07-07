#!/bin/bash
# =============================================================================
# Submit stage 2a (exact dedup: --identify + --remove) as a multi-node SLURM job.
#
# Run from the project root:
#   bash scripts/submit_default/02a_submit_exact_dedup.sh
#
# Assumes stage 1 already ran and populated $DATA_DIR/cleaned_extracted.
# =============================================================================
set -euo pipefail

if [[ ! -f scripts/slurm/run_stage.sbatch ]]; then
    echo "ERROR: run this from the project root (scripts/slurm/run_stage.sbatch not found here)." >&2
    exit 1
fi
: "${SCRATCH:?SCRATCH not set — expected on CSCS Clariden/Alps}"

# --- SLURM resources ---------------------------------------------------------
# Every node on this cluster has 4 GPUs, 288 CPUs, ~870GB RAM (confirmed via
# `scontrol show node`) — since --gpus-per-node=4 already claims a node
# exclusively regardless, request the whole thing explicitly rather than
# getting whatever Slurm's default per-CPU memory ratio happens to grant.
ACCOUNT=infra01
PARTITION=normal
NODES=4
GPUS_PER_NODE=4
CPUS_PER_TASK=288
MEM=850000   # MB; a little under the node's 870000 to leave OS/slurmd headroom
TIME=04:00:00

# --- Stage 2a args -----------------------------------------------------------
DATA_DIR="${SCRATCH}/nemotron-cc-data"

STEP_SCRIPT="src/nemotron-cc/step_2a-exact_dedup.py"
# --num-gpus/--num-cpus intentionally omitted: their default ("all
# available") lets SlurmRayClient auto-detect the full multi-node
# allocation instead of us guessing per-node vs. cluster-total semantics.
STEP_ARGS="--identify --remove \
--input-dir ${DATA_DIR}/cleaned_extracted --cache-dir ${DATA_DIR}/cache/exact_dedup \
--output-dir ${DATA_DIR}/exact_deduplicated"

# NOTE: --remove nests a copy of --output-dir's own basename inside itself —
# actual output lands in "$DATA_DIR/exact_deduplicated/exact_deduplicated/",
# not flat under "$DATA_DIR/exact_deduplicated/" (confirmed live). Stage
# 2b's --input-dir accounts for this.
echo "Submitting stage 2a (exact dedup): ${NODES} nodes, ${GPUS_PER_NODE} GPUs/node"
STEP_SCRIPT="${STEP_SCRIPT}" STEP_ARGS="${STEP_ARGS}" \
sbatch -A "${ACCOUNT}" -p "${PARTITION}" \
    --nodes="${NODES}" --gpus-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEM}" --time="${TIME}" \
    --exclusive --no-requeue \
    scripts/slurm/run_stage.sbatch
