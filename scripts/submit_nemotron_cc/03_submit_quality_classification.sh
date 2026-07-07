#!/bin/bash
# =============================================================================
# Submit stage 3 (quality classification: --classify + --ensemble) as a
# multi-node SLURM job.
#
# Run from the project root:
#   bash scripts/submit_default/03_submit_quality_classification.sh
#
# Assumes stage 2c already ran.
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

# --- Stage 3 args ------------------------------------------------------------
DATA_DIR="${SCRATCH}/nemotron-cc-data"

STEP_SCRIPT="src/nemotron-cc/step_3-quality_classification.py"
# --num-gpus/--num-cpus intentionally omitted — see 02a's comment.
STEP_ARGS="--classify --ensemble \
--input-dir ${DATA_DIR}/substring_deduplicated --output-dir ${DATA_DIR}/quality_labeling"

echo "Submitting stage 3 (quality classification): ${NODES} nodes, ${GPUS_PER_NODE} GPUs/node"
STEP_SCRIPT="${STEP_SCRIPT}" STEP_ARGS="${STEP_ARGS}" \
sbatch -A "${ACCOUNT}" -p "${PARTITION}" \
    --nodes="${NODES}" --gpus-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEM}" --time="${TIME}" \
    --exclusive --no-requeue \
    scripts/slurm/run_stage.sbatch
