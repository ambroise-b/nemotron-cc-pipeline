#!/bin/bash
# =============================================================================
# Submit stage 2b (fuzzy dedup: --identify + --remove) as a multi-node SLURM job.
#
# Run from the project root:
#   bash scripts/submit_default/02b_submit_fuzzy_dedup.sh
#
# Assumes stage 2a already ran.
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
NODES=20
GPUS_PER_NODE=4
CPUS_PER_TASK=288
MEM=850000   # MB; a little under the node's 870000 to leave OS/slurmd headroom
TIME=04:00:00

# --- SLURM reservation (optional) --------------------------------------------
# When RESERVATION is non-empty, --reservation=<RESERVATION> is appended to the
# sbatch call below; when it's "", the flag is omitted entirely and Slurm
# schedules normally. Comment/uncomment to switch pools or disable it quickly.
#RESERVATION="SD-69241-apertus-1-5-0"
RESERVATION=""

# --- Stage 2b args -----------------------------------------------------------
#DATA_DIR="${SCRATCH}/nemotron-cc-data"
DATA_DIR="${SCRATCH}/nemotron-cc-pipeline-CC-MAIN-2017-13"

STEP_SCRIPT="src/nemotron-cc/step_2b-fuzzy_dedup.py"
# --num-gpus/--num-cpus intentionally omitted — see 02a's comment.
# --input-dir points at stage 2a's nested output dir (see its own script's
# NOTE for why).
STEP_ARGS="--identify --remove \
--input-dir ${DATA_DIR}/exact_deduplicated/exact_deduplicated --cache-dir ${DATA_DIR}/cache/fuzzy_dedup \
--output-dir ${DATA_DIR}/fuzzy_deduplicated"

# NOTE: same output-nesting caveat as 2a applies here — real data lands in
# "$DATA_DIR/fuzzy_deduplicated/fuzzy_deduplicated/". Stage 2c's INPUT_PATH
# accounts for this.
echo "Submitting stage 2b (fuzzy dedup): ${NODES} nodes, ${GPUS_PER_NODE} GPUs/node"
STEP_SCRIPT="${STEP_SCRIPT}" STEP_ARGS="${STEP_ARGS}" \
sbatch -A "${ACCOUNT}" -p "${PARTITION}" \
    --nodes="${NODES}" --gpus-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEM}" --time="${TIME}" \
    --exclusive --no-requeue \
    ${RESERVATION:+--reservation="${RESERVATION}"} \
    scripts/slurm/run_stage.sbatch
