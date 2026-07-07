#!/bin/bash
# =============================================================================
# Submit stage 2c (substring dedup) as a SLURM job.
#
# Run from the project root:
#   bash scripts/submit_default/02c_submit_substring_dedup.sh
#
# Assumes stage 2b already ran. Unlike the other stages, this is a plain
# shell/CPU pipeline with NO multi-node or Ray support (no --slurm flag),
# and it takes its paths via env vars, not CLI args.
#
# IMPORTANT: even though NODES=4 is requested below (for consistency with
# the other submit scripts), scripts/slurm/run_stage.sbatch detects the ".sh"
# step and deliberately runs it on only ONE of the 4 allocated nodes — with
# this script's normal one-task-per-node model, all 4 nodes would
# concurrently rm-rf/rebuild the same shared $MAIN_CACHE_PATH/$OUTPUT_PATH
# and corrupt each other's work. The other 3 nodes just sit idle for this
# step; that's expected, not a bug.
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
# (The single node this step actually runs on — see comment above — still
# gets the full --cpus-per-task/--mem below; the other 3 just sit idle.)
ACCOUNT=infra01
PARTITION=normal
NODES=4
GPUS_PER_NODE=4
CPUS_PER_TASK=288
MEM=850000   # MB; a little under the node's 870000 to leave OS/slurmd headroom
TIME=04:00:00

# --- Stage 2c args (env vars, not CLI args) ----------------------------------
DATA_DIR="${SCRATCH}/nemotron-cc-data"

STEP_SCRIPT="src/nemotron-cc/step_2c-substring_dedup/exact_substring_dedup.sh"
STEP_WORKDIR="src/nemotron-cc/step_2c-substring_dedup"
STEP_ARGS=""  # this stage takes no CLI args — see INPUT_PATH/etc below

# INPUT_PATH points at stage 2b's nested output dir (see its script's NOTE).
export INPUT_PATH="${DATA_DIR}/fuzzy_deduplicated/fuzzy_deduplicated"
export MAIN_CACHE_PATH="${DATA_DIR}/cache/substring_dedup"
export OUTPUT_PATH="${DATA_DIR}/substring_deduplicated"
# Matches CPUS_PER_TASK above — the script defaults to 128 (its upstream
# example value) unless overridden.
export NUM_THREADS="${CPUS_PER_TASK}"

echo "Submitting stage 2c (substring dedup): ${NODES} nodes requested, runs on 1 (see comment above)"
STEP_SCRIPT="${STEP_SCRIPT}" STEP_WORKDIR="${STEP_WORKDIR}" STEP_ARGS="${STEP_ARGS}" \
INPUT_PATH="${INPUT_PATH}" MAIN_CACHE_PATH="${MAIN_CACHE_PATH}" OUTPUT_PATH="${OUTPUT_PATH}" \
NUM_THREADS="${NUM_THREADS}" \
sbatch -A "${ACCOUNT}" -p "${PARTITION}" \
    --nodes="${NODES}" --gpus-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEM}" --time="${TIME}" \
    --exclusive --no-requeue \
    scripts/slurm/run_stage.sbatch
