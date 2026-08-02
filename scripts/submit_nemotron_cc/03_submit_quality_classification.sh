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
NODES=20
GPUS_PER_NODE=4
CPUS_PER_TASK=288
MEM=850000   # MB; a little under the node's 870000 to leave OS/slurmd headroom
TIME=08:00:00

# --- GPU classifier stage sizing ---------------------------------------------
# The two edu classifiers each claim a whole GPU per actor. Left to Xenna's
# autoscaler they drift (40/40 <-> 41/39) and, with every GPU already taken, a
# reallocated actor lands on an occupied device and OOMs — that killed job
# 2965494 at 54%. So pin both stages to a fixed count computed from the
# allocation. GPU_RESERVE leaves a couple of devices free for restarts; set it
# to 0 to use every GPU.
GPU_RESERVE=2
CLASSIFIER_NUM_WORKERS=$(( (NODES * GPUS_PER_NODE - GPU_RESERVE) / 2 ))

# --- SLURM reservation (optional) --------------------------------------------
# When RESERVATION is non-empty, --reservation=<RESERVATION> is appended to the
# sbatch call below; when it's "", the flag is omitted entirely and Slurm
# schedules normally. Comment/uncomment to switch pools or disable it quickly.
#RESERVATION="SD-69241-apertus-1-5-0"
#RESERVATION=""

# --- Stage 3 args ------------------------------------------------------------
#DATA_DIR="${SCRATCH}/nemotron-cc-data"
DATA_DIR="${SCRATCH}/nemotron-cc-pipeline-CC-MAIN-2019-04"

#TODO change back to the original input dir

STEP_SCRIPT="src/nemotron-cc/step_3-quality_classification.py"
# --num-gpus/--num-cpus intentionally omitted — see 02a's comment.
# here we also skip substring dedup : input dir would be ${DATA_DIR}/substring_deduplicated otherwise
#
# TEMP: reading the RESHARDED fuzzy-dedup output (files <=480MB) instead of the
# raw fuzzy_deduplicated dir, whose 2.5GB files OOM'd the classifier actors.
# Revert to the commented line once resharding moves upstream (after the URL/PII
# filter). Run 01c_submit_reshard.sh first to produce this directory.
# ORIGINAL: --input-dir ${DATA_DIR}/fuzzy_deduplicated/fuzzy_deduplicated/
STEP_ARGS="--classify --ensemble \
--input-dir ${DATA_DIR}/substring_dedup/substring_dedup --output-dir ${DATA_DIR}/quality_labeling \
--classifier-num-workers ${CLASSIFIER_NUM_WORKERS}"

# Stage 3 only: 26.07 image, for with_(num_workers=...). Other stages use container.toml.
CONTAINER_ENV="$(pwd)/container/container_new.toml"

echo "Submitting stage 3 (quality classification): ${NODES} nodes, ${GPUS_PER_NODE} GPUs/node"
echo "  GPU classifier stages pinned to ${CLASSIFIER_NUM_WORKERS} workers each (of $(( NODES * GPUS_PER_NODE )) GPUs)"
echo "  Container: ${CONTAINER_ENV}"
STEP_SCRIPT="${STEP_SCRIPT}" STEP_ARGS="${STEP_ARGS}" CONTAINER_ENV="${CONTAINER_ENV}" \
sbatch -A "${ACCOUNT}" -p "${PARTITION}" \
    --nodes="${NODES}" --gpus-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEM}" --time="${TIME}" \
    --exclusive --no-requeue \
    ${RESERVATION:+--reservation="${RESERVATION}"} \
    scripts/slurm/run_stage.sbatch
