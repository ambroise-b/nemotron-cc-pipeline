#!/bin/bash
# =============================================================================
# Submit stage 4 (SDG: synthetic data generation via a local vLLM server) as
# a multi-node SLURM job.
#
# Run from the project root:
#   bash scripts/submit_default/04_submit_sdg.sh
#
# Assumes stage 3 already ran. UNVERIFIED end-to-end — see
# scripts/CSCS-QUICKSTART.md (vllm / --serve-model on this cluster hasn't
# been confirmed working yet). On a tiny sample, don't be surprised if
# buckets 18-19 (this stage's input) are empty or sparse.
# =============================================================================
set -euo pipefail

if [[ ! -f scripts/run_stage.sbatch ]]; then
    echo "ERROR: run this from the project root (scripts/run_stage.sbatch not found here)." >&2
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

# --- Stage 4 args ------------------------------------------------------------
DATA_DIR="${SCRATCH}/nemotron-cc-data"
TASK=diverse_qa  # one of: all, diverse_qa, distill, extract_knowledge, knowledge_list

STEP_SCRIPT="src/nemotron-cc/step_4-sdg.py"
# --serve-model defaults to True (local vLLM server); --tensor-parallel-size
# defaults to all available GPUs. See step_4-sdg.py --help for the full set
# of generation/server knobs if you need to override them.
STEP_ARGS="--task ${TASK} \
--input-dir ${DATA_DIR}/quality_labeling/bucketed_results --output-dir ${DATA_DIR}/sdg_output"

echo "Submitting stage 4 (SDG, task=${TASK}): ${NODES} nodes, ${GPUS_PER_NODE} GPUs/node"
STEP_SCRIPT="${STEP_SCRIPT}" STEP_ARGS="${STEP_ARGS}" \
sbatch -A "${ACCOUNT}" -p "${PARTITION}" \
    --nodes="${NODES}" --gpus-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEM}" --time="${TIME}" \
    --exclusive --no-requeue \
    scripts/run_stage.sbatch
