#!/bin/bash
# =============================================================================
# Stage 4 (SDG) — UNIFIED multi-node serving via Ray Serve.
#
# One Ray cluster across ALL allocated nodes (SlurmRayClient): node 0 is the
# head/driver, the rest join as workers. step_4-sdg.py --serve-model deploys
# vLLM through Ray Serve, which places one TP=${TP} replica per node across the
# cluster's GPUs and load-balances requests to them. Ray Serve runs an HTTP
# proxy on every node, so the pipeline's LLM actors just call localhost and Ray
# routes onward. The data pipeline runs on the same cluster.
#
# Replaces the old split design (dedicated vLLM server nodes + separate compute
# nodes over HTTP), which relied on vLLM cross-node data-parallel — that kept
# collapsing here (gloo _synchronize_dp_ranks). Ray Serve routes instead, so
# there is NO cross-node model-parallel collective, and SlurmRayClient owns the
# whole allocation (its designed use), unlike the split design's compute subset.
#
# Run from the project root:
#   sbatch scripts/slurm/run_sdg_split.sbatch
# =============================================================================

#SBATCH --job-name=nemotron-cc-sdg
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=850000
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/users/%u/repos/nemotron_cc_pipeline/logs/nemotron_cc_sdg_split_%j.log
#SBATCH --error=/users/%u/repos/nemotron_cc_pipeline/logs/nemotron_cc_sdg_split_%j.log

set -euo pipefail

REPO_DIR="/users/${USER}/repos/nemotron_cc_pipeline"
: "${SCRATCH:?SCRATCH not set — expected on CSCS Clariden/Alps}"

# --- Config ------------------------------------------------------------------
# Pick a model AND a matching TP: TP must DIVIDE the model's attention-head
# count. Qwen3-30B-A3B has 32 heads -> TP=4 (one replica per node). Qwen2.5-0.5B
# has 14 heads -> TP=1 (14 % 4 != 0, so TP=4 fails vLLM's head-divisibility check).
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"; TP=4    # real model (~10min load; --serve-health-timeout below covers it)
#MODEL="Qwen/Qwen2.5-0.5B-Instruct"; TP=1         # fast test model (~1GB, loads in seconds)
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90
TASKS=(diverse_qa)                # one model load serves every task in this list
DATA_DIR="${SCRATCH}/nemotron-cc-pipeline-CC-MAIN-2026-21"

CONTAINER_ENV="${CONTAINER_ENV:-${REPO_DIR}/container/container.toml}"
CONTAINER_FLAGS=(--environment="${CONTAINER_ENV}" --container-writable)
export HF_HOME="${HF_HOME:-${SCRATCH}/nemotron-cc-hf-cache}"

mkdir -p "${REPO_DIR}/logs"

echo "=================================================="
echo "  Nemotron-CC — SDG (unified Ray Serve)"
echo "  Job ${SLURM_JOB_ID} | ${SLURM_JOB_NUM_NODES} node(s) | ${MODEL} (TP=${TP})"
echo "  Tasks: ${TASKS[*]}"
echo "=================================================="

# NOTE: no in-job resource monitor here. A background `srun --overlap` clashes
# with the main step on CSCS's Slingshot interconnect ("Error configuring
# interconnect"), which blocks the main task on the node the monitor grabbed and
# starves the Ray cluster (N-1/N workers -> timeout). To watch resources, run
# the monitor in a SEPARATE terminal AFTER the job's steps are up (see the
# one-liner in the repo docs / below the sbatch invocation notes).

# --- Run each task across ALL nodes ------------------------------------------
# One task per node; SlurmRayClient makes node 0 the head (runs the pipeline +
# Ray Serve deploy) and the rest Ray workers. --serve-model + --tensor-parallel-
# size ${TP} => one vLLM replica per node, Ray Serve load-balances requests.
for TASK in "${TASKS[@]}"; do
    echo "--- stage 4 task: ${TASK} ---"
    srun --label \
        --nodes="${SLURM_JOB_NUM_NODES}" --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
        "${CONTAINER_FLAGS[@]}" \
        bash -c "
export HF_HOME='${HF_HOME}'
# SlurmRayClient rendezvous file must be on the shared FS; ray tmp node-local.
export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}
export RAY_PORT_BROADCAST_DIR='${REPO_DIR}/logs'
# Conservative Ray data-side CPU count. These nodes ALSO host the vLLM replicas,
# and the giant input docs (100k+ tokens) make preprocessing memory-heavy — a
# high actor count OOM-killed a worker. Keep it modest to bound peak memory and
# leave headroom for the co-located replicas.
NUM_CPUS=64
echo \"[\$(hostname)] NODEID=\${SLURM_NODEID} num_cpus=\${NUM_CPUS}\"
python3 '${REPO_DIR}/src/nemotron-cc/step_4-sdg.py' \
    --serve-model \
    --slurm \
    --tensor-parallel-size ${TP} \
    --model-name '${MODEL}' \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --serve-health-timeout 1800 \
    --num-cpus \${NUM_CPUS} \
    --task ${TASK} \
    --input-dir '${DATA_DIR}/quality_labeling/bucketed_results' \
    --output-dir '${DATA_DIR}/sdg_output'
"
done

echo "=================================================="
echo "  DONE"
echo "=================================================="
