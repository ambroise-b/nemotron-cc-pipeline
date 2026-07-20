#!/bin/bash
# =============================================================================
# Stage 4 (SDG) — INDEPENDENT-ARRAY design (scales to many nodes).
#
# A Slurm array where each task is a fully self-contained SINGLE-NODE job:
#   - runs step_4-sdg.py --serve-model (local vLLM on the node's 4 GPUs, plain
#     RayClient, NO --slurm) — no cross-node Ray/Serve/interconnect at all;
#   - reads ONE shard of the bucket 18/19 files (--num-shards/--shard-id, keyed
#     off the array index) so tasks process disjoint data;
#   - writes to its own per-shard output dir (no filename collisions).
#
# Nothing is shared across tasks: a task dying kills only its shard (rerun that
# index), and it scales linearly to as many nodes as you allocate. Contrast
# run_sdg_split.sbatch (one unified Ray/Serve cluster) which is simpler to
# operate at small scale but centralizes on one head — see that file's header.
#
# Submit via one of the per-task wrappers
# scripts/submit_nemotron_cc/04A3_submit_sdg_array_<task>.sh (each sets the SDG
# TASK plus the --array range / concurrency). Direct `sbatch` needs TASK
# exported, e.g.
#   TASK=diverse_qa sbatch scripts/slurm/run_sdg_array.sbatch
# (no --array) to run a single task over the whole dataset (num_shards=1).
# =============================================================================

#SBATCH --job-name=nemotron-cc-sdg-array
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=850000
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/users/%u/repos/nemotron_cc_pipeline/logs/nemotron_cc_sdg_array_%A_%a.log
#SBATCH --error=/users/%u/repos/nemotron_cc_pipeline/logs/nemotron_cc_sdg_array_%A_%a.log

set -euo pipefail

REPO_DIR="/users/${USER}/repos/nemotron_cc_pipeline"
: "${SCRATCH:?SCRATCH not set — expected on CSCS Clariden/Alps}"

# --- Config ------------------------------------------------------------------
# Same per-node serving shape as run_sdg_split.sbatch: one TP=${TP} replica per
# node. TP must divide the model's attention-head count (30B: 32 heads -> TP=4;
# 0.5B test: 14 heads -> TP=1).
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"; TP=1    # real model; TP=1 => 4 independent replicas/node, no all-reduce (30B ~57GB fits one 96GB GPU)
#MODEL="Qwen/Qwen2.5-0.5B-Instruct"; TP=1         # fast test model
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-32}"   # per-actor in-flight cap; set per-task by the submit wrapper (fallback 32)
NUM_CPUS="${NUM_CPUS:-64}"      # Ray CPU slots = max concurrent map tasks; lower via NUM_CPUS=… to throttle splitter concurrency on OOM-prone shards
# Memory-safe preprocessing (no per-line explosion). Set STREAMING_CHUNKER=1 to
# enable; the _low submit wrappers turn it on.
CHUNKER_ARG=""; [[ -n "${STREAMING_CHUNKER:-}" ]] && CHUNKER_ARG="--streaming-chunker"
# Which SDG task to generate. Exported by the per-task submit wrapper
# (scripts/submit_nemotron_cc/04A3_submit_sdg_array_<task>.sh); "all" runs all
# four per shard in one job. Must be set — no default, so a bad submit fails
# loudly instead of silently generating the wrong task.
TASK="${TASK:?TASK not set — submit via scripts/submit_nemotron_cc/04A3_submit_sdg_array_<task>.sh, or export TASK=... before sbatch}"
DATA_DIR="${SCRATCH}/nemotron-cc-pipeline-CC-MAIN-2026-21"

CONTAINER_ENV="${CONTAINER_ENV:-${REPO_DIR}/container/container.toml}"
CONTAINER_FLAGS=(--environment="${CONTAINER_ENV}" --container-writable)
export HF_HOME="${HF_HOME:-${SCRATCH}/nemotron-cc-hf-cache}"
# Fully-offline HF: the Qwen model is pre-cached in $HF_HUB_CACHE (warm it once
# via scripts/submit_utils/00_submit_warm_hf_cache.sh — see CSCS-QUICKSTART.md
# Part 0.5). Without this, every array task hits huggingface.co at startup
# (AutoTokenizer -> model_info(), vLLM -> tree/main) and a wide fan-out blows the
# 1000-req/5-min API quota -> HTTP 429 -> shard dies. Offline reads come straight
# from the cache with zero HTTP. Set =0 to restore online. Matches run_stage.sbatch.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Shard divisor comes from the NUM_SHARDS env (exported by the submit wrapper),
# NOT SLURM_ARRAY_TASK_COUNT — the latter is the submitted range's size, so a
# partial submit (e.g. --array=0-0 for a test, or rerunning failed shards) would
# wrongly change the file->shard mapping. Keeping NUM_SHARDS fixed means shard k
# always maps to the same files regardless of which indices you submit.
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
OUT_DIR="${DATA_DIR}/sdg_output_dp/shard_$(printf '%04d' "${SHARD_ID}")"
# Per-task success marker: the pipeline writes each task to OUT_DIR/<task>/, so
# several tasks can share a shard dir. Keying _SUCCESS on TASK means finishing
# diverse_qa doesn't make a later distill submit skip the same shard.
SUCCESS_MARKER="${OUT_DIR}/_SUCCESS_${TASK}"

mkdir -p "${REPO_DIR}/logs"

echo "=================================================="
echo "  Nemotron-CC — SDG array task"
echo "  Job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} | shard ${SHARD_ID}/${NUM_SHARDS} | $(hostname)"
echo "  Model ${MODEL} (TP=${TP}) | task ${TASK} | num_cpus ${NUM_CPUS}"
echo "  Output ${OUT_DIR}"
echo "=================================================="

# Resumability: skip shards that already finished (rerun the array and only
# missing/failed shards do work). Delete a shard's _SUCCESS to force redo.
if [[ -f "${SUCCESS_MARKER}" ]]; then
    echo "Shard ${SHARD_ID} task ${TASK} already complete (${SUCCESS_MARKER} present) — skipping."
    exit 0
fi

# --- Per-node resource monitor (plain loop on the batch node — NO srun, so it
# can't clash with the step's Slingshot interconnect setup). --------------------
(
    while true; do
        gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | paste -sd" | " -)
        ram=$(free -g | awk '/Mem:/{printf "%d/%d", $3, $2}')
        top3=$(ps --no-headers -eo rss,comm --sort=-rss | head -3 | awk '{printf "%s(%.0fG) ", $2, $1/1048576}')
        printf '[MON %s shard %s] load %s | RAM %s GB | topRSS %s| GPU(util%%,used,totMiB) %s\n' \
            "$(hostname)" "${SHARD_ID}" "$(awk '{print $1}' /proc/loadavg)" "${ram}" "${top3}" "${gpu}"
        sleep 30
    done
) &
MON_PID=$!
trap 'kill "${MON_PID}" 2>/dev/null; wait 2>/dev/null' EXIT SIGTERM SIGINT

# --- Startup stagger (defense-in-depth) --------------------------------------
# Offline mode above already removes all HF API calls, so under normal operation
# this changes nothing. But if someone runs with HF_HUB_OFFLINE=0 (e.g. to pull a
# new model), a wide array all hitting the Hub in the same instant would blow the
# 1000-req/5-min quota -> 429. Spread task startups deterministically by shard id
# (reproducible, unlike $RANDOM) so a concurrency wave doesn't stampede. Capped so
# it never adds meaningful latency to the multi-hour job.
STAGGER_S=$(( SHARD_ID % 40 * 3 ))   # 0,3,6,...,117s across a 40-wide wave
if [[ "${STAGGER_S}" -gt 0 ]]; then
    echo "Staggering startup by ${STAGGER_S}s (shard ${SHARD_ID}) to avoid Hub stampede."
    sleep "${STAGGER_S}"
fi

# --- Run this shard: single-node --serve-model (no --slurm) ------------------
srun --label \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    "${CONTAINER_FLAGS[@]}" \
    bash -c "
export HF_HOME='${HF_HOME}'
# HF_HUB_CACHE is where the model actually lives (warmed capstor cache); it must
# be forwarded into the container or HF falls back to the empty HF_HOME dir and
# re-fetches from the Hub. Offline flags forwarded too. See run_stage.sbatch.
export HF_HUB_CACHE='${HF_HUB_CACHE:-}'
export HF_HUB_OFFLINE='${HF_HUB_OFFLINE:-1}'
export TRANSFORMERS_OFFLINE='${TRANSFORMERS_OFFLINE:-1}'
export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}
# Put src/lib on PYTHONPATH so Ray workers (not just the driver) can import
# sdg_chunker when --streaming-chunker deserializes the stage on a worker.
export PYTHONPATH='${REPO_DIR}/src/lib':\${PYTHONPATH:-}
# Node-local vLLM / torch.compile cache so re-enabling CUDA graphs below does NOT
# write the compile cache to the shared FS (where it hit 'Stale file handle').
export VLLM_CACHE_ROOT=/tmp/vllm_cache_\${SLURM_JOB_ID}
# --no-enforce-eager: re-enable CUDA graphs for throughput. Safe now that TP=1
# removed the all-reduce crash AND the compile cache is node-local (above). If it
# crashes at startup ('Stale file handle' / cudagraph), drop this flag to fall
# back to the known-good eager path.
python3 '${REPO_DIR}/src/nemotron-cc/step_4-sdg.py' \
    --serve-model \
    --tensor-parallel-size ${TP} \
    --model-name '${MODEL}' \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --serve-health-timeout 1800 \
    --num-cpus ${NUM_CPUS} \
    --max-concurrent-requests ${MAX_CONCURRENT_REQUESTS} \
    ${CHUNKER_ARG} \
    --no-enforce-eager \
    --num-shards ${NUM_SHARDS} --shard-id ${SHARD_ID} \
    --task ${TASK} \
    --input-dir '${DATA_DIR}/quality_labeling/bucketed_results' \
    --output-dir '${OUT_DIR}'
"

# Success marker for resumability (rerun only shards missing this).
touch "${SUCCESS_MARKER}"
echo "=================================================="
echo "  DONE shard ${SHARD_ID} task ${TASK} -> ${OUT_DIR}"
echo "=================================================="
