#!/bin/bash
# =============================================================================
# Submit stage 4 (SDG) as ONE Slurm job split into two disjoint node subsets:
#   - "server" nodes: a standalone multi-node vLLM deployment — one
#     self-contained TP replica per node, wired together via vLLM's own
#     --data-parallel-* flags for cross-node request routing (not Ray, and
#     not nemo_curator's InferenceServer — that one hardcodes its endpoint
#     host to "localhost", which is unreachable from any other node; see
#     scripts/CSCS-QUICKSTART.md's dependency-gotcha history)
#   - "compute" nodes: step_4-sdg.py --no-serve-model, calling the server
#     over plain HTTP
#
# Unlike submit_default/04_submit_sdg.sh (which relies on --serve-model
# spinning up a LOCAL, single-node-only server), this scales serving across
# multiple nodes, and — since it's one Slurm job doing its own node-list
# slicing — needs no cross-job service discovery (no shared file, no
# `sbatch --dependency=`): both node lists are known up front from
# $SLURM_JOB_NODELIST.
#
# Data-parallel (independent per-node replicas) rather than tensor/pipeline
# parallel across nodes: pipeline-parallel hit a confirmed-live vLLM bug on
# this model (get_layers_from_vllm_config KeyError — see the comment above
# the server srun loop below), and a single TP group spanning all nodes
# isn't valid either (32 attention heads don't divide across e.g. TP=12).
#
# This lives in scripts/ rather than scripts/submit_default/ because it IS
# the sbatch script (its own #SBATCH directives, its own srun/vllm calls)
# rather than a thin wrapper that submits scripts/run_stage.sbatch, unlike
# everything under submit_default/.
#
# Run from the project root:
#   sbatch scripts/run_sdg_split.sbatch
#
# All the values you're likely to want to change live in the block below —
# edit those, not the srun/ray/vllm calls themselves.
# =============================================================================

#SBATCH --job-name=nemotron-cc-sdg-split
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=850000
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/users/aborbely/repos/nemotron_cc_pipeline/logs/nemotron_cc_sdg_split_%j.log
#SBATCH --error=/users/aborbely/repos/nemotron_cc_pipeline/logs/nemotron_cc_sdg_split_%j.log

set -euo pipefail

REPO_DIR="/users/aborbely/repos/nemotron_cc_pipeline"
: "${SCRATCH:?SCRATCH not set — expected on CSCS Clariden/Alps}"

# --- Node split --------------------------------------------------------------
# NUM_SERVER_NODES + NUM_COMPUTE_NODES must equal the #SBATCH --nodes above
# (4 by default) — if you change either, update both.
# NUM_COMPUTE_NODES > 1 needs one more change below: add --slurm to the
# step_4-sdg.py call so it forms its own SlurmRayClient cluster across that
# subset (a second, independent Ray cluster from the server's — same
# reasoning as the port/connection-file discussion for this design).
NUM_SERVER_NODES=3
NUM_COMPUTE_NODES=1

# --- Model / vLLM config ------------------------------------------------------
# Data-parallel ACROSS nodes (one independent full replica per node, each
# purely local — no cross-node model-parallel comms at all) instead of
# pipeline-parallel: vLLM's PP has a confirmed-live bug on this model/version
# (get_layers_from_vllm_config KeyError on layers owned by another PP rank,
# hit both during cudagraph/compile setup AND during KV-cache attn-backend
# setup — --enforce-eager only dodges the first, not the second). TP=12 in
# one group was tried too but rejected: 32 attention heads isn't divisible
# by 12 (vLLM requires exact division), so TP must be a divisor of 32 that
# also fits GPUS_PER_NODE — 4 fits perfectly (one full TP group per node).
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"   # matches step_4-sdg.py's own default
PORT=8000
DP_RPC_PORT=13345
GPUS_PER_NODE=4
TP="${GPUS_PER_NODE}"          # tensor-parallel WITHIN a node (fast NVLink)
DP="${NUM_SERVER_NODES}"       # data-parallel ACROSS nodes (one full replica/node)

MAX_MODEL_LEN=8192              # matches step_4-sdg.py's own default
GPU_MEMORY_UTILIZATION=0.90     # matches step_4-sdg.py's own default

# --- Container (same one every other stage uses) ------------------------------
CONTAINER_ENV="${CONTAINER_ENV:-${REPO_DIR}/container/container.toml}"
CONTAINER_FLAGS=(--environment="${CONTAINER_ENV}" --container-writable)

# HF model weights — cache on $SCRATCH so a second run doesn't re-download
# (same convention as run_stage.sbatch).
export HF_HOME="${HF_HOME:-${SCRATCH}/nemotron-cc-hf-cache}"

# --- Stage 4 args --------------------------------------------------------------
DATA_DIR="${SCRATCH}/nemotron-cc-data"
# One or more of: diverse_qa, distill, extract_knowledge, knowledge_list.
# Since the server stays up for the whole job, running several tasks here
# costs one model load total instead of one per task (the whole point of
# this split design) — add/remove entries freely.
TASKS=(diverse_qa)

# ------------------------------------------------------------------------------
# Nothing below this line should need editing for a normal run.
# ------------------------------------------------------------------------------

ALL_NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
if [[ $(( NUM_SERVER_NODES + NUM_COMPUTE_NODES )) -ne ${#ALL_NODES[@]} ]]; then
    echo "ERROR: NUM_SERVER_NODES (${NUM_SERVER_NODES}) + NUM_COMPUTE_NODES (${NUM_COMPUTE_NODES}) != allocated nodes (${#ALL_NODES[@]})." >&2
    echo "        Update #SBATCH --nodes and/or the NUM_*_NODES variables above." >&2
    exit 1
fi
SERVER_NODES=("${ALL_NODES[@]:0:${NUM_SERVER_NODES}}")
COMPUTE_NODES=("${ALL_NODES[@]:${NUM_SERVER_NODES}:${NUM_COMPUTE_NODES}}")
HEAD_NODE="${SERVER_NODES[0]}"
HEAD_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}')

echo "=================================================="
echo "  Nemotron-CC — SDG split-node run"
echo "=================================================="
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Server nodes (${NUM_SERVER_NODES}): ${SERVER_NODES[*]}"
echo "  Compute nodes (${NUM_COMPUTE_NODES}): ${COMPUTE_NODES[*]}"
echo "  Model        : ${MODEL}  (TP=${TP}, DP=${DP})"
echo "  Tasks        : ${TASKS[*]}"
echo "=================================================="

mkdir -p "${REPO_DIR}/logs"

# --- Server: one DIFFERENTIATED vllm serve per node, no Ray -------------------
# Each node runs a fully self-contained TP=${TP} replica over its own local
# GPUs (default mp executor — no cross-node executor needed since nothing
# model-parallel crosses a node boundary). vLLM's --data-parallel-* flags
# only wire up lightweight cross-node request routing between replicas:
#   - rank 0 (head node): runs the API server other nodes' vllm CAN'T (no
#     --headless), listens on 0.0.0.0:${PORT}.
#   - ranks 1..N-1: --headless, engine-core-only, no API server of their own.
# Tried first: identical `--data-parallel-size ${DP}` command broadcast to
# all nodes via Ray (mirroring the old PP approach) — that made vLLM's own
# launcher assume a single-host multi-GPU layout and slice
# CUDA_VISIBLE_DEVICES into a global range, which broke because each node
# only ever sees its own 4 local GPU indices (confirmed live: "Error setting
# CUDA_VISIBLE_DEVICES: local range: [4, 8) base value: 0,1,2,3"). Explicit
# per-rank flags below sidestep that entirely.
SERVER_PIDS=()
for i in "${!SERVER_NODES[@]}"; do
    NODE="${SERVER_NODES[$i]}"
    # Headless/internal-LB DP: rank 0 (head) runs the sole API server and
    # load-balances across all engines; remote ranks run --headless and
    # declare their position via --data-parallel-start-rank. NB: do NOT set
    # --data-parallel-rank here — that flag switches vLLM into external-LB
    # mode, which forbids --headless ("Remote engine N must not use
    # --headless in external ... dp lb mode" — confirmed live).
    if [[ "$i" -eq 0 ]]; then
        RANK_FLAGS="--host 0.0.0.0 --port ${PORT}"
    else
        RANK_FLAGS="--headless --data-parallel-start-rank ${i}"
    fi
    srun --label \
        --nodes=1 --ntasks=1 --ntasks-per-node=1 \
        -w "${NODE}" \
        "${CONTAINER_FLAGS[@]}" \
        bash -c "
export TRITON_CACHE_DIR=/tmp/triton-cache-${SLURM_JOB_ID}-\$(hostname)
mkdir -p \"\${TRITON_CACHE_DIR}\"
vllm serve ${MODEL} \
    --tensor-parallel-size ${TP} \
    --data-parallel-size ${DP} \
    --data-parallel-size-local 1 \
    --data-parallel-address ${HEAD_IP} \
    --data-parallel-rpc-port ${DP_RPC_PORT} \
    --enable-expert-parallel \
    ${RANK_FLAGS} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-seqs 256 \
    --disable-log-requests
" &
    SERVER_PIDS+=($!)
done

#personnal note : I removed the --enforce-eager this has still to be tested

# Guarantee every per-node server process is torn down on normal exit, error,
# or the SIGTERM from --signal above (fired 60s before the hard kill at
# --time, giving this trap time to actually run).
trap 'kill "${SERVER_PIDS[@]}" 2>/dev/null; wait 2>/dev/null' EXIT SIGTERM SIGINT

echo "Waiting for vLLM on ${HEAD_NODE}:${PORT}..."
until curl -sf "http://${HEAD_NODE}:${PORT}/v1/models" >/dev/null 2>&1; do
    STILL_ALIVE=0
    for PID in "${SERVER_PIDS[@]}"; do
        kill -0 "$PID" 2>/dev/null && STILL_ALIVE=1
    done
    if [[ "$STILL_ALIVE" -eq 0 ]]; then
        echo "ERROR: all server processes (srun/vllm) exited before becoming ready — see the '0:' lines above for the actual vllm traceback (each per-node srun is its own single-task step, so all are labeled 0:; the APIServer pid / node ip in each line tells them apart)." >&2
        exit 1
    fi
    sleep 5
done
echo "vLLM ready at http://${HEAD_NODE}:${PORT}/v1"

# --- Compute: run each task in turn against the one already-loaded server ---
for TASK in "${TASKS[@]}"; do
    echo "--- stage 4 task: ${TASK} ---"
    srun --label \
        --nodes="${NUM_COMPUTE_NODES}" --ntasks="${NUM_COMPUTE_NODES}" --ntasks-per-node=1 \
        -w "$(IFS=,; echo "${COMPUTE_NODES[*]}")" \
        "${CONTAINER_FLAGS[@]}" \
        bash -c "
echo \"[COMPUTE] running SDG task '${TASK}' on host \$(hostname) (\$(hostname -i)) -> server http://${HEAD_NODE}:${PORT}/v1\"
export HF_HOME='${HF_HOME}'
python3 '${REPO_DIR}/src/nemotron-cc/step_4-sdg.py' \
    --no-serve-model --base-url 'http://${HEAD_NODE}:${PORT}/v1' \
    --api-key 'EMPTY' \
    --model-name '${MODEL}' \
    --task ${TASK} \
    --input-dir '${DATA_DIR}/quality_labeling/bucketed_results' \
    --output-dir '${DATA_DIR}/sdg_output'
"
done

echo "=================================================="
echo "  DONE"
echo "=================================================="
# Script reaches here -> trap fires -> server torn down -> job ends
