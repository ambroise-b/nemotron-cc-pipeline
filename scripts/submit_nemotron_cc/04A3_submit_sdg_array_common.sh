#!/bin/bash
# =============================================================================
# Shared submit logic for the stage 4 (SDG) INDEPENDENT-ARRAY design, one SDG
# task per array submit. Do NOT run this directly — run one of the per-task
# wrappers (04A3_submit_sdg_array_diverse_qa.sh, ..._distill.sh,
# ..._extract_knowledge.sh, ..._knowledge_list.sh). Each sets TASK and sources
# this file.
#
# The chosen task is passed to scripts/slurm/run_sdg_array.sbatch via the TASK
# env var (exported below); everything else (model, TP, paths) still lives in
# that sbatch — edit it to change resources/model.
#
# Run from the project root, e.g.:
#   bash scripts/submit_nemotron_cc/04A3_submit_sdg_array_diverse_qa.sh
# =============================================================================
set -euo pipefail

: "${TASK:?TASK not set — run a per-task wrapper (e.g. 04A3_submit_sdg_array_diverse_qa.sh), not the common file directly}"

if [[ ! -f scripts/slurm/run_sdg_array.sbatch ]]; then
    echo "ERROR: run this from the project root (scripts/slurm/run_sdg_array.sbatch not found here)." >&2
    exit 1
fi

# --- Array sizing ------------------------------------------------------------
# NUM_SHARDS   = how many disjoint pieces to split the input into (= array size;
#                each task = 1 node). Default 480 => ~40 files/shard, which was
#                CALIBRATED to keep the DocumentJoiner's per-worker buffer memory-
#                safe (~500 GB peak of 856; bigger shards, e.g. ~96 files, OOM'd).
#                Don't raise much above this without re-checking node RAM.
# MAX_CONCURRENT = how many array tasks (nodes) run at once (%N in --array).
#                Bump toward your node budget for the full run (e.g. 100).
export NUM_SHARDS="${NUM_SHARDS:-650}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"

# Ray CPU slots per node (max concurrent map tasks). Default 64; lower via
# NUM_CPUS=… to throttle splitter concurrency on OOM-prone shards.
export NUM_CPUS="${NUM_CPUS:-64}"

# Per-actor in-flight request cap, per task (higher => better GPU batching).
# Env-overridable; distill outputs are longest so it gets a lower cap.
case "${TASK}" in
    distill) _DEFAULT_MCR=128 ;;
    *)       _DEFAULT_MCR=256 ;;
esac
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-${_DEFAULT_MCR}}"

# ARRAY_RANGE = which shard indices to actually run. Defaults to all of them.
# Override to run a subset WITHOUT changing the file->shard mapping, e.g.:
#   ARRAY_RANGE=0-0            # single-shard smoke test (still 1/NUM_SHARDS of data)
#   ARRAY_RANGE=3,17,42        # rerun only these failed shards
ARRAY_RANGE="${ARRAY_RANGE:-0-$(( NUM_SHARDS - 1 ))}"

# --- SLURM reservation (optional) --------------------------------------------
#RESERVATION="SD-69241-apertus-1-5-0"
RESERVATION=""

echo "Submitting stage 4 (SDG array) task '${TASK}': NUM_SHARDS=${NUM_SHARDS}, NUM_CPUS=${NUM_CPUS}, MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS}, running indices [${ARRAY_RANGE}], up to ${MAX_CONCURRENT} nodes at once"
# NUM_SHARDS and TASK are exported so the sbatch uses them regardless of the
# submitted --array range. --job-name tags the task so squeue shows which one.
sbatch ${RESERVATION:+--reservation="${RESERVATION}"} \
    --job-name="nemotron-cc-sdg-${TASK}" \
    --export=ALL,NUM_SHARDS="${NUM_SHARDS}",TASK="${TASK}",NUM_CPUS="${NUM_CPUS}",MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS}",STREAMING_CHUNKER="${STREAMING_CHUNKER:-}" \
    --array="${ARRAY_RANGE}"%"${MAX_CONCURRENT}" \
    scripts/slurm/run_sdg_array.sbatch
