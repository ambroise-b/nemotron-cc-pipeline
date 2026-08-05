#!/bin/bash
# =============================================================================
# Submit stage 2c (substring dedup) as a SLURM ARRAY: shard the input dataset
# into N directories of symlinks (round-robin by file count — see
# src/lib/shard_symlinks.py, no data is copied), then run
# scripts/slurm/run_substring_dedup_array.sbatch, one array task per shard.
#
# Contrast 02c_submit_substring_dedup.sh (single job over the whole dataset):
# this one scales to many nodes at once and lets a failed shard be retried in
# isolation, at the cost of each shard's exact_substring_dedup.sh redundantly
# cloning+building its own Rust binary copy (see that sbatch's header).
#
# Two top-level trees under ${SHARD_ROOT} (see run_substring_dedup_array.sbatch):
#   sharding/shard_XXXX/         — ONLY the file-mapping symlinks, written by
#                                   shard_symlinks.py below.
#   substring_dedup/shard_XXXX/  — that shard's cache/output/workdir, written
#                                   by the array job (cache and workdir are
#                                   deleted once the shard succeeds).
# A full restart from scratch is `rm -rf ${SHARD_ROOT}`; to redo just one
# shard's run without re-sharding, `rm -rf ${SHARD_ROOT}/substring_dedup/shard_0003`.
# Resharding here only ever touches sharding/ — it's safe to re-run this
# script on its own without disturbing already-completed shards; the array's
# _SUCCESS marker (see that sbatch) then skips redoing them. Caveat: if
# NUM_SHARDS or the upstream input files change between runs, shard IDs remap
# to different files, but a stale _SUCCESS from the old mapping would still
# skip that shard — do the manual rm -rf above first if you change either of
# those. Also don't resubmit while a previous array job on the same
# SHARD_ROOT is still running — nothing stops two jobs racing on the same
# shard directory concurrently.
#
# Run from the project root:
#   bash scripts/submit_nemotron_cc/02cA2_submit_substring_dedup_array.sh
# =============================================================================
set -euo pipefail

if [[ ! -f scripts/slurm/run_substring_dedup_array.sbatch ]]; then
    echo "ERROR: run this from the project root (scripts/slurm/run_substring_dedup_array.sbatch not found here)." >&2
    exit 1
fi
: "${SCRATCH:?SCRATCH not set — expected on CSCS Clariden/Alps}"

REPO_DIR="${REPO_DIR:-/users/${USER}/repos/nemotron_cc_pipeline}"
DATA_DIR="${SCRATCH}/nemotron-cc-pipeline-CC-MAIN-2025-21"

# --- SLURM resources ----------------------------------------------------------
ACCOUNT=infra01
PARTITION=normal
CPUS_PER_TASK=288
MEM=850000   # MB; a little under the node's 870000 to leave OS/slurmd headroom
GPUS_PER_NODE=4
TIME=12:00:00
MAXPAR="${MAXPAR:-30}" # max array tasks (= nodes) running at once 

# --- SLURM reservation (optional) --------------------------------------------
#RESERVATION="SD-69241-apertus-1-5-0"
RESERVATION=""

# --- What to shard -------------------------------------------------------------
# Matches 02c_submit_substring_dedup.sh's INPUT_PATH (stage 2b's nested output
# dir — see its script's NOTE).
INPUT_PATH="${DATA_DIR}/fuzzy_deduplicated/fuzzy_deduplicated"
SHARD_ROOT="${DATA_DIR}/substring_dedup"
NUM_SHARDS="${NUM_SHARDS:-75}" # we try to have shard of ~58GB MAX

# Which array indices to actually submit — defaults to all of them
# (0-(NUM_SHARDS-1)), capped at MAXPAR concurrent. Override to smoke-test on
# a couple of shards without changing the sharding itself, e.g.:
#   NUM_SHARDS=50 ARRAY_RANGE=0-1 bash scripts/submit_nemotron_cc/02cA2_submit_substring_dedup_array.sh
# still shards the full dataset into 50 pieces but only submits shard_0000
# and shard_0001; the rest sit unprocessed under SHARD_ROOT until resubmitted
# with a wider range (or no ARRAY_RANGE override at all).
ARRAY_RANGE="${ARRAY_RANGE:-0-$(( NUM_SHARDS - 1 ))}"

echo "Sharding ${INPUT_PATH} into ${NUM_SHARDS} shard(s) of symlinks under ${SHARD_ROOT}/sharding..."
python3 "${REPO_DIR}/src/lib/shard_symlinks.py" \
    --input-path "${INPUT_PATH}" \
    --output-path "${SHARD_ROOT}/sharding" \
    --num-shards "${NUM_SHARDS}"

export SHARD_ROOT
export DATA_DIR
export NUM_THREADS="${CPUS_PER_TASK}"

echo "Submitting substring dedup array: indices ${ARRAY_RANGE} (of ${NUM_SHARDS} shard(s) total), up to ${MAXPAR} concurrent"
sbatch -A "${ACCOUNT}" -p "${PARTITION}" \
    --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEM}" --gpus-per-node="${GPUS_PER_NODE}" \
    --time="${TIME}" --exclusive --no-requeue \
    --array="${ARRAY_RANGE}%${MAXPAR}" \
    ${RESERVATION:+--reservation="${RESERVATION}"} \
    scripts/slurm/run_substring_dedup_array.sbatch
