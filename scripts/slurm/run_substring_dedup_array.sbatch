#!/bin/bash
# =============================================================================
# Stage 2c (substring dedup) as a SLURM array: one array task per pre-built
# shard, each running exact_substring_dedup.sh entirely against its own shard
# directory (own INPUT_PATH/MAIN_CACHE_PATH/OUTPUT_PATH) — no cross-task
# sharing, so tasks scale to as many nodes as allocated and a failed shard
# can be rerun on its own (see resumability marker below).
#
# Shards are directories of SYMLINKS to the original files (built by
# src/lib/shard_symlinks.py — see the submit script), not copies, so
# sharding itself costs no extra disk.
#
# Two top-level trees under ${SHARD_ROOT}: sharding/shard_XXXX/ holds ONLY
# the file-mapping symlinks (owned by shard_symlinks.py — this script never
# writes there), and substring_dedup/shard_XXXX/ holds this task's own
# cache/output/workdir. Restarting one shard's run from scratch is
# `rm -rf ${SHARD_ROOT}/substring_dedup/shard_XXXX` (leaves the file mapping
# untouched); a full restart is `rm -rf ${SHARD_ROOT}`.
#
# exact_substring_dedup.sh always clones+builds the Rust binary into a fixed
# relative "deduplicate-text-datasets" directory next to itself — running it
# unmodified from several array tasks at once would have them race on that
# same directory. So each task instead runs from its OWN copy of the driver
# scripts (copied fresh below, NOT the deduplicate-text-datasets build
# output), which gives it an isolated place to clone+build. This costs a
# redundant clone+cargo-build per shard (same as any single run of the
# script) rather than a shared one, in exchange for not touching
# exact_substring_dedup.sh itself.
#
# Not submitted directly — use
# scripts/submit_nemotron_cc/02cA2_submit_substring_dedup_array.sh, which
# shards the dataset and submits this with the right --array range.
# Required env: SHARD_ROOT, DATA_DIR.
# Optional env: NUM_THREADS (defaults to cpus-per-task below).
# =============================================================================

#SBATCH --job-name=nemotron-cc-substring-dedup-array
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=850000
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=12:00:00
#SBATCH --output=/users/%u/repos/nemotron_cc_pipeline/logs/substring_dedup_array_%A_%a.log
#SBATCH --error=/users/%u/repos/nemotron_cc_pipeline/logs/substring_dedup_array_%A_%a.log

set -euo pipefail

: "${SHARD_ROOT:?set SHARD_ROOT (parent dir containing the sharding/ and substring_dedup/ trees)}"

REPO_DIR="${REPO_DIR:-/users/${USER}/repos/nemotron_cc_pipeline}"
STEP_SRC_DIR="${REPO_DIR}/src/nemotron-cc/step_2c-substring_dedup"
CONTAINER_ENV="${CONTAINER_ENV:-${REPO_DIR}/container/container.toml}"
CONTAINER_FLAGS=(--environment="${CONTAINER_ENV}" --container-writable)

SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_TAG="shard_$(printf '%04d' "${SHARD_ID}")"
DATA_SUBDIR="${SHARD_ROOT}/substring_dedup/${SHARD_TAG}"

export INPUT_PATH="${SHARD_ROOT}/sharding/${SHARD_TAG}"
export MAIN_CACHE_PATH="${DATA_SUBDIR}/cache"
export OUTPUT_PATH="${DATA_SUBDIR}/output"
export NUM_THREADS="${NUM_THREADS:-288}"

# Isolated per-task copy of the driver scripts (see header) — NOT a copy of
# deduplicate-text-datasets itself, which exact_substring_dedup.sh clones
# fresh here every run.
TASK_WORKDIR="${DATA_SUBDIR}/workdir"

# Resumability: skip shards already completed by a prior run of this array
# (rerun the array and only missing/failed shards do work). Delete this
# marker to force a redo of that shard.
SUCCESS_MARKER="${OUTPUT_PATH}/_SUCCESS"
if [[ -f "${SUCCESS_MARKER}" ]]; then
    echo "Shard ${SHARD_ID} already complete (${SUCCESS_MARKER} present) — skipping."
    exit 0
fi

if [[ ! -d "${INPUT_PATH}" ]]; then
    echo "ERROR: ${INPUT_PATH} does not exist — did shard_symlinks.py run with a matching --num-shards?" >&2
    exit 1
fi

rm -rf "${TASK_WORKDIR}"
mkdir -p "${TASK_WORKDIR}" "${REPO_DIR}/logs"
cp "${STEP_SRC_DIR}"/*.py "${STEP_SRC_DIR}"/*.sh "${TASK_WORKDIR}/"

echo "=================================================="
echo "  Nemotron-CC — substring dedup array task"
echo "  Job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} | shard ${SHARD_ID} | $(hostname)"
echo "  Input    ${INPUT_PATH}"
echo "  Cache    ${MAIN_CACHE_PATH}"
echo "  Output   ${OUTPUT_PATH}"
echo "  Workdir  ${TASK_WORKDIR}"
echo "=================================================="

srun --label --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    "${CONTAINER_FLAGS[@]}" \
    bash -c "
cd '${TASK_WORKDIR}'
export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}_${SHARD_ID}
export RAY_PORT_BROADCAST_DIR='${REPO_DIR}/logs'
export INPUT_PATH='${INPUT_PATH}'
export MAIN_CACHE_PATH='${MAIN_CACHE_PATH}'
export OUTPUT_PATH='${OUTPUT_PATH}'
export NUM_THREADS='${NUM_THREADS}'
bash exact_substring_dedup.sh
"

touch "${SUCCESS_MARKER}"
echo "=================================================="
echo "  DONE shard ${SHARD_ID} -> ${OUTPUT_PATH}"
echo "=================================================="
