#!/bin/bash
# =============================================================================
# Download the original Nemotron-CC dataset from Common Crawl — SLURM ARRAY job
# (one task per shard).
#
# Each array task reads its shard list (runs/<RUN_NAME>/shard_<idx>.txt, produced
# by src/original_nem_cc/prepare_download_list.py) and downloads just those
# files. Every shard line is two tab-separated columns:
#
#   <remote path from the index>\t<local path relative to OUTPUT_DIR>
#
# so this script only has to prepend BASE_URL to column 1 and write to column 2.
#
# No container: this is plain curl against data.commoncrawl.org over the compute
# node's outbound HTTPS.
#
# Downloads are idempotent — a file that already exists at its final path is
# skipped, and partial transfers land on <file>.part so an interrupted job never
# leaves a truncated .zstd behind.
#
# This is not launched directly — scripts/submit_original_nem_cc/ prepares the
# shard lists on the host, then submits this with `--array=0-(N-1)`.
#
# Required env vars (exported by the submit script):
#   RUN_NAME, OUTPUT_DIR
# Optional:
#   BASE_URL, CONCURRENCY, RETRIES, VERIFY, FORCE, REPO_DIR
# =============================================================================

#SBATCH --job-name=dl-original-nem-cc
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/users/%u/repos/nemotron_cc_pipeline/logs/download_original_nem_cc_%A_%a.log
#SBATCH --error=/users/%u/repos/nemotron_cc_pipeline/logs/download_original_nem_cc_%A_%a.log

set -euo pipefail

: "${RUN_NAME:?Set RUN_NAME}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${SLURM_ARRAY_TASK_ID:?This script must be run as a SLURM array job}"

REPO_DIR="${REPO_DIR:-/users/${USER}/repos/nemotron_cc_pipeline}"

BASE_URL="${BASE_URL:-https://data.commoncrawl.org}"
BASE_URL="${BASE_URL%/}"
CONCURRENCY="${CONCURRENCY:-8}"   # parallel curl transfers within this task
RETRIES="${RETRIES:-5}"           # curl --retry per file
VERIFY="${VERIFY:-1}"             # 1 => zstd -t each downloaded file (if zstd exists)
FORCE="${FORCE:-0}"               # 1 => re-download files that already exist

# Zero-padding MUST match prepare_download_list.py's _SHARD_PAD (5).
SHARD_ID="$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")"
SHARD_FILE="${REPO_DIR}/runs/${RUN_NAME}/shard_${SHARD_ID}.txt"

if [[ ! -f "${SHARD_FILE}" ]]; then
    echo "ERROR: shard list not found: ${SHARD_FILE}" >&2
    echo "       (did prepare_download_list.py run with the same RUN_NAME?)" >&2
    exit 1
fi

NUM_FILES=$(wc -l < "${SHARD_FILE}")

echo "=================================================="
echo "  Nemotron-CC download — array task ${SLURM_ARRAY_TASK_ID}"
echo "=================================================="
echo "  Job        : ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME:-$(hostname)}"
echo "  Run name   : ${RUN_NAME}"
echo "  Shard file : ${SHARD_FILE} (${NUM_FILES} file(s))"
echo "  Base URL   : ${BASE_URL}"
echo "  Output     : ${OUTPUT_DIR}"
echo "  Concurrency: ${CONCURRENCY}  retries: ${RETRIES}  verify: ${VERIFY}  force: ${FORCE}"
echo "=================================================="

mkdir -p "${REPO_DIR}/logs" "${OUTPUT_DIR}"

# Per-file worker, run by xargs -P. Args: <remote path> <local relative path>.
# Exits non-zero on failure so xargs reports it and the task fails loudly.
fetch_one() {
    local remote="$1" rel="$2"
    local dest="${OUTPUT_DIR}/${rel}"
    local part="${dest}.part"

    if [[ -s "${dest}" && "${FORCE}" != "1" ]]; then
        echo "SKIP  ${rel} (already present)"
        return 0
    fi

    mkdir -p "$(dirname "${dest}")"
    if ! curl --fail --location --silent --show-error \
              --retry "${RETRIES}" --retry-delay 5 --retry-connrefused \
              --connect-timeout 30 --continue-at - \
              --output "${part}" "${BASE_URL}/${remote}"; then
        echo "FAIL  ${rel} (download error)" >&2
        rm -f "${part}"
        return 1
    fi

    if [[ "${VERIFY}" == "1" ]] && command -v zstd >/dev/null 2>&1; then
        if ! zstd --test --quiet "${part}"; then
            echo "FAIL  ${rel} (zstd integrity check failed)" >&2
            rm -f "${part}"
            return 1
        fi
    fi

    # Rename only after a complete (and, if enabled, verified) transfer, so the
    # final path is always a good file and a re-run can skip it.
    mv -f "${part}" "${dest}"
    echo "OK    ${rel} ($(du -h "${dest}" | cut -f1))"
}
export -f fetch_one
export OUTPUT_DIR BASE_URL RETRIES VERIFY FORCE

# Feed the shard list to xargs as NUL-free tab-separated pairs. -L1 gives each
# worker exactly one (remote, local) pair; word splitting on the tab is fine
# because Common Crawl paths contain no whitespace.
START=$(date +%s)
set +e
tr '\t' '\n' < "${SHARD_FILE}" \
    | xargs -r -n2 -P "${CONCURRENCY}" bash -c 'fetch_one "$0" "$1"'
STATUS=$?
set -e
ELAPSED=$(( $(date +%s) - START ))

PRESENT=$(awk -F'\t' -v out="${OUTPUT_DIR}" \
    '{ cmd = "test -s \"" out "/" $2 "\""; if (system(cmd) == 0) n++ } END { print n+0 }' \
    "${SHARD_FILE}")

echo "=================================================="
echo "  Shard ${SHARD_ID}: ${PRESENT}/${NUM_FILES} file(s) present after ${ELAPSED}s"
if [[ ${STATUS} -ne 0 || ${PRESENT} -ne ${NUM_FILES} ]]; then
    echo "  FAILED (xargs status ${STATUS}) — re-submit this task to resume; existing"
    echo "          files are skipped."
    echo "=================================================="
    exit 1
fi
echo "  DONE"
echo "=================================================="
