#!/bin/bash
# =============================================================================
# URL (robots.txt) + PII filtering — SLURM ARRAY job (one task per shard)
#
# Each array task processes a single shard of the dump: it reads the shard's
# paths file (runs/<RUN_NAME>/shard_<idx>.txt, produced by prepare_dumps.py)
# and runs src/url_pii_filter/url_pii_filter.py over just those parquet files,
# writing to <OUTPUT_DIR>/shard_<idx>/.
#
# This is not launched directly — scripts/submit_url_pii_filter/ submits the
# prepare job, then submits this with `--array=0-(N-1)` and a dependency on it.
#
# Required env vars (exported by the submit script):
#   RUN_NAME, INPUT_DIR, OUTPUT_DIR, ROBOTS_LIST, TASKS, WORKERS,
#   OUTPUT_FILETYPE, COMPRESSION, LOGGING_DIR
# =============================================================================

#SBATCH --job-name=url-pii-filter
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=850000
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=04:00:00
#SBATCH --output=/users/aborbely/repos/nemotron_cc_pipeline/logs/url_pii_filter_%A_%a.log
#SBATCH --error=/users/aborbely/repos/nemotron_cc_pipeline/logs/url_pii_filter_%A_%a.log

set -euo pipefail

: "${RUN_NAME:?Set RUN_NAME}"
: "${INPUT_DIR:?Set INPUT_DIR}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${SLURM_ARRAY_TASK_ID:?This script must be run as a SLURM array job}"

REPO_DIR="${REPO_DIR:-/users/aborbely/repos/nemotron_cc_pipeline}"
STEP_SCRIPT="src/url_pii_filter/url_pii_filter.py"

ROBOTS_LIST="${ROBOTS_LIST:-}"           # empty => script uses its submodule default
# Removed docs go to a separate tree; default sibling of OUTPUT_DIR.
REMOVED_DIR="${REMOVED_DIR:-${OUTPUT_DIR%/}_removed}"
TASKS="${TASKS:-32}"
WORKERS="${WORKERS:--1}"
OUTPUT_FILETYPE="${OUTPUT_FILETYPE:-jsonl}"
COMPRESSION="${COMPRESSION:-none}"   # plain .jsonl, matching step_1's output
LOGGING_DIR="${LOGGING_DIR:-}"

# Zero-padding MUST match prepare_dumps.py's _SHARD_PAD (5).
SHARD_ID="$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")"
PATHS_FILE="${REPO_DIR}/runs/${RUN_NAME}/shard_${SHARD_ID}.txt"

if [[ ! -f "${PATHS_FILE}" ]]; then
    echo "ERROR: shard paths file not found: ${PATHS_FILE}" >&2
    echo "       (did the prepare job run and use the same RUN_NAME / num-shards?)" >&2
    exit 1
fi

CONTAINER_ENV="${CONTAINER_ENV:-${REPO_DIR}/container/container.toml}"
CONTAINER_FLAGS=(--environment="${CONTAINER_ENV}" --container-writable)
RUN_PYTHON="python3"

# Assemble url_pii_filter.py args. All shards write into the SAME flat
# OUTPUT_DIR / REMOVED_DIR, distinguished by the per-shard filename prefix.
ARGS="--input-dir ${INPUT_DIR} \
--paths-file ${PATHS_FILE} \
--output-dir ${OUTPUT_DIR} \
--removed-dir ${REMOVED_DIR} \
--output-name-prefix shard_${SHARD_ID}_ \
--output-filetype ${OUTPUT_FILETYPE} \
--compression ${COMPRESSION} \
--tasks ${TASKS} --workers ${WORKERS}"
[[ -n "${ROBOTS_LIST}" ]] && ARGS="${ARGS} --robots-list ${ROBOTS_LIST}"
[[ -n "${LOGGING_DIR}" ]] && ARGS="${ARGS} --logging-dir ${LOGGING_DIR%/}/shard_${SHARD_ID}"

echo "=================================================="
echo "  URL + PII filter — array task ${SLURM_ARRAY_TASK_ID}"
echo "=================================================="
echo "  Job       : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME:-?}"
echo "  Run name  : ${RUN_NAME}"
echo "  Shard file: ${PATHS_FILE}"
echo "  Output    : ${OUTPUT_DIR} (prefix shard_${SHARD_ID}_)"
echo "=================================================="

mkdir -p "${REPO_DIR}/logs"

srun \
    --label \
    --nodes=1 --ntasks=1 \
    "${CONTAINER_FLAGS[@]}" \
    bash -c "
cd '${REPO_DIR}'
echo \"[\$(hostname)] python=\$(${RUN_PYTHON} --version 2>&1)\"
${RUN_PYTHON} '${REPO_DIR}/${STEP_SCRIPT}' ${ARGS}
"

echo "=================================================="
echo "  DONE (shard ${SHARD_ID})"
echo "=================================================="
