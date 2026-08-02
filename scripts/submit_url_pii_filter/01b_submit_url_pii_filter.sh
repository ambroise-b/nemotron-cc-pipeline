#!/bin/bash
# =============================================================================
# Submit the URL (robots.txt) + PII filtering stage as a SLURM array.
#
# Flow:
#   1. prepare  — run HERE on the host (not a SLURM job): prepare_dumps.py is a
#                 lightweight stdlib-only file-list + split. It writes NUM_SHARDS
#                 shard lists under runs/<RUN_NAME>/ and prints the shard count.
#   2. array    — submit one array job with that many tasks (one per shard),
#                 each running url_pii_filter.py over its shard.
#
# The array replaces launching NUM_SHARDS individual jobs.
#
# Run from the project root:
#   bash scripts/submit_url_pii_filter/submit_url_pii_filter.sh
#
# Edit the variables below, not the sbatch call.
# =============================================================================
set -euo pipefail

if [[ ! -f scripts/slurm/run_url_pii_filter.sbatch ]]; then
    echo "ERROR: run this from the project root (scripts/slurm/run_url_pii_filter.sbatch not found here)." >&2
    exit 1
fi
: "${SCRATCH:?SCRATCH not set — expected on CSCS Clariden/Alps}"

# --- Run identity ------------------------------------------------------------
RUN_NAME="CC-MAIN-2026-21_all"     # shard lists go to runs/<RUN_NAME>/
NUM_SHARDS=100                 # number of shards == number of array tasks
MAX_PARALLEL=50                # max array tasks running at once (SLURM '%' cap);
                               # empty = no cap (scheduler/QOS decides)

# --- Paths (everything under $SCRATCH) ---------------------------------------
# INPUT_DIR is step_1-extract_local_warc.py's extracted JSONL output.
# The filtered output is a schema-identical drop-in for step_2a to consume.
#DATA_DIR="${SCRATCH}/nemotron-cc-data"
DATA_DIR="${SCRATCH}/nemotron-cc-pipeline-CC-MAIN-2026-21"
INPUT_DIR="${DATA_DIR}/extracted/${RUN_NAME}"          # step_1 extracted JSONL (dump root)
OUTPUT_DIR="${DATA_DIR}/url_pii_filtered/${RUN_NAME}"  # kept docs (flat, shard-prefixed files)
REMOVED_DIR="${DATA_DIR}/url_pii_removed/${RUN_NAME}"  # robots-excluded docs (separate tree)
LOGGING_DIR="${DATA_DIR}/logs/url_pii_filter/${RUN_NAME}"

# robots.txt domain exclusion list. Leave empty to use url_pii_filter.py's
# default (the create_robots_txt_filter_scalable submodule's domain list).
ROBOTS_LIST=""

# --- Input / output / parallelism --------------------------------------------
PATTERN='*.jsonl'              # step_1's JsonlWriter emits plain .jsonl files
OUTPUT_FILETYPE=jsonl          # keep 'jsonl' to replicate step_1's output
COMPRESSION=none               # 'none' = plain .jsonl, matching step_1's JsonlWriter
TASKS=20                       # datatrove tasks per shard (intra-node parallelism)
WORKERS=-1                     # -1 => workers == tasks

# --- SLURM resources ---------------------------------------------------------
ACCOUNT=infra01
PARTITION=normal
CPUS_PER_TASK=288
MEM=850000
TIME=04:00:00

# --- SLURM reservation (optional) --------------------------------------------
# When RESERVATION is non-empty, --reservation=<RESERVATION> is appended to the
# sbatch call below; when it's "", the flag is omitted entirely and Slurm
# schedules normally. Comment/uncomment to switch pools or disable it quickly.
#RESERVATION="SD-69241-apertus-1-5-0"
RESERVATION=""

echo "Submitting URL + PII filter run '${RUN_NAME}'"
echo "  input:  ${INPUT_DIR}"
echo "  output: ${OUTPUT_DIR}"

# 1. Prepare shard lists HERE on the host (lightweight stdlib-only script — no
#    SLURM job). It writes runs/<RUN_NAME>/shard_*.txt and prints the number of
#    shards actually created on stdout (logs go to stderr).
echo "Preparing shard lists (${NUM_SHARDS} shards) ..."
NUM_SHARDS_MADE=$(
    python3 src/url_pii_filter/prepare_dumps.py \
        --input-dir "${INPUT_DIR}" --name "${RUN_NAME}" \
        --num-shards "${NUM_SHARDS}" --pattern "${PATTERN}" \
    | tail -n 1
)
echo "  prepared ${NUM_SHARDS_MADE} shard list(s) under runs/${RUN_NAME}/"

# 2. Submit ONE array job with exactly that many tasks (one per shard).
#    Append '%MAX_PARALLEL' to cap how many run simultaneously (if set).
ARRAY_SPEC="0-$((NUM_SHARDS_MADE - 1))"
[[ -n "${MAX_PARALLEL}" ]] && ARRAY_SPEC="${ARRAY_SPEC}%${MAX_PARALLEL}"

ARRAY_JID=$(
    RUN_NAME="${RUN_NAME}" INPUT_DIR="${INPUT_DIR}" OUTPUT_DIR="${OUTPUT_DIR}" \
    REMOVED_DIR="${REMOVED_DIR}" \
    ROBOTS_LIST="${ROBOTS_LIST}" TASKS="${TASKS}" WORKERS="${WORKERS}" \
    OUTPUT_FILETYPE="${OUTPUT_FILETYPE}" COMPRESSION="${COMPRESSION}" \
    LOGGING_DIR="${LOGGING_DIR}" \
    sbatch --parsable -A "${ACCOUNT}" -p "${PARTITION}" \
        --array="${ARRAY_SPEC}" \
        --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEM}" --time="${TIME}" \
        --exclusive --no-requeue \
        ${RESERVATION:+--reservation="${RESERVATION}"} \
        scripts/slurm/run_url_pii_filter.sbatch
)
echo "  array job:   ${ARRAY_JID} (array=${ARRAY_SPEC})"
