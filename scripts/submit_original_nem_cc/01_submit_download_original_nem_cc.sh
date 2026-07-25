#!/bin/bash
# =============================================================================
# Download the ORIGINAL Nemotron-CC dataset (NVIDIA's released JSONL) straight
# from Common Crawl, as a SLURM array.
#
# Flow:
#   1. prepare  — run HERE on the host (not a SLURM job): prepare_download_list.py
#                 re-downloads the path index
#                   https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz
#                 filters it on quality / kind / kind2, caps it at MAX_FILES, and
#                 writes NUM_SHARDS shard lists under runs/<RUN_NAME>/.
#   2. array    — submit one array job with that many tasks (one per shard), each
#                 curl-ing its files into OUTPUT_DIR.
#
# Local layout mirrors the remote partitioning, minus the constant prefix:
#   <OUTPUT_DIR>/quality=high/kind=actual/kind2=actual/CC-MAIN-2013-20-part-00016.jsonl.zstd
#
# Re-running is safe and resumes: files already present are skipped, partial
# transfers live at <file>.part and never become a final filename.
#
# Run from the project root:
#   bash scripts/submit_original_nem_cc/01_submit_download_original_nem_cc.sh
#
# Edit the variables below, not the sbatch call.
# =============================================================================
set -euo pipefail

if [[ ! -f scripts/slurm/run_download_original_nem_cc.sbatch ]]; then
    echo "ERROR: run this from the project root (scripts/slurm/run_download_original_nem_cc.sbatch not found here)." >&2
    exit 1
fi
: "${SCRATCH:?SCRATCH not set — expected on CSCS Clariden/Alps}"

# --- Run identity ------------------------------------------------------------
RUN_NAME="original-nem-cc_high_actual"   # shard lists go to runs/<RUN_NAME>/
NUM_SHARDS=4                   # number of shards == number of array tasks
MAX_PARALLEL=4                 # max array tasks running at once (SLURM '%' cap);
                               # empty = no cap (scheduler/QOS decides).
                               # Keep this modest — every task hammers
                               # data.commoncrawl.org with CONCURRENCY transfers.

# --- Subset selection --------------------------------------------------------
QUALITY="high"                 # high | medium-high | medium | medium-low | low | ""
KIND="actual"                  # actual | synthetic | ""
KIND2="actual"                 # actual | distill | diverse_qa_pairs |
                               # extract_knowledge | knowledge_list | wrap_medium | ""

MAX_FILES=2000                  # cap on files to download; 0 = everything matched
SELECT=spread                  # how MAX_FILES picks files when several
                               # quality/kind/kind2 groups match:
                               #   spread = round-robin across groups (even sample)
                               #   head   = first N in sorted order (contiguous parts)

# --- Paths (everything under $SCRATCH) ---------------------------------------
DATA_DIR="${SCRATCH}/original-nem-cc"
OUTPUT_DIR="${DATA_DIR}/data-jsonl"   # download root; remote partitioning preserved

# --- Download behaviour ------------------------------------------------------
BASE_URL="https://data.commoncrawl.org"
CONCURRENCY=4                  # parallel curl transfers per array task
RETRIES=5                      # curl --retry per file
VERIFY=1                       # 1 => zstd -t every downloaded file
FORCE=0                        # 1 => re-download files that already exist

# --- SLURM resources ---------------------------------------------------------
# This stage is network-bound, not compute-bound: ask for few CPUs and no
# --exclusive so it schedules fast and doesn't hold a whole node hostage.
ACCOUNT=infra01
PARTITION=normal
CPUS_PER_TASK=32
TIME=06:00:00

# --- SLURM reservation (optional) --------------------------------------------
# When RESERVATION is non-empty, --reservation=<RESERVATION> is appended to the
# sbatch call below; when it's "", the flag is omitted entirely.
#RESERVATION="SD-69241-apertus-1-5-0"
RESERVATION=""

echo "Submitting original Nemotron-CC download '${RUN_NAME}'"
echo "  selection: quality='${QUALITY:-<any>}' kind='${KIND:-<any>}' kind2='${KIND2:-<any>}'"
echo "  max files: ${MAX_FILES} (select=${SELECT})"
echo "  output:    ${OUTPUT_DIR}"

# 1. Prepare shard lists HERE on the host (stdlib-only script — no SLURM job).
#    It re-downloads the path index, filters it, and prints the number of shards
#    actually created on stdout (logs go to stderr).
echo "Preparing shard lists (${NUM_SHARDS} requested) ..."
NUM_SHARDS_MADE=$(
    python3 src/original_nem_cc/prepare_download_list.py \
        --name "${RUN_NAME}" \
        --num-shards "${NUM_SHARDS}" \
        --quality "${QUALITY}" --kind "${KIND}" --kind2 "${KIND2}" \
        --max-files "${MAX_FILES}" --select "${SELECT}" \
        --base-url "${BASE_URL}" \
    | tail -n 1
)
echo "  prepared ${NUM_SHARDS_MADE} shard list(s) under runs/${RUN_NAME}/"

# 2. Submit ONE array job with exactly that many tasks (one per shard).
#    Append '%MAX_PARALLEL' to cap how many run simultaneously (if set).
ARRAY_SPEC="0-$((NUM_SHARDS_MADE - 1))"
[[ -n "${MAX_PARALLEL}" ]] && ARRAY_SPEC="${ARRAY_SPEC}%${MAX_PARALLEL}"

ARRAY_JID=$(
    RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" BASE_URL="${BASE_URL}" \
    CONCURRENCY="${CONCURRENCY}" RETRIES="${RETRIES}" \
    VERIFY="${VERIFY}" FORCE="${FORCE}" \
    sbatch --parsable -A "${ACCOUNT}" -p "${PARTITION}" \
        --array="${ARRAY_SPEC}" \
        --cpus-per-task="${CPUS_PER_TASK}" --time="${TIME}" \
        --no-requeue \
        ${RESERVATION:+--reservation="${RESERVATION}"} \
        scripts/slurm/run_download_original_nem_cc.sbatch
)
echo "  array job:   ${ARRAY_JID} (array=${ARRAY_SPEC})"
