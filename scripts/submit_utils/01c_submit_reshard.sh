#!/bin/bash
# =============================================================================
# Submit the shard-normalization stage (see scripts/slurm/reshard_jsonl.sbatch):
# rewrite a JSONL dir into equivalent files each <= $SHARD_SIZE, so downstream
# stages never see oversized, unsplittable partitions.
#
# PERMANENT placement is right after the URL/PII filter (stage 1.5) and before
# exact dedup (stage 2a) — fixing shard size once there keeps 2a/2b/3 healthy,
# because the nemo_curator dedup stages self-regulate to ~blocksize when their
# inputs are already small. The two RESHARD_INPUT_DIR lines below toggle between
# that permanent placement and the current one-off (resharding after fuzzy dedup
# to unblock stage 3 on already-produced data).
#
# Run from the project root:
#   bash scripts/submit_nemotron_cc/01c_submit_reshard.sh
# =============================================================================
set -euo pipefail

if [[ ! -f scripts/slurm/reshard_jsonl.sbatch ]]; then
    echo "ERROR: run this from the project root (scripts/slurm/reshard_jsonl.sbatch not found here)." >&2
    exit 1
fi
: "${SCRATCH:?SCRATCH not set — expected on CSCS Clariden/Alps}"

REPO_DIR="${REPO_DIR:-/users/${USER}/repos/nemotron_cc_pipeline}"
DATA_DIR="${SCRATCH}/nemotron-cc-pipeline-CC-MAIN-2014-10"

# --- What to reshard ---------------------------------------------------------
# PERMANENT (full runs): reshard the URL/PII filter output, before exact dedup.
export RESHARD_INPUT_DIR="${DATA_DIR}/url_pii_filtered"
export RESHARD_OUTPUT_DIR="${DATA_DIR}/url_pii_filtered_resharded"

export SHARD_SIZE="${SHARD_SIZE:-480M}"
FILES_PER_TASK="${FILES_PER_TASK:-32}"
MAXPAR="${MAXPAR:-16}"          # max array tasks (= nodes) running at once
export FILES_PER_TASK

# --- SLURM reservation (optional) --------------------------------------------
#RESERVATION="SD-69241-apertus-1-5-0"
RESERVATION=""

# --- Build the file manifest + array range -----------------------------------
mkdir -p "${RESHARD_OUTPUT_DIR}" "${REPO_DIR}/logs"
export RESHARD_MANIFEST="${REPO_DIR}/logs/reshard_manifest.txt"
find "${RESHARD_INPUT_DIR}" -maxdepth 2 -name '*.jsonl' | sort > "${RESHARD_MANIFEST}"

NFILES=$(wc -l < "${RESHARD_MANIFEST}")
if [[ "${NFILES}" -eq 0 ]]; then
    echo "ERROR: no .jsonl files found under ${RESHARD_INPUT_DIR}" >&2
    exit 1
fi
NTASKS=$(( (NFILES + FILES_PER_TASK - 1) / FILES_PER_TASK ))

echo "Resharding ${NFILES} file(s) from:"
echo "  in : ${RESHARD_INPUT_DIR}"
echo "  out: ${RESHARD_OUTPUT_DIR}"
echo "  ${FILES_PER_TASK} files/task -> ${NTASKS} array task(s), shard size ${SHARD_SIZE}, up to ${MAXPAR} concurrent"

sbatch ${RESERVATION:+--reservation="${RESERVATION}"} \
    --array=0-$(( NTASKS - 1 ))%"${MAXPAR}" \
    scripts/slurm/reshard_jsonl.sbatch
