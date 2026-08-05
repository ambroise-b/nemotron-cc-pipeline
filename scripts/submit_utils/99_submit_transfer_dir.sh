#!/bin/bash
# =============================================================================
# Submit a bulk directory copy (see scripts/slurm/transfer_dir.sbatch).
#
# Semantics: the source folder KEEPS ITS NAME inside the destination parent.
#   TRANSFER_SRC=/path1/folder_to_copy
#   TRANSFER_DST=/path2
#   ->  /path2/folder_to_copy
# So TRANSFER_DST is the PARENT directory, not the final path. Trailing slashes
# on either variable are stripped and make no difference.
#
# This is the only file you edit: set the two paths below, then submit. Safe to
# re-submit at any time — the copy is resumable and re-running an already complete
# transfer is a cheap no-op (rsync just re-stats everything and skips it).
#
# Run from the project root:
#   bash scripts/submit_utils/99_submit_transfer_dir.sh
# =============================================================================
set -euo pipefail

if [[ ! -f scripts/slurm/transfer_dir.sbatch ]]; then
    echo "ERROR: run this from the project root (scripts/slurm/transfer_dir.sbatch not found here)." >&2
    exit 1
fi

: "${SCRATCH:?SCRATCH not set — expected on CSCS Clariden/Alps}"
: "${STORE:?STORE not set — expected on CSCS Clariden/Alps}"

# Repo root only — this exists so the logs/ dir the sbatch writes into is created
# before submit. Must match --output in scripts/slurm/transfer_dir.sbatch, or
# sbatch fails with no log file to explain why. Not a data path.
REPO_DIR="/users/aborbely/repos/nemotron_cc_pipeline"

# --- What to copy ------------------------------------------------------------
#export TRANSFER_SRC="${SCRATCH}/nemotron-cc-pipeline-CC-MAIN-2023-06/substring_dedup"
#export TRANSFER_DST="${STORE}/datasets/apertus-implementation-nemotron-cc/CC-MAIN-2023-06"

export TRANSFER_SRC="${SCRATCH}/nemotron-cc-pipeline-CC-MAIN-2023-06/quality_labeling"
export TRANSFER_DST="${STORE}/datasets/apertus-implementation-nemotron-cc/CC-MAIN-2023-06"

# Walltime: ~1-3 GB/s aggregate on Lustre, so 1-6 TB lands well inside 12h.
# The xfer partition caps at 24h (1-00:00:00) if you ever need more.
TIME_LIMIT="12:00:00"

# --- Preflight ---------------------------------------------------------------
if [[ -z "${TRANSFER_SRC}" || -z "${TRANSFER_DST}" ]]; then
    echo "ERROR: set TRANSFER_SRC and TRANSFER_DST at the top of this script." >&2
    exit 1
fi
[[ -d "${TRANSFER_SRC}" ]] || { echo "ERROR: source does not exist: ${TRANSFER_SRC}" >&2; exit 1; }

mkdir -p "${REPO_DIR}/logs"

BASE="$(basename "${TRANSFER_SRC%/}")"
SRC_SIZE="$(du -sh --apparent-size "${TRANSFER_SRC}" | cut -f1)"

echo "Transferring ${SRC_SIZE}:"
echo "  from: ${TRANSFER_SRC}"
echo "  to  : ${TRANSFER_DST%/}/${BASE}"
echo "  time: ${TIME_LIMIT} on partition xfer (billing=0 — does not use the compute allocation)"

sbatch --time="${TIME_LIMIT}" scripts/slurm/transfer_dir.sbatch
