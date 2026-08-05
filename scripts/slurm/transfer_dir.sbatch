#!/bin/bash
# =============================================================================
# Copy $TRANSFER_SRC into $TRANSFER_DST, preserving the source folder's name:
#   src=/path1/folder_to_copy  dst=/path2  ->  /path2/folder_to_copy
#
# Runs on the `xfer` partition: TRESBillingWeights are all 0 there, so transfers
# do not spend the compute allocation, and PriorityTier=30 means they start
# ~immediately. Verified that the xfer nodes (nid001154, nid001306) mount both
# /capstor and /iopsstor. Never run a multi-TB copy on the login node instead —
# long-running processes get reaped there and there is no restart story.
#
# Resumable: just re-submit the same job. rsync skips every file whose size and
# mtime already match at the destination, and --partial-dir keeps a file that was
# interrupted mid-write aside so the next run resumes it instead of restarting
# from byte 0. (Deliberately NOT --inplace: that would save space but leaves a
# corrupt, plausible-looking file behind if the job is killed mid-write.)
#
# Not submitted directly — use scripts/submit_utils/98_submit_transfer_dir.sh.
# Required env: TRANSFER_SRC, TRANSFER_DST
# =============================================================================

#SBATCH --job-name=transfer-dir
#SBATCH --account=infra01
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/users/%u/repos/nemotron_cc_pipeline/logs/transfer_%j.log
#SBATCH --error=/users/%u/repos/nemotron_cc_pipeline/logs/transfer_%j.log

set -euo pipefail

: "${TRANSFER_SRC:?set TRANSFER_SRC (the folder to copy)}"
: "${TRANSFER_DST:?set TRANSFER_DST (the PARENT dir it is copied into)}"

# Strip trailing slashes and build the destination path explicitly instead of
# relying on rsync's "src/" (contents-of) vs "src" (the-dir-itself) rule — that
# rule is the classic way to land one level off. Copying "$SRC/" into
# "$DST/$BASE/" behaves identically whether or not the caller wrote a slash.
SRC="${TRANSFER_SRC%/}"
BASE="$(basename "${SRC}")"
DEST="${TRANSFER_DST%/}/${BASE}"

[[ -d "${SRC}" ]] || { echo "ERROR: source is not a directory: ${SRC}" >&2; exit 1; }
mkdir -p "${DEST}"

echo "[$(hostname)] job ${SLURM_JOB_ID:-local}: ${SRC} -> ${DEST}"

rsync -a --numeric-ids --partial-dir=.rsync-partial --stats --human-readable \
      "${SRC}/" "${DEST}/"

echo "done: ${SRC} -> ${DEST}"
