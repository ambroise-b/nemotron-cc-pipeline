#!/bin/bash
# =============================================================================
# Shard-normalization stage: rewrite a directory of JSONL into an equivalent
# directory where no file exceeds $SHARD_SIZE (default 480M, safely under the
# 512MB FilePartitioningStage limit). Content is unchanged — same records, same
# schema — only split on line boundaries.
#
# Why: the dedup/filter stages preserve their input's file COUNT (one output
# file per partition), so oversized files born at stage 1.5 (URL/PII filter)
# propagate all the way down and eventually OOM a heavy stage (stage 3 loads a
# 2.3GB FastText model per actor + GPU classifiers). Normalizing shard size
# ONCE — ideally right after the URL/PII filter — keeps every nemo_curator
# stage healthy, since those stages self-regulate to ~blocksize when inputs are
# already small. See scripts/CSCS-QUICKSTART.md.
#
# Tech: plain `split -C` (coreutils) — streams bytes on line boundaries, so no
# JSON parsing and none of the DataFrame memory blowup we're avoiding. Runs as
# a Slurm ARRAY; each task splits a CHUNK of input files in parallel (xargs).
#
# Not submitted directly — use scripts/submit_nemotron_cc/01c_submit_reshard.sh,
# which builds the file manifest and sets the --array range. Required env:
#   RESHARD_INPUT_DIR, RESHARD_OUTPUT_DIR, RESHARD_MANIFEST
# Optional env: SHARD_SIZE (480M), FILES_PER_TASK (32), PARALLEL (32)
# =============================================================================

#SBATCH --job-name=reshard-jsonl
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/users/%u/repos/nemotron_cc_pipeline/logs/reshard_%A_%a.log
#SBATCH --error=/users/%u/repos/nemotron_cc_pipeline/logs/reshard_%A_%a.log

set -euo pipefail

: "${RESHARD_INPUT_DIR:?set RESHARD_INPUT_DIR}"
: "${RESHARD_OUTPUT_DIR:?set RESHARD_OUTPUT_DIR}"
: "${RESHARD_MANIFEST:?set RESHARD_MANIFEST (file list, one path per line)}"
SHARD_SIZE="${SHARD_SIZE:-480M}"
FILES_PER_TASK="${FILES_PER_TASK:-32}"
PARALLEL="${PARALLEL:-32}"

mkdir -p "${RESHARD_OUTPUT_DIR}"

# This array task owns manifest lines [start..end] (1-indexed).
start=$(( SLURM_ARRAY_TASK_ID * FILES_PER_TASK + 1 ))
end=$(( start + FILES_PER_TASK - 1 ))
mapfile -t files < <(sed -n "${start},${end}p" "${RESHARD_MANIFEST}")

echo "[$(hostname)] array task ${SLURM_ARRAY_TASK_ID}: manifest lines ${start}-${end}, ${#files[@]} file(s), shard size ${SHARD_SIZE}"
if [[ ${#files[@]} -eq 0 ]]; then
    echo "  nothing to do for this task index — exiting cleanly."
    exit 0
fi

export RESHARD_OUTPUT_DIR SHARD_SIZE

# One `split` per input file, up to $PARALLEL at once. `-C` keeps whole lines
# per shard (never splits a JSON record); `_part000.jsonl` suffix preserves the
# source file's identity so a shard is always traceable to its origin.
printf '%s\n' "${files[@]}" | xargs -P "${PARALLEL}" -I{} bash -c '
    infile="$1"
    [[ -s "$infile" ]] || { echo "  skip empty/missing: $infile"; exit 0; }
    base=$(basename "$infile" .jsonl)
    split -C "$SHARD_SIZE" -d -a 3 --additional-suffix=.jsonl \
        "$infile" "$RESHARD_OUTPUT_DIR/${base}_part"
    echo "  sharded $(basename "$infile")"
' _ {}

echo "[$(hostname)] array task ${SLURM_ARRAY_TASK_ID}: done."
