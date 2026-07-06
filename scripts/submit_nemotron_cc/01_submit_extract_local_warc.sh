#!/bin/bash
# =============================================================================
# Submit stage 1 (LOCAL WARC extraction) as a multi-node SLURM job.
#
# Alternative to 01_submit_download.sh: instead of downloading Common Crawl,
# extract from WARC files already on the cluster. Uses
# src/nemotron-cc/step_1-extract_local_warc.py, which applies the SAME
# nemo_curator JusText extraction + language-ID + unicode stages as the
# download path, so its JSONL output is interchangeable.
#
# Run from the project root:
#   bash scripts/submit_default/01b_submit_extract_local_warc.sh
#
# Edit the variables below, not the sbatch call itself.
# =============================================================================
set -euo pipefail

if [[ ! -f scripts/run_stage.sbatch ]]; then
    echo "ERROR: run this from the project root (scripts/run_stage.sbatch not found here)." >&2
    exit 1
fi
: "${SCRATCH:?SCRATCH not set — expected on CSCS Clariden/Alps}"

# --- SLURM resources ---------------------------------------------------------
ACCOUNT=infra01
PARTITION=normal
NODES=4
GPUS_PER_NODE=4
CPUS_PER_TASK=288
MEM=850000   # MB; a little under the node's 870000 to leave OS/slurmd headroom
TIME=04:00:00

# --- Stage 1 (local WARC) args ----------------------------------------------
RUN_NAME=CC-MAIN-2026-21
# Pre-downloaded WARC dump (read recursively, nested subdirs supported):
WARC_DIR="/capstor/store/cscs/swissai/infra01/kpitas/common-crawl-CC-MAIN-2026-21/data/crawl-data/CC-MAIN-2026-21"
# Output/cache stay on $SCRATCH. This output dir is the INPUT_DIR of the
# url+pii filtering step (scripts/submit_url_pii_filter/).
DATA_DIR="${SCRATCH}/nemotron-cc-data"
OUTPUT_DIR="${DATA_DIR}/extracted/${RUN_NAME}"
CACHE_DIR="${DATA_DIR}/cache/step1_local/${RUN_NAME}"
LANGUAGES=EN

STEP_SCRIPT="src/nemotron-cc/step_1-extract_local_warc.py"
STEP_ARGS="--warc-dir ${WARC_DIR} \
--languages ${LANGUAGES} \
--output-dir ${OUTPUT_DIR} --cache-dir ${CACHE_DIR}"

echo "Submitting stage 1 (local WARC extraction): ${NODES} nodes, ${GPUS_PER_NODE} GPUs/node"
echo "  WARC:   ${WARC_DIR}"
echo "  output: ${OUTPUT_DIR}"
STEP_SCRIPT="${STEP_SCRIPT}" STEP_ARGS="${STEP_ARGS}" \
sbatch -A "${ACCOUNT}" -p "${PARTITION}" \
    --nodes="${NODES}" --gpus-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEM}" --time="${TIME}" \
    --exclusive --no-requeue \
    scripts/run_stage.sbatch
