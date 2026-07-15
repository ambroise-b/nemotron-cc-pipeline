#!/bin/bash
# =============================================================================
# Submit stage 1 (download & extract) as a multi-node SLURM job.
#
# Run from the project root:
#   bash scripts/submit_default/01_submit_download.sh
#
# Values below are pre-filled from scripts/CSCS-QUICKSTART.md / configs/sample.env,
# scaled to 4 nodes. Edit the variables below, not the sbatch call itself.
# =============================================================================
set -euo pipefail

if [[ ! -f scripts/slurm/run_stage.sbatch ]]; then
    echo "ERROR: run this from the project root (scripts/slurm/run_stage.sbatch not found here)." >&2
    exit 1
fi
: "${SCRATCH:?SCRATCH not set — expected on CSCS Clariden/Alps}"

# --- SLURM resources ---------------------------------------------------------
# Every node on this cluster has 4 GPUs, 288 CPUs, ~870GB RAM (confirmed via
# `scontrol show node`) — since --gpus-per-node=4 already claims a node
# exclusively regardless, request the whole thing explicitly rather than
# getting whatever Slurm's default per-CPU memory ratio happens to grant.
ACCOUNT=infra01
PARTITION=normal
NODES=4
GPUS_PER_NODE=4
CPUS_PER_TASK=288
MEM=850000   # MB; a little under the node's 870000 to leave OS/slurmd headroom
TIME=04:00:00

# --- SLURM reservation (optional) --------------------------------------------
# When RESERVATION is non-empty, --reservation=<RESERVATION> is appended to the
# sbatch call below; when it's "", the flag is omitted entirely and Slurm
# schedules normally. Comment/uncomment to switch pools or disable it quickly.
RESERVATION="SD-69241-apertus-1-5-0"
#RESERVATION=""

# --- Stage 1 args ------------------------------------------------------------
DATA_DIR="${SCRATCH}/nemotron-cc-data"
START_SNAPSHOT=2024-46
END_SNAPSHOT=2024-46
# Leave URL_LIMIT/RECORD_LIMIT empty ("") for no limit (full-scale run) — the
# underlying script takes plain ints and errors on the literal string "None",
# so "no limit" means omitting the flag entirely, which the ${VAR:+...}
# expansions below do automatically when the variable is empty.
URL_LIMIT=300
RECORD_LIMIT=""
# Space-separated language codes to keep (e.g. "EN" or "EN DE FR").
# Leave EMPTY to keep ALL languages: the --languages flag is then omitted (via
# the ${LANGUAGES:+...} expansion), so no language filtering is applied —
# language-ID still runs and is recorded.
LANGUAGES=""

STEP_SCRIPT="src/nemotron-cc/step_1-download_extract.py"
STEP_ARGS="--start-snapshot ${START_SNAPSHOT} --end-snapshot ${END_SNAPSHOT} \
${URL_LIMIT:+--url-limit ${URL_LIMIT}} ${RECORD_LIMIT:+--record-limit ${RECORD_LIMIT}} \
${LANGUAGES:+--languages ${LANGUAGES}} \
--output-dir ${DATA_DIR}/cleaned_extracted --cache-dir ${DATA_DIR}/cache/step1"

echo "Submitting stage 1 (download & extract): ${NODES} nodes, ${GPUS_PER_NODE} GPUs/node"
STEP_SCRIPT="${STEP_SCRIPT}" STEP_ARGS="${STEP_ARGS}" \
sbatch -A "${ACCOUNT}" -p "${PARTITION}" \
    --nodes="${NODES}" --gpus-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEM}" --time="${TIME}" \
    --exclusive --no-requeue \
    ${RESERVATION:+--reservation="${RESERVATION}"} \
    scripts/slurm/run_stage.sbatch
