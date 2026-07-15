#!/bin/bash
# =============================================================================
# Warm the shared HuggingFace cache ($HF_HUB_CACHE) so every later stage can
# run fully offline (HF_HUB_OFFLINE=1, the default in run_stage.sbatch /
# run_sdg_split.sbatch). Run this ONCE, before the pipeline stages.
#
# Thin wrapper around scripts/slurm/warm_hf_cache.sbatch — edit that file, not
# this one, to change resources or the model list.
#
# Run from the project root:
#   bash scripts/submit_init/00_submit_warm_hf_cache.sh          # + Qwen (SDG)
#   INCLUDE_QWEN=0 bash scripts/submit_init/00_submit_warm_hf_cache.sh  # skip Qwen
# =============================================================================
set -euo pipefail

if [[ ! -f scripts/slurm/warm_hf_cache.sbatch ]]; then
    echo "ERROR: run this from the project root (scripts/slurm/warm_hf_cache.sbatch not found here)." >&2
    exit 1
fi

# --- SLURM reservation (optional) --------------------------------------------
# When RESERVATION is non-empty, --reservation=<RESERVATION> is appended to the
# sbatch call below; when it's "", the flag is omitted entirely and Slurm
# schedules normally. (All other resources live in the .sbatch file.)
RESERVATION="SD-69241-apertus-1-5-0"
#RESERVATION=""

echo "Submitting HF cache warm-up (see scripts/slurm/warm_hf_cache.sbatch for config)"
sbatch ${RESERVATION:+--reservation="${RESERVATION}"} scripts/slurm/warm_hf_cache.sbatch
