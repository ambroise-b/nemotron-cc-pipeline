#!/bin/bash
# =============================================================================
# Submit stage 4 (SDG) using the split-node design: dedicated server nodes
# running a real multi-node vLLM deployment (Ray's `symmetric-run`), a
# dedicated compute node calling it over HTTP.
#
# Run from the project root:
#   bash scripts/submit_default/04b_submit_sdg_split.sh
#
# Alternative to 04_submit_sdg.sh (single-node --serve-model, limited to
# whatever fits on one node). All the actual config (node counts, model,
# vLLM flags, tasks) lives in scripts/slurm/run_sdg_split.sbatch itself — this is
# just a thin, consistently-located pointer to it; edit that file, not this
# one, to change resources/model/etc.
# =============================================================================
set -euo pipefail

if [[ ! -f scripts/slurm/run_sdg_split.sbatch ]]; then
    echo "ERROR: run this from the project root (scripts/slurm/run_sdg_split.sbatch not found here)." >&2
    exit 1
fi

# --- SLURM reservation (optional) --------------------------------------------
# When RESERVATION is non-empty, --reservation=<RESERVATION> is appended to the
# sbatch call below; when it's "", the flag is omitted entirely and Slurm
# schedules normally. Comment/uncomment to switch pools or disable it quickly.
# (All other resources live in scripts/slurm/run_sdg_split.sbatch.)
RESERVATION="SD-69241-apertus-1-5-0"
#RESERVATION=""

echo "Submitting stage 4, split-node design (see scripts/slurm/run_sdg_split.sbatch for config)"
sbatch ${RESERVATION:+--reservation="${RESERVATION}"} scripts/slurm/run_sdg_split.sbatch
