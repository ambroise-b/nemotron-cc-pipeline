#!/bin/bash
# =============================================================================
# Submit stage 4 (SDG) using the UNIFIED multi-node design: one Ray cluster
# across all nodes, vLLM served via Ray Serve (one replica per node), and the
# data pipeline on the same cluster. (Replaces the old split-node design; see
# scripts/slurm/run_sdg_split.sbatch header for why.)
#
# Run from the project root:
#   bash scripts/submit_nemotron_cc/04A2_submit_sdg_split.sh
#
# All the actual config (node count, model, TP, tasks) lives in
# scripts/slurm/run_sdg_split.sbatch itself — this is just a thin,
# consistently-located pointer to it; edit that file to change resources/model.
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

echo "Submitting stage 4, unified Ray Serve design (see scripts/slurm/run_sdg_split.sbatch for config)"
sbatch ${RESERVATION:+--reservation="${RESERVATION}"} scripts/slurm/run_sdg_split.sbatch
