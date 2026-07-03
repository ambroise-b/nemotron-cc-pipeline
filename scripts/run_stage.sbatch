#!/bin/bash
# =============================================================================
# Nemotron-CC pipeline — generic SLURM submit script
#
# Runs any one of the src/nemotron-cc/step_*.py scripts across a multi-node
# SLURM allocation using NeMo Curator's SlurmRayClient (see
# src/nemotron-cc/README.md and https://github.com/NVIDIA-NeMo/Curator ->
# tutorials/slurm/ for background on how SlurmRayClient works: srun launches
# one process per node, node 0 becomes the Ray head, the rest join as workers
# and block until the head's pipeline finishes).
#
# Runs stage scripts directly against the container's own system Python —
# the default container (container/container.toml) is NVIDIA's own official
# nemo-curator image, which already ships a matched torch/RAPIDS/
# nemo_curator stack (confirmed live end-to-end), so no install step is
# needed. If you ever need a different container that doesn't already have
# nemo-curator installed, see scripts/CSCS-QUICKSTART.md's "Background"
# section — that's a real trap (unpinned torch resolves to a broken
# CPU-only build), not something to casually improvise around.
#
# Prerequisites:
#   - None — this repo (/users/aborbely/repos/nemotron_cc_pipeline) is on a
#     shared filesystem already visible from every compute node, so it's
#     run in place; nothing needs to be copied anywhere. Only the actual
#     data being processed lives on $SCRATCH — see scripts/CSCS-QUICKSTART.md.
#     Not repo-location-agnostic: REPO_DIR and --output/--error below are
#     this repo's actual absolute path, not derived at runtime (sbatch
#     copies scripts into a per-job spool file, so $0 isn't reliable here —
#     see REPO_DIR's own comment). If this repo ever moves, update those
#     three lines (or pass REPO_DIR= as an override).
#
# Usage:
#   STEP_SCRIPT=src/nemotron-cc/step_1-download_extract.py \
#   STEP_ARGS="--start-snapshot 2024-46 --end-snapshot 2024-46 --url-limit 2" \
#   sbatch scripts/run_stage.sbatch
#
# Override resources without editing this file:
#   sbatch --nodes=4 --time=02:00:00 scripts/run_stage.sbatch
#
# Defaults below target a single-node run on the "debug" partition under the
# "infra01" account, claiming a full GH200 node (288 CPUs, ~870GB RAM, 4
# GPUs — every node on this cluster has GPUs, confirmed via `scontrol show
# node`) since --gpus-per-node=4 already makes the node exclusively ours
# regardless — `--exclusive`/`--mem`/`--cpus-per-task` just make that
# explicit instead of getting whatever Slurm's default per-CPU memory ratio
# happens to grant. --mem is set a little under the node's 870000MB to
# leave headroom for the OS/slurmd. Override --partition/--account/--nodes
# on the sbatch (or salloc, for interactive runs — see
# scripts/CSCS-QUICKSTART.md) command line for other clusters/accounts.
# =============================================================================

#SBATCH --job-name=nemotron-cc
#SBATCH --partition=debug
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=850000
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=00:30:00
#SBATCH --output=/users/aborbely/repos/nemotron_cc_pipeline/logs/nemotron_cc_%j.log
#SBATCH --error=/users/aborbely/repos/nemotron_cc_pipeline/logs/nemotron_cc_%j.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Required: which stage script to run, and its CLI arguments. STEP_ARGS may
# be empty for a ".sh" step (e.g. step_2c, which takes its paths via env
# vars instead — see its own submit script).
# ---------------------------------------------------------------------------
: "${STEP_SCRIPT:?Set STEP_SCRIPT to one of src/nemotron-cc/step_*.py (see README)}"
if [[ "${STEP_SCRIPT}" != *.sh ]]; then
    : "${STEP_ARGS:?Set STEP_ARGS to the CLI arguments for that stage}"
fi
STEP_ARGS="${STEP_ARGS:-}"

# Optional: working directory to `cd` into before running STEP_SCRIPT
# (step_2c-substring_dedup/exact_substring_dedup.sh calls other scripts by
# relative path, so it must run from its own directory).
STEP_WORKDIR="${STEP_WORKDIR:-}"

# Hardcoded rather than derived from $0: sbatch typically copies the script
# into a per-job spool file before executing it, so $0 at runtime often
# isn't this repo's path at all (a real, common sbatch footgun).
REPO_DIR="${REPO_DIR:-/users/aborbely/repos/nemotron_cc_pipeline}"

# Shared directory for Ray port broadcast — must be visible to ALL nodes.
# /tmp is typically node-local on HPC clusters, so use the shared filesystem.
export RAY_PORT_BROADCAST_DIR="${REPO_DIR}/logs"
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"

# Path to a CSCS "--environment=" container definition (see
# scripts/CSCS-QUICKSTART.md). Every srun call below runs inside it.
CONTAINER_ENV="${CONTAINER_ENV:-${REPO_DIR}/container/container.toml}"
CONTAINER_FLAGS=(--environment="${CONTAINER_ENV}" --container-writable)

# HF model weights (stage 4/SDG downloads these automatically, tens of GB
# for the default model) are large — keep them on $SCRATCH rather than
# /users (small home quota, already hit once this project).
export HF_HOME="${HF_HOME:-${SCRATCH:-${REPO_DIR}}/nemotron-cc-hf-cache}"

RUN_PYTHON="python3"

echo "=================================================="
echo "  Nemotron-CC — SLURM stage run"
echo "=================================================="
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Nodes     : ${SLURM_JOB_NODELIST} (${SLURM_JOB_NUM_NODES} nodes)"
echo "  GPUs/node : ${SLURM_GPUS_ON_NODE:-none}"
echo "  Stage     : ${STEP_SCRIPT}"
echo "  Args      : ${STEP_ARGS}"
echo "  Repo dir  : ${REPO_DIR}"
echo "  Container : ${CONTAINER_ENV}"
echo "=================================================="

mkdir -p "${REPO_DIR}/logs"

RUN_DIR="${REPO_DIR}${STEP_WORKDIR:+/${STEP_WORKDIR}}"

# A `.sh` step is a plain shell script (e.g. step_2c-substring_dedup), not a
# nemo_curator/Ray script — it has no --slurm/SlurmRayClient support, and
# running one copy per node (this script's normal one-task-per-node model)
# would have every node concurrently rm-rf/rebuild the *same* shared output
# paths, corrupting them. So: no --slurm flag, and force it onto a single
# task on a single node regardless of how many nodes were allocated — extra
# nodes just sit idle for this step (harmless, only wasteful).
SRUN_TASK_FLAGS=(--ntasks-per-node=1)
SLURM_FLAG=""
RUN_CMD="${RUN_PYTHON} '${REPO_DIR}/${STEP_SCRIPT}'"
if [[ "${STEP_SCRIPT}" == *.sh ]]; then
    SRUN_TASK_FLAGS=(--nodes=1 --ntasks=1)
    RUN_CMD="bash '${REPO_DIR}/${STEP_SCRIPT}'"
    if [[ "${SLURM_JOB_NUM_NODES}" -gt 1 ]]; then
        echo "NOTE: ${STEP_SCRIPT} has no multi-node support — running on 1 of the ${SLURM_JOB_NUM_NODES} allocated nodes; the rest sit idle for this step."
    fi
elif [[ "${SLURM_JOB_NUM_NODES}" -gt 1 ]]; then
    # --slurm is only meaningful (and only added) for multi-node jobs; a
    # single-node allocation runs fine with the scripts' default RayClient.
    SLURM_FLAG="--slurm"
fi

srun \
    --label \
    "${SRUN_TASK_FLAGS[@]}" \
    "${CONTAINER_FLAGS[@]}" \
    bash -c "
cd '${RUN_DIR}'
export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}
export RAY_PORT_BROADCAST_DIR='${REPO_DIR}/logs'
export HF_HOME='${HF_HOME:-}'
echo \"[\$(hostname)] SLURM_NODEID=\${SLURM_NODEID} python=\$(${RUN_PYTHON} --version 2>&1)\"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
    | sed \"s/^/  [\$(hostname)] GPU /\" || echo \"  [\$(hostname)] no GPUs\"
${RUN_CMD} ${STEP_ARGS} ${SLURM_FLAG}
"

echo "=================================================="
echo "  DONE"
echo "=================================================="
