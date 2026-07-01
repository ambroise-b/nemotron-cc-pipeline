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
# By default this runs stage scripts directly against the container's own
# system Python — the default container (container/container.toml) is
# NVIDIA's own official nemo-curator image, which already ships a matched
# torch/RAPIDS/nemo_curator stack, so no install step is needed. Set
# INSTALL_VENV=1 to fall back to building a disposable `uv sync` venv in a
# job-scoped tmp directory instead (for a different container that doesn't
# already have nemo-curator installed — see scripts/CSCS-QUICKSTART.md for
# why plain `uv pip install`/an unpinned `torch` requirement is a trap
# there).
#
# Prerequisites:
#   - None — this repo (/users/aborbely/repos/nemotron_cc_pipeline) is on a
#     shared filesystem already visible from every compute node, so it's
#     run in place; nothing needs to be copied anywhere. Only the actual
#     data being processed lives on $SCRATCH — see scripts/CSCS-QUICKSTART.md.
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
#SBATCH --output=logs/nemotron_cc_%j.log
#SBATCH --error=logs/nemotron_cc_%j.log

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

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

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

# ---------------------------------------------------------------------------
# Python to run stage scripts with. Default: the container's own system
# Python (nemo-curator/torch/RAPIDS already installed there — nothing to
# do). Set INSTALL_VENV=1 for a different container that needs its
# dependencies installed via `uv sync` into a disposable job-scoped venv.
# ---------------------------------------------------------------------------
RUN_PYTHON="python3"
EXTRA_SETUP_LOG=""

if [[ "${INSTALL_VENV:-0}" == "1" ]]; then
    JOB_TMP_DIR="${JOB_TMP_DIR:-${SCRATCH:-${REPO_DIR}}/tmp/nemotron-cc-${SLURM_JOB_ID}}"
    mkdir -p "${JOB_TMP_DIR}"
    if [[ "${KEEP_JOB_TMP_DIR:-0}" != "1" ]]; then
        trap 'rm -rf "${JOB_TMP_DIR}"' EXIT
    fi

    export UV_CACHE_DIR="${JOB_TMP_DIR}/uv-cache"
    export UV_INSTALL_DIR="${JOB_TMP_DIR}/uv-bin"
    export UV_PROJECT_ENVIRONMENT="${JOB_TMP_DIR}/venv"
    export PATH="${UV_INSTALL_DIR}:${PATH}"

    if ! command -v uv >/dev/null 2>&1; then
        echo "uv not found — installing standalone uv into ${UV_INSTALL_DIR}"
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi

    cd "${REPO_DIR}"

    # system-site-packages so anything the container provides that we don't
    # install ourselves stays importable.
    uv venv --system-site-packages "${UV_PROJECT_ENVIRONMENT}" --python 3.12

    # `uv sync`, NOT `uv pip install -e .` — only `uv sync` honors the
    # pyproject.toml index routing that gets torch's real CUDA build
    # instead of a broken CPU-only default-PyPI one (see
    # scripts/CSCS-QUICKSTART.md's dependency gotcha).
    uv sync ${INSTALL_EXTRAS:+--extra "${INSTALL_EXTRAS}"}

    RUN_PYTHON="uv run --no-sync python"
    EXTRA_SETUP_LOG="  Venv      : ${UV_PROJECT_ENVIRONMENT}"
fi

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
[[ -n "${EXTRA_SETUP_LOG}" ]] && echo "${EXTRA_SETUP_LOG}"
echo "=================================================="

mkdir -p logs

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
    "${SRUN_TASK_FLAGS[@]}" \
    "${CONTAINER_FLAGS[@]}" \
    bash -c "
cd '${RUN_DIR}'
export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}
export RAY_PORT_BROADCAST_DIR='${REPO_DIR}/logs'
export UV_CACHE_DIR='${UV_CACHE_DIR:-}'
export UV_INSTALL_DIR='${UV_INSTALL_DIR:-}'
export UV_PROJECT_ENVIRONMENT='${UV_PROJECT_ENVIRONMENT:-}'
export HF_HOME='${HF_HOME:-}'
export PATH='${UV_INSTALL_DIR:-}:\${PATH}'
echo \"[\$(hostname)] SLURM_NODEID=\${SLURM_NODEID} python=\$(${RUN_PYTHON} --version 2>&1)\"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
    | sed \"s/^/  [\$(hostname)] GPU /\" || echo \"  [\$(hostname)] no GPUs\"
${RUN_CMD} ${STEP_ARGS} ${SLURM_FLAG}
"

echo "=================================================="
echo "  DONE"
echo "=================================================="
