#!/bin/bash
# =============================================================================
# Pre-download every HuggingFace artifact the pipeline needs into the shared
# cache ($HF_HUB_CACHE), so the actual stages can run fully offline
# (HF_HUB_OFFLINE=1, the default set in run_stage.sbatch / run_sdg_split.sbatch).
#
# Why: NeMo Curator's TokenizerStage.setup_on_node() calls
# snapshot_download(local_files_only=False) ONCE PER NODE. At ~50-node scale
# that per-node network round-trip to huggingface.co flakes (OSError: couldn't
# connect ... and couldn't find them in the cached files) and aborts the stage.
# Warming the cache once here + running offline removes that failure entirely.
#
# Runs ONLINE on a single node inside the container (matching the runtime's own
# huggingface_hub/transformers), so it writes exactly the cache layout the
# stages later read. Idempotent: already-present repos are skipped.
#
# Run from the project root:
#   sbatch scripts/slurm/warm_hf_cache.sbatch          # classifiers + Qwen (SDG)
#   INCLUDE_QWEN=0 sbatch scripts/slurm/warm_hf_cache.sbatch   # skip the 60GB Qwen
# =============================================================================

#SBATCH --job-name=warm-hf-cache
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/users/%u/repos/nemotron_cc_pipeline/logs/warm_hf_cache_%j.log
#SBATCH --error=/users/%u/repos/nemotron_cc_pipeline/logs/warm_hf_cache_%j.log

set -euo pipefail

REPO_DIR="${REPO_DIR:-/users/${USER}/repos/nemotron_cc_pipeline}"
CONTAINER_ENV="${CONTAINER_ENV:-${REPO_DIR}/container/container.toml}"

# Where models land. Must match what the stages use at runtime (your login env
# sets HF_HUB_CACHE=/capstor/.../hf_models). Fall back to HF_HOME/hub if unset.
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-${SCRATCH}/nemotron-cc-hf-cache}/hub}"
INCLUDE_QWEN="${INCLUDE_QWEN:-1}"

echo "=================================================="
echo "  Warming HF cache"
echo "  HF_HUB_CACHE : ${HF_HUB_CACHE}"
echo "  Include Qwen : ${INCLUDE_QWEN}"
echo "=================================================="

srun --label --environment="${CONTAINER_ENV}" --container-writable bash -c "
set -euo pipefail
# Force ONLINE for this job regardless of any inherited offline default.
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HUB_CACHE='${HF_HUB_CACHE}'
export INCLUDE_QWEN='${INCLUDE_QWEN}'
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download, hf_hub_download

# Full snapshots (weights + tokenizer + config) go via snapshot_download so the
# runtime's setup_on_node() finds a complete, up-to-date snapshot offline.
snapshots = [
    'nvidia/nemocurator-fineweb-mixtral-edu-classifier',
    'nvidia/nemocurator-fineweb-nemotron-4-edu-classifier',
]
if os.environ.get('INCLUDE_QWEN', '1') == '1':
    snapshots.append('Qwen/Qwen3-30B-A3B-Instruct-2507')

for repo in snapshots:
    print(f'>>> snapshot_download {repo}', flush=True)
    p = snapshot_download(repo_id=repo)
    print(f'    OK -> {p}', flush=True)

# fasttext is a single .bin fetched by hf_hub_download in step_3.
print('>>> hf_hub_download mlfoundations/fasttext-oh-eli5', flush=True)
p = hf_hub_download(
    repo_id='mlfoundations/fasttext-oh-eli5',
    filename='openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin',
)
print(f'    OK -> {p}', flush=True)

print('All requested artifacts are cached.', flush=True)
PY
"

echo "=================================================="
echo "  DONE — cache warmed at ${HF_HUB_CACHE}"
echo "=================================================="
