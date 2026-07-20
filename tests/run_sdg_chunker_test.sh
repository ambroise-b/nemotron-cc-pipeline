#!/bin/bash
# =============================================================================
# Run the streaming-chunker equivalence/memory test on a debug node.
#
# Direct srun (no sbatch): stands up the nemo-curator container CPU-only (no
# vLLM), runs tests/test_sdg_chunker.py which compares the stock preprocessing
# against the streaming chunker in-process. Fast — meant for the debug partition.
#
# Run from the project root, e.g.:
#   # equivalence on 2 files of shard 0
#   bash tests/run_sdg_chunker_test.sh --shard-id 0 --num-files 2
#   # chunker-only on the OOM-prone shard (stock path would blow up)
#   MEM=850000 bash tests/run_sdg_chunker_test.sh \
#       --shard-id 1 --num-files 39 --new-only
#
# Any args after the script are forwarded to the test (see its --help). The
# --input-dir defaults to the standard bucketed_results path below.
# =============================================================================
set -euo pipefail

REPO_DIR="/users/${USER}/repos/nemotron_cc_pipeline"
: "${SCRATCH:?SCRATCH not set — expected on CSCS Clariden/Alps}"
DATA_DIR="${SCRATCH}/nemotron-cc-pipeline-CC-MAIN-2026-21"

# --- Slurm sizing (override via env) -----------------------------------------
PARTITION="${PARTITION:-debug}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
CPUS="${CPUS:-64}"
MEM="${MEM:-200000}"          # MB; raise (e.g. 850000) for --new-only on the bad shard

CONTAINER_ENV="${CONTAINER_ENV:-${REPO_DIR}/container/container.toml}"
export HF_HOME="${HF_HOME:-${SCRATCH}/nemotron-cc-hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Default the input dir if the caller didn't pass one.
TEST_ARGS=("$@")
if [[ ! " $* " == *" --input-dir "* ]]; then
    TEST_ARGS=(--input-dir "${DATA_DIR}/quality_labeling/bucketed_results" "$@")
fi

echo "Running chunker test on ${PARTITION} (cpus=${CPUS}, mem=${MEM}MB): ${TEST_ARGS[*]}"

srun --account=infra01 --partition="${PARTITION}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" --mem="${MEM}" \
    --time="${TIME_LIMIT}" \
    --environment="${CONTAINER_ENV}" --container-writable \
    bash -c "
export HF_HOME='${HF_HOME}'
export HF_HUB_CACHE='${HF_HUB_CACHE:-}'
export HF_HUB_OFFLINE='${HF_HUB_OFFLINE}'
export TRANSFORMERS_OFFLINE='${TRANSFORMERS_OFFLINE}'
python3 '${REPO_DIR}/tests/test_sdg_chunker.py' ${TEST_ARGS[*]@Q}
"
