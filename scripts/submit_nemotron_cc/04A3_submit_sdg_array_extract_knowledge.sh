#!/bin/bash
# Submit the stage 4 (SDG) array for the "extract_knowledge" task. All shared
# sizing / submit logic lives in 04A3_submit_sdg_array_common.sh; this wrapper just picks the task.
#
# Run from the project root:
#   # all shards (the full run)
#   bash scripts/submit_nemotron_cc/04A3_submit_sdg_array_extract_knowledge.sh
#
#   # one shard (e.g. shard 0 — a smoke test over 1/NUM_SHARDS of the data)
#   ARRAY_RANGE=0 bash scripts/submit_nemotron_cc/04A3_submit_sdg_array_extract_knowledge.sh
#
#   # a range of shards (indices 0..9 inclusive)
#   ARRAY_RANGE=0-9 bash scripts/submit_nemotron_cc/04A3_submit_sdg_array_extract_knowledge.sh
#
#   # specific shards (e.g. rerun failed ones)
#   ARRAY_RANGE=3,17,42 bash scripts/submit_nemotron_cc/04A3_submit_sdg_array_extract_knowledge.sh
#
# ARRAY_RANGE selects WHICH indices run without changing the file->shard mapping
# (that is fixed by NUM_SHARDS). MAX_CONCURRENT caps how many run at once.
export TASK="extract_knowledge"
source "$(dirname "$0")/04A3_submit_sdg_array_common.sh"
