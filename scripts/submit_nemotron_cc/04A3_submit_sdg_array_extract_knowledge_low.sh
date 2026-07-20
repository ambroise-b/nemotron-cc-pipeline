#!/bin/bash
# Memory-safe rerun of the stage 4 (SDG) "extract_knowledge" task, for shards that OOM'd at
# the default. Enables the streaming chunker (no per-line explosion) and caps
# MAX_CONCURRENT_REQUESTS=32 to reduce generation memory pressure. NUM_CPUS is
# left at the common default (64). All shared logic lives in
# 04A3_submit_sdg_array_common.sh.
#
# Rerun specific failed shards (mapping fixed by NUM_SHARDS, unchanged):
#   ARRAY_RANGE=1,249 bash scripts/submit_nemotron_cc/04A3_submit_sdg_array_extract_knowledge_low.sh
export TASK="extract_knowledge"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-32}"
export STREAMING_CHUNKER="${STREAMING_CHUNKER:-1}"
source "$(dirname "$0")/04A3_submit_sdg_array_common.sh"
