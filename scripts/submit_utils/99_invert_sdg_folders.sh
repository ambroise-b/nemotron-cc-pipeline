#!/usr/bin/env bash
#
# invert_sdg_folders.sh
#
# Permute the two top levels of an SDG output tree:
#   <shard>/<task>/*.parquet   ->   <task>/<shard>/*.parquet
#
# This is a pure folder-order swap: filenames are untouched, no flattening,
# no renaming, no collision risk, and it is trivially reversible.
#
# Source layout (SRC):
#   shard_0000/
#     distill/            *.parquet
#     diverse_qa/         *.parquet
#     extract_knowledge/  *.parquet
#     knowledge_list/     *.parquet
#   shard_0001/
#     ...
#
# Destination layout (DEST):
#   distill/
#     shard_0000/         *.parquet
#     shard_0001/         *.parquet
#     ...
#   diverse_qa/
#     ...

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configuration — edit these, or override on the command line via env vars.
# --------------------------------------------------------------------------- #

# Root that contains the shard_* folders.
SRC="${SRC:-/iopsstor/scratch/cscs/aborbely/nemotron-cc-pipeline-CC-MAIN-2019-04/sdg_output}"

# Where the permuted (task-first) tree is written.
DEST="${DEST:-/iopsstor/scratch/cscs/aborbely/nemotron-cc-pipeline-CC-MAIN-2019-04/data_ablations/sdg_inverted}"

# How to place each shard's task folder into the new layout:
#   cp       -> full independent copy (safe, but duplicates all the data)
#   hardlink -> recursive hardlinks, instant, no extra disk (same filesystem only)
#   symlink  -> symlink the whole folder, instant (breaks if SRC is moved/deleted)
MODE="${MODE:-cp}"

# Glob for the shard folders at the root of SRC.
SHARD_GLOB="${SHARD_GLOB:-shard_*}"

# Task folders to process. Leave empty to auto-detect every sub-directory
# found inside the first shard.
TASKS="${TASKS:-}"

# Set DRY_RUN=1 to print what would happen without touching anything.
DRY_RUN="${DRY_RUN:-0}"

# --------------------------------------------------------------------------- #
# Implementation
# --------------------------------------------------------------------------- #

log() { printf '%s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -d "$SRC" ] || die "SRC does not exist: $SRC"

# place SRC_DIR DEST_DIR  — reproduce SRC_DIR at DEST_DIR using the chosen MODE.
case "$MODE" in
  cp)       place() { cp -r -- "$1" "$2"; } ;;
  hardlink) place() { cp -al -- "$1" "$2"; } ;;
  symlink)  place() { ln -s -- "$1" "$2"; } ;;
  *) die "MODE must be one of: cp | hardlink | symlink (got '$MODE')" ;;
esac

# Collect shard directories (sorted).
mapfile -t SHARDS < <(find "$SRC" -mindepth 1 -maxdepth 1 -type d -name "$SHARD_GLOB" -printf '%f\n' | sort)
[ "${#SHARDS[@]}" -gt 0 ] || die "No shard folders matching '$SHARD_GLOB' under $SRC"

# Determine task list.
if [ -n "$TASKS" ]; then
  read -r -a TASK_ARR <<< "$TASKS"
else
  mapfile -t TASK_ARR < <(find "$SRC/${SHARDS[0]}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi
[ "${#TASK_ARR[@]}" -gt 0 ] || die "Could not determine any task folders"

log "SRC     : $SRC"
log "DEST    : $DEST"
log "MODE    : $MODE"
log "shards  : ${#SHARDS[@]} (${SHARDS[0]} .. ${SHARDS[-1]})"
log "tasks   : ${TASK_ARR[*]}"
log "dry-run : $DRY_RUN"
log "-----------------------------------------------------------------"

placed=0
skipped=0
for task in "${TASK_ARR[@]}"; do
  task_dest="$DEST/$task"
  [ "$DRY_RUN" = "1" ] || mkdir -p -- "$task_dest"

  for shard in "${SHARDS[@]}"; do
    src_dir="$SRC/$shard/$task"
    dest_dir="$task_dest/$shard"

    [ -d "$src_dir" ] || continue

    if [ -e "$dest_dir" ]; then
      log "  skip (exists): $dest_dir"
      skipped=$((skipped + 1))
      continue
    fi

    if [ "$DRY_RUN" = "1" ]; then
      log "  $MODE  $src_dir  ->  $dest_dir"
    else
      place "$src_dir" "$dest_dir"
    fi
    placed=$((placed + 1))
  done
done

log "-----------------------------------------------------------------"
log "Done. placed=$placed skipped=$skipped  (task/shard folders) into $DEST"
