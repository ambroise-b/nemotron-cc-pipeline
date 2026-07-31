#!/usr/bin/env python3
"""Fan a directory of .jsonl files out into N shard directories of symlinks
(round-robin by file count, no size accounting), so downstream per-shard
processing (e.g. step_2c-substring_dedup) never needs to copy the data.

Writes only the file-mapping — output_path/shard_XXXX/<symlinks> — and
nothing else; a caller's own per-shard working data (cache/output/workdir)
belongs in a separate tree, not under here. Rerunning this script wipes and
rebuilds each shard_XXXX/ directory it writes, so it's safe to call again to
pick up new/changed input files, as long as that data lives elsewhere.

Pure stdlib — no nemo_curator / project dependency — so it runs with
whatever system python3 is on hand (login node or inside the container).
"""

import argparse
import shutil
from pathlib import Path


def list_jsonl_files(input_path):
    return sorted(str(p) for p in Path(input_path).rglob("*.jsonl") if p.is_file())


def build_shards(input_path, output_path, num_shards):
    files = list_jsonl_files(input_path)
    if not files:
        raise SystemExit(f"no .jsonl files found under {input_path}")

    shard_dirs = []
    for shard_id in range(num_shards):
        shard_dir = Path(output_path) / f"shard_{shard_id:04d}"
        shutil.rmtree(shard_dir, ignore_errors=True)
        shard_dir.mkdir(parents=True)
        shard_dirs.append(shard_dir)

    counts = [0] * num_shards
    for i, file_path in enumerate(files):
        shard_id = i % num_shards
        # Count-prefixed link name: files land round-robin from many source
        # subdirectories, so two picks in the same shard can share a basename.
        link_path = shard_dirs[shard_id] / f"{counts[shard_id]:06d}_{Path(file_path).name}"
        link_path.symlink_to(Path(file_path).resolve())
        counts[shard_id] += 1
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, help="Directory of .jsonl files to shard (searched recursively)")
    parser.add_argument("--output-path", required=True, help="Directory under which shard_XXXX/ symlink dirs are created")
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    counts = build_shards(args.input_path, args.output_path, args.num_shards)
    for shard_id, count in enumerate(counts):
        print(f"shard_{shard_id:04d}: {count} file(s)")
