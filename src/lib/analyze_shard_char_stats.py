#!/usr/bin/env python3
"""Per-shard character-count statistics for the SDG input parquet files.

Replicates the exact file-sharding stride used by step_4-sdg.py
(`all_files[shard_id::num_shards]`) so the numbers match what each array
task actually reads. For every shard it reports the mean / std / min / max
number of characters *per document*, plus the single largest document — the
metric that predicts the DocumentSplitter OOM (one pathological doc blows up
a single worker).

Reads only the `text` column, and only its utf8 lengths, so it stays cheap
even over ~19k files. Files are processed in a multiprocessing pool.

Example:
    python src/lib/analyze_shard_char_stats.py \
        --input-dir /iopsstor/scratch/cscs/aborbely/nemotron-cc-pipeline-CC-MAIN-2026-21/quality_labeling/bucketed_results \
        --num-shards 500 --buckets 18 19 --top 30 --workers 8
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from concurrent.futures import ProcessPoolExecutor

import pyarrow.compute as pc
import pyarrow.parquet as pq


def file_stats(path: str, text_field: str) -> dict:
    """Per-file aggregates of utf8 character length of `text_field`.

    Returns count, sum, sum-of-squares, min, max and the max document's char
    count. Everything stays in Arrow — no python strings materialized.
    """
    table = pq.read_table(path, columns=[text_field])
    col = table.column(text_field)
    lengths = pc.utf8_length(col)                 # int32 per row
    lengths = pc.cast(lengths, "int64")           # avoid overflow in sumsq
    n = len(lengths)
    if n == 0:
        return {"path": path, "n": 0, "sum": 0, "sumsq": 0, "min": 0, "max": 0,
                "sum_lines": 0, "max_lines": 0}
    total = pc.sum(lengths).as_py()
    sumsq = pc.sum(pc.multiply(lengths, lengths)).as_py()
    # Number of "\n" per document ≈ number of segments DocumentSplitter produces.
    # This is the real OOM predictor (row-count explosion), not char count.
    newlines = pc.cast(pc.count_substring(col, pattern="\n"), "int64")
    return {
        "path": path,
        "n": n,
        "sum": total,
        "sumsq": sumsq,
        "min": pc.min(lengths).as_py(),
        "max": pc.max(lengths).as_py(),
        "sum_lines": pc.sum(newlines).as_py(),
        "max_lines": pc.max(newlines).as_py(),
    }


def combine(stats: list[dict]) -> dict:
    """Merge per-file aggregates into one shard-level record."""
    n = sum(s["n"] for s in stats)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0, "max": 0,
                "mean_lines": 0.0, "max_lines": 0, "n_files": len(stats)}
    total = sum(s["sum"] for s in stats)
    sumsq = sum(s["sumsq"] for s in stats)
    mean = total / n
    var = max(sumsq / n - mean * mean, 0.0)       # clamp tiny negatives
    nonempty = [s for s in stats if s["n"] > 0]
    return {
        "n": n,
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(s["min"] for s in nonempty),
        "max": max(s["max"] for s in nonempty),
        "mean_lines": sum(s["sum_lines"] for s in stats) / n,
        "max_lines": max(s["max_lines"] for s in nonempty),
        "n_files": len(stats),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="bucketed_results directory")
    ap.add_argument("--num-shards", type=int, default=500)
    ap.add_argument("--buckets", type=int, nargs="+", default=[18, 19])
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--top", type=int, default=30, help="how many worst shards to print (by max lines/doc)")
    ap.add_argument("--csv", default=None, help="optional path to write full per-shard CSV")
    args = ap.parse_args()

    input_paths = []
    for bucket in args.buckets:
        d = os.path.join(args.input_dir, f"ensemble-max-int={bucket}")
        if os.path.isdir(d):
            input_paths.append(d)
        else:
            print(f"WARNING: bucket dir not found: {d}")
    if not input_paths:
        raise SystemExit(f"No bucket directories found in {args.input_dir} for {args.buckets}")

    # SAME ordering + stride as step_4-sdg.py
    all_files = sorted(f for d in input_paths for f in glob.glob(os.path.join(d, "*.parquet")))
    if not all_files:
        raise SystemExit(f"No .parquet files under {input_paths}")
    print(f"Found {len(all_files)} files across buckets {args.buckets}; num_shards={args.num_shards}")

    # Compute per-file stats once (each file belongs to exactly one shard).
    print(f"Scanning with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        per_file = list(ex.map(file_stats, all_files, [args.text_field] * len(all_files)))
    by_path = {s["path"]: s for s in per_file}

    # Group files into shards using the exact stride, then aggregate.
    rows = []
    for shard_id in range(args.num_shards):
        shard_files = all_files[shard_id::args.num_shards]
        if not shard_files:
            continue
        agg = combine([by_path[p] for p in shard_files])
        agg["shard_id"] = shard_id
        rows.append(agg)

    # Global view
    grand = combine(per_file)
    print("\n=== GLOBAL (all documents) ===")
    print(f"  docs={grand['n']:,}  chars: mean={grand['mean']:.0f} std={grand['std']:.0f} "
          f"min={grand['min']:,} max={grand['max']:,}  |  lines/doc: mean={grand['mean_lines']:.0f} "
          f"max={grand['max_lines']:,}")

    header = (f"{'shard':>6} {'files':>6} {'docs':>9} {'mean':>10} {'std':>10} {'min':>8} "
              f"{'max':>14} {'meanLines':>10} {'maxLines':>14}")

    def print_row(r: dict) -> None:
        print(f"{r['shard_id']:>6} {r['n_files']:>6} {r['n']:>9,} {r['mean']:>10.0f} "
              f"{r['std']:>10.0f} {r['min']:>8,} {r['max']:>14,} {r['mean_lines']:>10.0f} "
              f"{r['max_lines']:>14,}")

    rows.sort(key=lambda r: r["max_lines"], reverse=True)
    print(f"\n=== TOP {args.top} SHARDS BY MAX LINES/DOC (likely OOM offenders) ===")
    print(header)
    for r in rows[:args.top]:
        print_row(r)

    print(f"\n=== ALL {len(rows)} SHARDS (sorted by shard_id) ===")
    print(header)
    for r in sorted(rows, key=lambda r: r["shard_id"]):
        print_row(r)

    if args.csv:
        import csv
        fields = ["shard_id", "n_files", "n", "mean", "std", "min", "max", "mean_lines", "max_lines"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in sorted(rows, key=lambda r: r["shard_id"]):
                w.writerow({k: r[k] for k in fields})
        print(f"\nWrote full per-shard CSV to {args.csv}")


if __name__ == "__main__":
    main()
