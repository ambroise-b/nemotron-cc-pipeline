"""Split a Common Crawl dump into N shard file-lists for a SLURM array.

Recursively lists every parquet file under --input-dir, splits them into
--num-shards balanced groups, and writes one paths file per shard under
runs/<name>/. Each shard file contains one parquet path per line, relative to
--input-dir — exactly the format datatrove's ParquetReader(paths_file=...)
expects.

The SLURM array then launches one task per shard: array task i reads
runs/<name>/shard_<i>.txt and processes only those files (see
url_pii_filter.py and scripts/slurm/run_url_pii_filter.sbatch).

This runs once, inside the submit job, before the array fans out. It depends
only on the Python standard library (no datatrove / fsspec / loguru), so it
stays cheap to launch.

Usage:
    python src/url_pii_filter/prepare_dumps.py \
        --input-dir /path/to/cc_parquets \
        --name my_run \
        --num-shards 100
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_dumps")

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Shard filenames are zero-padded to this width so lexical and numeric order
# agree; must match run_url_pii_filter.sbatch's printf width (5).
_SHARD_PAD = 5


def shard_filename(index: int) -> str:
    return f"shard_{index:0{_SHARD_PAD}d}.txt"


def human_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string (e.g. '1.3 GiB')."""
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < 1024 or unit == "PiB":
            return f"{size:.1f} {unit}"
        size /= 1024


def list_parquet_files(input_dir: Path, pattern: str) -> list:
    """Recursively list files matching pattern under input_dir.

    Returns (relative POSIX path, size_bytes) tuples, sorted by path. Local
    filesystem only (stdlib pathlib) — the dump is expected on a POSIX path.
    """
    out = []
    for p in input_dir.rglob(pattern):
        if p.is_file():
            out.append((p.relative_to(input_dir).as_posix(), p.stat().st_size))
    return sorted(out)


def split_into_shards(files: list, num_shards: int):
    """Balance files across num_shards buckets by total bytes.

    Greedy largest-processing-time: assign each file (largest first) to the
    shard with the least bytes so far. Balances wall-clock better than a
    round-robin by file count when file sizes vary.

    Returns (shards, shard_bytes) where shards[i] is a list of relative paths.
    """
    shards = [[] for _ in range(num_shards)]
    shard_bytes = [0] * num_shards
    for rel, size in sorted(files, key=lambda x: x[1], reverse=True):
        j = min(range(num_shards), key=lambda k: shard_bytes[k])
        shards[j].append(rel)
        shard_bytes[j] += size
    return shards, shard_bytes


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"--input-dir is not a directory: {input_dir}")

    run_dir = Path(args.runs_dir) / args.name
    if not run_dir.is_absolute():
        run_dir = _REPO_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Listing %s files under %s ...", args.pattern, input_dir)
    files = list_parquet_files(input_dir, args.pattern)
    if not files:
        raise SystemExit(f"No files matching {args.pattern} found under {input_dir}")

    num_shards = args.num_shards
    if num_shards > len(files):
        logger.warning(
            "--num-shards (%d) > number of files (%d); %d shard(s) will be empty "
            "(their array tasks are no-ops).",
            num_shards, len(files), num_shards - len(files),
        )

    shards, shard_bytes = split_into_shards(files, num_shards)
    for i, shard_files in enumerate(shards):
        (run_dir / shard_filename(i)).write_text("".join(f"{p}\n" for p in shard_files))

    total_bytes = sum(shard_bytes)
    manifest = {
        "name": args.name,
        "input_dir": str(input_dir),
        "num_shards": num_shards,
        "total_files": len(files),
        "total_bytes": total_bytes,
        "pattern": args.pattern,
        "shard_pad": _SHARD_PAD,
        "shards": [
            {"shard": shard_filename(i), "files": len(shards[i]), "bytes": shard_bytes[i]}
            for i in range(num_shards)
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Print the job list: one line per shard / array task, with its size.
    logger.info(
        "Prepared %d shards from %d files (%s) in %s",
        num_shards, len(files), human_size(total_bytes), run_dir,
    )
    logger.info("%-8s %-10s %12s  %s", "ARRAY_ID", "FILES", "SIZE", "SHARD_FILE")
    for i in range(num_shards):
        logger.info(
            "%-8d %-10d %12s  %s",
            i, len(shards[i]), human_size(shard_bytes[i]), shard_filename(i),
        )
    # Also emit the shard count on stdout so a caller can size the array if needed.
    print(num_shards)


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a Common Crawl dump into N shard file-lists for a SLURM array.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Dump root containing the parquet files (searched recursively).",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Run name; shard lists are written to <runs-dir>/<name>/.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="Number of shards / SLURM array tasks to split the dump into.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Base directory for run shard-lists (relative paths are under the repo root).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob pattern (matched recursively) for input files (e.g. '*.jsonl' or '*.parquet').",
    )
    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
