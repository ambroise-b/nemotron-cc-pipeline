"""Build shard download-lists for the original Nemotron-CC dataset.

Downloads Common Crawl's authoritative path index

    https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz

on every invocation (so the selection always reflects the current index),
filters it on the three partition fields encoded in each path

    contrib/Nemotron/Nemotron-CC/data-jsonl/quality=high/kind=actual/kind2=actual/CC-MAIN-2013-20-part-00016.jsonl.zstd
                                            ^^^^^^^^^^^^ ^^^^^^^^^^^ ^^^^^^^^^^^^

caps the selection at --max-files, and splits the result into --num-shards
balanced lists under runs/<name>/.

The SLURM array then launches one task per shard: array task i reads
runs/<name>/shard_<i>.txt and downloads only those files (see
scripts/slurm/run_download_original_nem_cc.sbatch).

Each shard line is two tab-separated columns:

    <remote path, as it appears in the index>\t<local path, relative to OUTPUT_DIR>

so the sbatch script needs no path logic of its own — it appends column 1 to
the base URL and writes to column 2.

This runs once on the host, before the array fans out, and depends only on the
Python standard library (no datatrove / fsspec / loguru) so it stays cheap to
launch.

Usage:
    python src/original_nem_cc/prepare_download_list.py \
        --name nemotron-cc_high_actual \
        --num-shards 20 \
        --quality high --kind actual --kind2 actual \
        --max-files 200
"""

import argparse
import gzip
import json
import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_download_list")

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Shard filenames are zero-padded to this width so lexical and numeric order
# agree; must match run_download_original_nem_cc.sbatch's printf width (5).
_SHARD_PAD = 5

_DEFAULT_INDEX_URL = (
    "https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz"
)
# Common prefix of every path in the index. Stripped from the local layout so
# the download tree starts at quality=<...>/ instead of four dead directories.
_DEFAULT_STRIP_PREFIX = "contrib/Nemotron/Nemotron-CC/data-jsonl/"

# The three partition fields, in the order they appear in a path.
_FIELDS = ("quality", "kind", "kind2")


def shard_filename(index: int) -> str:
    return f"shard_{index:0{_SHARD_PAD}d}.txt"


def fetch_index(url: str, cache_path: Path) -> list:
    """Download the gzipped path index and return its lines.

    The raw .gz is kept at cache_path for provenance (it is overwritten on
    every run — the index is the source of truth, not the cache).
    """
    logger.info("Downloading path index %s ...", url)
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed https URL
        raw = resp.read()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(raw)
    lines = [ln.strip() for ln in gzip.decompress(raw).decode().splitlines()]
    lines = [ln for ln in lines if ln]
    logger.info("Index: %d path(s), cached at %s", len(lines), cache_path)
    return lines


def parse_fields(path: str) -> dict:
    """Extract the quality / kind / kind2 values from a 'key=value' path.

    Returns a dict with a None value for any field absent from the path.
    """
    found = {f: None for f in _FIELDS}
    for part in path.split("/"):
        key, sep, value = part.partition("=")
        if sep and key in found:
            found[key] = value
    return found


def filter_paths(paths: list, selectors: dict) -> list:
    """Keep paths whose fields match every non-empty selector.

    An empty (or None) selector value means "any value for this field".
    Returns (path, fields) tuples, sorted by path.
    """
    active = {k: v for k, v in selectors.items() if v}
    out = []
    for path in paths:
        fields = parse_fields(path)
        if all(fields.get(k) == v for k, v in active.items()):
            out.append((path, fields))
    return sorted(out, key=lambda x: x[0])


def group_key(fields: dict) -> str:
    return "/".join(f"{f}={fields.get(f)}" for f in _FIELDS)


def cap_selection(selected: list, max_files: int, strategy: str) -> list:
    """Reduce selected to at most max_files entries.

    'spread' (default) round-robins across the distinct quality/kind/kind2
    groups, so a cap applied to a multi-group selection samples every group
    instead of exhausting the alphabetically first one. 'head' simply takes
    the first max_files paths in sorted order (contiguous part numbers within
    a single group).
    """
    if max_files <= 0 or len(selected) <= max_files:
        return selected
    if strategy == "head":
        return selected[:max_files]

    groups = {}
    for entry in selected:
        groups.setdefault(group_key(entry[1]), []).append(entry)
    out = []
    order = sorted(groups)
    i = 0
    while len(out) < max_files:
        # Groups are consumed in lockstep; exhausted ones are simply skipped.
        progressed = False
        for key in order:
            bucket = groups[key]
            if i < len(bucket):
                out.append(bucket[i])
                progressed = True
                if len(out) == max_files:
                    break
        if not progressed:
            break
        i += 1
    return sorted(out, key=lambda x: x[0])


def local_path(remote: str, strip_prefix: str) -> str:
    """Map a remote index path to its path relative to the output dir."""
    if strip_prefix and remote.startswith(strip_prefix):
        return remote[len(strip_prefix):]
    return remote


def split_into_shards(entries: list, num_shards: int) -> list:
    """Deal entries round-robin into num_shards buckets.

    Round-robin (not size-balanced like url_pii_filter's prepare_dumps.py):
    remote sizes are unknown before downloading, and Nemotron-CC parts are
    close enough in size that file count is a good proxy for wall clock.
    """
    shards = [[] for _ in range(num_shards)]
    for i, entry in enumerate(entries):
        shards[i % num_shards].append(entry)
    return shards


def main(args: argparse.Namespace) -> None:
    run_dir = Path(args.runs_dir) / args.name
    if not run_dir.is_absolute():
        run_dir = _REPO_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = fetch_index(args.index_url, run_dir / "data-jsonl.paths.gz")

    selectors = {"quality": args.quality, "kind": args.kind, "kind2": args.kind2}
    logger.info(
        "Selectors: %s",
        ", ".join(f"{k}={v or '<any>'}" for k, v in selectors.items()),
    )
    selected = filter_paths(paths, selectors)
    if not selected:
        available = sorted({group_key(parse_fields(p)) for p in paths})
        raise SystemExit(
            "No paths matched the selectors. Available combinations:\n  "
            + "\n  ".join(available)
        )
    logger.info("Matched %d of %d path(s)", len(selected), len(paths))

    if args.max_files > 0:
        capped = cap_selection(selected, args.max_files, args.select)
        if len(capped) < len(selected):
            logger.info(
                "Capped to %d file(s) with --select %s (dropped %d)",
                len(capped), args.select, len(selected) - len(capped),
            )
        selected = capped

    num_shards = min(args.num_shards, len(selected))
    if num_shards < args.num_shards:
        logger.warning(
            "--num-shards (%d) > selected files (%d); using %d shard(s) so no "
            "array task is a no-op.",
            args.num_shards, len(selected), num_shards,
        )

    entries = [(remote, local_path(remote, args.strip_prefix)) for remote, _ in selected]
    shards = split_into_shards(entries, num_shards)
    for i, shard_entries in enumerate(shards):
        (run_dir / shard_filename(i)).write_text(
            "".join(f"{remote}\t{local}\n" for remote, local in shard_entries)
        )

    per_group = {}
    for _, fields in selected:
        key = group_key(fields)
        per_group[key] = per_group.get(key, 0) + 1

    manifest = {
        "name": args.name,
        "index_url": args.index_url,
        "base_url": args.base_url,
        "selectors": selectors,
        "select": args.select,
        "max_files": args.max_files,
        "strip_prefix": args.strip_prefix,
        "index_paths": len(paths),
        "total_files": len(entries),
        "num_shards": num_shards,
        "shard_pad": _SHARD_PAD,
        "files_per_group": per_group,
        "shards": [
            {"shard": shard_filename(i), "files": len(shards[i])}
            for i in range(num_shards)
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("Prepared %d shard(s) for %d file(s) in %s", num_shards, len(entries), run_dir)
    for key in sorted(per_group):
        logger.info("  %-60s %6d file(s)", key, per_group[key])
    logger.info("%-8s %-10s  %s", "ARRAY_ID", "FILES", "SHARD_FILE")
    for i in range(num_shards):
        logger.info("%-8d %-10d  %s", i, len(shards[i]), shard_filename(i))
    # Emit the shard count on stdout so the submit script can size the array.
    print(num_shards)


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter the Nemotron-CC path index and split it into shard download-lists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Number of shards / SLURM array tasks to split the download into.",
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="",
        help="quality= partition to keep (e.g. high, medium-high, medium, medium-low, low). "
             "Empty = all.",
    )
    parser.add_argument(
        "--kind",
        type=str,
        default="",
        help="kind= partition to keep (actual or synthetic). Empty = all.",
    )
    parser.add_argument(
        "--kind2",
        type=str,
        default="",
        help="kind2= partition to keep (actual, distill, diverse_qa_pairs, extract_knowledge, "
             "knowledge_list, wrap_medium). Empty = all.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Cap on the number of files to download; 0 or negative = no cap.",
    )
    parser.add_argument(
        "--select",
        choices=("spread", "head"),
        default="spread",
        help="How --max-files picks files: 'spread' round-robins across the matched "
             "quality/kind/kind2 groups, 'head' takes the first N in sorted order.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Base directory for run shard-lists (relative paths are under the repo root).",
    )
    parser.add_argument(
        "--index-url",
        type=str,
        default=_DEFAULT_INDEX_URL,
        help="Gzipped path index, re-downloaded on every run.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://data.commoncrawl.org",
        help="Base URL the index paths are appended to (recorded in manifest.json).",
    )
    parser.add_argument(
        "--strip-prefix",
        type=str,
        default=_DEFAULT_STRIP_PREFIX,
        help="Prefix removed from each remote path to form the local layout.",
    )
    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
