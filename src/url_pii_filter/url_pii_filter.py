"""URL (robots.txt) + PII filtering of extracted Common Crawl documents.

A datatrove pipeline that runs *after* the Nemotron-CC extraction stage
(src/nemotron-cc/step_1-download_extract.py in --warc-dir mode) and produces a
schema-identical, filtered replica of its JSONL output:

  1. Reads step_1's JSONL (or parquet). When --paths-file is given, only the
     files listed there are read (one shard of a larger dump) — this is how the
     SLURM-array setup gives each task its own slice of work.
  2. Drops documents whose URL domain is on the robots.txt exclusion list
     (datatrove URLFilter); excluded docs are written to a `removed/` subdir.
  3. Redacts emails / IPs / IBANs in the text (PIIFormatter, from the
     apply_robots_txt_filter submodule) and records a `pii_count` in metadata.
  4. Writes the surviving docs back out with the SAME columns as the input
     (text, url, warc_id, source_id, language, file_name, ...) plus pii_count,
     as plain .jsonl — a drop-in replacement that the dedup steps read exactly
     like the unfiltered step_1 output.

Extraction is NOT done here: the raw HTML -> text step must match Nemotron-CC's
JusText extractor exactly, so it lives in step_1 (nemo_curator). This pass only
filters already-extracted documents.

Parallelism is intra-node (datatrove LocalPipelineExecutor, --tasks/--workers);
multi-node scale-out is one SLURM-array task per shard (see prepare_dumps.py).

Usage (single shard):
    python src/url_pii_filter/url_pii_filter.py \
        --input-dir  /path/to/step1_extracted \
        --paths-file runs/<name>/shard_00000.txt \
        --output-dir /path/to/filtered/shard_00000 \
        --tasks 32 --workers -1
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

# Make the repo root importable so `scripts.lib` (which vendors PIIFormatter
# from the submodule) resolves regardless of the working directory we launch
# from. Repo root = three levels up: src/url_pii_filter/<file> -> repo/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datatrove.executor.local import LocalPipelineExecutor  # noqa: E402
from datatrove.io import get_datafolder  # noqa: E402
from datatrove.pipeline.filters import URLFilter  # noqa: E402
from datatrove.pipeline.readers import JsonlReader, ParquetReader  # noqa: E402
from datatrove.pipeline.writers import JsonlWriter, ParquetWriter  # noqa: E402

from scripts.lib.robots_txt_filter import PIIFormatter  # noqa: E402

# Default robots.txt domain exclusion list, vendored in the
# create_robots_txt_filter_scalable submodule. Used if --robots-list is omitted.
_DEFAULT_ROBOTS_LIST = (
    _REPO_ROOT
    / "modules"
    / "create_robots_txt_filter_scalable"
    / "domains_out_crawls_2022_2026.txt"
)


def _make_schema_adapter(id_field: str):
    """Return a datatrove writer adapter that reproduces the input schema.

    datatrove's default writer nests everything under a "metadata" key and
    renames the id to "id". To keep the output a drop-in replica of step_1's
    JSONL (columns: text, url, warc_id, source_id, language, file_name, ...),
    we lift metadata back to the top level and re-emit the id under its
    original column name (id_field, e.g. "warc_id"). PIIFormatter's added
    `pii_count` rides along in metadata.

    NOTE: datatrove calls the adapter as `adapter(self, document)` — the writer
    instance is passed as the first argument — hence the `self` parameter.
    """

    def adapter(self, document):  # noqa: ANN001
        return {
            "text": document.text,
            id_field: document.id,
            **(document.metadata or {}),
        }

    return adapter


def _make_datafolder(path: str, storage_options: dict):
    """Resolve a path to a datatrove DataFolder, passing any fsspec storage options.

    datatrove's get_datafolder maps a (path, dict) tuple to
    DataFolder(path, **dict), which is how storage options are threaded through.
    """
    return get_datafolder((path, storage_options)) if storage_options else path


def _make_writer(path, filetype, compression, storage_options, adapter, name_prefix=""):
    """Build a datatrove writer for the requested output filetype.

    name_prefix lets every SLURM-array shard write into ONE flat output dir
    with collision-free filenames (e.g. "shard_00007_00000.jsonl"), so the
    downstream reader consumes a single flat directory rather than per-shard
    subdirectories.
    """
    folder = _make_datafolder(path, storage_options)
    ext = "jsonl" if filetype == "jsonl" else "parquet"
    output_filename = f"{name_prefix}${{rank}}.{ext}"
    if filetype == "jsonl":
        return JsonlWriter(
            folder, compression=compression, adapter=adapter, output_filename=output_filename
        )
    return ParquetWriter(
        folder, compression=compression or "zstd", adapter=adapter, output_filename=output_filename
    )


def build_pipeline(args: argparse.Namespace) -> list:
    """Build the datatrove pipeline stage list."""
    storage_options = json.loads(args.storage_options) if args.storage_options else {}

    # Load the robots.txt domain exclusion list into a set (domain-level filter).
    robots_path = Path(args.robots_list)
    if not robots_path.exists():
        raise FileNotFoundError(
            f"Robots list not found at {robots_path}. Supply one with --robots-list, "
            "or initialise the create_robots_txt_filter_scalable submodule "
            "(git submodule update --init --recursive)."
        )
    with open(robots_path, "r") as f:
        robots_domains = set(line.strip() for line in f if line.strip())
    logger.info(f"Loaded {len(robots_domains)} robots.txt domains from {robots_path}")

    input_folder = _make_datafolder(args.input_dir, storage_options)
    adapter = _make_schema_adapter(args.id_field)
    # Removed docs go to a SEPARATE tree (not under --output-dir), so the
    # downstream reader over --output-dir never re-includes dropped documents.
    removed_dir = args.removed_dir or f"{args.output_dir.rstrip('/')}_removed"

    # 1. Read step_1's extracted docs. When paths_file is set, only those files
    #    (relative to input-dir) are read. `url` and every other column beyond
    #    text_key/id_key land in document.metadata (and are re-emitted verbatim
    #    by the schema adapter on write).
    if args.input_filetype == "parquet":
        reader = ParquetReader(
            input_folder, paths_file=args.paths_file,
            text_key=args.text_field, id_key=args.id_field,
        )
    else:
        reader = JsonlReader(
            input_folder, paths_file=args.paths_file,
            text_key=args.text_field, id_key=args.id_field,
        )

    return [
        reader,
        # 2. Coarse-grained (domain-level) robots.txt filtering. Excluded docs
        #    are written out (same schema) so the removal is auditable.
        URLFilter(
            extra_domains=robots_domains,
            use_integrated_lists=args.use_integrated_lists,
            exclusion_writer=_make_writer(
                removed_dir, args.output_filetype, args.compression,
                storage_options, adapter, args.output_name_prefix,
            ),
        ),
        # 3. Redact emails / IPs / IBANs; adds `pii_count` to metadata.
        PIIFormatter(),
        # 4. Write the filtered replica (input schema + pii_count).
        _make_writer(
            args.output_dir, args.output_filetype, args.compression,
            storage_options, adapter, args.output_name_prefix,
        ),
    ]


def main(args: argparse.Namespace) -> None:
    logger.info("Starting URL + PII filtering pipeline")
    logger.info(f"  Input:      {args.input_dir} ({args.input_filetype})")
    logger.info(f"  Paths file: {args.paths_file or '(all files under input-dir)'}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Parallelism: tasks={args.tasks}, workers={args.workers}")

    pipeline = build_pipeline(args)
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=args.logging_dir,
    )
    executor.run()
    logger.info("Pipeline complete")


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply robots.txt (URL) + PII filtering to extracted Common Crawl documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory of step_1's extracted documents (the dump root).",
    )
    parser.add_argument(
        "--paths-file",
        type=str,
        default=None,
        help="File with one input path per line (relative to --input-dir) to process. "
        "Omit to read every matching file under --input-dir. Produced by prepare_dumps.py.",
    )
    parser.add_argument(
        "--input-filetype",
        type=str,
        default="jsonl",
        choices=["jsonl", "parquet"],
        help="Input format. 'jsonl' matches step_1's output.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write the filtered (kept) documents.",
    )
    parser.add_argument(
        "--removed-dir",
        type=str,
        default=None,
        help="Directory to write robots-excluded docs. Default: '<output-dir>_removed' "
        "(kept SEPARATE from --output-dir so downstream never re-reads dropped docs).",
    )
    parser.add_argument(
        "--output-name-prefix",
        type=str,
        default="",
        help="Prefix for output filenames (e.g. 'shard_00007_'). Lets every SLURM-array "
        "shard write into one flat --output-dir without filename collisions.",
    )
    parser.add_argument(
        "--robots-list",
        type=str,
        default=str(_DEFAULT_ROBOTS_LIST),
        help="Path to the newline-separated robots.txt domain exclusion list.",
    )
    parser.add_argument(
        "--logging-dir",
        type=str,
        default=None,
        help="Directory for datatrove executor/pipeline logs.",
    )

    # Field / format parity with step_1's JsonlWriter output
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Name of the column holding the document text.",
    )
    parser.add_argument(
        "--id-field",
        type=str,
        default="warc_id",
        help="Name of the column holding the document id (step_1 uses 'warc_id').",
    )
    parser.add_argument(
        "--output-filetype",
        type=str,
        default="jsonl",
        choices=["jsonl", "parquet"],
        help="Output format. Keep 'jsonl' to replicate step_1's output.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="none",
        help="Output compression ('none', 'gzip', 'zstd'). Default 'none' matches "
        "step_1's plain .jsonl output.",
    )

    # URL filtering
    parser.add_argument(
        "--use-integrated-lists",
        action="store_true",
        help="Also apply datatrove's built-in URL blocklists in addition to the robots list.",
    )

    # Intra-node parallelism (datatrove LocalPipelineExecutor)
    parser.add_argument(
        "--tasks",
        type=int,
        default=8,
        help="Number of datatrove tasks (parallel processes) within this node.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Concurrent workers. -1 means workers == tasks.",
    )

    # Cloud storage
    parser.add_argument(
        "--storage-options",
        type=str,
        default=None,
        help='JSON string of fsspec storage options for cloud I/O (e.g., \'{"key": "...", "secret": "..."}\').',
    )

    return parser


if __name__ == "__main__":
    _args = attach_args().parse_args()
    if _args.compression and _args.compression.lower() == "none":
        _args.compression = None
    main(_args)
