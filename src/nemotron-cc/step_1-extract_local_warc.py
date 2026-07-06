# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract and preprocess PRE-DOWNLOADED Common Crawl WARC for the Nemotron-CC pipeline.

A variant of step_1-download_extract.py for the case where the WARC files are
already on disk (no download). It reads a local folder of .warc.gz, extracts
text with the SAME nemo_curator JusText extractor the download path uses (so
extraction is byte-for-byte identical), then runs the identical language-ID,
language filtering, and unicode stages, and writes the same JSONL output.

Everything below the extraction stage is intentionally kept identical to
step_1-download_extract.py.
"""

import argparse
import ast
import json
import os
import pickle
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.core.client import RayClient, SlurmRayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.download.base.iterator import DocumentIterateExtractStage
from nemo_curator.stages.text.download.common_crawl.extract import CommonCrawlHTMLExtractor
from nemo_curator.stages.text.download.common_crawl.warc_iterator import CommonCrawlWarcIterator
from nemo_curator.stages.text.filters import ScoreFilter
from nemo_curator.stages.text.filters.fasttext import FastTextLangId
from nemo_curator.stages.text.modifiers.unicode import UnicodeReformatter
from nemo_curator.stages.text.modifiers import Modify
from nemo_curator.tasks import DocumentBatch, EmptyTask, FileGroupTask
from nemo_curator.tasks.utils import TaskPerfUtils
from nemo_curator.stages.text.io.writer import JsonlWriter

FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_MODEL_FILENAME = "lid.176.bin"


class LanguageFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Extract language codes from FastTextLangId scores, optionally filtering to specific languages.

    FastTextLangId produces scores in the format "[0.95, 'EN']" (stringified list).
    This stage parses that field and replaces it with just the language code.
    If target_languages is provided, only documents matching those languages are kept.
    """

    def __init__(
        self, target_languages: list[str] | None = None, language_field: str = "language"
    ) -> None:
        self.target_languages = (
            {lang.upper() for lang in target_languages} if target_languages else None
        )
        self.language_field = language_field
        self.name = "language_filter"

    def process(self, task: DocumentBatch) -> DocumentBatch | None:
        df = task.to_pandas()
        # Parse "[0.95, 'EN']" -> 'EN'
        df[self.language_field] = df[self.language_field].apply(lambda v: ast.literal_eval(v)[1])
        if self.target_languages:
            df = df[df[self.language_field].isin(self.target_languages)]
            if len(df) == 0:
                return None
        task.data = df
        return task


def list_warc_files(warc_dir: str, glob_pattern: str, storage_options: dict) -> list[str]:
    """Recursively resolve the exact WARC files to extract, under warc_dir.

    We do discovery ourselves rather than via nemo_curator's FilePartitioningStage
    because that stage matches only the FINAL filename extension (".gz"), so it
    cannot target the compound ".warc.gz" without also pulling in .warc.wet.gz /
    .warc.wat.gz and the robotstxt/crawldiagnostics WARCs. The default glob
    "**/warc/*.warc.gz" selects only the main-content WARCs in segments/*/warc/.
    Works for local and fsspec-supported remote paths.
    """
    fs, root = url_to_fs(warc_dir, **storage_options)
    # `**` matches any depth; fsspec returns protocol-stripped absolute paths.
    paths = fs.glob(f"{root.rstrip('/')}/{glob_pattern}")
    return sorted(paths)


@dataclass
class WarcFileGroupStage(ProcessingStage[EmptyTask, FileGroupTask]):
    """Emit FileGroupTasks from a precomputed list of WARC file paths.

    A drop-in replacement for FilePartitioningStage that takes the exact file
    list from list_warc_files() (see why there) and just chunks it into groups
    of `files_per_partition`. Mirrors FilePartitioningStage's fan-out contract
    (single worker producing all partitions; IS_FANOUT_STAGE) and the same
    FileGroupTask metadata (incl. `source_files` for deterministic write names).
    """

    files: list
    files_per_partition: int = 1
    dataset_name: str = "cc-local-warc"
    name: str = "warc_file_groups"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def ray_stage_spec(self) -> dict:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def num_workers(self) -> int | None:
        return 1

    def process(self, _: EmptyTask) -> list[FileGroupTask]:
        fpp = max(1, self.files_per_partition)
        partitions = [self.files[i : i + fpp] for i in range(0, len(self.files), fpp)]
        return [
            FileGroupTask(
                task_id=f"{self.dataset_name}_{i}",
                dataset_name=self.dataset_name,
                data=file_group,
                _metadata={
                    "partition_index": i,
                    "total_partitions": len(partitions),
                    "source_files": file_group,
                },
                reader_config={},
            )
            for i, file_group in enumerate(partitions)
        ]


def download_fasttext_model(model_dir: str) -> str:
    """Download the FastText language identification model if not already present.

    Args:
        model_dir: Directory that should contain the FastText model file.

    Returns:
        The full path to the model file.
    """
    model_path = os.path.join(model_dir, FASTTEXT_MODEL_FILENAME)

    if os.path.exists(model_path):
        logger.info(f"FastText model already exists at {model_path}")
        return model_path

    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Downloading FastText language ID model to {model_path}")
    urllib.request.urlretrieve(FASTTEXT_MODEL_URL, model_path)  # noqa: S310
    logger.info("Download complete")
    return model_path


def create_pipeline(args: argparse.Namespace) -> Pipeline:
    """Build the download-extract-preprocess pipeline."""
    output_dir = args.output_dir
    cache_dir = str(Path(args.cache_dir).resolve())
    model_dir = os.path.join(cache_dir, "model")

    # Ensure FastText model is available locally (downloads if missing)
    fasttext_model_path = download_fasttext_model(model_dir)

    storage_options = json.loads(args.storage_options) if args.storage_options else {}

    # Recursively resolve the exact WARC files up-front (nested subdirs) — see
    # list_warc_files for why we don't let FilePartitioningStage discover them.
    warc_files = list_warc_files(args.warc_dir, args.warc_glob, storage_options)
    if args.file_limit is not None:
        warc_files = warc_files[: args.file_limit]
    if not warc_files:
        raise SystemExit(
            f"No files matching '{args.warc_glob}' found under {args.warc_dir}"
        )
    logger.info(f"Found {len(warc_files)} WARC file(s) under {args.warc_dir}")

    stages = [
        # 1a. Group the local, pre-downloaded WARC files into partitions
        #     (one FileGroupTask per `files_per_partition` files) — no download.
        WarcFileGroupStage(
            files=warc_files,
            files_per_partition=args.warc_files_per_partition,
        ),
        # 1b. Iterate WARC records and extract text with JusText — the SAME
        #     nemo_curator extractor the download path uses, so extraction is
        #     byte-for-byte identical to step_1-download_extract.py.
        #     max_calls_per_worker=2 mirrors nemo_curator's JusText OOM mitigation.
        DocumentIterateExtractStage(
            iterator=CommonCrawlWarcIterator(),
            extractor=CommonCrawlHTMLExtractor("justext"),
            add_filename_column=True,
            record_limit=args.record_limit,
            max_calls_per_worker=2,
        ),
        # 2. Language identification using FastText lid.176.bin (threshold 0.3 per paper).
        ScoreFilter(
            FastTextLangId(
                model_path=fasttext_model_path,
                min_langid_score=0.3,
            ),
            score_field="language",
        ),
        # 3. Extract language code, optionally filter to requested languages.
        LanguageFilter(
            target_languages=args.languages,
            language_field="language",
        ),
        # 4. Fix unicode issues on all documents.
        Modify(UnicodeReformatter()),
        # 5. Write output
        JsonlWriter(
            output_dir, write_kwargs={"storage_options": storage_options, "force_ascii": False}
        ),
    ]

    return Pipeline(
        name="nemotron-cc-extract-local-warc",
        description="Extract (JusText) and preprocess pre-downloaded Common Crawl WARC with language ID and unicode fixing.",
        stages=stages,
    )


def main(args: argparse.Namespace) -> None:
    storage_options = json.loads(args.storage_options) if args.storage_options else {}
    fs, fs_path = url_to_fs(args.output_dir, **storage_options)
    fs.mkdirs(fs_path, exist_ok=True)
    cache_dir = str(Path(args.cache_dir).resolve())
    os.makedirs(cache_dir, exist_ok=True)

    ray_client = SlurmRayClient(num_cpus=args.num_cpus) if args.slurm else RayClient(num_cpus=args.num_cpus)
    ray_client.start()

    logger.info("Starting Nemotron-CC local-WARC extraction pipeline")
    logger.info(f"  WARC dir: {args.warc_dir}")
    logger.info(f"  Languages: {args.languages or 'all'}")
    logger.info(f"  Cache dir: {cache_dir}")
    logger.info(f"  Output dir: {args.output_dir}")
    if args.file_limit is not None:
        logger.info(f"  File limit: {args.file_limit}")
    if args.record_limit is not None:
        logger.info(f"  Record limit: {args.record_limit}")

    pipeline = create_pipeline(args)
    logger.info(f"\n{pipeline.describe()}")

    executor = RayDataExecutor()

    start_time = time.perf_counter()
    results = pipeline.run(executor=executor)
    elapsed = time.perf_counter() - start_time

    total_documents = sum(task.num_items for task in results) if results else 0
    logger.info(f"Pipeline completed in {elapsed:.1f}s")
    logger.info(f"Total output files: {total_documents}")

    # Dump result tasks (with _stage_perf timing stats) for later analysis
    results_file = os.path.join(cache_dir, "results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Task results saved to {results_file}")

    # Aggregate and save per-stage metrics (mean/std/sum for each metric)
    metrics = TaskPerfUtils.aggregate_task_metrics(results)
    metrics_file = os.path.join(cache_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Aggregated metrics saved to {metrics_file}")

    ray_client.stop()


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and preprocess pre-downloaded Common Crawl WARC for Nemotron-CC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Local WARC input
    parser.add_argument(
        "--warc-dir",
        type=str,
        required=True,
        help="Root directory of the pre-downloaded Common Crawl dump (searched with --warc-glob).",
    )
    parser.add_argument(
        "--warc-glob",
        type=str,
        default="**/warc/*.warc.gz",
        help="Recursive glob (relative to --warc-dir) selecting the WARC files to extract. "
        "Default targets main-content WARCs (segments/*/warc/*.warc.gz), excluding "
        "wet/wat/robotstxt/crawldiagnostics. Use '**/*.warc.gz' to include everything named *.warc.gz.",
    )
    parser.add_argument(
        "--warc-files-per-partition",
        type=int,
        default=1,
        help="Number of WARC files per partition/task (extraction parallelism granularity).",
    )

    # Paths
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/cleaned_extracted",
        help="Directory to write the preprocessed extracted content.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data/cache",
        help="Cache directory for intermediate files. Layout: cache_dir/model (FastText model), plus results.pkl and metrics.json.",
    )

    # Limits (useful for testing)
    parser.add_argument(
        "--file-limit",
        type=int,
        default=None,
        help="Limit number of WARC files processed (useful for testing).",
    )
    parser.add_argument(
        "--record-limit",
        type=int,
        default=None,
        help="Limit number of records to extract per WARC file (useful for testing).",
    )

    # Language filtering
    parser.add_argument(
        "--languages",
        nargs="+",
        type=str,
        default=None,
        help="Language codes to keep (e.g., EN DE FR). If omitted, all languages are written.",
    )
    # Cloud storage
    parser.add_argument(
        "--storage-options",
        type=str,
        default=None,
        help='JSON string of fsspec storage options for cloud output paths (e.g., \'{"key": "...", "secret": "..."}\').',
    )

    # Ray cluster
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs for the Ray cluster (default: all available).",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Use SlurmRayClient to form a multi-node Ray cluster from the SLURM allocation "
        "(set when running via srun inside an sbatch script; see scripts/).",
    )

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
