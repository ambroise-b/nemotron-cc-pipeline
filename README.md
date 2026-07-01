# Nemotron-CC Pipeline

A local, SLURM/Ray-adapted copy of NVIDIA's [Nemotron-CC data curation
pipeline](https://docs.nvidia.com/nemotron/nightly/nemotron/data/curation/nemotron-cc.html),
which turns raw Common Crawl snapshots into curated pretraining text plus
LLM-generated synthetic training data.

## Provenance

The code under `src/nemotron-cc/` is vendored verbatim (Apache-2.0) from
[`NVIDIA-NeMo/Nemotron`](https://github.com/NVIDIA-NeMo/Nemotron), commit
[`993dd4f`](https://github.com/NVIDIA-NeMo/Nemotron/commit/993dd4f540ca9c397be0232b7233c256b16008a7)
(2026-06-24), path `src/nemotron/recipes/data/curation/nemotron-cc/`. See
`src/nemotron-cc/README.md` for NVIDIA's original documentation.

A few minimal modifications are made on top of the upstream files, all kept
to single-line diffs so it stays easy to diff against future upstream
updates:
- A `--slurm` flag added to each `step_*.py` script's CLI, switching
  `RayClient` to NeMo Curator's `SlurmRayClient` for multi-node runs — a
  one-line, NVIDIA-documented pattern (see [Curator's SLURM tutorial](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/slurm)).
- In `step_2c-substring_dedup/exact_substring_dedup.sh`, the hardcoded
  `INPUT_PATH`/`MAIN_CACHE_PATH`/`OUTPUT_PATH` placeholders are changed to
  `${VAR:-default}` so they're overridable via env vars instead of editing
  the script — upstream ships it as a "TODO: Update paths" template.
- Same file: `NUM_THREADS` changed from a hardcoded `128` to
  `${NUM_THREADS:-128}`, so it's overridable to match a node's real core
  count (288 on this cluster's GH200 nodes) instead of being stuck at
  upstream's example value.

Everything else — pipeline stages, executors, CLI args — is unchanged from
upstream.

## Pipeline stages

| Stage | Script | What it does | Resources |
|---|---|---|---|
| 1 | `step_1-download_extract.py` | Download CC WARC snapshots, extract text (JusText), language-ID + filter, Unicode cleanup | CPU-only |
| 2a | `step_2a-exact_dedup.py` | Hash-based exact deduplication (`--identify` / `--remove`) | GPU (identify), CPU (remove) |
| 2b | `step_2b-fuzzy_dedup.py` | MinHash + LSH fuzzy deduplication | GPU (identify), CPU (remove) |
| 2c | `step_2c-substring_dedup/` | Suffix-array exact substring dedup (needs Rust/cargo) | CPU-only |
| 3 | `step_3-quality_classification.py` | Ensemble quality classifiers → 0-19 buckets (`--classify` / `--ensemble`) | GPU classify, CPU ensemble |
| 4 | `step_4-sdg.py` | LLM synthetic data generation (diverse QA, distill, extract knowledge, knowledge list) on buckets 18-19 | GPU inference server |

## Setup and running

For the concrete, tested-on-cluster command flow (CSCS Clariden/Alps,
`infra01` account) — verifying the container, a single-node (4 GPU)
validation run, and multi-node — see
[`scripts/CSCS-QUICKSTART.md`](scripts/CSCS-QUICKSTART.md). Start there.

Runs inside the container defined in
[`container/container.toml`](container/container.toml) — NVIDIA's own
official `nemo-curator` image, which already ships a matched
torch/RAPIDS/`nemo_curator` stack, so no install step is needed for stages
1-3. If you ever need a different container that doesn't already have
`nemo-curator` installed: `nemo-curator`'s `torch` dependency is unpinned,
so a naive `pip`/`uv install` there installs a broken CPU-only torch — the
quickstart's "Background" section covers the fix (`uv sync`, never
`uv pip install`). `vllm` (only needed for stage 4) is split into an
optional `sdg` extra in `pyproject.toml` for the same reason.

Stage 2c additionally needs a Rust toolchain (`cargo`) — see
`src/nemotron-cc/step_2c-substring_dedup/README.md`.

For `sbatch`-based multi-node submission outside the CSCS-specific
quickstart, see `scripts/README.md`.

## Layout

```
src/nemotron-cc/     vendored pipeline code (upstream-mirrored, minus --slurm)
scripts/             SLURM submission scripts (new, cluster-specific)
container/           CSCS "--environment=" container definition
configs/             sample run configuration
data/                gitignored working/output directory
```
