# Nemotron-CC pipeline for Apertus

This repo adapts NVIDIA's [Nemotron-CC data-curation
pipeline](https://docs.nvidia.com/nemotron/nightly/nemotron/data/curation/nemotron-cc.html)
for the **Apertus** project. It turns raw Common Crawl into curated pretraining
text (and optional LLM-generated synthetic data), adding one Apertus-specific
step: **URL (robots.txt) + PII filtering** between extraction and deduplication,
so excluded domains and redacted PII never enter the curated dataset.

Two things are adapted on top of the upstream pipeline:

1. A **local-WARC extraction** entry point (`step_1-extract_local_warc.py`) for
   when the Common Crawl WARC files are already on the cluster (no download).
2. A **URL + PII filtering** stage (`src/url_pii_filter/`, built on
   [datatrove](https://github.com/huggingface/datatrove)), run as a SLURM array.

Everything else is NVIDIA's pipeline, run in place on CSCS Clariden/Alps via
SLURM + Ray. See [Provenance](#provenance) for exactly what is vendor vs ours.

## Running the pipeline

### 1. Clone (first time)

Run in place from `~/repos/nemotron_cc_pipeline`. Clone **with submodules** (the
robots.txt domain lists live in `modules/`):

```bash
mkdir -p ~/repos
git clone --recurse-submodules \
    https://github.com/ambroise-b/nemotron-cc-pipeline.git \
    ~/repos/nemotron_cc_pipeline
cd ~/repos/nemotron_cc_pipeline
# already cloned but modules/ is empty?
#   git submodule update --init --recursive
```

### 2. Run it

- **Manually / interactively (start here)** — the concrete, tested command flow
  on CSCS Clariden/Alps (`infra01`): verify the containers, then a single-node
  4-GPU validation run, then multi-node. See
  [`scripts/CSCS-QUICKSTART.md`](scripts/CSCS-QUICKSTART.md).
- **Automated multi-node (`sbatch`)** — one pre-filled launcher per stage (the
  **Submit script** column below). **Edit the variables at the top** of a
  launcher if the defaults don't fit (paths, run name, node counts), then `bash`
  it from the repo root — e.g. `bash scripts/submit_nemotron_cc/01_submit_download.sh`.
  Each launcher picks the right container automatically. See
  [`scripts/README.md`](scripts/README.md) for the shared `sbatch` template.

## Pipeline stages

Run in order; each stage's output feeds the next. Stage 1.5 is the Apertus
addition; stage 2c is **optional**.

| Stage | What it does | Submit script (under `scripts/`) |
|---|---|---|
| 1 | Get Common Crawl — download a snapshot **or** read pre-downloaded local WARC — extract text (JusText), language-ID + filter, Unicode cleanup | [`submit_nemotron_cc/01_submit_download.sh`](scripts/submit_nemotron_cc/01_submit_download.sh) **or** [`01_submit_extract_local_warc.sh`](scripts/submit_nemotron_cc/01_submit_extract_local_warc.sh) |
| **1.5** | **Apertus step:** drop documents whose domain is on the robots.txt exclusion list, redact emails / IPs / IBANs. Sharded SLURM array | [`submit_url_pii_filter/submit_url_pii_filter.sh`](scripts/submit_url_pii_filter/submit_url_pii_filter.sh) |
| 2a | Hash-based exact deduplication | [`submit_nemotron_cc/02a_submit_exact_dedup.sh`](scripts/submit_nemotron_cc/02a_submit_exact_dedup.sh) |
| 2b | MinHash + LSH fuzzy deduplication | [`submit_nemotron_cc/02b_submit_fuzzy_dedup.sh`](scripts/submit_nemotron_cc/02b_submit_fuzzy_dedup.sh) |
| 2c | Suffix-array exact substring dedup — **optional** (needs a Rust toolchain) | [`submit_nemotron_cc/02c_submit_substring_dedup.sh`](scripts/submit_nemotron_cc/02c_submit_substring_dedup.sh) |
| 3 | Ensemble quality classifiers → 0–19 buckets | [`submit_nemotron_cc/03_submit_quality_classification.sh`](scripts/submit_nemotron_cc/03_submit_quality_classification.sh) |
| 4 | LLM synthetic data generation on the top buckets | [`submit_nemotron_cc/04A1_submit_sdg.sh`](scripts/submit_nemotron_cc/04A1_submit_sdg.sh) (or [`04A2_submit_sdg_split.sh`](scripts/submit_nemotron_cc/04A2_submit_sdg_split.sh) for the split-node design) |

Stage 1.5 produces a schema-identical, filtered replica of stage 1's JSONL, so
stage 2a consumes it exactly as it would the unfiltered extraction output. Which
underlying `step_*.py` each stage runs, and its container, is in
[Provenance](#provenance) and [Containers](#containers) below.

### Containers

Two CSCS `--environment=` containers (see `container/`):

- [`container/container.toml`](container/container.toml) — NVIDIA's official
  `nemo-curator` image; all nemo_curator/Ray stages (1, 2, 3, 4).
- [`container/datatrove.toml`](container/datatrove.toml) — the
  `data-pipeline-pretrain` image; **only** stage 1.5 (datatrove).

## Layout

```
src/nemotron-cc/     NVIDIA's pipeline (vendored — see Provenance)
src/url_pii_filter/  stage 1.5: URL + PII filtering (ours, datatrove)
src/lib/             small shared helper (vendors PIIFormatter from a submodule)
scripts/slurm/       sbatch templates (run_stage / run_url_pii_filter / run_sdg_split)
scripts/submit_*/    per-stage submit launchers (ours)
container/           CSCS container definitions (nemo-curator + datatrove)
modules/             git submodules: robots.txt domain lists + PII formatter
runs/                generated shard lists (per URL+PII run; gitignored)
configs/             sample run configuration
data/, logs/         gitignored working/output dirs
```

## Provenance

The code under `src/nemotron-cc/` is vendored (Apache-2.0) from
[`NVIDIA-NeMo/Nemotron`](https://github.com/NVIDIA-NeMo/Nemotron), commit
[`993dd4f`](https://github.com/NVIDIA-NeMo/Nemotron/commit/993dd4f540ca9c397be0232b7233c256b16008a7)
(2026-06-24), path `src/nemotron/recipes/data/curation/nemotron-cc/`. Vendored
files are kept close to upstream, with only single-line diffs so they stay easy
to re-diff against future updates:

- a `--slurm` flag on each `step_*.py` (switches `RayClient` →
  `SlurmRayClient` for multi-node — NVIDIA's
  [documented pattern](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/slurm));
- in `step_2c-substring_dedup/exact_substring_dedup.sh`, the path placeholders
  and `NUM_THREADS` made overridable via `${VAR:-default}`.

Everything outside `src/nemotron-cc/` — `src/url_pii_filter/`, `src/lib/`,
`scripts/`, `container/` — is written for Apertus.

### `src/nemotron-cc/` — file by file

| File | Origin |
|---|---|
| `step_1-download_extract.py` | Vendor (+ `--slurm`) |
| `step_1-extract_local_warc.py` | **Ours (Apertus)** — variant of stage 1 that extracts from a pre-downloaded local WARC dump instead of downloading; reuses the same JusText extractor + downstream stages so output matches |
| `step_2a-exact_dedup.py` | Vendor (+ `--slurm`) |
| `step_2b-fuzzy_dedup.py` | Vendor (+ `--slurm`) |
| `step_2c-substring_dedup/` | Vendor (+ overridable paths / `NUM_THREADS`) |
| `step_3-quality_classification.py` | Vendor (+ `--slurm`) |
| `step_4-sdg.py` | Vendor (+ `--slurm`) |
| `README.md` | Vendor (NVIDIA's original stage documentation) |
