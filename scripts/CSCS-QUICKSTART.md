# CSCS Clariden/Alps quickstart

Concrete, copy-paste command flow for this pipeline on CSCS's Clariden/Alps
cluster, using the `infra01` account and the container defined in
[`container/container.toml`](../container/container.toml) — NVIDIA's own
official `nemo-curator` image
(`nvidia+nemo-curator+26.04.sqsh`).

**Run every command in this doc yourself, inside your own interactive job.**
None of it should be run from a login node.

The repo itself lives at `/users/aborbely/repos/nemotron_cc_pipeline` and is
already visible from every compute node (shared filesystem) — it's run in
place, nothing gets copied anywhere. Only the actual data being processed
lives on `$SCRATCH`.

Because this is NVIDIA's own official `nemo-curator` build, it should
already ship a matched `torch`/RAPIDS/`nemo_curator` stack — no install
step needed, unlike the generic containers this doc used to target (see
"Background" below if you land on a different container and need that
history). **This hasn't been verified live yet** — Part 0 below is exactly
that verification, and it's cheap (no downloads, just imports).

---

## Part 0 — verify the container has what we need

Start an interactive job:

```bash
srun -A infra01 -p debug --gpus-per-node=4 --time=00:30:00 \
    --environment=/users/aborbely/repos/nemotron_cc_pipeline/container/container.toml \
    --pty bash
```

Inside that shell, check the pieces this pipeline needs, directly against
the container's system Python — no venv, no install:

```bash
python3 -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available(), torch.cuda.device_count(), 'GPU(s)')"
python3 -c "import cudf; print('cudf', cudf.__version__)"
python3 -c "import nemo_curator; print('nemo_curator', nemo_curator.__version__ if hasattr(nemo_curator, '__version__') else 'OK')"
python3 -c "from nemo_curator.core.client import RayClient, SlurmRayClient; print('RayClient/SlurmRayClient OK')"
python3 -c "import loguru, huggingface_hub, transformers, tiktoken; print('extra deps OK')"
python3 -c "import vllm; print('vllm', vllm.__version__)"   # only needed for stage 4
```

Expect all of these to print cleanly, with `cuda.is_available()` returning
`True` and device count `4`. If everything above works, **you're done with
setup** — go straight to Part 1, running stage scripts with plain `python3`
against the container's system install (no venv to activate).

If something's missing or `torch`/`cudf` look wrong, see "Fallback: this
container doesn't have everything" below.

---

## Part 1 — single node, 4 GPU (small-sample validation)

Reuse the same interactive job/shell from Part 0. Data reads/writes go
under `$SCRATCH`; the repo runs in place from `/users/...`:

```bash
cd /users/aborbely/repos/nemotron_cc_pipeline

DATA_DIR="$SCRATCH/nemotron-cc-data"
set -a; source configs/sample.env; set +a   # START_SNAPSHOT, URL_LIMIT, etc.

# Stage 1 — extract (CPU-only). TWO options; both write the same JSONL layout
# to $DATA_DIR/cleaned_extracted, so everything downstream is identical.
#
#   Option A — download + extract from Common Crawl by snapshot:
python3 src/nemotron-cc/step_1-download_extract.py \
    --start-snapshot "$START_SNAPSHOT" --end-snapshot "$END_SNAPSHOT" \
    --url-limit "$URL_LIMIT" --record-limit "$RECORD_LIMIT" --languages "$LANGUAGES" \
    --output-dir "$DATA_DIR/cleaned_extracted" --cache-dir "$DATA_DIR/cache/step1"
#
#   Option B — extract from WARC files already on the cluster (no download).
#   Uses the SAME JusText extractor, so output matches Option A byte-for-byte.
#   Reads --warc-dir recursively; --file-limit keeps the sample small.
#   WARC_DIR --> we only take part of it to actually test
WARC_DIR="/capstor/store/cscs/swissai/infra01/kpitas/common-crawl-CC-MAIN-2026-21/data/crawl-data/CC-MAIN-2026-21/segments/1778213377585.61"
python3 src/nemotron-cc/step_1-extract_local_warc.py \
    --warc-dir "$WARC_DIR" --file-limit 2 --languages "$LANGUAGES" \
    --output-dir "$DATA_DIR/cleaned_extracted" --cache-dir "$DATA_DIR/cache/step1"

# Stage 1.5 — URL (robots.txt) + PII filtering (CPU-only, datatrove). This is
# the SAME two-step flow the multi-node array uses (Part 2), run by hand on a
# single shard so you can validate it interactively.
#
#   (a) prepare — split step 1's extracted JSONL into ~50 shard lists under
#       runs/<RUN_NAME>/ (stdlib-only, runs right here; prints the shard count).
#       Use a throwaway run name for this test. NOTE: with a tiny 2-WARC sample
#       there are only a couple of .jsonl files, so most of the 50 shards will
#       be empty and only shard_00000 (maybe a few) will have work — that's fine.
PII_RUN=quickstart-test
python3 src/url_pii_filter/prepare_dumps.py \
    --input-dir "$DATA_DIR/cleaned_extracted" \
    --name "$PII_RUN" --num-shards 50 --pattern '*.jsonl'
#
#   (b) filter ONE shard (shard_00000): reads only the files listed in that
#       shard and writes a schema-identical, filtered replica. Removed docs go
#       to a SEPARATE dir so they aren't re-read by stage 2. --output-name-prefix
#       matches the array's flat-output convention. The robots domain list
#       defaults to the create_robots_txt_filter_scalable submodule
#       (override with --robots-list).
python3 src/url_pii_filter/url_pii_filter.py \
    --input-dir "$DATA_DIR/cleaned_extracted" \
    --paths-file "runs/$PII_RUN/shard_00000.txt" \
    --output-dir "$DATA_DIR/url_pii_filtered" \
    --removed-dir "$DATA_DIR/url_pii_removed" \
    --output-name-prefix shard_00000_ \
    --tasks 4

# Stage 2a — exact dedup (identify phase uses GPU). NOTE: the --remove
# writer nests a copy of --output-dir's own basename inside itself — actual
# output lands in "$DATA_DIR/exact_deduplicated/exact_deduplicated/", not
# flat under "$DATA_DIR/exact_deduplicated/" — confirmed live. The next
# stage's --input-dir below accounts for this.
python3 src/nemotron-cc/step_2a-exact_dedup.py --identify --remove \
    --input-dir "$DATA_DIR/url_pii_filtered" --cache-dir "$DATA_DIR/cache/exact_dedup" \
    --output-dir "$DATA_DIR/exact_deduplicated" --num-gpus 4

# Stage 2b — fuzzy dedup. Same nesting caveat applies to its own
# --output-dir (real data lands in ".../fuzzy_deduplicated/fuzzy_deduplicated/").
python3 src/nemotron-cc/step_2b-fuzzy_dedup.py --identify --remove \
    --input-dir "$DATA_DIR/exact_deduplicated/exact_deduplicated" --cache-dir "$DATA_DIR/cache/fuzzy_dedup" \
    --output-dir "$DATA_DIR/fuzzy_deduplicated" --num-gpus 4

# Stage 2c — substring dedup (CPU). This container has no Rust toolchain;
# install one once (persists under /users across jobs, like `uv`):
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#   source "$HOME/.cargo/env"
#
# Unlike the other stages, paths are set via env vars, not CLI args (see
# the script itself — one-line change from upstream so these are
# overridable, same spirit as the --slurm flag; see top-level README).
# It also calls prepare_dataset.py/make_suffix_array.py/etc by relative
# path, so it must be run with its own directory as the working directory
# — run it in a subshell so this doesn't change your shell's cwd.
# INPUT_PATH points at the nested dir per the stage 2b caveat above.
export INPUT_PATH="$DATA_DIR/fuzzy_deduplicated/fuzzy_deduplicated"
export MAIN_CACHE_PATH="$DATA_DIR/cache/substring_dedup"
export OUTPUT_PATH="$DATA_DIR/substring_deduplicated"
( cd src/nemotron-cc/step_2c-substring_dedup && bash exact_substring_dedup.sh )

# Stage 3 — quality classification
python3 src/nemotron-cc/step_3-quality_classification.py --classify --ensemble \
    --input-dir "$DATA_DIR/substring_deduplicated" --output-dir "$DATA_DIR/quality_labeling" --num-gpus 4

# Stage 4 — SDG (only if vllm import succeeded in Part 0)
python3 src/nemotron-cc/step_4-sdg.py --task diverse_qa \
    --input-dir "$DATA_DIR/quality_labeling/bucketed_results" --output-dir "$DATA_DIR/sdg_output"
```

Check `$DATA_DIR/quality_labeling/bucketed_results/` for the 0-19 bucket
output. On a tiny sample, don't be surprised if high buckets (18-19,
needed for stage 4) are empty or sparse — that's a real limitation of
testing on a handful of records, not a bug.

---

## Part 2 — multi-node

Interactive `--pty` only gives you one shell on one node, which doesn't fit
`SlurmRayClient`'s one-process-per-node model — so multi-node runs go
through `sbatch` (still your own submission, just not a `--pty` shell).
`scripts/run_stage.sbatch` defaults to this same container and runs stage
scripts directly against its system Python — nothing to install:

```bash
cd /users/aborbely/repos/nemotron_cc_pipeline

STEP_SCRIPT=src/nemotron-cc/step_1-download_extract.py \
STEP_ARGS="--start-snapshot 2024-46 --end-snapshot 2024-51 --output-dir $SCRATCH/nemotron-cc-data/cleaned_extracted --cache-dir $SCRATCH/nemotron-cc-data/cache/step1" \
sbatch -A infra01 -p normal --nodes=2 --gpus-per-node=4 --time=02:00:00 \
    scripts/run_stage.sbatch
```

`scripts/run_stage.sbatch` detects `--nodes > 1` and automatically adds
`--slurm` to the target script's CLI, switching it to `SlurmRayClient` so
all allocated nodes join one Ray cluster (node 0 = head, runs the pipeline;
the rest block as workers). Logs land in `logs/nemotron_cc_<jobid>.log` in
the repo itself:

```bash
tail -f logs/nemotron_cc_<jobid>.log
```

Swap `STEP_SCRIPT`/`STEP_ARGS` for any other stage the same way, following
the Part 1 examples above (with the same `$SCRATCH/nemotron-cc-data` paths).
Override the container with `CONTAINER_ENV=<path>` if needed.

### Ready-made submit scripts

`scripts/submit_default/` has pre-filled launchers (edit the variables at the
top, then `bash` them from the repo root):

- `01_submit_download.sh` — stage 1, **Option A** (download + extract by snapshot).
- `01b_submit_extract_local_warc.sh` — stage 1, **Option B** (extract from local
  WARC, no download). Pre-set to the CC-MAIN-2026-21 dump on `/capstor`, output
  to `$SCRATCH`.
- `02a_submit_exact_dedup.sh`, … — the later stages.

**URL + PII filtering runs between stage 1 and stage 2.** It's a CPU-only
datatrove job, so it goes through its own SLURM array (one task per shard)
rather than `run_stage.sbatch`:

```bash
bash scripts/submit_url_pii_filter/submit_url_pii_filter.sh
```

It first runs a lightweight prepare step **on the login node** (a stdlib-only
file-list + split — no SLURM job) that writes `NUM_SHARDS` shard lists under
`runs/<RUN_NAME>/` and reports each shard's size, then submits a single array
job (one task per shard) that filters each shard. All shards write into one
flat `$SCRATCH/nemotron-cc-data/url_pii_filtered/<RUN_NAME>` (removed docs go to
`.../url_pii_removed/<RUN_NAME>`). Point stage 2a's `--input-dir` at that
filtered directory.

---

## Fallback: this container doesn't have everything

If Part 0's verification fails (missing package, wrong torch build, no
CUDA), fall back to installing this repo's dependencies into a disposable
venv via `uv sync`:

```bash
export UV_PROJECT_ENVIRONMENT="$SCRATCH/nemotron-cc-venv/.venv"
export PATH="/users/$USER/.local/bin:$PATH"   # uv already installed here
uv venv --system-site-packages "$UV_PROJECT_ENVIRONMENT" --python 3.12
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
uv sync   # NOT `uv pip install -e .` — see "Background" below for why
```

For `sbatch`/multi-node with a non-default container, `run_stage.sbatch`
doesn't build a venv for you — it only ever runs stage scripts directly
against `python3`. Activate the venv above yourself (e.g. from a wrapper
`sbatch` script that sources it before calling `srun`), or ask to have that
support added back if you hit this in practice — it existed at one point
and was removed for simplicity once the default container was confirmed
sufficient, but the pattern is easy to restore if actually needed.

---

## Background: the dependency gotcha on generic containers

This only matters if you're not using the default container above, or the
fallback triggers. `nemo-curator`'s `torch` requirement is **unpinned**.
Plain PyPI `torch` is CPU-only for Linux, so an unqualified install pulled
in a CPU-only `torch==2.10.0` that crashed on import
(`AttributeError: module 'torch._C' has no attribute '_dlpack_exchange_api'`)
— confirmed live, on a generic NGC PyTorch container.

**Root cause, part 1**: NVIDIA's own `NVIDIA-NeMo/Curator` repo publishes a
`uv.lock` that pins the *same* `torch==2.10.0`, just as `2.10.0+cu129` — a
real CUDA build. Their `pyproject.toml` routes
`torch`/`torchvision`/`torchaudio` to PyTorch's own wheel index
(`https://download.pytorch.org/whl/cu129`, aarch64 included) via
`[[tool.uv.index]]`/`[tool.uv.sources]`, instead of default PyPI. This
repo's `pyproject.toml` has the same routing.

**Root cause, part 2**: `[tool.uv.sources]`/`[[tool.uv.index]]` are
*project-workflow* config — only honored by `uv sync`/`uv lock`/`uv run`,
**not** by `uv pip install`, which deliberately mirrors plain `pip` and
ignores that TOML config entirely. Using `uv pip install -e .` anywhere
silently skips the routing above — confirmed live via `uv pip show torch`
showing a bare `2.10.0` with none of the CUDA build's dependencies
(`nvidia-cublas-cu12`, `cuda-bindings`, etc.), even with the fix from part 1
in place and a from-scratch venv. **Always use `uv sync`, never
`uv pip install`, for this project.**

A `--dry-run` plan on the (different) `evals-sglang` container resolved
`cudf-cu12==25.10.0`, `cuml-cu12==25.10.0`, `pylibcugraph-cu12==25.10.1`,
etc. — versions matching NVIDIA's own tested `uv.lock` exactly, all from
plain PyPI, no special index needed. So `uv sync` alone (no
`--no-deps`/exclusion-list workaround) is the right fallback recipe.

**Note on verification**: the crash, the `uv pip show torch` diagnosis, and
the RAPIDS-version match were all confirmed live, but *whether `uv sync`
itself resolves a working CUDA torch end-to-end* has not been confirmed —
that fallback path is still unverified. This is exactly why we switched to
the pre-built official image as the default instead of continuing to debug
generic-container dependency resolution.
