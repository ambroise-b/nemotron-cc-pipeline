# Init submit scripts

One-time setup steps to run **before** the pipeline stages in
`scripts/submit_nemotron_cc/`. Run from the project root.

- `00_submit_warm_hf_cache.sh` — pre-download every HuggingFace artifact the
  pipeline needs (both quality classifiers, the fasttext-oh-eli5 model, and —
  unless `INCLUDE_QWEN=0` — the Qwen SDG model) into the shared cache
  (`$HF_HUB_CACHE`).

Why this exists: the stage launchers (`scripts/slurm/run_stage.sbatch`,
`run_sdg_split.sbatch`) set `HF_HUB_OFFLINE=1` by default. NeMo Curator's
`TokenizerStage.setup_on_node()` otherwise makes an unconditional
`snapshot_download(local_files_only=False)` network call **once per node**,
which flakes at ~50-node scale (`OSError: couldn't connect ... and couldn't
find them in the cached files`) and aborts the whole stage. Warming the cache
once here, then running offline, removes that failure mode entirely.

```bash
bash scripts/submit_init/00_submit_warm_hf_cache.sh
```

Idempotent — already-cached artifacts are skipped, so it's safe to re-run.
See `scripts/CSCS-QUICKSTART.md` (Part 0.5) for the full explanation.
