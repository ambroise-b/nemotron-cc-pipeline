# Default submit scripts

One `sbatch` submission script per pipeline stage, numbered in run order,
pre-filled with the same values as `scripts/CSCS-QUICKSTART.md` /
`configs/sample.env`, scaled to 4 nodes. Run from the project root:

```bash
bash scripts/submit_default/01_submit_download.sh
bash scripts/submit_default/02a_submit_exact_dedup.sh
bash scripts/submit_default/02b_submit_fuzzy_dedup.sh
bash scripts/submit_default/02c_submit_substring_dedup.sh
bash scripts/submit_default/03_submit_quality_classification.sh
bash scripts/submit_default/04_submit_sdg.sh
```

Each stage depends on the previous one's output — run them one at a time,
check the output directory, then submit the next. All variables (SLURM
resources, snapshot range, paths) are declared at the top of each file —
edit those, not the `sbatch` call itself.

`02c` (substring dedup) requests 4 nodes like the others for consistency,
but `scripts/run_stage.sbatch` deliberately runs it on only 1 of them —
that script has no multi-node support, and running one copy per node would
have every node corrupt the same shared output paths. See the comment in
that submit script for why.

`04` (SDG) is unverified end-to-end on this cluster (`vllm`/`--serve-model`
— see `scripts/CSCS-QUICKSTART.md`).
