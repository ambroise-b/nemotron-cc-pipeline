# SLURM submission

For the actual, tested-on-this-cluster command flow (CSCS Clariden/Alps,
`infra01` account), see [`CSCS-QUICKSTART.md`](CSCS-QUICKSTART.md) — start
there. It covers verifying the container, a single-node (4 GPU) validation
run, and multi-node.

For ready-to-submit, pre-filled `sbatch` scripts (one per pipeline stage,
4 nodes), see [`submit_default/`](submit_default/).

`run_stage.sbatch` is the generic `sbatch` template that doc uses for
multi-node runs: it runs any single stage script from `src/nemotron-cc/`
across a SLURM allocation, using NeMo Curator's `SlurmRayClient` (see the
`--slurm` flag added to each `step_*.py` script) to form a multi-node Ray
cluster automatically — `srun` launches one process per node, node 0 becomes
the Ray head and runs the pipeline, the rest join as workers. On a
single-node allocation the scripts fall back to the default `RayClient`.

It defaults to the container defined in
[`container/container.toml`](../container/container.toml) — NVIDIA's own
official `nemo-curator` image, which already ships a matched
torch/RAPIDS/`nemo_curator` stack — and runs stage scripts directly against
its system Python. No install step, nothing to sync.

For a different container that doesn't already have `nemo-curator`
installed, set `INSTALL_VENV=1`: this builds a disposable venv via
`uv sync` in a job-scoped tmp directory under `$SCRATCH`, torn down on exit
(set `KEEP_JOB_TMP_DIR=1` to keep it for debugging a failed job).
`INSTALL_EXTRAS=sdg` also installs `vllm`. See the quickstart's
"Background" section for why this project must use `uv sync`, never
`uv pip install`.

Defaults target a single-node job on the `debug` partition under the
`infra01` account, claiming a full GH200 node (288 CPUs, ~870GB RAM, 4
GPUs — every node on this cluster has GPUs, confirmed via `scontrol show
node`) since `--gpus-per-node=4` already makes it exclusively ours
regardless. Override on the command line for other partitions, accounts,
or node counts, e.g. `sbatch --partition=<other> scripts/run_stage.sbatch`.
Set `CONTAINER_ENV=<path to a CSCS --environment= toml>` to use a different
container.

## Multi-node

Bump `--nodes` to span more machines — `run_stage.sbatch` detects
`SLURM_JOB_NUM_NODES > 1` and automatically adds `--slurm` so the target
script uses `SlurmRayClient`. CPU/mem/exclusivity defaults already claim a
full node each, so nothing else needs overriding:

```bash
STEP_SCRIPT=src/nemotron-cc/step_1-download_extract.py \
STEP_ARGS="--start-snapshot 2024-46 --end-snapshot 2024-51" \
sbatch --nodes=4 --time=04:00:00 scripts/run_stage.sbatch
```

If `/tmp` is node-local on your cluster (the common case), `run_stage.sbatch`
already points `RAY_PORT_BROADCAST_DIR` at a path inside the repo checkout;
make sure the repo lives on a shared filesystem visible to every allocated
node (on Clariden/Alps: `$SCRATCH`).
