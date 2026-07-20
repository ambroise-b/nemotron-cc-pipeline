#!/usr/bin/env python3
"""Enclosed equivalence + memory test for the streaming SDG chunker.

Runs the STOCK preprocessing stages (step_4-sdg._add_preprocessing_stages) and
the NEW streaming chunker (src/lib/sdg_chunker) on the same input, IN-PROCESS
(no Ray, no vLLM), and asserts the resulting tables match row-for-row. Also
supports a chunker-only mode to prove it completes on a shard whose stock path
OOMs.

This is self-contained: it imports the real production stages so there is no
behavioural drift, but it does not touch the main pipeline or its flags.

Run inside the nemo-curator container (needs nemo_curator + the tokenizer):

  # equivalence on a couple of normal files of shard 0
  python3 tests/test_sdg_chunker.py \
      --input-dir "$SCRATCH/nemotron-cc-pipeline-CC-MAIN-2026-21/quality_labeling/bucketed_results" \
      --num-shards 500 --shard-id 0 --num-files 2

  # chunker-only on the known-bad shard (stock path would OOM)
  python3 tests/test_sdg_chunker.py --input-dir ... --shard-id 1 --num-files 39 --new-only
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import resource
import sys
import time

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src", "lib"))

KEEP = ["id", "segment_id", "text", "segment_token_count", "document_token_count"]


def load_step4():
    """Import the hyphenated step_4-sdg.py as a module (real production stages)."""
    path = os.path.join(REPO, "src", "nemotron-cc", "step_4-sdg.py")
    spec = importlib.util.spec_from_file_location("step4_sdg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_stages(stages, batch):
    """Run a list of ProcessingStages sequentially, in-process."""
    for stage in stages:
        setup = getattr(stage, "setup", None)
        if callable(setup):
            try:
                setup()
            except Exception:  # noqa: BLE001 - setup may expect a WorkerMetadata; no-op for ours
                pass
        batch = stage.process(batch)
    return batch


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def shard_files(input_dir, buckets, num_shards, shard_id):
    """Same sorted-glob + stride as step_4-sdg.build_pipeline."""
    paths = []
    for b in buckets:
        d = os.path.join(input_dir, f"ensemble-max-int={b}")
        if os.path.isdir(d):
            paths.append(d)
    all_files = sorted(f for d in paths for f in glob.glob(os.path.join(d, "*.parquet")))
    return all_files[shard_id::num_shards] if num_shards > 1 else all_files


def normalize(df):
    df = df.copy()
    for c in ("segment_id", "segment_token_count", "document_token_count"):
        if c in df.columns:
            df[c] = df[c].astype("int64")
    df["text"] = df["text"].astype("string").astype("object")
    df = df[[c for c in KEEP if c in df.columns]]
    return df.sort_values(["id", "segment_id"]).reset_index(drop=True)


def compare(old_df, new_df):
    """Return (ok, diff_df). Aligns on (id, segment_id)."""
    a, b = normalize(old_df), normalize(new_df)
    if a.shape != b.shape:
        merged = a.merge(b, on=["id", "segment_id"], how="outer",
                         suffixes=("_old", "_new"), indicator=True)
        return False, merged[merged["_merge"] != "both"].head(10)
    ok = (
        a["id"].equals(b["id"])
        and a["segment_id"].equals(b["segment_id"])
        and a["text"].equals(b["text"])
        and a["segment_token_count"].equals(b["segment_token_count"])
    )
    if ok:
        return True, None
    merged = a.merge(b, on=["id", "segment_id"], how="outer",
                     suffixes=("_old", "_new"), indicator=True)
    bad = merged[
        (merged["_merge"] != "both")
        | (merged["text_old"] != merged["text_new"])
        | (merged["segment_token_count_old"] != merged["segment_token_count_new"])
    ]
    return False, bad.head(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=500)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-files", type=int, default=1, help="how many files of the shard to test")
    ap.add_argument("--buckets", type=int, nargs="+", default=[18, 19])
    ap.add_argument("--task", default="diverse_qa", choices=["diverse_qa", "distill", "extract_knowledge", "knowledge_list"])
    ap.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--max-docs", type=int, default=None, help="cap docs per file (speed)")
    ap.add_argument("--new-only", action="store_true", help="run only the chunker (for shards where the stock path OOMs)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.tasks import DocumentBatch

    import sdg_chunker
    step4 = load_step4()

    task_config = step4.TASK_CONFIGS[args.task]
    tok = AutoTokenizer.from_pretrained(args.model_name)
    prefix = step4._get_prefix_token_count(tok, task_config["system_prompt"], task_config["prompt_template"])
    max_seg = task_config["max_input_tokens"] - prefix - 2
    print(f"task={args.task}  max_segment_tokens={max_seg}  (prefix={prefix})")

    files = shard_files(args.input_dir, args.buckets, args.num_shards, args.shard_id)[: args.num_files]
    if not files:
        raise SystemExit("no files selected — check --input-dir/--shard-id/--num-shards")
    print(f"testing {len(files)} file(s) of shard {args.shard_id}/{args.num_shards}\n")

    all_ok = True
    total_new = 0
    for fi, f in enumerate(files):
        df = pd.read_parquet(f)
        if "id" not in df.columns:  # mirrors step_4-sdg add-document-id
            df["id"] = range(len(df))
        if args.max_docs:
            df = df.head(args.max_docs).copy()

        # --- NEW (streaming chunker) ---
        p_new = Pipeline(name="new")
        sdg_chunker.add_streaming_chunker_stages(p_new, task_config, tok, None, max_seg)
        t0 = time.perf_counter()
        new_df = run_stages(p_new.stages, DocumentBatch(task_id="t", dataset_name="t", data=df.copy())).to_pandas()
        dt_new = time.perf_counter() - t0
        total_new += len(new_df)

        if args.new_only:
            print(f"[{fi}] {os.path.basename(f)}: docs={len(df):>6} chunks={len(new_df):>7} "
                  f"{dt_new:6.1f}s  peakRSS={peak_rss_gb():.1f}GB")
            continue

        # --- OLD (stock split/explode/join) ---
        p_old = Pipeline(name="old")
        step4._add_preprocessing_stages(p_old, task_config, tok, None)
        old_df = run_stages(p_old.stages, DocumentBatch(task_id="t", dataset_name="t", data=df.copy())).to_pandas()

        ok, diff = compare(old_df, new_df)
        all_ok = all_ok and ok
        print(f"[{fi}] {os.path.basename(f)}: docs={len(df):>6} old={len(old_df):>7} new={len(new_df):>7} "
              f"-> {'OK' if ok else 'MISMATCH'}  (peakRSS={peak_rss_gb():.1f}GB)")
        if not ok:
            print(diff.to_string(), "\n")

    if args.new_only:
        print(f"\nchunker-only complete: {total_new} chunks over {len(files)} file(s), peakRSS={peak_rss_gb():.1f}GB")
        return
    print(f"\n{'ALL FILES MATCH' if all_ok else 'MISMATCHES FOUND'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
