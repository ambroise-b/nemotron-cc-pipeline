"""Streaming, memory-safe replacement for the SDG preprocessing split->join.

The stock preprocessing (DocumentSplitter -> per-segment TokenCountFilter ->
DocumentJoiner -> min-segment Filter) explodes each document into one row per
line before greedily re-joining lines into <=max_segment_tokens chunks. A
document with millions of newlines explodes into millions of rows inside a
single worker -> OOM (see logs/nemotron_cc_sdg_array_*.log).

This produces the SAME chunks in one streaming pass, never materializing the
per-line rows, so memory is bounded by chunk size instead of line count.

Semantics replicate nemo_curator 26.04 exactly (verified against the container
source; guarded by tests/test_sdg_chunker.py):
  * split on `separator` (str.split)
  * per-line token count = len(tokenizer.encode(line)); drop lines whose count
    exceeds max_segment_tokens (the per-segment TokenCountFilter has min=0)
  * greedy pack kept lines: proposed = acc_len + line_len + len(separator);
    while proposed <= max_segment_tokens append with separator, else emit the
    current chunk and start a new one. A chunk's length is the ACCUMULATED SUM
    (DocumentJoiner does NOT re-tokenize the joined text).
  * drop chunks whose length < min_segment_tokens
"""

from __future__ import annotations


def chunk_document_text(
    text: str,
    tokenizer,
    max_segment_tokens: int,
    min_segment_tokens: int,
    separator: str = "\n",
    token_batch: int = 10_000,
) -> list[tuple[str, int]]:
    """Return [(chunk_text, chunk_token_len), ...] for one document.

    Pure function (only needs a HF tokenizer). Peak memory is the line list plus
    one `token_batch` of tokenizations — never the exploded per-line row set.
    """
    lines = text.split(separator)
    chunks: list[tuple[str, int]] = []
    acc_text: str | None = None
    acc_len = 0

    for start in range(0, len(lines), token_batch):
        batch = lines[start : start + token_batch]
        # Batched tokenization; per-line count matches len(tokenizer.encode(line))
        # (special tokens included) == TokenCountFilter.score_document.
        lengths = [len(ids) for ids in tokenizer(batch)["input_ids"]]
        for line, line_len in zip(batch, lengths):
            if line_len > max_segment_tokens:
                continue  # dropped by the per-segment max_tokens filter
            if acc_text is None:
                acc_text, acc_len = line, line_len
                continue
            proposed = acc_len + line_len + len(separator)
            if proposed <= max_segment_tokens:
                acc_text = acc_text + separator + line
                acc_len = proposed
            else:
                chunks.append((acc_text, acc_len))
                acc_text, acc_len = line, line_len

    if acc_text is not None:
        chunks.append((acc_text, acc_len))

    return [(t, n) for t, n in chunks if n >= min_segment_tokens]


def add_streaming_chunker_stages(pipeline, task_config, tokenizer, hf_token, max_segment_tokens):
    """Memory-safe drop-in for step_4-sdg._add_preprocessing_stages.

    Adds the same min_document_tokens ScoreFilter (unchanged) then a single
    streaming chunker stage. Output columns match the stock chain: text, id,
    segment_id, segment_token_count, document_token_count, plus carried columns.
    """
    import pandas as pd
    from nemo_curator.stages.function_decorators import processing_stage
    from nemo_curator.stages.text.filters import ScoreFilter
    from nemo_curator.stages.text.filters.token import TokenCountFilter
    from nemo_curator.tasks import DocumentBatch

    min_document_tokens = task_config["min_document_tokens"]
    min_segment_tokens = task_config["min_segment_tokens"]

    # Identical to the stock path: drop whole docs below min_document_tokens and
    # add document_token_count. Per-document, cheap, no explosion.
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(tokenizer=tokenizer, hf_token=hf_token, min_tokens=min_document_tokens),
            text_field="text",
            score_field="document_token_count",
        ),
    )

    @processing_stage(name="streaming_chunker")
    def streaming_chunker(batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        rows = []
        for _, row in df.iterrows():
            for seg_id, (ctext, clen) in enumerate(
                chunk_document_text(row["text"], tokenizer, max_segment_tokens, min_segment_tokens)
            ):
                new = row.copy()
                new["text"] = ctext
                new["segment_id"] = seg_id
                new["segment_token_count"] = clen
                rows.append(new)
        if rows:
            out_df = pd.DataFrame(rows).reset_index(drop=True)
        else:
            out_df = df.iloc[0:0].copy()
            out_df["segment_id"] = pd.Series(dtype="int64")
            out_df["segment_token_count"] = pd.Series(dtype="int64")
        batch.data = out_df
        return batch

    pipeline.add_stage(streaming_chunker)
    return pipeline
