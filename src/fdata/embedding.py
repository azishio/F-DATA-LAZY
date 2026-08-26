"""Phase 2: batched Sentence-BERT embeddings streamed into the final parquet.

Model inference cannot be expressed lazily, so the phase-1 parquet is read
back in bounded record batches; memory use is the model (~90 MB) plus one
batch, independent of dataset size.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from .schema import EMBED_DIM, EMBED_MODEL, EMBED_TEXT_COL, OUTPUT_COLUMNS


def load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, device="cpu")


def add_embeddings(
    intermediate: Path,
    output: Path,
    batch_size: int = 8192,
    model=None,
    encode_batch_size: int = 256,
) -> int:
    """Read the phase-1 parquet, append embeddings batch by batch, write the
    final single parquet in the public column order. Returns the row count."""
    if model is None:
        model = load_model()

    rows = 0
    writer = None
    try:
        pf = pq.ParquetFile(intermediate)
        for batch in pf.iter_batches(batch_size=batch_size):
            df = pl.from_arrow(batch)
            texts = df[EMBED_TEXT_COL].fill_null("").to_list()
            vecs = model.encode(
                texts,
                batch_size=encode_batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
            # list<float> (not fixed_size_list) to match the published files
            embedding = pl.Series(
                "embedding", vecs, dtype=pl.Array(pl.Float32, EMBED_DIM)
            ).cast(pl.List(pl.Float32))
            table = (
                df.with_columns(embedding)
                .drop(EMBED_TEXT_COL)
                .select(OUTPUT_COLUMNS)
                .to_arrow()
            )
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(df)
    finally:
        if writer is not None:
            writer.close()
    return rows
