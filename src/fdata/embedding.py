"""Phase 3/4: deduplicated Sentence-BERT encoding and the final Polars sink.

The embedding input text is a function of (usr, jnam, jobenv_req) only, so
the number of distinct texts is typically orders of magnitude smaller than
the row count. Each distinct text is encoded exactly once (identical rows
therefore get bit-identical vectors) and the final pass is a lazy join and
streaming parquet sink. Memory use is the model plus the distinct-text
embedding table (~1.5 KB per distinct text), independent of the row count.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import polars as pl

from .anonymize import anonymize
from .schema import EMBED_DIM, EMBED_MODEL, EMBED_TEXT_COL, OUTPUT_COLUMNS
from .split import with_split


def load_model(backend: str = "torch"):
    from sentence_transformers import SentenceTransformer

    # Loading by model ID queries the HF Hub (list_repo_files) to locate the
    # backend file even when the model is cached, which breaks offline runs;
    # a local directory is resolved entirely on disk. The container bakes
    # the models as directories and sets FDATA_MODEL_DIR.
    model_dir = os.environ.get("FDATA_MODEL_DIR")
    source = str(Path(model_dir) / backend) if model_dir else EMBED_MODEL
    return SentenceTransformer(source, device="cpu", backend=backend)


def encode_unique(
    texts: list[str],
    model,
    chunk_size: int = 8192,
    encode_batch_size: int = 256,
    log=lambda msg: None,
) -> pl.DataFrame:
    """Encode each distinct text once; returns a {text -> embedding} table."""
    total = len(texts)
    t0 = time.monotonic()
    chunks = []
    for start in range(0, total, chunk_size):
        chunk = texts[start : start + chunk_size]
        vecs = model.encode(
            chunk,
            batch_size=encode_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        chunks.append(
            pl.DataFrame(
                {
                    EMBED_TEXT_COL: pl.Series(chunk, dtype=pl.Utf8),
                    # list<float> (not fixed_size_list) to match the
                    # published files
                    "embedding": pl.Series(
                        vecs, dtype=pl.Array(pl.Float32, EMBED_DIM)
                    ).cast(pl.List(pl.Float32)),
                }
            )
        )
        done = min(start + chunk_size, total)
        log(f"  encoded {done}/{total} unique texts ({time.monotonic() - t0:.1f}s)")
    return pl.concat(chunks)


def write_final(
    intermediate: Path,
    output: Path,
    maps: dict[str, pl.DataFrame],
    threshold: str | None,
    embeddings: pl.DataFrame,
) -> None:
    """Write one adt-sorted parquet with standard row-group statistics."""
    lf = pl.scan_parquet(intermediate).with_columns(
        pl.col(EMBED_TEXT_COL).fill_null("")
    )
    lf = with_split(anonymize(lf, maps), threshold).sort("adt")
    (
        lf.join(
            embeddings.lazy(),
            on=EMBED_TEXT_COL,
            how="left",
            maintain_order="left",
        )
        .select(OUTPUT_COLUMNS)
        .sink_parquet(
            output,
            compression="zstd",
            row_group_size=64_000,
            statistics=True,
        )
    )
