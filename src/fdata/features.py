"""Derived features (generate_derived_features.py semantics)."""

from __future__ import annotations

import polars as pl

from .schema import EMBED_SOURCE, EMBED_TEXT_COL, RIDGE_POINT


def derive(lf: pl.LazyFrame) -> pl.LazyFrame:
    denom = pl.col("elp") - pl.col("idle_time_ave")
    lf = lf.with_columns(
        ((pl.col("perf2") + pl.col("perf3") * 4) / denom).alias("flops"),
        ((((pl.col("perf4") + pl.col("perf5")) / 12) * 256) / denom).alias("mbwidth"),
    ).with_columns(
        (pl.col("flops") / pl.col("mbwidth")).alias("opint"),
    )
    return lf.with_columns(
        pl.when(pl.col("opint") >= RIDGE_POINT)
        .then(pl.lit("compute-bound"))
        .otherwise(pl.lit("memory-bound"))
        .alias("pclass"),
        pl.when(pl.col("ec") == 0)
        .then(pl.lit("completed"))
        .otherwise(pl.lit("failed"))
        .alias("exit state"),
        pl.col("elp").alias("duration"),
        # Comma-join of truthy source values (pre-anonymization originals);
        # when() without otherwise() yields null for empty strings, which
        # ignore_nulls skips — matching the original convert_to_str's
        # falsy filter.
        pl.concat_str(
            [pl.when(pl.col(c) != "").then(pl.col(c)) for c in EMBED_SOURCE],
            separator=",",
            ignore_nulls=True,
        ).alias(EMBED_TEXT_COL),
    )
