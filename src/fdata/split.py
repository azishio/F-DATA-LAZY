"""Temporal train/test split on adt.

adt strings are zero-padded ("YYYY-MM-DD hh:mm:ss"), so lexicographic order
is chronological; comparing strings avoids assuming an exact datetime format.
The threshold is the k-th smallest adt (k = ceil(n * ratio)); ties at the
threshold all land in train, so the realized train fraction is >= ratio.
"""

from __future__ import annotations

import math

import polars as pl

from .schema import SPLIT_COL


def compute_threshold(lf: pl.LazyFrame, ratio: float, n: int) -> str | None:
    """Return the adt value below which rows are train, or None for k == 0.

    `n` is the row count of `lf`, supplied by the caller so it can be
    computed alongside other aggregations in a shared collect_all pass.
    """
    if n == 0:
        raise ValueError("No rows survived cleaning; nothing to split")
    k = min(math.ceil(n * ratio), n)
    if k <= 0:
        return None
    return (
        lf.select("adt")
        .sort("adt")
        .slice(k - 1, 1)
        .collect(engine="streaming")
        .item()
    )


def with_split(lf: pl.LazyFrame, threshold: str | None) -> pl.LazyFrame:
    if threshold is None:
        expr = pl.lit("test")
    else:
        expr = (
            pl.when(pl.col("adt") <= threshold)
            .then(pl.lit("train"))
            .otherwise(pl.lit("test"))
        )
    return lf.with_columns(expr.alias(SPLIT_COL))
