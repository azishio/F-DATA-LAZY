"""Row filters and numeric casts (clean_and_anonmyze_data.py semantics)."""

from __future__ import annotations

import polars as pl

from .schema import DATE_FILTER_COLS, FLOAT_COLS, INT_COLS, SOFT_FLOAT_COLS


def clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    # Drop rows whose adt/sdt/edt is null, empty, or an epoch placeholder.
    for c in DATE_FILTER_COLS:
        lf = lf.filter(
            pl.col(c).is_not_null()
            & (pl.col(c) != "")
            & ~pl.col(c).str.starts_with("1970")
        )

    # The original crashed on non-numeric strings (float()/int() after a
    # null/"" filter); we instead drop such rows (strict=False -> null ->
    # filtered), a strict superset of what the original could process.
    # Int columns go through Float64 so values serialized as "3.0" survive.
    lf = lf.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in FLOAT_COLS]
        + [
            pl.col(c).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
            for c in INT_COLS
        ]
    ).filter(pl.all_horizontal(pl.col(FLOAT_COLS + INT_COLS).is_not_null()))

    return lf.with_columns(
        pl.col(c).cast(pl.Float64, strict=False) for c in SOFT_FLOAT_COLS
    )
