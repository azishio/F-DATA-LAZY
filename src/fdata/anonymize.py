"""First-appearance sequential anonymization ("usr_0", "usr_1", ...).

The original script accumulated a value -> "{feat}_{n}" dict in row order
across the monthly files. With a single input, first appearance = minimum
row index; the mapping tables (unique users/job names) are tiny compared to
the data, so we build them in one streaming pass and apply them via small
in-memory left joins.

Divergences from the original, both documented in the README:
- the original built its maps before the numeric row-drops, so values that
  only occur in later-dropped rows still consumed a pseudonym number; we map
  the cleaned data, which yields an equivalent (denser) bijection.
- the original also mapped NaN to a pseudonym; we keep nulls null.
"""

from __future__ import annotations

import polars as pl

from .schema import ANON_FEATURES


def build_maps(lf: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    lf_idx = lf.with_row_index("_ridx")
    plans = [
        lf_idx.filter(pl.col(feat).is_not_null())
        .group_by(feat)
        .agg(pl.col("_ridx").min())
        .sort("_ridx")
        .with_row_index("_rank")
        .select(
            pl.col(feat).alias(f"{feat}_or"),
            pl.format("{}_{}", pl.lit(feat), pl.col("_rank")).alias(feat),
        )
        for feat in ANON_FEATURES
    ]
    return dict(zip(ANON_FEATURES, pl.collect_all(plans, engine="streaming")))


def anonymize(lf: pl.LazyFrame, maps: dict[str, pl.DataFrame]) -> pl.LazyFrame:
    for feat, mapping in maps.items():
        lf = lf.rename({feat: f"{feat}_or"}).join(
            mapping.lazy(), on=f"{feat}_or", how="left", maintain_order="left"
        )
    return lf
