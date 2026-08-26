"""Unified input scanning: CSV or parquet -> all-Utf8 LazyFrame."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .schema import RAW_INPUT_COLUMNS, RAW_RENAMES


def infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in (".parquet", ".pq"):
        return "parquet"
    raise ValueError(
        f"Cannot infer input format from '{path.name}'; pass --format csv|parquet"
    )


def scan_input(path: Path, fmt: str | None = None) -> pl.LazyFrame:
    """Scan a raw dump as a LazyFrame with every column read as Utf8.

    Treating all raw values as strings makes CSV and parquet inputs behave
    identically and mirrors how the original pandas script saw the data
    before its explicit float()/int() casts.
    """
    fmt = fmt or infer_format(path)
    if fmt == "csv":
        lf = pl.scan_csv(path, infer_schema=False, truncate_ragged_lines=True)
    elif fmt == "parquet":
        lf = pl.scan_parquet(path)
        lf = lf.with_columns(pl.all().cast(pl.Utf8, strict=False))
    else:
        raise ValueError(f"Unsupported input format: {fmt}")

    names = lf.collect_schema().names()
    renames = {src: dst for src, dst in RAW_RENAMES.items() if src in names and dst not in names}
    if renames:
        lf = lf.rename(renames)
        names = [renames.get(n, n) for n in names]

    missing = [c for c in RAW_INPUT_COLUMNS if c not in names]
    if missing:
        raise ValueError(
            "Input is missing required raw columns: " + ", ".join(missing)
        )

    # Whitelist select: the output schema is fixed, so dropping junk columns
    # (ermsg, fjprofiler, elpl.1, all-null columns, ...) by omission is
    # equivalent to the original's explicit drops and streaming-safe.
    return lf.select(RAW_INPUT_COLUMNS)
