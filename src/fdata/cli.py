"""fdata CLI: one-shot generate/plot commands."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import polars as pl

from .anonymize import anonymize, build_maps
from .cleaning import clean
from .embedding import add_embeddings
from .features import derive
from .io import scan_input
from .schema import PHASE1_COLUMNS
from .split import compute_threshold, with_split


def _log(msg: str) -> None:
    print(f"[fdata] {msg}", file=sys.stderr, flush=True)


def cmd_generate(args: argparse.Namespace) -> None:
    if not 0 < args.train_ratio < 1:
        raise SystemExit(f"--train-ratio must be in (0, 1), got {args.train_ratio}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else output.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    intermediate = tmp_dir / f".{output.stem}.phase1.parquet"

    t0 = time.monotonic()
    lf = clean(scan_input(Path(args.input), args.format))

    _log("building anonymization maps (streaming pass A)")
    maps = build_maps(lf)
    for feat, mapping in maps.items():
        _log(f"  {feat}: {len(mapping)} unique values")
    lf = derive(anonymize(lf, maps))

    _log(f"computing train/test threshold at ratio {args.train_ratio} (pass B)")
    threshold = compute_threshold(lf, args.train_ratio)
    _log(f"  train = adt <= {threshold!r}")
    lf = with_split(lf, threshold)

    _log(f"sinking cleaned data to {intermediate} (streaming pass C)")
    try:
        lf.select(PHASE1_COLUMNS).sink_parquet(intermediate, row_group_size=64_000)

        _log("computing embeddings and writing final parquet (pass D)")
        rows = add_embeddings(intermediate, output, batch_size=args.batch_size)
    finally:
        intermediate.unlink(missing_ok=True)

    _log(f"wrote {rows} rows to {output} in {time.monotonic() - t0:.1f}s")


def cmd_plot(args: argparse.Namespace) -> None:
    from .plots import render

    t0 = time.monotonic()
    render(Path(args.input), Path(args.output_dir))
    _log(f"wrote plots to {args.output_dir} in {time.monotonic() - t0:.1f}s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fdata",
        description="Generate the public F-DATA parquet from a raw Fugaku job log dump",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="raw CSV/parquet -> single F-DATA parquet")
    gen.add_argument("--input", required=True, help="raw dump (.csv or .parquet)")
    gen.add_argument("--output", required=True, help="output parquet path")
    gen.add_argument(
        "--train-ratio",
        type=float,
        default=float(os.environ.get("FDATA_TRAIN_RATIO", "0.8")),
        help="fraction of earliest-adt rows labeled 'train' "
        "(default 0.8, env FDATA_TRAIN_RATIO)",
    )
    gen.add_argument("--format", choices=["csv", "parquet"], default=None,
                     help="input format (default: inferred from extension)")
    gen.add_argument("--batch-size", type=int, default=8192,
                     help="rows per embedding batch (default 8192)")
    gen.add_argument("--tmp-dir", default=None,
                     help="directory for the phase-1 intermediate "
                     "(default: alongside the output)")
    gen.set_defaults(func=cmd_generate)

    plot = sub.add_parser("plot", help="generated parquet -> monthly + aggregate plots")
    plot.add_argument("--input", required=True, help="generated F-DATA parquet")
    plot.add_argument("--output-dir", required=True, help="directory for the plots")
    plot.set_defaults(func=cmd_plot)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
