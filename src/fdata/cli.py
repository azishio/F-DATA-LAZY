"""fdata CLI: one-shot generate/plot commands."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import polars as pl

from .anonymize import build_maps, generate_salt
from .cleaning import clean
from .embedding import encode_unique, load_model, write_final
from .features import derive
from .io import scan_input
from .schema import EMBED_TEXT_COL, INTERMEDIATE_COLUMNS
from .split import compute_threshold


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
    try:
        # Pass 1: the only read of the (possibly CSV) source — clean, derive,
        # and sink to an intermediate parquet all later passes read instead.
        _log("cleaning and deriving features (streaming pass 1)")
        lf = derive(clean(scan_input(Path(args.input), args.format)))
        lf.select(INTERMEDIATE_COLUMNS).sink_parquet(
            intermediate, row_group_size=64_000
        )
        _log(f"  sunk intermediate parquet ({time.monotonic() - t0:.1f}s)")

        salt = args.anon_salt
        if not salt:
            salt = generate_salt()
            _log(f"generated anonymization salt: {salt}")
            _log(
                "  pass this via --anon-salt (or FDATA_ANON_SALT) to future"
                " runs whose output must be combinable with this one"
            )

        # Pass 2: one shared collect_all over the intermediate for the
        # anonymization maps, the row count, and the distinct embedding
        # texts; plus a single-column scan for the split threshold.
        lf_i = pl.scan_parquet(intermediate)
        _log("collecting anonymization maps, row count, unique texts (pass 2)")
        maps = build_maps(lf_i, salt)
        for feat, mapping in maps.items():
            _log(f"  {feat}: {len(mapping)} unique values")
        n_rows, texts_df = pl.collect_all(
            [
                lf_i.select(pl.len()),
                lf_i.select(pl.col(EMBED_TEXT_COL).fill_null("")).unique(),
            ],
            engine="streaming",
        )
        n = n_rows.item()
        texts = texts_df.to_series().to_list()
        threshold = compute_threshold(lf_i, args.train_ratio, n)
        _log(f"  {n} rows, {len(texts)} unique embedding texts")
        _log(f"  train = adt <= {threshold!r} (ratio {args.train_ratio})")

        # Pass 3: encode each distinct text exactly once.
        _log(f"encoding unique texts with the {args.encoder_backend} backend (pass 3)")
        embeddings = encode_unique(
            texts,
            load_model(args.encoder_backend),
            chunk_size=args.batch_size,
            log=_log,
        )

        # Pass 4: stream the intermediate, applying anonymization, split,
        # and the embedding lookup per batch.
        _log("writing final parquet (pass 4)")
        rows = write_final(
            intermediate,
            output,
            maps,
            threshold,
            embeddings,
            batch_size=args.batch_size,
            log=_log,
        )
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
                     help="rows per encode chunk and final-write batch "
                     "(default 8192)")
    gen.add_argument(
        "--anon-salt",
        default=os.environ.get("FDATA_ANON_SALT"),
        help="secret salt for hmac anonymization; keep it private and reuse "
        "it across runs whose outputs must be combinable. Auto-generated "
        "(and logged) from the current time plus random bits when omitted. "
        "Prefer the FDATA_ANON_SALT env var over the flag (command lines "
        "leak via process listings)",
    )
    gen.add_argument(
        "--encoder-backend",
        choices=["torch", "onnx"],
        default=os.environ.get("FDATA_ENCODER_BACKEND", "torch"),
        help="sentence-transformers inference backend (default torch, "
        "env FDATA_ENCODER_BACKEND; onnx is ~2-3x faster on CPU and "
        "requires the [onnx] extra)",
    )
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
