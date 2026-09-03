# f-data-lazy

A one-shot containerized pipeline that turns a **single raw Fugaku job log
dump** (CSV or parquet, not split by month) into a **single parquet** with the
same schema as the published
[F-DATA](https://doi.org/10.5281/zenodo.11467483) dataset
([`docs/feature_list.csv`](docs/feature_list.csv), 45 columns), plus a
`split` column for a temporal train/test split.

Built on [polars](https://pola.rs) lazy/streaming execution: the dataset is
never fully loaded into memory. The only non-lazy stage — Sentence-BERT
embedding inference — encodes each **distinct** input text exactly once
(the text depends only on `usr`/`jnam`/`jobenv_req`, so distinct texts are
typically orders of magnitude fewer than rows) and runs over bounded
chunks, so memory use stays flat regardless of input size. Progress (rows
processed, elapsed time) is logged to stderr throughout.

Based on F-DATA: *A Fugaku Workload Dataset for Job-centric Predictive
Modelling in HPC Systems* (Antici et al., Scientific Data 12, 1321, 2025).
The original scripts are preserved unchanged in [`legacy/`](legacy/).

## Usage

The image is published to GHCR by CI; `latest` tracks the default branch.

```bash
# Raw dump -> F-DATA parquet (temporal 80/20 train/test split)
docker run --rm -v "$PWD/data:/data" ghcr.io/azishio/f-data-lazy:latest \
  generate --input /data/raw.csv --output /data/fdata.parquet --train-ratio 0.8

# Monthly + aggregate plots from the generated parquet
docker run --rm -v "$PWD/data:/data" ghcr.io/azishio/f-data-lazy:latest \
  plot --input /data/fdata.parquet --output-dir /data/plots

# The split ratio can also come from an environment variable
docker run --rm -e FDATA_TRAIN_RATIO=0.9 -v "$PWD/data:/data" \
  ghcr.io/azishio/f-data-lazy:latest \
  generate --input /data/raw.parquet --output /data/fdata.parquet
```

The sentence-transformers model (`all-MiniLM-L6-v2`) is baked into the image
as local model directories (`/opt/models/{torch,onnx}`, selected via
`FDATA_MODEL_DIR`); the container needs no network access at runtime. Note
that an HF cache alone would not be enough: sentence-transformers queries
the HF Hub to locate the backend file when given a model ID, even when the
model is cached — only local-directory loading is fully offline, and the
image build verifies this with an offline load check.

### `generate` options

| Option | Default | Description |
|---|---|---|
| `--input` | (required) | Raw dump, `.csv` or `.parquet` |
| `--output` | (required) | Output parquet path |
| `--train-ratio` | `0.8` (or `FDATA_TRAIN_RATIO`) | Fraction of earliest-`adt` rows labeled `train` |
| `--format` | inferred from extension | Force `csv` or `parquet` |
| `--batch-size` | `8192` | Unique texts per embedding chunk |
| `--tmp-dir` | alongside the output | Where the phase-1 intermediate parquet lives |
| `--anon-salt` | auto-generated + logged (or `FDATA_ANON_SALT`) | Secret salt for the anonymization hashes; reuse one salt across runs whose outputs must be combinable. Prefer the env var (command lines leak via process listings) |
| `--encoder-backend` | `torch` (or `FDATA_ENCODER_BACKEND`) | `onnx` is ~2-3x faster on CPU (needs the `[onnx]` extra); the container image defaults to `onnx` |

### Input schema

The input must contain the raw columns listed in
[`src/fdata/schema.py`](src/fdata/schema.py) (`RAW_INPUT_COLUMNS`): every
non-derived column of `docs/feature_list.csv` plus `elp`. The raw names
`cr_jobenv_req`, `cr_freq_req`, `cr_freq_alloc` are accepted and renamed.
Extra columns (`ermsg`, `fjprofiler`, ...) are ignored.

## What the pipeline does

Reimplements `legacy/generation_scripts/` as four passes; the (possibly
CSV) source is parsed exactly once, and every later pass reads the fast
columnar intermediate instead:

1. **Clean + derive** (streaming) — drop rows with null/empty/epoch
   (`1970-*`) `adt`/`sdt`/`edt`; cast the numeric columns, dropping rows
   that fail (datetime columns stay strings, exactly as in the published
   data); compute `flops`, `mbwidth`, `opint`, `pclass`, `exit state`,
   `duration` (roofline model constants of Fugaku) and the embedding input
   text; sink everything to an intermediate parquet.
2. **Aggregate** (streaming, one shared pass) — anonymization maps for
   `jid`/`usr`/`jnam`/`jobenv_req`, the row count, the distinct embedding
   texts, and the temporal split threshold: rows are ordered by `adt` and
   the earliest `ceil(n * ratio)` become `train`, the rest `test` (ties at
   the threshold go to `train`, so the realized train fraction is ≥ the
   ratio). Anonymization labels are salted hashes
   (`usr_<64 hex of HMAC-SHA256(salt, value)>`): runs sharing a salt
   produce identical labels, so their outputs can be combined, while the
   secret salt prevents dictionary attacks on guessable identifiers. The
   pipeline verifies that each anonymized column has the same number of
   unique values before and after anonymization, aborting on a collision.
   When no salt is given one is generated (current time + random bits) and
   logged for reuse.
3. **Encode** — each distinct text is embedded exactly once (384-dim
   Sentence-BERT over the comma-joined original `usr`/`jnam`/`jobenv_req`),
   with chunked progress logging; identical rows get bit-identical vectors.
4. **Write** — Polars lazily applies the pseudonym maps, the `split` label,
   and the embedding lookup, sorts by `adt`, then streams a single Zstd
   parquet with standard row-group statistics (`min`, `max`, `null_count`).

`plot` reproduces the figures of `legacy/generate_plots.py` from the
generated parquet — per-month exit-code/duration/power distributions under
`<output-dir>/<YY_MM>/` and the aggregate 2×3 `pairplot` — with all
aggregation done in one streaming polars pass and months derived from `adt`.

### Known divergences from the original scripts

- The original crashed on non-numeric strings in numeric columns; this
  pipeline drops those rows instead.
- Anonymization labels are salted hashes (`usr_9f3a...`) instead of the
  published first-appearance numbering (`usr_0`, `usr_1`, ...): sequential
  labels depend on row order, so separately processed datasets could not
  be combined (the same value gets different labels, and different values
  collide on labels like `usr_0`).
- A null anonymized value stays null instead of receiving a pseudonym.
- Duration histograms use 50 fixed bins instead of seaborn's automatic
  binning.

## Development

```bash
pip install -e .[dev]
pytest tests/
docker build -t f-data-lazy .
```

Tests run the full pipeline on a synthetic fixture with a stub embedding
model, so they don't need torch or a model download.

## Cite

```bibtex
@article{antici2025fdata,
  title={F-DATA: A Fugaku Workload Dataset for Job-centric Predictive Modelling in HPC Systems},
  author={Antici, Francesco and Bartolini, Andrea and Domke, Jens and Kiziltan, Zeynep and Yamamoto, Keiji},
  journal = {Scientific Data},
  volume={12},
  pages={1321},
  year={2025},
  publisher={Nature Publishing Group},
  doi={https://doi.org/10.1038/s41597-025-05633-1}
}
```
