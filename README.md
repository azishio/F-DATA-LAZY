# f-data-lazy

A one-shot containerized pipeline that turns a **single raw Fugaku job log
dump** (CSV or parquet, not split by month) into a **single parquet** with the
same schema as the published
[F-DATA](https://doi.org/10.5281/zenodo.11467483) dataset
([`docs/feature_list.csv`](docs/feature_list.csv), 45 columns), plus a
`split` column for a temporal train/test split.

Built on [polars](https://pola.rs) lazy/streaming execution: the dataset is
never fully loaded into memory. The only non-lazy stage — Sentence-BERT
embedding inference — runs over bounded record batches, so memory use stays
flat regardless of input size.

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

The sentence-transformers model (`all-MiniLM-L6-v2`) is baked into the image;
the container needs no network access at runtime.

### `generate` options

| Option | Default | Description |
|---|---|---|
| `--input` | (required) | Raw dump, `.csv` or `.parquet` |
| `--output` | (required) | Output parquet path |
| `--train-ratio` | `0.8` (or `FDATA_TRAIN_RATIO`) | Fraction of earliest-`adt` rows labeled `train` |
| `--format` | inferred from extension | Force `csv` or `parquet` |
| `--batch-size` | `8192` | Rows per embedding batch (memory knob) |
| `--tmp-dir` | alongside the output | Where the phase-1 intermediate parquet lives |

### Input schema

The input must contain the raw columns listed in
[`src/fdata/schema.py`](src/fdata/schema.py) (`RAW_INPUT_COLUMNS`): every
non-derived column of `docs/feature_list.csv` plus `elp`. The raw names
`cr_jobenv_req`, `cr_freq_req`, `cr_freq_alloc` are accepted and renamed.
Extra columns (`ermsg`, `fjprofiler`, ...) are ignored.

## What the pipeline does

Reimplements `legacy/generation_scripts/` as four streaming passes:

1. **Clean** — drop rows with null/empty/epoch (`1970-*`) `adt`/`sdt`/`edt`;
   cast the numeric columns, dropping rows that fail; datetime columns stay
   strings, exactly as in the published data.
2. **Anonymize** — `jid`, `usr`, `jnam`, `jobenv_req` are replaced by
   first-appearance sequential pseudonyms (`usr_0`, `usr_1`, ...), built as
   small mapping tables in one streaming pass and applied via joins.
3. **Derive + split** — `flops`, `mbwidth`, `opint`, `pclass`,
   `exit state`, `duration` (roofline model constants of Fugaku), and the
   `split` column: rows are ordered by `adt` and the earliest
   `ceil(n * ratio)` become `train`, the rest `test` (ties at the threshold go
   to `train`, so the realized train fraction is ≥ the ratio).
4. **Embed** — `embedding` (384-dim Sentence-BERT over the comma-joined
   original `usr`/`jnam`/`jobenv_req`), computed batch-by-batch while
   streaming the final parquet to disk.

`plot` reproduces the figures of `legacy/generate_plots.py` from the
generated parquet — per-month exit-code/duration/power distributions under
`<output-dir>/<YY_MM>/` and the aggregate 2×3 `pairplot` — with all
aggregation done in one streaming polars pass and months derived from `adt`.

### Known divergences from the original scripts

- The original crashed on non-numeric strings in numeric columns; this
  pipeline drops those rows instead.
- Pseudonym numbering starts after cleaning, so values occurring only in
  dropped rows don't consume a number (labels are opaque either way).
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
