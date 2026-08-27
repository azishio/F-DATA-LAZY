"""Schema constants for the F-DATA pipeline.

The output contract is docs/feature_list.csv (the 45 public columns of the
Zenodo release) plus the `split` column added by this app.
"""

# Raw dumps name some columns with a cr_ prefix; the published schema does not.
# Inputs may carry either name.
RAW_RENAMES = {
    "cr_jobenv_req": "jobenv_req",
    "cr_freq_req": "freq_req",
    "cr_freq_alloc": "freq_alloc",
}

# Columns replaced by first-appearance sequential pseudonyms ("usr_0", ...).
ANON_FEATURES = ["jid", "usr", "jnam", "jobenv_req"]

# Numeric casts that drop the row when the value is null/empty/unparseable
# (clean_and_anonmyze_data.py filtered null/"" then applied float()/int()).
FLOAT_COLS = [
    "perf1", "perf2", "perf3", "perf4", "perf5", "perf6",
    "elpl", "elp", "idle_time_ave", "econ",
    "avgpcon", "minpcon", "maxpcon", "mmszu", "mszl",
]
INT_COLS = [
    "cnumr", "cnumat", "cnumut", "nnumr", "nnuma", "nnumu",
    "freq_req", "freq_alloc", "ec", "pri", "msza",
]
# Cast without dropping rows (no cast rule in the original scripts; typed as
# float in docs/feature_list.csv).
SOFT_FLOAT_COLS = ["uctmut", "sctmut", "usctmut"]

# Rows with a null/empty/epoch ("1970-...") value in these are dropped.
DATE_FILTER_COLS = ["adt", "sdt", "edt"]

# All datetime columns stay strings in the published data (the to_datetime
# call in the original script is commented out); we never convert them.
DATE_COLS = ["adt", "qdt", "schedsdt", "deldt", "sdt", "edt"]

# Fugaku roofline ridge point: peak GFLOPs / memory bandwidth GiB/s
# (generate_derived_features.py).
RIDGE_POINT = 537_000_000 / 163_000_000

# Sentence-BERT input: comma-joined non-empty original (pre-anonymization)
# values of these columns. Derived before anonymization, so original names.
EMBED_SOURCE = ["usr", "jnam", "jobenv_req"]
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
EMBED_TEXT_COL = "_embed_text"

# Final column order: docs/feature_list.csv verbatim.
FINAL_COLUMNS = [
    "jid", "usr", "jnam",
    "cnumr", "cnumat", "cnumut", "nnumr",
    "adt", "qdt", "schedsdt", "deldt",
    "ec", "elpl", "sdt", "edt",
    "nnuma", "idle_time_ave", "nnumu",
    "perf1", "perf2", "perf3", "perf4", "perf5", "perf6",
    "mszl", "pri", "econ",
    "avgpcon", "minpcon", "maxpcon",
    "msza", "mmszu",
    "uctmut", "sctmut", "usctmut",
    "jobenv_req", "freq_req", "freq_alloc",
    "flops", "mbwidth", "opint", "pclass",
    "embedding", "exit state", "duration",
]

SPLIT_COL = "split"
OUTPUT_COLUMNS = FINAL_COLUMNS + [SPLIT_COL]

# Raw input columns required after cr_* renames: every non-derived final
# column, plus elp (renamed to duration at the end).
_DERIVED = {"flops", "mbwidth", "opint", "pclass", "embedding", "exit state", "duration"}
RAW_INPUT_COLUMNS = [c for c in FINAL_COLUMNS if c not in _DERIVED] + ["elp"]

# Columns written by the phase-1 lazy sink: every final column except
# embedding and split (both applied in the final pass), with the anonymized
# features still carrying their original values.
INTERMEDIATE_COLUMNS = [
    c for c in FINAL_COLUMNS if c != "embedding"
] + [EMBED_TEXT_COL]
