import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from make_fixture import EXPECTED_ROWS, write_fixture  # noqa: E402

from fdata import cli  # noqa: E402
from fdata.schema import EMBED_DIM, FINAL_COLUMNS, OUTPUT_COLUMNS, RIDGE_POINT  # noqa: E402

TRAIN_RATIO = 0.8


class DummyModel:
    """Stands in for SentenceTransformer so tests need no torch download.

    Vectors are a function of the text (first dim = text length) so tests
    can verify that the deduplicated encoding assigns each row the vector
    of its own text.
    """

    def encode(self, texts, **kwargs):
        vecs = np.full((len(texts), EMBED_DIM), 0.5, dtype=np.float32)
        vecs[:, 0] = [len(t) for t in texts]
        return vecs


def _patch_model(mp: pytest.MonkeyPatch) -> None:
    mp.setattr("fdata.cli.load_model", lambda backend: DummyModel())


@pytest.fixture(scope="session")
def fixture_paths(tmp_path_factory):
    return write_fixture(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def generated(fixture_paths, tmp_path_factory) -> Path:
    csv_path, _ = fixture_paths
    out = tmp_path_factory.mktemp("out") / "fdata.parquet"
    mp = pytest.MonkeyPatch()
    _patch_model(mp)
    try:
        cli.main(
            [
                "generate",
                "--input", str(csv_path),
                "--output", str(out),
                "--train-ratio", str(TRAIN_RATIO),
                "--batch-size", "512",
            ]
        )
    finally:
        mp.undo()
    return out


@pytest.fixture(scope="session")
def df(generated) -> pl.DataFrame:
    return pl.read_parquet(generated)


def test_output_schema(df):
    assert df.columns == OUTPUT_COLUMNS
    assert df.schema["embedding"] == pl.List(pl.Float32)
    assert df.schema["ec"] == pl.Int64
    assert df.schema["perf1"] == pl.Float64
    assert df.schema["duration"] == pl.Float64
    for c in ["adt", "qdt", "schedsdt", "deldt", "sdt", "edt"]:
        assert df.schema[c] == pl.Utf8, c
    assert df["embedding"].list.len().unique().to_list() == [EMBED_DIM]


def test_dirty_rows_dropped(df):
    # 3 dirty rows (empty sdt, epoch edt, non-numeric perf1) dropped,
    # null-usr row kept.
    assert len(df) == EXPECTED_ROWS
    assert df["usr"].null_count() == 1


def test_anonymization(df):
    assert df["usr"][0] == "usr_0"
    assert df["jid"][0] == "jid_0"
    # jid is unique per row, so labels are dense 0..n-1 in row order
    assert df["jid"].to_list()[:3] == ["jid_0", "jid_1", "jid_2"]
    non_null_usr = df["usr"].drop_nulls()
    assert set(non_null_usr.to_list()) == {f"usr_{i}" for i in range(20)}
    assert set(df["jobenv_req"].to_list()) <= {"jobenv_req_0", "jobenv_req_1", "jobenv_req_2"}


def test_split_is_temporal(df):
    train = df.filter(pl.col("split") == "train")
    test = df.filter(pl.col("split") == "test")
    assert len(train) + len(test) == len(df)
    assert train["adt"].max() <= test["adt"].min()
    realized = len(train) / len(df)
    assert TRAIN_RATIO <= realized < TRAIN_RATIO + 0.01


def test_derived_features(df):
    denom = df["duration"] - df["idle_time_ave"]
    flops = (df["perf2"] + df["perf3"] * 4) / denom
    mbwidth = (((df["perf4"] + df["perf5"]) / 12) * 256) / denom
    assert np.allclose(df["flops"].to_numpy(), flops.to_numpy())
    assert np.allclose(df["mbwidth"].to_numpy(), mbwidth.to_numpy())
    assert np.allclose(df["opint"].to_numpy(), (flops / mbwidth).to_numpy())
    expected_pclass = (
        pl.when(df["opint"] >= RIDGE_POINT)
        .then(pl.lit("compute-bound"))
        .otherwise(pl.lit("memory-bound"))
    )
    assert df["pclass"].to_list() == pl.select(expected_pclass).to_series().to_list()
    assert df.filter(pl.col("ec") == 0)["exit state"].unique().to_list() == ["completed"]
    assert df.filter(pl.col("ec") != 0)["exit state"].unique().to_list() == ["failed"]


def test_deduplicated_embeddings(df):
    # Every row's vector must be the one for its own text: first dim equals
    # the length of the comma-joined original (usr, jnam, jobenv_req), which
    # is constant within an anonymized triple. Rows sharing a triple share
    # an identical vector; different text lengths give different vectors.
    firsts = df.select(
        pl.col("embedding").list.first().alias("e0"), "usr", "jnam", "jobenv_req"
    )
    per_triple = firsts.group_by("usr", "jnam", "jobenv_req").agg(
        pl.col("e0").n_unique()
    )
    assert per_triple["e0"].max() == 1
    assert firsts["e0"].n_unique() > 1
    assert firsts["e0"].min() > 0  # the null-usr row still embeds jnam+env


def test_final_columns_match_docs():
    docs = pl.read_csv(Path(__file__).parent.parent / "docs" / "feature_list.csv")
    assert FINAL_COLUMNS == docs["Column"].to_list()


def test_parquet_input(fixture_paths, tmp_path):
    _, parquet_path = fixture_paths
    out = tmp_path / "fdata_from_parquet.parquet"
    mp = pytest.MonkeyPatch()
    _patch_model(mp)
    try:
        cli.main(["generate", "--input", str(parquet_path), "--output", str(out)])
    finally:
        mp.undo()
    df2 = pl.read_parquet(out)
    assert len(df2) == EXPECTED_ROWS
    assert df2.columns == OUTPUT_COLUMNS


def test_plot(generated, tmp_path):
    plot_dir = tmp_path / "plots"
    cli.main(["plot", "--input", str(generated), "--output-dir", str(plot_dir)])
    for ym in ["21_03", "21_04", "21_05", "21_06"]:
        for name in ["ec_distribution", "dr_distribution", "pcon"]:
            assert (plot_dir / ym / f"{name}.png").exists()
            assert (plot_dir / ym / f"{name}.pdf").exists()
    assert (plot_dir / "pairplot.png").exists()
    assert (plot_dir / "pairplot.pdf").exists()
