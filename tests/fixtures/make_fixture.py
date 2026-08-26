"""Synthetic raw Fugaku job log fixture for pipeline tests.

Produces the raw-dump column set (cr_* names, junk columns) with ~4 months of
adt, repeated users/job names (to exercise first-appearance numbering), and a
tail of dirty rows: empty sdt, epoch edt, non-numeric perf1 (all dropped by
cleaning) plus a null usr row (kept, anonymized to null).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

N_CLEAN = 2000
N_DROPPED_DIRTY = 3  # empty sdt, 1970 edt, non-numeric perf1
N_KEPT_DIRTY = 1  # null usr
EXPECTED_ROWS = N_CLEAN + N_KEPT_DIRTY

_BASE = datetime(2021, 3, 1, 0, 0, 0)


def _ts(seconds: int) -> str:
    return (_BASE + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")


def make_raw_df(seed: int = 7) -> pl.DataFrame:
    rng = random.Random(seed)
    users = [f"user{k:02d}" for k in range(20)]
    jobs = [f"job-{k}" for k in range(50)]
    envs = ["vn", "hy", "th"]

    rows = []
    for i in range(N_CLEAN):
        t = i * 5000  # ~116 days total: 2021-03 .. 2021-06
        elp = float(rng.randint(60, 86400))
        nnuma = rng.randint(1, 4)
        minp = rng.uniform(20, 100)
        rows.append(
            {
                "jid": f"J{i:06d}",
                "usr": rng.choice(users),
                "jnam": rng.choice(jobs),
                "cnumr": 48, "cnumat": 48, "cnumut": rng.randint(1, 48),
                "nnumr": nnuma,
                "adt": _ts(t),
                "qdt": _ts(t + 1), "schedsdt": _ts(t + 2), "deldt": _ts(t + 100),
                "ec": rng.choice([0, 0, 0, 0, 1, 2, 512]),
                "elpl": 86400.0,
                "elp": elp,
                "sdt": _ts(t + 60),
                "edt": _ts(t + 60 + int(elp)),
                "nnuma": nnuma,
                "idle_time_ave": rng.uniform(0, 30),
                "nnumu": nnuma,
                "perf1": f"{rng.uniform(1e9, 1e12):.1f}",
                "perf2": rng.uniform(1e9, 1e12),
                "perf3": rng.uniform(1e9, 1e12),
                "perf4": rng.uniform(1e6, 1e9),
                "perf5": rng.uniform(1e6, 1e9),
                "perf6": rng.uniform(1e6, 1e9),
                "mszl": 28000.0,
                "pri": rng.randint(0, 5),
                "econ": rng.uniform(1e3, 1e6),
                "avgpcon": (minp + 10) * nnuma,
                "minpcon": minp * nnuma,
                "maxpcon": (minp + 20) * nnuma,
                "msza": rng.randint(1000, 28000),
                "mmszu": rng.uniform(100, 28000),
                "uctmut": rng.uniform(0, 1e6),
                "sctmut": rng.uniform(0, 1e4),
                "usctmut": rng.uniform(0, 1e6),
                "cr_jobenv_req": rng.choice(envs),
                "cr_freq_req": 2200, "cr_freq_alloc": 2000,
                "ermsg": "", "fjprofiler": "",
            }
        )

    template = dict(rows[-1])
    t_dirty = N_CLEAN * 5000
    dirty = [
        {**template, "jid": "Jdirty0", "adt": _ts(t_dirty), "sdt": ""},
        {**template, "jid": "Jdirty1", "adt": _ts(t_dirty + 1), "edt": "1970-01-01 00:00:00"},
        {**template, "jid": "Jdirty2", "adt": _ts(t_dirty + 2), "perf1": "abc"},
        {**template, "jid": "Jkept0", "adt": _ts(t_dirty + 3), "usr": None},
    ]
    return pl.DataFrame(rows + dirty)


def write_fixture(out_dir: Path, seed: int = 7) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = make_raw_df(seed)
    csv_path = out_dir / "raw.csv"
    parquet_path = out_dir / "raw.parquet"
    df.write_csv(csv_path)
    df.write_parquet(parquet_path)
    return csv_path, parquet_path


if __name__ == "__main__":
    write_fixture(Path(__file__).parent)
