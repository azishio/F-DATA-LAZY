"""Plot subcommand: per-month and aggregate figures from the generated parquet.

Reproduces generate_plots.py, but every figure is drawn from a small polars
aggregation (one streaming pass computes them all) instead of row-level
seaborn histplots. The month key is derived from adt ("2021-03-..." ->
"21_03"), matching the filename-derived key of the original.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

PCON_SERIES = ["nminpcon", "navgpcon", "nmaxpcon"]
PCON_LABELS = {"nminpcon": "minpcon", "navgpcon": "avgpcon", "nmaxpcon": "maxpcon"}
NNUMA_EDGES = [0, 10, 100, 1000, 10000, 100000, 200000]
NNUMA_LABELS = ["[1, 10)", "[10, 100)", "[100, 1K)", "[1k, 10K)", "[10K, 100K)", "[100K, 1M)"]


def _ym_expr() -> pl.Expr:
    return (
        pl.col("adt").str.slice(2, 5).str.replace("-", "_", literal=True).alias("ym")
    )


def _dur_min() -> pl.Expr:
    return (pl.col("duration") / 60).cast(pl.Int64).alias("dur_min")


def _nnuma_bucket() -> pl.Expr:
    expr = pl.lit(None, dtype=pl.Utf8)
    for lo, hi, label in zip(NNUMA_EDGES[:-1], NNUMA_EDGES[1:], NNUMA_LABELS):
        expr = (
            pl.when((pl.col("nnuma") >= lo) & (pl.col("nnuma") < hi))
            .then(pl.lit(label))
            .otherwise(expr)
        )
    return expr.alias("nnuma_bucket")


def _aggregate(lf: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    lf = lf.with_columns(_ym_expr())
    # Same per-node power normalization and (10, 300) filters as the original
    # (generate_plots.py lines 61-67). The original accumulated its aggregate
    # stats from the already power-filtered rows, so the aggregate panels use
    # lf_p, while the per-month ec/duration plots use the unfiltered frame.
    lf_p = lf.with_columns(
        (pl.col(s.removeprefix("n")) / pl.col("nnuma")).alias(s) for s in PCON_SERIES
    ).filter(
        pl.all_horizontal((pl.col(s) > 10) & (pl.col(s) < 300) for s in PCON_SERIES)
    )

    plans = {
        "ec": lf.filter(pl.col("ec").is_not_null()).group_by("ym", "ec").len(),
        "dur": lf.filter(pl.col("duration").is_not_null())
        .group_by("ym", _dur_min())
        .len(),
        "jobs": lf_p.group_by("ym").len(),
        "nnuma": lf_p.group_by(_nnuma_bucket()).len(),
        "dur_agg": lf_p.group_by(_dur_min()).len(),
        "exit": lf_p.group_by("ym", "exit state").len(),
        "pclass": lf_p.group_by("ym", "pclass").len(),
    }
    for s in PCON_SERIES:
        plans[f"pcon_{s}"] = (
            lf_p.group_by("ym", pl.col(s).floor().cast(pl.Int64).alias("watt"))
            .len()
        )
    keys = list(plans)
    frames = pl.collect_all([plans[k] for k in keys], engine="streaming")
    return dict(zip(keys, frames))


def _save(out_dir: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.png")
    plt.savefig(out_dir / f"{name}.pdf", format="pdf")
    plt.clf()


def _weighted_hist(ax, values: np.ndarray, weights: np.ndarray, bins: int = 50, **bar_kw):
    if len(values) == 0:
        return
    counts, edges = np.histogram(values, bins=bins, weights=weights)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", **bar_kw)


def _month_axis(ax, n_labels: int) -> None:
    labels = [t.get_text() for t in ax.get_xticklabels()]
    ax.set_xticks(
        ax.get_xticks(),
        [labels[i] if i % 3 == 0 else "" for i in range(len(labels))],
        rotation=45,
    )


def _stacked_by_month(ax, df: pl.DataFrame, hue: str) -> None:
    """Stacked per-month bars with '(total)' legend labels like the original."""
    months = sorted(df["ym"].unique().to_list())
    totals = df.group_by(hue).agg(pl.col("len").sum()).sort(hue)
    bottom = np.zeros(len(months))
    x = np.arange(len(months))
    for value, total in totals.iter_rows():
        counts = dict(df.filter(pl.col(hue) == value).select("ym", "len").iter_rows())
        y = np.array([counts.get(m, 0) for m in months], dtype=float)
        ax.bar(x, y, bottom=bottom, label=f"{value} ({total})")
        bottom += y
    ax.set_xticks(x, months)
    ax.legend()
    _month_axis(ax, len(months))


def render(input_path: Path, output_dir: Path) -> None:
    sns.set_style("whitegrid")
    aggs = _aggregate(pl.scan_parquet(input_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    months = sorted(aggs["ec"]["ym"].unique().to_list())
    for ym in months:
        ym_dir = output_dir / ym
        ym_dir.mkdir(exist_ok=True)

        # Exit code distribution: top 9 codes + Others, log y
        ec = aggs["ec"].filter(pl.col("ym") == ym).sort("len", descending=True)
        top, rest = ec.head(9), ec.slice(9)
        x_v = [str(v) for v in top["ec"].to_list()] + ["Others"]
        y_v = top["len"].to_list() + [rest["len"].sum()]
        sns.barplot(x=x_v, y=y_v)
        plt.ylabel("# of jobs")
        plt.xlabel("Exit code of the job")
        plt.yscale("log")
        plt.xticks(rotation=45)
        _save(ym_dir, "ec_distribution")

        # Duration distribution (minutes)
        dur = aggs["dur"].filter(pl.col("ym") == ym)
        ax = plt.gca()
        _weighted_hist(ax, dur["dur_min"].to_numpy(), dur["len"].to_numpy())
        ax.set_ylabel("# of jobs")
        ax.set_xlabel("Duration (in minutes)")
        ax.set_yscale("log")
        _save(ym_dir, "dr_distribution")

        # Per-node power consumption, min/avg/max overlaid
        ax = plt.gca()
        for s in PCON_SERIES:
            pc = aggs[f"pcon_{s}"].filter(pl.col("ym") == ym)
            ax.bar(
                pc["watt"].to_numpy(),
                pc["len"].to_numpy(),
                width=1,
                align="edge",
                alpha=0.7,
                label=PCON_LABELS[s],
            )
        ax.legend()
        ax.set_ylabel("# of jobs")
        ax.set_xlabel("Power consumption (in Watts)")
        ax.set_yscale("log")
        _save(ym_dir, "pcon")

    # Aggregate 2x3 figure (panels a-f of the original pairplot)
    fig = plt.figure(figsize=(21, 13))
    axes = [fig.add_subplot(2, 3, i + 1) for i in range(6)]
    ax1, ax2, ax3, ax4, ax5, ax6 = axes

    jobs = aggs["jobs"].sort("ym")
    ax1.bar(jobs["ym"].to_list(), jobs["len"].to_numpy())
    ax1.set_ylabel("# of jobs")
    ax1.set_xlabel("Year month")
    _month_axis(ax1, len(jobs))
    ax1.set_title("a)", y=-0.175)

    nnuma = {b: n for b, n in aggs["nnuma"].iter_rows() if b is not None}
    ax2.bar(NNUMA_LABELS, [nnuma.get(b, 0) for b in NNUMA_LABELS])
    ax2.set_ylabel("# of jobs")
    ax2.set_xlabel("# of nodes allocated")
    ax2.set_yscale("log")
    ax2.set_title("b)", y=-0.175)

    dur = aggs["dur_agg"]
    _weighted_hist(ax3, dur["dur_min"].to_numpy(), dur["len"].to_numpy())
    ax3.set_ylabel("# of jobs")
    ax3.set_xlabel("Duration (in minutes)")
    ax3.set_yscale("log")
    ax3.set_title("c)", y=-0.175)

    _stacked_by_month(ax4, aggs["exit"], "exit state")
    ax4.set_ylabel("# of jobs")
    ax4.set_xlabel("Year month")
    ax4.set_title("d)", y=-0.175)

    _stacked_by_month(ax5, aggs["pclass"], "pclass")
    ax5.set_ylabel("# of jobs")
    ax5.set_xlabel("Year month")
    ax5.set_title("e)", y=-0.175)

    for s in PCON_SERIES:
        pc = (
            aggs[f"pcon_{s}"]
            .group_by("watt")
            .agg(pl.col("len").sum())
            .sort("watt")
        )
        ax6.bar(
            pc["watt"].to_numpy(),
            pc["len"].to_numpy(),
            width=1,
            align="edge",
            alpha=0.35,
            label=PCON_LABELS[s],
        )
    ax6.legend()
    ax6.set_ylabel("# of jobs")
    ax6.set_xlabel("Power consumption (in Watts)")
    ax6.set_yscale("log")
    ax6.set_title("f)", y=-0.175)

    plt.tight_layout()
    plt.savefig(output_dir / "pairplot.png", dpi=300)
    plt.savefig(output_dir / "pairplot.pdf", format="pdf")
    plt.clf()
    plt.close(fig)
