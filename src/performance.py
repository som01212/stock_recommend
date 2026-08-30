"""Forward-return performance validation for clustered rebalancing snapshots.

For every rebalancing date, each ticker's forward return is measured from
that date's Close to the *next* rebalancing date's Close in the same file
(so the holding period matches the file's own rebalancing interval:
60 trading days for rebalance_60df, 120 for rebalance_120df). The last
rebalancing date has no next date, so it gets NaN and is dropped from
summaries.

This answers: did the "stable cluster" (is_stable_cluster == True) actually
hold up better/worse than the rest, on average, across every independent
rebalancing decision made in the backtest?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def add_forward_returns(clustered_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach each row's forward return to the next rebalancing date.

    ``prices`` must have Date, Ticker, Close columns (e.g. final_df).
    """
    dates = sorted(clustered_df["Date"].unique())
    next_date_map = dict(zip(dates[:-1], dates[1:]))

    df = clustered_df.copy()
    df["next_date"] = df["Date"].map(next_date_map)

    entry = prices[["Date", "Ticker", "Close"]].rename(columns={"Close": "entry_price"})
    df = df.merge(entry, on=["Date", "Ticker"], how="left")

    exit_ = prices[["Date", "Ticker", "Close"]].rename(
        columns={"Date": "next_date", "Close": "exit_price"}
    )
    df = df.merge(exit_, on=["next_date", "Ticker"], how="left")

    df["forward_return"] = df["exit_price"] / df["entry_price"] - 1.0
    return df.drop(columns=["entry_price", "exit_price"])


def summarize_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Per-date mean forward return, stable cluster vs the rest.

    Rows with cluster == -1 (warmup period, not enough history) or a
    missing forward return (last rebalancing date) are excluded.
    """
    valid = df[(df["cluster"] != -1) & df["forward_return"].notna()]
    summary = (
        valid.groupby(["Date", "is_stable_cluster"])["forward_return"]
        .mean()
        .unstack("is_stable_cluster")
        .rename(columns={True: "stable_mean_return", False: "other_mean_return"})
    )
    summary["stable_minus_other"] = summary["stable_mean_return"] - summary["other_mean_return"]
    return summary


def overall_stats(summary: pd.DataFrame) -> pd.Series:
    """Aggregate the per-date comparison into headline strategy stats."""
    diff = summary["stable_minus_other"].dropna()
    return pd.Series(
        {
            "n_snapshots": len(diff),
            "stable_mean_return": summary["stable_mean_return"].mean(),
            "other_mean_return": summary["other_mean_return"].mean(),
            "avg_stable_minus_other": diff.mean(),
            "win_rate": (diff > 0).mean(),
        }
    )


def risk_adjusted_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Pool every (ticker, snapshot) forward return by group and compare
    return per unit of risk (mean / std), not just raw mean return.

    A low-volatility strategy is not supposed to win on raw return — the
    thesis is a smoother ride (lower std) for less return given up than
    that. This is what actually tests that thesis.
    """
    valid = df[(df["cluster"] != -1) & df["forward_return"].notna()]
    grouped = valid.groupby("is_stable_cluster")["forward_return"]
    stats = grouped.agg(n="count", mean="mean", std="std")
    stats["return_per_risk"] = stats["mean"] / stats["std"]
    return stats.rename(index={True: "stable_cluster", False: "other_clusters"})


def sharpe_ratio_stats(
    df: pd.DataFrame,
    window: int,
    annual_risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """Annualized Sharpe ratio per group, pooled across (ticker, snapshot).

    This project doesn't collect an actual risk-free rate time series (no
    T-bill data source), so a constant annual rate is assumed and compounded
    down to the holding period length (``window`` trading days) before being
    subtracted from each observation's return. Treat the absolute Sharpe
    values as approximate — the stable-vs-other *comparison* is what matters,
    and that comparison is not sensitive to the exact risk-free assumption
    (subtracting the same constant from both groups barely moves their gap).
    """
    valid = df[(df["cluster"] != -1) & df["forward_return"].notna()]
    period_risk_free = (1 + annual_risk_free_rate) ** (window / TRADING_DAYS_PER_YEAR) - 1
    periods_per_year = TRADING_DAYS_PER_YEAR / window

    grouped = valid.groupby("is_stable_cluster")["forward_return"]
    stats = grouped.agg(n="count", mean="mean", std="std")
    stats["excess_mean"] = stats["mean"] - period_risk_free
    stats["sharpe_annualized"] = stats["excess_mean"] / stats["std"] * np.sqrt(periods_per_year)
    return stats.rename(index={True: "stable_cluster", False: "other_clusters"})
