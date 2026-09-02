"""Forward-return performance validation for clustered rebalancing snapshots.

For every rebalancing date T, each ticker's forward return is measured from
the Close on the trading day *after* T (T+1) to the Close on the trading day
*after* the next rebalancing date (so the holding period still matches the
file's own rebalancing interval: 60 trading days for rebalance_60df, 120 for
rebalance_120df -- only the entry/exit day is shifted by one). The last
rebalancing date has no next date, so it gets NaN and is dropped from
summaries.

Why T+1 and not T itself (2026-09-02 fix): the feature values at T (beta_Nd,
volatility_Nd, return_Nd, rsi_Nd) are rolling windows computed *inclusive* of
T's own Close (see feature.py's "on or before" contract), so a signal at T
already knows T's closing price. Using that same T Close as the trade price
means "observe the close, then trade at that exact same close" -- not
achievable in real trading, where the earliest you can act on a T-close
signal is the T+1 session. Shifting both entry and exit by one trading day
removes this. Note market_returns_like() in notebooks/10_full_backtest.ipynb
must use the identical T+1 convention for the benchmark, or the strategy vs.
benchmark comparison stops being apples-to-apples.

Terminal-exit policy for mid-holding-period delisting (2026-09-02 fix): if a
ticker has no price at its scheduled exit date, the old behavior was to
silently drop that row (NaN forward_return) -- an audit of every such case in
this project's data (60 tickers, 61 occurrences, ~0.3% of clustered rows)
found they're almost all M&A (CELG, TIF, WFM, RHT, XLNX, ...), not bankruptcy
or a data gap, and the last available close is typically already sitting at
(or very near) the deal price, since the market prices in an announced deal
well before the ticker stops trading. Dropping these silently understates
returns, since M&A exits skew flat-to-positive, not catastrophic.

The new policy, applied only when the scheduled exit price is missing:
1. If the ticker still shows S&P 500 membership at the *next* rebalancing
   date, this is NOT a real delisting (probably a data-provider gap) --
   leave it unresolved and excluded, same as before.
2. Otherwise, if there's a valid close between entry and the scheduled exit,
   force-liquidate there (``exit_method='terminal_last_close'``).
3. Otherwise, leave it unresolved and excluded.

This assumes cash liquidation at the last trade for every terminal exit,
including stock-for-stock mergers (e.g. XLNX -> AMD) where an investor could
in principle have held the acquirer's shares onward instead. That's a
deliberate simplifying assumption, not a claim every deal was all-cash --
documented here so it isn't mistaken for an oversight.

This answers: did the "stable cluster" (is_stable_cluster == True) actually
hold up better/worse than the rest, on average, across every independent
rebalancing decision made in the backtest?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .get_tickers import _load_membership_history

TRADING_DAYS_PER_YEAR = 252


def add_forward_returns(clustered_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach each row's forward return, entering/exiting one trading day
    after the rebalancing date (see module docstring for why), falling back
    to a terminal-last-close liquidation for genuine mid-holding delistings
    (see module docstring's "Terminal-exit policy").

    ``prices`` must have Date, Ticker, Close columns (e.g. final_df).

    Adds ``entry_date``, ``intended_exit_date`` (the scheduled T+1 exit),
    ``actual_exit_date``, ``exit_price``, and ``exit_method`` (one of
    ``scheduled_close``, ``terminal_last_close``, ``unresolved``) so a
    caller can audit exactly how each row's exit was resolved.
    """
    trading_dates = sorted(prices["Date"].unique())
    next_trading_day = dict(zip(trading_dates[:-1], trading_dates[1:]))

    dates = sorted(clustered_df["Date"].unique())
    next_date_map = dict(zip(dates[:-1], dates[1:]))

    df = clustered_df.copy()
    df["next_date"] = df["Date"].map(next_date_map)
    df["entry_date"] = df["Date"].map(next_trading_day)
    df["intended_exit_date"] = df["next_date"].map(next_trading_day)

    entry = prices[["Date", "Ticker", "Close"]].rename(
        columns={"Date": "entry_date", "Close": "entry_price"}
    )
    df = df.merge(entry, on=["entry_date", "Ticker"], how="left")

    scheduled = prices[["Date", "Ticker", "Close"]].rename(
        columns={"Date": "intended_exit_date", "Close": "exit_price"}
    )
    df = df.merge(scheduled, on=["intended_exit_date", "Ticker"], how="left")

    df["actual_exit_date"] = df["intended_exit_date"]
    df["exit_method"] = np.where(df["exit_price"].notna(), "scheduled_close", "unresolved")

    # 스케줄된 청산가가 없는 행만 강제청산 정책 대상 -- entry_price도 없거나
    # next_date가 없는(마지막 리밸런싱 시점) 행은 애초에 대상에서 제외.
    needs_fallback = (
        df["exit_price"].isna() & df["entry_price"].notna() & df["next_date"].notna()
    )
    if needs_fallback.any():
        membership = _load_membership_history()[["ticker", "start_date", "end_date"]]
        prices_sorted = prices.sort_values("Date")

        for idx in df.index[needs_fallback]:
            ticker = df.at[idx, "Ticker"]
            next_date = df.at[idx, "next_date"]
            entry_date = df.at[idx, "entry_date"]
            intended_exit = df.at[idx, "intended_exit_date"]

            m = membership[membership["ticker"] == ticker]
            still_member = bool(
                (
                    (next_date >= m["start_date"])
                    & (m["end_date"].isna() | (next_date < m["end_date"]))
                ).any()
            )
            if still_member:
                continue  # 데이터 공급자 결측 등으로 추정 -- 실제 상장폐지가 아니므로 unresolved 유지

            candidates = prices_sorted[
                (prices_sorted["Ticker"] == ticker)
                & (prices_sorted["Date"] >= entry_date)
                & (prices_sorted["Date"] < intended_exit)
            ]
            if candidates.empty:
                continue  # 보유기간 중 유효한 마지막 종가도 없음 -- unresolved 유지

            last_row = candidates.iloc[-1]
            df.at[idx, "actual_exit_date"] = last_row["Date"]
            df.at[idx, "exit_price"] = last_row["Close"]
            df.at[idx, "exit_method"] = "terminal_last_close"

    df["forward_return"] = df["exit_price"] / df["entry_price"] - 1.0
    return df.drop(columns=["entry_price"])


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


def _portfolio_return_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-date equal-weight portfolio return series for each group (2026-09-02
    fix): the mean forward return across every ticker held in that group on
    that date, i.e. exactly what an investor equal-weighting the group would
    have earned at each rebalancing snapshot.

    Used as the basis for risk stats instead of pooling raw (ticker, snapshot)
    forward returns. Pooling treats every ticker-date observation as an
    independent draw, so its std measures each stock's own (idiosyncratic)
    volatility. A real equal-weight portfolio's risk is lower than that,
    because on any given date the stocks' moves partly cancel out
    (diversification) -- std of the pooled observations overstates the
    portfolio's actual risk and understates its Sharpe ratio.
    """
    summary = summarize_by_group(df)
    return {
        "stable_cluster": summary["stable_mean_return"].dropna(),
        "other_clusters": summary["other_mean_return"].dropna(),
    }


def risk_adjusted_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-date equal-weight portfolio return per group, compared by
    return per unit of risk (mean / std) -- not just raw mean return.

    A low-volatility strategy is not supposed to win on raw return — the
    thesis is a smoother ride (lower std) for less return given up than
    that. This is what actually tests that thesis. std is computed on the
    date-level portfolio return series (see _portfolio_return_series), not
    pooled individual-stock returns, so it reflects diversified portfolio
    risk rather than idiosyncratic stock-level volatility.
    """
    series = _portfolio_return_series(df)
    stats = pd.DataFrame(
        {label: {"n": s.count(), "mean": s.mean(), "std": s.std()} for label, s in series.items()}
    ).T
    stats["return_per_risk"] = stats["mean"] / stats["std"]
    return stats


def sharpe_ratio_stats(
    df: pd.DataFrame,
    window: int,
    annual_risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """Annualized Sharpe ratio per group, computed from each group's
    date-level equal-weight portfolio return series (see
    _portfolio_return_series) rather than pooled (ticker, snapshot) returns
    -- pooling would measure idiosyncratic stock-level volatility instead of
    the diversified portfolio risk an equal-weight holder actually bears.

    This project doesn't collect an actual risk-free rate time series (no
    T-bill data source), so a constant annual rate is assumed and compounded
    down to the holding period length (``window`` trading days) before being
    subtracted from each observation's return. Treat the absolute Sharpe
    values as approximate — the stable-vs-other *comparison* is what matters,
    and that comparison is not sensitive to the exact risk-free assumption
    (subtracting the same constant from both groups barely moves their gap).
    """
    series = _portfolio_return_series(df)
    period_risk_free = (1 + annual_risk_free_rate) ** (window / TRADING_DAYS_PER_YEAR) - 1
    periods_per_year = TRADING_DAYS_PER_YEAR / window

    rows = {}
    for label, s in series.items():
        mean = s.mean()
        std = s.std()
        rows[label] = {
            "n": s.count(),
            "mean": mean,
            "std": std,
            "excess_mean": mean - period_risk_free,
            "sharpe_annualized": (mean - period_risk_free) / std * np.sqrt(periods_per_year),
        }
    return pd.DataFrame(rows).T
