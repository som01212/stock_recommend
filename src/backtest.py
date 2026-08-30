"""Time-series backtest: compound per-snapshot portfolio returns into a
cumulative equity curve, and compute drawdown/Sharpe from that curve.

notebooks 06-09 only pooled snapshot-level returns into a single mean/std
comparison — good for testing whether a selection rule has skill, but it
throws away the *sequence* of returns, so it can't show when a strategy
lost or gained, or how real capital would have evolved holding it. This
module restores that sequence.

Portfolio construction rule used everywhere here: equal weight across
every selected stock at each rebalancing date (simplest, most defensible
default given nothing in this project's data argues for a different
weighting scheme).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
DEFAULT_ROUND_TRIP_COST = 0.002  # 매수+매도 왕복 0.2% 가정 (유동성 높은 S&P500 종목 기준 수수료+스프레드+슬리피지 근사치)


def portfolio_returns(df: pd.DataFrame, selection_col: str, return_col: str = "forward_return") -> pd.Series:
    """Equal-weight portfolio return per rebalancing date: the mean forward
    return of every stock selected (``selection_col`` truthy) at that date."""
    selected = df[df[selection_col].astype(bool)]
    return selected.groupby("Date")[return_col].mean().sort_index()


def turnover_by_date(df: pd.DataFrame, selection_col: str) -> pd.Series:
    """Fraction of each date's holdings that are names not held the
    previous rebalancing date. The first date has no prior holdings to
    compare against, so it's excluded (portfolio inception, not an
    ongoing rebalancing cost)."""
    selected = df[df[selection_col].astype(bool)]
    holdings_by_date = selected.groupby("Date")["Ticker"].apply(set).sort_index()

    dates = holdings_by_date.index.tolist()
    turnover = {}
    for prev_date, date in zip(dates[:-1], dates[1:]):
        current = holdings_by_date[date]
        if not current:
            turnover[date] = 0.0
            continue
        new_names = current - holdings_by_date[prev_date]
        turnover[date] = len(new_names) / len(current)
    return pd.Series(turnover, name="turnover").sort_index()


def apply_transaction_costs(
    period_returns: pd.Series,
    df: pd.DataFrame,
    selection_col: str,
    round_trip_cost: float = DEFAULT_ROUND_TRIP_COST,
) -> pd.Series:
    """Subtract a round-trip trading cost from each period's return, scaled
    by that period's turnover (what fraction of the equal-weight portfolio
    actually had to be traded). The very first period keeps its gross
    return — there's no prior portfolio to have turned over from."""
    turnover = turnover_by_date(df, selection_col)
    cost = turnover.reindex(period_returns.index).fillna(0.0) * round_trip_cost
    net_returns = period_returns - cost
    net_returns.name = "net_return"
    return net_returns


def equity_curve(period_returns: pd.Series) -> pd.Series:
    """Cumulative compounded equity, starting at 1.0."""
    return (1 + period_returns).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min()


def annualized_sharpe(period_returns: pd.Series, window: int, annual_risk_free_rate: float = 0.02) -> float:
    period_rf = (1 + annual_risk_free_rate) ** (window / TRADING_DAYS_PER_YEAR) - 1
    periods_per_year = TRADING_DAYS_PER_YEAR / window
    excess = period_returns - period_rf
    return excess.mean() / period_returns.std() * np.sqrt(periods_per_year)


def summarize(period_returns: pd.Series, window: int, annual_risk_free_rate: float = 0.02) -> pd.Series:
    equity = equity_curve(period_returns)
    return pd.Series(
        {
            "n_periods": len(period_returns),
            "total_return": equity.iloc[-1] - 1.0,
            "annualized_sharpe": annualized_sharpe(period_returns, window, annual_risk_free_rate),
            "max_drawdown": max_drawdown(equity),
            "win_rate": (period_returns > 0).mean(),
        }
    )
