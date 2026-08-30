"""Point-in-time trailing P/E ratio, built from yfinance's annual financials.

yfinance's free tier only exposes about 5 years of annual financial
statements per ticker, which is exactly why this module is scoped to the
PER experiment's 2021-2026 window (see notebooks/09_per_experiment.ipynb)
rather than the main 10-year pipeline.

"Point-in-time" matters here: a fiscal year's EPS isn't public knowledge
the day the fiscal year ends — the annual report is filed weeks to months
later. Using the fiscal year-end date directly as the point PER becomes
available would leak future information into earlier rebalancing
snapshots (lookahead bias). REPORTING_LAG_DAYS approximates the filing
delay so only EPS that would have actually been public by a given
rebalancing date is used.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "cache" / "eps_history.csv"
REPORTING_LAG_DAYS = 90


def fetch_eps_history(tickers: list[str], force_refresh: bool = False, sleep: float = 0.3) -> pd.DataFrame:
    """Fetch each ticker's annual diluted (or basic) EPS history.

    Caches to disk — fetching ~600 tickers one at a time takes several
    minutes and yfinance has no bulk endpoint for this.
    """
    if CACHE_PATH.is_file() and not force_refresh:
        return pd.read_csv(CACHE_PATH, parse_dates=["fiscal_date"])

    rows = []
    failed = []
    for ticker in tickers:
        try:
            financials = yf.Ticker(ticker).financials
            eps_row = None
            for candidate in ("Diluted EPS", "Basic EPS"):
                if candidate in financials.index:
                    eps_row = financials.loc[candidate]
                    break
            if eps_row is None:
                failed.append(ticker)
                continue
            for fiscal_date, eps in eps_row.items():
                rows.append({"Ticker": ticker, "fiscal_date": fiscal_date, "eps": eps})
        except Exception:
            failed.append(ticker)
        time.sleep(sleep)

    print(f"[fundamentals] EPS 확보: {len(tickers) - len(failed)}/{len(tickers)}종목, 실패 {len(failed)}종목")
    eps_df = pd.DataFrame(rows)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    eps_df.to_csv(CACHE_PATH, index=False)
    return eps_df


def add_point_in_time_per(df: pd.DataFrame, eps_df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Attach trailing P/E as of each row's Date, using only EPS that was
    actually public knowledge by then (fiscal_date + REPORTING_LAG_DAYS)."""
    df = df.copy()
    eps_df = eps_df.dropna(subset=["eps"]).copy()
    eps_df["available_date"] = eps_df["fiscal_date"] + pd.Timedelta(days=REPORTING_LAG_DAYS)

    matched = []
    for ticker, group in df.groupby("Ticker", sort=False):
        ticker_eps = eps_df.loc[eps_df["Ticker"] == ticker, ["available_date", "eps"]].sort_values("available_date")
        # merge_asof drops the index, so carry it through as a column and restore it after
        group_sorted = group.sort_values("Date").reset_index(names="_orig_index")
        if ticker_eps.empty:
            merged = group_sorted.assign(trailing_eps=float("nan"))
        else:
            merged = pd.merge_asof(
                group_sorted, ticker_eps, left_on="Date", right_on="available_date", direction="backward",
            ).rename(columns={"eps": "trailing_eps"})
        matched.append(merged.set_index("_orig_index"))

    result = pd.concat(matched).loc[df.index]
    result.index.name = df.index.name
    result["per"] = result[price_col] / result["trailing_eps"]
    result.loc[result["trailing_eps"] <= 0, "per"] = float("nan")  # 적자 기업 PER은 의미 없음
    return result.drop(columns=["available_date"], errors="ignore")
