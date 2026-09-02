"""S&P 500 point-in-time membership filter.

Keeps only (Ticker, Date) rows where the ticker was an actual S&P 500
constituent on that date. Having a price on a date is not the same as being
an index member on that date (e.g. a ticker delisted from the index in 2001
can still trade normally for years afterward, or rejoin later) — this filter
removes those false-candidate rows using cached membership interval history.

Data source: fja05680/sp500 GitHub repository's
"sp500_ticker_start_end.csv" (ticker, start_date, end_date), cached locally
as data/sp500_membership_history.csv. A ticker can appear on multiple rows
if it left and later rejoined the index (e.g. AAL: 1996-1997, 2015-2024).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_MEMBERSHIP_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "sp500_membership_history.csv"
)

# Each entry was confirmed via news/filings as "the same company" before
# being added by hand (kept in sync with the main project's
# src/get_tickers.py). Verification method: checked that the old ticker's
# last end_date and the new ticker's first start_date in the raw membership
# history line up with zero gap (boundary-date matching).
# Caveat: boundary-date matching alone cannot discover new renames -- an
# unrelated ticker replacing another on the same day produces the identical
# pattern (e.g. BSC exited the day ISRG entered; not a rename), so this
# method only sanity-checks mappings we already believe, never detects new
# ones.
#
# TODO: SEC EDGAR's free ticker<->CIK (SEC registration number) mapping
# would be a stronger check -- a company's CIK never changes even if its
# ticker/name does, so it verifies "same legal entity" far more directly
# than date patterns.
TRUSTED_RENAMES = {
    "ABC": "COR", "ANTM": "ELV", "BLL": "BALL", "CDAY": "DAY",
    "FB": "META", "FBHS": "FBIN", "FLT": "CPAY", "FISV": "FI",
    "GPS": "GAP", "NLOK": "GEN", "PKI": "RVTY", "RE": "EG",
    "VIAC": "PARA", "WLTW": "WTW", "WRK": "SW",
}


def norm_ticker(value: str) -> str:
    """Convert a symbol to the Yahoo convention (for example BRK.B -> BRK-B)."""
    return str(value).strip().upper().replace(".", "-")


def load_membership_history(path: Path = DEFAULT_MEMBERSHIP_PATH) -> pd.DataFrame:
    """Load and normalize ticker-level S&P 500 membership intervals."""
    membership = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    membership["ticker"] = membership["ticker"].map(norm_ticker)
    membership["ticker"] = membership["ticker"].map(lambda t: TRUSTED_RENAMES.get(t, t))
    return membership


def filter_by_membership(
    df: pd.DataFrame,
    ticker_col: str = "Ticker",
    date_col: str = "Date",
    membership_path: Path = DEFAULT_MEMBERSHIP_PATH,
) -> pd.DataFrame:
    """Keep only rows where ticker_col was an actual S&P 500 member on date_col."""
    membership = load_membership_history(membership_path)[["ticker", "start_date", "end_date"]]
    merged = df.merge(membership, left_on=ticker_col, right_on="ticker", how="left")
    is_member = (merged[date_col] >= merged["start_date"]) & (
        merged["end_date"].isna() | (merged[date_col] < merged["end_date"])
    )
    merged["_is_member"] = is_member.fillna(False)
    # The same (ticker, date) can match multiple membership intervals; count it
    # as eligible if any interval covers it.
    eligible = (
        merged.groupby([ticker_col, date_col])["_is_member"]
        .any()
        .rename("_eligible")
        .reset_index()
    )
    result = df.merge(eligible, on=[ticker_col, date_col], how="left")
    dropped = int((~result["_eligible"].fillna(False)).sum())
    if dropped:
        print(f"[INFO] excluded {dropped} / {len(df)} (ticker, date) rows: not an S&P 500 member on that date")
    return result.loc[result["_eligible"].fillna(False)].drop(columns="_eligible").reset_index(drop=True)
