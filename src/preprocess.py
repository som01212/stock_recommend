"""Clean collected OHLCV without creating ML features or technical indicators."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .collect_prices import END_DATE, START_DATE

SOURCE_PRIORITY = {"yahoo": 0, "yahoo:chart": 1, "tiingo": 2}
FINAL_COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]


def _norm_ticker(value: object) -> str:
    return str(value).strip().upper().replace(".", "-")


def _empty_missing() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "Ticker", "company", "sector", "fail_stage", "fail_reason",
        "n_rows_raw", "n_rows_final",
    ])


def _build_final_missing(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    universe: Optional[pd.DataFrame],
    collection_missing: Optional[pd.DataFrame],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Combine collection failures with symbols unusable after cleaning."""
    if universe is not None:
        info = universe.copy().rename(columns={"ticker": "Ticker"})
        info["Ticker"] = info["Ticker"].map(_norm_ticker)
        for column in ("company", "sector"):
            if column not in info:
                info[column] = "Unknown"
        info = info[["Ticker", "company", "sector"]].drop_duplicates("Ticker")
        targets = set(info["Ticker"])
    else:
        targets = set(raw["Ticker"].dropna())
        info = pd.DataFrame({"Ticker": sorted(targets), "company": "Unknown", "sector": "Unknown"})

    raw_counts = raw.groupby("Ticker", dropna=False).size()
    successful = set(cleaned["Ticker"])
    existing: dict[str, dict[str, object]] = {}
    if collection_missing is not None and len(collection_missing):
        prior = collection_missing.copy().rename(columns={"ticker": "Ticker"})
        prior["Ticker"] = prior["Ticker"].map(_norm_ticker)
        for _, row in prior.iterrows():
            existing[row["Ticker"]] = {
                "fail_stage": "collection",
                "fail_reason": row.get("fail_reason", "all collection fallbacks failed"),
            }

    rows = []
    period_start, period_end = pd.Timestamp(start), pd.Timestamp(end)
    for ticker in sorted(targets - successful):
        n_raw = int(raw_counts.get(ticker, 0))
        prior = existing.get(ticker)
        if prior is not None:
            stage, reason = str(prior["fail_stage"]), str(prior["fail_reason"])
        elif n_raw == 0:
            stage, reason = "collection", "수집된 가격 데이터가 없음"
        else:
            ticker_raw = raw.loc[raw["Ticker"] == ticker]
            valid_dates = pd.to_datetime(ticker_raw["Date"], errors="coerce")
            if not (valid_dates.ge(period_start) & valid_dates.lt(period_end)).any():
                stage, reason = "analysis_period", "분석 기간 내 관측치가 없음"
            else:
                stage, reason = "preprocessing", "날짜/종가 정제 후 유효한 OHLCV 행이 남지 않음"
        rows.append({
            "Ticker": ticker, "fail_stage": stage, "fail_reason": reason,
            "n_rows_raw": n_raw, "n_rows_final": 0,
        })

    if not rows:
        return _empty_missing()
    missing = pd.DataFrame(rows).merge(info, on="Ticker", how="left")
    missing[["company", "sector"]] = missing[["company", "sector"]].fillna("Unknown")
    return missing[[
        "Ticker", "company", "sector", "fail_stage", "fail_reason",
        "n_rows_raw", "n_rows_final",
    ]].sort_values("Ticker").reset_index(drop=True)


def build_ml_dataset(
    prices: pd.DataFrame,
    universe: Optional[pd.DataFrame] = None,
    final_missing_df: Optional[pd.DataFrame] = None,
    start: str = START_DATE,
    end: str = END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return cleaned prices, coverage, and unified final failures.

    A symbol is retained when at least one valid Date/Close row remains. Short
    history is reported in coverage_df and is never an exclusion criterion.
    """
    frame = prices.copy().rename(columns={
        "date": "Date", "ticker": "Ticker", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
    })
    missing_columns = sorted(set(FINAL_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"수집 원본 필수 컬럼 누락: {missing_columns}")
    if "source" not in frame:
        frame["source"] = "unknown"

    frame["Ticker"] = frame["Ticker"].map(_norm_ticker)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if isinstance(frame["Date"].dtype, pd.DatetimeTZDtype):
        frame["Date"] = frame["Date"].dt.tz_localize(None)
    frame["Date"] = frame["Date"].dt.normalize()
    for column in ("Open", "High", "Low", "Close", "Volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    raw_for_audit = frame.copy()
    frame = frame[
        frame["Date"].ge(pd.Timestamp(start)) & frame["Date"].lt(pd.Timestamp(end))
    ].copy()
    frame = frame.dropna(subset=["Date", "Ticker", "Close"])
    frame = frame[(frame["Ticker"] != "") & (frame["Ticker"] != "NAN") & (frame["Close"] > 0)]
    frame["_priority"] = frame["source"].map(SOURCE_PRIORITY).fillna(9)
    frame = (
        frame.sort_values(["Ticker", "Date", "_priority"])
        .drop_duplicates(["Ticker", "Date"], keep="first")
        .drop(columns="_priority")
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    frame["_price_imputed"] = frame[["Open", "High", "Low"]].isna().any(axis=1)
    for column in ("Open", "High", "Low"):
        frame[column] = frame[column].fillna(frame["Close"])
    frame["_volume_missing"] = frame["Volume"].isna()
    frame["Volume"] = frame["Volume"].fillna(0).clip(lower=0).round().astype("int64")
    frame["_ohlc_inconsistent"] = (
        (frame["High"] < frame["Low"])
        | (frame["Close"] > frame["High"]) | (frame["Close"] < frame["Low"])
        | (frame["Open"] > frame["High"]) | (frame["Open"] < frame["Low"])
    )
    ohlc = frame[["Open", "High", "Low", "Close"]]
    frame["High"] = ohlc.max(axis=1)
    frame["Low"] = ohlc.min(axis=1)

    expected_rows = max(1, int(np.busday_count(pd.Timestamp(start).date(), pd.Timestamp(end).date())))
    coverage = (
        frame.groupby("Ticker")
        .agg(
            n_rows=("Date", "size"), actual_start_date=("Date", "min"),
            actual_end_date=("Date", "max"), n_price_imputed=("_price_imputed", "sum"),
            n_ohlc_inconsistent=("_ohlc_inconsistent", "sum"),
            n_volume_missing=("_volume_missing", "sum"),
            sources=("source", lambda values: ",".join(sorted(set(values)))),
        )
        .reset_index()
    )
    coverage["expected_rows_10y"] = expected_rows
    coverage["coverage_10y"] = (coverage["n_rows"] / expected_rows).clip(upper=1).round(4)
    coverage["short_history"] = (
        (coverage["actual_start_date"] > pd.Timestamp(start) + pd.Timedelta(days=30))
        | (coverage["coverage_10y"] < 0.80)
    )
    coverage["has_quality_issue"] = (
        coverage[["n_price_imputed", "n_ohlc_inconsistent", "n_volume_missing"]].sum(axis=1) > 0
    )
    if universe is not None:
        info = universe.rename(columns={"ticker": "Ticker"}).copy()
        info["Ticker"] = info["Ticker"].map(_norm_ticker)
        info_columns = [column for column in ("company", "sector") if column in info]
        coverage = coverage.merge(info[["Ticker", *info_columns]].drop_duplicates("Ticker"), on="Ticker", how="left")
    coverage = coverage.sort_values("Ticker").reset_index(drop=True)

    unified_missing = _build_final_missing(raw_for_audit, frame, universe, final_missing_df, start, end)
    final_df = frame[[*FINAL_COLUMNS, "source"]].copy()
    return final_df, coverage, unified_missing
