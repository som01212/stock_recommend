"""Validation checks for cleaned S&P 500 OHLCV data."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]


def validate_ml_dataset(
    final_df: pd.DataFrame,
    final_missing_df: pd.DataFrame,
    sp500_universe: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> dict[str, object]:
    """Validate schema, quality, and universe success/failure accounting."""
    errors: list[str] = []
    warnings: list[str] = []
    absent = sorted(set(REQUIRED_COLUMNS) - set(final_df.columns))
    if absent:
        raise AssertionError(f"필수 컬럼 누락: {absent}")
    if not pd.api.types.is_datetime64_any_dtype(final_df["Date"]):
        errors.append(f"Date dtype이 datetime이 아님: {final_df['Date'].dtype}")
    elif isinstance(final_df["Date"].dtype, pd.DatetimeTZDtype):
        errors.append("Date에 timezone 정보가 남아 있음")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if not pd.api.types.is_numeric_dtype(final_df[column]):
            errors.append(f"{column} dtype이 숫자형이 아님: {final_df[column].dtype}")
    duplicates = int(final_df.duplicated(["Ticker", "Date"]).sum())
    if duplicates:
        errors.append(f"(Ticker, Date) 중복 {duplicates}건")
    if not final_df.reset_index(drop=True).equals(
        final_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    ):
        errors.append("Ticker/Date 정렬이 깨져 있음")
    for column in REQUIRED_COLUMNS:
        nulls = int(final_df[column].isna().sum())
        if nulls:
            errors.append(f"{column} 결측 {nulls}건")
    if (final_df["Close"] <= 0).any():
        errors.append("Close <= 0인 행이 남아 있음")

    universe_tickers = set(sp500_universe["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False))
    successful = set(final_df["Ticker"])
    failed = set(final_missing_df["Ticker"])
    if successful - universe_tickers:
        errors.append(f"유니버스 밖 성공 티커: {sorted(successful - universe_tickers)[:10]}")
    if failed - universe_tickers:
        errors.append(f"유니버스 밖 실패 티커: {sorted(failed - universe_tickers)[:10]}")
    if successful & failed:
        errors.append(f"성공/실패 중복 티커: {sorted(successful & failed)[:10]}")
    unaccounted = universe_tickers - successful - failed
    if unaccounted:
        errors.append(f"성공/실패 어디에도 없는 티커 {len(unaccounted)}개: {sorted(unaccounted)[:10]}")

    coverage_tickers = set(coverage_df["Ticker"]) if "Ticker" in coverage_df else set()
    if coverage_tickers != successful:
        errors.append("coverage_df 티커 집합이 final_df 성공 티커 집합과 다름")
    if "n_rows" not in coverage_df:
        errors.append("coverage_df에 n_rows 컬럼 누락")
    else:
        actual = final_df.groupby("Ticker").size()
        reported = coverage_df.set_index("Ticker")["n_rows"]
        mismatch = actual.index[actual.ne(reported.reindex(actual.index))]
        if len(mismatch):
            errors.append(f"coverage_df 행 수 불일치 티커 {len(mismatch)}개")
    if "n_rows_final" in final_missing_df and (final_missing_df["n_rows_final"] != 0).any():
        errors.append("final_missing_df에 유효한 최종 행이 있는 종목이 포함됨")
    if "short_history" in coverage_df and int(coverage_df["short_history"].sum()):
        warnings.append(f"short_history 종목 {int(coverage_df['short_history'].sum())}개(제외하지 않음)")

    forbidden = {
        "return", "simple_return", "log_return", "volatility", "mdd", "rsi",
        "sharpe", "sortino", "beta", "calmar", "moving_average", "momentum",
    }
    generated = sorted(column for column in final_df.columns if column.lower() in forbidden)
    if generated:
        errors.append(f"전처리 범위 밖 파생변수 존재: {generated}")

    report = {
        "valid": not errors, "errors": errors, "warnings": warnings,
        "final_shape": final_df.shape, "n_final_tickers": len(successful),
        "n_missing_tickers": len(failed), "n_universe_tickers": len(universe_tickers),
    }
    if errors:
        raise AssertionError("검증 실패:\n- " + "\n- ".join(errors))
    return report
