"""Window-configurable clustering features for daily OHLCV data.

All rolling features use only information available on or before each row's
date. Beta always uses a separately collected S&P 500 index DataFrame.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
REQUIRED_COLUMNS = {"Date", "Ticker", "Close"}
FEATURE_METRICS = ("beta", "volatility", "return", "rsi")


def feature_columns(
    windows: Iterable[int],
) -> list[str]:
    """Return the four clustering feature names for each requested window."""
    normalized_windows = tuple(int(window) for window in windows)
    return [
        f"{metric}_{window}d"
        for window in normalized_windows
        for metric in FEATURE_METRICS
    ]


def _prepare_benchmark_returns(
    sp500_beta_df: pd.DataFrame,
) -> pd.Series:
    """Return date-indexed S&P 500 daily returns for beta calculation."""
    if not isinstance(sp500_beta_df, pd.DataFrame):
        raise TypeError("sp500_beta_df는 pandas DataFrame이어야 합니다.")
    if "Date" not in sp500_beta_df or "Close" not in sp500_beta_df:
        raise ValueError("sp500_beta_df에는 Date와 Close 컬럼이 필요합니다.")

    market = sp500_beta_df.copy()
    market["Date"] = pd.to_datetime(market["Date"], errors="coerce")
    market["Close"] = pd.to_numeric(market["Close"], errors="coerce")
    market = market.dropna(subset=["Date", "Close"])
    market = market.loc[market["Close"] > 0]
    market = market.sort_values("Date").drop_duplicates("Date", keep="last")
    series = market.set_index("Date")["Close"].pct_change(fill_method=None)

    if series.index.hasnans or series.isna().all():
        raise ValueError("benchmark 날짜 또는 값이 유효하지 않습니다.")
    return series.sort_index()


def add_features(
    prices: pd.DataFrame,
    sp500_beta_df: pd.DataFrame,
    windows: Iterable[int],
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Add beta, volatility, period return, and RSI features.

    The caller must explicitly choose ``windows``; the feature code does not
    prescribe a 60- or 120-day horizon. ``sp500_beta_df`` must contain the
    separately collected S&P 500 ``Date`` and ``Close`` and is used only for
    beta. Period returns compare today's close with the close exactly N trading
    days earlier. Features are NaN until enough history exists.
    """
    missing = sorted(REQUIRED_COLUMNS - set(prices.columns))
    if missing:
        raise ValueError(f"피처 생성 필수 컬럼 누락: {missing}")

    normalized_windows = tuple(int(window) for window in windows)
    if not normalized_windows or any(window < 2 for window in normalized_windows):
        raise ValueError("windows에는 2 이상의 기간이 하나 이상 필요합니다.")
    if len(set(normalized_windows)) != len(normalized_windows):
        raise ValueError("windows에 중복 기간이 있습니다.")
    if annualization <= 0:
        raise ValueError("annualization은 양수여야 합니다.")

    frame = prices.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    if frame[["Date", "Ticker", "Close"]].isna().any().any():
        raise ValueError("Date, Ticker, Close에는 결측 또는 변환 불가 값이 없어야 합니다.")
    if (frame["Close"] <= 0).any():
        raise ValueError("Close는 모두 양수여야 합니다.")
    if frame.duplicated(["Ticker", "Date"]).any():
        raise ValueError("(Ticker, Date) 중복 행이 있습니다.")

    frame = frame.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    market_returns = _prepare_benchmark_returns(sp500_beta_df)
    frame["_market_return"] = frame["Date"].map(market_returns)
    ticker_group = frame.groupby("Ticker", sort=False, group_keys=False)
    daily_returns = ticker_group["Close"].pct_change(fill_method=None)

    for window in normalized_windows:
        return_std = daily_returns.groupby(frame["Ticker"], sort=False).transform(
            lambda values: values.rolling(window, min_periods=window).std(ddof=1)
        )
        # 연환산변동성, 최근 N일 일별 수익률의 표준편차를 연간 기준으로 환산
        # 값이 클수록 주가가 크게 흔들림
        frame[f"volatility_{window}d"] = return_std * np.sqrt(annualization)

        # 베타, 종목 수익률이 snp500 지수 수익률에 얼마나 민감하게 움직이는지 계산
        # 종목 수익률: final_df의 종목별 Close
        # 시장 수익률: sp500_beta_df의 Close
        market_variance = frame.groupby("Ticker", sort=False)["_market_return"].transform(
            lambda values: values.rolling(window, min_periods=window).var(ddof=1)
        )
        covariance = daily_returns.groupby(frame["Ticker"], sort=False).transform(
            lambda values: values.rolling(window, min_periods=window).cov(
                frame.loc[values.index, "_market_return"]
            )
        )
        frame[f"beta_{window}d"] = (covariance / market_variance).where(market_variance > 0)

        # RSI, 최근 N일 동안 평균 상승 폭과 평균 하락 폭을 비교
        # 범위: 0~100
        # 상승만 있고 하락이 없으면 100
        # 하락만 있고 상승이 없으면 0
        # 변동이 전혀 없으면 50
        # 일반적으로 높을수록 상승 압력이 강함
        gains = daily_returns.clip(lower=0.0)
        losses = -daily_returns.clip(upper=0.0)
        avg_gain = gains.groupby(frame["Ticker"], sort=False).transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
        avg_loss = losses.groupby(frame["Ticker"], sort=False).transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
        relative_strength = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.where(avg_loss > 0, 100.0).where(avg_gain > 0, 0.0)
        frame[f"rsi_{window}d"] = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)

        # 기간 수익률, 현재 가격과 정확히 N거래일 전 가격을 비교
        period_return = ticker_group["Close"].transform(
            lambda close: close / close.shift(window) - 1.0
        )
        frame[f"return_{window}d"] = period_return

    return frame.drop(columns="_market_return")


def build_feature_dataset(
    windows: Iterable[int],
    input_path: str = "data/processed/final_df.parquet",
    sp500_beta_path: str = "data/raw/sp500_beta_df.parquet",
    output_path: str | None = None,
) -> pd.DataFrame:
    """Load processed prices, create requested features, and optionally save."""
    featured = add_features(
        pd.read_parquet(input_path),
        sp500_beta_df=pd.read_parquet(sp500_beta_path),
        windows=windows,
    )
    if output_path is not None:
        featured.to_parquet(output_path, index=False)
    return featured
