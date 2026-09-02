"""Window-configurable clustering features for daily OHLCV data.

All rolling features use only information available on or before each row's
date. Beta always uses a separately collected S&P 500 index DataFrame.

feature.py 대비 변경점
----------------------
1. RSI 워밍업 버그 수정
   기존 `rsi.where(avg_loss > 0, 100.0).where(avg_gain > 0, 0.0)` 는
   `NaN > 0` 이 False 로 평가되는 탓에 워밍업 구간(avg_gain/avg_loss 가 NaN)을
   100.0 -> 0.0 순으로 덮어써서 최종적으로 **0.0** 을 남겼다.
   `.where` 대신 `.mask` 를 쓰고 마지막에 유효성 게이트를 걸어 NaN 을 보존한다.

2. 극단 수익률 방어선 (max_abs_daily_return / max_abs_period_return)
   기존 `.where(market_variance > 0)` 는 beta 의 **분모**만 막았다. 실제 폭발은
   분자(공분산)에서 나온다 — 미조정 분할 등으로 생긴 하루짜리 이상 수익률 하나가
   롤링 윈도우에 window 행 머물면서 volatility 와 beta 를 동시에 폭주시킨다.
   클리핑이 아니라 **NaN 마스킹**을 한다. 없는 값을 지어내지 않고,
   min_periods=window 덕분에 오염된 구간이 통째로 빠진다.

3. 진단 출력
   마스킹된 행, 시장 수익률 결측, beta 이상치를 조용히 넘기지 않고 보고한다.

4. find_price_anomalies() 추가
   원본 가격 데이터에서 이상 점프를 먼저 찾아내기 위한 헬퍼.
   지표를 만들기 전에 이걸로 데이터를 청소하는 게 정석이다.

5. add_idio_vol (기본 False, opt-in)
   beta = rho * sigma_i / sigma_m 이라 beta 와 volatility 는 구조적으로 상관된다.
   고유변동성(시장 회귀 잔차의 표준편차)으로 바꾸면 beta 와 거의 직교해져서
   "안정성 가중치"가 의도대로 작동한다. 단, 데이터 청소 결과와 섞이지 않도록
   기본값은 False 로 두고 별도 실험에서 켜는 것을 권장한다.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
REQUIRED_COLUMNS = {"Date", "Ticker", "Close"}
FEATURE_METRICS = ("beta", "volatility", "return", "rsi")

# 하루 +/-75% 를 넘는 움직임은 S&P500 구성종목에서는 사실상 데이터 오류(미조정 분할 등)다.
DEFAULT_MAX_ABS_DAILY_RETURN = 0.75
# 기간 수익률 +1000% 초과도 마찬가지. period_return 은 Close 에서 직접 계산되므로
# 일간 수익률 마스킹이 적용되지 않아 별도 가드가 필요하다.
DEFAULT_MAX_ABS_PERIOD_RETURN = 10.0
# 정상적인 종목 베타는 넉넉히 잡아도 |beta| < 5 다. 20 을 넘으면 계산이 깨진 것.
DEFAULT_MAX_ABS_BETA = 20.0


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


def find_price_anomalies(
    prices: pd.DataFrame,
    max_abs_daily_return: float = DEFAULT_MAX_ABS_DAILY_RETURN,
) -> pd.DataFrame:
    """이상 일간 수익률을 크기순으로 반환한다 (지표 생성 전 데이터 청소용).

    ``price_range_ratio`` 는 그 종목 전체 기간의 max(Close)/min(Close) 다.
    이 값이 1000 을 넘으면 분할 미조정을 강하게 의심할 수 있다.
    """
    frame = prices.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Ticker", "Close"])
    frame = frame.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    grouped = frame.groupby("Ticker", sort=False)["Close"]
    daily_returns = grouped.pct_change(fill_method=None)
    previous_close = grouped.shift(1)
    flagged = daily_returns.abs() > max_abs_daily_return

    columns = ["Date", "Ticker", "Close"]
    if "source" in frame.columns:
        columns.append("source")

    report = frame.loc[flagged, columns].copy()
    report["prev_close"] = previous_close[flagged]
    report["daily_return"] = daily_returns[flagged]

    price_range_ratio = frame.groupby("Ticker")["Close"].agg(lambda s: s.max() / s.min())
    report["price_range_ratio"] = report["Ticker"].map(price_range_ratio)

    return (
        report.sort_values("daily_return", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )


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


def _rolling_by_ticker(
    values: pd.Series,
    tickers: pd.Series,
    window: int,
    how: str,
) -> pd.Series:
    """종목별 롤링 통계 (min_periods=window 고정)."""
    return values.groupby(tickers, sort=False).transform(
        lambda group: getattr(group.rolling(window, min_periods=window), how)(ddof=1)
    )


def add_features(
    prices: pd.DataFrame,
    sp500_beta_df: pd.DataFrame,
    windows: Iterable[int],
    annualization: int = TRADING_DAYS_PER_YEAR,
    max_abs_daily_return: float | None = DEFAULT_MAX_ABS_DAILY_RETURN,
    max_abs_period_return: float | None = DEFAULT_MAX_ABS_PERIOD_RETURN,
    max_abs_beta: float | None = DEFAULT_MAX_ABS_BETA,
    add_idio_vol: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Add beta, volatility, period return, and RSI features.

    The caller must explicitly choose ``windows``; the feature code does not
    prescribe a 60- or 120-day horizon. ``sp500_beta_df`` must contain the
    separately collected S&P 500 ``Date`` and ``Close`` and is used only for
    beta. Period returns compare today's close with the close exactly N trading
    days earlier. Features are NaN until enough history exists.

    Parameters
    ----------
    max_abs_daily_return
        이 값을 넘는 |일간 수익률| 을 NaN 으로 마스킹한다. 클리핑이 아니라
        마스킹이므로 해당 행이 포함된 롤링 윈도우(window 행)가 통째로 NaN 이 된다.
        ``None`` 이면 기존 feature.py 와 동일하게 동작한다(방어선 없음).
    max_abs_period_return
        이 값을 넘는 |기간 수익률| 을 NaN 으로 마스킹한다.
    max_abs_beta
        이 값을 넘는 |beta| 를 NaN 으로 마스킹한다. 수익률 마스킹이 제대로 걸리면
        보통 0건이며, 0건이 아니면 아직 남은 데이터 문제가 있다는 신호다.
    add_idio_vol
        True 면 ``idio_vol_{window}d`` (시장 회귀 잔차의 연환산 표준편차) 를 추가한다.
        beta 와 거의 직교하는 위험 지표. 기본 False.
    verbose
        마스킹/결측 진단을 출력한다.
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
    for name, bound in (
        ("max_abs_daily_return", max_abs_daily_return),
        ("max_abs_period_return", max_abs_period_return),
        ("max_abs_beta", max_abs_beta),
    ):
        if bound is not None and bound <= 0:
            raise ValueError(f"{name}은 양수이거나 None이어야 합니다.")

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
    tickers = frame["Ticker"]

    # ------------------------------------------------------------------
    # 시장 수익률 결합 — 결측을 조용히 넘기지 않는다.
    # 결측 1행은 롤링 윈도우 window 행을 NaN 으로 만들기 때문에 영향이 크다.
    # ------------------------------------------------------------------
    market_returns = _prepare_benchmark_returns(sp500_beta_df)
    frame["_market_return"] = frame["Date"].map(market_returns)

    market_missing = frame["_market_return"].isna()
    # 벤치마크 시계열의 첫 거래일은 pct_change 특성상 항상 NaN 이므로 제외하고 센다.
    first_market_date = market_returns.index.min()
    unexpected_missing = market_missing & (frame["Date"] != first_market_date)
    if verbose and unexpected_missing.any():
        n_dates = frame.loc[unexpected_missing, "Date"].nunique()
        print(
            f"[경고] 시장 수익률 결측 {int(unexpected_missing.sum()):,}행 "
            f"({n_dates}개 날짜) — 해당 구간 beta가 NaN 처리됩니다."
        )
        print(
            "        결측 날짜 예시:",
            sorted(frame.loc[unexpected_missing, "Date"].unique())[:5],
        )

    # ------------------------------------------------------------------
    # 일간 수익률 + 극단값 마스킹
    # volatility / beta(공분산) / RSI 가 모두 이 시리즈에서 파생되므로
    # 여기서 한 번 막으면 세 지표가 동시에 보호된다.
    # ------------------------------------------------------------------
    daily_returns = frame.groupby("Ticker", sort=False)["Close"].pct_change(fill_method=None)

    if max_abs_daily_return is not None:
        extreme = daily_returns.abs() > max_abs_daily_return
        if extreme.any():
            if verbose:
                report = (
                    frame.loc[extreme, ["Date", "Ticker", "Close"]]
                    .assign(daily_return=daily_returns[extreme])
                    .sort_values("daily_return", key=lambda s: s.abs(), ascending=False)
                )
                print(
                    f"[경고] |일간수익률| > {max_abs_daily_return:.0%} 인 행 "
                    f"{int(extreme.sum()):,}건 / {report['Ticker'].nunique()}종목 → NaN 마스킹"
                )
                print(report.head(10).to_string(index=False))
                print(
                    "        find_price_anomalies()로 원본 가격을 확인하세요 "
                    "(price_range_ratio > 1000 이면 분할 미조정 의심)."
                )
            daily_returns = daily_returns.mask(extreme)

    for window in normalized_windows:
        # ------------------------------------------------------------------
        # 연환산 변동성 — 최근 N일 일별 수익률의 표준편차를 연간 기준으로 환산.
        # 값이 클수록 주가가 크게 흔들림.
        # ------------------------------------------------------------------
        return_std = _rolling_by_ticker(daily_returns, tickers, window, "std")
        frame[f"volatility_{window}d"] = return_std * np.sqrt(annualization)

        # ------------------------------------------------------------------
        # 베타 — 종목 수익률이 S&P500 지수 수익률에 얼마나 민감하게 움직이는지.
        # market_variance 를 종목별로 계산하는 이유: covariance 가 그 종목의 행만
        # 사용하므로, 분모도 정확히 같은 행 집합에서 나와야 정합적이다
        # (상장 중단/거래일 누락이 있는 종목에서 차이가 난다).
        # ------------------------------------------------------------------
        market_variance = _rolling_by_ticker(
            frame["_market_return"], tickers, window, "var"
        )
        covariance = daily_returns.groupby(tickers, sort=False).transform(
            lambda values: values.rolling(window, min_periods=window).cov(
                frame.loc[values.index, "_market_return"]
            )
        )
        beta = (covariance / market_variance).where(market_variance > 0)

        if max_abs_beta is not None:
            beta_extreme = beta.abs() > max_abs_beta
            if beta_extreme.any():
                if verbose:
                    worst = (
                        frame.loc[beta_extreme, ["Date", "Ticker"]]
                        .assign(beta=beta[beta_extreme])
                        .sort_values("beta", key=lambda s: s.abs(), ascending=False)
                    )
                    print(
                        f"[경고] |beta_{window}d| > {max_abs_beta} 인 행 "
                        f"{int(beta_extreme.sum()):,}건 / {worst['Ticker'].nunique()}종목 "
                        "→ NaN 마스킹 (수익률 마스킹 후에도 남았다면 데이터 문제가 더 있습니다)"
                    )
                    print(worst.head(5).to_string(index=False))
                beta = beta.mask(beta_extreme)

        frame[f"beta_{window}d"] = beta

        # ------------------------------------------------------------------
        # 고유변동성(opt-in) — 시장 회귀 r_i = a + b*r_m + e 의 잔차 표준편차.
        # sigma_idio = sigma_i * sqrt(1 - rho^2),  rho = beta * sigma_m / sigma_i
        # beta(체계적 위험)와 거의 직교하므로, 둘을 함께 쓰면 위험이 깔끔하게 분해된다.
        # ------------------------------------------------------------------
        if add_idio_vol:
            market_std_annual = np.sqrt(market_variance) * np.sqrt(annualization)
            total_vol = frame[f"volatility_{window}d"]
            with np.errstate(divide="ignore", invalid="ignore"):
                rho = (beta * market_std_annual / total_vol).clip(-1.0, 1.0)
            frame[f"idio_vol_{window}d"] = total_vol * np.sqrt(1.0 - rho**2)

        # ------------------------------------------------------------------
        # RSI — 최근 N일 평균 상승 폭과 평균 하락 폭의 비교. 범위 0~100.
        #   상승만 있고 하락이 없으면 100 / 하락만 있으면 0 / 변동이 없으면 50.
        # [수정] `.where` 대신 `.mask` 사용 + 마지막에 유효성 게이트.
        #   `.where(cond, other)` 는 cond 가 False 인 곳을 바꾸는데 `NaN > 0` 이
        #   False 라, 기존 코드는 워밍업 구간을 100.0 -> 0.0 으로 덮어써서
        #   최종적으로 0.0 을 남겼다. `.mask` 는 조건이 True 인 곳만 바꾸므로
        #   NaN 이 자연스럽게 통과하고, 마지막 .where 로 확실히 NaN 을 보장한다.
        # ------------------------------------------------------------------
        gains = daily_returns.clip(lower=0.0)
        losses = -daily_returns.clip(upper=0.0)
        avg_gain = _rolling_mean_by_ticker(gains, tickers, window)
        avg_loss = _rolling_mean_by_ticker(losses, tickers, window)

        with np.errstate(divide="ignore", invalid="ignore"):
            relative_strength = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = (
            rsi.mask(avg_loss == 0, 100.0)
            .mask(avg_gain == 0, 0.0)
            .mask((avg_gain == 0) & (avg_loss == 0), 50.0)
            .where(avg_gain.notna() & avg_loss.notna())  # 워밍업 구간은 NaN 유지
        )
        frame[f"rsi_{window}d"] = rsi

        # ------------------------------------------------------------------
        # 기간 수익률 — 현재 가격과 정확히 N거래일 전 가격을 비교.
        # daily_returns 가 아니라 Close 에서 직접 계산되므로 위의 일간 마스킹이
        # 적용되지 않는다. 별도 가드가 필요한 이유.
        # ------------------------------------------------------------------
        period_return = frame.groupby("Ticker", sort=False)["Close"].transform(
            lambda close: close / close.shift(window) - 1.0
        )
        if max_abs_period_return is not None:
            period_extreme = period_return.abs() > max_abs_period_return
            if period_extreme.any():
                if verbose:
                    print(
                        f"[경고] |return_{window}d| > {max_abs_period_return:.0f} 인 행 "
                        f"{int(period_extreme.sum()):,}건 / "
                        f"{frame.loc[period_extreme, 'Ticker'].nunique()}종목 → NaN 마스킹"
                    )
                period_return = period_return.mask(period_extreme)
        frame[f"return_{window}d"] = period_return

    if verbose:
        produced = feature_columns(normalized_windows)
        if add_idio_vol:
            produced += [f"idio_vol_{w}d" for w in normalized_windows]
        coverage = frame[produced].notna().mean().sort_index()
        print("[요약] 피처별 유효값 비율")
        print(coverage.round(4).to_string())

    return frame.drop(columns="_market_return")


def _rolling_mean_by_ticker(
    values: pd.Series,
    tickers: pd.Series,
    window: int,
) -> pd.Series:
    """종목별 롤링 평균 (rolling.mean 은 ddof 인자를 받지 않아 별도 헬퍼)."""
    return values.groupby(tickers, sort=False).transform(
        lambda group: group.rolling(window, min_periods=window).mean()
    )


def build_feature_dataset(
    windows: Iterable[int],
    input_path: str = "data/processed/final_df.parquet",
    sp500_beta_path: str = "data/raw/sp500_beta_df.parquet",
    output_path: str | None = None,
    **feature_kwargs,
) -> pd.DataFrame:
    """Load processed prices, create requested features, and optionally save."""
    featured = add_features(
        pd.read_parquet(input_path),
        sp500_beta_df=pd.read_parquet(sp500_beta_path),
        windows=windows,
        **feature_kwargs,
    )
    if output_path is not None:
        featured.to_parquet(output_path, index=False)
    return featured


# ======================================================================
# 자체 검증 — `python src/new_feature.py` 로 실행
#
# 실제 데이터 없이 합성 데이터로 아래를 확인한다:
#   (1) 오염 행 탐지        - find_price_anomalies 가 범인을 집어내는가
#   (2) 방어선 OFF          - 오염 1건이 지표를 얼마나 망가뜨리는가 (기존 동작)
#   (3) 방어선 ON           - 마스킹 후 정상 범위로 복귀하는가
#   (4) RSI 워밍업          - NaN 이 0.0 으로 덮이지 않는가
#   (5) 고유변동성          - beta 와 직교해지는가
#   (6) 수식 정확도         - 오염 없는 데이터에서 참 beta 를 복원하는가
# ======================================================================


def _make_synthetic_data(
    n_days: int = 500,
    n_tickers: int = 20,
    contaminate: bool = True,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """검증용 합성 가격 데이터. (prices, sp500, 참값 beta) 를 돌려준다."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    market_returns = rng.normal(0.0004, 0.011, n_days)
    market_returns[0] = 0.0
    sp500 = pd.DataFrame(
        {"Date": dates, "Close": 3000 * np.exp(np.cumsum(market_returns))}
    )

    true_beta = pd.Series(
        {f"T{i:02d}": 0.4 + 0.12 * i for i in range(n_tickers)}, name="true_beta"
    )

    blocks = []
    for i in range(n_tickers):
        # 고유위험은 beta 와 무관하게 생성 -> corr(beta, idio_vol) ~ 0 이어야 정상
        idiosyncratic = rng.normal(0, 0.012, n_days)
        returns = true_beta.iloc[i] * market_returns + idiosyncratic
        returns[0] = 0.0
        blocks.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Ticker": f"T{i:02d}",
                    "Close": 50 * np.exp(np.cumsum(returns)),
                }
            )
        )
    prices = pd.concat(blocks, ignore_index=True)

    if contaminate:
        # 미조정 분할 시뮬레이션: T07 의 특정 시점 이후 가격에 96배 점프 1건
        victim, split_date = "T07", dates[250]
        target = (prices["Ticker"] == victim) & (prices["Date"] >= split_date)
        prices.loc[target, "Close"] *= 96.0

    return prices, sp500, true_beta


def _self_test(window: int = 60) -> None:
    """합성 데이터로 버그 재현과 수정을 검증한다."""

    def header(text: str) -> None:
        print()
        print("=" * 70)
        print(text)
        print("=" * 70)

    prices, sp500, true_beta = _make_synthetic_data()
    vol_col, beta_col, rsi_col = (
        f"volatility_{window}d",
        f"beta_{window}d",
        f"rsi_{window}d",
    )

    header("1) find_price_anomalies — 범인 특정")
    anomalies = find_price_anomalies(prices)
    print(anomalies.to_string(index=False))
    assert len(anomalies) == 1 and anomalies.loc[0, "Ticker"] == "T07"

    header("2) 방어선 OFF (= 기존 feature.py 동작)")
    unguarded = add_features(
        prices,
        sp500,
        windows=(window,),
        max_abs_daily_return=None,
        max_abs_period_return=None,
        max_abs_beta=None,
        verbose=False,
    )
    corr_off = unguarded[[vol_col, beta_col]].dropna().corr().iloc[0, 1]
    print(f"corr(volatility, beta) = {corr_off:.3f}   <- 오염된 값")
    print(f"beta 범위       : {unguarded[beta_col].min():.2f} ~ {unguarded[beta_col].max():.1f}")
    print(f"volatility 범위 : {unguarded[vol_col].min():.2f} ~ {unguarded[vol_col].max():.1f}")
    assert unguarded[beta_col].abs().max() > 20, "오염이 재현되지 않음"

    header("3) 방어선 ON (new_feature.py 기본값)")
    guarded = add_features(prices, sp500, windows=(window,), verbose=True)
    corr_on = guarded[[vol_col, beta_col]].dropna().corr().iloc[0, 1]
    print()
    print(f"corr(volatility, beta) = {corr_on:.3f}   <- 오염 제거 후")
    print(f"beta 범위       : {guarded[beta_col].min():.2f} ~ {guarded[beta_col].max():.2f}")
    print(f"volatility 범위 : {guarded[vol_col].min():.2f} ~ {guarded[vol_col].max():.2f}")
    assert guarded[beta_col].abs().max() < 10, "방어선이 작동하지 않음"
    assert guarded[vol_col].max() < 5, "방어선이 작동하지 않음"

    header("4) RSI 워밍업 버그 확인")
    print(guarded[guarded["Ticker"] == "T00"].head(3)[["Date", vol_col, rsi_col]].to_string(index=False))
    warmup = guarded[guarded[vol_col].isna()]
    n_zero = int((warmup[rsi_col] == 0.0).sum())
    n_nan = int(warmup[rsi_col].isna().sum())
    print(f"\n워밍업 행 {len(warmup):,}개 중 -> rsi=0.0: {n_zero}개, rsi=NaN: {n_nan:,}개")
    assert n_zero == 0 and n_nan == len(warmup), "RSI 워밍업이 여전히 0으로 채워짐"
    print("OK: 워밍업 RSI가 전부 NaN")

    legacy = pd.Series([np.nan] * 3)
    legacy = legacy.where(legacy > 0, 100.0).where(legacy > 0, 0.0)
    print(f"참고 - 기존 .where 체인에 NaN 입력 시: {legacy.tolist()}  <- 이게 버그였음")

    header("5) 고유변동성 (add_idio_vol=True)")
    with_idio = add_features(
        prices, sp500, windows=(window,), add_idio_vol=True, verbose=False
    )
    idio_col = f"idio_vol_{window}d"
    corr_matrix = with_idio[[beta_col, vol_col, idio_col]].dropna().corr()
    print(corr_matrix.round(3).to_string())
    print(f"\ncorr(beta, volatility) = {corr_matrix.loc[beta_col, vol_col]:.3f}")
    print(f"corr(beta, idio_vol)   = {corr_matrix.loc[beta_col, idio_col]:.3f}  <- 직교에 가까움")
    assert abs(corr_matrix.loc[beta_col, idio_col]) < 0.3

    header("6) 수식 정확도 — 오염 없는 데이터에서 참 beta 복원")
    clean_prices, clean_sp500, true_beta = _make_synthetic_data(contaminate=False)
    clean = add_features(clean_prices, clean_sp500, windows=(window,), verbose=False)
    estimated = clean.groupby("Ticker")[beta_col].mean().rename("est_beta")
    print(pd.concat([true_beta, estimated.round(3)], axis=1).head(6).to_string())
    mae = (estimated - true_beta).abs().mean()
    print(f"\n평균 절대오차: {mae:.4f}  (수식 정상)")
    assert mae < 0.1, "beta 추정이 참값에서 크게 벗어남"

    print("\n모든 검증 통과")


if __name__ == "__main__":
    _self_test()
