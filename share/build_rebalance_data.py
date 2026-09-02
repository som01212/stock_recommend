"""Build the combined feature dataset and per-interval rebalancing snapshots.

Input:
    data/final_df.parquet              cleaned daily OHLCV (all constituents,
                                        including historically delisted ones)
    data/sp500_total_return_df.parquet S&P 500 *total return* index daily
                                        close (^SP500TR, dividends reinvested)
                                        -- used for beta (2026-09-02). Every
                                        constituent's Close is already
                                        dividend-adjusted, so beta's market
                                        side is now computed on the same
                                        return convention. data/sp500_beta_df
                                        .parquet (^GSPC, price-only) is kept
                                        for reference but no longer read here
                                        -- see README.md for the sensitivity
                                        check that motivated this (cluster
                                        label agreement was only 84%/90% at
                                        60/120-day windows, not negligible).
    data/sp500_membership_history.csv  ticker start/end membership intervals

Output (written to data/):
    final_df.parquet (overwritten in place with feature columns added)
        OHLCV + 4 metrics x 4 windows (30/60/120/252) = 16 feature columns
    cluster_df.parquet
        Date, Ticker + the 16 feature columns only (daily, every row)
    rebalance_30df.parquet, rebalance_60df.parquet,
    rebalance_120df.parquet, rebalance_252df.parquet
        cluster_df sampled every 30 / 60 / 120 / 252 trading days, then
        filtered to rows where the ticker was an actual S&P 500 member on
        that date. All four files carry the SAME 16 feature columns — only
        the sampled date set (rebalancing cadence) differs between them.

Note: a feature's lookback window (how far back a value is computed from)
and the rebalancing interval (how far apart the sampled snapshot dates are)
are independent choices. All 16 feature columns are valid at every snapshot
regardless of which rebalancing file they come from.

Run:
    python build_rebalance_data.py                    # default windows: 30,60,120,252
    python build_rebalance_data.py --windows 45,90     # custom windows (source unchanged)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.feature import add_features
from src.membership import filter_by_membership

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# The 4 features confirmed after multicollinearity analysis:
# beta/volatility (stability, priority 1) + return/rsi (profitability, priority 2).
# mdd, downside_volatility, ma_gap, momentum, cagr are intentionally not
# computed — see src/feature.py.
CLUSTER_METRICS = ["beta", "volatility", "return", "rsi"]
DEFAULT_WINDOWS = (30, 60, 120, 252)


def parse_windows(raw: str) -> tuple[int, ...]:
    windows = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if not windows:
        raise ValueError("--windows에는 최소 1개 이상의 정수가 필요합니다 (예: --windows 30,90).")
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows", type=parse_windows, default=DEFAULT_WINDOWS,
        help="리밸런싱/피처 윈도우(거래일 기준), 쉼표로 구분. 기본값: 30,60,120,252",
    )
    windows = parser.parse_args().windows
    print(f"윈도우: {windows}")

    final_df = pd.read_parquet(DATA_DIR / "final_df.parquet")
    # beta는 총수익지수(^SP500TR) 기준 -- data/sp500_beta_df.parquet(^GSPC)는
    # 더 이상 안 읽음, 이유는 이 파일 상단 docstring 참고.
    sp500_beta_df = pd.read_parquet(DATA_DIR / "sp500_total_return_df.parquet")
    print(f"final_df (입력): {final_df.shape}, {final_df['Date'].min()} ~ {final_df['Date'].max()}")

    # 예전 실행(다른 --windows)에서 남은 피처 컬럼이 있으면 지우고 새로 만든다.
    # 그래야 --windows를 바꿔가며 여러 번 돌려도 final_df.parquet에 옛날
    # 윈도우 컬럼이 계속 쌓이지 않는다.
    stale_cols = [c for c in final_df.columns if c.split("_")[0] in CLUSTER_METRICS and c.endswith("d")]
    final_df = final_df.drop(columns=stale_cols)

    # 한 번의 호출로 요청한 모든 윈도우의 피처 컬럼을 전부 추가한다
    # (add_features는 windows를 순회하며 각 윈도우의 4개 지표를 컬럼으로 붙인다).
    featured = add_features(final_df, sp500_beta_df=sp500_beta_df, windows=windows)
    featured.to_parquet(DATA_DIR / "final_df.parquet", index=False)
    print(f"final_df (출력, 피처 포함): {featured.shape}")

    feature_cols = [f"{metric}_{window}d" for window in windows for metric in CLUSTER_METRICS]
    cluster_df = featured[["Date", "Ticker", *feature_cols]].copy()
    cluster_df.to_parquet(DATA_DIR / "cluster_df.parquet", index=False)
    print(f"cluster_df: {cluster_df.shape}, 컬럼={cluster_df.columns.tolist()}")

    all_dates = sorted(cluster_df["Date"].unique())
    for window in windows:
        rebalance_dates = all_dates[::window]
        snapshots = cluster_df[cluster_df["Date"].isin(rebalance_dates)].copy()
        rebalance_df = filter_by_membership(snapshots)
        rebalance_df.to_parquet(DATA_DIR / f"rebalance_{window}df.parquet", index=False)
        print(
            f"[{window}일 간격] rebalance_{window}df: {rebalance_df.shape} "
            f"({rebalance_df['Ticker'].nunique()}종목, {rebalance_df['Date'].nunique()}개 시점)"
        )


if __name__ == "__main__":
    main()
