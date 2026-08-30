"""Point-in-time fundamental ratios (PER, PBR, ROE, debt ratio, revenue
growth, log market cap), built from yfinance's annual financials and
balance sheet.

yfinance's free tier only exposes about 5 years of annual financial
statements per ticker, which is exactly why this module is scoped to a
recent-years window (see notebooks/09_per_experiment.ipynb and
notebooks/11_fundamentals_expansion.ipynb) rather than the main 10-year
pipeline.

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

import numpy as np
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


# ======================================================================
# 확장 재무지표 — PBR / ROE / 부채비율 / 매출성장률 / 시가총액
#
# PER과 같은 point-in-time 원칙(회계연도 종료일 + REPORTING_LAG_DAYS)을
# 그대로 적용한다. income statement(순이익, 매출)와 balance sheet(자기자본,
# 부채, 발행주식수)는 서로 다른 yfinance 엔드포인트라 별도로 캐싱하되,
# 종목별 merge_asof 로직 자체는 ``_attach_point_in_time``으로 통일해서
# add_point_in_time_per처럼 매번 반복하지 않는다.
# ======================================================================

CACHE_PATH_INCOME = Path(__file__).resolve().parents[1] / "data" / "raw" / "cache" / "income_statement_history.csv"
CACHE_PATH_BALANCE = Path(__file__).resolve().parents[1] / "data" / "raw" / "cache" / "balance_sheet_history.csv"


def fetch_income_statement_history(
    tickers: list[str], force_refresh: bool = False, sleep: float = 0.3
) -> pd.DataFrame:
    """Fetch each ticker's annual EPS, net income, and revenue in one pass
    (all three live in yfinance's ``.financials``, so this is one API call
    per ticker regardless of how many of the three we need).

    Also computes ``revenue_growth_yoy`` per ticker (NaN in each ticker's
    earliest fiscal year on file, since growth needs a prior year).
    """
    if CACHE_PATH_INCOME.is_file() and not force_refresh:
        return pd.read_csv(CACHE_PATH_INCOME, parse_dates=["fiscal_date"])

    rows = []
    failed = []
    for ticker in tickers:
        try:
            financials = yf.Ticker(ticker).financials
            if financials.empty:
                failed.append(ticker)
                continue

            eps_row = next((financials.loc[c] for c in ("Diluted EPS", "Basic EPS") if c in financials.index), None)
            net_income_row = financials.loc["Net Income"] if "Net Income" in financials.index else None
            revenue_row = financials.loc["Total Revenue"] if "Total Revenue" in financials.index else None
            if eps_row is None and net_income_row is None and revenue_row is None:
                failed.append(ticker)
                continue

            for fiscal_date in financials.columns:
                rows.append({
                    "Ticker": ticker,
                    "fiscal_date": fiscal_date,
                    "eps": eps_row[fiscal_date] if eps_row is not None else float("nan"),
                    "net_income": net_income_row[fiscal_date] if net_income_row is not None else float("nan"),
                    "revenue": revenue_row[fiscal_date] if revenue_row is not None else float("nan"),
                })
        except Exception:
            failed.append(ticker)
        time.sleep(sleep)

    print(f"[fundamentals] 손익계산서 확보: {len(tickers) - len(failed)}/{len(tickers)}종목, 실패 {len(failed)}종목")
    income_df = pd.DataFrame(rows).sort_values(["Ticker", "fiscal_date"])
    income_df["revenue_growth_yoy"] = income_df.groupby("Ticker")["revenue"].pct_change(fill_method=None)

    CACHE_PATH_INCOME.parent.mkdir(parents=True, exist_ok=True)
    income_df.to_csv(CACHE_PATH_INCOME, index=False)
    return income_df


def fetch_balance_sheet_history(
    tickers: list[str], force_refresh: bool = False, sleep: float = 0.3
) -> pd.DataFrame:
    """Fetch each ticker's annual stockholders' equity, total debt, and
    shares outstanding from yfinance's ``.balance_sheet``."""
    if CACHE_PATH_BALANCE.is_file() and not force_refresh:
        return pd.read_csv(CACHE_PATH_BALANCE, parse_dates=["fiscal_date"])

    rows = []
    failed = []
    for ticker in tickers:
        try:
            balance_sheet = yf.Ticker(ticker).balance_sheet
            if balance_sheet.empty:
                failed.append(ticker)
                continue

            equity_row = next(
                (balance_sheet.loc[c] for c in ("Stockholders Equity", "Common Stock Equity")
                 if c in balance_sheet.index), None
            )
            debt_row = next(
                (balance_sheet.loc[c] for c in ("Total Debt", "Net Debt") if c in balance_sheet.index), None
            )
            shares_row = next(
                (balance_sheet.loc[c] for c in ("Ordinary Shares Number", "Share Issued")
                 if c in balance_sheet.index), None
            )
            if equity_row is None and debt_row is None and shares_row is None:
                failed.append(ticker)
                continue

            for fiscal_date in balance_sheet.columns:
                rows.append({
                    "Ticker": ticker,
                    "fiscal_date": fiscal_date,
                    "equity": equity_row[fiscal_date] if equity_row is not None else float("nan"),
                    "total_debt": debt_row[fiscal_date] if debt_row is not None else float("nan"),
                    "shares_outstanding": shares_row[fiscal_date] if shares_row is not None else float("nan"),
                })
        except Exception:
            failed.append(ticker)
        time.sleep(sleep)

    print(f"[fundamentals] 재무상태표 확보: {len(tickers) - len(failed)}/{len(tickers)}종목, 실패 {len(failed)}종목")
    balance_df = pd.DataFrame(rows)
    CACHE_PATH_BALANCE.parent.mkdir(parents=True, exist_ok=True)
    balance_df.to_csv(CACHE_PATH_BALANCE, index=False)
    return balance_df


def _attach_point_in_time(df: pd.DataFrame, annual_df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Attach each row's trailing annual value(s) as of its Date, using only
    values that were actually public by then (fiscal_date + REPORTING_LAG_DAYS).
    Shared by every fundamental ratio below so the merge_asof-per-ticker
    (and its index-preservation workaround) is written once."""
    annual_df = annual_df.copy()
    annual_df["available_date"] = annual_df["fiscal_date"] + pd.Timedelta(days=REPORTING_LAG_DAYS)

    matched = []
    for ticker, group in df.groupby("Ticker", sort=False):
        ticker_annual = (
            annual_df.loc[annual_df["Ticker"] == ticker, ["available_date", *value_cols]]
            .sort_values("available_date")
        )
        # merge_asof drops the index, so carry it through as a column and restore it after
        group_sorted = group.sort_values("Date").reset_index(names="_orig_index")
        if ticker_annual.empty:
            merged = group_sorted.assign(**{c: float("nan") for c in value_cols})
        else:
            merged = pd.merge_asof(
                group_sorted, ticker_annual, left_on="Date", right_on="available_date", direction="backward",
            )
        matched.append(merged.set_index("_orig_index"))

    result = pd.concat(matched).loc[df.index]
    result.index.name = df.index.name
    return result.drop(columns=["available_date"], errors="ignore")


def add_fundamental_ratios(
    df: pd.DataFrame,
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Attach PER, PBR, ROE, debt_ratio, revenue_growth, and log_market_cap
    as of each row's Date, all point-in-time per REPORTING_LAG_DAYS.

    Each ratio is masked to NaN where its denominator is non-positive
    (mirrors add_point_in_time_per's "적자 기업 PER은 의미 없음" rule) —
    a negative-equity company's PBR/ROE/debt_ratio isn't a meaningful
    ratio, it's just a sign flip. market_cap is stored log-transformed
    (``log_market_cap``) rather than raw: firm size spans several orders
    of magnitude, which would dominate the linear/logistic model's
    distance-based scaling and skew the tree models' splits toward the
    mega-caps if left untransformed.
    """
    result = _attach_point_in_time(df, income_df, ["eps", "net_income", "revenue", "revenue_growth_yoy"])
    result = _attach_point_in_time(result, balance_df, ["equity", "total_debt", "shares_outstanding"])
    result = result.rename(columns={"revenue_growth_yoy": "revenue_growth"})

    result["per"] = result[price_col] / result["eps"]
    result.loc[result["eps"] <= 0, "per"] = float("nan")

    book_value_per_share = result["equity"] / result["shares_outstanding"]
    result["pbr"] = result[price_col] / book_value_per_share
    result.loc[(result["equity"] <= 0) | (result["shares_outstanding"] <= 0), "pbr"] = float("nan")

    result["roe"] = result["net_income"] / result["equity"]
    result.loc[result["equity"] <= 0, "roe"] = float("nan")

    result["debt_ratio"] = result["total_debt"] / result["equity"]
    result.loc[result["equity"] <= 0, "debt_ratio"] = float("nan")

    market_cap = result[price_col] * result["shares_outstanding"]
    result["log_market_cap"] = np.log(market_cap.mask(market_cap <= 0))

    return result


# ======================================================================
# 자체 검증 — `python src/fundamentals.py` 로 실행
# ======================================================================


def _self_test() -> None:
    income_df = pd.DataFrame([
        # T00: 2년치, 흑자 -> 두 번째 해에 revenue_growth_yoy가 채워져야 함
        {"Ticker": "T00", "fiscal_date": pd.Timestamp("2021-12-31"), "eps": 2.0, "net_income": 100.0, "revenue": 1000.0},
        {"Ticker": "T00", "fiscal_date": pd.Timestamp("2022-12-31"), "eps": 3.0, "net_income": 150.0, "revenue": 1200.0},
        # T01: 적자 -> per/roe가 마스킹돼야 함
        {"Ticker": "T01", "fiscal_date": pd.Timestamp("2021-12-31"), "eps": -1.0, "net_income": -50.0, "revenue": 500.0},
    ])
    income_df["revenue_growth_yoy"] = income_df.groupby("Ticker")["revenue"].pct_change(fill_method=None)

    balance_df = pd.DataFrame([
        {"Ticker": "T00", "fiscal_date": pd.Timestamp("2021-12-31"), "equity": 500.0, "total_debt": 200.0, "shares_outstanding": 100.0},
        {"Ticker": "T00", "fiscal_date": pd.Timestamp("2022-12-31"), "equity": 600.0, "total_debt": 250.0, "shares_outstanding": 100.0},
        {"Ticker": "T01", "fiscal_date": pd.Timestamp("2021-12-31"), "equity": -20.0, "total_debt": 300.0, "shares_outstanding": 50.0},
    ])

    prices = pd.DataFrame([
        # 2021-12-31 + 90일 = 2022-03-31 이전이라 아직 2021 실적이 안 잡혀야 함(NaN)
        {"Date": pd.Timestamp("2022-03-01"), "Ticker": "T00", "Close": 50.0},
        # 2022-03-31 이후라 2021 실적이 잡혀야 함
        {"Date": pd.Timestamp("2022-04-01"), "Ticker": "T00", "Close": 50.0},
        # 2022-12-31 + 90일 = 2023-03-31 이후라 2022 실적이 잡혀야 함
        {"Date": pd.Timestamp("2023-04-01"), "Ticker": "T00", "Close": 60.0},
        {"Date": pd.Timestamp("2022-04-01"), "Ticker": "T01", "Close": 10.0},
    ])

    result = add_fundamental_ratios(prices, income_df, balance_df)
    print(result[["Date", "Ticker", "eps", "revenue_growth", "per", "pbr", "roe", "debt_ratio", "log_market_cap"]])

    row = result.set_index(["Date", "Ticker"])

    print("\n1) point-in-time lag: 보고 지연 전에는 NaN")
    assert pd.isna(row.loc[(pd.Timestamp("2022-03-01"), "T00"), "eps"])
    print("OK")

    print("\n2) 보고 지연 이후엔 값이 붙음 (2022-04-01 -> 2021 실적)")
    r = row.loc[(pd.Timestamp("2022-04-01"), "T00")]
    assert r["eps"] == 2.0
    assert r["per"] == 50.0 / 2.0
    assert r["pbr"] == 50.0 / (500.0 / 100.0)
    assert r["roe"] == 100.0 / 500.0
    assert r["debt_ratio"] == 200.0 / 500.0
    assert pd.isna(r["revenue_growth"])  # T00의 첫 연도라 growth 없음
    print("OK")

    print("\n3) 두 번째 회계연도가 잡히면 revenue_growth_yoy가 채워짐 (2023-04-01 -> 2022 실적)")
    r = row.loc[(pd.Timestamp("2023-04-01"), "T00")]
    assert abs(r["revenue_growth"] - (1200.0 / 1000.0 - 1)) < 1e-9
    print("OK")

    print("\n4) 자기자본이 음수면 per/pbr/roe/debt_ratio 전부 마스킹")
    r = row.loc[(pd.Timestamp("2022-04-01"), "T01")]
    assert pd.isna(r["per"]), "적자(eps<0)라 per는 NaN이어야 함"
    assert pd.isna(r["pbr"]) and pd.isna(r["roe"]) and pd.isna(r["debt_ratio"])
    print("OK")

    print("\n5) log_market_cap = log(Close * shares_outstanding)")
    r = row.loc[(pd.Timestamp("2022-04-01"), "T00")]
    expected = np.log(50.0 * 100.0)
    assert abs(r["log_market_cap"] - expected) < 1e-9
    print("OK")

    print("\n모든 검증 통과")


if __name__ == "__main__":
    _self_test()
