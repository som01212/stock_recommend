"""Collect OHLCV using Yahoo, Yahoo chart API, then Tiingo."""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm.auto import tqdm

from .get_tickers import norm_ticker

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "cache"
START_DATE = "2016-01-01"
# All collectors treat `end` as exclusive, so this includes 2026-06-30.
END_DATE = "2026-07-01"
AUTO_ADJUST = True
MAX_WORKERS = 4
MAX_RETRIES = 3
BASE_SLEEP = 1.5
MIN_ROWS_KEEP = 20
USE_TIINGO = True
TIINGO_MAX_REQ_PER_HOUR = 50
TIINGO_API_KEY = os.environ.get("TIINGO_API_KEY", "")
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
SCHEMA = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "source"]

try:
    from curl_cffi import requests as curl_requests

    HAS_CURL = True
except ImportError:
    HAS_CURL = False

_local = threading.local()


def get_session():
    if not hasattr(_local, "session"):
        if HAS_CURL:
            _local.session = curl_requests.Session(impersonate="chrome")
        else:
            _local.session = requests.Session()
            _local.session.headers.update({"User-Agent": UA})
    return _local.session


def _make_ticker(symbol: str):
    try:
        return yf.Ticker(symbol, session=get_session())
    except Exception:
        return yf.Ticker(symbol)


def _tidy(raw: pd.DataFrame, ticker: str, source: str) -> pd.DataFrame:
    frame = raw.reset_index()
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_localize(None)
    output = pd.DataFrame({"Date": dates.dt.normalize()})
    for column in ("Open", "High", "Low", "Close", "Volume"):
        output[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
    output["Ticker"] = ticker
    output["source"] = source
    output = output.dropna(subset=["Date", "Close"])
    return output.loc[output["Close"] > 0, SCHEMA].sort_values("Date").reset_index(drop=True)


def _clip(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (frame["Date"] >= pd.Timestamp(start)) & (frame["Date"] < pd.Timestamp(end))
    return frame.loc[mask].reset_index(drop=True)


def fetch_yahoo(symbol: str, start: str, end: str, label_as: Optional[str] = None):
    error = "unknown"
    ticker = _make_ticker(symbol)
    for attempt in range(MAX_RETRIES):
        try:
            raw = ticker.history(start=start, end=end, auto_adjust=AUTO_ADJUST, actions=False)
            if raw is not None and not raw.empty and raw["Close"].notna().any():
                return _clip(_tidy(raw, label_as or symbol, "yahoo"), start, end), None
            error = "empty response"
        except Exception as exc:
            error = type(exc).__name__
        if attempt < MAX_RETRIES - 1:
            time.sleep(BASE_SLEEP * (2**attempt) + random.uniform(0, 0.5))
    return None, error


def fetch_yahoo_chart(symbol: str, start: str, end: str, label_as: Optional[str] = None):
    try:
        response = get_session().get(
            f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={
                "period1": int(pd.Timestamp(start).timestamp()),
                "period2": int(pd.Timestamp(end).timestamp()),
                "interval": "1d",
                "events": "div,splits",
            },
            headers={"User-Agent": UA, "Accept": "application/json"},
            timeout=30,
        )
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"
        chart = json.loads(response.text).get("chart") or {}
        if chart.get("error"):
            return None, str(chart["error"].get("description"))[:80]
        result = (chart.get("result") or [None])[0]
        if not result or not result.get("timestamp"):
            return None, "no data"
        quote = (result["indicators"].get("quote") or [{}])[0]
        if not quote.get("close"):
            return None, "no close series"
        timezone = result.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
        dates = (
            pd.to_datetime(result["timestamp"], unit="s", utc=True)
            .tz_convert(timezone).tz_localize(None).normalize()
        )
        frame = pd.DataFrame({
            "Date": dates, "Open": quote.get("open"), "High": quote.get("high"),
            "Low": quote.get("low"), "Close": quote.get("close"), "Volume": quote.get("volume"),
        })
        if AUTO_ADJUST:
            adjusted = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose")
            if adjusted is not None:
                ratio = (pd.Series(adjusted, dtype="float64") / frame["Close"]).replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(1.0)
                for column in ("Open", "High", "Low", "Close"):
                    frame[column] *= ratio
        output = _clip(_tidy(frame, label_as or symbol, "yahoo:chart"), start, end)
        return (output, None) if len(output) else (None, "empty after clip")
    except Exception as exc:
        return None, type(exc).__name__


def collect_sp500_index(
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Collect the S&P 500 *price* index (``^GSPC``) separately for beta
    calculation.

    Deliberately a price index, not a total-return one (2026-09-02 note):
    beta is a covariance/variance ratio, and dividends add a roughly
    constant/smooth daily drift to the market return series that a
    sensitivity check found barely moves beta (mean abs diff ~0.002,
    see 05_clustering.ipynb) or cluster labels -- not worth the risk of
    silently mixing return conventions elsewhere. Use
    ``collect_sp500_total_return_index()`` for anything that compares
    cumulative strategy/benchmark performance, where the dividend drag
    compounds over years and does matter.

    The returned frame contains only ``Date`` and adjusted ``Close`` and is
    never mixed into the constituent OHLCV panel.
    """
    frame, yahoo_error = fetch_yahoo("^GSPC", start, end, label_as="^GSPC")
    if frame is None or len(frame) < MIN_ROWS_KEEP:
        frame, chart_error = fetch_yahoo_chart("^GSPC", start, end, label_as="^GSPC")
        if frame is None or len(frame) < MIN_ROWS_KEEP:
            raise RuntimeError(
                "S&P 500 지수(^GSPC) 수집 실패: "
                f"yfinance={yahoo_error}, yahoo_chart={chart_error}"
            )
    return (
        frame[["Date", "Close"]]
        .dropna()
        .drop_duplicates("Date", keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )


def collect_sp500_total_return_index(
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Collect the S&P 500 *total return* index (``^SP500TR``, dividends
    reinvested) for strategy-vs-benchmark performance comparison.

    Added 2026-09-02: every constituent in the OHLCV panel is fetched with
    ``auto_adjust=True`` (dividend-adjusted, i.e. already a total-return
    series per stock), so comparing dividend-adjusted portfolio returns
    against ``^GSPC`` (which excludes dividends) understates the benchmark
    by the S&P 500's dividend yield (~1.3-1.5%/year, compounding over a
    10-year window). ``^SP500TR`` matches the constituents' own return
    convention. Kept as a separate file from ``sp500_beta_df`` -- see
    ``collect_sp500_index()`` -- so a file's name always tells you which
    return convention it holds, instead of silently swapping the meaning
    of an existing file in place.

    The returned frame contains only ``Date`` and adjusted ``Close`` and is
    never mixed into the constituent OHLCV panel.
    """
    frame, yahoo_error = fetch_yahoo("^SP500TR", start, end, label_as="^SP500TR")
    if frame is None or len(frame) < MIN_ROWS_KEEP:
        frame, chart_error = fetch_yahoo_chart("^SP500TR", start, end, label_as="^SP500TR")
        if frame is None or len(frame) < MIN_ROWS_KEEP:
            raise RuntimeError(
                "S&P 500 총수익지수(^SP500TR) 수집 실패: "
                f"yfinance={yahoo_error}, yahoo_chart={chart_error}"
            )
    return (
        frame[["Date", "Close"]]
        .dropna()
        .drop_duplicates("Date", keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )


class TiingoFallback:
    """Rate-limited third source with positive and negative caches."""

    def __init__(self, api_key: str = "", max_req_per_hour: int = TIINGO_MAX_REQ_PER_HOUR):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self.pos_cache = CACHE_DIR / "tiingo_prices.csv"
        self.neg_cache = CACHE_DIR / "tiingo_failed.csv"
        self.sleep = 3600 / max(1, max_req_per_hour) + 1

    def _load_pos(self) -> pd.DataFrame:
        if not self.pos_cache.exists():
            return pd.DataFrame(columns=SCHEMA)
        return pd.read_csv(self.pos_cache, parse_dates=["Date"]).drop_duplicates(
            ["Ticker", "Date"], keep="last"
        )

    def _load_neg(self) -> set[str]:
        if not self.neg_cache.exists():
            return set()
        return set(pd.read_csv(self.neg_cache)["Ticker"].unique())

    def _fetch_one(self, symbol: str, start: str, end: str, max_retries: int = 3):
        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        params = {"startDate": start, "endDate": end, "format": "json", "token": self.api_key}
        error = "unknown"
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 404:
                    return None, "not found"
                if response.status_code == 429:
                    time.sleep(60)
                    continue
                response.raise_for_status()
                data = response.json()
                if not data:
                    return None, "empty response"
                frame = pd.DataFrame(data)
                frame["Date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
                frame = frame.rename(columns={
                    "adjOpen": "Open", "adjHigh": "High", "adjLow": "Low",
                    "adjClose": "Close", "adjVolume": "Volume",
                })
                return _clip(_tidy(frame, symbol, "tiingo"), start, end), None
            except Exception as exc:
                error = type(exc).__name__
            time.sleep(2 * (attempt + 1))
        return None, error

    def recover(self, targets: Sequence[str], start: str, end: str):
        cached = self._load_pos()
        start_ts = pd.Timestamp(start)
        last_needed_ts = pd.Timestamp(end) - pd.Timedelta(days=1)
        if len(cached):
            coverage = cached.groupby("Ticker")["Date"].agg(["min", "max"])
            have = set(
                coverage[
                    (coverage["min"] <= start_ts) & (coverage["max"] >= last_needed_ts)
                ].index
            )
        else:
            have = set()
        bad = self._load_neg()
        todo = [ticker for ticker in targets if ticker not in have and ticker not in bad]
        errors: dict[str, str] = {ticker: "tiingo: cached failure" for ticker in targets if ticker in bad}
        if todo and not self.api_key:
            errors.update({ticker: "tiingo: API key missing" for ticker in todo})
            todo = []
        frames, new_negative = [], []
        # 종목당 self.sleep(약 73초)씩 강제 대기가 있어 오래 걸린다.
        # tqdm으로 진행률을 표시해 "멈춘 것처럼" 보이지 않게 한다.
        for index, ticker in enumerate(tqdm(todo, desc="3차 Tiingo")):
            frame, error = self._fetch_one(ticker, start, end)
            if frame is not None and len(frame):
                frames.append(frame)
                frame.to_csv(self.pos_cache, mode="a", index=False, header=not self.pos_cache.exists())
            else:
                errors[ticker] = f"tiingo: {error}"
                new_negative.append({"Ticker": ticker, "error": error, "checked_at": str(pd.Timestamp.now().date())})
            if index < len(todo) - 1:
                time.sleep(self.sleep)
        if new_negative:
            old = pd.read_csv(self.neg_cache) if self.neg_cache.exists() else pd.DataFrame()
            pd.concat([old, pd.DataFrame(new_negative)], ignore_index=True).drop_duplicates(
                "Ticker", keep="last"
            ).to_csv(self.neg_cache, index=False)
        available = [frame for frame in [cached, *frames] if len(frame)]
        merged = pd.concat(available, ignore_index=True) if available else pd.DataFrame(columns=SCHEMA)
        if len(merged):
            merged = merged.drop_duplicates(["Ticker", "Date"], keep="last")
            merged = _clip(merged[merged["Ticker"].isin(targets)].reset_index(drop=True), start, end)
        return merged, errors


def make_df(
    tickers: Sequence[str], start: str = START_DATE, end: str = END_DATE,
    universe: Optional[pd.DataFrame] = None, use_tiingo: bool = USE_TIINGO,
    tiingo_key: str = TIINGO_API_KEY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return collected prices and only the final unrecovered symbols/reasons."""
    tickers = list(dict.fromkeys(norm_ticker(ticker) for ticker in tickers))
    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}

    def work(ticker: str):
        frame, error = fetch_yahoo(ticker, start, end)
        return ticker, frame, error

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(work, ticker): ticker for ticker in tickers}
        for future in tqdm(as_completed(futures), total=len(tickers), desc="1차 yfinance"):
            ticker, frame, error = future.result()
            if frame is not None and len(frame) >= MIN_ROWS_KEEP:
                frames.append(frame)
            else:
                errors[ticker] = f"yahoo: {error or f'rows={0 if frame is None else len(frame)}'}"

    for ticker in tqdm(sorted(errors), desc="2차 Yahoo chart"):
        frame, error = fetch_yahoo_chart(ticker, start, end)
        if frame is not None and len(frame) >= MIN_ROWS_KEEP:
            frames.append(frame)
            errors.pop(ticker, None)
        else:
            errors[ticker] += f" | chart: {error}"
        time.sleep(0.2)

    if errors and use_tiingo:
        recovered, tiingo_errors = TiingoFallback(tiingo_key).recover(sorted(errors), start, end)
        for ticker, group in recovered.groupby("Ticker"):
            if len(group) >= MIN_ROWS_KEEP:
                frames.append(group.reset_index(drop=True))
                errors.pop(ticker, None)
        for ticker, error in tiingo_errors.items():
            if ticker in errors:
                errors[ticker] += f" | {error}"

    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCHEMA)
    info = universe[["ticker", "company", "sector"]].drop_duplicates("ticker") if universe is not None else pd.DataFrame(
        {"ticker": tickers, "company": "Unknown", "sector": "Unknown"}
    )
    counts = prices.groupby("Ticker").size() if len(prices) else pd.Series(dtype="int64")
    rows = []
    for ticker in tickers:
        row_count = int(counts.get(ticker, 0))
        if row_count >= MIN_ROWS_KEEP:
            continue
        reason = errors.get(ticker, "unknown")
        stage = "yahoo+chart+tiingo" if "tiingo" in reason else "yahoo+chart" if "chart" in reason else "yahoo"
        rows.append({"ticker": ticker, "collected": False, "n_rows": row_count, "fail_stage": stage, "fail_reason": reason})
    missing = pd.DataFrame(rows, columns=["ticker", "collected", "n_rows", "fail_stage", "fail_reason"])
    missing = missing.merge(info, on="ticker", how="left")
    missing[["company", "sector"]] = missing[["company", "sector"]].fillna("Unknown")
    missing = missing[["ticker", "company", "sector", "collected", "n_rows", "fail_stage", "fail_reason"]]
    return prices, missing
