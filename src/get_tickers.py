"""Retrieve and normalize the current S&P 500 universe."""

from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "cache"
REPO_ZIP = "https://codeload.github.com/fja05680/sp500/zip/refs/heads/master"
REPO_RAW = "https://raw.githubusercontent.com/fja05680/sp500/master/"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# 각 항목은 뉴스/공시로 "같은 회사다"를 먼저 확인한 뒤 손으로 추가한 것.
# 검증 방법: 옛 티커의 마지막 end_date와 새 티커의 첫 start_date가 하루도
# 안 어긋나게 정확히 맞물리는지 raw 멤버십 이력에서 대조 (날짜 경계 매칭).
# 주의: 날짜 경계 매칭만으로는 "같은 회사"를 새로 찾아낼 수 없음 — 아무 관련
# 없는 종목이 같은 날 교체돼도 똑같은 패턴이 나오기 때문에(예: BSC 편출 당일
# ISRG 편입, 리네임 아님), 이 방식은 이미 알고 있는 매핑이 데이터와 모순되지
# 않는지 사후 확인하는 용도로만 쓸 것 — 신규 리네임 탐지에는 쓰지 말 것.
#
# TODO: SEC EDGAR가 무료로 제공하는 티커<->CIK(SEC 등록번호) 매핑을 추가하면
# 좋을듯. CIK는 미국 상장사가 SEC에 등록될 때 받는 번호로, 사명/티커가 바뀌어도
# 안 바뀌므로 "같은 법인"인지 날짜 패턴보다 훨씬 직접적으로 검증 가능함.
TRUSTED_RENAMES = {
    "ABC": "COR", "ANTM": "ELV", "BLL": "BALL", "CDAY": "DAY",
    "FB": "META", "FBHS": "FBIN", "FLT": "CPAY", "FISV": "FI",
    "GPS": "GAP", "NLOK": "GEN", "PKI": "RVTY", "RE": "EG",
    "VIAC": "PARA", "WLTW": "WTW", "WRK": "SW",
}


def norm_ticker(value: str) -> str:
    """Convert a symbol to the Yahoo convention (for example BRK.B -> BRK-B)."""
    return str(value).strip().upper().replace(".", "-")


def _mirror_urls(url: str) -> list[str]:
    match = re.match(
        r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)", url
    )
    if not match:
        return [url]
    owner, repo, branch, path = match.groups()
    return [
        f"https://cdn.jsdelivr.net/gh/{owner}/{repo}@{branch}/{path}",
        f"https://github.com/{owner}/{repo}/raw/{branch}/{path}",
        url,
    ]


def _read_csv_url(url: str, retries: int = 2) -> Optional[pd.DataFrame]:
    for candidate in _mirror_urls(url):
        host = candidate.split("/")[2]
        for attempt in range(retries):
            try:
                response = requests.get(candidate, timeout=30, headers={"User-Agent": UA})
                if response.status_code == 429:
                    time.sleep(4 * (attempt + 1))
                    continue
                response.raise_for_status()
                frame = pd.read_csv(io.StringIO(response.text))
                print(f"[INFO] {host} 에서 확보 ({len(frame):,}행)")
                return frame
            except Exception as exc:
                print(f"[WARN] {host} 실패: {type(exc).__name__}")
                time.sleep(2)
    return None


def _read_from_repo_zip(filename: str = "sp500.csv") -> Optional[pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = CACHE_DIR / "sp500_repo.zip"
    if not archive_path.exists():
        try:
            response = requests.get(REPO_ZIP, timeout=180, headers={"User-Agent": UA})
            response.raise_for_status()
            archive_path.write_bytes(response.content)
            print(f"[INFO] 저장소 zip 확보 ({len(response.content) / 1024:.0f} KB)")
        except Exception as exc:
            print(f"[WARN] zip 조회 실패: {type(exc).__name__}")
            return None
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = [name for name in archive.namelist() if name.endswith(f"/{filename}")]
            return pd.read_csv(archive.open(names[0])) if names else None
    except Exception as exc:
        print(f"[WARN] zip 읽기 실패: {type(exc).__name__}")
        return None


def _read_from_wikipedia() -> Optional[pd.DataFrame]:
    try:
        return pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    except Exception as exc:
        print(f"[WARN] 위키백과 폴백 실패: {type(exc).__name__}")
        return None


def get_sp500_universe(force_refresh: bool = False) -> pd.DataFrame:
    """Return the current constituents as ticker/company/sector data."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "sp500_universe.csv"
    if cache_path.exists() and not force_refresh:
        frame = pd.read_csv(cache_path)
        print(f"[캐시] 구성종목 {len(frame)}건 재사용 ({cache_path})")
        return frame

    raw = _read_from_repo_zip()
    if raw is None:
        raw = _read_csv_url(REPO_RAW + "sp500.csv")
    if raw is None:
        print("[WARN] 저장소 실패 -> 위키백과로 폴백")
        raw = _read_from_wikipedia()
    if raw is None:
        raise RuntimeError("현재 S&P 500 구성종목 목록을 어느 소스에서도 받지 못했습니다.")

    columns = {str(column).lower().strip(): column for column in raw.columns}

    def pick(*names: str):
        return next((columns[name] for name in names if name in columns), None)

    symbol_col = pick("symbol", "ticker")
    name_col = pick("security", "company", "name", "security name")
    sector_col = pick("gics sector", "sector")
    if symbol_col is None:
        raise RuntimeError(f"티커 컬럼을 찾지 못했습니다. 실제 컬럼: {list(raw.columns)}")

    frame = pd.DataFrame({"ticker": raw[symbol_col].map(norm_ticker)})
    frame["company"] = raw[name_col].astype(str).str.strip() if name_col else frame["ticker"]
    frame["sector"] = raw[sector_col].astype(str).str.strip() if sector_col else "Unknown"
    frame["ticker_original"] = frame["ticker"]
    frame["ticker"] = frame["ticker"].map(lambda ticker: TRUSTED_RENAMES.get(ticker, ticker))
    frame = (
        frame.dropna(subset=["ticker"])
        .query("ticker != '' and ticker != 'NAN'")
        .drop_duplicates("ticker", keep="first")
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    frame.to_csv(cache_path, index=False)
    print(f"[INFO] 정규화/중복 제거 후 구성종목 {len(frame)}건")
    return frame


# --- 생존자 편향(survivorship bias) 보정 ---
#
# get_sp500_universe()는 "오늘" 기준 S&P500 구성종목만 준다. 그런데 10년치 과거
# 데이터를 이 503개 종목으로만 채우면, 그 사이 파산·상장폐지·인수합병으로
# 편출된 종목은 애초에 존재하지 않는 셈이 된다. 결과적으로 "지금까지 살아남은
# 우량 종목"만 남은 데이터로 과거를 분석하게 되어, 백테스팅 성과가 실제보다
# 좋게 나오는 착시가 생긴다 (survivorship bias). 예: 2018년에 상장폐지된
# 종목은 2016~2017년에는 분명히 실존했지만, "오늘 기준" 유니버스에는 없어서
# 2016~2017년 분석에서도 통째로 빠지게 된다.
#
# 아래 함수는 fja05680/sp500 저장소의 sp500_ticker_start_end.csv(티커별
# 편입일~편출일 이력)를 이용해, 분석 기간 [start, end) 동안 단 하루라도
# S&P500 구성종목이었던 모든 티커(편출된 종목 포함)를 복원한다.
def _load_membership_history(force_refresh: bool = False) -> pd.DataFrame:
    """Load and normalize the ticker-level S&P 500 membership interval history.

    A ticker can appear multiple times (rejoined after being removed), so the
    same ticker may have several disjoint [start_date, end_date) intervals.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "sp500_membership_history.csv"
    if cache_path.exists() and not force_refresh:
        membership = pd.read_csv(cache_path, parse_dates=["start_date", "end_date"])
        print(f"[캐시] 편입/편출 이력 {len(membership)}건 재사용 ({cache_path})")
    else:
        raw = _read_from_repo_zip("sp500_ticker_start_end.csv")
        if raw is None:
            raise RuntimeError("S&P 500 편입/편출 이력을 저장소에서 받지 못했습니다.")
        membership = raw.copy()
        membership["start_date"] = pd.to_datetime(membership["start_date"], errors="coerce")
        membership["end_date"] = pd.to_datetime(membership["end_date"], errors="coerce")
        membership.to_csv(cache_path, index=False)
        print(f"[INFO] 편입/편출 이력 {len(membership)}건 확보")
    membership["ticker"] = membership["ticker"].map(norm_ticker)
    membership["ticker"] = membership["ticker"].map(lambda ticker: TRUSTED_RENAMES.get(ticker, ticker))
    return membership


def get_historical_sp500_universe(
    start: str, end: str, force_refresh: bool = False
) -> pd.DataFrame:
    """Return every ticker that was an S&P 500 member at any point in [start, end).

    Unlike get_sp500_universe() (today's constituents only), this also includes
    tickers removed before ``end`` (delisted, acquired, merged) so that
    historical backtests are not built on a survivor-only universe.
    """
    membership = _load_membership_history(force_refresh)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    # 편입일이 분석기간 끝보다 이전이고, 편출일이 없거나(현재도 편입 상태) 편출일이
    # 분석기간 시작 이후면, 그 구간 동안 최소 하루는 실제로 구성종목이었다는 뜻이다.
    overlaps = (membership["start_date"] < end_ts) & (
        membership["end_date"].isna() | (membership["end_date"] >= start_ts)
    )
    tickers = sorted(membership.loc[overlaps, "ticker"].dropna().unique())
    frame = pd.DataFrame({"ticker_original": tickers})
    frame["ticker"] = frame["ticker_original"].map(norm_ticker)
    frame["ticker"] = frame["ticker"].map(lambda ticker: TRUSTED_RENAMES.get(ticker, ticker))
    frame = frame.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    print(f"[INFO] {start}~{end} 기간 중 한 번이라도 구성종목이었던 티커 {len(frame)}건")
    return frame


# --- 리밸런싱 시점별 자격(point-in-time membership) 필터 ---
#
# get_historical_sp500_universe()는 "수집 대상"을 정하는 용도라 "한 번이라도"
# 구성종목이었으면 포함시킨다 (넓게 모으는 게 목적). 하지만 리밸런싱 스냅샷은
# 성격이 다르다 — "그 날짜에 실제로 골라 담을 수 있었던 종목"만 후보여야 한다.
# 예: 2010년에 S&P500에 재편입된 종목이라도, 2016년 스냅샷에 그 종목이 있으면
# 안 된다는 게 아니라(2016년엔 이미 편입 상태였으므로 정상) — 반대로 2026년에
# 막 편입된 종목이 2016년 스냅샷에 나타나면 안 된다는 뜻이다. 한 종목이 여러 번
# 편입/편출됐을 수 있으므로(예: AAL은 1996~1997, 2015~2024 두 구간), 특정
# 구간 하나만 보면 안 되고 전체 이력 중 하나라도 그 날짜를 포함하는지 확인해야 한다.
def filter_by_membership(
    df: pd.DataFrame, ticker_col: str = "Ticker", date_col: str = "Date",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Keep only rows where ``ticker_col`` was an actual S&P 500 member on ``date_col``."""
    membership = _load_membership_history(force_refresh)[["ticker", "start_date", "end_date"]]
    merged = df.merge(membership, left_on=ticker_col, right_on="ticker", how="left")
    is_member = (merged[date_col] >= merged["start_date"]) & (
        merged["end_date"].isna() | (merged[date_col] < merged["end_date"])
    )
    merged["_is_member"] = is_member.fillna(False)
    # 같은 (종목, 날짜)가 여러 편입 구간과 매칭될 수 있으므로, 그중 하나라도
    # 참이면(그 구간에 속하면) 최종적으로 자격 있는 것으로 인정한다.
    eligible = (
        merged.groupby([ticker_col, date_col])["_is_member"]
        .any()
        .rename("_eligible")
        .reset_index()
    )
    result = df.merge(eligible, on=[ticker_col, date_col], how="left")
    dropped = int((~result["_eligible"].fillna(False)).sum())
    if dropped:
        print(f"[INFO] 시점별 자격 미달로 제외된 (종목,날짜) 조합: {dropped}건 / {len(df)}건")
    return result.loc[result["_eligible"].fillna(False)].drop(columns="_eligible").reset_index(drop=True)
