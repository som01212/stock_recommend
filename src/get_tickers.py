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

# --- 사명/티커 변경 매핑 (동일 법인 확인된 것만) ---
#
# 이 매핑이 없으면 무슨 일이 생기나 (2026-09-02에 실측으로 확인):
#   BNY(뉴욕멜론은행)는 2016~2026 내내 S&P500 구성종목이고 가격 데이터도
#   10년치(2,637행)를 전부 갖고 있는데, 리밸런싱 스냅샷에는 **한 번도**
#   등장하지 않았다. 멤버십 이력은 옛 티커 `BK`로 남아있고 가격은 새 티커
#   `BNY`로 수집돼서, 둘이 서로 못 만난 것이다. MRSH(1행), DOC, GL, J,
#   LUMN도 같은 이유로 구성종목 기간의 상당 부분을 잃고 있었다.
#   즉 이 매핑은 편의 기능이 아니라 생존자 편향 보정의 필수 부품이다.
#
# 검증 절차 (3단계) — 2026-09-02 확립.
#   1단계 · 후보 발견: 야후 검색 API(`/v1/finance/search`)에 **회사명**을 넣어
#           현재 티커를 역추적한다. 옛 티커로는 조회가 안 되므로(상장폐지된
#           심볼은 야후가 서빙하지 않음) 반드시 회사명으로 출발해야 한다.
#   2단계 · 법인 동일성 확정: SEC EDGAR의 무료 공개 데이터로 대조한다.
#           `https://www.sec.gov/files/company_tickers.json` (티커→CIK) 로 CIK를
#           얻고, `https://data.sec.gov/submissions/CIK##########.json` 의
#           `formerNames`(과거 사명과 변경일)를 확인한다. CIK는 사명/티커가
#           바뀌어도 안 바뀌므로 "같은 법인"을 날짜 패턴보다 직접 증명한다.
#           (SEC는 User-Agent에 연락처 명시를 요구하니 헤더를 붙일 것.)
#   3단계 · 시계열 연속성 확인: 새 티커의 가격대가 옛 종목과 맞는지 실측한다.
#           야후 메타데이터를 믿으면 안 되는 실제 사례 — VMRK의 `longName`은
#           "AvalonBay Communities"로 **잘못** 붙어 있었는데, 가격대가
#           $45~63(AvalonBay는 $128~192)이라 EQR 계열임이 드러났고, SEC
#           formerNames가 "EQUITY RESIDENTIAL(~2026-08-12)"로 확정해줬다.
#
# 왜 3단계가 다 필요한가: CIK만으로는 부족하다. company_tickers.json에는
# **현재** 티커만 있어서 옛 티커로 역조회가 안 되고(1단계가 필요), 사명 없이
# 티커만 바뀐 경우(BK→BNY)는 formerNames가 비어 있으며, 법인이 갈렸는데
# 가격은 이어지는 경우(아래 HFC 참고)는 CIK만 보면 판단이 갈린다(3단계가 필요).
#
# 기존 방식(날짜 경계 매칭 — 옛 티커의 end_date와 새 티커의 start_date가
# 맞물리는지 확인)은 **검증용으로만** 유효하다. 아무 관련 없는 종목이 같은 날
# 교체돼도 같은 패턴이 나오므로(예: BSC 편출 당일 ISRG 편입, 리네임 아님)
# 신규 리네임 탐지에는 절대 쓰지 말 것.
#
# 검증 결과 의도적으로 **제외**한 것 (가격 데이터는 이어지지만 법인이 다름):
#   HFC → DINO : HF Sinclair의 CIK가 1915657(2022년 신규 등록)이라 HollyFrontier와
#                별개 법인. 지주회사 전환으로 신설된 케이스.
#   CCE → CCEP : CCEP의 CIK 1650107은 2015년 "Spark Orange Ltd"로 신설된 법인이
#                이후 개명한 것. Coca-Cola Enterprises가 여기 합병된 구조라 연속이 아님.
#
# 매핑을 만들 수 없어 데이터가 영구 소실된 것 (무료 소스 기준, 2026-09-02 조사):
#   파산·청산      : BBBY, ENDP, MNK
#   현금 인수      : CSRA(→GD), JNPR(→HPE), NFX(→OVV) — 인수 주체의 주가 이력이라
#                    이어붙이면 완전히 다른 회사 데이터가 섞인다. 절대 매핑 금지.
#   합병 후 상장폐지: COG, CBS, ESV
#     · COG는 3단계 절차의 2단계(SEC)가 아니었으면 끝까지 원인 불명이었을 케이스다.
#       야후 검색은 "Coterra 결과 없음"으로 막혔지만, CIK 858470을 직접 조회하니
#       과거명 CABOT OIL & GAS CORP(~2021-09-29) → 현재명 Coterra Energy Inc.이고,
#       2026-05-19자 Form 15-12G(증권 등록 말소)가 제출돼 상장폐지된 것이 확인됐다.
#       등록 티커/거래소 필드도 비어 있다.
#   유료 데이터(CRSP 등)는 상장폐지 시점까지 이력을 보존하므로 이 그룹 상당수가
#   복구 가능하다 — 즉 이건 방법론 한계가 아니라 데이터 소스 한계다.
TRUSTED_RENAMES = {
    # 날짜 경계 매칭으로 확인한 기존 항목
    "ABC": "COR", "ANTM": "ELV", "BLL": "BALL", "CDAY": "DAY",
    "FB": "META", "FBHS": "FBIN", "FLT": "CPAY", "FISV": "FI",
    "GPS": "GAP", "NLOK": "GEN", "PKI": "RVTY", "RE": "EG",
    "VIAC": "PARA", "WLTW": "WTW", "WRK": "SW",
    # 2026-09-02 추가 — 위 3단계 절차로 검증. 괄호 안은 CIK와 SEC formerNames 근거.
    "ADS": "BFH",    # CIK 1101215, 과거명 ALLIANCE DATA SYSTEMS CORP (~2022-03-23)
    "BK": "BNY",     # CIK 1390777, Bank of New York Mellon Corp — 사명 유지, 티커만 변경
    "CTL": "LUMN",   # CIK 18926,   과거명 CENTURYLINK, INC (~2021-01-20)
    "EQR": "VMRK",   # CIK 906107,  과거명 EQUITY RESIDENTIAL (~2026-08-12)
    "JEC": "J",      # CIK 52988,   과거명 JACOBS ENGINEERING GROUP INC (~2022-08-19)
    "MMC": "MRSH",   # CIK 62709,   MARSH & MCLENNAN — 사명 유지, 티커만 변경
    "PEAK": "DOC",   # CIK 765880,  과거명 HCP, INC. (~2019-10-01)
    "TMK": "GL",     # CIK 320335,  과거명 TORCHMARK CORP (~2019-08-02)
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
