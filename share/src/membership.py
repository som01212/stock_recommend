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
