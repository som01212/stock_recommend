"""Fama-French factor attribution for the clustered strategy.

Added 2026-09-07. Everything before this answered "did the stable cluster
score higher"; nothing answered the question a factor researcher asks
first: **is this just a repackaging of factors that are already known?**

A portfolio built by selecting low beta and low volatility mechanically
tilts toward large, defensive, value-ish names. Those tilts are exactly
what the Fama-French factors price. So "the stable group earned more per
unit of risk" is not yet evidence of anything new -- the test is whether
an ALPHA survives after the known factor exposures are regressed out:

    r_p - r_f  =  alpha  +  b_mkt*(Mkt-RF) + b_smb*SMB + b_hml*HML + b_mom*MOM  +  e

A significant positive alpha would mean the selection rule found something
the four factors do not already explain. An alpha indistinguishable from
zero means the strategy is a (possibly fine) way to buy known factor
exposure, not a new source of return.

Data: Ken French's data library (free, public). Daily factor returns get
compounded over EACH holding period so the regression's left and right
sides cover exactly the same days -- the strategy's own T+1-to-next-T+1
windows, never calendar months.

Sample-size honesty: the regression spends 5 parameters. At 30-day
rebalancing that leaves 80 degrees of freedom (workable); at 252-day
rebalancing there are 9 observations total, so the model is not
identifiable and this module refuses rather than printing a number that
looks like an answer.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "cache"
FF_CACHE = CACHE_DIR / "ff_factors_daily.csv"

FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF_3FACTOR = "F-F_Research_Data_Factors_daily_CSV.zip"
FF_MOMENTUM = "F-F_Momentum_Factor_daily_CSV.zip"

FACTORS = ("Mkt-RF", "SMB", "HML", "MOM")
# 회귀가 쓰는 파라미터 수(절편 + 팩터 4개)의 최소 3배는 되어야 결과를 신뢰한다.
MIN_OBS_PER_PARAM = 3


def _read_ff_zip(filename: str) -> pd.DataFrame:
    """Ken French zip 하나를 날짜 인덱스 DataFrame으로 읽는다."""
    request = urllib.request.Request(FF_BASE + filename, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    text = archive.read(archive.namelist()[0]).decode("utf-8", "ignore")

    rows = []
    for line in text.splitlines():
        head = line.split(",")[0].strip()
        # 일별 데이터 행만 취한다 — 파일 앞뒤의 설명문과 연간 요약표를 건너뛴다.
        if len(head) == 8 and head.isdigit():
            rows.append(line)
    if not rows:
        raise RuntimeError(f"{filename}에서 일별 데이터 행을 찾지 못했습니다.")

    header = next(l for l in text.splitlines() if l.startswith(","))
    frame = pd.read_csv(io.StringIO(header + "\n" + "\n".join(rows)))
    frame = frame.rename(columns={frame.columns[0]: "Date"})
    frame.columns = [c.strip() for c in frame.columns]
    frame["Date"] = pd.to_datetime(frame["Date"], format="%Y%m%d")
    # Ken French는 퍼센트로 배포한다 (0.53 = 0.53%).
    for column in frame.columns:
        if column != "Date":
            frame[column] = pd.to_numeric(frame[column], errors="coerce") / 100.0
    return frame.set_index("Date").sort_index()


def load_ff_factors(force_refresh: bool = False) -> pd.DataFrame:
    """일별 Mkt-RF / SMB / HML / MOM / RF. 한 번 받아 캐시한다."""
    if FF_CACHE.exists() and not force_refresh:
        cached = pd.read_csv(FF_CACHE, parse_dates=["Date"]).set_index("Date").sort_index()
        print(f"[캐시] Fama-French 팩터 {len(cached):,}일 재사용 ({FF_CACHE})")
        return cached

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    three = _read_ff_zip(FF_3FACTOR)
    momentum = _read_ff_zip(FF_MOMENTUM)
    momentum.columns = ["MOM" if "Mom" in c else c for c in momentum.columns]

    factors = three.join(momentum[["MOM"]], how="inner")
    factors.reset_index().to_csv(FF_CACHE, index=False)
    print(f"[INFO] Fama-French 팩터 {len(factors):,}일 확보 "
          f"({factors.index.min().date()} ~ {factors.index.max().date()})")
    return factors


def compound_factors_to_periods(
    perf: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """각 보유기간(entry_date ~ actual_exit_date)에 대해 일별 팩터를 복리 누적한다.

    전략 수익률과 **정확히 같은 날짜 구간**을 덮어야 회귀가 성립한다. 달력 월이나
    고정 구간으로 근사하면 좌변과 우변이 서로 다른 기간을 보게 된다.
    """
    windows = (
        perf.loc[perf["forward_return"].notna(), ["Date", "entry_date", "actual_exit_date"]]
        .dropna()
        .drop_duplicates("Date")
        .sort_values("Date")
    )

    rows = []
    for _, w in windows.iterrows():
        span = factors.loc[(factors.index >= w["entry_date"]) & (factors.index <= w["actual_exit_date"])]
        if span.empty:
            continue
        row = {"Date": w["Date"], "n_days": len(span)}
        for column in span.columns:
            # 팩터 수익률도 전략 수익률과 같은 방식으로 복리 누적한다.
            row[column] = float((1.0 + span[column]).prod() - 1.0)
        rows.append(row)

    return pd.DataFrame(rows).set_index("Date")


def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    """절편 포함 OLS — 계수, 표준오차, t값, R², 조정 R²."""
    n, k = X.shape
    design = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = n - k - 1
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(cov))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return {
        "beta": beta, "se": se, "t": beta / se, "dof": dof,
        "r2": r2, "adj_r2": 1.0 - (1.0 - r2) * (n - 1) / dof if dof > 0 else float("nan"),
    }


def factor_regression(
    period_returns: pd.Series,
    period_factors: pd.DataFrame,
    label: str = "",
    subtract_rf: bool = True,
) -> pd.Series:
    """전략 수익률을 4팩터에 회귀해 알파가 남는지 본다.

    ``period_returns``는 보유기간별 전략 수익률(long-short 스프레드라면
    ``subtract_rf=False`` — 자기자금이 들어가지 않으므로 무위험수익률을 빼지 않는다).
    """
    joined = pd.concat([period_returns.rename("ret"), period_factors], axis=1).dropna()
    n = len(joined)
    params = len(FACTORS) + 1
    if n < params * MIN_OBS_PER_PARAM:
        return pd.Series({
            "label": label, "n": n,
            "note": f"관측 {n}개 < 파라미터 {params}개 x {MIN_OBS_PER_PARAM} — 회귀 불가",
        })

    y = (joined["ret"] - joined["RF"]).to_numpy() if subtract_rf else joined["ret"].to_numpy()
    X = joined[list(FACTORS)].to_numpy()
    fit = _ols(y, X)

    out = {"label": label, "n": n, "alpha": fit["beta"][0], "alpha_t": fit["t"][0]}
    for i, factor in enumerate(FACTORS, start=1):
        out[f"b_{factor}"] = fit["beta"][i]
        out[f"t_{factor}"] = fit["t"][i]
    out["adj_r2"] = fit["adj_r2"]
    # |t| >= 2 를 유의 기준으로 쓴다 (양측 5%의 통상적 근사).
    out["alpha_significant"] = bool(abs(fit["alpha_t"] if "alpha_t" in fit else fit["t"][0]) >= 2)
    return pd.Series(out)
