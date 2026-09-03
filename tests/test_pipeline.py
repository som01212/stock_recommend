"""performance.py / backtest.py 계산 로직 회귀 테스트.

2026-09-02 하루 감사에서 계산 버그가 다섯 종류 나왔다. 전략을 검증하는 코드가
정작 전략만큼 검증되지 않았던 게 원인이라, 그때 나온 버그를 **하나씩 재현하는**
합성 데이터 테스트를 남겨둔다. 각 테스트는 "고치기 전 코드였다면 실패했을"
형태로 작성했다 — 그래야 회귀를 실제로 잡는다.

커버하는 버그:
  1. 진입/청산이 신호 관찰일(T) 종가였던 것 → T+1이어야 함
  2. 위험대비수익·Sharpe를 종목x시점 풀링으로 계산한 것
     → 날짜별 동일가중 포트폴리오 시계열 기준이어야 함
  3. 보유기간 중 상장폐지 종목을 조용히 버린 것
     → 조건부 강제청산(terminal_last_close) / unresolved 구분
  4. 거래비용을 회전율에 비례시키지 않거나 첫 기간에도 물린 것
  5. 누적 곡선/MDD/Sharpe의 기본 산식

실행:
    python tests/test_pipeline.py        # 의존성 없이 그대로 실행
    pytest tests/test_pipeline.py        # pytest가 있으면 그대로 수집됨
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import performance as perf_mod
from src.backtest import (
    annualized_sharpe,
    apply_transaction_costs,
    equity_curve,
    max_drawdown,
    portfolio_returns,
    turnover_by_date,
)
from src.performance import (
    add_forward_returns,
    risk_adjusted_stats,
    sharpe_ratio_stats,
    summarize_by_group,
)

TRADING_DAYS = pd.bdate_range("2020-01-01", periods=12)


def _prices(rows: dict[str, list[float]], dates=TRADING_DAYS) -> pd.DataFrame:
    """{ticker: [close, ...]} → Date/Ticker/Close 패널. None은 미거래(행 없음)."""
    frames = []
    for ticker, closes in rows.items():
        frame = pd.DataFrame({"Date": dates[: len(closes)], "Ticker": ticker, "Close": closes})
        frames.append(frame.dropna(subset=["Close"]))
    return pd.concat(frames, ignore_index=True)


def _snapshots(dates, tickers, stable=True) -> pd.DataFrame:
    """리밸런싱 스냅샷(클러스터 라벨 포함) 최소 구성."""
    rows = []
    for date in dates:
        for ticker in tickers:
            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "cluster": 0,
                    "is_stable_cluster": stable if isinstance(stable, bool) else stable[ticker],
                }
            )
    return pd.DataFrame(rows)


def _always_member(_=False) -> pd.DataFrame:
    """모든 티커가 영구 구성종목이라고 보는 멤버십 스텁."""
    return pd.DataFrame(
        {"ticker": ["AAA", "BBB", "CCC", "GONE", "GAP"], "start_date": pd.Timestamp("2000-01-01"), "end_date": pd.NaT}
    )


# ----------------------------------------------------------------------
# 1. 진입/청산은 T가 아니라 T+1이어야 한다
# ----------------------------------------------------------------------
def test_entry_and_exit_shift_one_trading_day():
    # 리밸런싱 T0=0번째, T1=4번째 거래일. T와 T+1의 종가를 일부러 다르게 준다.
    closes = [100.0, 110.0, 111.0, 112.0, 200.0, 240.0, 241.0, 242.0]
    prices = _prices({"AAA": closes})
    clustered = _snapshots([TRADING_DAYS[0], TRADING_DAYS[4]], ["AAA"])

    perf_mod._load_membership_history = _always_member  # 멤버십 조회 차단
    result = add_forward_returns(clustered, prices)
    first = result[result["Date"] == TRADING_DAYS[0]].iloc[0]

    # T+1(110) → 다음 리밸런싱일 T+1(240)
    assert np.isclose(first["forward_return"], 240.0 / 110.0 - 1.0), first["forward_return"]
    # 고치기 전 동작(T종가 100 → 200 = +100%)이 아니어야 한다
    assert not np.isclose(first["forward_return"], 1.0)
    assert first["entry_date"] == TRADING_DAYS[1]
    assert first["intended_exit_date"] == TRADING_DAYS[5]
    assert first["exit_method"] == "scheduled_close"


# ----------------------------------------------------------------------
# 2. 위험은 포트폴리오 단위로 재야 한다 (종목x시점 풀링 금지)
# ----------------------------------------------------------------------
def test_risk_is_measured_on_portfolio_series_not_pooled():
    # 안정군 두 종목이 매 시점 정확히 반대로 움직인다 →
    #   동일가중 포트폴리오 수익률은 매번 같은 값(변동성 0에 가까움)
    #   반면 종목x시점을 풀링하면 큰 변동성이 잡힌다.
    dates = [TRADING_DAYS[0], TRADING_DAYS[2], TRADING_DAYS[4]]
    prices = _prices(
        {
            "AAA": [100, 100, 100, 130, 130, 171.6, 171.6, 171.6],
            "BBB": [100, 100, 100, 70, 70, 50.4, 50.4, 50.4],
            "CCC": [100, 100, 100, 105, 105, 110.25, 110.25, 110.25],
        }
    )
    clustered = _snapshots(dates, ["AAA", "BBB", "CCC"],
                           stable={"AAA": True, "BBB": True, "CCC": False})
    perf_mod._load_membership_history = _always_member

    result = add_forward_returns(clustered, prices)
    stats = risk_adjusted_stats(result, min_group_size=1)

    # 날짜가 3개, 그중 마지막은 다음 시점이 없어 제외 → 유효 시점 2개
    assert stats.loc["stable_cluster", "n"] == 2, stats

    per_date = summarize_by_group(result, min_group_size=1)["stable_mean_return"].dropna()
    assert np.isclose(stats.loc["stable_cluster", "std"], per_date.std()), "포트폴리오 시계열 std가 아님"

    # 풀링 std는 포트폴리오 std보다 훨씬 크다 — 그 값이 잡히면 회귀
    valid = result[result["forward_return"].notna()]
    pooled_std = valid["forward_return"].std()
    assert stats.loc["stable_cluster", "std"] > 0
    assert pooled_std > stats.loc["stable_cluster", "std"] * 10, (pooled_std, stats)


def test_sharpe_uses_same_portfolio_series():
    dates = [TRADING_DAYS[0], TRADING_DAYS[2], TRADING_DAYS[4]]
    prices = _prices({"AAA": [100, 100, 100, 130, 130, 171.6, 171.6, 171.6],
                      "BBB": [100, 100, 100, 70, 70, 50.4, 50.4, 50.4],
                      "CCC": [100, 100, 100, 105, 105, 110, 110, 116.0]})
    clustered = _snapshots(dates, ["AAA", "BBB", "CCC"],
                           stable={"AAA": True, "BBB": True, "CCC": False})
    perf_mod._load_membership_history = _always_member

    result = add_forward_returns(clustered, prices)
    ra = risk_adjusted_stats(result, min_group_size=1)
    sh = sharpe_ratio_stats(result, window=60, min_group_size=1)
    # 두 함수가 같은 시계열을 봐야 한다 (n / mean / std 동일)
    for col in ("n", "mean", "std"):
        assert np.isclose(ra.loc["stable_cluster", col], sh.loc["stable_cluster", col]), col


# ----------------------------------------------------------------------
# 3. 보유기간 중 상장폐지 처리 — 조건부 강제청산
# ----------------------------------------------------------------------
def _membership_gone_only(_=False) -> pd.DataFrame:
    """GONE만 편출됨(구성종목 종료), GAP은 계속 구성종목."""
    return pd.DataFrame(
        {
            "ticker": ["AAA", "GONE", "GAP"],
            "start_date": [pd.Timestamp("2000-01-01")] * 3,
            "end_date": [pd.NaT, TRADING_DAYS[3], pd.NaT],
        }
    )


def test_delisted_with_membership_ended_is_force_liquidated():
    # GONE은 3번째 거래일까지만 거래되고 사라진다 + 멤버십도 종료됨
    prices = _prices(
        {
            "AAA": [100, 101, 102, 103, 104, 105, 106, 107],
            "GONE": [100, 120, 150, None, None, None, None, None],
        }
    )
    clustered = _snapshots([TRADING_DAYS[0], TRADING_DAYS[4]], ["AAA", "GONE"])
    perf_mod._load_membership_history = _membership_gone_only

    result = add_forward_returns(clustered, prices)
    gone = result[(result["Ticker"] == "GONE") & (result["Date"] == TRADING_DAYS[0])].iloc[0]

    assert gone["exit_method"] == "terminal_last_close", gone["exit_method"]
    assert gone["actual_exit_date"] == TRADING_DAYS[2]          # 마지막 유효 거래일
    assert np.isclose(gone["forward_return"], 150.0 / 120.0 - 1.0)  # 진입 T+1(120) → 마지막(150)
    # 고치기 전에는 조용히 NaN으로 버려졌다
    assert not np.isnan(gone["forward_return"])


def test_missing_price_while_still_a_member_stays_unresolved():
    # GAP은 가격만 끊기고 멤버십은 살아있다 → 상장폐지가 아니므로 추정 금지
    prices = _prices(
        {
            "AAA": [100, 101, 102, 103, 104, 105, 106, 107],
            "GAP": [100, 120, 150, None, None, None, None, None],
        }
    )
    clustered = _snapshots([TRADING_DAYS[0], TRADING_DAYS[4]], ["AAA", "GAP"])
    perf_mod._load_membership_history = _membership_gone_only

    result = add_forward_returns(clustered, prices)
    gap = result[(result["Ticker"] == "GAP") & (result["Date"] == TRADING_DAYS[0])].iloc[0]

    assert gap["exit_method"] == "unresolved", gap["exit_method"]
    assert np.isnan(gap["forward_return"])


def test_warmup_and_last_snapshot_rows_are_excluded_from_summary():
    prices = _prices({"AAA": [100, 110, 120, 130, 140, 150, 160, 170],
                      "BBB": [100, 105, 110, 115, 120, 125, 130, 135],
                      "CCC": [100, 102, 104, 106, 108, 110, 112, 114]})
    clustered = _snapshots([TRADING_DAYS[0], TRADING_DAYS[4]], ["AAA", "BBB", "CCC"],
                           stable={"AAA": True, "BBB": True, "CCC": False})
    clustered.loc[clustered["Ticker"] == "BBB", "cluster"] = -1   # 워밍업 행
    perf_mod._load_membership_history = _always_member

    result = add_forward_returns(clustered, prices)
    summary = summarize_by_group(result, min_group_size=1)
    assert len(summary) == 1                      # 마지막 스냅샷은 다음 시점이 없어 제외
    assert summary.index[0] == TRADING_DAYS[0]
    # cluster == -1 행이 평균에 섞이면 안 된다
    assert np.isclose(summary.iloc[0]["stable_mean_return"], 150.0 / 110.0 - 1.0)


# ----------------------------------------------------------------------
# 3b. 퇴화 스냅샷(한쪽 그룹이 1~2종목)은 비교에서 빠지고, 반드시 보고된다
# ----------------------------------------------------------------------
def test_degenerate_snapshot_is_dropped_and_reported(capsys=None):
    import io, contextlib

    # 2번째 리밸런싱 시점에서 나머지군이 CCC 한 종목뿐이 되도록 구성
    prices = _prices({
        "AAA": [100, 100, 100, 110, 110, 121, 121, 121],
        "BBB": [100, 100, 100, 108, 108, 118, 118, 118],
        "CCC": [100, 100, 100, 105, 105, 5, 5, 5],      # 마지막 구간 -95%
        "DDD": [100, 100, 100, 106, 106, 112, 112, 112],
    })
    dates = [TRADING_DAYS[0], TRADING_DAYS[2], TRADING_DAYS[4]]
    stable = {"AAA": True, "BBB": True, "CCC": False, "DDD": True}
    clustered = _snapshots(dates, ["AAA", "BBB", "CCC", "DDD"], stable=stable)
    perf_mod._load_membership_history = _always_member
    result = add_forward_returns(clustered, prices)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary = summarize_by_group(result, min_group_size=2)
    printed = buf.getvalue()

    # 나머지군이 1종목인 시점은 전부 빠져야 한다
    assert (summary["other_n"] >= 2).all(), summary
    assert len(summary) == 0, "이 구성에서는 모든 시점의 나머지군이 1종목이라 전부 제외"
    # 조용히 넘어가면 안 된다 — 반드시 보고
    assert "제외한 스냅샷" in printed, printed


def test_group_sizes_are_reported_in_summary():
    prices = _prices({
        "AAA": [100, 100, 100, 110, 110, 121, 121, 121],
        "BBB": [100, 100, 100, 108, 108, 118, 118, 118],
        "CCC": [100, 100, 100, 105, 105, 110, 110, 110],
        "DDD": [100, 100, 100, 106, 106, 112, 112, 112],
    })
    dates = [TRADING_DAYS[0], TRADING_DAYS[2]]
    stable = {"AAA": True, "BBB": True, "CCC": False, "DDD": False}
    clustered = _snapshots(dates, ["AAA", "BBB", "CCC", "DDD"], stable=stable)
    perf_mod._load_membership_history = _always_member

    summary = summarize_by_group(add_forward_returns(clustered, prices), min_group_size=2)
    assert {"stable_n", "other_n"} <= set(summary.columns), summary.columns.tolist()
    assert summary.iloc[0]["stable_n"] == 2 and summary.iloc[0]["other_n"] == 2


# ----------------------------------------------------------------------
# 4~5. 백테스트 산식
# ----------------------------------------------------------------------
def _selection_frame() -> pd.DataFrame:
    d0, d1, d2 = TRADING_DAYS[0], TRADING_DAYS[1], TRADING_DAYS[2]
    return pd.DataFrame(
        [
            {"Date": d0, "Ticker": "AAA", "picked": True, "forward_return": 0.10},
            {"Date": d0, "Ticker": "BBB", "picked": True, "forward_return": 0.20},
            {"Date": d0, "Ticker": "CCC", "picked": False, "forward_return": 0.90},
            {"Date": d1, "Ticker": "AAA", "picked": True, "forward_return": -0.10},
            {"Date": d1, "Ticker": "CCC", "picked": True, "forward_return": 0.30},
            {"Date": d2, "Ticker": "CCC", "picked": True, "forward_return": 0.05},
            {"Date": d2, "Ticker": "DDD", "picked": True, "forward_return": 0.15},
        ]
    )


def test_portfolio_returns_are_equal_weighted_and_ignore_unpicked():
    df = _selection_frame()
    got = portfolio_returns(df, "picked")
    assert np.isclose(got.iloc[0], 0.15)   # (0.10+0.20)/2 — 0.90짜리 미선정 종목 제외
    assert np.isclose(got.iloc[1], 0.10)   # (-0.10+0.30)/2
    assert np.isclose(got.iloc[2], 0.10)   # (0.05+0.15)/2


def test_turnover_excludes_inception_and_counts_new_names():
    df = _selection_frame()
    got = turnover_by_date(df, "picked")
    assert TRADING_DAYS[0] not in got.index          # 최초 편입은 회전이 아님
    assert np.isclose(got.loc[TRADING_DAYS[1]], 0.5)  # {AAA,CCC} 중 CCC가 신규
    assert np.isclose(got.loc[TRADING_DAYS[2]], 0.5)  # {CCC,DDD} 중 DDD가 신규


def test_costs_scale_with_turnover_and_skip_first_period():
    df = _selection_frame()
    gross = portfolio_returns(df, "picked")
    net = apply_transaction_costs(gross, df, "picked", round_trip_cost=0.01)
    assert np.isclose(net.iloc[0], gross.iloc[0])                 # 첫 기간은 비용 없음
    assert np.isclose(net.iloc[1], gross.iloc[1] - 0.5 * 0.01)    # 회전율 비례
    assert np.isclose(net.iloc[2], gross.iloc[2] - 0.5 * 0.01)


def test_equity_curve_and_drawdown():
    returns = pd.Series([0.10, -0.20, 0.25])
    equity = equity_curve(returns)
    assert np.isclose(equity.iloc[-1], 1.10 * 0.80 * 1.25)
    # 고점 1.10 → 저점 0.88 → -20%
    assert np.isclose(max_drawdown(equity), 0.88 / 1.10 - 1.0)


def test_annualized_sharpe_matches_formula():
    returns = pd.Series([0.05, 0.01, 0.03, -0.02])
    window, rate = 63, 0.02
    period_rf = (1 + rate) ** (window / 252) - 1
    expected = (returns - period_rf).mean() / returns.std() * np.sqrt(252 / window)
    assert np.isclose(annualized_sharpe(returns, window=window, annual_risk_free_rate=rate), expected)


# ----------------------------------------------------------------------
def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = []
    for name, fn in tests:
        original = perf_mod._load_membership_history
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
        finally:
            perf_mod._load_membership_history = original

    print(f"\n{len(tests) - len(failures)}/{len(tests)} 통과")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
