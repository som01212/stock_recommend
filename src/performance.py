"""Forward-return performance validation for clustered rebalancing snapshots.

For every rebalancing date T, each ticker's forward return is measured from
the Close on the trading day *after* T (T+1) to the Close on the trading day
*after* the next rebalancing date (so the holding period still matches the
file's own rebalancing interval: 60 trading days for rebalance_60df, 120 for
rebalance_120df -- only the entry/exit day is shifted by one). The last
rebalancing date has no next date, so it gets NaN and is dropped from
summaries.

Why T+1 and not T itself (2026-09-02 fix): the feature values at T (beta_Nd,
volatility_Nd, return_Nd, rsi_Nd) are rolling windows computed *inclusive* of
T's own Close (see feature.py's "on or before" contract), so a signal at T
already knows T's closing price. Using that same T Close as the trade price
means "observe the close, then trade at that exact same close" -- not
achievable in real trading, where the earliest you can act on a T-close
signal is the T+1 session. Shifting both entry and exit by one trading day
removes this. Note market_returns_like() in notebooks/10_full_backtest.ipynb
must use the identical T+1 convention for the benchmark, or the strategy vs.
benchmark comparison stops being apples-to-apples.

Terminal-exit policy for mid-holding-period delisting (2026-09-02 fix): if a
ticker has no price at its scheduled exit date, the old behavior was to
silently drop that row (NaN forward_return) -- an audit of every such case in
this project's data (60 tickers, 61 occurrences, ~0.3% of clustered rows)
found they're almost all M&A (CELG, TIF, WFM, RHT, XLNX, ...), not bankruptcy
or a data gap, and the last available close is typically already sitting at
(or very near) the deal price, since the market prices in an announced deal
well before the ticker stops trading. Dropping these silently understates
returns, since M&A exits skew flat-to-positive, not catastrophic.

The new policy, applied only when the scheduled exit price is missing:
1. If the ticker still shows S&P 500 membership at the *next* rebalancing
   date, this is NOT a real delisting (probably a data-provider gap) --
   leave it unresolved and excluded, same as before.
2. Otherwise, if there's a valid close between entry and the scheduled exit,
   force-liquidate there (``exit_method='terminal_last_close'``).
3. Otherwise, leave it unresolved and excluded.

This assumes cash liquidation at the last trade for every terminal exit,
including stock-for-stock mergers (e.g. XLNX -> AMD) where an investor could
in principle have held the acquirer's shares onward instead. That's a
deliberate simplifying assumption, not a claim every deal was all-cash --
documented here so it isn't mistaken for an oversight.

This answers: did the "stable cluster" (is_stable_cluster == True) actually
hold up better/worse than the rest, on average, across every independent
rebalancing decision made in the backtest?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .get_tickers import _load_membership_history

TRADING_DAYS_PER_YEAR = 252


def add_forward_returns(clustered_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach each row's forward return, entering/exiting one trading day
    after the rebalancing date (see module docstring for why), falling back
    to a terminal-last-close liquidation for genuine mid-holding delistings
    (see module docstring's "Terminal-exit policy").

    ``prices`` must have Date, Ticker, Close columns (e.g. final_df).

    Adds ``entry_date``, ``intended_exit_date`` (the scheduled T+1 exit),
    ``actual_exit_date``, ``exit_price``, and ``exit_method`` (one of
    ``scheduled_close``, ``terminal_last_close``, ``unresolved``) so a
    caller can audit exactly how each row's exit was resolved.
    """
    trading_dates = sorted(prices["Date"].unique())
    next_trading_day = dict(zip(trading_dates[:-1], trading_dates[1:]))

    dates = sorted(clustered_df["Date"].unique())
    next_date_map = dict(zip(dates[:-1], dates[1:]))

    df = clustered_df.copy()
    df["next_date"] = df["Date"].map(next_date_map)
    df["entry_date"] = df["Date"].map(next_trading_day)
    df["intended_exit_date"] = df["next_date"].map(next_trading_day)

    entry = prices[["Date", "Ticker", "Close"]].rename(
        columns={"Date": "entry_date", "Close": "entry_price"}
    )
    df = df.merge(entry, on=["entry_date", "Ticker"], how="left")

    scheduled = prices[["Date", "Ticker", "Close"]].rename(
        columns={"Date": "intended_exit_date", "Close": "exit_price"}
    )
    df = df.merge(scheduled, on=["intended_exit_date", "Ticker"], how="left")

    df["actual_exit_date"] = df["intended_exit_date"]
    df["exit_method"] = np.where(df["exit_price"].notna(), "scheduled_close", "unresolved")

    # 스케줄된 청산가가 없는 행만 강제청산 정책 대상 -- entry_price도 없거나
    # next_date가 없는(마지막 리밸런싱 시점) 행은 애초에 대상에서 제외.
    needs_fallback = (
        df["exit_price"].isna() & df["entry_price"].notna() & df["next_date"].notna()
    )
    if needs_fallback.any():
        membership = _load_membership_history()[["ticker", "start_date", "end_date"]]
        prices_sorted = prices.sort_values("Date")

        for idx in df.index[needs_fallback]:
            ticker = df.at[idx, "Ticker"]
            next_date = df.at[idx, "next_date"]
            entry_date = df.at[idx, "entry_date"]
            intended_exit = df.at[idx, "intended_exit_date"]

            m = membership[membership["ticker"] == ticker]
            still_member = bool(
                (
                    (next_date >= m["start_date"])
                    & (m["end_date"].isna() | (next_date < m["end_date"]))
                ).any()
            )
            if still_member:
                continue  # 데이터 공급자 결측 등으로 추정 -- 실제 상장폐지가 아니므로 unresolved 유지

            candidates = prices_sorted[
                (prices_sorted["Ticker"] == ticker)
                & (prices_sorted["Date"] >= entry_date)
                & (prices_sorted["Date"] < intended_exit)
            ]
            if candidates.empty:
                continue  # 보유기간 중 유효한 마지막 종가도 없음 -- unresolved 유지

            last_row = candidates.iloc[-1]
            df.at[idx, "actual_exit_date"] = last_row["Date"]
            df.at[idx, "exit_price"] = last_row["Close"]
            df.at[idx, "exit_method"] = "terminal_last_close"

    df["forward_return"] = df["exit_price"] / df["entry_price"] - 1.0
    return df.drop(columns=["entry_price"])


DEFAULT_MIN_GROUP_SIZE = 5


def summarize_by_group(
    df: pd.DataFrame,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    verbose: bool = True,
) -> pd.DataFrame:
    """Per-date equal-weight return of the stable cluster vs the rest.

    Rows with cluster == -1 (warmup period, not enough history) or a missing
    forward return (last rebalancing date) are excluded. Snapshots where
    either side has fewer than ``min_group_size`` tickers are also dropped --
    see the "퇴화 스냅샷" note below for why -- and reported when that happens.

    Returns per-date ``stable_mean_return`` / ``other_mean_return`` /
    ``stable_minus_other`` plus the group sizes (``stable_n`` / ``other_n``)
    that produced them, so a reader can always see how many tickers each
    average was computed from.
    """
    valid = df[(df["cluster"] != -1) & df["forward_return"].notna()]
    grouped = valid.groupby(["Date", "is_stable_cluster"])["forward_return"]
    summary = (
        grouped.mean()
        .unstack("is_stable_cluster")
        .rename(columns={True: "stable_mean_return", False: "other_mean_return"})
    )
    sizes = (
        grouped.size()
        .unstack("is_stable_cluster")
        .rename(columns={True: "stable_n", False: "other_n"})
        .reindex(columns=["stable_n", "other_n"])
        .fillna(0)
        .astype(int)
    )
    summary = summary.join(sizes)

    # ------------------------------------------------------------------
    # 퇴화 스냅샷 제외 (2026-09-03 결정)
    #
    # 왜 필요한가 — 실제로 겪은 사례:
    #   FRC(First Republic Bank)를 복구하고 나니, 30일 리밸런싱의 2023-04-12
    #   시점에서 군집이 **안정군 500종목 / 나머지군 1종목(FRCB)**으로 갈렸다.
    #   붕괴 중이던 FRC의 beta/volatility가 극단값이라 Ward가 이 점 하나를
    #   따로 떼어낸 것이다(알고리즘은 정상 동작). 그 결과 "나머지군 평균
    #   수익률"이 FRCB 한 종목의 -97.9%가 되어버렸다.
    #
    #   영향은 작지 않았다 — 이 1개 시점이 30일 구간 전체를 이렇게 흔들었다:
    #     나머지군 위험대비수익   0.264 → 0.075
    #     안정-나머지 평균 격차  -0.53% → +0.57%  (부호가 뒤집힘)
    #   즉 "안정군 우위"라는 결론의 일부가 1종목짜리 가짜 포트폴리오에
    #   떠받쳐지고 있었다. 제외하면 결과가 이 프로젝트 주장에 **불리해지므로**,
    #   유리하게 만들기 위한 보정이 아니다.
    #
    # 왜 클러스터링이 아니라 여기서 막는가:
    #   Ward는 잘못한 게 없다. 최소 군집 크기를 강제하면 문제 1건을 고치려고
    #   161개 전 시점의 군집 소속을 바꾸게 된다. 반면 여기서 막으면 "이 날짜는
    #   유의미한 그룹 비교를 만들 수 없다"는 판단만 하는 것이고, 이미 하고 있는
    #   제외(cluster == -1 워밍업, forward_return 결측)와 성격이 같다.
    #   또 위험을 포트폴리오 단위로 재기로 한 결정(_portfolio_return_series)과도
    #   일관된다 — 1종목은 포트폴리오가 아니다.
    #
    # 임계값이 자의적이지 않은 이유:
    #   윈도우별 최소 그룹 크기가 30일 1 / 60일 25 / 120일 62 / 252일 64로,
    #   1과 25 사이가 비어 있다. 2~25 사이 어떤 값을 골라도 결과가 동일하다.
    #
    # 반드시 보고한다:
    #   이 사건의 진짜 문제는 500대1 분할이 헤드라인 숫자를 만들었는데 아무도
    #   몰랐다는 점이다. 그래서 제외가 발생하면 조용히 넘어가지 않고 출력한다.
    # ------------------------------------------------------------------
    degenerate = summary[summary[["stable_n", "other_n"]].min(axis=1) < min_group_size]
    if len(degenerate):
        if verbose:
            print(
                f"[경고] 한쪽 그룹이 {min_group_size}종목 미만이라 제외한 스냅샷 "
                f"{len(degenerate)}건 / 전체 {len(summary)}건 "
                "— 1~2종목짜리 그룹은 포트폴리오로 볼 수 없어 비교에서 뺍니다."
            )
            print(
                degenerate[["stable_n", "other_n", "stable_mean_return", "other_mean_return"]]
                .to_string()
            )
        summary = summary.drop(index=degenerate.index)

    summary["stable_minus_other"] = summary["stable_mean_return"] - summary["other_mean_return"]
    return summary[
        ["stable_mean_return", "other_mean_return", "stable_minus_other", "stable_n", "other_n"]
    ]


def overall_stats(summary: pd.DataFrame) -> pd.Series:
    """Aggregate the per-date comparison into headline strategy stats."""
    diff = summary["stable_minus_other"].dropna()
    return pd.Series(
        {
            "n_snapshots": len(diff),
            "stable_mean_return": summary["stable_mean_return"].mean(),
            "other_mean_return": summary["other_mean_return"].mean(),
            "avg_stable_minus_other": diff.mean(),
            "win_rate": (diff > 0).mean(),
        }
    )


def _portfolio_return_series(
    df: pd.DataFrame,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> dict[str, pd.Series]:
    """Per-date equal-weight portfolio return series for each group (2026-09-02
    fix): the mean forward return across every ticker held in that group on
    that date, i.e. exactly what an investor equal-weighting the group would
    have earned at each rebalancing snapshot.

    Used as the basis for risk stats instead of pooling raw (ticker, snapshot)
    forward returns. Pooling treats every ticker-date observation as an
    independent draw, so its std measures each stock's own (idiosyncratic)
    volatility. A real equal-weight portfolio's risk is lower than that,
    because on any given date the stocks' moves partly cancel out
    (diversification) -- std of the pooled observations overstates the
    portfolio's actual risk and understates its Sharpe ratio.
    """
    summary = summarize_by_group(df, min_group_size=min_group_size)
    return {
        "stable_cluster": summary["stable_mean_return"].dropna(),
        "other_clusters": summary["other_mean_return"].dropna(),
    }


def risk_adjusted_stats(
    df: pd.DataFrame,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Per-date equal-weight portfolio return per group, compared by
    return per unit of risk (mean / std) -- not just raw mean return.

    A low-volatility strategy is not supposed to win on raw return — the
    thesis is a smoother ride (lower std) for less return given up than
    that. This is what actually tests that thesis. std is computed on the
    date-level portfolio return series (see _portfolio_return_series), not
    pooled individual-stock returns, so it reflects diversified portfolio
    risk rather than idiosyncratic stock-level volatility.
    """
    series = _portfolio_return_series(df, min_group_size=min_group_size)
    stats = pd.DataFrame(
        {label: {"n": s.count(), "mean": s.mean(), "std": s.std()} for label, s in series.items()}
    ).T
    stats["return_per_risk"] = stats["mean"] / stats["std"]
    return stats


def sharpe_ratio_stats(
    df: pd.DataFrame,
    window: int,
    annual_risk_free_rate: float = 0.02,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Annualized Sharpe ratio per group, computed from each group's
    date-level equal-weight portfolio return series (see
    _portfolio_return_series) rather than pooled (ticker, snapshot) returns
    -- pooling would measure idiosyncratic stock-level volatility instead of
    the diversified portfolio risk an equal-weight holder actually bears.

    This project doesn't collect an actual risk-free rate time series (no
    T-bill data source), so a constant annual rate is assumed and compounded
    down to the holding period length (``window`` trading days) before being
    subtracted from each observation's return. Treat the absolute Sharpe
    values as approximate — the stable-vs-other *comparison* is what matters,
    and that comparison is not sensitive to the exact risk-free assumption
    (subtracting the same constant from both groups barely moves their gap).
    """
    series = _portfolio_return_series(df, min_group_size=min_group_size)
    period_risk_free = (1 + annual_risk_free_rate) ** (window / TRADING_DAYS_PER_YEAR) - 1
    periods_per_year = TRADING_DAYS_PER_YEAR / window

    rows = {}
    for label, s in series.items():
        mean = s.mean()
        std = s.std()
        rows[label] = {
            "n": s.count(),
            "mean": mean,
            "std": std,
            "excess_mean": mean - period_risk_free,
            "sharpe_annualized": (mean - period_risk_free) / std * np.sqrt(periods_per_year),
        }
    return pd.DataFrame(rows).T


DEFAULT_N_BOOT = 10000


def significance_stats(
    df: pd.DataFrame,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 42,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> pd.Series:
    """Test whether the stable cluster's edge is statistically distinguishable
    from zero, rather than only reporting that the point estimate is positive.

    Added 2026-09-07. Everything above answers "which group scored higher";
    nothing answered "could this gap have come from chance at this sample
    size" -- and the samples are small (n = 85 / 42 / 20 / 9 by window), so
    that question is not decoration.

    Two different claims get two different tests, because they are not the
    same claim and they do not point the same way here:

    1. ``mean_gap`` -- the per-date difference in raw mean return
       (stable minus other). A low-volatility strategy is NOT supposed to win
       this one: the thesis is a smoother ride bought by giving up some
       return. Tested with a one-sample t-test against zero, plus a
       bootstrap CI since period returns are fat-tailed and n is small.
    2. ``rpr_diff`` -- the difference in return per unit of risk
       (mean / std), which IS the project's actual claim. It's a ratio of
       two statistics, so there is no closed-form t-test; the bootstrap
       resamples DATES (keeping both groups paired on the same dates, since
       they share a market) and recomputes the whole ratio each draw.

    Why "no closed-form t-test" is true, and not just a convenient excuse
    (added 2026-09-08, from a presentation Q&A -- see 유의성검정_발표자료.pdf):

    - For a SINGLE group, rpr actually does reduce to something with a known
      distribution: ``rpr * sqrt(n) = mean / (std / sqrt(n))`` is exactly the
      one-sample t statistic testing whether that group's mean is zero. So
      "is rpr_stable different from 0" has a textbook answer under normality.
    - What this function asks instead is the DIFFERENCE of two such ratios,
      ``rpr_stable - rpr_other``, computed from two samples that are (a)
      paired on the same dates (they share a market), (b) fat-tailed, not
      normal, and (c) tiny (n = 9 to 85). Its variance is
      ``Var(A) + Var(B) - 2*Cov(A, B)``, and every term in that expression is
      itself only an asymptotic approximation:
        * Var(a ratio of mean/std) needs the delta method, which needs large
          n and a reliable kurtosis estimate -- neither holds here.
        * Cov(rpr_stable, rpr_other) has a named textbook test (Jobson-Korkie
          1981, corrected by Memmel 2003, for comparing two correlated Sharpe
          ratios) -- but it too assumes joint normality and large-sample
          asymptotics. Plugging n=9 fat-tailed data into it would produce a
          confidently narrow, wrong confidence interval, not a missing one.
    - That is what "no closed-form test" means precisely: the formulas exist
      in the literature, but they need assumptions this sample cannot meet.
      The bootstrap sidesteps all three assumptions at once -- it never
      needs normality, kurtosis, or a covariance formula, because it just
      recomputes ``rpr_stable - rpr_other`` on resampled data and reads the
      spread off directly. Its failure mode is also more honest: when n is
      too small (see the 252-day row, n=9) the CI balloons visibly instead
      of a formula silently reporting false precision.

    Reading the result: a CI that straddles zero means this data cannot
    distinguish the observed edge from chance -- it does NOT mean the edge is
    absent. With n <= 85 the study has little power to detect a modest
    effect, so "not significant" here is a statement about the sample size as
    much as about the strategy.

    Multiple testing: four rebalancing windows get tested, but they are NOT
    four independent experiments -- they resample the same underlying price
    history at different intervals, so a Bonferroni-style correction would be
    too harsh and treating them as independent confirmation would be too
    generous. Report them as one family and say so.
    """
    summary = summarize_by_group(df, min_group_size=min_group_size, verbose=False)
    paired = summary[["stable_mean_return", "other_mean_return"]].dropna()
    stable = paired["stable_mean_return"].to_numpy()
    other = paired["other_mean_return"].to_numpy()
    gap = stable - other
    n = len(gap)

    if n < 3:
        return pd.Series({"n": n, "note": "표본이 3개 미만이라 검정 불가"})

    def _rpr(x: np.ndarray) -> float:
        sd = x.std(ddof=1)
        return float("nan") if sd == 0 else x.mean() / sd

    # 날짜 단위로 대응 재표집한다 -- 두 그룹은 같은 시장을 공유하므로
    # 각각 따로 뽑으면 상관을 끊어버려 차이의 분산을 과대평가한다.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_gap = gap[idx].mean(axis=1)
    boot_rpr = np.array([_rpr(stable[i]) - _rpr(other[i]) for i in idx])
    boot_rpr = boot_rpr[np.isfinite(boot_rpr)]

    def _two_sided_p(draws: np.ndarray) -> float:
        return float(2 * min((draws <= 0).mean(), (draws >= 0).mean()))

    mean_gap = float(gap.mean())
    se = gap.std(ddof=1) / np.sqrt(n)
    t_stat = mean_gap / se if se > 0 else float("nan")

    rpr_stable, rpr_other = _rpr(stable), _rpr(other)
    rpr_lo, rpr_hi = np.percentile(boot_rpr, [2.5, 97.5])
    gap_lo, gap_hi = np.percentile(boot_gap, [2.5, 97.5])

    return pd.Series({
        "n": n,
        "mean_gap": mean_gap,
        "mean_gap_t": t_stat,
        "mean_gap_ci_low": gap_lo,
        "mean_gap_ci_high": gap_hi,
        "mean_gap_p_boot": _two_sided_p(boot_gap),
        "rpr_stable": rpr_stable,
        "rpr_other": rpr_other,
        "rpr_diff": rpr_stable - rpr_other,
        "rpr_ci_low": rpr_lo,
        "rpr_ci_high": rpr_hi,
        "rpr_p_boot": _two_sided_p(boot_rpr),
        "rpr_significant": bool(rpr_lo > 0),
        "win_rate": float((gap > 0).mean()),
    })
