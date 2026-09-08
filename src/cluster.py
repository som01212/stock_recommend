"""Hierarchical clustering of rebalancing snapshots into stability tiers.

For every rebalancing date independently (cross-sectional — a ticker's
cluster on date T only ever depends on other tickers' values on that same
date T, never on other dates, so this cannot leak future information):

1. The 4 features are standardized (z-score) within that date's
   cross-section only.
2. The stability features (beta, volatility) are multiplied by
   ``stability_weight`` per the strategy's "stability first, profitability
   second" priority, before clustering.
3. Ward hierarchical clustering (``sklearn.cluster.AgglomerativeClustering``)
   groups tickers into ``n_clusters`` clusters using the weighted, scaled
   features.
4. The cluster with the lowest average (unweighted) stability z-score is
   labeled the "stable" cluster (``is_stable_cluster``).

Label churn between consecutive snapshots (measured 2026-09-08, ad hoc in a
Claude Code session -- not yet a notebook cell or test, so treat the exact
numbers as a one-off check rather than a tracked metric): comparing
``is_stable_cluster`` for the same ticker across consecutive rebalancing
dates in ``clustered_60df.parquet`` gives a ~25.7% flip rate (a ticker
labeled "stable" this period has roughly a 1-in-4 chance of being labeled
"other" next period, or vice versa). That is well below what full random
reassignment would produce (~42-47%, estimated as 2*p*(1-p) from the average
stable-cluster share), so the label is not noise -- but over a ticker's full
multi-year history, 79-88% of tickers flip label at least once; only ~12-21%
stay in one cluster for their entire run.

This is not a bug and does not need fixing -- it is the expected consequence
of classifying by RELATIVE rank (see step 1: z-scored within that date's
cross-section only) rather than an absolute threshold. A ticker's own beta
or volatility can stay nearly unchanged while its z-score, and therefore its
cluster, moves simply because the rest of that day's universe shifted around
it. The raw metrics themselves drift less than this (e.g. beta_60d's
period-to-period drift ratio is ~0.84, see notebooks/07_robustness.ipynb) --
so persistence at the raw-metric level does not carry over 1:1 to
persistence at the label level, and the two should not be conflated.

Practical reading: "is_stable_cluster" does not identify a fixed watchlist
of permanently-safe blue chips. It is closer to "which tickers rank as
relatively less risky as of THIS rebalancing date" than to "which companies
are safe" -- a claim about the strategy's re-selection rule at each
rebalancing, not about any stock's fixed identity. This does not invalidate
downstream significance testing (each period's groups are reconstituted
honestly, not assumed to persist), but it does mean the "stable cluster"
should be described as a rules-based re-sort, not a durable set of safe
companies.
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import AgglomerativeClustering

DEFAULT_STABILITY_METRICS = ("beta", "volatility")
DEFAULT_PROFITABILITY_METRICS = ("return", "rsi")
DEFAULT_STABILITY_WEIGHT = 2.0
DEFAULT_N_CLUSTERS = 2


def cluster_snapshot(
    group: pd.DataFrame,
    window: int,
    stability_metrics: tuple[str, ...] = DEFAULT_STABILITY_METRICS,
    profitability_metrics: tuple[str, ...] = DEFAULT_PROFITABILITY_METRICS,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    stability_weight: float = DEFAULT_STABILITY_WEIGHT,
) -> pd.DataFrame:
    """Cluster a single rebalancing date's cross-section of tickers.

    ``group`` must already be filtered to one Date (use
    ``cluster_all_snapshots`` to apply this across every date). Rows with a
    missing feature value are dropped (too little history yet) and returned
    with ``cluster = -1``.
    """
    stability_cols = [f"{m}_{window}d" for m in stability_metrics]
    profitability_cols = [f"{m}_{window}d" for m in profitability_metrics]
    feature_cols = stability_cols + profitability_cols

    group = group.copy()
    valid = group.dropna(subset=feature_cols)
    if len(valid) < n_clusters:
        group["cluster"] = -1
        group["is_stable_cluster"] = False
        return group

    # 그 날짜의 종목 단면(cross-section)끼리만 비교하는 z-score.
    # 다른 날짜 데이터는 전혀 안 쓰므로 lookahead bias가 없다.
    z = (valid[feature_cols] - valid[feature_cols].mean()) / valid[feature_cols].std(ddof=0)
    z = z.fillna(0.0)  # 그 날짜에 표준편차가 0인(전종목 동일값) 경우 방어

    weights = pd.Series(1.0, index=feature_cols)
    weights[stability_cols] = stability_weight
    weighted = z * weights

    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(
        weighted.to_numpy()
    )

    # "안정적" 클러스터 = beta/volatility 표준화 점수(가중치 적용 전) 평균이
    # 가장 낮은 클러스터. 가중치는 군집 경계를 정하는 데만 쓰고, 라벨링은
    # 원래 정의(낮을수록 안정적)로 판단한다.
    stability_score = z[stability_cols].mean(axis=1)
    cluster_stability = stability_score.groupby(labels).mean()
    stable_cluster_id = cluster_stability.idxmin()

    group["cluster"] = -1
    group["is_stable_cluster"] = False
    group.loc[valid.index, "cluster"] = labels
    group.loc[valid.index, "is_stable_cluster"] = labels == stable_cluster_id
    group["is_stable_cluster"] = group["is_stable_cluster"].astype(bool)
    return group


def cluster_all_snapshots(
    rebalance_df: pd.DataFrame,
    window: int,
    **kwargs,
) -> pd.DataFrame:
    """Apply ``cluster_snapshot`` independently to every Date in rebalance_df."""
    results = [
        cluster_snapshot(group, window=window, **kwargs)
        for _, group in rebalance_df.groupby("Date", sort=False)
    ]
    return pd.concat(results, ignore_index=True)
