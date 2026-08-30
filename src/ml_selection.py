"""Supervised stock selection: predict which stocks will outperform their
same-date peers over the next rebalancing period, using an ensemble of
independently trained classifiers (Random Forest, Logistic Regression,
XGBoost). A stock is "recommended" when all three models agree.

The label is deliberately cross-sectional (relative to that date's median
forward return), not the raw forward return itself. notebooks 06/07 showed
raw forward return is dominated by market direction (correlation ~-0.5 to
-0.7 with the S&P500's own return over the same window) — a model trained
on raw return would mostly learn to predict the market, not pick stocks.
Comparing each stock only to its same-date peers removes that regime
confound and keeps the label about relative stock-picking skill.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURE_METRICS = ("beta", "volatility", "return", "rsi")
RANDOM_STATE = 42

# 6-팩터 구조 (Risk/Momentum/Growth/Valuation/Quality/Size) — 11번 노트북에서
# beta/volatility/return/rsi(가격) + PER/PBR/ROE/debt_ratio/revenue_growth/
# log_market_cap(재무)를 하나로 합친 뒤, 개별 피처 대신 "어떤 투자 요인 자체가
# 기여하는가"를 보려고 만든 매핑. MDD는 의도적으로 뺐다 — 04번에서 다중공선성
# 때문에 제거한 지표를 팩터 구조 맞추자고 다시 넣으면 그 결정과 모순되고,
# 애초에 MDD는 가격 경로에서 파생된 값이라 이미 있는 volatility/return과
# 정보가 겹친다. MDD는 대신 backtest.py의 전략 성과 평가(max_drawdown)에서만
# 쓴다 — "이 종목을 고를까"(feature)와 "고른 포트폴리오가 실제로 얼마나
# 위험했나"(평가지표)는 서로 다른 질문이기 때문이다.
FACTOR_GROUPS = {
    "Risk": ("beta", "volatility", "debt_ratio"),
    "Momentum": ("return", "rsi"),
    "Growth": ("revenue_growth",),
    "Valuation": ("per", "pbr"),
    "Quality": ("roe",),
    "Size": ("log_market_cap",),
}


def metric_to_factor(column: str) -> str:
    """Map a feature column (e.g. ``beta_30d`` or ``per``) to its factor
    group. Strips a trailing ``_Nd`` window suffix if present, since price
    features are windowed but fundamental ratios aren't."""
    base = column
    parts = column.rsplit("_", 1)
    if len(parts) == 2 and parts[1].endswith("d") and parts[1][:-1].isdigit():
        base = parts[0]
    for factor, metrics in FACTOR_GROUPS.items():
        if base in metrics:
            return factor
    raise ValueError(f"'{column}'이 어느 팩터에도 안 걸립니다 — FACTOR_GROUPS를 확인하세요.")


def add_relative_label(df: pd.DataFrame, forward_return_col: str = "forward_return") -> pd.DataFrame:
    """Binary label: 1 if forward_return beats that date's median, else 0."""
    df = df.copy()
    median_by_date = df.groupby("Date")[forward_return_col].transform("median")
    df["outperform_peers"] = (df[forward_return_col] > median_by_date).astype(int)
    return df


def time_split(dates: list, train_ratio: float = 0.7) -> tuple[list, list]:
    """Split dates chronologically (never randomly) to avoid leaking future
    snapshots into training."""
    dates = sorted(dates)
    cutoff = int(len(dates) * train_ratio)
    return dates[:cutoff], dates[cutoff:]


def walk_forward_splits(dates: list, min_train_years: int = 5) -> list[tuple[list, list]]:
    """Expanding-window walk-forward splits, one fold per calendar year.

    Fold for test year Y trains on every date strictly before year Y and
    tests on year Y alone — "train 2016-2020 -> test 2021, train
    2016-2021 -> test 2022, ..." A single 70/30 split only tells you
    whether the model generalizes to *one* held-out period; this checks
    whether it generalizes to *every* year, which is what actually matters
    for a strategy meant to run indefinitely.

    The first ``min_train_years`` calendar years are never used as a test
    fold (only as training history) — otherwise the earliest folds would
    train on too little data to mean anything.
    """
    dates = sorted(dates)
    years = sorted({d.year for d in dates})

    splits = []
    for test_year in years[min_train_years:]:
        train_dates = [d for d in dates if d.year < test_year]
        test_dates = [d for d in dates if d.year == test_year]
        if train_dates and test_dates:
            splits.append((train_dates, test_dates))
    return splits


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Fit all three models. Logistic regression gets standardized inputs
    (it's distance/coefficient-scale sensitive); the tree models don't."""
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, random_state=RANDOM_STATE
        ),
        "logistic": LogisticRegression(max_iter=1000),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE, eval_metric="logloss",
        ),
    }
    models["random_forest"].fit(X_train, y_train)
    models["logistic"].fit(X_train_scaled, y_train)
    models["xgboost"].fit(X_train, y_train)
    return {"models": models, "scaler": scaler}


def predict_all(fitted: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Each model's binary prediction, plus a ``consensus`` column that's
    True only when every model predicts 1 (outperform)."""
    models = fitted["models"]
    X_scaled = fitted["scaler"].transform(X)

    preds = {
        "random_forest": models["random_forest"].predict(X),
        "logistic": models["logistic"].predict(X_scaled),
        "xgboost": models["xgboost"].predict(X),
    }
    pred_df = pd.DataFrame(preds, index=X.index)
    pred_df["consensus"] = pred_df.all(axis=1)
    return pred_df
