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
