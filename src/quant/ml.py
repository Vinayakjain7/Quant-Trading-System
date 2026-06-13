"""Optional ML ranking layer.

The rule-based strategy decides WHEN a name is a valid candidate. When more
candidates appear than you have open slots, this model helps decide WHICH ones
to take, by estimating the probability that the next `horizon_days` return is
positive.

Leakage guards (this is where most ML-for-trading projects quietly cheat):
  * Features at row t use only information available at the close of t.
  * The label is a FORWARD return; the final `horizon` rows (whose label peeks
    into data we don't have yet) are dropped from training.
  * Train/test split is TIME-ORDERED (no shuffling) — the model is always
    tested on data chronologically after what it trained on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import add_indicators

FEATURES = ["bb_pct", "rsi", "vol_ratio", "ret_1d", "ret_5d", "dist_trend", "atr_pct"]


def _build_features(df: pd.DataFrame, strat_cfg: dict) -> pd.DataFrame:
    data = add_indicators(df, strat_cfg)
    close = data["Close"]
    data["ret_1d"] = close.pct_change(1)
    data["ret_5d"] = close.pct_change(5)
    data["dist_trend"] = (close - data["trend_ma"]) / data["trend_ma"]
    data["atr_pct"] = data["atr"] / close
    return data


def build_dataset(
    price_data: dict[str, pd.DataFrame], strat_cfg: dict, horizon: int
) -> pd.DataFrame:
    """Stack a supervised dataset across the universe.

    label = 1 if the forward `horizon`-day return is positive, else 0.
    """
    frames = []
    for ticker, df in price_data.items():
        feat = _build_features(df, strat_cfg)
        fwd = feat["Close"].shift(-horizon) / feat["Close"] - 1.0
        feat = feat.assign(ticker=ticker, label=(fwd > 0).astype(int), fwd_ret=fwd)
        # Drop the last `horizon` rows: their label looks into the future.
        feat = feat.iloc[:-horizon] if horizon > 0 else feat
        frames.append(feat)
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames).dropna(subset=FEATURES + ["label"])
    return full


def _make_model(name: str):
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    # default
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(random_state=42)


@dataclass
class MLResult:
    model: object
    test_auc: float
    test_accuracy: float
    n_train: int
    n_test: int
    feature_importance: dict


def train_ranker(price_data: dict[str, pd.DataFrame], config) -> MLResult | None:
    """Train the ranking model with a time-ordered holdout. Returns None if
    ML is disabled or there isn't enough data."""
    ml_cfg = config.section("ml")
    if not ml_cfg.get("enabled", False):
        return None

    strat_cfg = config.section("strategy")
    horizon = int(ml_cfg.get("horizon_days", 5))
    ds = build_dataset(price_data, strat_cfg, horizon)
    if len(ds) < int(ml_cfg.get("min_train_rows", 500)):
        return None

    ds = ds.sort_index()  # chronological
    test_size = float(ml_cfg.get("test_size", 0.25))
    split = int(len(ds) * (1 - test_size))
    train, test = ds.iloc[:split], ds.iloc[split:]

    X_train, y_train = train[FEATURES], train["label"]
    X_test, y_test = test[FEATURES], test["label"]

    model = _make_model(ml_cfg.get("model", "gradient_boosting"))
    model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score, roc_auc_score
    proba = _proba(model, X_test)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")
    acc = accuracy_score(y_test, (proba > 0.5).astype(int))

    importance = _importance(model)
    return MLResult(
        model=model,
        test_auc=round(float(auc), 3),
        test_accuracy=round(float(acc), 3),
        n_train=len(train),
        n_test=len(test),
        feature_importance=importance,
    )


def _proba(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


def _importance(model) -> dict:
    est = model
    if hasattr(model, "steps"):  # pipeline
        est = model.steps[-1][1]
    if hasattr(est, "feature_importances_"):
        return {f: round(float(v), 3) for f, v in zip(FEATURES, est.feature_importances_)}
    if hasattr(est, "coef_"):
        return {f: round(float(v), 3) for f, v in zip(FEATURES, np.ravel(est.coef_))}
    return {}


def score_latest(model, df: pd.DataFrame, strat_cfg: dict) -> float:
    """Probability that the latest bar leads to a positive forward return."""
    feat = _build_features(df, strat_cfg).dropna(subset=FEATURES)
    if feat.empty:
        return float("nan")
    x = feat[FEATURES].iloc[[-1]]
    return float(_proba(model, x)[0])
