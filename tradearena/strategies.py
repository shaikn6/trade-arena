"""Each strategy maps a price history to a daily long/flat position decided at the close (applied next day)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tradearena.features import build, target


def buy_hold(close):
    return pd.Series(1.0, index=close.index)


def sma_cross(close, fast=50, slow=200):
    return (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float)


def momentum(close):
    return (build(close)["mom_12_1"] > 0).astype(float)


MODELS = {
    "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1)),
    "gradient boosting": lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=3, min_samples_leaf=50, random_state=0),
    "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier((32, 16), alpha=1e-2, early_stopping=True, max_iter=200, random_state=0)),
}
MIN_TRAIN = 1000


def walk_forward(close, make_model, min_train=MIN_TRAIN, target_fn=target):
    """Retrain every calendar year on data strictly before that year; predict that year's days.

    Returns (position, p_up, first_test_date). Rows whose target peeks past the train cut are excluded (purge).
    """
    X, y = build(close), target_fn(close)
    data = X.assign(y=y).dropna(subset=list(X.columns))
    p = pd.Series(np.nan, index=close.index)
    first = None
    for year in sorted(set(close.index.year)):
        test_idx = X.index[X.index.year == year].intersection(data.index)
        train = data[data.index.year < year].dropna(subset=["y"])
        train = train.iloc[:-1] if len(train) else train  # purge: last train target uses the first test day's return
        if len(train) < min_train or len(test_idx) == 0 or train["y"].nunique() < 2:
            continue
        first = first or test_idx[0]
        m = make_model().fit(train[X.columns], train["y"])
        p.loc[test_idx] = m.predict_proba(data.loc[test_idx, X.columns])[:, 1]
    pos = (p > 0.5).astype(float).where(p.notna())
    return pos, p, first
