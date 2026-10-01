"""Price-only features known at the close of day t, and the next-day direction target."""
import numpy as np
import pandas as pd


def rsi(close, n=14):
    d = close.diff()
    up, dn = d.clip(lower=0).rolling(n).mean(), (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build(close: pd.Series) -> pd.DataFrame:
    r = close.pct_change()
    X = pd.DataFrame(index=close.index)
    for k in (1, 2, 3, 5, 10, 20, 60):
        X[f"ret_{k}"] = close.pct_change(k)
    X["vol_20"], X["vol_60"] = r.rolling(20).std(), r.rolling(60).std()
    X["rsi_14"] = rsi(close)
    X["dist_sma50"], X["dist_sma200"] = close / close.rolling(50).mean() - 1, close / close.rolling(200).mean() - 1
    X["mom_12_1"] = close.shift(21) / close.shift(252) - 1
    return X


def shuffled_target(seed=0):
    """Control experiment: the true target with its labels randomly permuted (carries no information)."""
    def fn(close: pd.Series) -> pd.Series:
        return target(close).sample(frac=1.0, random_state=seed).set_axis(close.index)
    return fn


def target(close: pd.Series) -> pd.Series:
    """1 if the NEXT day's return is positive. Last row is NaN (unknown)."""
    nxt = close.pct_change().shift(-1)
    return (nxt > 0).astype(float).where(nxt.notna())
