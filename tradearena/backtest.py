"""Net-of-cost backtest: position decided at close t earns day t+1's return; costs on turnover."""
import numpy as np
import pandas as pd

COST = {"equity": 0.0010, "crypto": 0.0020}  # one-way, fraction of traded notional (fees + slippage)


def run(close, position, start, cost):
    pos = position.reindex(close.index).fillna(0.0).loc[start:]
    ret = close.pct_change().shift(-1).loc[start:]  # return earned by today's position
    valid = ret.notna()
    pos, ret = pos[valid], ret[valid]
    turnover = pos.diff().abs().fillna(pos.abs())
    return pos * ret - turnover * cost, pos, turnover


def stats(net, pos, turnover, ann):
    eq = (1 + net).cumprod()
    years = len(net) / ann
    dd = (eq / eq.cummax() - 1).min()
    return dict(cagr=float(eq.iloc[-1] ** (1 / years) - 1), sharpe=float(net.mean() / net.std() * np.sqrt(ann)) if net.std() > 0 else 0.0,
                max_drawdown=float(dd), turnover_per_year=float(turnover.sum() / years), time_in_market=float(pos.mean()),
                final_wealth=float(eq.iloc[-1]))


def sharpe_diff_ci(a, b, ann, block=20, n=2000, seed=0):
    """Stationary-style block bootstrap CI for Sharpe(a) - Sharpe(b) on aligned daily net returns."""
    d = pd.concat([a, b], axis=1).dropna().values
    rng = np.random.default_rng(seed)
    T = len(d)
    out = np.empty(n)
    for i in range(n):
        idx = np.concatenate([(rng.integers(0, T) + np.arange(block)) % T for _ in range(T // block + 1)])[:T]
        s = d[idx]
        sa = s[:, 0].mean() / (s[:, 0].std() + 1e-12) * np.sqrt(ann)
        sb = s[:, 1].mean() / (s[:, 1].std() + 1e-12) * np.sqrt(ann)
        out[i] = sa - sb
    point = (d[:, 0].mean() / d[:, 0].std() - d[:, 1].mean() / d[:, 1].std()) * np.sqrt(ann)
    return float(point), [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
