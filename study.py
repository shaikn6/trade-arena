"""Paper study on top of run.py's walk-forward setup:
1. break-even cost: the per-trade cost at which each strategy's net Sharpe falls to buy-and-hold's;
2. a no-trade band (enter above p_hi, exit below 1 - p_hi) to cut turnover;
3. a matched random-timing control: random long/flat paths with the same exposure and turnover as the model.
Writes study.json (aggregates only)."""
import json

import numpy as np
import pandas as pd

from tradearena.backtest import COST, run, sharpe_diff_ci, stats
from tradearena.strategies import MODELS, buy_hold, momentum, sma_cross, walk_forward

COSTS_BPS = (0, 2, 5, 10, 15, 20, 30)
BANDS = (0.50, 0.52, 0.55, 0.60)
N_RANDOM = 200


def banded(p, hi):
    """Long when p > hi, flat when p < 1 - hi, otherwise keep yesterday's position."""
    pos, cur = [], 0.0
    for v in p.values:
        if np.isnan(v):
            pos.append(np.nan)
            continue
        cur = 1.0 if v > hi else 0.0 if v < 1 - hi else cur
        pos.append(cur)
    return pd.Series(pos, index=p.index)


def random_matched(pos, rng):
    """Two-state Markov long/flat path with the same time in market and switch rate as `pos`."""
    x = pos.dropna().values
    e, sw = x.mean(), np.abs(np.diff(x)).mean()
    p_on = min(1.0, sw / (2 * max(1 - e, 1e-9)))  # flat -> long
    p_off = min(1.0, sw / (2 * max(e, 1e-9)))     # long -> flat
    out, cur = np.empty(len(x)), float(rng.random() < e)
    for i in range(len(x)):
        out[i] = cur
        cur = (1.0 - cur) if rng.random() < (p_off if cur else p_on) else cur
    return pd.Series(out, index=pos.dropna().index)


def sharpe(net, ann):
    return float(net.mean() / net.std() * np.sqrt(ann)) if net.std() > 0 else 0.0


def break_even(close, pos, start, ann, bh_sharpe):
    """Largest cost on the grid at which the strategy's net Sharpe still matches buy-and-hold (None if never)."""
    ok = [c for c in COSTS_BPS if sharpe(run(close, pos, start, c / 1e4)[0], ann) >= bh_sharpe]
    return max(ok) if ok else None


if __name__ == "__main__":
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
    rng = np.random.default_rng(0)
    out = {}
    for asset in prices.columns:
        close = prices[asset].dropna()
        kind = "crypto" if "USD" in asset else "equity"
        ann, cost = (365 if kind == "crypto" else 252), COST[kind]
        probs, starts = {}, []
        for name, mk in MODELS.items():
            _, p, first = walk_forward(close, mk)
            probs[name] = p
            starts.append(first)
        start = max(starts)
        bh = buy_hold(close)
        bh_net = run(close, bh, start, cost)[0]
        bh_s = sharpe(bh_net, ann)
        a = {"buy_hold_sharpe": round(bh_s, 3), "kind": kind, "strategies": {}}
        rules = {"SMA 50/200 crossover": sma_cross(close), "12-1 momentum": momentum(close)}
        for name, pos in rules.items():
            a["strategies"][name] = {"sharpe_by_cost": {c: round(sharpe(run(close, pos, start, c / 1e4)[0], ann), 3) for c in COSTS_BPS},
                                     "break_even_bps": break_even(close, pos, start, ann, bh_s)}
        for name, p in probs.items():
            for hi in BANDS:
                pos = banded(p, hi)
                net, pp, to = run(close, pos, start, cost)
                st = stats(net, pp, to, ann)
                rand = [sharpe(run(close, random_matched(pos.loc[start:], rng), start, cost)[0], ann) for _ in range(N_RANDOM)]
                a["strategies"][f"{name} | band {hi}"] = {
                    "sharpe_net": round(st["sharpe"], 3), "turnover_per_year": round(st["turnover_per_year"], 1),
                    "time_in_market": round(st["time_in_market"], 3),
                    "sharpe_by_cost": {c: round(sharpe(run(close, pos, start, c / 1e4)[0], ann), 3) for c in COSTS_BPS},
                    "break_even_bps": break_even(close, pos, start, ann, bh_s),
                    "random_matched_pctile": round(float((np.array(rand) < st["sharpe"]).mean()), 3),
                    "random_matched_median_sharpe": round(float(np.median(rand)), 3),
                    "sharpe_minus_bh_ci95": [round(c, 3) for c in sharpe_diff_ci(net, bh_net, ann)[1]]}
        out[asset] = a
        print(asset, flush=True)
    json.dump({"costs_bps": COSTS_BPS, "bands": BANDS, "n_random": N_RANDOM, "assets": out}, open("study.json", "w"), indent=2)
