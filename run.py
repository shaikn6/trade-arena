"""Walk-forward model arena across stocks, ETFs and crypto. Net of costs, against buy-and-hold."""
import json
import pandas as pd
from tradearena.features import shuffled_target, target
from tradearena.strategies import MODELS, buy_hold, momentum, sma_cross, walk_forward
from tradearena.backtest import COST, run, sharpe_diff_ci, stats

prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
results, hit = {}, {}
for asset in prices.columns:
    close = prices[asset].dropna()
    kind = "crypto" if "USD" in asset else "equity"
    ann = 365 if kind == "crypto" else 252
    strategies, starts = {"buy & hold": buy_hold(close), "SMA 50/200 crossover": sma_cross(close), "12-1 momentum": momentum(close)}, []
    p_up = {}
    for name, mk in MODELS.items():
        pos, p, first = walk_forward(close, mk)
        strategies[name], p_up[name] = pos, p
        starts.append(first)
    # control: same gradient-boosting pipeline trained on shuffled labels (should have no edge -> leak check)
    pos, _, _ = walk_forward(close, MODELS["gradient boosting"], target_fn=shuffled_target(0))
    strategies["CONTROL: shuffled labels"] = pos
    start = max(starts)
    nets = {}
    for name, pos in strategies.items():
        net, pp, to = run(close, pos, start, COST[kind])
        nets[name] = net
        results.setdefault(asset, {})[name] = stats(net, pp, to, ann)
        results[asset][name]["sharpe_gross"] = stats(run(close, pos, start, 0.0)[0], pp, to, ann)["sharpe"]
    for name in strategies:
        if name != "buy & hold":
            d, ci = sharpe_diff_ci(nets[name], nets["buy & hold"], ann)
            results[asset][name]["sharpe_minus_bh"], results[asset][name]["sharpe_minus_bh_ci95"] = round(d, 3), [round(c, 3) for c in ci]
    actual = target(close)  # NaN on the last day, whose next-day return is unknown
    for name, p in p_up.items():
        m = p.loc[start:].dropna().index.intersection(actual.dropna().index)
        hit.setdefault(asset, {})[name] = round(float(((p[m] > 0.5) == (actual[m] > 0.5)).mean()), 4)
    hit[asset]["always-up base rate"] = round(float(actual.loc[start:].dropna().mean()), 4)
    hit[asset]["test_start"] = str(start.date())
    print(asset, hit[asset]["test_start"], {k: round(v["sharpe"], 2) for k, v in results[asset].items()}, flush=True)
json.dump(dict(per_asset=results, hit_rate=hit, cost=COST), open("results.json", "w"), indent=2)
