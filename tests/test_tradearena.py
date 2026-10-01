import numpy as np
import pandas as pd
from tradearena.backtest import run, stats, sharpe_diff_ci
from tradearena.features import build, target
from tradearena.strategies import MODELS, walk_forward


def prices(n=2600, seed=0, drift=0.0003):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))), index=pd.bdate_range("2010-01-01", periods=n))


def test_buy_and_hold_with_no_cost_equals_asset_return():
    c = prices()
    net, pos, to = run(c, pd.Series(1.0, index=c.index), c.index[300], 0.0)
    assert np.isclose((1 + net).prod(), c.iloc[-1] / c.iloc[300], rtol=1e-6)


def test_costs_charged_on_turnover_only():
    c = prices()
    flip = pd.Series(np.tile([1.0, 0.0], len(c) // 2 + 1)[: len(c)], index=c.index)
    gross, _, to = run(c, flip, c.index[300], 0.0)
    net, _, _ = run(c, flip, c.index[300], 0.001)
    assert np.isclose((gross - net).sum(), to.sum() * 0.001)
    hold, _, to1 = run(c, pd.Series(1.0, index=c.index), c.index[300], 0.001)
    assert to1.sum() == 1.0  # one entry trade, nothing after


def test_position_uses_next_day_return_not_same_day():
    c = pd.Series([100.0, 110.0, 99.0, 99.0], index=pd.bdate_range("2020-01-01", periods=4))
    pos = pd.Series([1.0, 0.0, 0.0, 0.0], index=c.index)
    net, _, _ = run(c, pos, c.index[0], 0.0)
    assert np.isclose(net.iloc[0], 0.10)  # long at day-0 close earns day-1 return (+10%)


def test_features_are_causal():
    c = prices()
    X1 = build(c)
    c2 = c.copy()
    c2.iloc[2000:] *= 3  # change the future
    X2 = build(c2)
    assert X1.iloc[:2000].equals(X2.iloc[:2000])


def test_walk_forward_positions_do_not_depend_on_future_data():
    c = prices()
    p1, _, _ = walk_forward(c, MODELS["logistic regression"])
    c2 = c.copy()
    cut = c.index[c.index.year < 2019][-1]
    c2.loc[c2.index > cut] *= np.exp(np.random.default_rng(1).normal(0, 0.05, (c2.index > cut).sum()).cumsum())
    p2, _, _ = walk_forward(c2, MODELS["logistic regression"])
    same = p1.loc[:cut].fillna(-1).equals(p2.loc[:cut].fillna(-1))
    assert same, "positions before the cut changed when only later prices changed (look-ahead leak)"


def test_ml_has_no_edge_on_a_random_walk():
    """On an i.i.d. random walk direction is unpredictable, so hit rate must be ~ the base rate, not better."""
    c = prices(seed=3, drift=0.0)
    _, p, first = walk_forward(c, MODELS["gradient boosting"])
    actual = (c.pct_change().shift(-1) > 0)
    m = p.dropna().index.intersection(c.pct_change().shift(-1).dropna().index)
    assert abs(((p[m] > 0.5) == actual[m]).mean() - 0.5) < 0.04


def test_bootstrap_ci_contains_zero_for_identical_returns():
    r = pd.Series(np.random.default_rng(0).normal(0.0004, 0.01, 1500))
    d, ci = sharpe_diff_ci(r, r.copy(), 252, n=200)
    assert d == 0 and ci[0] <= 0 <= ci[1]


def test_stats_max_drawdown_known_case():
    net = pd.Series([0.1, -0.5, 0.0, 0.2])
    s = stats(net, pd.Series(1.0, index=net.index), pd.Series(0.0, index=net.index), 252)
    assert np.isclose(s["max_drawdown"], -0.5)


def test_target_last_row_is_unknown():
    assert np.isnan(target(prices()).iloc[-1])
