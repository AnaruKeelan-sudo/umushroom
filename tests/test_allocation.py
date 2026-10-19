from __future__ import annotations

import pytest

from umushroom.allocation import generate_allocation
from umushroom.features import build_training_set
from umushroom.model import RankingModel


@pytest.fixture
def trained_model(synthetic_prices):
    X, y = build_training_set(synthetic_prices, forward_return_days=5,
                               step=5, min_history_days=70)
    return RankingModel(alpha=1.0, random_seed=42).fit(X, y)


def test_allocation_never_overspends_balance(synthetic_prices, trained_model):
    balance = 100_000.0
    alloc = generate_allocation(synthetic_prices, trained_model, top_n=4,
                                 balance=balance)
    assert alloc.total_deployed <= balance
    assert alloc.cash_remaining >= 0


def test_allocation_has_requested_number_of_positions(synthetic_prices, trained_model):
    alloc = generate_allocation(synthetic_prices, trained_model, top_n=3,
                                 balance=100_000.0)
    assert len(alloc.orders) == 3


def test_allocation_no_negative_shares(synthetic_prices, trained_model):
    alloc = generate_allocation(synthetic_prices, trained_model, top_n=4,
                                 balance=100_000.0)
    assert (alloc.orders["shares"] >= 0).all()


def test_allocation_no_duplicate_tickers(synthetic_prices, trained_model):
    alloc = generate_allocation(synthetic_prices, trained_model, top_n=4,
                                 balance=100_000.0)
    assert not alloc.orders.index.duplicated().any()


def test_allocation_rejects_zero_balance(synthetic_prices, trained_model):
    with pytest.raises(ValueError):
        generate_allocation(synthetic_prices, trained_model, top_n=4, balance=0)


def test_allocation_rejects_negative_balance(synthetic_prices, trained_model):
    with pytest.raises(ValueError):
        generate_allocation(synthetic_prices, trained_model, top_n=4, balance=-100)


def test_allocation_rejects_zero_top_n(synthetic_prices, trained_model):
    with pytest.raises(ValueError):
        generate_allocation(synthetic_prices, trained_model, top_n=0, balance=100_000)


def test_allocation_scales_with_balance(synthetic_prices, trained_model):
    small = generate_allocation(synthetic_prices, trained_model, top_n=3,
                                 balance=10_000.0)
    large = generate_allocation(synthetic_prices, trained_model, top_n=3,
                                 balance=100_000.0)
    # Same picks (same model/data), but larger balance should deploy more capital.
    assert large.total_deployed > small.total_deployed


def test_allocation_exposes_full_universe_scores(synthetic_prices, trained_model):
    """all_scores must cover the whole clean-feature universe, not just
    the top_n picks -- needed for prediction logging / IC computation."""
    alloc = generate_allocation(synthetic_prices, trained_model, top_n=3,
                                 balance=100_000.0)
    assert len(alloc.all_scores) >= len(alloc.orders)
    assert set(alloc.orders.index).issubset(set(alloc.all_scores.index))


def test_allocation_to_csv_writes_file(tmp_path, synthetic_prices, trained_model):
    alloc = generate_allocation(synthetic_prices, trained_model, top_n=3,
                                 balance=100_000.0)
    out_path = tmp_path / "alloc.csv"
    alloc.to_csv(str(out_path))
    assert out_path.exists()
    content = out_path.read_text()
    assert "shares" in content


# ---------------------------------------------------------------- v4 tests
def test_allocation_reports_listing_currency_and_local_price(synthetic_prices,
                                                             trained_model):
    local = synthetic_prices * 1.25
    ccy = {t: "USD" for t in synthetic_prices.columns}
    alloc = generate_allocation(synthetic_prices, trained_model, top_n=3,
                                balance=50_000, local_prices=local, currencies=ccy)
    assert {"listing_currency", "local_price"} <= set(alloc.orders.columns)
    t = alloc.orders.index[0]
    assert alloc.orders.loc[t, "local_price"] == pytest.approx(
        round(float(local[t].iloc[-1]), 4))
    assert (alloc.orders["shares"] * alloc.orders["price"]).sum() <= 50_000


def test_allocation_hrp_runs_natively(synthetic_prices, trained_model):
    alloc = generate_allocation(synthetic_prices, trained_model, top_n=4,
                                balance=100_000, weighting="hrp")
    assert alloc.orders["target_weight"].sum() == pytest.approx(1.0, abs=1e-3)
    assert alloc.total_deployed <= 100_000


def test_fractional_allocation_buys_fractions_and_never_overspends(
        synthetic_prices, trained_model):
    expensive = synthetic_prices * 1000          # like BTC: one unit > budget
    alloc = generate_allocation(expensive, trained_model, top_n=3, balance=10_000,
                                fractional=True)
    assert (alloc.orders["shares"] > 0).all()
    assert (alloc.orders["shares"] < 1).any()
    assert alloc.total_deployed <= 10_000
    assert alloc.total_deployed > 9_900          # nearly fully invested


def test_allocation_skips_implausibly_low_price(synthetic_prices, trained_model):
    """v4.3.1 regression test for the IGLT.L pence/pound error found on the
    first real run: a 100x-too-low price must never become an order."""
    local = synthetic_prices.copy()
    alloc_all = generate_allocation(synthetic_prices, trained_model, top_n=8,
                                    balance=100_000)
    victim = alloc_all.orders.index[0]
    local[victim] = 0.0952                            # like IGLT.L: 0.0952 instead of 9.48
    alloc = generate_allocation(synthetic_prices, trained_model, top_n=8,
                                balance=100_000, local_prices=local,
                                min_local_price=0.10)
    assert victim not in alloc.orders.index
    assert len(alloc.orders) == 7


# ---------------------------------------------------------------- v4.5
from umushroom.allocation import cap_weights


def test_cap_weights_limits_and_redistributes():
    import pandas as pd
    w = pd.Series({"A": 0.59, "B": 0.2, "C": 0.11, "D": 0.1})
    capped = cap_weights(w, 0.5)
    assert capped.max() <= 0.5 + 1e-12
    assert capped.sum() == pytest.approx(1.0)
    # excess goes to the others in proportion to their original weights
    assert capped["B"] / capped["C"] == pytest.approx(0.2 / 0.11)


def test_cap_weights_infeasible_cap_gives_equal_weight():
    import pandas as pd
    capped = cap_weights(pd.Series({"A": 0.9, "B": 0.1}), 0.3)
    assert capped.tolist() == pytest.approx([0.5, 0.5])


def test_low_price_candidate_is_replaced_by_next_ranked(synthetic_prices, trained_model):
    """v4.5: an untradeable top pick must be swapped for the next-ranked name,
    not leave its money in cash (v4.4 run deployed only 27.6k of 70k)."""
    base = generate_allocation(synthetic_prices, trained_model, top_n=4,
                               balance=100_000)
    victim = base.orders.index[0]
    local = synthetic_prices.copy()
    local[victim] = 0.031                         # like CEIR.L
    alloc = generate_allocation(synthetic_prices, trained_model, top_n=4,
                                balance=100_000, local_prices=local,
                                min_local_price=0.10)
    assert victim not in alloc.orders.index
    assert len(alloc.orders) == 4                 # slot refilled
    assert alloc.total_deployed > 95_000          # money not left in cash


def test_hrp_weights_are_capped_at_twice_equal(synthetic_prices, trained_model):
    alloc = generate_allocation(synthetic_prices, trained_model, top_n=5,
                                balance=100_000, weighting="hrp")
    assert alloc.orders["target_weight"].max() <= 2 / 5 + 1e-4


def test_hold_all_uses_every_available_instrument(synthetic_prices, trained_model):
    """v4.5: the ETF sleeve crashed when one ETF lacked data today."""
    data = synthetic_prices.copy()
    data.iloc[-1, 0] = float("nan")               # one instrument missing today
    alloc = generate_allocation(data, trained_model, top_n=8, balance=100_000,
                                hold_all=True)
    assert len(alloc.orders) == 7


def test_stale_priced_instrument_is_excluded_and_replaced(synthetic_prices, trained_model):
    base = generate_allocation(synthetic_prices, trained_model, top_n=3,
                               balance=100_000)
    victim = base.orders.index[0]
    alloc = generate_allocation(synthetic_prices, trained_model, top_n=3,
                                balance=100_000, exclude=frozenset({victim}))
    assert victim not in alloc.orders.index and len(alloc.orders) == 3


def test_unaffordable_pick_is_replaced_never_a_zero_share_order(synthetic_prices,
                                                                 trained_model):
    """v4.11: SGLD.L-style instrument whose ONE unit costs more than its share of
    the budget used to appear as a 0-share order with its money unused."""
    base = generate_allocation(synthetic_prices, trained_model, top_n=4,
                               balance=100_000)
    victim = base.orders.index[0]
    data = synthetic_prices.copy()
    data[victim] = data[victim] * 1e6            # same returns, huge unit price
    alloc = generate_allocation(data, trained_model, top_n=4, balance=100_000)
    assert victim not in alloc.orders.index
    assert len(alloc.orders) == 4                 # slot refilled by next-ranked
    assert (alloc.orders["shares"] > 0).all()
    assert alloc.total_deployed > 95_000


def test_hold_all_drops_unaffordable_and_respreads_budget(synthetic_prices,
                                                          trained_model):
    data = synthetic_prices.copy()
    victim = data.columns[0]
    data[victim] = data[victim] * 1e6
    alloc = generate_allocation(data, trained_model, top_n=8, balance=100_000,
                                hold_all=True)
    assert victim not in alloc.orders.index
    assert len(alloc.orders) == data.shape[1] - 1
    assert (alloc.orders["shares"] > 0).all()
    assert alloc.orders["target_weight"].round(4).nunique() == 1


# ------------------------------------- v4.20: factor weighting, twin guard
import numpy as np
import pandas as pd


def _two_group_prices(n=600, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    f1, f2 = rng.normal(0.0002, 0.010, n), rng.normal(0.0002, 0.020, n)
    cols, groups = {}, {}
    for i in range(6):
        cols[f"A{i}"], groups[f"A{i}"] = f1 + rng.normal(0, 0.001, n), "ga"
        cols[f"B{i}"], groups[f"B{i}"] = f2 + rng.normal(0, 0.001, n), "gb"
    return pd.DataFrame({k: 100 * np.cumprod(1 + v) for k, v in cols.items()},
                        index=idx), groups


def test_factor_weighting_sizes_by_factor_risk_and_reports_it(trained_model):
    model = trained_model
    px, groups = _two_group_prices()
    alloc = generate_allocation(px, model, top_n=4, balance=100_000, hold_all=True,
                                weighting="factor", factor_groups=groups)
    assert alloc.factor_risk is not None
    w = alloc.orders["target_weight"]
    # the low-vol group (A) carries twice the money for equal factor risk
    assert w[[t for t in w.index if t.startswith("A")]].sum() == pytest.approx(2 / 3, abs=0.05)
    assert alloc.total_deployed <= 100_000
    assert alloc.factor_risk.contributions["ga"] == pytest.approx(0.5, abs=0.05)
    assert "idiosyncratic" in alloc.factor_risk.summary()


def test_factor_weighting_falls_back_to_equal_without_groups(trained_model, caplog):
    model = trained_model
    px, _groups = _two_group_prices()
    alloc = generate_allocation(px, model, top_n=4, balance=100_000, hold_all=True,
                                weighting="factor")
    assert alloc.factor_risk is None
    assert alloc.orders["target_weight"].round(6).nunique() == 1
    assert "falling back to equal weight" in caplog.text


def test_invalid_weighting_is_rejected(synthetic_prices, trained_model):
    model = trained_model
    with pytest.raises(ValueError):
        generate_allocation(synthetic_prices, model, top_n=3, balance=100_000,
                            weighting="markowitz")   # type: ignore[arg-type]


def test_live_correlation_cap_keeps_twins_apart(synthetic_prices, trained_model):
    model = trained_model
    px = synthetic_prices.copy()
    rng = np.random.default_rng(1)
    for t in list(px.columns):
        px[t + "X"] = px[t] * (1 + rng.normal(0, 5e-4, len(px)))     # a clone of every name
    base = generate_allocation(px, model, top_n=4, balance=100_000)
    held = set(base.orders.index)
    assert any(t + "X" in held for t in held)       # without the guard: twins together
    alloc = generate_allocation(px, model, top_n=4, balance=100_000,
                                max_pick_correlation=0.95)
    held = list(alloc.orders.index)
    assert len(held) == 4
    assert not any(t + "X" in held or (t.endswith("X") and t[:-1] in held) for t in held)
    # hold-all sleeves are never filtered: they hold the basket as given
    assert len(generate_allocation(px, model, top_n=4, balance=100_000, hold_all=True,
                                   max_pick_correlation=0.95).orders) == 16
