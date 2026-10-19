"""v4.18 sector cap: selection rule, decision rule, backtest + allocation."""
from __future__ import annotations

import pytest

from umushroom.allocation import generate_allocation
from umushroom.features import build_training_set
from umushroom.model import RankingModel
from umushroom.purged_backtest import run_purged_walk_forward
from umushroom.selection import capped_top, decide_cap, top_sector_share

SECT = {"HSBA": "bank", "LLOY": "bank", "STAN": "bank", "BARC": "bank",
        "AZN": "pharma", "GSK": "pharma", "SHEL": "energy"}


def test_cap_skips_to_next_ranked_other_sector():
    ranked = ["HSBA", "LLOY", "STAN", "BARC", "AZN", "GSK", "SHEL"]
    assert capped_top(ranked, 4, SECT, 2) == ["HSBA", "LLOY", "AZN", "GSK"]
    assert capped_top(ranked, 4) == ["HSBA", "LLOY", "STAN", "BARC"]     # off


def test_cap_fills_by_rank_when_too_few_sectors():
    ranked = ["HSBA", "LLOY", "STAN", "AZN"]
    assert capped_top(ranked, 4, SECT, 1) == ["HSBA", "AZN", "LLOY", "STAN"]


def test_unknown_sectors_are_not_lumped_together():
    assert capped_top(["X", "Y", "Z"], 3, SECT, 1) == ["X", "Y", "Z"]


def test_top_sector_share():
    assert top_sector_share(["HSBA", "LLOY", "AZN", "SHEL"], SECT) == 0.5
    assert top_sector_share([], SECT) == 0.0


@pytest.mark.parametrize("au,ac,vu,vc,adopt", [
    (0.10, 0.09, 0.030, 0.025, True),    # keeps 90%, less risk
    (0.10, 0.08, 0.030, 0.025, True),    # keeps exactly 80% > 75%
    (0.10, 0.07, 0.030, 0.025, False),   # keeps 70%
    (0.10, 0.12, 0.030, 0.025, True),    # better alpha
    (0.10, 0.12, 0.030, 0.031, False),   # but MORE risk
    (-0.02, -0.03, 0.03, 0.02, False),   # worse and nothing to keep
])
def test_pre_declared_decision_rule(au, ac, vu, vc, adopt):
    assert decide_cap(au, ac, vu, vc).adopt is adopt


@pytest.fixture
def sectors_for(synthetic_prices):
    cols = list(synthetic_prices.columns)
    # put half the universe in one "sector" so concentration is possible
    return {t: ("bank" if i < len(cols) // 2 else f"s{i}") for i, t in enumerate(cols)}


def test_backtest_cap_limits_every_period(planted_signal_prices):
    cols = list(planted_signal_prices.columns)
    sectors = {t: f"sec{i % 4}" for i, t in enumerate(cols)}       # 4 sectors
    kw = {"test_window_days": 21, "purge_days": 5, "embargo_days": 5,
          "rebalance_days": 5, "forward_return_days": 5, "top_n": 8,
          "min_train_days": 504, "model_alpha": 1.0, "random_seed": 42,
          "record_cross_sections": True}
    capped = run_purged_walk_forward(planted_signal_prices, sectors=sectors,
                                     max_per_sector=2, **kw)
    plain = run_purged_walk_forward(planted_signal_prices, **kw)
    assert len(capped.period_returns) == len(plain.period_returns)
    for _d, preds, _r in capped.cross_sections:
        picks = capped_top(preds.sort_values(ascending=False).index, 8, sectors, 2)
        assert len(picks) == 8 and top_sector_share(picks, sectors) <= 0.25
    # same signal, different selection -> different but related returns
    assert not capped.period_returns.equals(plain.period_returns)


def test_allocation_respects_cap(synthetic_prices, sectors_for):
    X, y = build_training_set(synthetic_prices, forward_return_days=5, step=5,
                              min_history_days=70)
    model = RankingModel(alpha=1.0, random_seed=42).fit(X, y)
    alloc = generate_allocation(synthetic_prices, model, top_n=4, balance=100_000,
                                sectors=sectors_for, max_per_sector=1)
    held = list(alloc.orders.index)
    assert len(held) == 4
    assert sum(sectors_for[t] == "bank" for t in held) <= 1


def test_allocation_refill_keeps_cap(synthetic_prices, sectors_for):
    X, y = build_training_set(synthetic_prices, forward_return_days=5, step=5,
                              min_history_days=70)
    model = RankingModel(alpha=1.0, random_seed=42).fit(X, y)
    base = generate_allocation(synthetic_prices, model, top_n=3, balance=100_000,
                               sectors=sectors_for, max_per_sector=1)
    victim = base.orders.index[0]
    data = synthetic_prices.copy()
    data[victim] = data[victim] * 1e6                  # unaffordable now
    alloc = generate_allocation(data, model, top_n=3, balance=100_000,
                                sectors=sectors_for, max_per_sector=1)
    held = list(alloc.orders.index)
    assert victim not in held and len(held) == 3
    assert sum(sectors_for[t] == "bank" for t in held) <= 1


def test_hold_all_ignores_cap(synthetic_prices, sectors_for):
    X, y = build_training_set(synthetic_prices, forward_return_days=5, step=5,
                              min_history_days=70)
    model = RankingModel(alpha=1.0, random_seed=42).fit(X, y)
    alloc = generate_allocation(synthetic_prices, model, top_n=3, balance=100_000,
                                hold_all=True, sectors=sectors_for, max_per_sector=1)
    assert len(alloc.orders) == synthetic_prices.shape[1]


def test_config_parses_and_validates_cap(tmp_path):
    from umushroom.config import ConfigError, load_config
    good = """tickers: [A, B, C, D]
starting_balance: 1000
top_n_holdings: 2
lookback_period: "5y"
forward_return_days: 5
rebalance_days: 5
min_history_days: 300
model_alpha: 1.0
random_seed: 1
cache_dir: ".cache"
cache_max_age_hours: 12
log_level: "WARNING"
purged_test_window_days: 21
purged_purge_days: 5
purged_embargo_days: 5
purged_min_train_days: 504
prediction_log_path: "p.csv"
sector_file: "s.csv"
"""
    (tmp_path / "s.csv").write_text("yahoo,sector\nA,bank\n")
    (tmp_path / "c.yaml").write_text(good + "max_per_sector: 3\n")
    cfg = load_config(tmp_path / "c.yaml")
    assert cfg.max_per_sector == 3 and cfg.all_sleeves[0].max_per_sector == 3
    (tmp_path / "d.yaml").write_text(good)
    assert load_config(tmp_path / "d.yaml").max_per_sector is None
    (tmp_path / "e.yaml").write_text(good.replace('sector_file: "s.csv"\n', "")
                                     + "max_per_sector: 3\n")
    with pytest.raises(ConfigError):
        load_config(tmp_path / "e.yaml")


# ------------------------------------------------ v4.20 near-duplicate guard
import numpy as np
import pandas as pd

from umushroom.selection import correlation_filter


def _clones_and_others(n=400, seed=5):
    """A, A2 move together (rho ~ 0.99); B, C, D are independent; NEW listed
    only 30 days ago (not judgeable)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    base = rng.normal(0, 0.01, n)
    r = {"A": base, "A2": base + rng.normal(0, 0.001, n),
         "B": rng.normal(0, 0.01, n), "C": rng.normal(0, 0.01, n),
         "D": rng.normal(0, 0.01, n), "NEW": rng.normal(0, 0.01, n)}
    px = pd.DataFrame({k: 100 * np.cumprod(1 + v) for k, v in r.items()}, index=idx)
    px.loc[px.index[:-30], "NEW"] = np.nan
    return px


def test_near_duplicate_is_skipped_for_the_next_ranked_name():
    px = _clones_and_others()
    sim = correlation_filter(px, len(px) - 1, max_correlation=0.9)
    assert sim("A2", ["A"]) and not sim("B", ["A"]) and not sim("A", [])
    assert capped_top(["A", "A2", "B", "C", "D"], 3, too_similar=sim) == ["A", "B", "C"]
    # combined with the sector cap: both constraints hold
    sect = {"A": "s1", "A2": "s1", "B": "s1", "C": "s2", "D": "s2"}
    assert capped_top(["A", "A2", "B", "C", "D"], 3, sect, 1, too_similar=sim) == \
        ["A", "C", "B"]


def test_duplicates_fill_last_only_when_nothing_else_is_left():
    px = _clones_and_others()
    sim = correlation_filter(px, len(px) - 1, max_correlation=0.9)
    # only A, A2 and B exist: a 3-name book must take the clone, last
    assert capped_top(["A", "A2", "B"], 3, too_similar=sim) == ["A", "B", "A2"]
    sect = {"A": "s1", "A2": "s1", "B": "s2"}
    # sector-capped names top up BEFORE near-duplicates
    assert capped_top(["A", "A2", "B", "C"], 4, sect, 1, too_similar=sim) == \
        ["A", "B", "C", "A2"]


def test_unjudgeable_pairs_are_allowed_and_unknown_names_pass():
    px = _clones_and_others()
    sim = correlation_filter(px, len(px) - 1, max_correlation=0.9)
    assert not sim("NEW", ["A", "B"])          # 30 days of overlap < MIN_CORR_OBS
    assert not sim("ZZZ", ["A"])               # not in the price frame


def test_correlation_filter_is_causal():
    """The predicate at idx must not change when future rows are appended
    or altered -- same discipline as the no-lookahead feature test."""
    px = _clones_and_others()
    idx = 300
    future = px.copy()
    future.iloc[idx + 1:] *= 3.0                   # corrupt everything after idx
    a = correlation_filter(px, idx, 0.9)
    b = correlation_filter(future, idx, 0.9)
    c = correlation_filter(px.iloc[: idx + 1], idx, 0.9)
    for cand, picks in (("A2", ["A"]), ("B", ["A"]), ("C", ["A", "B"])):
        assert a(cand, picks) == b(cand, picks) == c(cand, picks)


def test_correlation_filter_rejects_bad_arguments():
    px = _clones_and_others()
    with pytest.raises(ValueError):
        correlation_filter(px, len(px) - 1, 0.0)
    with pytest.raises(ValueError):
        correlation_filter(px, len(px) - 1, 1.5)
    with pytest.raises(ValueError):
        correlation_filter(px, len(px), 0.9)
