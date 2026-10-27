from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from umushroom.config import StrategyConfig, load_config
from umushroom.exceptions import ConfigError

VALID_YAML = textwrap.dedent("""\
    tickers: ["AAA", "BBB", "CCC"]
    starting_balance: 100000
    top_n_holdings: 2
    lookback_period: "2y"
    forward_return_days: 5
    rebalance_days: 5
    min_history_days: 70
    model_alpha: 1.0
    random_seed: 42
    cache_dir: ".cache"
    cache_max_age_hours: 12
    log_level: "INFO"
    purged_test_window_days: 20
    purged_purge_days: 5
    purged_embargo_days: 5
    purged_min_train_days: 70
    prediction_log_path: ".cache/prediction_log.csv"
""")


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


def _base_kwargs(tmp_path: Path) -> dict:
    """Shared baseline of valid StrategyConfig kwargs for direct
    dataclass construction tests. Each test overrides only the field(s)
    it's probing."""
    return {
        "tickers": ["AAA", "BBB"],
        "starting_balance": 100000,
        "top_n_holdings": 1,
        "lookback_period": "2y",
        "forward_return_days": 5,
        "rebalance_days": 5,
        "min_history_days": 70,
        "model_alpha": 1.0,
        "random_seed": 42,
        "cache_dir": tmp_path,
        "cache_max_age_hours": 12,
        "log_level": "INFO",
        "purged_test_window_days": 20,
        "purged_purge_days": 5,
        "purged_embargo_days": 5,
        "purged_min_train_days": 70,
        "prediction_log_path": tmp_path / "pred_log.csv",
    }


def test_load_valid_config(tmp_path):
    path = _write(tmp_path, VALID_YAML)
    cfg = load_config(path)
    assert cfg.tickers == ["AAA", "BBB", "CCC"]
    assert cfg.starting_balance == 100000.0
    assert cfg.top_n_holdings == 2
    assert cfg.purged_test_window_days == 20
    assert cfg.prediction_log_path == Path(".cache/prediction_log.csv")


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_missing_required_key_raises(tmp_path):
    bad = VALID_YAML.replace("starting_balance: 100000\n", "")
    path = _write(tmp_path, bad)
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_new_required_key_raises(tmp_path):
    bad = VALID_YAML.replace("purged_purge_days: 5\n", "")
    path = _write(tmp_path, bad)
    with pytest.raises(ConfigError):
        load_config(path)


def test_malformed_yaml_raises(tmp_path):
    path = _write(tmp_path, "tickers: [unclosed")
    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize("field,value", [
    ("starting_balance", 0),
    ("starting_balance", -500),
    ("top_n_holdings", 0),
    ("forward_return_days", 0),
    ("rebalance_days", 0),
    ("purged_test_window_days", 0),
    ("purged_embargo_days", -1),
])
def test_invalid_scalar_values_raise(tmp_path, field, value):
    kwargs = _base_kwargs(tmp_path)
    kwargs[field] = value
    with pytest.raises(ConfigError):
        StrategyConfig(**kwargs)


def test_top_n_exceeding_universe_raises(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    kwargs["top_n_holdings"] = 5  # only 2 tickers in base kwargs
    with pytest.raises(ConfigError):
        StrategyConfig(**kwargs)


def test_duplicate_tickers_raise(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    kwargs["tickers"] = ["AAA", "AAA"]
    with pytest.raises(ConfigError):
        StrategyConfig(**kwargs)


def test_invalid_log_level_raises(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    kwargs["log_level"] = "NOT_A_LEVEL"
    with pytest.raises(ConfigError):
        StrategyConfig(**kwargs)


def test_purge_days_less_than_forward_return_days_raises(tmp_path):
    """The leakage guard itself must be enforced at config-validation
    time, not discovered later as a silent correctness bug."""
    kwargs = _base_kwargs(tmp_path)
    kwargs["forward_return_days"] = 10
    kwargs["purged_purge_days"] = 3  # less than forward_return_days
    with pytest.raises(ConfigError):
        StrategyConfig(**kwargs)


def test_purged_min_train_days_below_min_history_days_raises(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    kwargs["min_history_days"] = 100
    kwargs["purged_min_train_days"] = 50
    with pytest.raises(ConfigError):
        StrategyConfig(**kwargs)


def test_env_var_overrides_scalar_field(tmp_path, monkeypatch):
    path = _write(tmp_path, VALID_YAML)
    monkeypatch.setenv("UMUSHROOM_STARTING_BALANCE", "50000")
    cfg = load_config(path)
    assert cfg.starting_balance == 50000.0


# ---------------------------------------------------------------- v4 tests
REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def test_v3_style_config_still_loads_with_v4_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, VALID_YAML))
    assert cfg.base_currency == "GBP"
    assert cfg.target == "rank"
    assert cfg.neutralize_by_exchange is True


def test_shipped_config_is_valid_and_multi_market():
    """Regression test for the v3 Swiss-only universe."""
    from umushroom.universe import exchange_mix, resolve_universe
    cfg = load_config(REPO_CONFIG)
    mix = exchange_mix(resolve_universe(cfg.tickers, cfg.currency_overrides))
    assert len(mix) >= 3
    assert max(mix.values()) <= cfg.max_single_exchange_fraction


@pytest.mark.parametrize("field,value", [
    ("base_currency", "XYZ"),
    ("target", "magic"),
    ("max_single_exchange_fraction", 0.0),
    ("fama_french_region", "Mars"),
    ("tickers", ["AAPL", "FOO.QQ"]),
    ("currency_overrides", {"NOT_IN_UNIVERSE.L": "USD"}),
])
def test_invalid_v4_fields_raise(tmp_path, field, value):
    kwargs = _base_kwargs(tmp_path)
    kwargs[field] = value
    with pytest.raises(ConfigError):
        StrategyConfig(**kwargs)


def test_env_var_bool_override(tmp_path, monkeypatch):
    path = _write(tmp_path, VALID_YAML + "neutralize_by_exchange: true\n")
    monkeypatch.setenv("UMUSHROOM_NEUTRALIZE_BY_EXCHANGE", "false")
    assert load_config(path).neutralize_by_exchange is False


def test_env_var_bad_bool_raises(tmp_path, monkeypatch):
    path = _write(tmp_path, VALID_YAML + "drop_incomplete_bar: true\n")
    monkeypatch.setenv("UMUSHROOM_DROP_INCOMPLETE_BAR", "maybe")
    with pytest.raises(ConfigError):
        load_config(path)


# ---------------------------------------------------------------- v4.2 sleeves
from umushroom.config import EQUITY_SLEEVE, SleeveConfig

SLEEVE_YAML = VALID_YAML + textwrap.dedent("""\
    sleeves:
      crypto:
        weight: 0.1
        top_n: 2
        fractional: true
        tickers: ["BTC-USD", "ETH-USD", "SOL-USD"]
      etf:
        weight: 0.2
        top_n: 1
        tickers: ["IWDA.L", "SGLD.L"]
    currency_overrides: {"IWDA.L": "USD"}
""")


def test_sleeves_parse_and_equity_gets_remainder(tmp_path):
    cfg = load_config(_write(tmp_path, SLEEVE_YAML))
    names = [s.name for s in cfg.all_sleeves]
    assert names == [EQUITY_SLEEVE, "crypto", "etf"]
    assert cfg.equity_weight == pytest.approx(0.7)
    crypto = cfg.all_sleeves[1]
    assert crypto.fractional and crypto.top_n == 2
    assert "BTC-USD" in cfg.all_tickers and "AAA" in cfg.all_tickers


def _sleeve(**kw):
    base = {"name": "crypto", "tickers": ["BTC-USD", "ETH-USD"], "weight": 0.1,
            "top_n": 1}
    base.update(kw)
    return SleeveConfig(**base)


@pytest.mark.parametrize("sleeve", [
    _sleeve(weight=0.0),
    _sleeve(weight=1.0),
    _sleeve(top_n=3),
    _sleeve(tickers=[]),
    _sleeve(name="equity"),
    _sleeve(tickers=["AAA", "BTC-USD"]),       # AAA already in equity sleeve
])
def test_invalid_sleeves_raise(tmp_path, sleeve):
    kwargs = _base_kwargs(tmp_path)
    kwargs["sleeves"] = [sleeve]
    with pytest.raises(ConfigError):
        StrategyConfig(**kwargs)


def test_sleeve_weights_must_leave_room_for_equity(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    kwargs["sleeves"] = [_sleeve(weight=0.6),
                         _sleeve(name="etf", tickers=["IWDA.L", "SGLD.L"], weight=0.5)]
    with pytest.raises(ConfigError):
        StrategyConfig(**kwargs)


def test_shipped_config_sleeves_are_valid():
    cfg = load_config(REPO_CONFIG)
    names = {s.name: s for s in cfg.all_sleeves}
    # v4.18: crypto sleeve switched off (no skill; only added volatility);
    # its 10% went to the equity sleeve
    assert set(names) == {"equity", "etf"}
    assert sum(s.weight for s in cfg.all_sleeves) == pytest.approx(1.0)
    assert names["equity"].weight == pytest.approx(0.80)
    # v4.20 (2026-10-19 head-to-heads): factor-risk weights for ETFs, HRP for
    # stocks; learner ridge and the near-duplicate guard off until their
    # rules say otherwise (see config.yaml for the numbers)
    assert names["etf"].selection == "auto" and names["etf"].weighting == "equal"
    assert names["equity"].weighting == "hrp" and cfg.weighting == "hrp"
    # gbm: won the learner head-to-head (6-week active t +3.95) and passed
    # the 7-check validation suite on 2026-10-19 -- for STOCKS; the ETF sleeve
    # keeps ridge (no learner had a contest-window edge there)
    assert cfg.model_type == "gbm" and cfg.max_pick_correlation is None
    assert sleeve_model_type(cfg, names["equity"]) == "gbm"
    assert sleeve_model_type(cfg, names["etf"]) == "ridge"
    assert cfg.max_missing_fraction == pytest.approx(0.10)


def test_sleeve_selection_all_and_weighting_parse(tmp_path):
    y = SLEEVE_YAML.replace("    top_n: 1\n",
                            "    selection: all\n    weighting: equal\n")
    assert y != SLEEVE_YAML
    cfg = load_config(_write(tmp_path, y))
    etf = next(s for s in cfg.all_sleeves if s.name == "etf")
    assert etf.selection == "all" and etf.weighting == "equal"
    assert etf.top_n == 2                   # defaults to the sleeve's size


@pytest.mark.parametrize("field,value", [("selection", "best"), ("weighting", "magic")])
def test_invalid_sleeve_selection_or_weighting_raises(tmp_path, field, value):
    kwargs = _base_kwargs(tmp_path)
    kwargs["sleeves"] = [_sleeve(**{field: value})]
    with pytest.raises(ConfigError):
        StrategyConfig(**kwargs)


def test_shipped_config_etf_sleeve_is_a_model_sleeve_over_the_universe():
    """v4.19: the ETF sleeve ranks the whole UMushroom ETF universe with the
    ETF feature set and falls back to the 13 core ETFs under `auto`."""
    etf = next(s for s in load_config(REPO_CONFIG).all_sleeves if s.name == "etf")
    assert etf.selection == "auto" and etf.weighting == "equal"
    assert etf.feature_set == "etf" and etf.group_by == "asset_class"
    assert len(etf.core_tickers) == 13 and set(etf.core_tickers) <= set(etf.tickers)
    assert len(etf.tickers) > 1000 and not etf.universe_missing
    assert etf.forward_return_days is not None and etf.forward_return_days <= 30
    assert set(etf.groups.values()) >= {"equity", "bond", "commodity"}


def test_shipped_config_iglt_quoted_in_pounds():
    """IGLT.L is quoted in GBP (not pence) on Yahoo -- verified against
    UMushroom's own price on 2026-10-08."""
    from umushroom.universe import resolve_listing
    cfg = load_config(REPO_CONFIG)
    lst = resolve_listing("IGLT.L", cfg.currency_overrides)
    assert lst.currency == "GBP" and lst.price_divisor == 1.0


def test_sector_file_required_and_must_exist(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    kwargs["neutralize_by_sector"] = True
    with pytest.raises(ConfigError):
        StrategyConfig(**kwargs)
    kwargs["sector_file"] = tmp_path / "nope.csv"
    with pytest.raises(ConfigError):
        StrategyConfig(**kwargs)


def test_shipped_config_v45_settings():
    """Equity sleeve = the setup that WON the 10-year head-to-head (v4.3
    settings), on the v4.4 data (10y, ragged history)."""
    from umushroom.config import load_sectors, sleeve_horizon
    cfg = load_config(REPO_CONFIG)
    eq, *_ = cfg.all_sleeves
    assert eq.neutralize_by_exchange
    assert not eq.neutralize_by_sector and not eq.use_volume_features
    assert sleeve_horizon(cfg, eq) == 5 and eq.top_n == 10
    assert cfg.ragged_history and cfg.lookback_period == "10y"
    sectors = load_sectors(cfg.sector_file)
    covered = sum(t in sectors for t in cfg.tickers) / len(cfg.tickers)
    assert covered > 0.99


def test_shipped_config_has_no_usd_quoted_london_gdrs():
    """v4.5: London GDRs trade in USD, not pence (CEIR.L showed 0.031
    instead of ~3.1); they are excluded from the universe."""
    import csv
    cfg = load_config(REPO_CONFIG)
    with open(cfg.sector_file, newline="") as fh:
        gdrs = {r["yahoo"] for r in csv.DictReader(fh) if "GDR" in r["name"].upper()}
    assert not gdrs & set(cfg.tickers)


def test_sleeve_horizon_may_exceed_purge_but_not_the_contest_window(tmp_path):
    """v4.19: the walk-forward lifts the purge to the horizon itself, so a
    21-day ETF horizon with a 5-day purge is fine; a 31-day horizon is not
    (the contest is 30 trading days)."""
    kwargs = _base_kwargs(tmp_path)
    kwargs["sleeves"] = [_sleeve(forward_return_days=21)]   # purge is 5
    StrategyConfig(**kwargs)
    kwargs["sleeves"] = [_sleeve(forward_return_days=31)]
    with pytest.raises(ConfigError, match="contest window"):
        StrategyConfig(**kwargs)
    kwargs["sleeves"] = []
    kwargs["forward_return_days"] = 40
    kwargs["purged_purge_days"] = 40
    with pytest.raises(ConfigError, match="contest window"):
        StrategyConfig(**kwargs)


def test_shipped_config_ticker_fixes_v47():
    cfg = load_config(REPO_CONFIG)
    for good in ("SBRY.L", "MNG.L", "BME.L", "JDW.L", "VZUG.SW", "RSGN.SW",
                 "SPGI", "TMUS"):
        assert good in cfg.tickers, good
    for bad in ("SBRY-J.L", "MNG-M.L", "BME-B.L", "JDW-J.L", "VZUG-V.SW", "RSGN-R.SW"):
        assert bad not in cfg.tickers, bad
    for t in ("CRH.L", "FERG.L", "INDV.L", "FLTR.L"):     # moved to the US
        assert cfg.currency_overrides[t] == "USD"
    assert cfg.yahoo_symbols["FLTR.L"] == "FLUT"
    assert cfg.yahoo_symbols["TM17.L"] == "EVPL.L"
    # never an OTC/ADR line (wrong price level, e.g. RHHBY = 1/8 Roche share)
    from umushroom.check_tickers import is_otc_or_adr
    assert not any(is_otc_or_adr(v) for v in cfg.yahoo_symbols.values())
    assert "ROG.SW" not in cfg.tickers and "RO.SW" in cfg.tickers


def test_yahoo_symbols_for_unknown_ticker_raises(tmp_path):
    kwargs = _base_kwargs(tmp_path)
    kwargs["yahoo_symbols"] = {"NOPE.L": "NOPE"}
    with pytest.raises(ConfigError):
        StrategyConfig(**kwargs)


# ---------------------------------------------------------------- v4.19 ETF universe

UNIVERSE_CSV = textwrap.dedent("""\
    yahoo,isin,name,umushroom_symbol,umushroom_currency,yahoo_currency,yahoo_exchange,yahoo_name,asset_class,style,manager,url_name,umushroom_price,price_date,history_days,first_date,last_date,name_similarity,via,status,note
    IWDA.L,IE00B4L5Y983,iShares Core MSCI World,IWDA,USD,USD,LSE,iShares Core MSCI World,equity,,BlackRock,iwda,147.5,2026-10-12,2500,2016-10-12,2026-10-12,1.0,search,ok,
    IGLT.L,IE00B1FZSB30,iShares Core UK Gilts,IGLT,GBP,GBp,LSE,iShares Core UK Gilts,bond,,BlackRock,iglt,9.5,2026-10-12,2500,2016-10-12,2026-10-12,1.0,search,ok,
    XESC.DE,LU0380865021,Xtrackers Euro Stoxx 50,XESC,EUR,EUR,GER,Xtrackers Euro Stoxx 50,equity,,DWS,xesc,110.0,2026-10-12,2500,2016-10-12,2026-10-12,1.0,search,ok,
    DEAD.L,XX0000000001,Dead ETF,DEAD,USD,,,,equity,,,dead,1.0,2026-10-12,0,,,,,no_yahoo_data,no data
""")


def _universe_yaml(extra: str = "") -> str:
    return VALID_YAML + textwrap.dedent(f"""\
        sleeves:
          etf:
            weight: 0.2
            top_n: 2
            feature_set: etf
            group_by: asset_class
            max_per_group: 1
            selection: auto
            universe_file: "universe.csv"
            core_tickers: ["IWDA.L", "SGLD.L"]
            {extra}
        currency_overrides: {{"SGLD.L": "USD"}}
    """)


def test_universe_file_fills_tickers_groups_meta_and_overrides(tmp_path):
    (tmp_path / "universe.csv").write_text(UNIVERSE_CSV)
    cfg = load_config(_write(tmp_path, _universe_yaml()))
    etf = cfg.sleeves[0]
    # core first, then the universe's ok rows (no duplicates, dead row left out)
    assert etf.tickers == ["IWDA.L", "SGLD.L", "IGLT.L", "XESC.DE"]
    assert etf.groups == {"IWDA.L": "equity", "IGLT.L": "bond", "XESC.DE": "equity"}
    assert etf.meta["IGLT.L"]["umushroom_symbol"] == "IGLT"
    assert etf.universe_status == {"ok": 3, "no_yahoo_data": 1}
    assert not etf.universe_missing
    # USD line on London from the file, plus the explicit override from the YAML
    assert cfg.currency_overrides == {"IWDA.L": "USD", "SGLD.L": "USD"}
    assert cfg.contest_window_days == 30


def test_missing_universe_file_degrades_to_core_tickers(tmp_path):
    cfg = load_config(_write(tmp_path, _universe_yaml()))
    etf = cfg.sleeves[0]
    assert etf.tickers == ["IWDA.L", "SGLD.L"] and etf.universe_missing
    assert etf.groups == {}


def test_explicit_override_wins_over_universe_file(tmp_path):
    (tmp_path / "universe.csv").write_text(UNIVERSE_CSV)
    yaml_text = _universe_yaml().replace('currency_overrides: {"SGLD.L": "USD"}',
                                         'currency_overrides: {"IWDA.L": "GBP", "SGLD.L": "USD"}')
    cfg = load_config(_write(tmp_path, yaml_text))
    assert cfg.currency_overrides["IWDA.L"] == "GBP"


@pytest.mark.parametrize("extra", [
    "feature_set: bonds",
    "group_by: sector",
    "max_per_group: -1",
])
def test_invalid_etf_sleeve_fields_raise(tmp_path, extra):
    (tmp_path / "universe.csv").write_text(UNIVERSE_CSV)
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, _universe_yaml(extra)))


def test_auto_selection_needs_core_tickers(tmp_path):
    (tmp_path / "universe.csv").write_text(UNIVERSE_CSV)
    yaml_text = _universe_yaml().replace('core_tickers: ["IWDA.L", "SGLD.L"]', "")
    with pytest.raises(ConfigError, match="core_tickers"):
        load_config(_write(tmp_path, yaml_text))


def test_asset_class_grouping_needs_a_universe(tmp_path):
    # no universe file -> no asset classes -> group_by: asset_class is an error
    yaml_text = _universe_yaml().replace('universe_file: "universe.csv"', "")
    with pytest.raises(ConfigError, match="asset_class"):
        load_config(_write(tmp_path, yaml_text))


def test_contest_window_days_from_yaml(tmp_path):
    cfg = load_config(_write(tmp_path, VALID_YAML + "contest_window_days: 20\n"))
    assert cfg.contest_window_days == 20
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, VALID_YAML + "contest_window_days: 0\n"))


# ------------------------------------------------------------- v4.20 fields
from umushroom.config import sleeve_model_type, sleeve_pick_correlation


def test_v420_defaults_keep_the_strategy_unchanged(tmp_path):
    cfg = load_config(_write(tmp_path, VALID_YAML))
    assert cfg.model_type == "ridge" and cfg.max_pick_correlation is None
    assert cfg.max_missing_fraction == pytest.approx(0.10)
    eq = cfg.all_sleeves[0]
    assert sleeve_model_type(cfg, eq) == "ridge"
    assert sleeve_pick_correlation(cfg, eq) is None


def test_v420_fields_parse_and_sleeves_inherit_or_override(tmp_path):
    yaml_text = VALID_YAML + textwrap.dedent("""\
        model_type: gbm
        max_pick_correlation: 0.9
        max_missing_fraction: 0.25
        sleeves:
          etf:
            weight: 0.2
            top_n: 1
            tickers: ["IWDA.L", "SGLD.L"]
            weighting: factor
            model_type: mlp
          crypto:
            weight: 0.1
            top_n: 1
            tickers: ["BTC-USD"]
            max_pick_correlation: 0
    """)
    cfg = load_config(_write(tmp_path, yaml_text))
    assert cfg.model_type == "gbm" and cfg.max_missing_fraction == 0.25
    eq, etf, crypto = cfg.all_sleeves
    assert sleeve_model_type(cfg, eq) == "gbm" and sleeve_model_type(cfg, etf) == "mlp"
    assert sleeve_pick_correlation(cfg, eq) == 0.9
    assert sleeve_pick_correlation(cfg, etf) == 0.9        # inherits the global
    assert sleeve_pick_correlation(cfg, crypto) is None    # 0 switches it OFF
    assert etf.weighting == "factor"


@pytest.mark.parametrize("extra", [
    "model_type: lstm\n", "max_pick_correlation: 1.5\n", "max_pick_correlation: 0\n",
    "max_missing_fraction: 1.5\n",
])
def test_invalid_v420_top_level_values_raise(tmp_path, extra):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, VALID_YAML + extra))


@pytest.mark.parametrize("field, value", [
    ("model_type", "lstm"), ("max_pick_correlation", 2), ("weighting", "markowitz"),
])
def test_invalid_v420_sleeve_values_raise(tmp_path, field, value):
    yaml_text = VALID_YAML + textwrap.dedent(f"""\
        sleeves:
          etf:
            weight: 0.2
            top_n: 1
            tickers: ["IWDA.L"]
            {field}: {value}
    """)
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, yaml_text))


def test_env_override_of_v420_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("UMUSHROOM_MAX_PICK_CORRELATION", "0.85")
    monkeypatch.setenv("UMUSHROOM_MODEL_TYPE", "gbm")
    cfg = load_config(_write(tmp_path, VALID_YAML + "max_pick_correlation: 0.9\n"
                             "model_type: ridge\n"))
    assert cfg.max_pick_correlation == 0.85 and cfg.model_type == "gbm"


def test_top_level_weighting_sets_the_stock_sleeve(tmp_path):
    cfg = load_config(_write(tmp_path, VALID_YAML + "weighting: hrp\n"))
    assert cfg.weighting == "hrp" and cfg.all_sleeves[0].weighting == "hrp"
    assert load_config(_write(tmp_path, VALID_YAML)).all_sleeves[0].weighting is None
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, VALID_YAML + "weighting: markowitz\n"))



# ------------------------------------------------------------- v4.21 fields
from umushroom.config import data_period


def test_v421_fields_parse_validate_and_default_off(tmp_path):
    cfg = load_config(_write(tmp_path, VALID_YAML))
    assert cfg.history_start is None and not cfg.require_fresh_data
    assert cfg.auto_confirm_runs == 1 and data_period(cfg) == "2y"
    cfg = load_config(_write(tmp_path, VALID_YAML + textwrap.dedent("""\
        history_start: "2016-10-01"
        require_fresh_data: true
        auto_confirm_runs: 3
    """)))
    assert data_period(cfg) == "start:2016-10-01" and cfg.require_fresh_data
    assert cfg.auto_confirm_runs == 3
    for bad in ("history_start: 2016/10/01\n", "auto_confirm_runs: 0\n",
                "auto_trade_on: maybe\n"):
        with pytest.raises(ConfigError):
            load_config(_write(tmp_path, VALID_YAML + bad))
    assert load_config(_write(tmp_path, VALID_YAML)).auto_trade_on == "pass"
    assert load_config(_write(tmp_path, VALID_YAML + "auto_trade_on: WARN\n")).auto_trade_on == "warn"


def test_shipped_config_v421_settings():
    cfg = load_config(REPO_CONFIG)
    assert cfg.history_start == "2016-10-01" and cfg.require_fresh_data
    assert cfg.auto_confirm_runs == 3 and cfg.auto_trade_on == "pass"
    etf = next(s for s in cfg.all_sleeves if s.name == "etf")
    # 2026-10-27: factor weights put 72% of the live ETF book in three silver
    # ETPs -> equal weight again, with the near-duplicate guard on
    assert etf.weighting == "equal" and sleeve_pick_correlation(cfg, etf) == 0.9
    assert sleeve_model_type(cfg, etf) == "ridge"
