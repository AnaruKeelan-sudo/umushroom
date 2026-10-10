"""End-to-end smoke test of the full CLI pipeline, offline: the data
layer is replaced by a fake fetcher, everything else runs for real."""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

from umushroom import cli
from umushroom import data as data_mod


def _fake_fetcher(symbols, period):
    rng = np.random.default_rng(0)
    n = 900
    idx = pd.bdate_range("2023-01-02", periods=n)
    cols = {}
    for s in symbols:
        if s.endswith("=X"):
            cols[s] = 0.8 * np.exp(rng.normal(0, 0.003, n).cumsum())
        else:
            level = 5000.0 if s.endswith(".L") else 150.0
            cols[s] = level * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    return pd.DataFrame(cols, index=idx)


def test_cli_full_pipeline_offline(tmp_path, monkeypatch, capsys):
    us = [f"U{i}" for i in range(8)]
    uk = [f"K{i}.L" for i in range(8)]
    ch = [f"C{i}.SW" for i in range(6)]
    tickers = us + uk + ch
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        tickers: {tickers}
        starting_balance: 100000
        top_n_holdings: 5
        lookback_period: "5y"
        forward_return_days: 5
        rebalance_days: 5
        min_history_days: 300
        model_alpha: 1.0
        random_seed: 42
        cache_dir: "{tmp_path / '.cache'}"
        cache_max_age_hours: 12
        log_level: "WARNING"
        purged_test_window_days: 21
        purged_purge_days: 5
        purged_embargo_days: 5
        purged_min_train_days: 504
        prediction_log_path: "{tmp_path / 'pred.csv'}"
        base_currency: "GBP"
    """))

    real = data_mod.get_market_data

    def fake_get_market_data(**kw):
        return real(**kw, fetcher=_fake_fetcher)

    monkeypatch.setattr(data_mod, "get_market_data", fake_get_market_data)
    out_csv = tmp_path / "alloc.csv"
    rc = cli.main(["--config", str(cfg), "--weighting", "hrp",
                   "--output", str(out_csv)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Purged & Embargoed Walk-Forward" in out
    assert "IC t-statistic" in out
    assert "Alpha vs. equal-weight universe" in out
    orders = pd.read_csv(out_csv, index_col=0)
    assert len(orders) == 5
    assert (orders["shares"] >= 0).all()
    assert (orders["shares"] * orders["price"]).sum() <= 100000
    # London pence must have been converted: a 5000p stock is ~GBP 50
    uk_rows = orders[orders.index.str.endswith(".L")]
    assert (uk_rows["price"] < 1000).all()
    assert (tmp_path / "pred.csv").exists()


def test_cli_bad_config_returns_2(tmp_path):
    assert cli.main(["--config", str(tmp_path / "missing.yaml")]) == 2


def _fake_fetcher_7day(symbols, period):
    """Calendar-day index: stocks/ETFs/FX are NaN at weekends, crypto
    trades every day -- like real Yahoo data."""
    rng = np.random.default_rng(1)
    n = 1300
    idx = pd.date_range("2023-01-02", periods=n, freq="D")
    weekend = idx.dayofweek >= 5
    cols = {}
    for s in symbols:
        if s.endswith("=X"):
            v = 0.8 * np.exp(rng.normal(0, 0.003, n).cumsum())
        elif s.endswith("-USD"):
            v = 30000.0 * np.cumprod(1 + rng.normal(0.0005, 0.04, n))
        else:
            level = 5000.0 if s.endswith(".L") else 150.0
            v = level * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
        if not s.endswith("-USD"):
            v = np.where(weekend, np.nan, v)
        cols[s] = v
    return pd.DataFrame(cols, index=idx)


def test_cli_runs_all_sleeves_and_combines_orders(tmp_path, monkeypatch, capsys):
    stocks = [f"U{i}" for i in range(8)] + [f"K{i}.L" for i in range(8)]
    crypto = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD"]
    etfs = ["IWDA.L", "SGLD.L", "ISF.L", "IGLT.L", "XESC.DE"]
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        tickers: {stocks}
        starting_balance: 100000
        top_n_holdings: 5
        lookback_period: "5y"
        forward_return_days: 5
        rebalance_days: 5
        min_history_days: 300
        model_alpha: 1.0
        random_seed: 42
        cache_dir: "{tmp_path / '.cache'}"
        cache_max_age_hours: 12
        log_level: "WARNING"
        purged_test_window_days: 21
        purged_purge_days: 5
        purged_embargo_days: 5
        purged_min_train_days: 504
        prediction_log_path: "{tmp_path / 'pred.csv'}"
        base_currency: "GBP"
        currency_overrides: {{"IWDA.L": "USD", "SGLD.L": "USD"}}
        sleeves:
          crypto: {{weight: 0.1, top_n: 2, fractional: true, tickers: {crypto}}}
          etf: {{weight: 0.2, selection: all, weighting: equal, tickers: {etfs}}}
    """))
    real = data_mod.get_market_data
    monkeypatch.setattr(data_mod, "get_market_data",
                        lambda **kw: real(**kw, fetcher=_fake_fetcher_7day))
    out_csv = tmp_path / "alloc.csv"
    rc = cli.main(["--config", str(cfg), "--output", str(out_csv)])
    out = capsys.readouterr().out
    assert rc == 0, out
    for name in ("EQUITY", "CRYPTO", "ETF"):
        assert f"SLEEVE: {name}" in out
    orders = pd.read_csv(out_csv, index_col=0)
    assert set(orders["sleeve"]) == {"equity", "crypto", "etf"}
    etf_rows = orders[orders["sleeve"] == "etf"]
    # selection: all -- except any ETF whose one unit costs more than its
    # equal share (v4.11: dropped and reported, never a 0-share order)
    assert set(etf_rows.index) <= set(etfs) and len(etf_rows) >= len(etfs) - 1
    assert (orders["shares"] > 0).all()
    assert etf_rows["target_weight"].round(4).nunique() == 1   # equal weight
    by = orders.groupby("sleeve")["allocated_value"].sum()
    assert by["equity"] <= 70_000 and 15_000 < by["etf"] <= 20_000 and by["crypto"] <= 10_000
    assert by["crypto"] > 9_000                       # fractional coins bought
    assert orders["allocated_value"].sum() <= 100_000
    assert (tmp_path / "pred_crypto.csv").exists() and (tmp_path / "pred.csv").exists()


def test_cli_v44_sector_volume_ragged_and_compare(tmp_path, monkeypatch, capsys):
    stocks = ([f"B{i}.L" for i in range(6)] + [f"T{i}" for i in range(6)]
              + [f"M{i}.SW" for i in range(6)])
    sectors = {t: ("finance" if t[0] == "B" else "technology" if t[0] == "T"
                   else "industrials") for t in stocks}
    (tmp_path / "sectors.csv").write_text(
        "yahoo,sector\n" + "\n".join(f"{t},{s}" for t, s in sectors.items()))

    def fetch(symbols, period):
        closes = _fake_fetcher_7day(symbols, period)
        closes.iloc[:400, 0] = np.nan                   # B0 lists late
        vol = pd.DataFrame(1e5, index=closes.index, columns=closes.columns)
        return closes, vol

    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        tickers: {stocks}
        starting_balance: 100000
        top_n_holdings: 6
        lookback_period: "10y"
        forward_return_days: 21
        rebalance_days: 5
        min_history_days: 300
        model_alpha: 1.0
        random_seed: 42
        cache_dir: "{tmp_path / '.cache'}"
        cache_max_age_hours: 12
        log_level: "WARNING"
        purged_test_window_days: 21
        purged_purge_days: 21
        purged_embargo_days: 5
        purged_min_train_days: 504
        prediction_log_path: "{tmp_path / 'pred.csv'}"
        neutralize_by_sector: true
        sector_file: "sectors.csv"
        use_volume_features: true
        ragged_history: true
    """))
    real = data_mod.get_market_data
    monkeypatch.setattr(data_mod, "get_market_data",
                        lambda **kw: real(**kw, fetcher=fetch))
    out_csv = tmp_path / "alloc.csv"
    rc = cli.main(["--config", str(cfg), "--output", str(out_csv),
                   "--compare-baseline", "--skip-backtest"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Forecast horizon: 21 trading days" in out
    assert "abn_volume_21_252" in out and "amihud_illiq_63" in out
    assert "Head-to-head on identical data" in out
    assert "v4.3 baseline" in out
    orders = pd.read_csv(out_csv, index_col=0)
    assert len(orders) == 6 and orders["allocated_value"].sum() <= 100_000


def test_cli_validate_flag_runs_suite(tmp_path, monkeypatch, capsys):
    stocks = [f"U{i}" for i in range(8)] + [f"K{i}.L" for i in range(8)]
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        tickers: {stocks}
        starting_balance: 100000
        top_n_holdings: 4
        lookback_period: "5y"
        forward_return_days: 5
        rebalance_days: 5
        min_history_days: 300
        model_alpha: 1.0
        random_seed: 42
        cache_dir: "{tmp_path / '.cache'}"
        cache_max_age_hours: 12
        log_level: "WARNING"
        purged_test_window_days: 21
        purged_purge_days: 5
        purged_embargo_days: 5
        purged_min_train_days: 504
        prediction_log_path: "{tmp_path / 'pred.csv'}"
    """))
    real = data_mod.get_market_data
    monkeypatch.setattr(data_mod, "get_market_data",
                        lambda **kw: real(**kw, fetcher=_fake_fetcher))
    rc = cli.main(["--config", str(cfg), "--output", str(tmp_path / "o.csv"),
                   "--skip-backtest", "--validate", "--placebo-runs", "2"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "=== Validation suite: equity sleeve ===" in out
    for check in ("Placebo test", "Decile monotonicity", "Stable over time",
                  "Works in every market", "Survives 1-day execution delay",
                  "Benchmark plausible", "Not dominated"):
        assert check in out


def test_one_failing_sleeve_does_not_stop_the_others(tmp_path, monkeypatch, capsys):
    """v4.8: the real run crashed in the ETF sleeve and produced NO orders."""
    stocks = [f"U{i}" for i in range(8)] + [f"K{i}.L" for i in range(8)]
    etfs = ["IWDA.L", "SGLD.L", "ISF.L"]

    def fetch(symbols, period):
        df = _fake_fetcher_7day(symbols, period)
        for e in etfs:                       # ETFs: no data at all in the last
            df.loc[df.index[-400]:, e] = np.nan   # 400 days -> no features today
        return df
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        tickers: {stocks}
        starting_balance: 100000
        top_n_holdings: 4
        lookback_period: "5y"
        forward_return_days: 5
        rebalance_days: 5
        min_history_days: 300
        model_alpha: 1.0
        random_seed: 42
        cache_dir: "{tmp_path / '.cache'}"
        cache_max_age_hours: 12
        log_level: "WARNING"
        purged_test_window_days: 21
        purged_purge_days: 5
        purged_embargo_days: 5
        purged_min_train_days: 504
        prediction_log_path: "{tmp_path / 'pred.csv'}"
        ragged_history: true
        currency_overrides: {{"IWDA.L": "USD", "SGLD.L": "USD"}}
        sleeves:
          etf: {{weight: 0.2, selection: all, weighting: equal, tickers: {etfs}}}
    """))
    real = data_mod.get_market_data
    monkeypatch.setattr(data_mod, "get_market_data",
                        lambda **kw: real(**kw, fetcher=fetch))
    out_csv = tmp_path / "o.csv"
    rc = cli.main(["--config", str(cfg), "--output", str(out_csv),
                   "--skip-backtest", "--skip-purged-backtest"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "SKIPPED" in out
    orders = pd.read_csv(out_csv, index_col=0)
    assert set(orders["sleeve"]) == {"equity"} and len(orders) == 4


def test_hold_all_sleeve_keeps_stale_priced_instruments(tmp_path, monkeypatch, capsys):
    """v4.8: London ETFs had no fresh close; the hold-all sleeve must still
    hold them (sized on the previous close) and say so."""
    stocks = [f"U{i}" for i in range(8)] + [f"K{i}.L" for i in range(8)]
    etfs = ["IWDA.L", "SGLD.L", "ISF.L"]

    def fetch(symbols, period):
        df = _fake_fetcher_7day(symbols, period)
        last_weekday = df.index[df.index.dayofweek < 5][-1]
        df.loc[last_weekday, etfs] = np.nan       # no close on the last day
        return df
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        tickers: {stocks}
        starting_balance: 100000
        top_n_holdings: 4
        lookback_period: "5y"
        forward_return_days: 5
        rebalance_days: 5
        min_history_days: 300
        model_alpha: 1.0
        random_seed: 42
        cache_dir: "{tmp_path / '.cache'}"
        cache_max_age_hours: 12
        log_level: "WARNING"
        purged_test_window_days: 21
        purged_purge_days: 5
        purged_embargo_days: 5
        purged_min_train_days: 504
        prediction_log_path: "{tmp_path / 'pred.csv'}"
        currency_overrides: {{"IWDA.L": "USD", "SGLD.L": "USD"}}
        sleeves:
          etf: {{weight: 0.2, selection: all, weighting: equal, tickers: {etfs}}}
    """))
    real = data_mod.get_market_data
    monkeypatch.setattr(data_mod, "get_market_data",
                        lambda **kw: real(**kw, fetcher=fetch))
    out_csv = tmp_path / "o.csv"
    rc = cli.main(["--config", str(cfg), "--output", str(out_csv),
                   "--skip-backtest", "--skip-purged-backtest"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "no fresh close yet" in out
    orders = pd.read_csv(out_csv, index_col=0)
    assert set(orders[orders["sleeve"] == "etf"].index) == set(etfs)


def test_cli_report_writes_dashboard_with_monte_carlo(tmp_path, monkeypatch, capsys):
    import json
    import re
    stocks = [f"U{i}" for i in range(8)] + [f"K{i}.L" for i in range(8)]
    crypto = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD"]
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        tickers: {stocks}
        starting_balance: 100000
        top_n_holdings: 4
        lookback_period: "5y"
        forward_return_days: 5
        rebalance_days: 5
        min_history_days: 300
        model_alpha: 1.0
        random_seed: 42
        cache_dir: "{tmp_path / '.cache'}"
        cache_max_age_hours: 12
        log_level: "WARNING"
        purged_test_window_days: 21
        purged_purge_days: 5
        purged_embargo_days: 5
        purged_min_train_days: 504
        prediction_log_path: "{tmp_path / 'pred.csv'}"
        sleeves:
          crypto: {{weight: 0.1, selection: all, weighting: equal, fractional: true,
                    tickers: {crypto}}}
    """))
    real = data_mod.get_market_data
    monkeypatch.setattr(data_mod, "get_market_data",
                        lambda **kw: real(**kw, fetcher=_fake_fetcher_7day))
    report = tmp_path / "report.html"
    rc = cli.main(["--config", str(cfg), "--output", str(tmp_path / "o.csv"),
                   "--skip-backtest", "--report", str(report), "--mc-weeks", "6",
                   "--mc-sims", "500"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Report written to" in out
    html = report.read_text(encoding="utf-8")
    data = json.loads(re.search(r'type="application/json">(.*?)</script>', html,
                                re.DOTALL).group(1))
    assert len(data["orders"]) == 4 + len(crypto)
    assert len(data["mc"]["weeks"]) == 7                  # week 0..6
    assert {s["name"] for s in data["sleeves"]} == {"equity", "crypto"}
    assert data["deciles"] and data["top"]


def test_cli_compare_sector_cap_prints_verdict(tmp_path, monkeypatch, capsys):
    stocks = ([f"B{i}.L" for i in range(8)] + [f"T{i}" for i in range(4)]
              + [f"M{i}.SW" for i in range(4)])
    (tmp_path / "sectors.csv").write_text(
        "yahoo,sector\n" + "\n".join(
            f"{t},{'banks' if t[0] == 'B' else 'tech' if t[0] == 'T' else 'industry'}"
            for t in stocks))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        tickers: {stocks}
        starting_balance: 100000
        top_n_holdings: 6
        lookback_period: "10y"
        forward_return_days: 5
        rebalance_days: 5
        min_history_days: 300
        model_alpha: 1.0
        random_seed: 42
        cache_dir: "{tmp_path / '.cache'}"
        cache_max_age_hours: 12
        log_level: "WARNING"
        purged_test_window_days: 21
        purged_purge_days: 5
        purged_embargo_days: 5
        purged_min_train_days: 504
        prediction_log_path: "{tmp_path / 'pred.csv'}"
        sector_file: "sectors.csv"
    """))
    real = data_mod.get_market_data
    monkeypatch.setattr(data_mod, "get_market_data",
                        lambda **kw: real(**kw, fetcher=_fake_fetcher_7day))
    rc = cli.main(["--config", str(cfg), "--output", str(tmp_path / "a.csv"),
                   "--skip-backtest", "--compare-sector-cap", "2"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Sector cap: head-to-head" in out and "VERDICT:" in out
    assert "no cap" in out and "max 2 per sector" in out


def test_cli_mc_default_is_six_weeks():
    p = cli.build_parser() if hasattr(cli, "build_parser") else None
    if p is None:
        import inspect
        assert "default=6" in inspect.getsource(cli)
    else:
        assert p.parse_args([]).mc_weeks == 6
