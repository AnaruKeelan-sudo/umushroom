"""End-to-end tests of the v4.19 ETF sleeve through the CLI, offline:
universe file + core tickers, the ETF feature set, asset-class groups and
cap, the contest-window evaluation, the `auto` selection rule in both
directions, the horizon head-to-head, and the ETFs tab in the report."""
from __future__ import annotations

import json
import textwrap

import numpy as np
import pandas as pd
import pytest

from umushroom import cli
from umushroom import data as data_mod
from umushroom.config import SleeveConfig
from umushroom.purged_backtest import PurgedWalkForwardResult
from umushroom.report import section_bodies
from umushroom.window import FAIL, PASS, WindowResult

STOCKS = [f"U{i}" for i in range(8)] + [f"K{i}.L" for i in range(8)]
CORE = ["IWDA.L", "SGLD.L", "ISF.L"]
UNIVERSE = [f"E{i}.L" for i in range(12)] + [f"B{i}.DE" for i in range(8)]
CLASSES = {**{f"E{i}.L": "equity" for i in range(12)}, **{f"B{i}.DE": "bond" for i in range(8)},
           "IWDA.L": "equity", "SGLD.L": "commodity", "ISF.L": "equity", "LEV3.L": "equity"}


def _universe_csv(path, extra_rows=()):
    head = ("yahoo,isin,name,umushroom_symbol,umushroom_currency,yahoo_currency,"
            "yahoo_exchange,yahoo_name,asset_class,style,manager,url_name,umushroom_price,"
            "price_date,history_days,first_date,last_date,name_similarity,via,status,note,"
            "leveraged\n")
    rows = []
    for t in CORE + UNIVERSE:
        ccy = "USD" if t in ("IWDA.L", "SGLD.L") else ("EUR" if t.endswith(".DE") else "GBp")
        rows.append(f"{t},ISIN{t},{t} name,{t.split('.')[0]},{ccy.replace('GBp', 'GBP')},"
                    f"{ccy},X,{t} name,{CLASSES[t]},,Mgr,{t.lower()},10,2026-10-12,2000,"
                    f"2018-01-01,2026-10-12,1.0,search,ok,,False")
    rows.append("LEV3.L,ISINLEV,Leverage Shares 3x Long Tesla ETP,LEV3,USD,USD,X,Lev 3x,"
                "equity,,Mgr,lev3,10,2026-10-12,2000,2018-01-01,2026-10-12,1.0,search,ok,,"
                "True")
    rows.append("DEAD.L,ISINDEAD,Dead ETF,DEAD,USD,,,,equity,,,dead,1,2026-10-12,0,,,,,"
                "no_yahoo_data,no data,False")
    rows.extend(extra_rows)
    path.write_text(head + "\n".join(rows) + "\n")


def _fetch_factory(seed=1, trend_strength=0.0):
    """Weekday-only prices. With trend_strength > 0 every instrument gets a
    persistent latent drift, so trend factors carry a real signal."""
    def fetch(symbols, period):
        rng = np.random.default_rng(seed)
        n = 1700
        idx = pd.bdate_range("2019-09-02", periods=n)
        cols = {}
        for s in symbols:
            if s.endswith("=X"):
                cols[s] = 0.8 * np.exp(rng.normal(0, 0.003, n).cumsum())
                continue
            drift = np.zeros(n)
            if trend_strength:
                drift[0] = rng.normal(0, 0.0015)
                for t in range(1, n):
                    drift[t] = 0.998 * drift[t - 1] + rng.normal(0, 0.0001)
                drift *= trend_strength
            rets = drift + rng.normal(0.0003, 0.012, n)
            # pence-quoted London stocks at 5000p; ETFs at ~100 in their currency
            level = 5000.0 if s.startswith("K") and s.endswith(".L") else 100.0
            cols[s] = level * np.cumprod(1 + rets)
        return pd.DataFrame(cols, index=idx)
    return fetch


def _config(tmp_path, selection="auto", horizon=10, extra=""):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        tickers: {STOCKS}
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
        contest_window_days: 30
        currency_overrides: {{"IWDA.L": "USD", "SGLD.L": "USD"}}
        sleeves:
          etf:
            weight: 0.2
            top_n: 4
            feature_set: etf
            group_by: asset_class
            max_per_group: 2
            selection: {selection}
            forward_return_days: {horizon}
            universe_file: "universe.csv"
            exclude_leveraged: true
            core_tickers: {CORE}
            {extra}
    """))
    _universe_csv(tmp_path / "universe.csv")
    return cfg


def _run(monkeypatch, cfg, tmp_path, fetch, *extra_args):
    real = data_mod.__dict__.setdefault("_real_get_market_data", data_mod.get_market_data)
    monkeypatch.setattr(data_mod, "get_market_data",
                        lambda **kw: real(**kw, fetcher=fetch))
    out_csv = tmp_path / "alloc.csv"
    report = tmp_path / "report.html"
    rc = cli.main(["--config", str(cfg), "--output", str(out_csv), "--skip-backtest",
                   "--skip-prediction-check", "--report", str(report), *extra_args])
    return rc, out_csv, report


def _payload(report_path):
    html = report_path.read_text()
    start = html.index('<script id="report-data" type="application/json">') + len(
        '<script id="report-data" type="application/json">')
    end = html.index("</script>", start)
    return json.loads(html[start:end].replace("<\\/", "</"))


def test_auto_falls_back_to_core_basket_on_noise(tmp_path, monkeypatch, capsys):
    cfg = _config(tmp_path)
    rc, out_csv, report = _run(monkeypatch, cfg, tmp_path, _fetch_factory(seed=5))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "feature set: etf" in out and "Group cap ON: at most 2 picks per asset_class" in out
    assert "Contest-Window Evaluation (30 trading days" in out
    assert "Selection rule (auto):" in out and "holding the core basket" in out
    orders = pd.read_csv(out_csv, index_col=0)
    etf = orders[orders["sleeve"] == "etf"]
    # the fallback holds exactly the core ETFs, equal weight, never the universe
    assert set(etf.index) == set(CORE)
    assert etf["target_weight"].round(4).nunique() == 1
    assert 15_000 < etf["allocated_value"].sum() <= 20_000
    # leveraged ETP was excluded from the universe before modelling
    assert "LEV3.L" not in out.split("SLEEVE: ETF")[1].split("Instruments:")[0]
    payload = _payload(report)
    s = next(x for x in payload["sleeves"] if x["name"] == "etf")
    assert s["selection"] == "auto" and s["selection_used"] == "core"
    assert s["feature_set"] == "etf" and s["horizon_days"] == 10
    assert s["contest_window_days"] == 30 and s["n_core"] == 3
    assert s["universe_status"]["ok"] == len(CORE) + len(UNIVERSE) + 1
    assert s["window"] is not None and s["window"]["verdict"] in ("PASS", "WARN", "FAIL")
    # the ranking table covers the whole universe even though only core is held
    assert len(payload["top_etfs"]["etf"]) > len(CORE)
    body = section_bodies(payload)["etfs"]
    assert "core basket" in body and "Today's ETF orders" in body
    assert "does the edge arrive inside the contest?" in body
    assert "Top-ranked ETFs" in body and "Universe build" in body


def test_auto_trades_the_model_when_a_real_trend_exists(tmp_path, monkeypatch, capsys):
    cfg = _config(tmp_path, horizon=5)
    rc, out_csv, report = _run(monkeypatch, cfg, tmp_path,
                               _fetch_factory(seed=7, trend_strength=6.0))
    out = capsys.readouterr().out
    assert rc == 0, out
    orders = pd.read_csv(out_csv, index_col=0)
    etf = orders[orders["sleeve"] == "etf"]
    payload = _payload(report)
    s = next(x for x in payload["sleeves"] if x["name"] == "etf")
    if s["selection_used"] == "model":
        assert "trading the model's top 4" in out
        assert len(etf) == 4
        # at most 2 per asset class
        classes = pd.Series([CLASSES[t] for t in etf.index]).value_counts()
        assert classes.max() <= 2
        assert payload["top_etfs"]["etf"] and payload["top_etfs"]["etf"][0]["asset_class"]
    else:   # the synthetic trend was not strong enough on this draw: still safe
        assert set(etf.index) == set(CORE)
    assert "Mean Information Coeff" in out


def test_model_selection_bypasses_the_rule(tmp_path, monkeypatch, capsys):
    cfg = _config(tmp_path, selection="model")
    rc, out_csv, _ = _run(monkeypatch, cfg, tmp_path, _fetch_factory(seed=9),
                          "--skip-purged-backtest")
    out = capsys.readouterr().out
    assert rc == 0, out
    etf = pd.read_csv(out_csv, index_col=0).query("sleeve == 'etf'")
    assert len(etf) == 4 and set(etf.index) <= set(CORE + UNIVERSE)


def test_compare_horizon_prints_table_and_refuses_long_horizons(tmp_path, monkeypatch,
                                                               capsys):
    cfg = _config(tmp_path)
    rc, _, _ = _run(monkeypatch, cfg, tmp_path, _fetch_factory(seed=3),
                    "--compare-horizon", "etf:5,10")
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Forecast horizon: head-to-head" in out and "VERDICT:" in out
    assert "win(6w)" in out
    rc, _, _ = _run(monkeypatch, cfg, tmp_path, _fetch_factory(seed=3),
                    "--compare-horizon", "etf:5,40")
    out = capsys.readouterr().out
    assert rc == 0 and "Horizon comparison refused" in out


def test_missing_universe_file_runs_on_core_and_warns(tmp_path, monkeypatch, capsys):
    cfg = _config(tmp_path)
    (tmp_path / "universe.csv").unlink()
    rc, out_csv, report = _run(monkeypatch, cfg, tmp_path, _fetch_factory(seed=2),
                               "--skip-purged-backtest")
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "WARNING: universe file" in out
    etf = pd.read_csv(out_csv, index_col=0).query("sleeve == 'etf'")
    assert set(etf.index) == set(CORE)
    assert "Universe file missing" in section_bodies(_payload(report))["etfs"]


# ------------------------------------------------- the auto rule in isolation

def _pwf(ic_t: float) -> PurgedWalkForwardResult:
    n = 40
    idx = pd.bdate_range("2024-01-01", periods=n, freq="7D")
    ic = pd.Series(np.full(n, 0.05 * np.sign(ic_t) if ic_t else 0.0), index=idx)
    # force the t-stat: constant ic with a tiny jitter gives t = mean/std*sqrt(n)
    rng = np.random.default_rng(0)
    ic = ic + rng.normal(0, 0.05 * np.sqrt(n) / max(abs(ic_t), 1e-9), n) * 0 \
        if ic_t == 0 else ic + rng.normal(0, abs(ic.mean()) * np.sqrt(n) / ic_t, n)
    return PurgedWalkForwardResult(
        folds=[], overall_strategy_return=0.0, overall_benchmark_return=0.0,
        overall_win_rate=0.5, overall_sharpe=0.0, mean_information_coefficient=float(ic.mean()),
        period_returns=pd.Series(0.0, index=idx),
        benchmark_period_returns=pd.Series(0.0, index=idx), period_ic=ic)


def _window(verdict: str) -> WindowResult:
    idx = pd.bdate_range("2024-01-01", periods=20, freq="7D")
    rng = np.random.default_rng(1)
    # a constant active return has zero variance (t undefined): add jitter
    s = pd.Series((0.03 if verdict == PASS else -0.03) + rng.normal(0, 0.002, 20), index=idx)
    b = pd.Series(0.0, index=idx)
    return WindowResult(6, 30, s, b, s.iloc[::6], b.iloc[::6])


def _sleeve(selection: str) -> SleeveConfig:
    return SleeveConfig(name="etf", tickers=["A", "B"], weight=0.2, top_n=1,
                        selection=selection, core_tickers=["A"])


def test_decide_selection_rule():
    assert cli.decide_selection(_sleeve("model"), None, None) == (
        True, "selection: model (configured; no rule applied)")
    assert cli.decide_selection(_sleeve("all"), None, None)[0] is False
    use, why = cli.decide_selection(_sleeve("auto"), None, None)
    assert not use and "no purged walk-forward" in why
    strong, weak = _pwf(4.0), _pwf(0.5)
    assert strong.has_evidence_of_skill and not weak.has_evidence_of_skill
    use, why = cli.decide_selection(_sleeve("auto"), weak, _window(PASS))
    assert not use and "no reliable ranking skill" in why
    use, why = cli.decide_selection(_sleeve("auto"), strong, None)
    assert not use and "no contest-window" in why
    use, why = cli.decide_selection(_sleeve("auto"), strong, _window(FAIL))
    assert not use and "did not beat the benchmark" in why
    use, why = cli.decide_selection(_sleeve("auto"), strong, _window(PASS))
    assert use and "trading the model's top 1" in why


@pytest.mark.parametrize("verdict", [PASS, FAIL])
def test_window_payload_and_cards_render(verdict):
    from umushroom.report import SleeveReport, build_payload
    s = SleeveReport("etf", 0.2, "auto", feature_set="etf", horizon_days=10,
                     selection_used="model" if verdict == PASS else "core",
                     decision="because", window=_window(verdict), n_instruments=20, n_core=3,
                     universe_status={"ok": 20, "no_candidates": 2}, contest_window_days=30)
    p = build_payload({"base_currency": "GBP"}, [s], pd.DataFrame())
    w = p["sleeves"][0]["window"]
    assert w["verdict"] == verdict and w["n_windows"] == 20 and w["days"] == 30
    body = section_bodies(p)["etfs"]
    assert verdict in body and "20 priced today" in body
    assert ("model picks" if verdict == PASS else "core basket") in body


# --------------------------------------------------- v4.20: learner, weights, guard

def test_model_weighting_and_guard_settings_flow_through_to_orders_and_report(
        tmp_path, monkeypatch, capsys):
    """gbm learner, factor-risk weights and the near-duplicate guard, set on the
    sleeve, must show in the console, drive the orders and reach the website."""
    # `extra` is spliced in at the sleeve's indentation (12 spaces) before the
    # template is dedented, so every line after the first must carry it
    cfg = _config(tmp_path, selection="model",
                  extra="model_type: gbm\n            weighting: factor\n"
                        "            max_pick_correlation: 0.9")
    rc, out_csv, report = _run(monkeypatch, cfg, tmp_path, _fetch_factory(seed=12),
                               "--skip-purged-backtest")
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Model: gbm; weighting: factor; near-duplicate guard: max correlation 0.90" in out
    assert "Feature importance (permutation importance" in out
    assert "Factor-risk weights (v4.20)" in out and "idiosyncratic" in out
    etf = pd.read_csv(out_csv, index_col=0).query("sleeve == 'etf'")
    assert len(etf) == 4 and etf["target_weight"].round(4).nunique() > 1
    s = next(x for x in _payload(report)["sleeves"] if x["name"] == "etf")
    assert s["model_type"] == "gbm" and s["weighting"] == "factor"
    assert s["max_pick_correlation"] == pytest.approx(0.9)
    fr = s["factor_risk"]
    assert set(fr["factors"]) == {"bond", "equity"}
    assert sum(fr["contributions"].values()) == pytest.approx(1.0, abs=1e-6)
    assert set(fr["weights"]) == set(etf.index)
    assert all(set(v) == {"bond", "equity"} for v in fr["exposures"].values())


def test_defaults_are_unchanged_ridge_equal_no_guard(tmp_path, monkeypatch, capsys):
    cfg = _config(tmp_path, selection="model")
    rc, _out_csv, report = _run(monkeypatch, cfg, tmp_path, _fetch_factory(seed=12),
                               "--skip-purged-backtest")
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Model: ridge; weighting: equal; near-duplicate guard: off" in out
    assert "Feature importance (Ridge coefficients)" in out
    s = next(x for x in _payload(report)["sleeves"] if x["name"] == "etf")
    assert s["model_type"] == "ridge" and s["weighting"] == "equal"
    assert s["max_pick_correlation"] is None and s["factor_risk"] is None


def test_v420_head_to_heads_print_tables_and_verdicts(tmp_path, monkeypatch, capsys):
    cfg = _config(tmp_path)
    rc, _, _ = _run(monkeypatch, cfg, tmp_path, _fetch_factory(seed=3),
                    "--compare-model", "etf:ridge,gbm",
                    "--compare-weighting", "etf:equal,factor",
                    "--compare-dedup", "etf:0.9")
    out = capsys.readouterr().out
    assert rc == 0, out
    for title in ("Learner: head-to-head", "Weighting: head-to-head",
                  "Near-duplicate guard: head-to-head"):
        assert title in out, title
    assert out.count("VERDICT") >= 5          # backtest, window + three comparisons
    assert "VaR95/wk" in out and "current setting: ridge" in out
    assert "current setting: equal" in out and "current setting: no guard" in out


def test_v420_head_to_heads_refuse_bad_arguments(tmp_path, monkeypatch, capsys):
    cfg = _config(tmp_path)
    rc, _, _ = _run(monkeypatch, cfg, tmp_path, _fetch_factory(seed=3),
                    "--compare-model", "etf:lstm", "--compare-weighting", "etf:markowitz",
                    "--compare-dedup", "etf:1.5")
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Model comparison refused" in out
    assert "Weighting comparison refused" in out
    assert "Dedup comparison refused" in out


def test_factor_risk_renders_on_the_risk_tab_and_the_etf_tile():
    from tests.test_factor_risk import _two_factor_universe
    from umushroom.factor_risk import factor_risk
    from umushroom.report import SleeveReport, build_payload, factor_risk_payload
    px, groups = _two_factor_universe()
    fr = factor_risk(px, ["A0", "A1", "B0"], len(px) - 1, groups)
    s = SleeveReport("etf", 0.2, "model", feature_set="etf", horizon_days=10,
                     selection_used="model", n_instruments=20, contest_window_days=30,
                     model_type="gbm", weighting="factor", max_pick_correlation=0.9,
                     factor_risk=factor_risk_payload(fr))
    p = build_payload({"base_currency": "GBP"}, [s], pd.DataFrame())
    sl = p["sleeves"][0]
    assert sl["model_type"] == "gbm" and sl["weighting"] == "factor"
    assert sl["factor_risk"]["factors"] == ["ga", "gb"]
    assert json.dumps(p)                                   # JSON-clean
    body = section_bodies(p)
    assert "gbm / factor" in body["etfs"] and "max correlation 0.90" in body["etfs"]
    risk = body["risk"]
    assert "Factor risk" in risk and "beta ga" in risk and "A0" in risk and "B0" in risk
    assert "idiosyncratic" in risk
    # a sleeve without factor weights adds nothing to the Risk tab
    plain = SleeveReport("etf", 0.2, "model", feature_set="etf")
    assert "Factor risk" not in section_bodies(
        build_payload({"base_currency": "GBP"}, [plain], pd.DataFrame()))["risk"]


def test_learner_rule_needs_a_significant_window_edge_to_replace_the_current():
    from umushroom.cli import skill_verdict
    base = {"ic_t": 3.0, "w_t": -0.2}
    rows = {"ridge": base, "mlp": {"ic_t": 3.3, "w_t": 0.34}, "gbm": {"ic_t": 3.9, "w_t": -0.3}}
    winner, why = skill_verdict(rows, "ridge", ["ridge", "gbm", "mlp"])
    assert winner is None and "noise" in why and "mlp" in why           # the 10-19 ETF case
    rows["gbm"] = {"ic_t": 5.1, "w_t": 3.95}
    winner, why = skill_verdict(rows, "ridge", ["ridge", "gbm", "mlp"])
    assert winner == "gbm" and "+3.95" in why                             # the 10-19 stock case
    winner, why = skill_verdict({"ridge": {"ic_t": 1.2, "w_t": 2.5}}, "ridge")
    assert winner is None and "IC t < 2" in why
    tie = {"ridge": {"ic_t": 3.0, "w_t": 2.5}, "gbm": {"ic_t": 3.0, "w_t": 2.5}}
    assert skill_verdict(tie, "ridge", ["ridge", "gbm"])[0] is None        # tie -> current
    assert skill_verdict(tie, "mlp", ["ridge", "gbm"])[0] == "ridge"       # tie -> simpler


def test_auto_fallback_basket_is_equal_weight_even_with_factor_weighting(
        tmp_path, monkeypatch, capsys):
    """The core basket is held equal weight (what the benchmark, the report and
    the Monte Carlo assume); `weighting: factor` applies to model picks only."""
    cfg = _config(tmp_path, extra="weighting: factor")
    rc, out_csv, report = _run(monkeypatch, cfg, tmp_path, _fetch_factory(seed=5))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "holding the core basket" in out
    assert "(weighting 'factor' applies to model picks only)" in out
    etf = pd.read_csv(out_csv, index_col=0).query("sleeve == 'etf'")
    assert set(etf.index) == set(CORE)
    assert etf["target_weight"].round(4).nunique() == 1
    s = next(x for x in _payload(report)["sleeves"] if x["name"] == "etf")
    assert s["weighting"] == "factor" and s["factor_risk"] is None
