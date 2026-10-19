"""Command-line entrypoint.

Orchestrates the full pipeline (data -> features -> train -> validate
-> backtest -> allocate) and is the only module that calls sys.exit.
Everything it calls raises typed exceptions on failure rather than
exiting directly, which keeps the rest of the package usable as a
library (e.g. from a notebook) without side effects.
"""

from __future__ import annotations

import argparse
import functools
import math
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from umushroom import data as data_mod
from umushroom import features as features_mod
from umushroom.allocation import Allocation, WeightingScheme, generate_allocation
from umushroom.backtest import run_backtest
from umushroom.config import (
    EQUITY_SLEEVE,
    WEIGHTINGS,
    SleeveConfig,
    StrategyConfig,
    load_config,
    load_sectors,
    sleeve_horizon,
    sleeve_model_type,
    sleeve_pick_correlation,
)
from umushroom.exceptions import DataError, InsufficientHistoryError, UMushroomError
from umushroom.factor_regression import (
    benchmark_alpha,
    fama_french_alpha,
    fetch_french_factors,
)
from umushroom.factor_risk import factor_weight
from umushroom.features import hierarchical_groups
from umushroom.logging_config import configure_logging, get_logger
from umushroom.model import MODEL_TYPES, RankingModel
from umushroom.montecarlo import combine_sleeves, simulate
from umushroom.portfolio_construction import hrp_weight
from umushroom.prediction_log import evaluate_latest_predictions, log_predictions
from umushroom.purged_backtest import (
    PurgedWalkForwardResult,
    WeightFn,
    run_purged_walk_forward,
)
from umushroom.report import (
    SleeveReport,
    build_payload,
    factor_risk_payload,
    stock_snapshot,
    write_report,
)
from umushroom.risk import var_es
from umushroom.robustness import bootstrap_sharpe
from umushroom.selection import (
    ALPHA_RETENTION,
    CapVerdict,
    capped_top,
    decide_cap,
    top_sector_share,
)
from umushroom.universe import check_concentration
from umushroom.validation import ValidationReport, decile_returns, run_validation
from umushroom.window import (
    FAIL,
    WINDOW_T_THRESHOLD,
    WindowResult,
    contest_windows,
    window_periods_for,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="umushroom",
        description="Factor-model portfolio engine for the UMushroom "
                     "investment contest.",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Bypass the data cache and re-fetch from the network.",
    )
    parser.add_argument(
        "--skip-backtest", action="store_true",
        help="Skip the simple historical backtest and only produce today's allocation.",
    )
    parser.add_argument(
        "--skip-purged-backtest", action="store_true",
        help="Skip the (slower) purged & embargoed walk-forward backtest.",
    )
    parser.add_argument(
        "--skip-prediction-check", action="store_true",
        help="Skip evaluating the previous run's prediction accuracy.",
    )
    parser.add_argument(
        "--weighting", choices=list(WEIGHTINGS), default="equal",
        help="Portfolio weighting scheme for sleeves without their own: "
             "'equal' (default, 1/N), 'hrp' (Hierarchical Risk Parity) or "
             "'factor' (v4.20 factor-risk parity, factor_risk.py). The walk-forward "
             "backtest scores the same scheme.",
    )
    parser.add_argument(
        "--fama-french", action="store_true",
        help="Also regress out-of-sample returns on Fama-French factors "
             "(downloads factor data from Ken French's library).",
    )
    parser.add_argument(
        "--compare-sector-cap", type=int, metavar="N", default=None,
        help="v4.18: head-to-head backtest of the equity sleeve with at most N "
             "picks per sector vs. no cap, with a pre-declared decision rule")
    parser.add_argument(
        "--compare-horizon", metavar="SLEEVE:DAYS,DAYS,...", default=None,
        help="v4.19: head-to-head of forecast horizons for one sleeve on identical "
             "data, e.g. etf:5,10,21, judged by the contest-window rule declared in "
             "window.py. Every horizon must fit inside contest_window_days.")
    parser.add_argument(
        "--compare-model", metavar="SLEEVE:TYPE,TYPE,...", default=None,
        help="v4.20: head-to-head of learners (ridge, mlp, gbm) for one sleeve on "
             "identical data, e.g. etf:ridge,mlp,gbm, judged by the contest-window "
             "rule; ties go to the simpler model.")
    parser.add_argument(
        "--compare-weighting", metavar="SLEEVE:SCHEME,SCHEME,...", default=None,
        help="v4.20: head-to-head of weighting schemes (equal, hrp, factor) for one "
             "sleeve, e.g. etf:equal,hrp,factor. Rule: adopt a scheme only if it keeps "
             ">= 75%% of equal weight's alpha AND does not raise the weekly 95%% VaR.")
    parser.add_argument(
        "--compare-dedup", metavar="SLEEVE:CORR,CORR,...", default=None,
        help="v4.20: head-to-head of the near-duplicate guard, e.g. etf:0.9,0.95 "
             "(max trailing return correlation between two picks) vs. no guard. "
             "Same rule as --compare-weighting.")
    parser.add_argument(
        "--compare-baseline", action="store_true",
        help="Also run the v4.3 equity setup on the same data and print a "
             "head-to-head table (one pre-declared comparison).",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run the validation suite (placebo test, deciles, per-year, "
             "per-market, execution delay, data sanity, concentration) for "
             "each model-driven sleeve. Does not change the model.",
    )
    parser.add_argument(
        "--placebo-runs", type=int, default=20,
        help="Shuffled-signal runs for the placebo test (default 20).",
    )
    parser.add_argument(
        "--report", nargs="?", const="umushroom_report.html", default=None,
        metavar="PATH",
        help="Write an HTML dashboard (picks, alpha charts, Monte Carlo) to PATH "
             "(default umushroom_report.html).",
    )
    parser.add_argument(
        "--mc-weeks", type=int, default=6,
        help="Contest length in weeks for the Monte Carlo simulation "
             "(default 6 = UMushroom UK Stage 1).",
    )
    parser.add_argument(
        "--mc-sims", type=int, default=5000,
        help="Number of Monte Carlo simulations (default 5000).",
    )
    parser.add_argument(
        "--output", default="umushroom_allocation.csv",
        help="Where to write the resulting allocation CSV.",
    )
    parser.add_argument(
        "--model-out", default=None,
        help="Optional path to persist the fitted model artifact.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except UMushroomError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 2

    configure_logging(cfg.log_level)
    logger = get_logger("cli")

    try:
        logger.info("=== UMushroom pipeline starting ===")

        market = data_mod.get_market_data(
            tickers=cfg.all_tickers,
            period=cfg.lookback_period,
            cache_dir=cfg.cache_dir,
            cache_max_age_hours=cfg.cache_max_age_hours,
            min_history_days=cfg.min_history_days,
            base_currency=cfg.base_currency,
            currency_overrides=cfg.currency_overrides,
            force_refresh=args.force_refresh,
            drop_today_bar=cfg.drop_incomplete_bar,
            calendar_tickers=cfg.tickers,
            ragged=cfg.ragged_history,
            yahoo_symbols=cfg.yahoo_symbols,
            max_missing_fraction=cfg.max_missing_fraction,
        )
        prices = market.prices
        print(f"\nData: {prices.shape[1]} instruments, {len(prices)} trading days "
              f"({prices.index[0].date()} .. {prices.index[-1].date()}), "
              f"prices in {cfg.base_currency}")

        allocations: list[tuple[SleeveConfig, Allocation]] = []
        collected: dict[str, SleeveReport] = {}
        for sleeve in cfg.all_sleeves:
            try:
                alloc = _run_sleeve(cfg, args, sleeve, market, collected)
            except (DataError, ValueError) as e:
                # v4.8: one sleeve's problem must not cost the others' orders
                logger.error(f"Sleeve '{sleeve.name}' failed: {e}")
                print(f"\nSLEEVE {sleeve.name.upper()} SKIPPED: {e}\n"
                      f"Its {sleeve.weight:.0%} stays in cash this run.")
                continue
            if alloc is not None:
                allocations.append((sleeve, alloc))
        if not allocations:
            raise DataError("No sleeve produced an allocation")

        frames = []
        for sleeve, alloc in allocations:
            f = alloc.orders.copy()
            f.insert(0, "sleeve", sleeve.name)
            frames.append(f)
        combined = pd.concat(frames)
        if cfg.yahoo_symbols:
            combined["priced_from"] = [cfg.yahoo_symbols.get(t, "") for t in
                                       combined.index]
        # whole shares print as 194, fractional coins as 0.753508
        combined["shares"] = [int(v) if float(v).is_integer() else float(v)
                              for v in combined["shares"]]
        deployed = float(combined["allocated_value"].sum())
        if deployed > cfg.starting_balance + 1e-6:
            raise DataError(f"Combined orders overspend: {deployed:,.2f} > "
                            f"{cfg.starting_balance:,.2f}")

        print(f"\n=== Today's Orders: all sleeves (enter into UMushroom; 'price' in "
              f"{cfg.base_currency}, 'local_price' in the listing currency -- "
              f"compare it with UMushroom's price) ===")
        print(combined.to_string())
        for sleeve, alloc in allocations:
            print(f"  {sleeve.name:<8} {alloc.total_deployed:>12,.2f} of "
                  f"{alloc.balance:>12,.2f}")
        print(f"\nDeployed: {deployed:,.2f} / {cfg.starting_balance:,.2f}  "
              f"(cash remaining: {cfg.starting_balance - deployed:,.2f})")
        combined.to_csv(args.output)

        if args.report:
            path = _write_report(cfg, args, market, combined, allocations, collected)
            print(f"\nReport written to {path.resolve()} -- open it in your browser.")

        logger.info("=== Pipeline completed successfully ===")
        return 0

    except UMushroomError as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # last-resort guard: never crash with a raw traceback
        logger.exception("Unexpected error")
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 1


MIN_LOCAL_PRICE = 0.10


def _write_report(cfg: StrategyConfig, args: argparse.Namespace,
                  market: data_mod.MarketData, combined: pd.DataFrame,
                  allocations: list[tuple[SleeveConfig, Allocation]],
                  collected: dict[str, SleeveReport]) -> Path:
    import csv
    from datetime import datetime

    from umushroom import __version__
    meta_rows: dict[str, dict[str, str]] = {}
    meta_file = cfg.sector_file
    if meta_file is not None and Path(meta_file).exists():
        with open(meta_file, newline="") as fh:
            meta_rows = {r["yahoo"]: r for r in csv.DictReader(fh) if r.get("yahoo")}

    def meta(t: str, key: str) -> str:
        return meta_rows.get(t, {}).get(key, "")

    etf_meta: dict[str, dict[str, str]] = {}
    for sl in cfg.sleeves:
        etf_meta.update(sl.meta)

    def imeta(t: str, key: str) -> str:
        return etf_meta.get(t, {}).get(key, "") or meta(t, key)

    orders = combined.copy()
    snap = stock_snapshot(market.prices, list(orders.index))
    orders = orders.join(snap)
    orders["name"] = [imeta(t, "name") for t in orders.index]
    orders["sector"] = [(etf_meta.get(t, {}).get("asset_class") or meta(t, "sector"))
                        .replace("-", " ") for t in orders.index]
    orders["umushroom_symbol"] = [etf_meta.get(t, {}).get("umushroom_symbol", "")
                                 for t in orders.index]
    orders["umushroom_currency"] = [etf_meta.get(t, {}).get("umushroom_currency", "")
                                   for t in orders.index]
    orders["portfolio_weight"] = orders["allocated_value"] / cfg.starting_balance

    top = pd.DataFrame()
    eq_alloc = next((a for s, a in allocations if s.name == EQUITY_SLEEVE), None)
    if eq_alloc is not None:
        scores = eq_alloc.all_scores.sort_values(ascending=False)
        rank_pct = scores.rank(pct=True)
        best = scores.head(25)
        top = pd.DataFrame({"rank": range(1, len(best) + 1),
                            "rank_pct": rank_pct[best.index]}, index=best.index)
        top["name"] = [meta(t, "name") for t in top.index]
        top["sector"] = [meta(t, "sector").replace("-", " ") for t in top.index]
        top["exchange"] = [market.listings[t].exchange for t in top.index]
        top["held"] = [1 if t in eq_alloc.orders.index else 0 for t in top.index]
        top = top.join(stock_snapshot(market.prices, list(top.index)))

    # v4.19: the ETF sleeve's own ranking table (top 25), with asset class
    top_etfs: dict[str, pd.DataFrame] = {}
    for sl, al in allocations:
        if sl.name == EQUITY_SLEEVE or sl.feature_set != "etf":
            continue
        info_s = collected.get(sl.name)
        full = info_s.scores if info_s is not None and info_s.scores is not None else None
        scores = (full if full is not None else al.all_scores).sort_values(ascending=False)
        rank_pct = scores.rank(pct=True)
        best = scores.head(25)
        tb = pd.DataFrame({"rank": range(1, len(best) + 1),
                           "rank_pct": rank_pct[best.index]}, index=best.index)
        tb["name"] = [imeta(t, "name") for t in tb.index]
        tb["asset_class"] = [etf_meta.get(t, {}).get("asset_class", "") for t in tb.index]
        tb["umushroom_symbol"] = [etf_meta.get(t, {}).get("umushroom_symbol", "")
                                 for t in tb.index]
        tb["currency"] = [market.listings[t].currency for t in tb.index]
        tb["held"] = [1 if t in al.orders.index else 0 for t in tb.index]
        top_etfs[sl.name] = tb.join(stock_snapshot(market.prices, list(tb.index)))

    deciles = None
    eq_info = collected.get(EQUITY_SLEEVE)
    if eq_info is not None and eq_info.backtest is not None:
        deciles = decile_returns(eq_info.backtest)

    mc = None
    mc_note = ""
    held = [(s, collected.get(s.name)) for s, _ in allocations]
    if held and all(i is not None and i.backtest is not None for _, i in held):
        strat_parts, bench_parts = [], []
        for s, i in held:
            assert i is not None and i.backtest is not None
            r = i.backtest
            # model sleeves earn the model's picks; hold-all sleeves earn the
            # equal-weight basket, which is exactly the benchmark series
            strat_parts.append((s.weight, r.period_returns
                                if i.selection_used == "model"
                                else r.benchmark_period_returns))
            bench_parts.append((s.weight, r.benchmark_period_returns))
        try:
            mc = simulate(combine_sleeves(strat_parts), combine_sleeves(bench_parts),
                          horizon_weeks=args.mc_weeks,
                          start_value=cfg.starting_balance, n_paths=args.mc_sims,
                          seed=cfg.random_seed)
            mc_note = ("Whole portfolio, all sleeves at their target weights; the "
                       "benchmark replaces the stock picks with the equal-weight "
                       "stock universe.")
        except ValueError as e:
            print(f"Monte Carlo skipped: {e}")
    else:
        print("Monte Carlo skipped: needs the purged backtest for every sleeve "
              "(don't use --skip-purged-backtest with --report).")

    sleeves = [collected[s.name] for s in cfg.all_sleeves if s.name in collected]
    payload = build_payload(
        meta={"as_of": str(market.prices.index[-1].date()),
              "generated": datetime.now(data_mod.LOCAL_TZ).strftime("%Y-%m-%d %H:%M"),
              "base_currency": cfg.base_currency, "version": __version__,
              "rebalance_days": cfg.rebalance_days,
              "contest_window_days": cfg.contest_window_days},
        sleeves=sleeves, orders=orders, top_ranked=top, deciles=deciles,
        mc=mc, mc_note=mc_note, top_etfs=top_etfs)
    return write_report(args.report, payload)


def _sleeve_log_path(cfg: StrategyConfig, sleeve: SleeveConfig) -> Path:
    p = cfg.prediction_log_path
    if sleeve.name == EQUITY_SLEEVE:
        return p        # unchanged location for the main stock sleeve
    return p.with_name(f"{p.stem}_{sleeve.name}{p.suffix}")


def _run_sleeve(cfg: StrategyConfig, args: argparse.Namespace, sleeve: SleeveConfig,
                market: data_mod.MarketData,
                collect: dict[str, SleeveReport] | None = None) -> Allocation | None:
    """Full research + allocation pipeline for ONE sleeve, on its own
    instruments only: its own model, CV, backtests, skill verdict, alpha
    and orders (sized to its share of the balance)."""
    picks = "all" if sleeve.selection == "all" else f"top {sleeve.top_n}"
    print(f"\n{'#' * 72}\n# SLEEVE: {sleeve.name.upper()}  "
          f"(weight {sleeve.weight:.0%}, {picks})\n{'#' * 72}")
    cols = [t for t in sleeve.tickers if t in market.prices.columns]
    dropped = sorted(set(sleeve.tickers) - set(cols))
    if dropped:
        print(f"Not available after data validation: {dropped}")
    if len(cols) < max(sleeve.top_n, 3):
        print(f"SKIPPED: only {len(cols)} usable instrument(s); need at least "
              f"{max(sleeve.top_n, 3)}. Its {sleeve.weight:.0%} stays in cash.")
        return None
    prices = market.prices[cols]
    listings = {t: market.listings[t] for t in cols}
    groups = _sleeve_groups(cfg, sleeve, listings)
    dvol = (market.dollar_volume[cols]
            if sleeve.use_volume_features and market.dollar_volume is not None
            else None)
    if sleeve.use_volume_features and dvol is None:
        print("NOTE: no volume data available; volume factors skipped.")
    sectors = (load_sectors(cfg.sector_file)
               if cfg.sector_file is not None and Path(cfg.sector_file).exists()
               else None)
    # v4.19: a per-group cap (e.g. at most 3 bond ETFs) reuses the sector-cap
    # machinery with the sleeve's own groups
    cap_groups, cap_n = sectors, sleeve.max_per_sector
    if sleeve.max_per_group and sleeve.groups:
        cap_groups = {t: sleeve.groups[t] for t in cols if t in sleeve.groups}
        cap_n = sleeve.max_per_group
    if sleeve.name == EQUITY_SLEEVE:
        for w in check_concentration(listings, cfg.max_single_exchange_fraction):
            print(f"WARNING: {w}")
        if sleeve.max_per_sector:
            print(f"Sector cap ON: at most {sleeve.max_per_sector} picks per sector")
    elif cap_n and cap_groups:
        print(f"Group cap ON: at most {cap_n} picks per {sleeve.group_by or 'group'}")
    core_cols = [t for t in sleeve.core_tickers if t in cols]
    print(f"Instruments: {len(cols)}" + (f" ({len(core_cols)} core)" if core_cols else "")
          + f"; feature set: {sleeve.feature_set}")
    if sleeve.universe_missing:
        print(f"WARNING: universe file {sleeve.universe_file} not found -- running on "
              f"the {len(cols)} listed instruments only. Build it with "
              f"`python -m umushroom.etf_universe`.")
    if sleeve.groups:
        mix = pd.Series([sleeve.groups.get(t, "other") for t in cols]).value_counts()
        print("By group: " + ", ".join(f"{k} {v}" for k, v in mix.items()))

    log_path = _sleeve_log_path(cfg, sleeve)
    if not args.skip_prediction_check:
        report = evaluate_latest_predictions(log_path, prices.iloc[-1], prices.index[-1])
        print("\n--- Previous Run's Prediction Accuracy ---")
        print(report.summary() if report is not None else
              "No prior predictions logged yet to evaluate (first run).")

    horizon = sleeve_horizon(cfg, sleeve)
    print(f"Forecast horizon: {horizon} trading days (rebalanced every "
          f"{cfg.rebalance_days}; contest window {cfg.contest_window_days})")
    # v4.20: learner, weighting scheme and near-duplicate guard are decided
    # here ONCE and used identically by the backtests and the live orders
    model_type = sleeve_model_type(cfg, sleeve)
    pick_corr = sleeve_pick_correlation(cfg, sleeve)
    weighting = cast(WeightingScheme, sleeve.weighting or args.weighting)
    if weighting == "factor" and not groups:
        print("NOTE: weighting 'factor' needs ranking groups (exchange or asset class) "
              "to build the factors; this sleeve has none -- using equal weight.")
        weighting = "equal"
    weight_fn = _weight_fn_for(weighting, groups)
    print(f"Model: {model_type}; weighting: {weighting}; near-duplicate guard: "
          + (f"max correlation {pick_corr:.2f}" if pick_corr else "off"))
    wf = _walk_forward_kwargs(cfg, sleeve, groups, dvol, cap_groups, cap_n, weight_fn)
    X, y = features_mod.build_training_set(
        prices, forward_return_days=horizon, step=cfg.rebalance_days,
        min_history_days=cfg.min_history_days, target=cfg.target, groups=groups,
        dollar_volume=dvol, feature_set=sleeve.feature_set)
    purge_dates = math.ceil(horizon / cfg.rebalance_days)
    model = RankingModel(alpha=cfg.model_alpha, random_seed=cfg.random_seed,
                         model_type=model_type, purge_dates=purge_dates)
    cv_stats = model.cross_validate(X, y, purge_dates=purge_dates)
    model.fit(X, y)
    print(f"\n--- Feature importance ({model.importance_label}) ---")
    print(model.feature_importance().to_string())
    print(f"\nCross-validated IC: {cv_stats['ic_mean']:+.4f} "
          f"(t = {cv_stats['ic_tstat']:+.2f} over {int(cv_stats['ic_n_dates'])} dates)")
    if args.model_out:
        out = Path(args.model_out)
        model.save(out if sleeve.name == EQUITY_SLEEVE
                   else out.with_name(f"{out.stem}_{sleeve.name}"))

    if not args.skip_backtest:
        bt = run_backtest(prices, model, top_n=sleeve.top_n,
                          rebalance_days=cfg.rebalance_days,
                          min_history_days=cfg.min_history_days, groups=groups,
                          dollar_volume=dvol, feature_set=sleeve.feature_set)
        print("\n--- Quick Sanity-Check Backtest (IN-SAMPLE -- see warning) ---")
        print(bt.summary())

    info = SleeveReport(sleeve.name, sleeve.weight, sleeve.selection,
                        feature_set=sleeve.feature_set, horizon_days=horizon,
                        n_instruments=len(cols), n_core=len(core_cols),
                        universe_status=dict(sleeve.universe_status),
                        universe_missing=sleeve.universe_missing,
                        contest_window_days=cfg.contest_window_days,
                        model_type=model_type, weighting=weighting,
                        max_pick_correlation=pick_corr)
    if collect is not None:
        collect[sleeve.name] = info
    pwf: PurgedWalkForwardResult | None = None
    if not args.skip_purged_backtest:
        pwf = _report_out_of_sample(cfg, args, sleeve, prices, wf)
        if pwf is not None:
            info.backtest = pwf
            try:
                a = benchmark_alpha(pwf.period_returns, pwf.benchmark_period_returns,
                                    252 / cfg.rebalance_days)
                info.alpha_annual, info.alpha_t = a.alpha_annualized, a.alpha_tstat
            except ValueError:
                pass
            info.window = _report_contest_window(cfg, pwf, sleeve, prices, wf)
        if args.compare_baseline and sleeve.name == EQUITY_SLEEVE:
            _compare_with_baseline(cfg, sleeve, prices, listings, wf)
        if args.compare_sector_cap and sleeve.name == EQUITY_SLEEVE:
            if sectors is None:
                print("\nSector-cap comparison skipped: no sector_file in config.")
            else:
                compare_sector_cap(cfg, sleeve, prices, wf, sectors,
                                   args.compare_sector_cap)
        if args.compare_horizon:
            name, _, days = str(args.compare_horizon).partition(":")
            if name == sleeve.name:
                compare_horizons(cfg, sleeve, prices, wf,
                                 [int(d) for d in days.split(",") if d.strip()])
        for flag, fn in (("compare_model", compare_models),
                         ("compare_weighting", compare_weightings),
                         ("compare_dedup", compare_dedup)):
            spec = getattr(args, flag, None)
            if spec:
                name, _, items = str(spec).partition(":")
                if name == sleeve.name:
                    fn(cfg, sleeve, prices, wf, groups,
                       [x.strip() for x in items.split(",") if x.strip()])
    if args.validate and sleeve.selection in ("model", "auto"):
        vr = _run_validation(cfg, args, sleeve, prices, listings, wf, cap_groups, cap_n)
        info.validation = [(c.name, c.status, c.detail) for c in vr.checks]

    # v4.19 "auto": trade the model's picks only while the out-of-sample
    # evidence holds up; otherwise hold the core basket, equal weight.
    use_model, reason = decide_selection(sleeve, pwf, info.window)
    info.selection_used = ("model" if use_model else
                           "core" if sleeve.selection == "auto" else "all")
    info.decision = reason
    if sleeve.selection == "auto":
        print(f"\nSelection rule (auto): {reason}")
    hold_all = not use_model
    alloc_prices, alloc_cols = prices, cols
    if hold_all and sleeve.selection == "auto":
        if not core_cols:
            raise DataError(f"sleeve '{sleeve.name}': auto fallback needs core "
                            f"instruments with data; none usable today")
        alloc_prices, alloc_cols = prices[core_cols], core_cols
    stale_here = sorted(market.stale_last & set(alloc_cols))
    if hold_all and stale_here:
        print(f"NOTE: no fresh close yet for {stale_here}; sizing them on the "
              f"previous close. Check their prices on UMushroom when entering.")
    # v4.20: a held-all basket is EQUAL weight whatever the sleeve's scheme --
    # that is what the walk-forward benchmark, the report's "what is actually
    # held" curve and the Monte Carlo assume, and the only sizing of the core
    # basket that has been evaluated. hrp/factor apply to the model's picks.
    alloc_weighting: WeightingScheme = "equal" if hold_all else weighting
    if hold_all:
        print(f"\nSelection: holding ALL {len(alloc_cols)} "
              f"{'core ' if sleeve.selection == 'auto' else ''}instruments "
              f"(equal weight); the model above is reported but does not pick."
              + (f" (weighting '{weighting}' applies to model picks only)"
                 if weighting != "equal" else ""))
    alloc = generate_allocation(
        alloc_prices, model, top_n=sleeve.top_n, hold_all=hold_all,
        balance=cfg.starting_balance * sleeve.weight,
        weighting=alloc_weighting, groups=groups,
        local_prices=market.local_prices[alloc_cols],
        currencies={t: listings[t].currency for t in alloc_cols},
        fractional=sleeve.fractional,
        # crypto can legitimately trade at fractions of a cent; stocks and
        # ETFs below 0.10 in their own currency are treated as data errors
        min_local_price=0.0 if sleeve.fractional else MIN_LOCAL_PRICE,
        dollar_volume=dvol,
        # model-picked sleeves never trade on an old price (it would also be
        sectors=cap_groups, max_per_sector=cap_n,
        # an old SIGNAL); hold-all sleeves just warn -- a day-old price only
        # shifts the share count slightly, it doesn't change what is held
        exclude=(frozenset() if hold_all
                 else frozenset(market.stale_last & set(alloc_cols))),
        feature_set=sleeve.feature_set,
        max_pick_correlation=pick_corr,
        factor_groups=groups,
        factor_prices=prices,          # the factors come from the WHOLE sleeve
    )
    if alloc.factor_risk is not None:
        print("\n--- Factor-risk weights (v4.20) ---")
        print(alloc.factor_risk.summary())
        info.factor_risk = factor_risk_payload(alloc.factor_risk)
    log_predictions(log_path=log_path, run_date=prices.index[-1],
                    all_scores=alloc.all_scores, prices=prices.iloc[-1],
                    picked_tickers=set(alloc.orders.index))
    info.scores = alloc.all_scores
    if alloc_cols is not cols:
        # held the core basket: still rank the whole universe for the report
        try:
            feats_all = features_mod.build_features_at_index(
                prices, len(prices) - 1, groups=groups, dollar_volume=dvol,
                feature_set=sleeve.feature_set).dropna()
            if not feats_all.empty:
                info.scores = model.predict(feats_all)
        except (InsufficientHistoryError, ValueError) as e:
            get_logger("cli").warning(
                f"Could not rank the full {sleeve.name} universe today: {e}")
    return alloc


def decide_selection(sleeve: SleeveConfig, pwf: PurgedWalkForwardResult | None,
                     window: WindowResult | None) -> tuple[bool, str]:
    """The pre-declared rule behind `selection: auto`.

    Use the model's picks only if BOTH hold out of sample:
      1. ranking skill: walk-forward IC t-stat >= SKILL_T_THRESHOLD, and
      2. contest-window verdict is not FAIL (window.py).
    Otherwise hold the core basket. "model"/"all" bypass the rule.
    Returns (use_model, reason)."""
    if sleeve.selection == "model":
        return True, "selection: model (configured; no rule applied)"
    if sleeve.selection == "all":
        return False, "selection: all (configured; no rule applied)"
    if pwf is None:
        return False, ("no purged walk-forward result this run, so the model's "
                       "picks are not trusted: holding the core basket")
    if not pwf.has_evidence_of_skill:
        return False, (f"walk-forward IC t = {pwf.ic_tstat:+.2f} (needs >= 2): no "
                       f"reliable ranking skill, holding the core basket")
    if window is None:
        return False, "no contest-window evaluation possible: holding the core basket"
    if window.verdict == FAIL:
        return False, (f"IC t = {pwf.ic_tstat:+.2f} but the picks did not beat the "
                       f"benchmark inside contest-length windows (win rate "
                       f"{window.win_rate:.0%}, t = {window.active_tstat:+.2f}): "
                       f"holding the core basket")
    return True, (f"IC t = {pwf.ic_tstat:+.2f} and contest-window verdict "
                  f"{window.verdict} (win rate {window.win_rate:.0%}, "
                  f"t = {window.active_tstat:+.2f}): trading the model's top "
                  f"{sleeve.top_n}")


def _weight_fn_for(weighting: str, groups: dict[str, str] | None) -> WeightFn | None:
    """The weighting scheme as a (prices, picks, as_of_idx) -> weights function
    for the walk-forward; None = equal weight."""
    if weighting == "hrp":
        return hrp_weight
    if weighting == "factor":
        return functools.partial(factor_weight, groups=groups or {})
    return None


def _walk_forward_kwargs(cfg: StrategyConfig, sleeve: SleeveConfig,
                         groups: dict[str, str] | None, dvol: pd.DataFrame | None,
                         cap_groups: dict[str, str] | None, cap_n: int | None,
                         weight_fn: WeightFn | None) -> dict[str, Any]:
    """ONE definition of how a sleeve is evaluated out of sample, shared by
    the headline backtest, the contest-window evaluation, every head-to-head
    and the validation suite -- so they cannot drift apart."""
    horizon = sleeve_horizon(cfg, sleeve)
    return {
        "test_window_days": cfg.purged_test_window_days,
        "purge_days": max(cfg.purged_purge_days, horizon),
        "embargo_days": cfg.purged_embargo_days,
        "rebalance_days": cfg.rebalance_days,
        "forward_return_days": horizon,
        "top_n": sleeve.top_n,
        "min_train_days": cfg.purged_min_train_days,
        "model_alpha": cfg.model_alpha,
        "random_seed": cfg.random_seed,
        "target": cfg.target,
        "groups": groups,
        "dollar_volume": dvol,
        "sectors": cap_groups,
        "max_per_sector": cap_n,
        "feature_set": sleeve.feature_set,
        "model_type": sleeve_model_type(cfg, sleeve),
        "max_pick_correlation": sleeve_pick_correlation(cfg, sleeve),
        "weight_fn": weight_fn,
    }


def _window_test_days(cfg: StrategyConfig) -> int:
    """Test-fold length for the contest-window evaluation: a window needs
    `window_periods` CONSECUTIVE rebalance periods inside one test fold
    (folds are separated by purge + embargo gaps), so the fold must be at
    least one contest window long."""
    periods = window_periods_for(cfg.contest_window_days, cfg.rebalance_days)
    return max(cfg.purged_test_window_days, periods * cfg.rebalance_days)


def _report_contest_window(cfg: StrategyConfig, pwf: PurgedWalkForwardResult,
                           sleeve: SleeveConfig, prices: pd.DataFrame,
                           wf: dict[str, Any]) -> WindowResult | None:
    periods = window_periods_for(cfg.contest_window_days, cfg.rebalance_days)
    test_days = _window_test_days(cfg)
    source = pwf
    note = ""
    if test_days > cfg.purged_test_window_days:
        # re-run the walk-forward with folds exactly one contest window long
        try:
            source = run_purged_walk_forward(prices, **{**wf, "test_window_days": test_days})
        except InsufficientHistoryError as e:
            print(f"\nContest-window evaluation skipped: {e}")
            return None
        note = (f" (separate walk-forward with {test_days}-day test folds, one contest "
                f"window each; {len(source.folds)} folds, IC t = {source.ic_tstat:+.2f})")
    try:
        w = contest_windows(source.period_returns, source.benchmark_period_returns,
                            source.period_end_dates, periods, cfg.rebalance_days)
    except ValueError as e:
        print(f"\nContest-window evaluation skipped: {e}")
        return None
    print(f"\n--- Contest-Window Evaluation ({cfg.contest_window_days} trading days, "
          f"out of sample){note} ---")
    print(w.summary())
    return w


def compare_horizons(cfg: StrategyConfig, sleeve: SleeveConfig, prices: pd.DataFrame,
                     wf: dict[str, Any], horizons: list[int]) -> int | None:
    """v4.19: ONE pre-declared head-to-head of forecast horizons on identical
    data. Rule (fixed before looking): keep the horizon with the highest
    contest-window active-return t-statistic among those whose walk-forward
    IC t is at least 2; ties go to the SHORTER horizon. Horizons longer than
    the contest window are refused -- their payoff could not arrive in time."""
    too_long = [h for h in horizons if h > cfg.contest_window_days or h < 1]
    if too_long:
        print(f"\nHorizon comparison refused: {too_long} do not fit inside the "
              f"{cfg.contest_window_days}-day contest window.")
        return None
    print(f"\n=== Forecast horizon: head-to-head on identical data "
          f"(sleeve {sleeve.name}) ===")
    print("Rule (decided BEFORE the results): among horizons with IC t >= 2, keep the "
          "one with the highest contest-window active t; ties -> shorter horizon.")
    periods = window_periods_for(cfg.contest_window_days, cfg.rebalance_days)
    test_days = _window_test_days(cfg)
    print(f"(test folds of {test_days} days so each holds one contest window)")
    rows: dict[int, dict[str, float]] = {}
    for h in sorted(set(horizons)):
        try:
            r = run_purged_walk_forward(
                prices, **{**wf, "test_window_days": test_days,
                           "purge_days": max(cfg.purged_purge_days, h),
                           "forward_return_days": h})
            w = contest_windows(r.period_returns, r.benchmark_period_returns,
                                r.period_end_dates, periods, cfg.rebalance_days)
        except (InsufficientHistoryError, ValueError) as e:
            print(f"  horizon {h}: skipped ({e})")
            continue
        rows[h] = {"ic": r.mean_information_coefficient, "ic_t": r.ic_tstat,
                   "active": float(r.active_returns.mean()), "active_t": r.active_tstat,
                   "win": w.win_rate, "w_active": w.mean_active, "w_t": w.active_tstat,
                   "sharpe": r.overall_sharpe}
    if not rows:
        print("No horizon produced a result.")
        return None
    print(f"{'horizon':<10}{'IC':>8}{'IC t':>7}{'active/wk':>11}{'act t':>7}"
          f"{'win(6w)':>9}{'active/6w':>11}{'6w t':>7}{'Sharpe':>8}")
    for h, m in rows.items():
        print(f"{h:<10}{m['ic']:>+8.4f}{m['ic_t']:>+7.2f}{m['active']:>+11.3%}"
              f"{m['active_t']:>+7.2f}{m['win']:>9.0%}{m['w_active']:>+11.2%}"
              f"{m['w_t']:>+7.2f}{m['sharpe']:>8.2f}")
    eligible = {h: m for h, m in rows.items()
                if np.isfinite(m["ic_t"]) and m["ic_t"] >= 2.0 and np.isfinite(m["w_t"])}
    if not eligible:
        print("\nVERDICT: no horizon shows reliable ranking skill (IC t < 2 for all); "
              "keep the current setting and do not trade on these results.")
        return None
    best = min(eligible.items(), key=lambda kv: (-kv[1]["w_t"], kv[0]))[0]
    current = sleeve_horizon(cfg, sleeve)
    print(f"\nVERDICT: horizon {best} days wins (contest-window active t = "
          f"{eligible[best]['w_t']:+.2f})" + (" -- already the configured horizon."
          if best == current else f". To adopt it, set forward_return_days: {best} "
          f"under sleeves.{sleeve.name} in config.yaml."))
    return best


def _sleeve_groups(cfg: StrategyConfig, sleeve: SleeveConfig,
                   listings: dict) -> dict[str, str] | None:
    """Ranking groups: sector-within-exchange -> sector -> pooled (v4.4),
    or exchange only (v4), or one global group. v4.19: `group_by:
    asset_class` ranks ETFs within their asset class (bond vs. bond)."""
    tickers = list(listings)
    exchange = {t: lst.exchange for t, lst in listings.items()}
    if sleeve.group_by == "asset_class":
        return {t: sleeve.groups.get(t, "other") for t in tickers}
    if sleeve.group_by == "exchange":
        return exchange
    if sleeve.neutralize_by_sector and cfg.sector_file is not None:
        sector = load_sectors(cfg.sector_file)
        levels = []
        if sleeve.neutralize_by_exchange:
            levels.append({t: f"{sector[t]}|{exchange[t]}" for t in tickers
                           if t in sector})
        levels.append({t: sector[t] for t in tickers if t in sector})
        if sleeve.neutralize_by_exchange:
            levels.append(exchange)
        return hierarchical_groups(tickers, levels)
    return exchange if sleeve.neutralize_by_exchange else None


def _run_validation(cfg: StrategyConfig, args: argparse.Namespace,
                    sleeve: SleeveConfig, prices: pd.DataFrame, listings: dict,
                    wf: dict[str, Any],
                    cap_groups: dict[str, str] | None = None,
                    cap_n: int | None = None) -> ValidationReport:
    if cap_groups is None and sleeve.max_per_sector and cfg.sector_file is not None:
        cap_groups, cap_n = load_sectors(cfg.sector_file), sleeve.max_per_sector
    # the live selection rule (caps, guard, weighting, learner) is what gets validated
    kwargs: dict[str, Any] = {**wf, "sectors": cap_groups if cap_n else None,
                              "max_per_sector": cap_n}
    exchanges = {t: lst.exchange for t, lst in listings.items()}
    multi_market = len(set(exchanges.values())) > 1
    is_crypto = all(lst.exchange == "Crypto" for lst in listings.values())
    print(f"\nRunning validation suite ({args.placebo_runs} placebo runs; "
          f"this takes a few minutes)...")
    report = run_validation(
        prices, sleeve.name, kwargs,
        groups_for_report=exchanges if multi_market else None,
        placebo_runs=args.placebo_runs,
        benchmark_annual_range=(-0.9, 3.0) if is_crypto else (-0.6, 1.0))
    print()
    print(report.summary())
    return report


def _compare_with_baseline(cfg: StrategyConfig, sleeve: SleeveConfig,
                           prices: pd.DataFrame, listings: dict,
                           wf: dict[str, Any]) -> None:
    """ONE pre-declared head-to-head on identical data: the v4.3 equity
    setup vs. the current one. Both are scored on the same weekly holding
    returns, so the comparison is like-for-like."""
    common: dict[str, Any] = {"test_window_days": cfg.purged_test_window_days,
              "embargo_days": cfg.purged_embargo_days,
              "rebalance_days": cfg.rebalance_days,
              "min_train_days": cfg.purged_min_train_days,
              "model_alpha": cfg.model_alpha, "random_seed": cfg.random_seed,
              "target": cfg.target}
    base_fwd = cfg.rebalance_days
    dvol = wf.get("dollar_volume")
    runs = {
        "v4.3 baseline (exch-neutral, 5d, top 10, no volume)": run_purged_walk_forward(
            prices, purge_days=max(base_fwd, cfg.rebalance_days),
            forward_return_days=base_fwd, top_n=min(10, prices.shape[1]),
            groups={t: lst.exchange for t, lst in listings.items()}, **common),
        f"current ({sleeve_horizon(cfg, sleeve)}d, top {sleeve.top_n}"
        f"{', sector' if sleeve.neutralize_by_sector else ''}"
        f"{', volume' if dvol is not None else ''})": run_purged_walk_forward(
            prices, **wf),
    }
    ppy = 252 / cfg.rebalance_days
    print("\n=== Head-to-head on identical data (scored on weekly holding returns) ===")
    print(f"{'':<55}{'IC':>8}{'IC t':>7}{'active/wk':>11}{'act t':>7}"
          f"{'Q5-Q1':>9}{'alpha/yr':>10}{'alpha t':>8}{'Sharpe':>8}")
    for name, r in runs.items():
        try:
            a = benchmark_alpha(r.period_returns, r.benchmark_period_returns, ppy)
            alpha, alpha_t = a.alpha_annualized, a.alpha_tstat
        except ValueError:
            alpha, alpha_t = float("nan"), float("nan")
        print(f"{name:<55}{r.mean_information_coefficient:>+8.4f}{r.ic_tstat:>+7.2f}"
              f"{r.active_returns.mean():>+11.3%}{r.active_tstat:>+7.2f}"
              f"{r.long_short_period_returns.mean():>+9.3%}{alpha:>+10.2%}"
              f"{alpha_t:>+8.2f}{r.overall_sharpe:>8.2f}")
    print("Decide BEFORE looking: keep the current setup only if its active-return "
          "t and alpha t are higher. Do not try further variants on this data.")


def compare_sector_cap(cfg: StrategyConfig, sleeve: SleeveConfig,
                       prices: pd.DataFrame, wf: dict[str, Any], sectors: dict[str, str],
                       cap: int) -> CapVerdict:
    """v4.18: ONE pre-declared head-to-head on identical data -- the equity
    sleeve with no sector cap vs. at most `cap` picks per sector."""
    print("\n=== Sector cap: head-to-head on identical data ===")
    print(f"Rule (decided BEFORE the results): switch the cap on only if it keeps "
          f"at least {ALPHA_RETENTION:.0%} of the uncapped alpha AND does not raise "
          f"the weekly 95% VaR.")
    kw: dict[str, Any] = {**wf, "record_cross_sections": True, "sectors": sectors}
    ppy = 252 / cfg.rebalance_days
    rows = {}
    for label, mps in (("no cap", None), (f"max {cap} per sector", cap)):
        r = run_purged_walk_forward(prices, **{**kw, "max_per_sector": mps})
        try:
            a = benchmark_alpha(r.period_returns, r.benchmark_period_returns, ppy)
            alpha, alpha_t = a.alpha_annualized, a.alpha_tstat
        except ValueError:
            alpha, alpha_t = float("nan"), float("nan")
        shares = [top_sector_share(capped_top(p.sort_values(ascending=False).index,
                                              sleeve.top_n, sectors, mps), sectors)
                  for _d, p, _r in r.cross_sections]
        var95, es95 = var_es(r.period_returns, 0.95)
        rows[label] = {"alpha": alpha, "alpha_t": alpha_t, "sharpe": r.overall_sharpe,
                       "var": var95, "es": es95, "worst": float(-r.period_returns.min()),
                       "share": float(np.mean(shares)) if shares else float("nan"),
                       "half": float(np.mean([x >= 0.5 for x in shares])) if shares
                       else float("nan")}
    print(f"{'':<22}{'alpha/yr':>10}{'alpha t':>9}{'Sharpe':>8}{'VaR95/wk':>10}"
          f"{'ES95/wk':>9}{'worst wk':>10}{'top sector':>12}{'wks >=50%':>11}")
    for label, m in rows.items():
        print(f"{label:<22}{m['alpha']:>+10.2%}{m['alpha_t']:>+9.2f}{m['sharpe']:>8.2f}"
              f"{m['var']:>10.2%}{m['es']:>9.2%}{m['worst']:>10.2%}{m['share']:>12.0%}"
              f"{m['half']:>11.0%}")
    u, c = rows["no cap"], rows[f"max {cap} per sector"]
    verdict = decide_cap(u["alpha"], c["alpha"], u["var"], c["var"])
    print(f"\nVERDICT: {'SWITCH THE CAP ON' if verdict.adopt else 'KEEP NO CAP'} "
          f"-- {verdict.reason}.")
    if verdict.adopt:
        print(f"To switch it on, add this line to config.yaml:  max_per_sector: {cap}")
    print("(top sector = average share of the picks in the single biggest sector; "
          "wks >=50% = how often one sector was half the picks or more)")
    return verdict


SKILL_RULE = "skill"      # keep the variant with the best contest-window active t
RISK_RULE = "risk"        # adopt only if alpha is kept and weekly VaR is not raised


def skill_verdict(rows: dict[str, dict[str, float]], current: str,
                  simplicity: list[str] | None = None) -> tuple[str | None, str]:
    """The learner rule (pure, unit-tested). Returns (variant to adopt or
    None, reason).

    Among variants with walk-forward IC t >= 2, the highest contest-window
    active t wins; ties go to the current setting, then the simpler one.
    A variant OTHER than the current one is adopted only if its
    contest-window active t is itself >= WINDOW_T_THRESHOLD: the first ETF
    comparison (2026-10-19) produced a "winner" at t = +0.34 against
    -0.18, a difference that is pure noise -- a live model is not changed
    on that. The floor is the same bar window.py uses for a PASS."""
    eligible = {n: m for n, m in rows.items()
                if np.isfinite(m["ic_t"]) and m["ic_t"] >= 2.0 and np.isfinite(m["w_t"])}
    if not eligible:
        return None, ("no variant shows reliable ranking skill (IC t < 2 for all); keep "
                      "the current setting and do not trade on these results")
    order = simplicity or list(rows)

    def key(item: tuple[str, dict[str, float]]) -> tuple[float, int, int]:
        n, m = item
        return (-m["w_t"], 0 if n == current else 1,
                order.index(n) if n in order else len(order))
    best = min(eligible.items(), key=key)[0]
    t = eligible[best]["w_t"]
    if best == current:
        return None, (f"{best} wins (contest-window active t = {t:+.2f}) -- already the "
                      f"current setting")
    if t < WINDOW_T_THRESHOLD:
        return None, (f"{best} has the highest contest-window active t ({t:+.2f}) but "
                      f"below the {WINDOW_T_THRESHOLD:.0f} needed to replace {current}: "
                      f"the difference is noise; keep {current}")
    return best, f"{best} wins (contest-window active t = {t:+.2f})"


def _variant_stats(cfg: StrategyConfig, r: PurgedWalkForwardResult,
                   periods: int) -> dict[str, float]:
    ppy = 252 / cfg.rebalance_days
    try:
        a = benchmark_alpha(r.period_returns, r.benchmark_period_returns, ppy)
        alpha, alpha_t = a.alpha_annualized, a.alpha_tstat
    except ValueError:
        alpha, alpha_t = float("nan"), float("nan")
    try:
        w = contest_windows(r.period_returns, r.benchmark_period_returns,
                            r.period_end_dates, periods, cfg.rebalance_days)
        win, w_active, w_t = w.win_rate, w.mean_active, w.active_tstat
    except ValueError:
        win = w_active = w_t = float("nan")
    try:
        var95, _es = var_es(r.period_returns, 0.95)
    except ValueError:
        var95 = float("nan")
    return {"ic": r.mean_information_coefficient, "ic_t": r.ic_tstat,
            "active": float(r.active_returns.mean()), "active_t": r.active_tstat,
            "win": win, "w_active": w_active, "w_t": w_t, "sharpe": r.overall_sharpe,
            "alpha": alpha, "alpha_t": alpha_t, "var": var95,
            "fallbacks": float(r.weight_fallbacks)}


def compare_variants(cfg: StrategyConfig, sleeve: SleeveConfig, prices: pd.DataFrame,
                     wf: dict[str, Any], title: str, variants: dict[str, dict[str, Any]],
                     current: str, rule: str, simplicity: list[str] | None = None
                     ) -> str | None:
    """v4.20: ONE pre-declared head-to-head of settings for one sleeve on
    identical data, with test folds one contest window long. Returns the
    name of the variant to adopt (None = keep the current setting).

    SKILL_RULE (learners): among variants with walk-forward IC t >= 2, keep
      the highest contest-window active t; ties -> the current setting, then
      the simpler one (`simplicity` order).
    RISK_RULE (weighting, near-duplicate guard): a variant is adopted only
      if it keeps >= ALPHA_RETENTION of the current setting's alpha AND does
      not raise the weekly 95% VaR (decide_cap); if several qualify, the one
      with the lowest VaR.
    """
    periods = window_periods_for(cfg.contest_window_days, cfg.rebalance_days)
    test_days = _window_test_days(cfg)
    print(f"\n=== {title}: head-to-head on identical data (sleeve {sleeve.name}) ===")
    if rule == SKILL_RULE:
        print("Rule (decided BEFORE the results): among variants with IC t >= 2, keep the "
              "one with the highest contest-window active t; ties -> the current, then "
              f"the simpler setting; a new setting needs that t >= "
              f"{WINDOW_T_THRESHOLD:.0f} to replace the current one.")
    else:
        print(f"Rule (decided BEFORE the results): adopt a variant only if it keeps at "
              f"least {ALPHA_RETENTION:.0%} of the current setting's alpha AND does not "
              f"raise the weekly 95% VaR; several -> the lowest VaR.")
    print(f"(test folds of {test_days} days so each holds one contest window; "
          f"current setting: {current})")
    rows: dict[str, dict[str, float]] = {}
    for name, overrides in variants.items():
        try:
            r = run_purged_walk_forward(
                prices, **{**wf, "test_window_days": test_days, **overrides})
        except (InsufficientHistoryError, ValueError) as e:
            print(f"  {name}: skipped ({e})")
            continue
        rows[name] = _variant_stats(cfg, r, periods)
    if not rows:
        print("No variant produced a result.")
        return None
    print(f"{'variant':<16}{'IC':>8}{'IC t':>7}{'active/wk':>11}{'act t':>7}"
          f"{'win(6w)':>9}{'active/6w':>11}{'6w t':>7}{'Sharpe':>8}{'alpha/yr':>10}"
          f"{'alpha t':>8}{'VaR95/wk':>10}")
    for name, m in rows.items():
        print(f"{name:<16}{m['ic']:>+8.4f}{m['ic_t']:>+7.2f}{m['active']:>+11.3%}"
              f"{m['active_t']:>+7.2f}{m['win']:>9.0%}{m['w_active']:>+11.2%}"
              f"{m['w_t']:>+7.2f}{m['sharpe']:>8.2f}{m['alpha']:>+10.2%}"
              f"{m['alpha_t']:>+8.2f}{m['var']:>10.2%}"
              + (f"   (equal-weight fallback in {m['fallbacks']:.0f} periods)"
                 if m["fallbacks"] else ""))
    if rule == SKILL_RULE:
        winner, reason = skill_verdict(rows, current, simplicity or list(variants))
        print(f"\nVERDICT: {reason}.")
        return winner
    if current not in rows:
        print(f"\nVERDICT: the current setting ({current}) produced no result; "
              f"nothing to compare against.")
        return None
    cur = rows[current]
    adoptable: dict[str, float] = {}
    for n, m in rows.items():
        if n == current:
            continue
        v = decide_cap(cur["alpha"], m["alpha"], cur["var"], m["var"])
        print(f"  {n}: {'ADOPTABLE' if v.adopt else 'keep ' + current} -- {v.reason}")
        if v.adopt:
            adoptable[n] = m["var"]
    if not adoptable:
        print(f"\nVERDICT: KEEP {current}.")
        return None
    best = min(adoptable.items(), key=lambda kv: (kv[1], list(rows).index(kv[0])))[0]
    print(f"\nVERDICT: SWITCH TO {best} (lowest weekly VaR among the variants that keep "
          f"the alpha).")
    return best


def compare_models(cfg: StrategyConfig, sleeve: SleeveConfig, prices: pd.DataFrame,
                   wf: dict[str, Any], groups: dict[str, str] | None,
                   types: list[str]) -> str | None:
    """`--compare-model SLEEVE:ridge,mlp,gbm` (v4.20)."""
    bad = [t for t in types if t not in MODEL_TYPES]
    if bad:
        print(f"\nModel comparison refused: unknown model type(s) {bad}; "
              f"choose from {MODEL_TYPES}.")
        return None
    current = sleeve_model_type(cfg, sleeve)
    variants = {t: {"model_type": t} for t in dict.fromkeys([*types])}
    best = compare_variants(cfg, sleeve, prices, wf, "Learner", variants, current,
                            SKILL_RULE, simplicity=list(MODEL_TYPES))
    if best:
        print(f"To adopt it, set model_type: {best} under sleeves.{sleeve.name} "
              f"(or at the top level for the stock sleeve) in config.yaml.")
    return best


def compare_weightings(cfg: StrategyConfig, sleeve: SleeveConfig, prices: pd.DataFrame,
                       wf: dict[str, Any], groups: dict[str, str] | None,
                       schemes: list[str]) -> str | None:
    """`--compare-weighting SLEEVE:equal,hrp,factor` (v4.20)."""
    bad = [s for s in schemes if s not in WEIGHTINGS]
    if bad:
        print(f"\nWeighting comparison refused: unknown scheme(s) {bad}; "
              f"choose from {WEIGHTINGS}.")
        return None
    if "factor" in schemes and not groups:
        print("\nNOTE: 'factor' needs ranking groups to build factors; this sleeve has "
              "none, so it is left out.")
        schemes = [s for s in schemes if s != "factor"]
    current = "equal" if wf.get("weight_fn") is None else (
        "factor" if isinstance(wf["weight_fn"], functools.partial) else "hrp")
    variants = {s: {"weight_fn": _weight_fn_for(s, groups)} for s in dict.fromkeys(schemes)}
    best = compare_variants(cfg, sleeve, prices, wf, "Weighting", variants, current,
                            RISK_RULE)
    if best:
        print(f"To adopt it, set weighting: {best} under sleeves.{sleeve.name} in "
              f"config.yaml (or pass --weighting {best} for the stock sleeve).")
    return best


def compare_dedup(cfg: StrategyConfig, sleeve: SleeveConfig, prices: pd.DataFrame,
                  wf: dict[str, Any], groups: dict[str, str] | None,
                  thresholds: list[str]) -> str | None:
    """`--compare-dedup SLEEVE:0.9,0.95` (v4.20): the near-duplicate guard at
    each threshold vs. no guard."""
    try:
        vals = [float(t) for t in thresholds]
    except ValueError:
        print(f"\nDedup comparison refused: thresholds must be numbers, got {thresholds}.")
        return None
    if any(not 0 < v <= 1 for v in vals):
        print("\nDedup comparison refused: thresholds must be in (0, 1].")
        return None
    cur_v = sleeve_pick_correlation(cfg, sleeve)
    current = f"max corr {cur_v:.2f}" if cur_v else "no guard"
    variants: dict[str, dict[str, Any]] = {"no guard": {"max_pick_correlation": None}}
    for v in sorted(set(vals)):
        variants[f"max corr {v:.2f}"] = {"max_pick_correlation": v}
    best = compare_variants(cfg, sleeve, prices, wf, "Near-duplicate guard", variants,
                            current, RISK_RULE)
    if best:
        value = "0" if best == "no guard" else best.split()[-1]
        print(f"To adopt it, set max_pick_correlation: {value} under sleeves."
              f"{sleeve.name} (or at the top level for the stock sleeve) in config.yaml.")
    return best


def _report_out_of_sample(cfg: StrategyConfig, args: argparse.Namespace,
                          sleeve: SleeveConfig, prices: pd.DataFrame,
                          wf: dict[str, Any]) -> PurgedWalkForwardResult | None:
    try:
        pwf = run_purged_walk_forward(
            prices, record_cross_sections=bool(getattr(args, "report", None)), **wf)
    except InsufficientHistoryError as e:
        print(f"\n--- Purged & Embargoed Walk-Forward Backtest ---\nSkipped: {e}")
        return None
    print("\n--- Purged & Embargoed Walk-Forward Backtest ---")
    print(pwf.summary())
    if pwf.weight_fallbacks:
        print(f"NOTE: the weighting scheme fell back to equal weight in "
              f"{pwf.weight_fallbacks} of {len(pwf.period_returns)} periods.")

    periods_per_year = 252 / cfg.rebalance_days
    print("\n--- Alpha Regression ---")
    try:
        print(benchmark_alpha(pwf.period_returns, pwf.benchmark_period_returns,
                              periods_per_year).summary())
    except ValueError as e:
        print(f"Benchmark alpha skipped: {e}")
    if args.fama_french and sleeve.name == EQUITY_SLEEVE:
        try:
            ff = fetch_french_factors(cfg.fama_french_region, cfg.cache_dir)
            print(fama_french_alpha(pwf.period_returns, pwf.period_end_dates, ff,
                                    periods_per_year, cfg.base_currency,
                                    cfg.fama_french_region).summary())
        except (DataError, ValueError) as e:
            print(f"Fama-French alpha skipped: {e}")
    try:
        ci = bootstrap_sharpe(pwf.period_returns, periods_per_year=periods_per_year)
        print("\n--- Bootstrap Sharpe Confidence Interval ---")
        print(ci.summary())
    except ValueError as e:
        print(f"\nBootstrap Sharpe CI skipped: {e}")
    return pwf


if __name__ == "__main__":
    sys.exit(main())
