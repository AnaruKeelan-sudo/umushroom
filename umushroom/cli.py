"""Command-line entrypoint.

Orchestrates the full pipeline (data -> features -> train -> validate
-> backtest -> allocate) and is the only module that calls sys.exit.
Everything it calls raises typed exceptions on failure rather than
exiting directly, which keeps the rest of the package usable as a
library (e.g. from a notebook) without side effects.
"""

from __future__ import annotations

import argparse
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
    SleeveConfig,
    StrategyConfig,
    load_config,
    load_sectors,
    sleeve_horizon,
)
from umushroom.exceptions import DataError, InsufficientHistoryError, UMushroomError
from umushroom.factor_regression import (
    benchmark_alpha,
    fama_french_alpha,
    fetch_french_factors,
)
from umushroom.features import hierarchical_groups
from umushroom.logging_config import configure_logging, get_logger
from umushroom.model import RankingModel
from umushroom.montecarlo import combine_sleeves, simulate
from umushroom.prediction_log import evaluate_latest_predictions, log_predictions
from umushroom.purged_backtest import PurgedWalkForwardResult, run_purged_walk_forward
from umushroom.report import SleeveReport, build_payload, stock_snapshot, write_report
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
        "--weighting", choices=["equal", "hrp"], default="equal",
        help="Portfolio weighting scheme for today's allocation: "
             "'equal' (default, 1/N) or 'hrp' (Hierarchical Risk Parity).",
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

    orders = combined.copy()
    snap = stock_snapshot(market.prices, list(orders.index))
    orders = orders.join(snap)
    orders["name"] = [meta(t, "name") for t in orders.index]
    orders["sector"] = [meta(t, "sector").replace("-", " ") for t in orders.index]
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
            strat_parts.append((s.weight, r.period_returns if s.selection == "model"
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
              "rebalance_days": cfg.rebalance_days},
        sleeves=sleeves, orders=orders, top_ranked=top, deciles=deciles,
        mc=mc, mc_note=mc_note)
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
    if sleeve.name == EQUITY_SLEEVE:
        for w in check_concentration(listings, cfg.max_single_exchange_fraction):
            print(f"WARNING: {w}")
        if sleeve.max_per_sector:
            print(f"Sector cap ON: at most {sleeve.max_per_sector} picks per sector")
    print(f"Instruments: {len(cols)}")

    log_path = _sleeve_log_path(cfg, sleeve)
    if not args.skip_prediction_check:
        report = evaluate_latest_predictions(log_path, prices.iloc[-1], prices.index[-1])
        print("\n--- Previous Run's Prediction Accuracy ---")
        print(report.summary() if report is not None else
              "No prior predictions logged yet to evaluate (first run).")

    horizon = sleeve_horizon(cfg, sleeve)
    print(f"Forecast horizon: {horizon} trading days (rebalanced every "
          f"{cfg.rebalance_days})")
    X, y = features_mod.build_training_set(
        prices, forward_return_days=horizon, step=cfg.rebalance_days,
        min_history_days=cfg.min_history_days, target=cfg.target, groups=groups,
        dollar_volume=dvol)
    model = RankingModel(alpha=cfg.model_alpha, random_seed=cfg.random_seed)
    cv_stats = model.cross_validate(
        X, y, purge_dates=math.ceil(horizon / cfg.rebalance_days))
    model.fit(X, y)
    print("\n--- Feature importance (Ridge coefficients) ---")
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
                          dollar_volume=dvol)
        print("\n--- Quick Sanity-Check Backtest (IN-SAMPLE -- see warning) ---")
        print(bt.summary())

    info = SleeveReport(sleeve.name, sleeve.weight, sleeve.selection)
    if collect is not None:
        collect[sleeve.name] = info
    if not args.skip_purged_backtest:
        pwf = _report_out_of_sample(cfg, args, sleeve, prices, groups, dvol, sectors)
        if pwf is not None:
            info.backtest = pwf
            try:
                a = benchmark_alpha(pwf.period_returns, pwf.benchmark_period_returns,
                                    252 / cfg.rebalance_days)
                info.alpha_annual, info.alpha_t = a.alpha_annualized, a.alpha_tstat
            except ValueError:
                pass
        if args.compare_baseline and sleeve.name == EQUITY_SLEEVE:
            _compare_with_baseline(cfg, sleeve, prices, listings, groups, dvol)
        if args.compare_sector_cap and sleeve.name == EQUITY_SLEEVE:
            if sectors is None:
                print("\nSector-cap comparison skipped: no sector_file in config.")
            else:
                compare_sector_cap(cfg, sleeve, prices, groups, dvol, sectors,
                                   args.compare_sector_cap)
    if args.validate and sleeve.selection == "model":
        vr = _run_validation(cfg, args, sleeve, prices, listings, groups, dvol)
        info.validation = [(c.name, c.status, c.detail) for c in vr.checks]

    hold_all = sleeve.selection == "all"
    weighting = cast(WeightingScheme, sleeve.weighting or args.weighting)
    stale_here = sorted(market.stale_last & set(cols))
    if hold_all and stale_here:
        print(f"NOTE: no fresh close yet for {stale_here}; sizing them on the "
              f"previous close. Check their prices on UMushroom when entering.")
    if hold_all:
        print(f"\nSelection: holding ALL {len(cols)} instruments ({weighting} weight); "
              f"the model above is reported but does not pick.")
    alloc = generate_allocation(
        prices, model, top_n=sleeve.top_n, hold_all=hold_all,
        balance=cfg.starting_balance * sleeve.weight,
        weighting=weighting, groups=groups,
        local_prices=market.local_prices[cols],
        currencies={t: lst.currency for t, lst in listings.items()},
        fractional=sleeve.fractional,
        # crypto can legitimately trade at fractions of a cent; stocks and
        # ETFs below 0.10 in their own currency are treated as data errors
        min_local_price=0.0 if sleeve.fractional else MIN_LOCAL_PRICE,
        dollar_volume=dvol,
        # model-picked sleeves never trade on an old price (it would also be
        sectors=sectors, max_per_sector=sleeve.max_per_sector,
        # an old SIGNAL); hold-all sleeves just warn -- a day-old price only
        # shifts the share count slightly, it doesn't change what is held
        exclude=(frozenset() if hold_all
                 else frozenset(market.stale_last & set(cols))),
    )
    log_predictions(log_path=log_path, run_date=prices.index[-1],
                    all_scores=alloc.all_scores, prices=prices.iloc[-1],
                    picked_tickers=set(alloc.orders.index))
    return alloc


def _sleeve_groups(cfg: StrategyConfig, sleeve: SleeveConfig,
                   listings: dict) -> dict[str, str] | None:
    """Ranking groups: sector-within-exchange -> sector -> pooled (v4.4),
    or exchange only (v4), or one global group."""
    tickers = list(listings)
    exchange = {t: lst.exchange for t, lst in listings.items()}
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
                    groups: dict[str, str] | None,
                    dvol: pd.DataFrame | None) -> ValidationReport:
    kwargs: dict[str, Any] = {
        "test_window_days": cfg.purged_test_window_days,
        "purge_days": cfg.purged_purge_days,
        "embargo_days": cfg.purged_embargo_days,
        "rebalance_days": cfg.rebalance_days,
        "forward_return_days": sleeve_horizon(cfg, sleeve),
        "top_n": sleeve.top_n,
        "min_train_days": cfg.purged_min_train_days,
        "model_alpha": cfg.model_alpha, "random_seed": cfg.random_seed,
        "target": cfg.target, "groups": groups, "dollar_volume": dvol,
        # the live selection rule is what gets validated
        "sectors": (load_sectors(cfg.sector_file)
                    if sleeve.max_per_sector and cfg.sector_file is not None else None),
        "max_per_sector": sleeve.max_per_sector,
    }
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
                           groups: dict[str, str] | None,
                           dvol: pd.DataFrame | None) -> None:
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
    runs = {
        "v4.3 baseline (exch-neutral, 5d, top 10, no volume)": run_purged_walk_forward(
            prices, purge_days=max(base_fwd, cfg.rebalance_days),
            forward_return_days=base_fwd, top_n=min(10, prices.shape[1]),
            groups={t: lst.exchange for t, lst in listings.items()}, **common),
        f"current ({sleeve_horizon(cfg, sleeve)}d, top {sleeve.top_n}"
        f"{', sector' if sleeve.neutralize_by_sector else ''}"
        f"{', volume' if dvol is not None else ''})": run_purged_walk_forward(
            prices, purge_days=cfg.purged_purge_days,
            forward_return_days=sleeve_horizon(cfg, sleeve), top_n=sleeve.top_n,
            groups=groups, dollar_volume=dvol, **common),
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
                       prices: pd.DataFrame, groups: dict[str, str] | None,
                       dvol: pd.DataFrame | None, sectors: dict[str, str],
                       cap: int) -> CapVerdict:
    """v4.18: ONE pre-declared head-to-head on identical data -- the equity
    sleeve with no sector cap vs. at most `cap` picks per sector."""
    print("\n=== Sector cap: head-to-head on identical data ===")
    print(f"Rule (decided BEFORE the results): switch the cap on only if it keeps "
          f"at least {ALPHA_RETENTION:.0%} of the uncapped alpha AND does not raise "
          f"the weekly 95% VaR.")
    kw: dict[str, Any] = {
        "test_window_days": cfg.purged_test_window_days,
        "purge_days": cfg.purged_purge_days, "embargo_days": cfg.purged_embargo_days,
        "rebalance_days": cfg.rebalance_days,
        "forward_return_days": sleeve_horizon(cfg, sleeve), "top_n": sleeve.top_n,
        "min_train_days": cfg.purged_min_train_days, "model_alpha": cfg.model_alpha,
        "random_seed": cfg.random_seed, "target": cfg.target, "groups": groups,
        "dollar_volume": dvol, "record_cross_sections": True, "sectors": sectors}
    ppy = 252 / cfg.rebalance_days
    rows = {}
    for label, mps in (("no cap", None), (f"max {cap} per sector", cap)):
        r = run_purged_walk_forward(prices, max_per_sector=mps, **kw)
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


def _report_out_of_sample(cfg: StrategyConfig, args: argparse.Namespace,
                          sleeve: SleeveConfig, prices: pd.DataFrame,
                          groups: dict[str, str] | None,
                          dvol: pd.DataFrame | None = None,
                          sectors: dict[str, str] | None = None
                          ) -> PurgedWalkForwardResult | None:
    try:
        pwf = run_purged_walk_forward(
            prices, test_window_days=cfg.purged_test_window_days,
            purge_days=cfg.purged_purge_days, embargo_days=cfg.purged_embargo_days,
            rebalance_days=cfg.rebalance_days,
            forward_return_days=sleeve_horizon(cfg, sleeve), top_n=sleeve.top_n,
            min_train_days=cfg.purged_min_train_days, model_alpha=cfg.model_alpha,
            random_seed=cfg.random_seed, target=cfg.target, groups=groups,
            dollar_volume=dvol,
            record_cross_sections=bool(getattr(args, "report", None)),
            sectors=sectors, max_per_sector=sleeve.max_per_sector)
    except InsufficientHistoryError as e:
        print(f"\n--- Purged & Embargoed Walk-Forward Backtest ---\nSkipped: {e}")
        return None
    print("\n--- Purged & Embargoed Walk-Forward Backtest ---")
    print(pwf.summary())

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
