"""Diagnose tickers that Yahoo Finance can't find, and suggest fixes.

Run:  python -m umushroom.check_tickers --config config.yaml

Why tickers go missing (found on the real run of 2026-10-08):
  1. MAPPING: the UMushroom -> Yahoo symbol rule was wrong (fixed in v4.7
     for names like "J Sainsbury", "M&G", "V-ZUG").
  2. MOVED: the company moved its main listing (CRH, Ferguson, Indivior ->
     US). Yahoo dropped the London line; UMushroom prices it from the US
     listing. Fix with `yahoo_symbols` (+ `currency_overrides`).
  3. RENAMED / RE-TICKERED on Yahoo (e.g. Team17 -> Everplay).
  4. Yahoo simply lacks the instrument (some tiny Swiss/AIM names).

For every ticker with no data, this searches Yahoo by ISIN and by company
name, test-downloads each candidate, and prints a ready-to-paste config
snippet. It never edits config.yaml itself -- you review the suggestions.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from umushroom.config import load_config
from umushroom.universe import SUFFIX_MAP, resolve_listing

SearchFn = Callable[[str], list[dict]]
DownloadFn = Callable[[list[str]], pd.DataFrame]
CurrencyFn = Callable[[str], str | None]


@dataclass(frozen=True)
class TickerCheck:
    ticker: str
    name: str
    isin: str
    status: str                 # "ok" | "fixable" | "not found"
    suggestion: str | None = None
    suggestion_currency: str | None = None
    note: str = ""


def _yf_search(query: str) -> list[dict]:
    import yfinance as yf
    search_cls = getattr(yf, "Search", None)
    if search_cls is None:          # very old yfinance
        return []
    try:
        return list(search_cls(query, max_results=8).quotes or [])
    except Exception:  # noqa: BLE001 -- network/parse errors vary by version
        return []


def _yf_download(symbols: list[str]) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(symbols, period="1mo", interval="1d", auto_adjust=True,
                      progress=False, threads=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    closes = raw.get("Close", raw)
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(symbols[0])
    return closes


def _yf_currency(symbol: str) -> str | None:
    import yfinance as yf
    try:
        return yf.Ticker(symbol).fast_info.get("currency")
    except Exception:  # noqa: BLE001
        return None


def _has_data(df: pd.DataFrame, symbol: str) -> bool:
    return symbol in df.columns and df[symbol].notna().sum() >= 5


def is_otc_or_adr(symbol: str) -> bool:
    """US over-the-counter lines and ADRs (5 letters ending F or Y, e.g.
    PIERF, RHHBY). Their price level usually differs from the home share --
    RHHBY is 1/8 of a Roche share -- so they'd size orders wrongly."""
    return "." not in symbol and len(symbol) == 5 and symbol[-1] in "FY"


def _rank_candidates(quotes: list[dict], ticker: str) -> list[str]:
    """Equity candidates, preferring the same exchange suffix as the
    UMushroom listing, then US listings (for companies that moved)."""
    suffix = ticker.rsplit(".", 1)[1] if "." in ticker else ""
    syms: list[str] = [str(q["symbol"]) for q in quotes
                       if q.get("symbol")
                       and q.get("quoteType", "EQUITY") in ("EQUITY", "ETF")]
    same = [s for s in syms if s.endswith("." + suffix)] if suffix else []
    us = [s for s in syms if "." not in s]
    known = [s for s in syms if "." in s and s.rsplit(".", 1)[1] in SUFFIX_MAP]
    out: list[str] = []
    for s in same + us + known:
        if s not in out and s != ticker:
            out.append(s)
    # OTC/ADR lines last: only used if nothing else exists
    return [s for s in out if not is_otc_or_adr(s)] + [s for s in out if is_otc_or_adr(s)]


def check_tickers(
    tickers: list[str],
    meta: dict[str, dict[str, str]],
    yahoo_symbols: dict[str, str] | None = None,
    search: SearchFn = _yf_search,
    download: DownloadFn = _yf_download,
    currency: CurrencyFn = _yf_currency,
) -> list[TickerCheck]:
    yahoo_symbols = yahoo_symbols or {}
    fetch = {t: yahoo_symbols.get(t, t) for t in tickers}
    data = download(sorted(set(fetch.values())))
    results: list[TickerCheck] = []
    for t in tickers:
        m = meta.get(t, {})
        name, isin = m.get("name", ""), m.get("isin", "")
        if _has_data(data, fetch[t]):
            results.append(TickerCheck(t, name, isin, "ok"))
            continue
        quotes = (search(isin) if isin else []) + (search(name) if name else [])
        candidates = _rank_candidates(quotes, t)
        found = None
        if candidates:
            test = download(candidates)
            found = next((c for c in candidates if _has_data(test, c)), None)
        if found and is_otc_or_adr(found):
            results.append(TickerCheck(
                t, name, isin, "not found", found, currency(found),
                note="only an OTC/ADR line found -- its price level differs from "
                     "the real share, so it would size orders wrongly. Not "
                     "suggested; remove the ticker or find its home listing."))
        elif found:
            ccy = currency(found)
            results.append(TickerCheck(
                t, name, isin, "fixable", found, ccy,
                note="different Yahoo symbol (moved listing or renamed) -- check "
                     "its price matches UMushroom's before relying on it"))
        else:
            results.append(TickerCheck(
                t, name, isin, "not found",
                note="Yahoo has no data under any symbol found; remove it or "
                     "check the company's status"))
    return results


def config_snippet(results: list[TickerCheck],
                   currency_overrides: dict[str, str]) -> str:
    fixes = [r for r in results if r.status == "fixable"]
    if not fixes:
        return ""
    lines = ["yahoo_symbols:"]
    lines += [f'  "{r.ticker}": "{r.suggestion}"   # {r.name}' for r in fixes]
    ccy_lines = []
    for r in fixes:
        default = resolve_listing(r.ticker, currency_overrides).quote_currency
        if r.suggestion_currency and r.suggestion_currency != default:
            ccy_lines.append(f'  "{r.ticker}": "{r.suggestion_currency}"')
    if ccy_lines:
        lines += ["currency_overrides:  # add these to the existing section"] + ccy_lines
    return "\n".join(lines)


def _load_meta(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {r["yahoo"]: r for r in csv.DictReader(fh) if r.get("yahoo")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="umushroom-check-tickers",
        description="Find tickers Yahoo can't price and suggest replacements.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--meta", default=None,
                        help="CSV with yahoo,name,isin columns "
                             "(default: the config's sector_file)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    meta = _load_meta(Path(args.meta) if args.meta else cfg.sector_file)
    print(f"Checking {len(cfg.all_tickers)} tickers against Yahoo Finance "
          f"(about a minute)...")
    results = check_tickers(cfg.all_tickers, meta, cfg.yahoo_symbols)

    bad = [r for r in results if r.status != "ok"]
    print(f"\n{len(results) - len(bad)} OK, "
          f"{sum(r.status == 'fixable' for r in bad)} fixable, "
          f"{sum(r.status == 'not found' for r in bad)} not found\n")
    for r in bad:
        sug = f"-> {r.suggestion} ({r.suggestion_currency})" if r.suggestion else ""
        print(f"  {r.status:<10} {r.ticker:<12} {r.name[:40]:<40} {sug}")
        print(f"             {r.note}")
    snippet = config_snippet(results, cfg.currency_overrides)
    if snippet:
        print("\nSuggested config.yaml additions (check each on UMushroom first):\n")
        print(snippet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
