"""ETF universe: UMushroom's catalogue -> a Yahoo-priced, tradeable list.

Three steps, each one testable offline with injected fetchers:

  1. CATALOGUE  (umushroom/catalogue.py, or the committed snapshot
     `umushroom_etf_catalogue.json`): every ETF UMushroom lists, with
     ISIN, UMushroom ticker, pricing currency and asset class.
  2. CANDIDATES: for every ISIN, the Yahoo Finance listings that carry
     it (Yahoo's symbol search by ISIN), ordered by how useful they are
     for a GBP account: London first, then the big European venues, then
     the US, with the listing currency UMushroom prices in preferred.
     When the search finds nothing usable, UMushroom's own ticker is
     tried with the usual exchange suffixes -- but such a GUESS is only
     accepted if Yahoo's fund name matches UMushroom's.
  3. VERIFY: a candidate counts only if Yahoo actually serves a daily
     price history for it (currency known, enough bars, still trading).

The result is `umushroom_etf_universe.csv`, one row per catalogue ETF
with a `status` column. Only `status == ok` rows become sleeve tickers;
the rest stay in the file so a human can see WHY an ETF is not traded
(no Yahoo listing, stale line, name mismatch) instead of it vanishing.

Why the ISIN mapping is this careful: a wrong symbol is the one error
that silently trades the wrong instrument. Every accepted row records
which rule accepted it (`via`), Yahoo's own name, and a name-similarity
score, so the mapping can be audited line by line.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from umushroom.exceptions import ConfigError, DataError
from umushroom.logging_config import get_logger
from umushroom.universe import SUFFIX_MAP

logger = get_logger("etf_universe")

STATUS_OK = "ok"
MIN_HISTORY_DAYS_DEFAULT = 60       # a listing with fewer bars is not modelable
MAX_STALE_DAYS_DEFAULT = 14         # last Yahoo close older than this = dead line
GUESS_NAME_SIMILARITY = 0.5         # a ticker GUESS must match the fund name this well
WEAK_NAME_SIMILARITY = 0.3          # below this an ISIN-search hit is flagged for review

# Yahoo search "exchange" code -> symbol suffix (what we trade on / fetch).
EXCHANGE_SUFFIX: dict[str, str] = {
    "LSE": "L", "IOB": "IL", "GER": "DE", "FRA": "F", "MUN": "MU", "EBS": "SW",
    "MIL": "MI", "PAR": "PA", "AMS": "AS", "LUX": "LU", "STO": "ST", "CPH": "CO",
    "HEL": "HE", "OSL": "OL", "MCE": "MC", "VIE": "VI", "TOR": "TO", "JPX": "T",
    "NMS": "", "NYQ": "", "PCX": "", "NGM": "", "ASE": "", "BTS": "",
    "STU": "SG",
}
# Venues an ETF line may be taken from: the price layer must know the suffix
# (universe.SUFFIX_MAP) AND it must be a market a UK account would plausibly
# trade -- a Tokyo or Hong Kong line of a European ETF is never the right
# pick, however the search ranks it.
SUPPORTED_SUFFIXES = frozenset(SUFFIX_MAP) - {"T", "HK", "AX", "TO"}
# Quote types Yahoo uses for exchange-traded products. ETCs/ETNs often come
# back as EQUITY; Stuttgart "ISIN.SG" fund quotes as MUTUALFUND.
TYPE_RANK = {"ETF": 0, "EQUITY": 1, "MUTUALFUND": 2}
# Listing-venue preference by the currency UMushroom prices the ETF in.
# Earlier = better. London first for a GBP account whenever it exists.
VENUE_PREFERENCE: dict[str, list[str]] = {
    "GBP": ["L", "IL", "DE", "SW", "MI", "PA", "AS", "", "MU", "LU"],
    "USD": ["L", "IL", "", "SW", "DE", "MI", "PA", "AS", "MU", "LU"],
    "EUR": ["DE", "MI", "PA", "AS", "L", "SW", "MU", "LU", "IL", ""],
    "CHF": ["SW", "DE", "L", "MI", "PA", "AS", "MU", "LU", "IL", ""],
}
DEFAULT_VENUE_PREFERENCE = ["L", "IL", "DE", "SW", "MI", "PA", "AS", "", "MU", "LU",
                            "ST", "CO", "HE", "OL", "MC", "VI", "F"]
GUESS_SUFFIXES = ["L", "DE", "SW", "MI", "PA", "AS"]
TICKER_RE = re.compile(r"^[A-Z0-9]{1,8}$")
_STOP = {"ucits", "etf", "etc", "etp", "etn", "acc", "dist", "distributing",
         "accumulating", "accumulation", "usd", "eur", "gbp", "chf", "the", "plc",
         "fund", "funds", "class", "share", "shares", "ii", "securities", "a", "c", "d",
         "1c", "1d", "2c", "2d", "hedged", "daily", "index", "tr", "inc", "ltd"}


# ------------------------------------------------------------------ types

@dataclass(frozen=True)
class CatalogueEtf:
    url_name: str
    name: str
    isin: str
    symbol: str            # UMushroom's own ticker (may be empty)
    currency: str          # currency UMushroom prices it in
    asset_class: str
    style: str = ""
    manager: str = ""
    price: float | None = None
    price_date: str = ""
    is_active: bool = True


@dataclass(frozen=True)
class Quote:
    """One Yahoo symbol-search hit."""
    symbol: str
    exchange: str          # Yahoo exchange code (LSE, GER, ...)
    quote_type: str        # ETF | EQUITY | MUTUALFUND | ...
    name: str = ""


@dataclass(frozen=True)
class Verified:
    """What Yahoo serves for a symbol (None from the verifier = nothing)."""
    symbol: str
    currency: str
    exchange: str = ""
    name: str = ""
    history_days: int = 0
    first_date: str = ""
    last_date: str = ""
    last_close: float | None = None


SearchFn = Callable[[str], list[Quote]]
VerifyFn = Callable[[str], "Verified | None"]

UNIVERSE_COLUMNS = [
    "yahoo", "isin", "name", "umushroom_symbol", "umushroom_currency",
    "yahoo_currency", "yahoo_exchange", "yahoo_name", "asset_class", "style",
    "manager", "url_name", "umushroom_price", "price_date", "history_days",
    "first_date", "last_date", "name_similarity", "via", "status", "note",
    "leveraged",
]

# Leveraged / inverse / short exchange-traded products, recognised from the
# name. They are ranked out of the model universe by default (config
# `exclude_leveraged`): a 3x daily product dominates every trend factor and
# its path-dependent decay is nothing the factors describe. "Short duration"
# bond funds and similar are NOT leveraged, so those phrases are stripped
# before the test.
_NOT_LEVERAGE = re.compile(r"(short[\s-]*(?:duration|maturity|dated|term)|ultra[\s-]*short"
                           r"|short-?dated|0-[0-9]|[0-9]-[0-9] ?(?:yr|year))", re.IGNORECASE)
_LEVERAGE = re.compile(r"(\b-?[1-9]x\b|\b[1-9]x[\s-]|leverag|\bshort\b|inverse|\bbear\b"
                       r"|\bbull\b|daily[\s-]+(?:long|short)|\bultra\b)", re.IGNORECASE)


def leveraged_or_inverse(name: str) -> bool:
    cleaned = _NOT_LEVERAGE.sub(" ", name or "")
    return bool(_LEVERAGE.search(cleaned))


# ------------------------------------------------------------- catalogue

def load_catalogue(path: str | Path) -> list[CatalogueEtf]:
    """Read the scraped catalogue snapshot (see catalogue.py for its shape)."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"Catalogue file not found: {path}")
    raw = json.loads(path.read_text())
    rows = raw["etfs"] if isinstance(raw, dict) else raw
    out: list[CatalogueEtf] = []
    seen: set[str] = set()
    for r in rows:
        isin = str(r.get("isin") or "").strip().upper()
        if not isin or isin in seen:
            continue
        seen.add(isin)
        price = r.get("price")
        out.append(CatalogueEtf(
            url_name=str(r.get("urlName") or r.get("url_name") or ""),
            name=str(r.get("name") or ""),
            isin=isin,
            symbol=str(r.get("symbol") or "").strip().upper(),
            currency=str(r.get("currency") or "").strip().upper(),
            asset_class=str(r.get("assetClass") or r.get("asset_class") or "other"),
            style=str(r.get("style") or ""),
            manager=str(r.get("manager") or ""),
            price=float(price) if price not in (None, "") else None,
            price_date=str(r.get("priceDate") or r.get("price_date") or ""),
            is_active=bool(r.get("isActive", True)),
        ))
    return out


# ------------------------------------------------------------ candidates

def _tokens(name: str) -> set[str]:
    words = re.findall(r"[a-z0-9&]+", name.lower().replace("&", " and "))
    return {w for w in words if w not in _STOP and len(w) > 1}


def name_similarity(a: str, b: str) -> float:
    """Jaccard overlap of the meaningful words in two fund names."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _suffix(symbol: str) -> str:
    return symbol.rsplit(".", 1)[1] if "." in symbol else ""


def venue_rank(suffix: str, umushroom_currency: str) -> int:
    pref = VENUE_PREFERENCE.get(umushroom_currency, DEFAULT_VENUE_PREFERENCE)
    if suffix in pref:
        return pref.index(suffix)
    if suffix in DEFAULT_VENUE_PREFERENCE:
        return 20 + DEFAULT_VENUE_PREFERENCE.index(suffix)
    return 50


def candidate_symbols(etf: CatalogueEtf, quotes: list[Quote],
                      prefer: frozenset[str] = frozenset()) -> list[tuple[str, str]]:
    """Ordered (symbol, via) pairs to try for one ETF. `via` is "search"
    for Yahoo ISIN-search hits and "guess" for UMushroom-ticker guesses.

    Venue preference comes first (a London line beats a Milan line for a
    GBP account whatever Yahoo calls it), then quote type (ETF before the
    EQUITY type Yahoo gives many ETCs). Symbols in `prefer` -- the lines
    the team already trades -- jump the queue, so a rebuild can never
    silently switch a core holding to another line of the same fund."""
    scored: list[tuple[tuple[int, int, int], str]] = []
    seen: set[str] = set()
    for q in quotes:
        sym = q.symbol.strip().upper()
        suf = _suffix(sym)
        if not sym or sym in seen or suf not in SUPPORTED_SUFFIXES:
            continue
        if q.quote_type not in TYPE_RANK:
            continue
        seen.add(sym)
        scored.append(((0 if sym in prefer else 1, venue_rank(suf, etf.currency),
                        TYPE_RANK[q.quote_type]), sym))
    ordered = [(sym, "search") for _k, sym in sorted(scored)]
    if TICKER_RE.match(etf.symbol or ""):
        guesses = []
        for suf in sorted(GUESS_SUFFIXES, key=lambda s: venue_rank(s, etf.currency)):
            sym = f"{etf.symbol}.{suf}"
            if sym not in seen:
                seen.add(sym)
                guesses.append((sym, "guess"))
        # a preferred (already-traded) line that the search missed goes first
        front = [g for g in guesses if g[0] in prefer]
        ordered = front + ordered + [g for g in guesses if g[0] not in prefer]
    return ordered


# ----------------------------------------------------------------- resolve

@dataclass
class UniverseRow:
    yahoo: str = ""
    isin: str = ""
    name: str = ""
    umushroom_symbol: str = ""
    umushroom_currency: str = ""
    yahoo_currency: str = ""
    yahoo_exchange: str = ""
    yahoo_name: str = ""
    asset_class: str = ""
    style: str = ""
    manager: str = ""
    url_name: str = ""
    umushroom_price: float | None = None
    price_date: str = ""
    history_days: int = 0
    first_date: str = ""
    last_date: str = ""
    name_similarity: float | None = None
    via: str = ""
    status: str = ""
    note: str = ""
    leveraged: bool = False


def _parse_date(s: str) -> date | None:
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc).date()
    except (TypeError, ValueError):
        return None


def _today() -> date:
    return datetime.now(timezone.utc).date()


def resolve_etf(etf: CatalogueEtf, quotes: list[Quote], verify: VerifyFn,
                as_of: date, min_history_days: int = MIN_HISTORY_DAYS_DEFAULT,
                max_stale_days: int = MAX_STALE_DAYS_DEFAULT,
                prefer: frozenset[str] = frozenset()) -> UniverseRow:
    """Pick the first candidate Yahoo serves usable data for."""
    row = UniverseRow(isin=etf.isin, name=etf.name, umushroom_symbol=etf.symbol,
                      umushroom_currency=etf.currency, asset_class=etf.asset_class,
                      style=etf.style, manager=etf.manager, url_name=etf.url_name,
                      umushroom_price=etf.price, price_date=etf.price_date,
                      leveraged=leveraged_or_inverse(etf.name))
    if not etf.is_active:
        row.status, row.note = "inactive", "UMushroom marks it inactive"
        return row
    cands = candidate_symbols(etf, quotes, prefer)
    if not cands:
        row.status, row.note = "no_candidates", "no Yahoo listing found for the ISIN"
        return row
    rejected: list[str] = []
    for sym, via in cands:
        v = verify(sym)
        if v is None or not v.currency:
            rejected.append(f"{sym}: no data")
            continue
        if v.history_days < min_history_days:
            rejected.append(f"{sym}: only {v.history_days} bars")
            continue
        last = _parse_date(v.last_date)
        if last is not None and (as_of - last).days > max_stale_days:
            rejected.append(f"{sym}: last close {v.last_date} is stale")
            continue
        sim = name_similarity(etf.name, v.name) if v.name else None
        if via == "guess" and (sim is None or sim < GUESS_NAME_SIMILARITY):
            rejected.append(f"{sym}: guessed ticker, name mismatch ({v.name!r})")
            continue
        row.yahoo, row.via = sym, via
        row.yahoo_currency, row.yahoo_exchange, row.yahoo_name = v.currency, v.exchange, v.name
        row.history_days, row.first_date, row.last_date = v.history_days, v.first_date, v.last_date
        row.name_similarity = None if sim is None else round(sim, 3)
        row.status = STATUS_OK
        notes = []
        if sim is not None and sim < WEAK_NAME_SIMILARITY:
            notes.append("weak name match -- check on UMushroom")
        if v.currency.replace("GBp", "GBP") != etf.currency:
            notes.append(f"Yahoo line is in {v.currency}, UMushroom shows {etf.currency}")
        row.note = "; ".join(notes)
        return row
    row.status = "no_yahoo_data"
    row.note = "; ".join(rejected)[:300]
    return row


def build_universe(catalogue: list[CatalogueEtf], search: SearchFn, verify: VerifyFn,
                   as_of: date, min_history_days: int = MIN_HISTORY_DAYS_DEFAULT,
                   max_stale_days: int = MAX_STALE_DAYS_DEFAULT,
                   progress_every: int = 200,
                   prefer: frozenset[str] = frozenset()) -> pd.DataFrame:
    rows: list[dict] = []
    taken: dict[str, str] = {}
    for i, etf in enumerate(catalogue, start=1):
        try:
            quotes = search(etf.isin)
        except Exception as e:  # noqa: BLE001 -- network errors vary by backend
            logger.warning(f"search failed for {etf.isin}: {e}")
            quotes = []
        row = resolve_etf(etf, quotes, verify, as_of, min_history_days, max_stale_days,
                          prefer)
        if row.status == STATUS_OK and row.yahoo in taken:
            # two catalogue entries mapping to one Yahoo line (e.g. share classes
            # that Yahoo merges) must never become two orders for one instrument
            row.status = "duplicate_symbol"
            row.note = f"{row.yahoo} already used by {taken[row.yahoo]}"
        elif row.status == STATUS_OK:
            taken[row.yahoo] = etf.isin
        rows.append(asdict(row))
        if progress_every and i % progress_every == 0:
            logger.info(f"resolved {i}/{len(catalogue)} ETFs")
    df = pd.DataFrame(rows, columns=UNIVERSE_COLUMNS)
    ok = int((df["status"] == STATUS_OK).sum())
    logger.info(f"ETF universe: {ok}/{len(df)} catalogue entries priced on Yahoo")
    return df


def write_universe(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    df[UNIVERSE_COLUMNS].sort_values(["status", "asset_class", "name"]).to_csv(
        path, index=False)
    return path


# ------------------------------------------------------------------ load

@dataclass(frozen=True)
class EtfUniverse:
    rows: pd.DataFrame
    path: Path

    @property
    def tickers(self) -> list[str]:
        ok = self.rows[self.rows["status"] == STATUS_OK]
        return sorted(ok["yahoo"].astype(str))

    @property
    def leveraged(self) -> set[str]:
        """Yahoo symbols flagged as leveraged / inverse / short products."""
        ok = self.rows[self.rows["status"] == STATUS_OK]
        col = ok["leveraged"] if "leveraged" in ok.columns else pd.Series("", index=ok.index)
        flags = col.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
        return {str(t) for t, f in zip(ok["yahoo"], flags) if f}

    @property
    def groups(self) -> dict[str, str]:
        ok = self.rows[self.rows["status"] == STATUS_OK]
        return {str(t): str(a or "other")
                for t, a in zip(ok["yahoo"], ok["asset_class"].fillna("other"))}

    @property
    def currency_overrides(self) -> dict[str, str]:
        """Listings whose Yahoo currency differs from their exchange's usual
        one (e.g. a USD line on London) -- exactly what data.py needs."""
        out: dict[str, str] = {}
        ok = self.rows[self.rows["status"] == STATUS_OK]
        for t, ccy in zip(ok["yahoo"], ok["yahoo_currency"]):
            t, ccy = str(t), str(ccy or "")
            suf = _suffix(t)
            if not ccy or suf not in SUFFIX_MAP:
                continue
            if ccy != SUFFIX_MAP[suf][1]:
                out[t] = ccy
        return out

    @property
    def meta(self) -> dict[str, dict[str, str]]:
        cols = ["name", "umushroom_symbol", "umushroom_currency", "asset_class",
                "isin", "yahoo_currency", "style"]
        ok = self.rows[self.rows["status"] == STATUS_OK]
        return {str(r["yahoo"]): {c: ("" if pd.isna(r[c]) else str(r[c])) for c in cols}
                for _, r in ok.iterrows()}

    @property
    def status_counts(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.rows["status"].value_counts().items()}

    def summary(self) -> str:
        counts = ", ".join(f"{k} {v}" for k, v in self.status_counts.items())
        by_class = self.rows[self.rows["status"] == STATUS_OK]["asset_class"].value_counts()
        classes = ", ".join(f"{k} {v}" for k, v in by_class.items())
        return (f"ETF universe {self.path.name}: {len(self.tickers)} tradeable of "
                f"{len(self.rows)} catalogued ({counts}); by asset class: {classes}; "
                f"leveraged/inverse: {len(self.leveraged)}")


def load_universe(path: str | Path) -> EtfUniverse:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"ETF universe file not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ("yahoo", "isin", "status", "asset_class") if c not in df.columns]
    if missing:
        raise ConfigError(f"ETF universe file {path} is missing columns {missing}")
    if "history_days" not in df.columns:
        df["history_days"] = 0
    df["history_days"] = pd.to_numeric(df["history_days"], errors="coerce").fillna(0)
    ok = df[df["status"] == STATUS_OK]
    if ok["yahoo"].duplicated().any():
        dupes = sorted(ok.loc[ok["yahoo"].duplicated(), "yahoo"].unique())
        raise ConfigError(f"ETF universe file {path} lists a Yahoo symbol twice: {dupes}")
    return EtfUniverse(rows=df, path=path)


# ------------------------------------------------------- live (yfinance)

def yahoo_search(isin: str) -> list[Quote]:
    import yfinance as yf
    search_cls = getattr(yf, "Search", None)
    if search_cls is None:
        return []
    try:
        quotes = list(search_cls(isin, max_results=10, news_count=0).quotes or [])
    except Exception:  # noqa: BLE001
        return []
    return [Quote(symbol=str(q.get("symbol", "")), exchange=str(q.get("exchange", "")),
                  quote_type=str(q.get("quoteType", "")),
                  name=str(q.get("longname") or q.get("shortname") or ""))
            for q in quotes]


def yahoo_verify(symbol: str) -> Verified | None:
    import yfinance as yf
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="max", interval="1d", auto_adjust=True)
    except Exception:  # noqa: BLE001
        return None
    if hist is None or hist.empty or "Close" not in hist:
        return None
    closes = hist["Close"].dropna()
    if closes.empty:
        return None
    try:
        info = t.fast_info
        currency = str(info.get("currency") or "")
        exchange = str(info.get("exchange") or "")
    except Exception:  # noqa: BLE001
        currency, exchange = "", ""
    try:
        name = str((t.info or {}).get("longName") or (t.info or {}).get("shortName") or "")
    except Exception:  # noqa: BLE001 -- yfinance .info is best-effort
        name = ""
    return Verified(symbol=symbol, currency=currency, exchange=exchange, name=name,
                    history_days=len(closes),
                    first_date=str(closes.index[0].date()),
                    last_date=str(closes.index[-1].date()),
                    last_close=float(closes.iloc[-1]))


# -------------------------------------------------------- replay helpers

def replay_search(candidates_path: str | Path) -> SearchFn:
    """Search function that replays a saved {isin: [quote, ...]} file."""
    raw = json.loads(Path(candidates_path).read_text())
    table = raw.get("candidates", raw)

    def search(isin: str) -> list[Quote]:
        return [Quote(symbol=str(c.get("s") or c.get("symbol") or ""),
                      exchange=str(c.get("x") or c.get("exchange") or ""),
                      quote_type=str(c.get("t") or c.get("quote_type") or ""),
                      name=str(c.get("n") or c.get("name") or ""))
                for c in table.get(isin, [])]
    return search


def replay_verify(verified_path: str | Path) -> VerifyFn:
    """Verify function that replays a saved {symbol: meta} file."""
    raw = json.loads(Path(verified_path).read_text())
    table = raw.get("verified", raw)

    def verify(symbol: str) -> Verified | None:
        m = table.get(symbol)
        if not m:
            return None
        return Verified(symbol=symbol, currency=str(m.get("currency") or m.get("cur") or ""),
                        exchange=str(m.get("exchange") or m.get("x") or ""),
                        name=str(m.get("name") or m.get("ln") or m.get("sn") or ""),
                        history_days=int(m.get("history_days") or m.get("n") or 0),
                        first_date=str(m.get("first_date") or m.get("first") or ""),
                        last_date=str(m.get("last_date") or m.get("last") or ""),
                        last_close=m.get("last_close"))
    return verify


def preferred_from_config(path: str | Path) -> set[str]:
    """Core/listed tickers of every non-equity sleeve in a config file, read
    from the raw YAML so a missing universe file cannot block the build."""
    import yaml
    try:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    out: set[str] = set()
    for spec in (raw.get("sleeves") or {}).values():
        if isinstance(spec, dict):
            for key in ("core_tickers", "tickers"):
                out |= {str(t).upper() for t in (spec.get(key) or [])}
    return out


def catalogue_as_of(path: str | Path) -> date | None:
    """The scrape date recorded in a catalogue snapshot, if any."""
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    stamp = raw.get("scraped_at") if isinstance(raw, dict) else None
    if not stamp:
        return None
    d = _parse_date(str(stamp))
    return d if d is not None and d <= _today() + timedelta(days=1) else None


# ------------------------------------------------------------------- CLI

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="umushroom-etf-universe",
        description="Map UMushroom's ETF catalogue to Yahoo Finance symbols.")
    p.add_argument("--catalogue", default="umushroom_etf_catalogue.json")
    p.add_argument("--out", default="umushroom_etf_universe.csv")
    p.add_argument("--candidates", default=None,
                   help="Replay saved Yahoo search results instead of searching live.")
    p.add_argument("--verified", default=None,
                   help="Replay saved Yahoo verification results instead of fetching.")
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--min-history-days", type=int, default=MIN_HISTORY_DAYS_DEFAULT)
    p.add_argument("--max-stale-days", type=int, default=MAX_STALE_DAYS_DEFAULT)
    p.add_argument("--limit", type=int, default=None, help="Only the first N ETFs (testing).")
    p.add_argument("--prefer", nargs="*", default=[],
                   help="Yahoo symbols to keep whenever they carry the ISIN (the lines "
                        "already traded, e.g. the core ETFs in config.yaml).")
    p.add_argument("--config", default=None,
                   help="Read --prefer from this config's sleeves (their core_tickers and "
                        "tickers).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        catalogue = load_catalogue(args.catalogue)
    except DataError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if args.limit:
        catalogue = catalogue[: args.limit]
    as_of = _parse_date(args.as_of) if args.as_of else _today()
    if as_of is None:
        print("ERROR: --as-of must be YYYY-MM-DD", file=sys.stderr)
        return 2
    search = replay_search(args.candidates) if args.candidates else yahoo_search
    verify = replay_verify(args.verified) if args.verified else yahoo_verify
    prefer = {str(t).upper() for t in args.prefer}
    if args.config:
        prefer |= preferred_from_config(args.config)
    df = build_universe(catalogue, search, verify, as_of,
                        min_history_days=args.min_history_days,
                        max_stale_days=args.max_stale_days, prefer=frozenset(prefer))
    out = write_universe(df, args.out)
    uni = load_universe(out)
    print(uni.summary())
    print(f"Written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

