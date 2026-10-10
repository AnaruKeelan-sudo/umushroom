"""Universe metadata: which exchange and currency each ticker trades in.

Why this module exists (v4): v3's config.yaml contained only 20 Swiss
SMI tickers (".SW" suffix), so every prediction was necessarily a Swiss
stock -- the model can only rank what it is given. v4 makes the
universe multi-market, which raises two problems this module solves:

  1. CURRENCY. Prices come back in each listing's local currency.
     Sizing a GBP balance against a USD or CHF price silently buys the
     wrong number of shares. Worse, London (".L") prices on Yahoo are
     quoted in PENCE (GBp), not pounds -- a 100x sizing error if not
     handled. Every price is converted to the account's base currency
     before anything else touches it.

  2. CONCENTRATION. The pipeline now reports the exchange mix of the
     universe and warns loudly when one exchange dominates, so a
     single-market universe can never again go unnoticed.

Exchange/currency are inferred from the Yahoo ticker suffix (a
deterministic, offline-testable rule). Any ticker that doesn't follow
its exchange's usual currency (e.g. a USD-quoted line on London) can be
pinned with `currency_overrides` in config.yaml.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from umushroom.exceptions import ConfigError
from umushroom.logging_config import get_logger

logger = get_logger("universe")

# Yahoo suffix -> (exchange name, quote currency). "GBp" = pence.
SUFFIX_MAP: dict[str, tuple[str, str]] = {
    "": ("US", "USD"),
    "L": ("LSE", "GBp"),
    "SW": ("SIX", "CHF"),
    "DE": ("XETRA", "EUR"),
    "F": ("Frankfurt", "EUR"),
    "PA": ("Euronext Paris", "EUR"),
    "AS": ("Euronext Amsterdam", "EUR"),
    "BR": ("Euronext Brussels", "EUR"),
    "MI": ("Borsa Italiana", "EUR"),
    "MC": ("BME Madrid", "EUR"),
    "LS": ("Euronext Lisbon", "EUR"),
    "IR": ("Euronext Dublin", "EUR"),
    "HE": ("Nasdaq Helsinki", "EUR"),
    "VI": ("Vienna", "EUR"),
    "ST": ("Nasdaq Stockholm", "SEK"),
    "CO": ("Nasdaq Copenhagen", "DKK"),
    "OL": ("Oslo", "NOK"),
    "TO": ("Toronto", "CAD"),
    "T": ("Tokyo", "JPY"),
    "HK": ("Hong Kong", "HKD"),
    "AX": ("ASX", "AUD"),
}

SUPPORTED_BASE_CURRENCIES = {"GBP", "USD", "EUR", "CHF"}


@dataclass(frozen=True)
class Listing:
    ticker: str
    exchange: str
    quote_currency: str      # as quoted by Yahoo, may be "GBp"

    @property
    def currency(self) -> str:
        """ISO currency (GBp -> GBP)."""
        return "GBP" if self.quote_currency == "GBp" else self.quote_currency

    @property
    def price_divisor(self) -> float:
        """Divide the quoted price by this to get a price in `currency`."""
        return 100.0 if self.quote_currency == "GBp" else 1.0


CRYPTO_PATTERN = re.compile(r"^[A-Z0-9]+-USD$")   # Yahoo crypto, e.g. BTC-USD


def _suffix(ticker: str) -> str:
    # Yahoo FX/index symbols ("GBPUSD=X", "^FTSE") are not equities.
    if ticker.startswith("^") or ticker.endswith("=X"):
        raise ConfigError(f"{ticker!r} is an index/FX symbol, not a tradable equity")
    return ticker.rsplit(".", 1)[1] if "." in ticker else ""


def resolve_listing(ticker: str, currency_overrides: dict[str, str] | None = None
                    ) -> Listing:
    overrides = currency_overrides or {}
    if CRYPTO_PATTERN.match(ticker):
        return Listing(ticker=ticker, exchange="Crypto", quote_currency="USD")
    suffix = _suffix(ticker)
    # US share classes use a hyphen on Yahoo (BRK-B); a dotted suffix we
    # don't recognise is almost certainly a typo or unsupported market.
    if suffix not in SUFFIX_MAP:
        raise ConfigError(
            f"Unrecognised exchange suffix '.{suffix}' on ticker {ticker!r}. "
            f"Known suffixes: {sorted(s for s in SUFFIX_MAP if s)}"
        )
    exchange, quote_ccy = SUFFIX_MAP[suffix]
    if ticker in overrides:
        quote_ccy = overrides[ticker]
    return Listing(ticker=ticker, exchange=exchange, quote_currency=quote_ccy)


def resolve_universe(tickers: list[str],
                     currency_overrides: dict[str, str] | None = None
                     ) -> dict[str, Listing]:
    return {t: resolve_listing(t, currency_overrides) for t in tickers}


def fx_ticker(from_ccy: str, to_ccy: str) -> str:
    """Yahoo symbol whose price = units of `to_ccy` per 1 `from_ccy`."""
    return f"{from_ccy}{to_ccy}=X"


def required_fx_tickers(listings: dict[str, Listing], base_currency: str) -> list[str]:
    ccys = sorted({lst.currency for lst in listings.values()} - {base_currency})
    return [fx_ticker(c, base_currency) for c in ccys]


def exchange_mix(listings: dict[str, Listing]) -> dict[str, float]:
    counts = Counter(lst.exchange for lst in listings.values())
    total = sum(counts.values())
    return {ex: n / total for ex, n in counts.most_common()}


def check_concentration(listings: dict[str, Listing],
                        max_single_exchange_fraction: float) -> list[str]:
    """Return warnings (and log them) if the universe is dominated by
    one exchange. Returned rather than raised: a deliberately
    single-market universe is legitimate, but it must be visible."""
    mix = exchange_mix(listings)
    mix_str = ", ".join(f"{ex} {frac:.0%}" for ex, frac in mix.items())
    logger.info(f"Universe: {len(listings)} tickers across {len(mix)} "
                f"exchange(s): {mix_str}")
    warnings = []
    for ex, frac in mix.items():
        if frac > max_single_exchange_fraction:
            msg = (f"{frac:.0%} of the universe is listed on {ex} "
                   f"(threshold {max_single_exchange_fraction:.0%}). "
                   f"Every pick will be concentrated in one market.")
            logger.warning(msg)
            warnings.append(msg)
    return warnings
