from __future__ import annotations

import pytest

from umushroom.exceptions import ConfigError
from umushroom.universe import (
    check_concentration,
    exchange_mix,
    fx_ticker,
    required_fx_tickers,
    resolve_listing,
    resolve_universe,
)

V3_SWISS_ONLY = ["NESN.SW", "NOVN.SW", "ROG.SW", "UBSG.SW", "ZURN.SW", "ABBN.SW"]


@pytest.mark.parametrize("ticker,exchange,ccy,divisor", [
    ("AAPL", "US", "USD", 1.0),
    ("BRK-B", "US", "USD", 1.0),
    ("AZN.L", "LSE", "GBP", 100.0),      # London quotes are in PENCE
    ("NESN.SW", "SIX", "CHF", 1.0),
    ("SAP.DE", "XETRA", "EUR", 1.0),
    ("ASML.AS", "Euronext Amsterdam", "EUR", 1.0),
])
def test_resolve_listing(ticker, exchange, ccy, divisor):
    lst = resolve_listing(ticker)
    assert (lst.exchange, lst.currency, lst.price_divisor) == (exchange, ccy, divisor)


def test_currency_override_for_usd_line_on_london():
    lst = resolve_listing("XYZ.L", {"XYZ.L": "USD"})
    assert lst.currency == "USD" and lst.price_divisor == 1.0


@pytest.mark.parametrize("bad", ["FOO.XX", "^FTSE", "GBPUSD=X"])
def test_rejects_unknown_suffix_and_non_equities(bad):
    with pytest.raises(ConfigError):
        resolve_listing(bad)


def test_required_fx_tickers_excludes_base_currency():
    listings = resolve_universe(["AAPL", "AZN.L", "NESN.SW", "SHEL.L"])
    assert required_fx_tickers(listings, "GBP") == ["CHFGBP=X", "USDGBP=X"]
    assert fx_ticker("USD", "GBP") == "USDGBP=X"


def test_v3_swiss_only_universe_is_flagged():
    """Regression test for the v3 problem: a single-exchange universe
    must produce a concentration warning, never pass silently."""
    listings = resolve_universe(V3_SWISS_ONLY)
    assert exchange_mix(listings) == {"SIX": 1.0}
    warnings = check_concentration(listings, max_single_exchange_fraction=0.6)
    assert len(warnings) == 1 and "SIX" in warnings[0]


def test_balanced_universe_not_flagged():
    listings = resolve_universe(["AAPL", "MSFT", "AZN.L", "SHEL.L", "NESN.SW"])
    assert check_concentration(listings, 0.6) == []


def test_crypto_tickers_resolve_to_crypto_usd():
    lst = resolve_listing("BTC-USD")
    assert (lst.exchange, lst.currency, lst.price_divisor) == ("Crypto", "USD", 1.0)
    # a US share class with a hyphen must NOT be mistaken for crypto
    assert resolve_listing("BRK-B").exchange == "US"


def test_mapped_universe_has_no_name_letter_share_classes():
    """v4.7 regression: 'J Sainsbury' must map to SBRY.L, not SBRY-J.L."""
    import csv
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "umushroom_universe_mapped.csv"
    with open(path, newline="") as fh:
        rows = {r["urlName"]: r["yahoo"] for r in csv.DictReader(fh)}
    assert rows["sbry-j-sainsbury"] == "SBRY.L"
    assert rows["mng-m-g"] == "MNG.L"
    assert rows["vzug-v-zug-hldg"] == "VZUG.SW"
    assert rows["brk-b-berkshire-hath"] == "BRK-B"      # real class kept
    assert rows["bt-a-bt-group"] == "BT-A.L"            # real class kept
