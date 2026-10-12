"""ETF universe: catalogue -> Yahoo listing -> tradeable universe file.
Every network call is replaced by an in-memory fake."""
from __future__ import annotations

import json
from datetime import date

import pytest

from umushroom import etf_universe as eu
from umushroom.exceptions import ConfigError

AS_OF = date(2026, 10, 12)


def _etf(**kw) -> eu.CatalogueEtf:
    base = {"url_name": "x", "name": "iShares Core MSCI World UCITS ETF USD (Acc)",
            "isin": "IE00B4L5Y983", "symbol": "IWDA", "currency": "USD",
            "asset_class": "equity", "style": "Large-cap", "manager": "BlackRock",
            "price": 147.57, "price_date": "2026-10-12", "is_active": True}
    base.update(kw)
    return eu.CatalogueEtf(**base)


def _verified(symbol, currency="USD", n=2500, last="2026-10-12", name=None,
              exchange="LSE"):
    return eu.Verified(symbol=symbol, currency=currency, exchange=exchange,
                       name=name if name is not None else "iShares Core MSCI World UCITS ETF",
                       history_days=n, first_date="2016-10-12", last_date=last,
                       last_close=100.0)


def test_load_catalogue_dedupes_and_normalises(tmp_path):
    p = tmp_path / "cat.json"
    p.write_text(json.dumps({"etfs": [
        {"urlName": "a", "name": "A", "isin": "ie00b4l5y983", "symbol": "iwda",
         "currency": "usd", "assetClass": "equity", "price": "1.5", "isActive": True},
        {"urlName": "b", "name": "A again", "isin": "IE00B4L5Y983", "symbol": "IWDA",
         "currency": "USD"},
        {"urlName": "c", "name": "No ISIN", "isin": "", "symbol": "X"},
    ]}))
    cat = eu.load_catalogue(p)
    assert len(cat) == 1
    assert cat[0].isin == "IE00B4L5Y983" and cat[0].symbol == "IWDA"
    assert cat[0].currency == "USD" and cat[0].price == 1.5


def test_candidate_order_prefers_london_for_usd_then_guesses():
    quotes = [eu.Quote("IWDA.AS", "AMS", "ETF"), eu.Quote("IWDA.L", "LSE", "ETF"),
              eu.Quote("IE00B4L5Y983.SG", "STU", "MUTUALFUND"),   # unsupported suffix
              eu.Quote("SWDA.L", "LSE", "ETF"), eu.Quote("IWDA.MX", "MEX", "ETF")]
    order = eu.candidate_symbols(_etf(currency="USD"), quotes)
    symbols = [s for s, _ in order]
    assert symbols[:3] == ["IWDA.L", "SWDA.L", "IWDA.AS"]
    assert "IE00B4L5Y983.SG" not in symbols and "IWDA.MX" not in symbols
    # ticker guesses follow, in venue-preference order, never repeating a hit
    guesses = [s for s, via in order if via == "guess"]
    assert guesses[0] == "IWDA.SW" and "IWDA.L" not in guesses


def test_candidate_order_prefers_xetra_for_eur():
    quotes = [eu.Quote("XESC.L", "LSE", "ETF"), eu.Quote("XESC.DE", "GER", "ETF"),
              eu.Quote("XESC.MI", "MIL", "EQUITY")]
    order = eu.candidate_symbols(_etf(currency="EUR", symbol="XESC"), quotes)
    # venue preference first (Milan is a EUR venue), then quote type
    assert [s for s, _ in order][:3] == ["XESC.DE", "XESC.MI", "XESC.L"]


def test_preferred_lines_jump_the_queue():
    quotes = [eu.Quote("INRG.SW", "EBS", "ETF")]
    order = eu.candidate_symbols(_etf(currency="USD", symbol="INRG"), quotes,
                                 prefer=frozenset({"INRG.L"}))
    assert order[0] == ("INRG.L", "guess") and order[1] == ("INRG.SW", "search")
    quotes = [eu.Quote("BRNG.L", "LSE", "EQUITY"), eu.Quote("BRNT.L", "LSE", "EQUITY")]
    order = eu.candidate_symbols(_etf(currency="USD", symbol="BRNT"), quotes,
                                 prefer=frozenset({"BRNT.L"}))
    assert [s for s, _ in order][:2] == ["BRNT.L", "BRNG.L"]


def test_far_east_lines_are_never_candidates():
    quotes = [eu.Quote("1673.T", "JPX", "ETF"), eu.Quote("PHAG.L", "LSE", "EQUITY")]
    order = eu.candidate_symbols(_etf(currency="USD", symbol="PHAG"), quotes)
    assert order[0] == ("PHAG.L", "search") and all(s != "1673.T" for s, _ in order)


def test_leveraged_classifier():
    yes = ["WisdomTree Gold 3x Daily Leveraged", "Leverage Shares -1x Short Disney ETP",
           "Xtrackers S&P 500 Inverse Daily Swap UCITS ETF 1C", "WisdomTree FTSE 250 1x Daily Short",
           "Amundi ETF Short Euro Stoxx 50 Daily", "GraniteShares 3x Long NVIDIA Daily ETP"]
    no = ["iShares $ Short Duration Corp Bond UCITS ETF", "PIMCO Sterling Short Maturity UCITS ETF",
          "UBS EUR Ultra-Short Bond Screened UCITS ETF", "iShares $ Treasury Bond 0-1yr UCITS ETF",
          "Vanguard S&P 500 UCITS ETF", "iShares Core UK Gilts UCITS ETF"]
    assert all(eu.leveraged_or_inverse(n) for n in yes)
    assert not any(eu.leveraged_or_inverse(n) for n in no)


def test_resolve_takes_first_verified_search_hit():
    quotes = [eu.Quote("IWDA.L", "LSE", "ETF"), eu.Quote("IWDA.AS", "AMS", "ETF")]
    seen = []

    def verify(sym):
        seen.append(sym)
        return None if sym == "IWDA.L" else _verified(sym, "EUR", exchange="AMS")

    row = eu.resolve_etf(_etf(), quotes, verify, AS_OF)
    assert row.status == eu.STATUS_OK and row.yahoo == "IWDA.AS" and row.via == "search"
    assert "Yahoo line is in EUR" in row.note
    assert seen == ["IWDA.L", "IWDA.AS"]


def test_resolve_rejects_stale_and_short_histories():
    quotes = [eu.Quote("IWDA.L", "LSE", "ETF")]
    stale = eu.resolve_etf(_etf(), quotes, lambda s: _verified(s, last="2025-01-10"), AS_OF)
    assert stale.status == "no_yahoo_data" and "stale" in stale.note
    short = eu.resolve_etf(_etf(symbol=""), quotes, lambda s: _verified(s, n=20), AS_OF)
    assert short.status == "no_yahoo_data" and "only 20 bars" in short.note


def test_guess_needs_matching_name():
    # no search hits at all: only ticker guesses remain
    def verify(sym):
        if sym == "IWDA.L":
            return _verified(sym, name="Totally Different Fund Plc")
        if sym == "IWDA.DE":
            return _verified(sym, "EUR", name="iShares Core MSCI World UCITS ETF")
        return None

    row = eu.resolve_etf(_etf(), [], verify, AS_OF)
    assert row.status == eu.STATUS_OK and row.yahoo == "IWDA.DE" and row.via == "guess"
    assert row.name_similarity is not None and row.name_similarity >= 0.5


def test_no_candidates_and_inactive():
    assert eu.resolve_etf(_etf(symbol=""), [], lambda s: None, AS_OF).status == "no_candidates"
    assert eu.resolve_etf(_etf(is_active=False), [], lambda s: None, AS_OF).status == "inactive"


def test_build_universe_never_maps_two_isins_to_one_symbol():
    cat = [_etf(isin="IE00000000A1", url_name="a"), _etf(isin="IE00000000B2", url_name="b")]
    df = eu.build_universe(cat, lambda isin: [eu.Quote("IWDA.L", "LSE", "ETF")],
                           lambda s: _verified(s), AS_OF)
    assert list(df["status"]) == [eu.STATUS_OK, "duplicate_symbol"]


def test_write_then_load_universe_exposes_tickers_groups_and_overrides(tmp_path):
    cat = [_etf(), _etf(isin="IE00B1FZSB30", symbol="IGLT", currency="GBP",
                        asset_class="bond", name="iShares Core UK Gilts UCITS ETF"),
           _etf(isin="DEAD00000001", symbol="", url_name="dead", name="Dead ETF")]

    def search(isin):
        return {"IE00B4L5Y983": [eu.Quote("IWDA.L", "LSE", "ETF")],
                "IE00B1FZSB30": [eu.Quote("IGLT.L", "LSE", "ETF")]}.get(isin, [])

    def verify(sym):
        return {"IWDA.L": _verified("IWDA.L", "USD"),
                "IGLT.L": _verified("IGLT.L", "GBp", name="iShares Core UK Gilts")}.get(sym)

    df = eu.build_universe(cat, search, verify, AS_OF)
    out = eu.write_universe(df, tmp_path / "u.csv")
    uni = eu.load_universe(out)
    assert uni.tickers == ["IGLT.L", "IWDA.L"]
    assert uni.groups == {"IGLT.L": "bond", "IWDA.L": "equity"}
    # a USD line on London needs an override; a pence line does not
    assert uni.currency_overrides == {"IWDA.L": "USD"}
    assert uni.meta["IWDA.L"]["umushroom_symbol"] == "IWDA"
    assert uni.status_counts == {"ok": 2, "no_candidates": 1}
    assert "2 tradeable of 3" in uni.summary()


def test_load_universe_rejects_duplicate_symbols_and_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("yahoo,isin,status,asset_class\nA.L,X,ok,equity\nA.L,Y,ok,bond\n")
    with pytest.raises(ConfigError):
        eu.load_universe(bad)
    bad.write_text("yahoo,isin\nA.L,X\n")
    with pytest.raises(ConfigError):
        eu.load_universe(bad)
    with pytest.raises(ConfigError):
        eu.load_universe(tmp_path / "missing.csv")


def test_cli_replays_saved_search_and_verification(tmp_path, capsys):
    cat = tmp_path / "cat.json"
    cat.write_text(json.dumps({"scraped_at": "2026-10-12T17:00:00Z", "etfs": [
        {"urlName": "iwda", "name": "iShares Core MSCI World UCITS ETF USD (Acc)",
         "isin": "IE00B4L5Y983", "symbol": "IWDA", "currency": "USD",
         "assetClass": "equity", "isActive": True}]}))
    cands = tmp_path / "cands.json"
    cands.write_text(json.dumps({"candidates": {
        "IE00B4L5Y983": [{"s": "IWDA.L", "x": "LSE", "t": "ETF", "n": "iShares Core"}]}}))
    verified = tmp_path / "verified.json"
    verified.write_text(json.dumps({"verified": {
        "IWDA.L": {"cur": "USD", "x": "LSE", "ln": "iShares Core MSCI World UCITS ETF",
                   "n": 2500, "first": "2016-10-12", "last": "2026-10-12"}}}))
    out = tmp_path / "universe.csv"
    rc = eu.main(["--catalogue", str(cat), "--candidates", str(cands),
                  "--verified", str(verified), "--out", str(out), "--as-of", "2026-10-12"])
    assert rc == 0
    assert "1 tradeable of 1" in capsys.readouterr().out
    assert eu.load_universe(out).tickers == ["IWDA.L"]
    assert eu.catalogue_as_of(cat) == AS_OF
    assert eu.main(["--catalogue", str(tmp_path / "nope.json")]) == 2


def test_name_similarity_ignores_boilerplate():
    a = "iShares Core MSCI World UCITS ETF USD (Acc)"
    b = "ISHARES III PLC ISHRS CORE MSCI WORLD"
    assert eu.name_similarity(a, b) > 0.5
    assert eu.name_similarity(a, "WisdomTree Physical Silver") < 0.2
    assert eu.name_similarity("", a) == 0.0
