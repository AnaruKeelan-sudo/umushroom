from __future__ import annotations

import numpy as np
import pandas as pd

from umushroom.check_tickers import check_tickers, config_snippet

IDX = pd.bdate_range("2026-09-01", periods=20)
ALIVE = {"AAPL", "SBRY.L", "CRH", "EVPL.L"}


def fake_download(symbols):
    return pd.DataFrame({s: (np.linspace(1, 2, len(IDX)) if s in ALIVE else np.nan)
                         for s in symbols}, index=IDX)


def fake_search(query):
    db = {
        "IE0001827041": [{"symbol": "CRH", "quoteType": "EQUITY"},
                         {"symbol": "CRG.IR", "quoteType": "EQUITY"}],
        "Team17 Group PLC": [{"symbol": "EVPL.L", "quoteType": "EQUITY"},
                             {"symbol": "TSVNF", "quoteType": "MUTUALFUND"}],
    }
    return db.get(query, [])


META = {
    "CRH.L": {"name": "CRH PLC", "isin": "IE0001827041"},
    "TM17.L": {"name": "Team17 Group PLC", "isin": "GB00BYVX2X20"},
    "GONE.SW": {"name": "Gone AG", "isin": "CH000"},
}


def test_classifies_ok_fixable_and_missing():
    res = {r.ticker: r for r in check_tickers(
        ["AAPL", "SBRY.L", "CRH.L", "TM17.L", "GONE.SW"], META,
        search=fake_search, download=fake_download, currency=lambda s: "USD"
        if s == "CRH" else "GBp")}
    assert res["AAPL"].status == "ok" and res["SBRY.L"].status == "ok"
    assert res["CRH.L"].status == "fixable" and res["CRH.L"].suggestion == "CRH"
    assert res["TM17.L"].suggestion == "EVPL.L"     # same-exchange match preferred
    assert res["GONE.SW"].status == "not found"


def test_existing_yahoo_symbol_mapping_is_respected():
    res = check_tickers(["CRH.L"], META, {"CRH.L": "CRH"},
                        search=fake_search, download=fake_download,
                        currency=lambda s: None)
    assert res[0].status == "ok"


def test_snippet_includes_currency_override_only_when_needed():
    res = check_tickers(["CRH.L", "TM17.L"], META, search=fake_search,
                        download=fake_download,
                        currency=lambda s: "USD" if s == "CRH" else "GBp")
    snip = config_snippet(res, {})
    assert '"CRH.L": "CRH"' in snip and '"TM17.L": "EVPL.L"' in snip
    assert '"CRH.L": "USD"' in snip           # London default is GBp -> override
    assert '"TM17.L": "GBp"' not in snip      # matches default -> no override


def test_otc_and_adr_lines_are_never_suggested():
    """v4.8: the real run suggested RHHBY for Roche -- a US ADR worth 1/8 of
    a share, which would size orders ~8x wrong. Must not be suggested."""
    from umushroom.check_tickers import is_otc_or_adr
    assert is_otc_or_adr("RHHBY") and is_otc_or_adr("PIERF")
    assert not is_otc_or_adr("FLUT") and not is_otc_or_adr("EVPL.L")
    assert not is_otc_or_adr("AAPL")

    def search(q):
        return [{"symbol": "RHHBY", "quoteType": "EQUITY"}]

    def download(symbols):
        return pd.DataFrame({s: np.linspace(1, 2, len(IDX)) if s == "RHHBY"
                             else np.nan for s in symbols}, index=IDX)
    res = check_tickers(["ROG.SW"], {"ROG.SW": {"name": "Roche", "isin": "CH1"}},
                        search=search, download=download, currency=lambda s: "USD")
    assert res[0].status == "not found" and "OTC/ADR" in res[0].note
    assert config_snippet(res, {}) == ""
