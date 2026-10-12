"""Catalogue scraper (catalogue.py) with fake HTTP: sitemap parsing,
record normalisation, retries, dead pages and incremental reruns."""
from __future__ import annotations

import json

from umushroom import catalogue as cat

SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>https://umushroom.com/en/etf/alpha-etf</loc><lastmod>2025-01-01</lastmod></url>
<url><loc>https://umushroom.com/en/etf/beta-etf</loc></url>
<url><loc>https://umushroom.com/en/etf/alpha-etf</loc></url>
<url><loc>https://umushroom.com/en/equity/not-an-etf</loc></url>
</urlset>"""


def _security(slug, isin):
    return {"code": 200, "security": {
        "id": 1, "urlName": slug, "name": slug.replace("-", " ").title(), "isin": isin,
        "symbol": slug[:4].upper(), "currencyCode": "USD", "type": 3, "typeName": "etf",
        "price": 10.5, "priceDate": "2026-10-09", "assetClass": "equity",
        "styleName": "Large-cap", "isActive": True,
        "assetManager": {"id": 9, "name": "Acme"}, "performance": {"1y": 0.1},
        "volatility": {"1y": 0.2}}}


def test_parse_sitemap_keeps_etf_slugs_unique_and_ordered():
    assert cat.parse_sitemap(SITEMAP) == ["alpha-etf", "beta-etf"]


def test_scrape_handles_ok_404_and_retries(monkeypatch):
    monkeypatch.setattr(cat.time, "sleep", lambda s: None)
    calls = {"gamma-etf": 0}

    def fetch_json(url):
        slug = url.rsplit("/", 1)[1]
        if slug == "alpha-etf":
            return 200, _security(slug, "IE00AAAA0001")
        if slug == "beta-etf":
            return 404, None
        calls[slug] += 1
        return (503, None) if calls[slug] < 3 else (200, _security(slug, "IE00GGGG0003"))

    res = cat.scrape(["alpha-etf", "beta-etf", "gamma-etf"], fetch_json, pause=0)
    assert [e["isin"] for e in res["etfs"]] == ["IE00AAAA0001", "IE00GGGG0003"]
    assert res["etfs"][0]["manager"] == "Acme" and res["etfs"][0]["currency"] == "USD"
    assert res["dead_pages"] == [{"slug": "beta-etf", "err": "404"}]
    assert res["sitemap_slugs"] == 3 and calls["gamma-etf"] == 3


def test_rerun_only_fetches_new_slugs(monkeypatch):
    monkeypatch.setattr(cat.time, "sleep", lambda s: None)
    fetched = []

    def fetch_json(url):
        slug = url.rsplit("/", 1)[1]
        fetched.append(slug)
        return 200, _security(slug, f"IE00{slug[:4].upper()}0000")

    first = cat.scrape(["alpha-etf", "beta-etf"], fetch_json, pause=0)
    again = cat.scrape(["alpha-etf", "beta-etf", "delta-etf"], fetch_json, first, pause=0)
    assert fetched == ["alpha-etf", "beta-etf", "delta-etf"]
    assert len(again["etfs"]) == 3


def test_main_with_local_sitemap(tmp_path, monkeypatch):
    monkeypatch.setattr(cat.time, "sleep", lambda s: None)
    sm = tmp_path / "sitemap.xml"
    sm.write_text(SITEMAP)
    monkeypatch.setattr(cat, "_http_json", lambda url: (
        200, _security(url.rsplit("/", 1)[1], "IE00" + url[-8:].upper())))
    out = tmp_path / "catalogue.json"
    rc = cat.main(["--sitemap", str(sm), "--out", str(out), "--extra", "extra-etf"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert [e["urlName"] for e in data["etfs"]] == ["alpha-etf", "beta-etf", "extra-etf"]
    assert data["scraped_at"]
