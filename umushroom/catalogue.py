"""UMushroom ETF catalogue scraper (v4.19).

Produces `umushroom_etf_catalogue.json`: every ETF page UMushroom
publishes, with the fields its own API returns for each one.

How it works (two public, unauthenticated sources -- no login, no
JavaScript, no guessing):
  1. https://umushroom.com/sitemap.en_etf.xml lists every ETF page URL
     (the sitemap exists precisely so crawlers can find them).
  2. GET https://api.umushroom.com/api/security/{slug} returns the ETF
     record behind each page: name, ISIN, UMushroom ticker, pricing
     currency, asset class, price and date, manager, active flag.
  Extra slugs (ETFs UMushroom lists but leaves out of the sitemap -- the
  team found IWDA, XESC and VUAA this way) can be added with --extra.

Politeness: requests are sequential by default with a pause between
them (about 4 per second at most), retried with backoff on 429/5xx, and
the result is cached so a rerun only fetches what is new. UMushroom's
terms frown on systematic extraction; the team decided this one-off
catalogue read (the same thing it did for the stock list on 2026-10-08)
is acceptable for an instrument list that the site itself publishes to
search engines. Keep it to a single run, not a daily job.

The fetchers are injectable so the whole module is unit-tested offline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from umushroom.exceptions import DataError
from umushroom.logging_config import get_logger

logger = get_logger("catalogue")

SITEMAP_URL = "https://umushroom.com/sitemap.en_etf.xml"
API_URL = "https://api.umushroom.com/api/security/{slug}"
PAUSE_SECONDS = 0.25
MAX_RETRIES = 3
TextFetch = Callable[[str], str]
JsonFetch = Callable[[str], "tuple[int, dict | None]"]   # (status, body)

RECORD_FIELDS = (
    "id", "urlName", "name", "isin", "valor", "wkn", "symbol", "exchange", "type",
    "typeName", "currency", "price", "priceDate", "assetClass", "style", "domicile",
    "category", "esg", "isActive", "manager", "perf", "vol", "website",
)


def parse_sitemap(xml: str) -> list[str]:
    """Slugs of every /en/etf/<slug> page in the sitemap, in order, unique."""
    slugs = re.findall(r"<loc>\s*https?://(?:www\.)?umushroom\.com/en/etf/([^<\s]+)\s*</loc>",
                       xml)
    out: list[str] = []
    seen: set[str] = set()
    for s in slugs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def normalise_record(security: dict) -> dict:
    """The subset of UMushroom's security record the pipeline keeps."""
    mgr = security.get("assetManager")
    return {
        "id": security.get("id"),
        "urlName": security.get("urlName"),
        "name": security.get("name"),
        "isin": security.get("isin"),
        "valor": security.get("valor"),
        "wkn": security.get("wkn"),
        "symbol": security.get("symbol"),
        "exchange": security.get("exchange"),
        "type": security.get("type"),
        "typeName": security.get("typeName"),
        "currency": security.get("currencyCode"),
        "price": security.get("price"),
        "priceDate": security.get("priceDate"),
        "assetClass": security.get("assetClass"),
        "style": security.get("styleName"),
        "domicile": security.get("domicile"),
        "category": security.get("categoryName"),
        "esg": security.get("esg"),
        "isActive": security.get("isActive"),
        "manager": mgr.get("name") if isinstance(mgr, dict) else None,
        "perf": security.get("performance"),
        "vol": security.get("volatility"),
        "website": security.get("website"),
    }


def _http_text(url: str) -> str:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "umushroom-strategy/4.19"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _http_json(url: str) -> tuple[int, dict | None]:
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "umushroom-strategy/4.19",
                                               "Accept-Language": "en",
                                               "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        return e.code, None


def fetch_record(slug: str, fetch_json: JsonFetch, pause: float = PAUSE_SECONDS
                 ) -> tuple[dict | None, str | None]:
    """(record, error). error is "404" for a dead page, else the last failure."""
    last = "unknown"
    for attempt in range(1, MAX_RETRIES + 1):
        status, body = fetch_json(API_URL.format(slug=slug))
        if status == 200 and body and isinstance(body.get("security"), dict):
            time.sleep(pause)
            return normalise_record(body["security"]), None
        if status == 404:
            time.sleep(pause)
            return None, "404"
        last = f"http{status}"
        if status in (429,) or status >= 500:
            time.sleep(2.0 * attempt)
            continue
        break
    return None, last


def scrape(slugs: list[str], fetch_json: JsonFetch, existing: dict | None = None,
           pause: float = PAUSE_SECONDS, progress_every: int = 200) -> dict:
    """Fetch every slug not already in `existing` (a previous catalogue)."""
    etfs: dict[str, dict] = {}
    dead: dict[str, str] = {}
    if existing:
        for r in existing.get("etfs", []):
            if r.get("urlName"):
                etfs[r["urlName"]] = r
        for d in existing.get("dead_pages", []):
            if d.get("slug"):
                dead[d["slug"]] = d.get("err", "404")
    todo = [s for s in slugs if s not in etfs and s not in dead]
    logger.info(f"catalogue: {len(slugs)} slugs, {len(todo)} to fetch")
    for i, slug in enumerate(todo, start=1):
        rec, err = fetch_record(slug, fetch_json, pause)
        if rec is not None:
            etfs[slug] = rec
        else:
            dead[slug] = err or "unknown"
        if progress_every and i % progress_every == 0:
            logger.info(f"catalogue: fetched {i}/{len(todo)}")
    return {
        "source": f"{SITEMAP_URL} + {API_URL}",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "sitemap_slugs": len(slugs),
        "etfs": [etfs[s] for s in slugs if s in etfs]
                + [r for s, r in etfs.items() if s not in set(slugs)],
        "dead_pages": [{"slug": s, "err": e} for s, e in dead.items()],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="umushroom-catalogue",
                                description="Scrape UMushroom's public ETF catalogue.")
    p.add_argument("--out", default="umushroom_etf_catalogue.json")
    p.add_argument("--extra", nargs="*", default=[],
                   help="Extra page slugs to fetch (ETFs missing from the sitemap).")
    p.add_argument("--sitemap", default=None, help="Local copy of the sitemap XML.")
    p.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    p.add_argument("--fresh", action="store_true",
                   help="Ignore the existing catalogue instead of only fetching new pages.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = Path(args.out)
    try:
        xml = Path(args.sitemap).read_text() if args.sitemap else _http_text(SITEMAP_URL)
    except OSError as e:
        print(f"ERROR: could not read the sitemap: {e}", file=sys.stderr)
        return 1
    slugs = parse_sitemap(xml) + [s for s in args.extra if s]
    if not slugs:
        print("ERROR: no ETF pages found in the sitemap", file=sys.stderr)
        return 1
    existing = None
    if out.exists() and not args.fresh:
        try:
            existing = json.loads(out.read_text())
        except ValueError:
            existing = None
    try:
        result = scrape(slugs, _http_json, existing, pause=args.pause)
    except DataError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    out.write_text(json.dumps(result, separators=(",", ":")))
    print(f"{len(result['etfs'])} ETFs, {len(result['dead_pages'])} dead pages -> "
          f"{out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
