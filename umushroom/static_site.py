"""The website as static, password-encrypted pages -- for GitHub Pages, so
the team can open it without your laptop being on.

    python -m umushroom.static_site --run --out public

  --run   first runs the full pipeline (like "Run now"), saving the run in
          ./site/reports exactly like the local website does
  --out   folder to write the finished site into (upload that folder)

Every page is ENCRYPTED with the team password (AES-256-GCM, key from
PBKDF2-SHA256 with 600,000 iterations). The published files are unreadable
without it -- the browser asks for the password and decrypts the page
locally. So even though GitHub Pages sites are public, the picks are not.

The password comes from the UMUSHROOM_PASSWORD environment variable (a
GitHub secret in the workflow) and must be at least 12 characters: anyone
can download the encrypted pages and try passwords offline, so a short one
could be guessed.

Differences from the live website: no "Run now" button (the nav links to the
GitHub workflow instead, where team members can press "Run workflow"), and
Tracker / Macro prices are as of the build, not live.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import secrets
import shutil
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

from umushroom.report import render_page, section_bodies
from umushroom.tracker import FetchFn, yahoo_closes
from umushroom.web import (
    _NAV_CSS,
    Runner,
    Site,
    _default_held,
    _pretty,
    default_command,
    macro_body,
    sell_body,
    tracker_body,
)

ITERATIONS = 600_000
MIN_PASSWORD = 12
KEEP_RUNS = 40

PAGES: list[tuple[str, str, str]] = [        # (key, label, file)
    ("overview", "Overview", "index.html"), ("stocks", "Stocks", "stocks.html"),
    ("etfs", "ETFs", "etfs.html"),
    ("sell", "When to sell", "sell.html"), ("tracker", "Tracker", "tracker.html"),
    ("graphs", "Graphs", "graphs.html"), ("montecarlo", "Monte Carlo", "montecarlo.html"),
    ("risk", "Risk (VaR)", "risk.html"), ("macro", "Macro", "macro.html"),
    ("checks", "Checks", "checks.html"), ("runs", "History", "history.html")]


# ---------------------------------------------------------------- pages

def _short(stamp: str) -> str:
    from datetime import datetime
    return datetime.strptime(stamp, "%Y%m%d-%H%M%S").strftime("%d %b %H:%M")


def static_nav(active: str, updated: str, run_url: str | None, prefix: str = "") -> str:
    tabs = "".join(
        f'<a href="{prefix}{f}"'
        f'{" aria-current=" + chr(34) + "page" + chr(34) if k == active else ""}>'
        f"{html.escape(label)}</a>" for k, label, f in PAGES)
    run = (f'<a class="umw-runlink" href="{html.escape(run_url)}" target="_blank" '
           f'rel="noopener">Run now</a>' if run_url else "")
    return (f"<style>{_NAV_CSS}.umw-runlink{{color:var(--series-1,#2a78d6)}}</style>"
            f'<nav class="umw-bar" aria-label="Sections"><div class="umw-in">'
            f'<span class="umw-brand">UMushroom</span><div class="umw-tabs">{tabs}</div>'
            f'<span class="umw-sp"></span><span class="umw-st" title="Last update">'
            f"{html.escape(updated)}</span>{run}</div></nav>")


def _with_nav(report_html: str, bar: str) -> str:
    m = re.search(r"<body[^>]*>", report_html, flags=re.IGNORECASE)
    return bar + report_html if not m else report_html[:m.end()] + bar + report_html[m.end():]


def build_pages(site: Site, run_url: str | None = None,
                today: date | None = None) -> dict[str, str]:
    """All pages, as plain HTML, keyed by their file name."""
    runner = site.runner
    latest = runner.latest_report()
    hist = runner.history()
    newest = hist[0] if hist else None
    if latest is None:
        bar = static_nav("overview", "never", run_url)
        msg = ("<h2>No report yet</h2><p>The first run hasn't finished. On GitHub open "
               "<b>Actions -> Website -> Run workflow</b>.</p>")
        return {f: render_page({"meta": {}}, msg, "UMushroom", nav=bar, sub="")
                for _, _, f in PAGES}

    payload = site.payload(latest) or {"meta": {}}
    updated = "Updated " + _short(latest)
    ccy = payload.get("meta", {}).get("base_currency", "")
    bodies = section_bodies(payload)
    warn = ""
    if newest and newest["stamp"] != latest and newest.get("status") != "succeeded":
        warn = (f'<div class="due now"><b>The latest run failed</b>The run of '
                f"{html.escape(_pretty(newest['stamp']))} did not finish, so this shows "
                f"the last good one ({html.escape(_pretty(latest))}). See History.</div>")

    out: dict[str, str] = {}
    for key, label, fname in PAGES:
        if key == "runs":
            continue
        if key == "sell":
            held = _default_held(site, latest, payload)
            body = sell_body(site, latest, payload, held, today=today, picker=False)
        elif key == "tracker":
            try:
                body = tracker_body(site.tracked(latest, payload, refresh=True), ccy)
            except Exception as e:  # noqa: BLE001 -- network errors vary
                body = f'<p class="note">Could not get prices: {html.escape(str(e))}</p>'
        elif key == "macro":
            try:
                body = macro_body(site.macro(latest, payload, refresh=True), ccy)
            except Exception as e:  # noqa: BLE001
                body = f'<p class="note">Could not get macro data: {html.escape(str(e))}</p>'
        else:
            body = bodies[key]
        title = "UMushroom" if key == "overview" else f"UMushroom: {label}"
        out[fname] = render_page(payload, (warn if key == "overview" else "") + body,
                                 title=title, nav=static_nav(key, updated, run_url))

    rows = []
    for r in hist:
        st = str(r.get("status", "unknown"))
        link = (f'<a href="reports/{r["stamp"]}.html">Full report</a>'
                if r["has_report"] and st == "succeeded" else "")
        rows.append(f"<tr><td>{html.escape(_pretty(r['stamp']))}</td>"
                    f'<td class="{html.escape(st)}">{html.escape(st)}</td><td>{link}</td></tr>')
        if link:
            src = runner.path(r["stamp"], "html").read_text(errors="replace")
            out[f"reports/{r['stamp']}.html"] = _with_nav(
                src, static_nav("", updated, run_url, prefix="../"))
    table = ('<div class="scroll"><table class="data runs-tbl"><thead><tr><th>Run</th>'
             "<th>Status</th><th>Report</th></tr></thead><tbody>" + "".join(rows)
             + "</tbody></table></div>")
    out["history.html"] = render_page(
        payload, '<p class="note">Every run, newest first.</p>' + table,
        title="Run history", nav=static_nav("runs", updated, run_url), sub="")
    return out


# ----------------------------------------------------------- encryption

def derive_key(password: str, salt: bytes, iterations: int = ITERATIONS) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt,
                     iterations=iterations)
    return kdf.derive(password.encode("utf-8"))


def encrypt(plaintext: str, key: bytes) -> tuple[bytes, bytes]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    iv = secrets.token_bytes(12)
    return iv, AESGCM(key).encrypt(iv, plaintext.encode("utf-8"), None)


def decrypt(iv: bytes, ciphertext: bytes, key: bytes) -> str:
    """Python twin of the in-browser decryption (used by the tests)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key).decrypt(iv, ciphertext, None).decode("utf-8")


_LOCK_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>UMushroom team</title>
<style>
:root{color-scheme:light;--bg:#f4f3f0;--card:#fcfcfb;--text:#0b0b0b;--muted:#5f5e5a;
--line:#e2e0da;--accent:#2a78d6;--bad:#b3261e}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;
--bg:#161615;--card:#1f1f1d;--text:#f2f1ee;--muted:#a7a59f;--line:#34332f;
--accent:#6aa7ff;--bad:#ff8a80}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#161615;--card:#1f1f1d;--text:#f2f1ee;
--muted:#a7a59f;--line:#34332f;--accent:#6aa7ff;--bad:#ff8a80}
body{margin:0;background:var(--bg);color:var(--text);
font:16px/1.5 system-ui,-apple-system,sans-serif;display:grid;place-items:center;
min-height:100vh}
form{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:24px;width:min(360px,calc(100vw - 32px));box-sizing:border-box}
h1{font-size:22px;margin:0 0 4px}p{margin:0 0 16px;color:var(--muted);font-size:14px}
label{display:block;font-size:14px;margin-bottom:6px}
input[type=password]{width:100%;box-sizing:border-box;font:inherit;padding:10px;
border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--text)}
.row{display:flex;align-items:center;gap:8px;margin:12px 0;font-size:14px}
button{font:inherit;width:100%;padding:10px;border:0;border-radius:8px;
background:var(--accent);color:#fff;cursor:pointer}
button:disabled{opacity:.6}#msg{color:var(--bad);margin:12px 0 0;min-height:1.5em}
</style></head><body>
<form id="f"><h1>UMushroom</h1><p>Team members only.</p>
<label for="pw">Team password</label>
<input id="pw" type="password" autocomplete="current-password" required autofocus>
<div class="row"><input id="rem" type="checkbox"><label for="rem" style="margin:0">
Remember on this device</label></div>
<button id="go" type="submit">Open</button><p id="msg" role="alert"></p></form>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(function(){
var d=JSON.parse(document.getElementById('payload').textContent);
var K='umk:'+d.s;
function b64(s){return Uint8Array.from(atob(s),function(c){return c.charCodeAt(0);});}
function enc(a){var s='';a.forEach(function(x){s+=String.fromCharCode(x);});return btoa(s);}
function get(){try{return sessionStorage.getItem(K)||localStorage.getItem(K);}catch(e){return null;}}
function put(raw,keep){try{sessionStorage.setItem(K,enc(raw));if(keep){localStorage.setItem(K,enc(raw));}}catch(e){}}
function drop(){try{sessionStorage.removeItem(K);localStorage.removeItem(K);}catch(e){}}
function show(raw,keep){
 return crypto.subtle.importKey('raw',raw,'AES-GCM',false,['decrypt'])
 .then(function(k){return crypto.subtle.decrypt({name:'AES-GCM',iv:b64(d.i)},k,b64(d.c));})
 .then(function(pt){put(raw,keep);var h=new TextDecoder().decode(pt);
  document.open();document.write(h);document.close();});
}
var saved=get(),form=document.getElementById('f');
function auto(){show(b64(saved),false).catch(function(){drop();form.style.visibility='visible';});}
if(saved){
 form.style.visibility='hidden';
 // document.write only replaces the page once parsing has finished
 if(document.readyState==='complete'){auto();}else{window.addEventListener('load',auto);}
}
document.getElementById('f').addEventListener('submit',function(ev){
 ev.preventDefault();var b=document.getElementById('go'),m=document.getElementById('msg');
 b.disabled=true;m.textContent='Unlocking...';
 var pw=new TextEncoder().encode(document.getElementById('pw').value);
 crypto.subtle.importKey('raw',pw,'PBKDF2',false,['deriveBits'])
 .then(function(base){return crypto.subtle.deriveBits({name:'PBKDF2',salt:b64(d.s),
   iterations:d.n,hash:'SHA-256'},base,256);})
 .then(function(bits){return show(new Uint8Array(bits),document.getElementById('rem').checked);})
 .catch(function(){b.disabled=false;m.textContent='Wrong password.';});
});
})();
</script></body></html>
"""


def lock(page_html: str, key: bytes, salt: bytes, iterations: int = ITERATIONS) -> str:
    iv, ct = encrypt(page_html, key)
    data = json.dumps({"s": base64.b64encode(salt).decode(),
                       "i": base64.b64encode(iv).decode(),
                       "c": base64.b64encode(ct).decode(), "n": iterations})
    return _LOCK_PAGE.replace("__PAYLOAD__", data)


def site_salt(site_dir: Path) -> bytes:
    """Kept between builds, so 'Remember on this device' survives daily updates."""
    f = site_dir / "site_salt.txt"
    if f.is_file():
        try:
            raw = base64.b64decode(f.read_text().strip())
            if len(raw) == 16:
                return raw
        except ValueError:
            pass
    raw = secrets.token_bytes(16)
    site_dir.mkdir(parents=True, exist_ok=True)
    f.write_text(base64.b64encode(raw).decode() + "\n")
    return raw


def prune(runner: Runner, keep: int = KEEP_RUNS) -> None:
    """Keep the newest `keep` runs (site size and cache stay bounded)."""
    for r in runner.history()[keep:]:
        for ext in ("html", "csv", "log", "json"):
            runner.path(r["stamp"], ext).unlink(missing_ok=True)


def write_site(pages: dict[str, str], out: Path, password: str, salt: bytes,
               iterations: int = ITERATIONS) -> None:
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"the team password must be at least {MIN_PASSWORD} characters "
                         "(the encrypted pages are public, so a short one can be guessed)")
    key = derive_key(password, salt, iterations)
    if out.exists():
        shutil.rmtree(out)
    for name, page in pages.items():
        f = out / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(lock(page, key, salt, iterations), encoding="utf-8")
    (out / ".nojekyll").write_text("")
    (out / "robots.txt").write_text("User-agent: *\nDisallow: /\n")


# v4.21: the pipeline exits 3 when Yahoo has not published yesterday's closes
# yet (data.check_freshness); the workflow then keeps the last good run and a
# later slot retries, instead of publishing a run on stale signals
NOT_READY_CODE = 3


def build(site_dir: Path, out: Path, password: str, run_url: str | None = None,
          fetch: FetchFn = yahoo_closes,
          command: Callable[[Path, Path], list[str]] | None = None,
          run_first: bool = False, run_timeout: float = 3 * 3600,
          iterations: int = ITERATIONS) -> int:
    if len(password) < MIN_PASSWORD:      # fail BEFORE spending an hour on a run
        raise ValueError(f"the team password must be at least {MIN_PASSWORD} characters")
    runner = Runner(site_dir.resolve(), command or default_command("config.yaml", []))
    ok = True
    if run_first:
        if runner.start() is None:
            raise RuntimeError("a run is already in progress")
        runner.wait(timeout=run_timeout)
        ok = runner.history()[0].get("status") == "succeeded"
        log = runner.path(runner.history()[0]["stamp"], "log")
        # Only the end of a FAILED run's log is printed: CI logs of a public
        # repo are public, and a successful run's log lists the picks.
        if not ok and log.exists():
            tail = log.read_text(errors="replace").splitlines()[-40:]
            print("Run failed. End of its log:\n" + "\n".join(tail))
    prune(runner)
    site = Site(runner, fetch)
    write_site(build_pages(site, run_url), out, password, site_salt(site_dir), iterations)
    not_ready = (not ok and run_first
                 and runner.history()[0].get("returncode") == NOT_READY_CODE)
    state = ("run OK" if ok else "DATA NOT READY -- showing last good" if not_ready
             else "RUN FAILED -- showing last good")
    print(f"Site written to {out} ({state})")
    return 0 if ok else NOT_READY_CODE if not_ready else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="umushroom-static-site",
                                description="Build the password-protected static site.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--site-dir", default="site")
    p.add_argument("--out", default="public")
    p.add_argument("--run", action="store_true", help="run the pipeline first")
    p.add_argument("--validate", action="store_true",
                   help="include the validation suite in the run (slower)")
    p.add_argument("--run-url", default=None,
                   help="link for the 'Run now (GitHub)' button")
    args = p.parse_args(argv)
    password = os.environ.get("UMUSHROOM_PASSWORD", "")
    if not password:
        print("Set the UMUSHROOM_PASSWORD environment variable (a GitHub secret).")
        return 2
    cmd = default_command(args.config, ["--validate"] if args.validate else [])
    try:
        return build(Path(args.site_dir), Path(args.out), password, args.run_url,
                     command=cmd, run_first=args.run)
    except ValueError as e:
        print(f"Error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
