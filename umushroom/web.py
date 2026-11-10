"""A local website for the whole pipeline -- runs on YOUR computer only.

    python -m umushroom.web

opens http://127.0.0.1:8000 in your browser:

  Overview     headline numbers         /
  Stocks       today's orders + top-ranked stocks
  Tracker      each pick's price now vs. its price when it was predicted
  Graphs       out-of-sample performance, alpha, deciles
  Monte Carlo  the range of contest outcomes
  Checks       the validation suite
  History      every past run (open any of them in all the tabs above)
  "Run now"  runs the full pipeline (same as the command line with
             --report) in the background and shows its progress live

Optional:  --daily 07:30  also runs it automatically every weekday at 07:30
           --port 8001    use another port

Safety:
  * listens on 127.0.0.1 only -- other computers can't reach it;
  * "Run now" needs a custom header + same-origin request, so another
    website open in your browser can't trigger runs behind your back;
  * file names are checked against a strict pattern (no path tricks);
  * only one run at a time.

Standard library only: nothing extra to install.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import html
import json
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import numpy as np

from umushroom import __version__
from umushroom.macro import FBY, MacroResult, analyse, talking_points
from umushroom.rebalance import DEFAULT_REBALANCE_DAYS, Cadence, books, cadence, plan
from umushroom.report import (
    _label,
    _price,
    _tile,
    bar_chart,
    line_chart,
    render_page,
    section_bodies,
    trade_link,
)
from umushroom.shell import Shell, report_bar
from umushroom.tracker import (
    BENCHMARK,
    FetchFn,
    RecordRow,
    TrackerResult,
    record,
    track,
    yahoo_closes,
)

STAMP_RE = re.compile(r"^\d{8}-\d{6}$")
RUN_HEADER = "X-UMushroom-Run"
CommandFactory = Callable[[Path, Path], list[str]]   # (report, orders) -> argv


def default_command(config: str, extra: list[str]) -> CommandFactory:
    def make(report: Path, orders: Path) -> list[str]:
        return [sys.executable, "-m", "umushroom.cli", "--config", config,
                "--report", str(report), "--output", str(orders), *extra]
    return make


# ----------------------------------------------------------------- runs

@dataclass
class Runner:
    """Starts pipeline runs in the background and records each one in
    site_dir/reports/<stamp>.{html,csv,log,json}."""
    site_dir: Path
    command: CommandFactory
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _proc: subprocess.Popen[bytes] | None = None
    _stamp: str | None = None

    def __post_init__(self) -> None:
        self.reports.mkdir(parents=True, exist_ok=True)
        for meta in self.reports.glob("*.json"):       # server was stopped mid-run
            d = _read_json(meta)
            if d.get("status") == "running":
                d.update(status="interrupted", finished=_now())
                meta.write_text(json.dumps(d))

    @property
    def reports(self) -> Path:
        return self.site_dir / "reports"

    def path(self, stamp: str, ext: str) -> Path:
        return self.reports / f"{stamp}.{ext}"

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> str | None:
        """Start a run; returns its stamp, or None if one is already going."""
        with self._lock:
            if self.running:
                return None
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            while self.path(stamp, "json").exists():     # two clicks in 1 second
                time.sleep(1)
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            argv = self.command(self.path(stamp, "html"), self.path(stamp, "csv"))
            self.path(stamp, "json").write_text(json.dumps(
                {"status": "running", "started": _now(), "command": argv}))
            log = open(self.path(stamp, "log"), "wb")   # noqa: SIM115 -- closed in _wait
            self._proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                          cwd=self.site_dir.parent)
            self._stamp = stamp
            threading.Thread(target=self._wait, args=(self._proc, stamp, log),
                             daemon=True).start()
            return stamp

    def _wait(self, proc: subprocess.Popen[bytes], stamp: str, log: Any) -> None:
        rc = proc.wait()
        log.close()
        ok = rc == 0 and self.path(stamp, "html").exists()
        d = _read_json(self.path(stamp, "json"))
        d.update(status="succeeded" if ok else "failed", returncode=rc, finished=_now())
        self.path(stamp, "json").write_text(json.dumps(d))

    def wait(self, timeout: float = 60) -> None:
        """Block until the current run ends (used by tests)."""
        end = time.time() + timeout
        while self.running and time.time() < end:
            time.sleep(0.05)
        time.sleep(0.2)                     # let _wait write the status
        if self.running:
            raise TimeoutError("run still going")

    def history(self) -> list[dict[str, Any]]:
        out = []
        for meta in sorted(self.reports.glob("*.json"), reverse=True):
            stamp = meta.stem
            if not STAMP_RE.match(stamp):
                continue
            d = _read_json(meta)
            d["stamp"] = stamp
            d["has_report"] = self.path(stamp, "html").exists()
            d["has_orders"] = self.path(stamp, "csv").exists()
            out.append(d)
        return out

    def latest_report(self) -> str | None:
        return next((r["stamp"] for r in self.history()
                     if r["has_report"] and r.get("status") == "succeeded"), None)

    def status(self) -> dict[str, Any]:
        stamp = self._stamp
        tail = ""
        if stamp and self.path(stamp, "log").exists():
            tail = "\n".join(self.path(stamp, "log").read_text(
                errors="replace").splitlines()[-40:])
        d = _read_json(self.path(stamp, "json")) if stamp else {}
        return {"running": self.running, "stamp": stamp,
                "status": d.get("status"), "log_tail": tail}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(p: Path) -> dict[str, Any]:
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _pretty(stamp: str) -> str:
    return datetime.strptime(stamp, "%Y%m%d-%H%M%S").strftime("%a %d %b %Y, %H:%M")


# ------------------------------------------------------------- daily runs

def next_weekday_time(now: datetime, hh: int, mm: int) -> datetime:
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if t <= now:
        t += timedelta(days=1)
    while t.weekday() >= 5:                # Saturday/Sunday: markets closed
        t += timedelta(days=1)
    return t


def _daily(runner: Runner, hh: int, mm: int) -> None:
    while True:
        target = next_weekday_time(datetime.now(), hh, mm)
        time.sleep(max(1.0, (target - datetime.now()).total_seconds()))
        runner.start()


# ------------------------------------------------------------------ pages

TABS: list[tuple[str, str, str]] = [   # (key, label, path)
    ("overview", "Overview", "/"), ("stocks", "Stocks", "/stocks"),
    ("etfs", "ETFs", "/etfs"),
    ("sell", "When to sell", "/sell"), ("tracker", "Tracker", "/tracker"),
    ("graphs", "Graphs", "/graphs"),
    ("montecarlo", "Monte Carlo", "/montecarlo"), ("risk", "Risk (VaR)", "/risk"),
    ("macro", "Macro", "/macro"),
    ("checks", "Checks", "/checks"),
    ("runs", "History", "/runs")]
_PATH_TO_TAB = {path: key for key, _, path in TABS}
_TAB_PATH = {key: path for key, _, path in TABS}
_TAB_LABEL = {key: label for key, label, _ in TABS}

_BAR_JS = """
(function(){
var b=document.getElementById('umw-run'),s=document.getElementById('umw-st');
if(!b){return;}
function poll(){fetch('/api/status').then(function(r){return r.json();}).then(function(d){
 if(d.running){b.disabled=true;b.textContent='Running...';
  var last=(d.log_tail||'').split('\\n').filter(Boolean).pop()||'';
  s.textContent=last.slice(0,120);s.title=last;setTimeout(poll,2000);}
 else{b.disabled=false;b.textContent='Run now';
  if(d.status&&b.dataset.started){
   if(d.status==='succeeded'){location.href=location.pathname;}
   else{s.textContent='Last run '+d.status+' - see History for the log.';}}}
}).catch(function(){setTimeout(poll,4000);});}
b.addEventListener('click',function(){
 b.disabled=true;b.dataset.started='1';s.textContent='Starting...';
 fetch('/api/run',{method:'POST',headers:{'X-UMushroom-Run':'1'}})
 .then(function(r){if(r.status===409){s.textContent='A run is already going.';}poll();});
});
poll();
})();
"""


def _short_stamp(stamp: str) -> str:
    return datetime.strptime(stamp, "%Y%m%d-%H%M%S").strftime("%a %d %b %H:%M")


def nav(active: str, run: str | None = None, latest: str | None = None,
        runner: Runner | None = None, payload: dict | None = None) -> Shell:
    """v4.22: the dashboard frame for the local website (every section is its
    own URL; the sidebar lists the recent runs; Run now lives in its foot)."""
    q = f"?run={run}" if run and run != latest else ""
    links = {key: f"{path}{q if key != 'runs' else ''}" for key, _, path in TABS}
    notice = ""
    if run and latest and run != latest:
        notice = (f"Viewing an older run ({html.escape(_pretty(run))}). "
                  f'<a href="{_TAB_PATH.get(active, "/")}">Back to the latest</a>')
    runs: list[tuple[str, str, str]] = []
    if runner is not None:
        for r in runner.history()[:6]:
            ok = r.get("status") == "succeeded" and bool(r["has_report"])
            href = f"/?run={r['stamp']}" if ok else f"/reports/{r['stamp']}.log"
            runs.append((_short_stamp(r["stamp"]), href, "ok" if ok else "bad"))
    meta = (payload or {}).get("meta", {})
    action = ('<button class="btn" id="umw-run" type="button">Run now</button>'
              '<span class="st" id="umw-st" aria-live="polite"></span>')
    return Shell(active=active, links=links, mode="pages",
                 as_of=str(meta.get("as_of") or "")[:10], notice=notice, action=action,
                 runs=tuple(runs), extra_js=_BAR_JS)


def inject_toolbar(report_html: str, latest: str | None, run: str | None = None) -> str:
    """Put a slim 'back to the dashboard' bar at the top of a stored full report."""
    m = re.search(r"<body[^>]*>", report_html, flags=re.IGNORECASE)
    bar = report_bar("/", _pretty(run) if run else "")
    if not m:
        return bar + report_html
    return report_html[:m.end()] + bar + report_html[m.end():]


_DATA_RE = re.compile(r'<script id="report-data" type="application/json">(.*?)</script>',
                      re.DOTALL)


def payload_from_report(report_html: str) -> dict[str, Any] | None:
    """The report embeds all its data as JSON; read it back."""
    m = _DATA_RE.search(report_html)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


_EMPTY: dict[str, Any] = {"meta": {}}


def _site_page(title: str, body: str, active: str, runner: Runner,
               run: str | None = None, payload: dict | None = None,
               sub: str | None = None) -> str:
    return render_page(payload or _EMPTY, body, title=title,
                       nav=nav(active, run, runner.latest_report(), runner, payload), sub=sub)


def empty_page(runner: Runner, active: str = "overview") -> str:
    return _site_page("UMushroom", (
        "<h2>No report yet</h2><p>Press <b>Run now</b> at the top. The first run "
        "downloads prices and backtests every sleeve, so it can take several "
        "minutes. This page updates by itself when it's done.</p>"),
        active, runner, sub=f"v{__version__}")


def history_page(runner: Runner) -> str:
    rows = []
    for r in runner.history():
        s = r["stamp"]
        links = []
        if r["has_report"]:
            links.append(f'<a href="/?run={s}">Open</a>')
            links.append(f'<a href="/reports/{s}.html">Full report</a>')
        if r["has_orders"]:
            links.append(f'<a href="/reports/{s}.csv">Orders CSV</a>')
        links.append(f'<a href="/reports/{s}.log">Log</a>')
        st = str(r.get("status", "unknown"))
        rows.append(f'<tr><td>{html.escape(_pretty(s))}</td>'
                    f'<td class="{html.escape(st)}">{html.escape(st)}</td>'
                    f'<td>{" · ".join(links)}</td></tr>')
    table = ('<div class="scroll"><table class="data runs-tbl"><thead><tr><th>Run</th>'
             '<th>Status</th><th>Files</th></tr></thead><tbody>' + "".join(rows)
             + "</tbody></table></div>" if rows
             else "<p>No runs yet. Press <b>Run now</b> at the top.</p>")
    return _site_page("Run history", '<p class="note">Every run is kept, newest first.'
                      "</p>" + table, "runs", runner, sub=f"v{__version__}")


def _cls(v: float | None) -> str:
    return "" if v is None else ("up" if v > 0 else "down" if v < 0 else "")


def _run_picker(runner: Runner, run: str, action: str) -> str:
    opts = "".join(
        f'<option value="{r["stamp"]}"{" selected" if r["stamp"] == run else ""}>'
        f'{html.escape(_pretty(r["stamp"]))}</option>'
        for r in runner.history() if r["has_report"] and r.get("status") == "succeeded")
    return (f'<form class="pick" method="get" action="{action}"><label for="run">'
            f'Prediction from</label><select id="run" name="run">{opts}</select>'
            f'<button type="submit">Show</button>'
            f'<a href="{action}?run={run}&amp;refresh=1">Refresh prices</a></form>')


def payload_urls(payload: dict) -> dict[str, str]:
    """v4.24: ticker -> its buy / sell page, from every table in the report."""
    out: dict[str, str] = {}
    rows = list(payload.get("orders") or []) + list(payload.get("top") or [])
    for lst in (payload.get("top_etfs") or {}).values():
        rows += list(lst or [])
    for r in rows:
        if r.get("url") and r.get("ticker"):
            out.setdefault(str(r["ticker"]), str(r["url"]))
    return out


def _fx_note(res: TrackerResult, ccy: str) -> str:
    """v4.25: what the values include -- said plainly either way."""
    if res.fx:
        return (f"Values are in {html.escape(ccy or 'the account currency')}, currency "
                "moves included, as the UMushroom account counts them; prices are in "
                "each listing's own currency.")
    missing = sorted({p.ticker for p in res.priced if not p.fx})
    return ("Values are approximate: no exchange rate was found for "
            f"{html.escape(', '.join(missing))}, so their currency moves are ignored.")


def tracker_body(res: TrackerResult, ccy: str, label: str = "",
                 urls: dict[str, str] | None = None) -> str:
    out = [(f'<h2>Since the prediction</h2><p class="note">{label}Each pick\'s latest '
            f"close compared with its close on {html.escape(res.as_of)}, when the model "
            f"chose it. {_fx_note(res, ccy)}</p>")]
    priced = res.priced
    if not res.positions:
        return out[0] + '<p class="note">This run has no orders.</p>'
    if not priced:
        out.append('<p class="note">No prices found yet (Yahoo had nothing newer, or '
                   "the download failed). Try <b>Refresh prices</b> later.</p>")
    elif not res.newer:
        # v4.21: on the morning a run is published its latest close IS the
        # prediction close -- every change is 0.00%, which the site used to
        # show as "Up / down 0 / 21" as if the picks had all failed
        out.append(f'<div class="due"><b>No close newer than {html.escape(res.as_of)} '
                   "yet</b>Every holding still shows the price the model chose it at, so "
                   "there is nothing to score: the first reading comes after the next "
                   "close. Earlier books are scored week by week below.</div>")
    else:
        up = res.up
        best = max(priced, key=lambda p: p.change or 0.0)
        worst = min(priced, key=lambda p: p.change or 0.0)
        last = max((p.last_date or "") for p in priced)
        out.append('<section class="tiles">' + "".join([
            _tile("Value at prediction", f"{res.value_then:,.0f} {ccy}",
                  f"{len(priced)} priced holdings"),
            _tile("Value now", f"{res.value_now:,.0f} {ccy}", f"prices to {last}"),
            _tile("Change", "n/a" if res.change is None else f"{res.change:+.2%}",
                  f"{res.value_now - res.value_then:+,.0f} {ccy}"),
            _tile("Up / down", f"{up} / {len(priced) - up}", "holdings"),
            _tile("Best", f"{best.change:+.1%}", best.ticker),
            _tile("Worst", f"{worst.change:+.1%}", worst.ticker),
        ]) + "</section>")
        path = res.portfolio_path()
        if len(path) >= 2:
            out.append(line_chart(
                "Your picks since the prediction",
                f"Value of this run's orders on each trading day since {res.as_of} "
                f"({ccy}, currency moves {'included' if res.fx else 'partly ignored'}).",
                [d.strftime("%Y-%m-%d") for d in path.index],
                [("Portfolio value", [float(v) for v in path], "--series-1")], "money"))
        ranked = sorted(priced, key=lambda p: p.change or 0.0, reverse=True)
        out.append(bar_chart(
            "Change since the prediction, by holding",
            f"Value now against the value on the prediction date, in {ccy}.",
            [p.ticker for p in ranked], [p.change or 0.0 for p in ranked], "pct"))
    head = ["Sleeve", "Ticker", "Trade", "Name", "Shares", "Price at prediction",
            "Price now", "Change", "Value then", "Value now", "Gain", "Last price"]
    rows = []
    for p in sorted(res.positions, key=lambda p: (p.sleeve, -(p.change or -9))):
        cur = f" {p.currency}" if p.currency else ""
        gain = None if p.value_now is None else p.value_now - p.value_then
        rows.append(
            f"<tr><td>{html.escape(p.sleeve)}</td><td>{html.escape(p.ticker)}</td>"
            f"<td>{trade_link((urls or {}).get(p.ticker))}</td>"
            f"<td>{html.escape(p.name)}</td>"
            f"<td>{'' if p.shares is None else html.escape(f'{p.shares:g}')}</td>"
            f"<td>{'n/a' if p.price_then is None else html.escape(_price(p.price_then) + cur)}"
            f"</td><td>{'n/a' if p.price_now is None else html.escape(_price(p.price_now) + cur)}"
            f'</td><td class="{_cls(p.change)}">'
            f"{'no price' if p.change is None else f'{p.change:+.2%}'}</td>"
            f"<td>{p.value_then:,.0f}</td>"
            f"<td>{'n/a' if p.value_now is None else f'{p.value_now:,.0f}'}</td>"
            f'<td class="{_cls(gain)}">{"n/a" if gain is None else f"{gain:+,.0f}"}</td>'
            f"<td>{html.escape(p.last_date or '')}</td></tr>")
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    out.append(f'<h2>Every holding</h2><div class="scroll"><table class="data"><thead>'
               f'<tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return "".join(out)


def record_body(rows: list[RecordRow], days: int, benchmark: str = BENCHMARK,
                ccy: str = "GBP") -> str:
    """v4.21: the live track record -- one holding period's book per row,
    scored to the close the next book was bought at, against the benchmark
    over exactly the same closes. v4.25: both in the account currency."""
    fx = all(r.fx for r in rows)
    money = (f"Book and benchmark are in {html.escape(ccy)}, currency moves included."
             if fx else "Currency moves are ignored where no exchange rate was found.")
    out = [(f'<h2>Week by week</h2><p class="note">Every holding period\'s book under '
            f"the {days}-trading-day rule (the first run of the period), scored from the "
            "close its signals were built on to the close the next book was bought at; "
            f"the current book is scored so far. {html.escape(benchmark)} (MSCI World) "
            f"is scored over the same closes. {money}</p>")]
    if not rows:
        return out[0] + '<p class="note">No books yet.</p>'
    done = [r for r in rows if r.until and r.change is not None]
    if done:
        act = [r.active for r in done if r.active is not None]
        beat = sum(1 for a in act if a > 0)
        out.append('<section class="tiles">' + "".join([
            _tile("Weeks scored", str(len(done)), "completed holding periods"),
            _tile("Beat MSCI World", f"{beat} of {len(act)}" if act else "n/a", "weeks"),
            _tile("Average week", f"{float(np.mean([r.change for r in done])):+.2%}",
                  f"the book, in {ccy}" if fx else "the book"),
            _tile("Average vs world", f"{float(np.mean(act)):+.2%}" if act else "n/a",
                  "per week"),
        ]) + "</section>")
    head = ["Book", "Signals of", "Held until", "Book", "Up / down", "MSCI World",
            "Book minus world"]
    trs = []
    for r in reversed(rows):                        # newest first
        scored = bool(r.until) or bool(r.last_date and r.last_date > r.as_of[:10])
        until = r.until or (f"so far, to {r.last_date}" if scored else "so far")
        if r.change is None:
            chg, ud, bench, act_s = "no prices", "", "", ""
        elif not scored:
            chg, ud, bench, act_s = "no close yet", "", "", ""
        else:
            chg, ud = f"{r.change:+.2%}", f"{r.up} / {r.n - r.up}"
            bench = "n/a" if r.benchmark is None else f"{r.benchmark:+.2%}"
            act_s = "n/a" if r.active is None else f"{r.active:+.2%}"
        trs.append(
            f"<tr><td>{html.escape(_pretty(r.stamp))}</td><td>{html.escape(r.as_of[:10])}"
            f"</td><td>{html.escape(until)}</td>"
            f'<td class="{_cls(r.change if scored else None)}">{chg}</td><td>{ud}</td>'
            f'<td class="{_cls(r.benchmark if scored else None)}">{bench}</td>'
            f'<td class="{_cls(r.active if scored else None)}">{act_s}</td></tr>')
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    out.append(f'<div class="scroll"><table class="data"><thead><tr>{th}</tr></thead>'
               f'<tbody>{"".join(trs)}</tbody></table></div>')
    return "".join(out)


def _pct(v: float | None, digits: int = 2) -> str:
    return "n/a" if v is None else f"{v:+.{digits}%}"


def _live_reading(v: dict) -> str:
    """One plain sentence on what the live record means, by the numbers."""
    n, share = int(v.get("n") or 0), v.get("share_as_bad")
    weeks = f"{n} independent period{'s' if n != 1 else ''}"
    if share is None:
        out = f"Too little walk-forward history to compare {weeks} against."
    elif share < 0.05:
        out = (f"The live record is worse than all but {share:.1%} of {weeks.split()[0]}-"
               "period stretches of the walk-forward: a real warning -- though by the "
               "rules this strategy runs on, a few weeks cannot overturn years of "
               "out-of-sample evidence.")
    elif share < 0.25:
        out = (f"Weaker than the walk-forward's average, but stretches like this "
               f"happened in {share:.0%} of it.")
    else:
        out = "In line with the walk-forward."
    streak, after, base = v.get("streak") or 0, v.get("after_streak_next"), v.get("all_next")
    if streak >= 2 and after is not None and base is not None and (v.get("after_streak_n") or 0) >= 5:
        tail = (f" In the walk-forward, after {streak} periods in a row of the picks lagging "
                f"the universe, the next period's picks {'beat' if after >= 0 else 'lagged'} "
                f"it by {abs(after):.2%} on average ({v['after_streak_n']} times), against "
                f"{base:+.2%} for every period: ")
        tail += ("a losing run was not, in the model's own history, a sign that it had "
                 "stopped working." if after >= base else
                 "in the model's own history, losing runs tended to continue.")
        out += tail
    return out


def live_body(payload: dict) -> str:
    """v4.25: is the model right? Every logged forecast scored at its horizon."""
    parts: list[str] = []
    for sl in payload.get("sleeves") or []:
        live = sl.get("live") or {}
        rows = live.get("rows") or []
        if not rows:
            continue
        h = int(live.get("horizon") or 0)
        name = _label(str(sl.get("name") or ""))
        parts.append(f"<h3>{html.escape(name)}: {h}-trading-day forecasts</h3>")
        v = live.get("verdict")
        if v:
            parts.append('<section class="tiles">' + "".join([
                _tile("Periods counted", str(v["n"]), "non-overlapping"),
                _tile("Live IC", f"{v['mean_ic']:+.3f}",
                      f"walk-forward {v['bt_mean_ic']:+.3f} a period"),
                _tile("Picks vs universe", _pct(v.get("mean_active")), "average a period"),
                _tile("As bad in the walk-forward",
                      "n/a" if v.get("share_as_bad") is None else f"{v['share_as_bad']:.1%}",
                      f"of {v['n']}-period stretches"),
            ]) + "</section>")
            parts.append(f'<p class="note">{html.escape(_live_reading(v))}</p>')
        head = ["Signals of", "Scored to", "Instruments", "IC", "Picks", "Universe",
                "Picks minus universe", "Counted"]
        trs = []
        for r in reversed(rows):                       # newest first
            trs.append(
                f"<tr><td>{html.escape(r['run_date'])}</td>"
                f"<td>{html.escape(r['scored_to'])}</td><td>{int(r['n'])}</td>"
                f'<td class="{_cls(r.get("ic"))}">'
                f"{'n/a' if r.get('ic') is None else format(r['ic'], '+.3f')}</td>"
                f'<td class="{_cls(r.get("picks"))}">{_pct(r.get("picks"))}</td>'
                f'<td class="{_cls(r.get("universe"))}">{_pct(r.get("universe"))}</td>'
                f'<td class="{_cls(r.get("active"))}">{_pct(r.get("active"))}</td>'
                f"<td>{'yes' if r.get('independent') else 'overlaps'}</td></tr>")
        th = "".join(f"<th>{html.escape(x)}</th>" for x in head)
        parts.append(f'<div class="scroll"><table class="data"><thead><tr>{th}</tr>'
                     f'</thead><tbody>{"".join(trs)}</tbody></table></div>')
    if not parts:
        return ""
    intro = ('<h2>Is the model right? The live record</h2><p class="note">Every run logs '
             "the model's score for every instrument it ranks. Once the forecast horizon "
             "has passed, each day's scores are checked against what the prices actually "
             "did: the IC is the rank correlation between scores and returns (+1 = perfect "
             "order, 0 = no skill, below 0 = the wrong way round), and the picks are "
             "compared with the average of everything ranked. Rows that share days with "
             "an earlier counted row are shown but not counted again. The walk-forward "
             "figures are the same model's out-of-sample record, so the comparison asks "
             "how often the backtest had a stretch this good or bad.</p>")
    return intro + "".join(parts)


# --------------------------------------------------------- when to sell
# --------------------------------------------------------- when to sell

_ACT_CLASS = {"SELL ALL": "act-sell", "SELL": "act-sell", "BUY NEW": "act-buy",
              "BUY MORE": "act-buy", "HOLD": "act-hold"}


def _as_of(payload: dict) -> str:
    return str(payload.get("meta", {}).get("as_of", ""))[:10]


def _days(payload: dict) -> int:
    return int(payload.get("meta", {}).get("rebalance_days") or DEFAULT_REBALANCE_DAYS)


def _runs_for(site: Site, latest: str, latest_payload: dict) -> list[tuple[str, str]]:
    """(stamp, as_of) of every successful run with a report, the latest included."""
    runs: list[tuple[str, str]] = []
    for r in site.runner.history():
        if not r["has_report"] or r.get("status") != "succeeded":
            continue
        p = latest_payload if r["stamp"] == latest else site.payload(r["stamp"])
        if p is not None and _as_of(p):
            runs.append((str(r["stamp"]), _as_of(p)))
    if latest not in {s for s, _ in runs} and _as_of(latest_payload):
        runs.append((latest, _as_of(latest_payload)))
    return runs


def _cadence_for(site: Site, latest: str, latest_payload: dict) -> Cadence:
    """v4.21: the weekly rule (rebalance.cadence) over the run history."""
    return cadence(_runs_for(site, latest, latest_payload), latest, _days(latest_payload))


def tracked_run(site: Site, latest: str, latest_payload: dict) -> str:
    """The run the Tracker scores: this period's book (what the team holds),
    or the latest run when it opens a new period."""
    try:
        cad = _cadence_for(site, latest, latest_payload)
    except ValueError:
        return latest
    if cad.rebalance_now:
        return cad.book or latest       # v4.26: a same-day repeat run tracks the book
    return cad.held or latest


def _default_held(site: Site, latest: str, latest_payload: dict) -> str | None:
    """The run whose orders you hold under the weekly rule: at a rebalance
    point, the previous period's book; mid-period, this period's first run."""
    try:
        return _cadence_for(site, latest, latest_payload).held
    except ValueError:
        return None


def _held_picker(site: Site, run: str, held: str | None) -> str:
    items: list[str] = []
    for r in site.runner.history():
        if not r["has_report"] or r.get("status") != "succeeded" or r["stamp"] == run:
            continue
        p = site.payload(r["stamp"])
        sig = f" (signals {_as_of(p)})" if p and _as_of(p) else ""
        items.append(f'<option value="{r["stamp"]}"{" selected" if r["stamp"] == held else ""}>'
                    f'{html.escape(_pretty(r["stamp"]) + sig)}</option>')
    opts = "".join(items)
    if not opts:
        return ""
    return (f'<form class="pick" method="get" action="/sell">'
            f'<input type="hidden" name="run" value="{html.escape(run)}">'
            f'<label for="held">I bought the orders from</label>'
            f'<select id="held" name="held">{opts}</select>'
            f'<button type="submit">Show</button></form>')


def _due_box(cad: Cadence, days: int, today: date) -> str:
    when = cad.next_rebalance.strftime("%a %d %b %Y")
    if cad.rebalance_now:
        return (f'<div class="due now"><b>Sell / rebalance now</b>This run opens a new '
                f"{days}-trading-day holding period (from {cad.period_start.strftime('%a %d %b')}"
                f"). Do the sells below first, then the buys; then hold until {when}.</div>")
    left = max(0, int(np.busday_count(today, cad.next_rebalance)))
    return (f'<div class="due"><b>Holding period in progress -- next sell / rebalance: '
            f"{when}</b>{left} trading day{'s' if left != 1 else ''} to go. The strategy "
            "holds this period's book until then -- no sells, and today's picks on the "
            "Stocks tab are for anyone entering now, not a signal to trade.</div>")


def _capital_note(meta: dict) -> str:
    """v4.27: what the orders were sized on -- the account value, or the fixed
    starting balance (then the buys can cost more than the account holds)."""
    cap, src = meta.get("capital"), meta.get("capital_source")
    if cap is None:
        return ""
    amount = html.escape(f"{float(cap):,.0f} {meta.get('base_currency', '')}".strip())
    if src == "account_value":
        return (f'<p class="note">Orders are sized on the account value <b>{amount}</b> '
                "(the ACCOUNT_VALUE repository variable). Update it to the UMushroom "
                "account's total (holdings + cash) before each rebalance.</p>")
    return (f'<p class="note"><b>Orders are sized on the fixed starting balance of '
            f"{amount}</b>, not on what the account is worth now: after losses the buys "
            "below can cost more than the cash the sells raise. Set the ACCOUNT_VALUE "
            "repository variable (GitHub: Settings, Secrets and variables, Actions, "
            "Variables) to the account's total value.</p>")


def sell_body(site: Site, run: str, latest_payload: dict, held: str | None,
              today: date | None = None, picker: bool = True) -> str:
    today = today or date.today()
    meta = latest_payload.get("meta", {})
    days = int(meta.get("rebalance_days") or DEFAULT_REBALANCE_DAYS)
    ccy = meta.get("base_currency", "")
    intro = ('<h2>When to sell</h2><p class="note">The strategy\'s own, backtested exit '
             f"rule: every {days} trading days the model re-ranks everything; anything "
             "that is no longer a pick is sold and the money moves to the new picks. "
             "There is no stop-loss or profit target -- none was tested, and an untested "
             "rule can quietly wipe out the measured alpha. This is the contest "
             "strategy's output, not financial advice.</p>") + _capital_note(meta)
    cad = _cadence_for(site, run, latest_payload)
    explicit = held is not None and held != cad.held      # the picker overrides
    held = held or cad.held
    held_payload = site.payload(held) if held else None
    if held_payload is None:
        return (intro + _due_box(cad, days, today)
                + '<p class="note">Only one prediction so far, so there is nothing to '
                  "sell yet: buy today's orders (Stocks tab). After the next run this "
                  "page lists exactly what to sell.</p>")
    if not cad.rebalance_now and not explicit:
        return (intro + (_held_picker(site, run, held) if picker else "")
                + _due_box(cad, days, today)
                + f'<p class="note">Your book is the orders of {html.escape(_pretty(str(held)))} '
                  f"(signals of {html.escape(_as_of(held_payload))}), the first run of "
                  "this period. Nothing to do until the next period starts.</p>")

    # v4.26: on a same-day repeat run, the period's book is the FIRST run's
    # orders (what the record scores and what the morning's readers bought)
    target, target_payload = run, latest_payload
    if cad.rebalance_now and cad.book and cad.book != run and not explicit:
        book_payload = site.payload(cad.book)
        if book_payload is not None:
            target, target_payload = cad.book, book_payload
    actions = plan(held_payload, target_payload)
    out = [intro, _held_picker(site, run, held) if picker else "", _due_box(cad, days, today)]

    gains: dict[str, float] = {}
    try:                                     # how each holding did since you bought
        res = site.tracked(held, held_payload, refresh=False)  # type: ignore[arg-type]
        gains = {p.ticker: p.change for p in res.positions if p.change is not None}
    except Exception:  # noqa: BLE001,S110 -- prices are a nice-to-have here
        pass

    counts = {k: sum(a.action == k for a in actions) for k in ORDER_KEYS}
    out.append('<section class="tiles">' + "".join([
        _tile("Sell completely", str(counts["SELL ALL"]), "dropped from the picks"),
        _tile("Trim", str(counts["SELL"]), "still picks, too big"),
        _tile("Buy new", str(counts["BUY NEW"]), "new picks"),
        _tile("Top up", str(counts["BUY MORE"]), "still picks, too small"),
        _tile("Hold", str(counts["HOLD"]), "no trade needed"),
    ]) + "</section>")

    head = ["Action", "Sleeve", "Ticker", "On UMushroom", "Name", "Held", "Target", "Trade",
            "≈ Value", "Since you bought", "Why"]
    urls = {**payload_urls(held_payload), **payload_urls(latest_payload)}
    rows = []
    for a in actions:
        t = a.trade_shares
        val = None if a.price is None else abs(t) * a.price
        g = gains.get(a.ticker)
        rows.append(
            f'<tr><td><span class="act {_ACT_CLASS[a.action]}">{a.action}</span></td>'
            f"<td>{html.escape(a.sleeve)}</td><td>{html.escape(a.ticker)}</td>"
            f"<td>{trade_link(urls.get(a.ticker))}</td>"
            f"<td>{html.escape(a.name)}</td><td>{_sh(a.held_shares)}</td>"
            f"<td>{_sh(a.target_shares)}</td>"
            f"<td>{'' if t == 0 else ('+' if t > 0 else '-') + _sh(abs(t))}</td>"
            f"<td>{'' if not val else html.escape(f'{val:,.2f} {a.currency}'.strip())}</td>"
            f'<td class="{_cls(g)}">{"" if g is None else f"{g:+.1%}"}</td>'
            f"<td>{html.escape(a.reason)}</td></tr>")
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    out.append(f'<h3 class="sec">Trade list (sells first)</h3><div class="scroll">'
               f'<table class="data sell-tbl"><thead><tr>{th}</tr></thead>'
               f'<tbody>{"".join(rows)}</tbody></table></div>')
    out.append(f'<p class="note">Held = the orders from {html.escape(_pretty(held))} '  # type: ignore[arg-type]
               f"(signals of {html.escape(_as_of(held_payload))}). Target = the orders "
               f"of {html.escape(_pretty(target))}"
               f"{', the first run of this period' if target != run else ''}. Values are in each listing's currency ({ccy} totals are on the "
               "Stocks tab). Check prices on UMushroom before trading; differences under "
               "2% of a position are left alone.</p>")
    return "".join(out)


ORDER_KEYS = ("SELL ALL", "SELL", "BUY NEW", "BUY MORE", "HOLD")


def _sh(v: float) -> str:
    return html.escape(f"{v:,.6f}".rstrip("0").rstrip(".") if v % 1 else f"{v:,.0f}")


# ----------------------------------------------------------------- macro

_SHORT = {"market": "Stocks", "us10y": "US 10y", "gilts": "Gilts", "oil": "Oil",
          "usd": "USD", "gbp": "GBP", "gold": "Gold", "vix": "VIX"}


def _lvl(row: dict) -> str:
    v = row["level"]
    if row["key"] == "us10y":
        return f"{v:.2f}%"
    if row["kind"] == "diff":
        return f"{v:.1f}"
    return _price(v)


def _chg(row: dict, v: float) -> str:
    if row["kind"] == "ret":
        return f"{v:+.1%}"
    return f"{v:+.2f} {row['unit']}"


def _r1(v: float) -> float:
    return round(v, 3)


def _signed(v: float) -> str:
    """+1.2% / -0.4% / 0.0% (never "-0.0%")."""
    return "0.0%" if abs(v) < 0.0005 else f"{v:+.1%}"


def macro_body(res: MacroResult, ccy: str) -> str:
    out = [('<h2>Macro: the economy and our exposure</h2><p class="note">What the '
            "big macro drivers are doing, and how sensitive today's holdings are to "
            "them (estimated from about 2 years of weekly returns). Reporting only -- "
            "the model is not changed by anything here.</p>")]
    pts = talking_points(res)
    if pts:
        items = "".join(f"<li>{html.escape(t)}</li>" for t in pts)
        out.append(f'<div class="card"><h3>For the pitch: macro &amp; risk talking '
                   f'points</h3><ul class="notes">{items}</ul></div>')

    if res.dashboard:
        rows = "".join(
            f"<tr><td>{html.escape(r['label'])}</td><td>{html.escape(_lvl(r))}</td>"
            f'<td class="{_cls(r["chg_1m"])}">{html.escape(_chg(r, r["chg_1m"]))}</td>'
            f'<td class="{_cls(r["chg_3m"])}">{html.escape(_chg(r, r["chg_3m"]))}</td>'
            f"<td>{r['pct_1y']:.0%}</td><td>{html.escape(r['date'])}</td></tr>"
            for r in res.dashboard)
        out.append('<h3 class="sec">What the economy is doing</h3><div class="scroll">'
                   '<table class="data"><thead><tr><th>Driver</th><th>Latest</th>'
                   "<th>1 month</th><th>3 months</th><th>Where in its 1-year range"
                   "</th><th>As of</th></tr></thead><tbody>" + rows
                   + "</tbody></table></div><p class=\"note\">“Where in its 1-year "
                   "range”: 100% = the highest level of the past year, 0% = the "
                   "lowest.</p>")

    keys = [f.key for f in FBY.values() if res.sensitivity(f.key) is not None]
    if keys:
        out.append(bar_chart(
            "How much the portfolio moves in a typical week for each driver",
            "Portfolio move when the driver has a normal-sized (1 standard deviation) "
            "week. Stocks = the market itself; the others are on top of the market.",
            [_SHORT[k] for k in keys], [res.sensitivity(k) or 0.0 for k in keys], "pct2"))
    if res.scenarios:
        out.append(bar_chart(
            "Stress tests: estimated portfolio move",
            "Linear, other-things-equal estimates from the sensitivities -- a sketch "
            "of the risk, not a forecast.",
            [_SHORT[s["key"]] for s in res.scenarios],
            [s["impact"] for s in res.scenarios], "pct"))
        rows_s = "".join(
            f"<tr><td>{html.escape(s['label'])}</td>"
            f'<td class="{_cls(_r1(s["impact"]))}">{_signed(s["impact"])}</td>'
            f"<td>{html.escape(s['note'])}</td></tr>" for s in res.scenarios)
        out.append('<div class="scroll"><table class="data"><thead><tr><th>Scenario</th>'
                   "<th>Portfolio</th><th>Note</th></tr></thead><tbody>" + rows_s
                   + "</tbody></table></div>")

    if res.currency_mix:
        rows_c = "".join(f"<tr><td>{html.escape(c)}</td><td>{w:.1%}</td></tr>"
                         for c, w in res.currency_mix.items())
        out.append('<h3 class="sec">Currency mix (by value)</h3><div class="scroll">'
                   '<table class="data"><thead><tr><th>Priced in</th><th>Share</th></tr>'
                   f"</thead><tbody>{rows_c}</tbody></table></div>")

    head = ["Ticker", "Sleeve", "Weight", "Market beta"] + [
        _SHORT[k] for k in keys if k != "market"]
    rows_h = []
    for h in sorted(res.holdings, key=lambda h: -h.weight):
        cells = [html.escape(h.ticker), html.escape(h.sleeve), f"{h.weight:.1%}",
                 "n/a" if "market" not in h.betas else f"{h.betas['market']:.2f}"]
        for k in keys:
            if k == "market":
                continue
            b, sd = h.betas.get(k), res.shock_sd.get(k)
            cells.append("n/a" if b is None or sd is None else
                         f'<span class="{_cls(b * sd)}">{b * sd:+.2%}</span>')
        rows_h.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    th = "".join(f"<th>{html.escape(x)}</th>" for x in head)
    out.append(f'<h3 class="sec">Each holding</h3><p class="note">Driver columns: the '
               f"holding's move in a typical week for that driver, on top of the market."
               f'</p><div class="scroll"><table class="data"><thead><tr>{th}</tr></thead>'
               f'<tbody>{"".join(rows_h)}</tbody></table></div>')
    out.append(f'<p class="note">Sensitivities cover {res.coverage:.0%} of the '
               "portfolio's value (holdings with under 40 weeks of shared history are "
               "left out). Prices: Yahoo Finance, refreshed at most hourly.</p>")
    return "".join(out)


# ----------------------------------------------------------------- server

_TYPES = {"html": "text/html; charset=utf-8", "csv": "text/csv; charset=utf-8",
          "log": "text/plain; charset=utf-8"}
PRICE_TTL = 15 * 60          # seconds before tracker prices are fetched again
MACRO_TTL = 60 * 60          # macro data: at most hourly
MAX_FAILS, LOCKOUT_WINDOW = 10, 600      # 10 wrong passwords in 10 min -> wait
LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}


@dataclass
class Site:
    runner: Runner
    fetch: FetchFn = yahoo_closes
    _payloads: dict[str, tuple[float, dict]] = field(default_factory=dict)
    _tracked: dict[str, tuple[float, TrackerResult]] = field(default_factory=dict)
    _record: dict[str, tuple[float, list[RecordRow]]] = field(default_factory=dict)
    password: str | None = None          # team mode: required for other computers
    trust_local: bool = True             # this computer itself needs no password
    _macro: dict[str, tuple[float, MacroResult]] = field(default_factory=dict)
    _fails: dict[str, list[float]] = field(default_factory=dict)

    def macro(self, stamp: str, payload: dict, refresh: bool) -> MacroResult:
        hit = self._macro.get(stamp)
        if hit and not refresh and time.time() - hit[0] < MACRO_TTL:
            return hit[1]
        res = analyse(payload, self.fetch)
        self._macro[stamp] = (time.time(), res)
        return res

    def locked_out(self, ip: str) -> bool:
        now = time.time()
        recent = [t for t in self._fails.get(ip, []) if now - t < LOCKOUT_WINDOW]
        self._fails[ip] = recent
        return len(recent) >= MAX_FAILS

    def check_password(self, ip: str, header: str | None) -> bool:
        if self.password is None:
            return True
        given = ""
        if header and header.startswith("Basic "):
            try:
                given = base64.b64decode(header[6:]).decode("utf-8", "replace")
                given = given.split(":", 1)[1] if ":" in given else ""
            except ValueError:
                given = ""
        ok = hmac.compare_digest(given.encode(), self.password.encode())
        if not ok and header:
            self._fails.setdefault(ip, []).append(time.time())
        return ok

    def payload(self, stamp: str) -> dict | None:
        f = self.runner.path(stamp, "html")
        if not f.is_file():
            return None
        mtime = f.stat().st_mtime
        hit = self._payloads.get(stamp)
        if hit and hit[0] == mtime:
            return hit[1]
        p = payload_from_report(f.read_text(errors="replace"))
        if p is not None:
            self._payloads[stamp] = (mtime, p)
        return p

    def tracked(self, stamp: str, payload: dict, refresh: bool) -> TrackerResult:
        hit = self._tracked.get(stamp)
        if hit and not refresh and time.time() - hit[0] < PRICE_TTL:
            return hit[1]
        res = track(payload, self.fetch)
        self._tracked[stamp] = (time.time(), res)
        return res

    def recorded(self, latest: str, payload: dict, refresh: bool) -> list[RecordRow]:
        """v4.21: every book of the weekly rule, scored to the close the next
        book was bought at (the current one so far) -- one price download."""
        hit = self._record.get(latest)
        if hit and not refresh and time.time() - hit[0] < PRICE_TTL:
            return hit[1]
        entries: list[tuple[str, dict, str | None]] = []
        for b in books(_runs_for(self, latest, payload), _days(payload)):
            p = payload if b.stamp == latest else self.payload(b.stamp)
            if p is not None:
                entries.append((b.stamp, p, b.until))
        rows = record(entries, self.fetch)
        self._record[latest] = (time.time(), rows)
        return rows


def make_handler(site: Site) -> type[BaseHTTPRequestHandler]:
    runner = site.runner

    class Handler(BaseHTTPRequestHandler):
        server_version = "UMushroom"

        def log_message(self, format: str, *args: Any) -> None:
            pass                                        # keep the terminal quiet

        def _send(self, code: int, body: str | bytes, ctype: str) -> None:
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code: int, obj: dict[str, Any]) -> None:
            self._send(code, json.dumps(obj), "application/json")

        def _not_found(self) -> None:
            self._send(404, "Not found", _TYPES["log"])

        def _allowed(self) -> bool:
            """Team mode: other computers need the team password."""
            ip = self.client_address[0]
            if site.password is None or (site.trust_local and ip in LOOPBACK):
                return True
            if site.locked_out(ip):
                self._send(429, "Too many wrong passwords; try again in 10 minutes.",
                           _TYPES["log"])
                return False
            if site.check_password(ip, self.headers.get("Authorization")):
                return True
            body = b"Password required"
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="UMushroom team", '
                             'charset="UTF-8"')
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False

        def do_GET(self) -> None:
            if not self._allowed():
                return
            path, _, query = self.path.partition("?")
            params = parse_qs(query)
            if path == "/runs":
                self._send(200, history_page(runner), _TYPES["html"])
            elif path == "/api/status":
                self._json(200, runner.status())
            elif m := re.fullmatch(r"/reports/(\d{8}-\d{6})\.(html|csv|log)", path):
                self._file(m.group(1), m.group(2))
            elif path in _PATH_TO_TAB:
                self._section(_PATH_TO_TAB[path], params)
            else:
                self._not_found()

        def _file(self, stamp: str, ext: str) -> None:
            f = runner.path(stamp, ext)
            if not f.is_file():
                self._not_found()
                return
            body: str | bytes = (inject_toolbar(f.read_text(errors="replace"),
                                                runner.latest_report(), stamp)
                                 if ext == "html" else f.read_bytes())
            self._send(200, body, _TYPES[ext])

        def _section(self, key: str, params: dict[str, list[str]]) -> None:
            latest = runner.latest_report()
            run = (params.get("run") or [latest or ""])[0]
            if latest is None:
                self._send(200, empty_page(runner, key), _TYPES["html"])
                return
            if not STAMP_RE.match(run):
                self._not_found()
                return
            payload = site.payload(run)
            if payload is None:
                self._not_found()
                return
            label = _TAB_LABEL[key]
            if key == "sell":
                held: str | None = (params.get("held") or [""])[0] or None
                if held is not None and not STAMP_RE.match(held):
                    self._not_found()
                    return
                body = sell_body(site, run, payload,
                                 held or _default_held(site, run, payload))
            elif key == "macro":
                ccy = payload.get("meta", {}).get("base_currency", "")
                try:
                    res_m = site.macro(run, payload, "refresh" in params)
                    body = macro_body(res_m, ccy)
                except Exception as e:  # noqa: BLE001 -- network errors vary
                    body = (f'<h2>Macro</h2><p class="note">Could not get macro data: '
                            f"{html.escape(str(e))}</p>")
            elif key == "tracker":
                ccy = payload.get("meta", {}).get("base_currency", "")
                try:
                    res = site.tracked(run, payload, "refresh" in params)
                    lp = payload if run == latest else (site.payload(latest) or payload)
                    rows = site.recorded(latest, lp, "refresh" in params)
                    body = (_run_picker(runner, run, "/tracker")
                            + tracker_body(res, ccy, urls=payload_urls(payload))
                            + record_body(rows, _days(lp), ccy=ccy or "GBP"))
                except Exception as e:  # noqa: BLE001 -- network errors vary
                    body = (_run_picker(runner, run, "/tracker")
                            + f'<p class="note">Could not get prices: {html.escape(str(e))}'
                              "</p>")
                # v4.25: needs no download -- the run itself scored its log
                body += live_body(site.payload(latest) or payload)
            else:
                body = section_bodies(payload)[key]
            title = "UMushroom" if key == "overview" else f"UMushroom: {label}"
            page = render_page(payload, body, title=title,
                               nav=nav(key, run, latest, runner, payload))
            self._send(200, page, _TYPES["html"])

        def do_POST(self) -> None:
            if not self._allowed():
                return
            if self.path != "/api/run":
                self._not_found()
                return
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            if self.headers.get(RUN_HEADER) != "1" or (
                    origin is not None and origin != f"http://{host}"):
                self._json(403, {"error": "forbidden"})
                return
            stamp = runner.start()
            if stamp is None:
                self._json(409, {"error": "a run is already in progress"})
            else:
                self._json(202, {"stamp": stamp})

    return Handler


def make_server(runner: Runner, host: str = "127.0.0.1", port: int = 8000,
                fetch: FetchFn = yahoo_closes, password: str | None = None,
                trust_local: bool = True) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1") and not password:
        raise ValueError("refusing to open the website to other computers without "
                         "a password")
    return ThreadingHTTPServer((host, port),
                               make_handler(Site(runner, fetch, password=password,
                                                 trust_local=trust_local)))


def team_password(site_dir: Path) -> str:
    """The team password: $UMUSHROOM_PASSWORD, else one generated once and kept
    in site/team_password.txt (readable only by you)."""
    import os
    env = os.environ.get("UMUSHROOM_PASSWORD")
    if env:
        return env
    f = site_dir / "team_password.txt"
    if f.is_file() and f.read_text().strip():
        return f.read_text().strip()
    site_dir.mkdir(parents=True, exist_ok=True)
    pw = "-".join(secrets.token_hex(3) for _ in range(3))     # e.g. 3f9a1c-77b0de-...
    f.write_text(pw + "\n")
    f.chmod(0o600)
    return pw


def lan_address() -> str | None:
    """This computer's address on the local network (nothing is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))
            addr = str(sk.getsockname()[0])
            return None if addr.startswith("127.") else addr
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="umushroom-web",
                                description="Local website for the UMushroom pipeline.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--site-dir", default="site",
                   help="where runs are stored (default: ./site)")
    p.add_argument("--daily", metavar="HH:MM", default=None,
                   help="also run automatically every weekday at this time")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--validate", action="store_true",
                   help="include the validation suite in each run (slower)")
    p.add_argument("--team", action="store_true",
                   help="let teammates open it from their own computers (password "
                        "protected)")
    args = p.parse_args(argv)

    if not Path(args.config).exists():
        print(f"Can't find {args.config}. Run this from the umushroom_pkg folder.")
        return 2
    extra = ["--validate"] if args.validate else []
    runner = Runner(Path(args.site_dir).resolve(), default_command(args.config, extra))
    if args.daily:
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", args.daily)
        if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
            print("--daily must look like 07:30")
            return 2
        threading.Thread(target=_daily, args=(runner, int(m.group(1)), int(m.group(2))),
                         daemon=True).start()
    password = team_password(runner.site_dir) if args.team else None
    try:
        server = make_server(runner, host="0.0.0.0" if args.team else "127.0.0.1",
                             port=args.port, password=password)
    except OSError:
        print(f"Port {args.port} is busy. Try:  python -m umushroom.web --port {args.port + 1}")
        return 2
    url = f"http://127.0.0.1:{args.port}"
    print(f"UMushroom website: {url}")
    if args.team:
        lan = lan_address()
        print("\nTEAM MODE -- teammates can open it from their own computers:")
        if lan:
            print(f"  same Wi-Fi:      http://{lan}:{args.port}")
        print(f"  anywhere:        http://<your Tailscale address>:{args.port}  "
              "(see README)")
        print(f"  password:        {password}   (any username)")
        print("  Your Mac must stay on with this window open.\n")
    if args.daily:
        print(f"Automatic runs every weekday at {args.daily} (while this is open).")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
