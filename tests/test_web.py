"""Local website: runs, history, status, and safety checks."""
from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from umushroom.report import build_payload, render_html
from umushroom.web import (
    Runner,
    inject_toolbar,
    make_server,
    next_weekday_time,
    payload_from_report,
)


def _report_html(as_of: str = "2026-10-02") -> str:
    orders = pd.DataFrame({"sleeve": ["equity"], "shares": [100],
                           "allocated_value": [1000.0], "local_price": [10.0],
                           "listing_currency": ["GBP"], "name": ["Alpha Co"]},
                          index=["AAA.L"])
    return render_html(build_payload({"as_of": as_of, "base_currency": "GBP",
                                      "version": "t"}, [], orders))


def _fake(ok: bool = True, as_of: str = "2026-10-02"):
    """A stand-in for the real pipeline: writes a real (small) report + orders."""
    def make(report: Path, orders: Path) -> list[str]:
        src = report.parent / f"_src_{as_of}.html"
        src.write_text(_report_html(as_of))
        code = (f"import shutil;print('step 1');print('step 2');"
                f"shutil.copy({str(src)!r},{str(report)!r});"
                f"open({str(orders)!r},'w').write('ticker,shares\\nAAA,1\\n')"
                if ok else "print('boom');raise SystemExit(3)")
        return [sys.executable, "-c", code]
    return make


DAYS = pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"])


def _yahoo():
    calls = []

    def f(symbols, start):
        calls.append(symbols)
        return pd.DataFrame({"AAA.L": [990.0, 1000.0, 1050.0, 1100.0]}, index=DAYS)
    f.calls = calls
    return f


@pytest.fixture
def site(tmp_path):
    runner = Runner(tmp_path / "site", _fake())
    yahoo = _yahoo()
    server = make_server(runner, port=0, fetch=yahoo)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    runner.yahoo = yahoo
    yield runner, base
    server.shutdown()
    server.server_close()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read().decode()


def _post(url, headers=None):
    req = urllib.request.Request(url, method="POST", data=b"", headers=headers or {})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_empty_site_explains_what_to_do(site):
    _, base = site
    code, body = _get(base + "/")
    assert code == 200 and "No report yet" in body and "Run now" in body


def test_run_now_produces_report_history_and_files(site):
    runner, base = site
    code, d = _post(base + "/api/run", {"X-UMushroom-Run": "1"})
    assert code == 202
    runner.wait()
    status = json.loads(_get(base + "/api/status")[1])
    assert status["status"] == "succeeded" and "step 2" in status["log_tail"]
    _, home = _get(base + "/")
    assert "Invested today" in home and 'class="umw-bar"' in home
    _, hist = _get(base + "/runs")
    assert d["stamp"] in hist and "succeeded" in hist
    assert "AAA,1" in _get(f"{base}/reports/{d['stamp']}.csv")[1]
    assert "step 1" in _get(f"{base}/reports/{d['stamp']}.log")[1]


def test_failed_run_is_marked_and_does_not_replace_latest_report(tmp_path):
    runner = Runner(tmp_path / "site", _fake())
    runner.start()
    runner.wait()
    good = runner.latest_report()
    runner.command = _fake(ok=False)
    runner.start()
    runner.wait()
    assert runner.history()[0]["status"] == "failed"
    assert runner.history()[0]["returncode"] == 3
    assert runner.latest_report() == good


def test_only_one_run_at_a_time(tmp_path):
    slow = lambda r, o: [sys.executable, "-c", "import time;time.sleep(1.5)"]
    runner = Runner(tmp_path / "site", slow)
    assert runner.start() is not None
    assert runner.start() is None
    runner.wait()


def test_run_requires_custom_header_and_same_origin(site):
    _, base = site
    assert _post(base + "/api/run")[0] == 403
    assert _post(base + "/api/run", {"X-UMushroom-Run": "1",
                                     "Origin": "http://evil.example"})[0] == 403


@pytest.mark.parametrize("path", ["/reports/../config.yaml", "/reports/x.html",
                                  "/reports/20260101-000000.py", "/etc/passwd",
                                  "/reports/20260101-000000.html"])
def test_no_file_outside_reports_can_be_read(site, path):
    _, base = site
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + path)
    assert e.value.code == 404


def test_interrupted_run_is_marked_on_restart(tmp_path):
    reports = tmp_path / "site" / "reports"
    reports.mkdir(parents=True)
    (reports / "20260101-120000.json").write_text(json.dumps({"status": "running"}))
    runner = Runner(tmp_path / "site", _fake())
    assert runner.history()[0]["status"] == "interrupted"


def test_toolbar_goes_inside_body():
    out = inject_toolbar('<html><body class="x"><p>hi</p></body></html>', None, None)
    assert out.index('class="umw-bar"') > out.index('<body class="x">')
    assert out.index('class="umw-bar"') < out.index("<p>hi</p>")


def test_daily_schedule_skips_weekends():
    fri_evening = datetime(2026, 10, 9, 18, 0)          # Friday
    assert next_weekday_time(fri_evening, 7, 30) == datetime(2026, 10, 12, 7, 30)
    mon_early = datetime(2026, 10, 12, 6, 0)
    assert next_weekday_time(mon_early, 7, 30) == datetime(2026, 10, 12, 7, 30)


# ---- v4.13: one tab per section + since-prediction tracker -------------

def _run_once(runner, base):
    _post(base + "/api/run", {"X-UMushroom-Run": "1"})
    runner.wait()


@pytest.mark.parametrize("path,expect,absent", [
    ("/", "Invested today", "Today's orders"),
    ("/stocks", "Today's orders", "Monte Carlo: what"),
    ("/etfs", "ETF", "Monte Carlo: what"),
    ("/graphs", "Out-of-sample performance", "Today's orders"),
    ("/montecarlo", "Monte Carlo", "Today's orders"),
    ("/checks", "Validation checks", "Today's orders"),
    ("/risk", "Value at Risk", "Today's orders"),
])
def test_each_section_has_its_own_page(site, path, expect, absent):
    runner, base = site
    _run_once(runner, base)
    _, body = _get(base + path)
    assert expect in body and absent not in body
    assert 'aria-current="page"' in body
    for label in ("Overview", "Stocks", "ETFs", "When to sell", "Tracker", "Graphs",
                  "Monte Carlo", "Risk (VaR)", "Macro", "Checks", "History"):
        assert f">{label}</a>" in body


def test_tracker_shows_price_now_vs_price_at_prediction(site):
    runner, base = site
    _run_once(runner, base)
    _, body = _get(base + "/tracker")
    assert "Since the prediction" in body
    assert "+10.00%" in body                     # 1000p -> 1100p
    assert "10.00 GBP" in body and "11.00 GBP" in body
    assert "1,100 GBP" in body                   # value now


def test_tracker_caches_prices_and_refresh_refetches(site):
    runner, base = site
    _run_once(runner, base)
    _get(base + "/tracker")
    _get(base + "/tracker")
    assert len(runner.yahoo.calls) == 1
    stamp = runner.latest_report()
    _get(f"{base}/tracker?run={stamp}&refresh=1")
    assert len(runner.yahoo.calls) == 2


def test_tracker_survives_download_failure(tmp_path):
    runner = Runner(tmp_path / "site", _fake())
    runner.start()
    runner.wait()

    def broken(symbols, start):
        raise ConnectionError("no internet")
    server = make_server(runner, port=0, fetch=broken)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        code, body = _get(f"http://127.0.0.1:{server.server_address[1]}/tracker")
        assert code == 200 and "Could not get prices: no internet" in body
    finally:
        server.shutdown()
        server.server_close()


def test_older_run_can_be_viewed_in_every_tab(site):
    runner, base = site
    _run_once(runner, base)
    first = runner.latest_report()
    import time
    time.sleep(1.1)
    _run_once(runner, base)
    assert runner.latest_report() != first
    _, body = _get(f"{base}/stocks?run={first}")
    assert "Viewing an older run" in body and f"run={first}" in body
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(f"{base}/stocks?run=../../etc")
    assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(f"{base}/stocks?run=20000101-000000")
    assert e.value.code == 404


def test_payload_round_trips_through_the_report():
    p = payload_from_report(_report_html())
    assert p["orders"][0]["ticker"] == "AAA.L"
    assert payload_from_report("<html></html>") is None


# ---- v4.15: when to sell ------------------------------------------------

def test_sell_page_with_one_run_says_nothing_to_sell_yet(site):
    runner, base = site
    _run_once(runner, base)
    _, body = _get(base + "/sell")
    assert "When to sell" in body and "nothing to sell yet" in body
    assert "Sell / rebalance now" in body or "Holding period in progress" in body


def test_sell_page_lists_trades_against_the_run_you_bought_from(tmp_path):
    from umushroom.web import Site, sell_body
    runner = Runner(tmp_path / "site", _fake(as_of="2026-10-02"))
    runner.start()
    runner.wait()
    old = runner.latest_report()
    import time
    time.sleep(1.1)
    runner.command = _fake(as_of="2026-10-09")
    runner.start()
    runner.wait()
    new = runner.latest_report()
    site = Site(runner, _yahoo())
    from umushroom.web import _default_held
    assert _default_held(site, new, site.payload(new)) == old
    body = sell_body(site, new, site.payload(new), old, today=date(2026, 10, 9))
    assert "Sell / rebalance now" in body and "HOLD" in body and "+10.0%" in body


def test_sell_page_due_dates(tmp_path):
    from umushroom.rebalance import Cadence
    from umushroom.web import _due_box
    mid = Cadence("20261012-080000", False, date(2026, 10, 12), date(2026, 10, 19))
    assert "3 trading days to go" in _due_box(mid, 5, date(2026, 10, 14))
    assert "Holding period in progress" in _due_box(mid, 5, date(2026, 10, 14))
    start = Cadence("20261005-080000", True, date(2026, 10, 12), date(2026, 10, 19))
    assert "Sell / rebalance now" in _due_box(start, 5, date(2026, 10, 12))


def test_sell_page_holds_mid_period_instead_of_churning_daily(tmp_path):
    """v4.21: two runs in the SAME week (signals of Mon 10-05 and Tue 10-06)
    must not produce a trade list -- the 2026-10-20..27 site told the team to
    turn the whole book over every morning."""
    from umushroom.web import Site, _default_held, sell_body
    runner = Runner(tmp_path / "site", _fake(as_of="2026-10-05"))
    runner.start()
    runner.wait()
    first = runner.latest_report()
    import time
    time.sleep(1.1)
    runner.command = _fake(as_of="2026-10-06")
    runner.start()
    runner.wait()
    latest = runner.latest_report()
    site = Site(runner, _yahoo())
    assert _default_held(site, latest, site.payload(latest)) == first
    body = sell_body(site, latest, site.payload(latest), None, today=date(2026, 10, 7),
                     picker=False)
    assert "Holding period in progress" in body and "Sell / rebalance now" not in body
    assert "Trade list" not in body and "first run of this period" in body
    # choosing the same run in the local site's picker changes nothing
    body = sell_body(site, latest, site.payload(latest), first, today=date(2026, 10, 7))
    assert "Trade list" not in body


def test_sell_page_rejects_bad_held_param(site):
    runner, base = site
    _run_once(runner, base)
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/sell?held=../../x")
    assert e.value.code == 404


# ---- v4.16: macro tab + team mode -------------------------------------

def test_macro_tab_renders_with_market_data(tmp_path):
    from tests.test_macro import _closes
    runner = Runner(tmp_path / "site", _fake())
    runner.start()
    runner.wait()
    frame = _closes()
    frame["AAA.L"] = frame["AAA.L"]

    def fetch(symbols, start):
        return frame[[s for s in symbols if s in frame.columns]]
    server = make_server(runner, port=0, fetch=fetch)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        code, body = _get(f"http://127.0.0.1:{server.server_address[1]}/macro")
        assert code == 200
        for text in ("What the economy is doing", "Stress tests", "talking points",
                     "Currency mix", "Each holding", "US 10-year yield"):
            assert text in body
    finally:
        server.shutdown()
        server.server_close()


def test_macro_tab_survives_download_failure(tmp_path):
    runner = Runner(tmp_path / "site", _fake())
    runner.start()
    runner.wait()

    def broken(symbols, start):
        raise ConnectionError("offline")
    server = make_server(runner, port=0, fetch=broken)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        _, body = _get(f"http://127.0.0.1:{server.server_address[1]}/macro")
        assert "Could not get macro data: offline" in body
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def team(tmp_path):
    runner = Runner(tmp_path / "site", _fake())
    runner.start()
    runner.wait()
    server = make_server(runner, port=0, fetch=_yahoo(), password="s3cret-pw",
                         trust_local=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _auth(pw):
    import base64
    return {"Authorization": "Basic " + base64.b64encode(f"team:{pw}".encode()).decode()}


def _get_h(url, headers):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as r:
        return r.status, r.read().decode()


def test_team_mode_needs_the_password(team):
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(team + "/stocks")
    assert e.value.code == 401
    assert "Basic" in e.value.headers["WWW-Authenticate"]
    with pytest.raises(urllib.error.HTTPError) as e:
        _get_h(team + "/stocks", _auth("wrong"))
    assert e.value.code == 401
    code, body = _get_h(team + "/stocks", _auth("s3cret-pw"))
    assert code == 200 and "Today's orders" in body


def test_team_mode_protects_run_now_and_files(team):
    req = urllib.request.Request(team + "/api/run", method="POST", data=b"",
                                 headers={"X-UMushroom-Run": "1"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 401
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(team + "/api/status")
    assert e.value.code == 401


def test_team_mode_locks_out_password_guessing(team):
    for _ in range(10):
        with pytest.raises(urllib.error.HTTPError):
            _get_h(team + "/", _auth("guess"))
    with pytest.raises(urllib.error.HTTPError) as e:
        _get_h(team + "/", _auth("s3cret-pw"))           # even the right one waits
    assert e.value.code == 429


def test_never_opens_to_other_computers_without_a_password(tmp_path):
    runner = Runner(tmp_path / "site", _fake())
    with pytest.raises(ValueError):
        make_server(runner, host="0.0.0.0", port=0)


def test_team_password_is_generated_once_and_private(tmp_path, monkeypatch):
    import stat

    from umushroom.web import team_password
    monkeypatch.delenv("UMUSHROOM_PASSWORD", raising=False)
    pw = team_password(tmp_path)
    assert len(pw) >= 15 and team_password(tmp_path) == pw
    mode = (tmp_path / "team_password.txt").stat().st_mode
    assert not mode & (stat.S_IRGRP | stat.S_IROTH)
    monkeypatch.setenv("UMUSHROOM_PASSWORD", "from-env")
    assert team_password(tmp_path) == "from-env"
