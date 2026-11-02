"""Static, encrypted site for GitHub Pages."""
from __future__ import annotations

import base64
import json
import re
import sys

import pytest

from umushroom.static_site import (
    MIN_PASSWORD,
    build,
    decrypt,
    derive_key,
    lock,
    site_salt,
)
from umushroom.web import Runner

PW = "team-pass-123456"
FAST = 1000          # PBKDF2 iterations for tests only (real builds use 600,000)


def _payload_of(locked: str) -> dict:
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', locked)
    return json.loads(m.group(1))


def _open(locked: str, password: str) -> str:
    d = _payload_of(locked)
    key = derive_key(password, base64.b64decode(d["s"]), d["n"])
    return decrypt(base64.b64decode(d["i"]), base64.b64decode(d["c"]), key)


def test_lock_round_trip_and_wrong_password_fails():
    salt = b"0123456789abcdef"
    page = "<html><body><h1>Secret picks: AAA.L</h1></body></html>"
    locked = lock(page, derive_key(PW, salt, FAST), salt, FAST)
    assert "AAA.L" not in locked and "Secret" not in locked
    assert _open(locked, PW) == page
    with pytest.raises(Exception):  # noqa: B017 -- cryptography's InvalidTag
        _open(locked, "wrong-password-xx")


def test_each_page_gets_a_fresh_iv():
    salt = b"0123456789abcdef"
    key = derive_key(PW, salt, FAST)
    a, b = lock("same", key, salt, FAST), lock("same", key, salt, FAST)
    assert _payload_of(a)["i"] != _payload_of(b)["i"]


def test_salt_is_kept_between_builds(tmp_path):
    assert site_salt(tmp_path) == site_salt(tmp_path)
    assert len(site_salt(tmp_path)) == 16


def _fake_cmd(tmp_path, ok=True):
    from tests.test_web import _report_html
    src = tmp_path / "src.html"
    src.write_text(_report_html())

    def make(report, orders):
        code = (f"import shutil;shutil.copy({str(src)!r},{str(report)!r})" if ok
                else "raise SystemExit(4)")
        return [sys.executable, "-c", code]
    return make


def _yahoo(symbols, start):
    import pandas as pd
    days = pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05"])
    return pd.DataFrame({s: [100.0, 100.0, 110.0] for s in symbols}, index=days)


def test_build_runs_and_writes_every_page_encrypted(tmp_path):
    out = tmp_path / "public"
    rc = build(tmp_path / "site", out, PW, run_url="https://github.com/x/y/actions",
               fetch=_yahoo, command=_fake_cmd(tmp_path), run_first=True,
               run_timeout=60, iterations=FAST)
    assert rc == 0
    names = {p.name for p in out.iterdir()}
    for f in ("index.html", "stocks.html", "etfs.html", "sell.html", "tracker.html",
              "graphs.html", "montecarlo.html", "risk.html", "macro.html", "checks.html",
              "history.html", ".nojekyll", "robots.txt"):
        assert f in names
    assert len(list((out / "reports").iterdir())) == 1
    for f in out.rglob("*.html"):
        raw = f.read_text()
        assert "AAA.L" not in raw and "Alpha Co" not in raw   # nothing readable
    home = _open((out / "index.html").read_text(), PW)
    assert "Invested today" in home and 'href="#stocks"' in home   # v4.22: one page, panes
    assert ">Run now</a>" in home and "github.com/x/y" in home and "/api/" not in home
    assert 'data-key="stocks"' in home and "AAA.L" in home          # the Stocks pane
    assert "+10.00%" in home                                        # the Tracker pane
    assert 'class="pal"' in home and "Find a ticker" in home       # command palette
    # the old per-section files are contentless redirects into the dashboard
    stub = (out / "stocks.html").read_text()
    assert "url=index.html#stocks" in stub and "AAA.L" not in stub
    rep = next((out / "reports").iterdir())
    assert 'href="../index.html"' in _open(rep.read_text(), PW)


def test_failed_run_still_publishes_last_good_with_a_warning(tmp_path):
    site = tmp_path / "site"
    out = tmp_path / "public"
    assert build(site, out, PW, fetch=_yahoo, command=_fake_cmd(tmp_path),
                 run_first=True, run_timeout=60, iterations=FAST) == 0
    import time
    time.sleep(1.1)
    rc = build(site, out, PW, fetch=_yahoo, command=_fake_cmd(tmp_path, ok=False),
               run_first=True, run_timeout=60, iterations=FAST)
    assert rc == 1
    home = _open((out / "index.html").read_text(), PW)
    assert "The latest run failed" in home and "Invested today" in home


def test_short_password_is_refused_before_running(tmp_path):
    with pytest.raises(ValueError):
        build(tmp_path / "site", tmp_path / "public", "x" * (MIN_PASSWORD - 1),
              command=_fake_cmd(tmp_path), run_first=True)
    assert not (tmp_path / "site" / "reports").exists() or not list(
        (tmp_path / "site" / "reports").glob("*.json"))


def test_old_runs_are_pruned(tmp_path):
    from umushroom.static_site import prune
    r = Runner(tmp_path / "site", _fake_cmd(tmp_path))
    for i in range(5):
        (r.reports / f"2026010{i}-000000.json").write_text('{"status": "succeeded"}')
        (r.reports / f"2026010{i}-000000.html").write_text("x")
    prune(r, keep=2)
    assert len(list(r.reports.glob("*.json"))) == 2
    assert (r.reports / "20260104-000000.json").exists()


def test_main_needs_a_password(monkeypatch, capsys):
    from umushroom.static_site import main
    monkeypatch.delenv("UMUSHROOM_PASSWORD", raising=False)
    assert main(["--out", "x"]) == 2
    assert "UMUSHROOM_PASSWORD" in capsys.readouterr().out


def test_successful_run_log_is_never_printed(tmp_path, capsys):
    """CI logs of a public repo are public: the picks must not appear there."""
    from tests.test_web import _report_html
    src = tmp_path / "src.html"
    src.write_text(_report_html())

    def chatty(report, orders):
        code = (f"print('BUY 100 AAA.L SECRET-PICK');import shutil;"
                f"shutil.copy({str(src)!r},{str(report)!r})")
        return [sys.executable, "-c", code]
    build(tmp_path / "site", tmp_path / "public", PW, fetch=_yahoo, command=chatty,
          run_first=True, run_timeout=60, iterations=FAST)
    assert "SECRET-PICK" not in capsys.readouterr().out



def test_not_ready_run_returns_3_and_keeps_the_last_good_site(tmp_path):
    """v4.21: exit code 3 from the pipeline (Yahoo has not published yesterday's
    closes) is not a failure: the site keeps the last good run and the
    workflow retries later."""
    from umushroom.static_site import NOT_READY_CODE
    site, out = tmp_path / "site", tmp_path / "public"
    assert build(site, out, PW, fetch=_yahoo, command=_fake_cmd(tmp_path),
                 run_first=True, run_timeout=60, iterations=FAST) == 0
    import time
    time.sleep(1.1)

    def not_ready(report, orders):
        return [sys.executable, "-c", "raise SystemExit(3)"]
    rc = build(site, out, PW, fetch=_yahoo, command=not_ready, run_first=True,
               run_timeout=60, iterations=FAST)
    assert rc == NOT_READY_CODE == 3
    assert "last good" in _open((out / "index.html").read_text(), PW)


# ------------------------------------------------- v4.21: the Tracker's book

def _cmd_with_dates(tmp_path, dates):
    """A fake pipeline whose report's as_of advances on every run."""
    from tests.test_web import _report_html
    it = iter(dates)

    def make(report, orders):
        src = tmp_path / f"src_{report.stem}.html"
        src.write_text(_report_html(next(it)))
        return [sys.executable, "-c",
                f"import shutil;shutil.copy({str(src)!r},{str(report)!r})"]
    return make


def _yahoo_weeks(symbols, start):
    import pandas as pd
    days = pd.to_datetime(["2026-10-02", "2026-10-09", "2026-10-12"])
    frame = pd.DataFrame({"AAA.L": [1000.0, 1100.0, 1210.0],
                          "IWDA.L": [100.0, 102.0, 103.0]}, index=days)
    return frame[[s for s in symbols if s in frame.columns]]


def _build_twice(tmp_path, dates, fetch):
    import time
    site, out = tmp_path / "site", tmp_path / "public"
    cmd = _cmd_with_dates(tmp_path, dates)
    for i in range(2):
        if i:
            time.sleep(1.1)                               # a distinct run stamp
        assert build(site, out, PW, fetch=fetch, command=cmd, run_first=True,
                     run_timeout=60, iterations=FAST) == 0
    return _open((out / "index.html").read_text(), PW)


def test_tracker_scores_the_new_book_and_the_week_by_week_record(tmp_path):
    # Fri 2 Oct signals -> week of 5 Oct; Fri 9 Oct signals -> a new week
    tracker = _build_twice(tmp_path, ["2026-10-02", "2026-10-09"], _yahoo_weeks)
    assert "close on 2026-10-09" in tracker and "+10.00%" in tracker   # 1100 -> 1210
    assert "Week by week" in tracker
    # the first week's book: 2 Oct -> 9 Oct +10%, MSCI World +2%, so +8% active
    assert "+2.00%" in tracker and "+8.00%" in tracker and "1 of 1" in tracker
    assert "so far, to 2026-10-12" in tracker


def test_tracker_mid_week_scores_the_book_you_hold_not_this_mornings_run(tmp_path):
    # Mon 5 Oct signals trade on Tue 6 Oct: the same week as the 2 Oct book
    tracker = _build_twice(tmp_path, ["2026-10-02", "2026-10-05"], _yahoo_weeks)
    assert "Your book is the orders of" in tracker and "close on 2026-10-02" in tracker
    assert "+21.00%" in tracker                                   # 1000 -> 1210


def test_tracker_says_no_close_yet_instead_of_zero_up(tmp_path):
    """The live site read 'Up / down 0 / 21' every morning: the latest close
    was the prediction close itself."""
    import pandas as pd

    def flat(symbols, start):
        days = pd.to_datetime(["2026-10-01", "2026-10-02"])
        return pd.DataFrame({s: [990.0, 1000.0] for s in symbols}, index=days)
    out = tmp_path / "public"
    assert build(tmp_path / "site", out, PW, fetch=flat, command=_fake_cmd(tmp_path),
                 run_first=True, run_timeout=60, iterations=FAST) == 0
    tracker = _open((out / "index.html").read_text(), PW)
    assert "No close newer than 2026-10-02 yet" in tracker and "no close yet" in tracker
    assert "0 / 1" not in tracker and ">Up / down</div>" not in tracker   # no tile
