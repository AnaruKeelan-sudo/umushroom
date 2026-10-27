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
    assert "Invested today" in home and 'href="stocks.html"' in home
    assert ">Run now</a>" in home and "github.com/x/y" in home and "/api/" not in home
    stocks = _open((out / "stocks.html").read_text(), PW)
    assert "AAA.L" in stocks
    tracker = _open((out / "tracker.html").read_text(), PW)
    assert "+10.00%" in tracker
    rep = next((out / "reports").iterdir())
    assert 'href="../stocks.html"' in _open(rep.read_text(), PW)


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
