"""The Website workflow itself (v4.28). Its shell steps hold the run history
-- on a rebalance day, which book the team holds -- so they are run here for
real, with the workflow's own `run:` blocks, in every branch that matters:
a save that does not read back is never kept, a missing password never
encrypts, and a state that cannot be read stops the build instead of being
overwritten. On CI this runs on the runner's own bash, tar and openssl."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "website.yml"
PASSWORD = "correct horse battery staple"
needs_shell = pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("bash", "openssl", "tar")),
    reason="needs bash, openssl and tar")


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _steps(job: str) -> list[dict]:
    return _workflow()["jobs"][job]["steps"]


def _step(job: str, name: str) -> dict:
    return next(s for s in _steps(job) if s.get("name") == name)


def test_the_workflow_is_wired_consistently():
    wf = _workflow()
    crons = [c["cron"] for c in wf[True]["schedule"]]        # YAML reads `on:` as True
    assert crons == [f"40 {h} * * 1-5" for h in (8, 11, 14, 17, 20)]
    build = wf["jobs"]["build"]
    # scheduled runs give the next slot its turn; only a manual validation run is long
    assert build["timeout-minutes"] == "${{ inputs.validate && 360 || 150 }}"
    ids = {s["id"] for s in build["steps"] if "id" in s}
    assert {"unpack", "site", "pack"} <= ids
    assert build["outputs"] == {"run_ok": "${{ steps.site.outputs.run_ok }}",
                                "state_ok": "${{ steps.pack.outputs.state_ok }}"}
    names = [s.get("name") for s in build["steps"]]
    # the state is unpacked before the run and packed, then saved, after it
    order = ["Decrypt the saved state", "Run the model and build the encrypted site",
             "Encrypt the state to save", "Save past runs and the price cache (encrypted)"]
    assert [n for n in names if n in order] == order
    pack = _step("build", "Encrypt the state to save")
    save = _step("build", "Save past runs and the price cache (encrypted)")
    # never pack or save over a state that could not be read
    assert "steps.unpack.outcome == 'success'" in pack["if"]
    assert "steps.unpack.outcome == 'success'" in save["if"]
    assert "hashFiles('state.enc') != ''" in save["if"]
    assert save["with"]["path"] == "state.enc"
    # a RE-RUN has the same run_id and caches are immutable: without the attempt
    # in the key its state would silently not be saved
    restore = _step("build", "Restore past runs and the price cache (encrypted)")
    key = "umushroom-state-${{ github.run_id }}-${{ github.run_attempt }}"
    assert save["with"]["key"] == restore["with"]["key"] == key
    assert restore["with"]["restore-keys"] == "umushroom-state-"
    # a missing state.enc is checked against the cache list BEFORE the one-time
    # migration (or an empty start) can be saved over an existing history
    guard = "Stop if a saved state exists but was not restored"
    migrate = "Restore pre-4.27 state (migration)"
    assert names.index(restore["name"]) < names.index(guard) < names.index(migrate)
    assert _step("build", guard)["if"] == _step("build", migrate)["if"] \
        == "hashFiles('state.enc') == ''"
    assert wf["permissions"]["actions"] == "read"
    # the run is stopped half an hour before the job's own limit, so the last
    # good site is still published and the state still saved
    site = _step("build", "Run the model and build the encrypted site")["run"]
    assert "--run-timeout ${{ inputs.validate && 330 || 120 }}" in site
    # a failed model run or an unsaved state turns the workflow red AFTER deploying
    deploy = [s.get("name") or s.get("uses") for s in _steps("deploy")]
    assert deploy == ["actions/deploy-pages@v4", "Report a failed model run",
                      "Report a state that could not be saved"]
    unsaved = _step("deploy", "Report a state that could not be saved")
    assert unsaved["if"] == "always() && needs.build.outputs.state_ok == 'false'"


def test_the_cloud_installs_the_pinned_numerical_libraries():
    install = _step("build", "Install")["run"]
    assert 'pip install -c constraints-site.txt -e ".[site]"' in install
    pins = {}
    for line in (ROOT / "constraints-site.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            m = re.fullmatch(r"([A-Za-z0-9_.-]+)==(\d+(?:\.\d+)+)", line)
            assert m, f"not an exact pin: {line!r}"
            pins[m.group(1).lower()] = m.group(2)
    assert {"numpy", "pandas", "scikit-learn", "scipy", "pyarrow"} <= set(pins)
    assert "yfinance" not in pins          # a Yahoo API change needs its next release
    # every pin satisfies the package's own declared minimum
    declared = dict(re.findall(r'"([A-Za-z0-9_-]+)>=([\d.]+)"',
                               (ROOT / "pyproject.toml").read_text()))

    def key(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in v.split("."))
    for name, version in pins.items():
        if name in declared:
            assert key(version) >= key(declared[name]), (name, version, declared[name])


def _run(script: str, cwd: Path, password: str) -> tuple[int, str, str]:
    """Run one workflow `run:` block the way the runner does (bash -e)."""
    out = cwd / "_github_output"
    env = dict(os.environ, UMUSHROOM_PASSWORD=password, GITHUB_OUTPUT=str(out))
    script_file = cwd / "_step.sh"
    script_file.write_text(script)
    r = subprocess.run(["bash", "-e", str(script_file)], cwd=cwd, env=env,
                       capture_output=True, text=True, check=False)
    written = out.read_text().strip() if out.exists() else ""
    out.unlink(missing_ok=True)
    script_file.unlink()
    return r.returncode, r.stdout + r.stderr, written


def _workspace(path: Path, with_state: bool = True) -> Path:
    path.mkdir()
    if with_state:
        (path / "site" / "reports").mkdir(parents=True)
        (path / ".cache").mkdir()
        (path / "site" / "reports" / "20261109-094013.json").write_text('{"status": "ok"}')
        (path / "site" / "reports" / "20261109-094013.html").write_text("<p>book</p>" * 4000)
        (path / "site" / "site_salt.txt").write_text("c2FsdHNhbHRzYWx0c2FsdA==\n")
        (path / ".cache" / "prediction_log.csv").write_text("run_date,ticker\n" * 3000)
    return path


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes()
            for d in ("site", ".cache") for p in sorted((root / d).rglob("*")) if p.is_file()}


@needs_shell
def test_the_saved_state_round_trips_and_is_never_overwritten_when_unreadable(tmp_path):
    unpack = _step("build", "Decrypt the saved state")["run"]
    pack = _step("build", "Encrypt the state to save")["run"]

    # the first run after the upgrade: plaintext folders, no state.enc yet
    first = _workspace(tmp_path / "first")
    rc, out, _ = _run(unpack, first, PASSWORD)
    assert rc == 0 and "::warning::" not in out
    rc, out, written = _run(pack, first, PASSWORD)
    assert rc == 0 and written == "state_ok=true", out
    archive = (first / "state.enc").read_bytes()
    assert b"book" not in archive and b"run_date" not in archive       # encrypted

    # the next run: a clean checkout plus the restored state.enc
    second = _workspace(tmp_path / "second", with_state=False)
    (second / "state.enc").write_bytes(archive)
    rc, out, _ = _run(unpack, second, PASSWORD)
    assert rc == 0, out
    assert _files(second) == _files(first)                             # byte for byte
    assert not (second / "state.enc").exists()

    # a changed password, or a truncated archive: the build STOPS (so the pack
    # and save steps, which need this step's success, never overwrite it)
    for name, blob, password in (("wrong", archive, "another password 123"),
                                 ("cut", archive[: len(archive) // 2], PASSWORD)):
        ws = _workspace(tmp_path / name, with_state=False)
        (ws / "state.enc").write_bytes(blob)
        rc, out, _ = _run(unpack, ws, password)
        assert rc != 0 and "Cannot decrypt the saved state" in out, (name, out)
        assert (ws / "state.enc").read_bytes() == blob

    # nothing restored at all: the site starts empty, and says so
    empty = _workspace(tmp_path / "empty", with_state=False)
    rc, out, _ = _run(unpack, empty, PASSWORD)
    assert rc == 0 and "the site starts empty" in out
    rc, out, written = _run(pack, empty, PASSWORD)
    assert rc == 0 and written == "" and not (empty / "state.enc").exists()


@needs_shell
def test_a_state_that_cannot_be_packed_is_not_saved_but_never_blocks_the_site(tmp_path):
    """Packing must end with exit 0 (the pages are uploaded by later steps) and
    with state_ok=false (the deploy job then turns the workflow red)."""
    pack = _step("build", "Encrypt the state to save")["run"]
    # a missing secret or a short password: never encrypt with it
    for i, password in enumerate(("", "short")):
        ws = _workspace(tmp_path / f"pw{i}")
        rc, out, written = _run(pack, ws, password)
        assert rc == 0 and written == "state_ok=false", out
        assert not (ws / "state.enc").exists() and "NOT saved" in out
    # tar fails part-way (an unreadable file): the broken archive is not kept
    ws = _workspace(tmp_path / "tar")
    bad = ws / "site" / "reports" / "unreadable.html"
    bad.write_text("x")
    bad.chmod(0)
    try:
        if os.access(bad, os.R_OK):                 # running as root: cannot simulate
            pytest.skip("the file is readable despite chmod 0 (root)")
        rc, out, written = _run(pack, ws, PASSWORD)
    finally:
        bad.chmod(0o644)
    assert rc == 0 and written == "state_ok=false", out
    assert not (ws / "state.enc").exists() and "Could not pack the state" in out


@needs_shell
@pytest.mark.parametrize("answer, code, expect", [
    ("echo 0", 0, "No encrypted state yet"),                 # first run after the upgrade
    ("echo 3", 1, "A saved state exists but could not be restored"),
    ("exit 1", 0, "::warning::Could not check"),             # the check itself failed: go on
])
def test_a_missing_state_is_checked_against_the_cache_list(tmp_path, answer, code, expect):
    """v4.28: state.enc missing because the cache SERVICE failed must not be
    taken for a first run (the pre-4.27 cache, or nothing, would be saved as the
    newest state). `gh` is replaced by a stand-in that answers the count."""
    guard = _step("build", "Stop if a saved state exists but was not restored")["run"]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(f"#!/bin/sh\n{answer}\n")
    gh.chmod(0o755)
    script = tmp_path / "guard.sh"
    script.write_text(guard)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
               GITHUB_REPOSITORY="owner/repo", GITHUB_REF="refs/heads/main", GH_TOKEN="x")
    r = subprocess.run(["bash", "-e", str(script)], cwd=tmp_path, env=env,
                       capture_output=True, text=True, check=False)
    assert r.returncode == code, r.stdout + r.stderr
    assert expect in r.stdout
