# Implementing the New Features & Running This in PyCharm

This document covers two things:
1. **What changed** — the purged walk-forward backtest and the morning
   prediction-accuracy check — and exactly where each piece lives.
2. **How to open and run this project in PyCharm**, step by step.

---

## Part 0a — v4.1/v4.2: universe from UMushroom + ETF and crypto sleeves

- **v4.1:** the stock universe is now built from UMushroom's own equity
  list (all 4,491 listed stocks are in `umushroom_universe_mapped.csv`):
  608 stocks = every London and Swiss stock UMushroom lists + 102 US large
  caps.
- **v4.2:** two extra **sleeves**, each with its own model and verdict:
  ETFs (20%, top 4 of 13) and crypto (10%, top 4 of 18). See the
  `sleeves:` section of `config.yaml` to change weights or holdings.
- Output is one combined order table with a `sleeve` column. Compare the
  `local_price` column with the price UMushroom shows: if one is ~100x
  off, that ticker's currency in `currency_overrides` is wrong.
- **Check before trading:** that UMushroom lets you buy fractions of a
  coin. If not, set `fractional: false` under `crypto` (expensive coins
  like BTC then can't be bought with a 10% sleeve).

## Part 0 — v4: what changed and why

### Why every prediction was a Swiss stock
`config.yaml` in v3 listed only the 20 SMI stocks (all `.SW`). The model
can only rank what it is given, so every pick was Swiss. v4 ships a
100-stock universe (40 FTSE 100, 40 US large caps, 20 SMI) and prints the
exchange mix on every run, with a WARNING if any one exchange exceeds
`max_single_exchange_fraction` (60%). **Check every ticker is tradable on
UMushroom and delete any that are not.**

### Changes aimed at real alpha
| Problem in v3 | Fix in v4 | File |
|---|---|---|
| Swiss-only universe | Multi-market universe + concentration warning | `config.yaml`, `universe.py` |
| No currency handling (London prices are in **pence**: 100x sizing error) | All prices converted to `base_currency` before anything else | `universe.py`, `data.py` |
| One late-listed ticker truncated the whole history | Low-coverage tickers dropped instead | `data.py` |
| Partial intraday bar could be used while UMushroom fills at yesterday's close | Today's bar dropped (`drop_incomplete_bar`) | `data.py` |
| 7 overlapping short-term factors | 9 published factors: 12-1 & 6-1 momentum, residual momentum, 52-week high, 21d & 5d reversal, low-vol, low-beta, RSI | `features.py` |
| Raw factor values (RSI ~50, vol ~0.01) fed to Ridge | Per-date centred percentile ranks, within each exchange | `features.py` |
| Target = raw return (mostly market noise) | Target = cross-sectional rank of forward return | `features.py` |
| CV split rows (could split one day) and scored RMSE | Date-based purged CV scoring IC + t-stat | `model.py` |
| Backtest reported returns only | IC t-stat, hit rate, active return t-stat, long-short spread, VERDICT | `purged_backtest.py` |
| Off-by-one: last test period could index past the data | Guard fixed | `purged_backtest.py` |
| No alpha number | Alpha vs universe (always) + Fama-French (`--fama-french`), Newey-West t-stats | `factor_regression.py` |
| HRP needed PyPortfolioOpt (won't install on Python 3.14) | Native scipy HRP; dependency removed | `portfolio_construction.py` |

### Verification
161 tests, all offline, verified on Python 3.13 and 3.14 with
PyPortfolioOpt absent. The key new ones are in `tests/test_alpha_recovery.py`:
the full pipeline finds a planted signal (IC t ≈ 5) and reports no skill on
pure noise (t ≈ 0.7). If those ever flip, there is a leakage bug.

### Install v4 (clean replace — don't merge into an old folder)
Delete your old `umushroom_pkg` folders (including `umushroom_pkg 2`/`3`),
unzip v4, then in a terminal opened in the new folder run one line at a time:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest
umushroom --config config.yaml --weighting hrp --fama-french
```

The first run downloads 5 years for 100 tickers (about 1 minute); later runs
use the cache.

---

## Part 1 — What changed in v2/v3 (history)

### 1a. Purged & embargoed walk-forward validation

**New file:** `umushroom/purged_backtest.py`

**What it does differently from the original `backtest.py`:**

| | `backtest.py` (original) | `purged_backtest.py` (new) |
|---|---|---|
| Model | Trained **once**, evaluated across all history | **Retrained at every fold**, using only data strictly before that fold |
| Leakage guard | None | **Purge gap** between training data and test window (≥ `forward_return_days`), so no training label's forward-return window overlaps the test period |
| Extra safety | None | **Embargo gap** after each test window before that data re-enters training, as a conservative buffer against serial correlation |
| Signal quality metric | Win rate, Sharpe | Adds **Information Coefficient (IC)** — Spearman rank correlation between predicted scores and realized returns per fold. This is the standard quant-research way to check whether a signal has any real skill, independent of whether one particular run happened to be profitable |

**Where it's wired in:** `umushroom/cli.py`, in `main()`, right after the
existing simple backtest block:

```python
if not args.skip_purged_backtest:
    try:
        pwf_result = run_purged_walk_forward(
            prices,
            test_window_days=cfg.purged_test_window_days,
            purge_days=cfg.purged_purge_days,
            embargo_days=cfg.purged_embargo_days,
            rebalance_days=cfg.rebalance_days,
            forward_return_days=cfg.forward_return_days,
            top_n=cfg.top_n_holdings,
            min_train_days=cfg.purged_min_train_days,
            model_alpha=cfg.model_alpha,
            random_seed=cfg.random_seed,
        )
        print("\n--- Purged & Embargoed Walk-Forward Backtest ---")
        print(pwf_result.summary())
    except InsufficientHistoryError as e:
        print(f"Skipped: {e}")
```

**New config fields** (in `config.yaml` and validated in `umushroom/config.py`):

```yaml
purged_test_window_days: 20     # length of each held-out test fold
purged_purge_days: 5            # gap before test fold; must be >= forward_return_days
purged_embargo_days: 5          # extra gap after test fold before next training
purged_min_train_days: 120      # minimum training history before the first fold
```

`config.py`'s `_validate()` function enforces `purged_purge_days >=
forward_return_days` at load time — if you set it too low, the program
refuses to start rather than silently running a leaky backtest.

**To disable it for a faster run:** `umushroom --skip-purged-backtest`

**Tests:** `tests/test_purged_backtest.py` — including a test that
explicitly checks each fold's training data ends at least `purge_days`
before its test window starts (`test_purged_train_end_precedes_test_start_by_at_least_purge_gap`),
and one proving the walk only ever moves forward in time with no overlapping
test windows.

---

### 1b. Morning prediction-accuracy check

**New file:** `umushroom/prediction_log.py`

**What it does:** every run logs today's full prediction table (ticker,
predicted score, price at prediction time, and whether it was a top-N
pick) to a CSV. The **next** run reads that log, finds the most recent
*prior* entry, and compares it against *today's* prices — before
generating any new predictions. It reports:

- **Directional accuracy** — did the predicted sign (up/down) match what
  actually happened
- **Information Coefficient** — rank correlation between what was
  predicted and what was realized, across the whole universe
- **Top-pick vs. rest-of-universe realized return** — the practical
  question: did the stocks you actually bought do better than the ones
  you didn't

**Where it's wired in:** `umushroom/cli.py`, in two places:

1. Right after data is fetched, **before** training a new model:
```python
if not args.skip_prediction_check:
    current_date = prices.index[-1]
    current_prices = prices.iloc[-1]
    report = evaluate_latest_predictions(
        cfg.prediction_log_path, current_prices, current_date,
    )
    if report is not None:
        print("\n--- Previous Run's Prediction Accuracy ---")
        print(report.summary())
```

2. Right after today's allocation is generated, to log this run's
   predictions for tomorrow to evaluate:
```python
log_predictions(
    log_path=cfg.prediction_log_path,
    run_date=prices.index[-1],
    all_scores=allocation.all_scores,
    prices=prices.iloc[-1],
    picked_tickers=set(allocation.orders.index),
)
```

**Changed file:** `umushroom/allocation.py` — the `Allocation` dataclass
gained a new field, `all_scores`, holding the predicted score for
*every* ticker with clean features that day (not just the top-N picks).
This is needed because scoring only the picks would bias the accuracy
metrics toward whatever the model already favored — a fair IC needs the
full universe.

**New config field:**
```yaml
prediction_log_path: ".cache/prediction_log.csv"
```

**To disable it:** `umushroom --skip-prediction-check`

**Tests:** `tests/test_prediction_log.py` — covers the first-run case
(nothing to evaluate), a same-day re-run not evaluating itself, log
idempotency, and a direct check that top-pick vs. rest-of-universe
splitting is computed correctly.

---

### 1c. HRP portfolio weighting

**New file:** `umushroom/portfolio_construction.py`

**What it does:** replaces flat 1/N equal-weighting with Hierarchical
Risk Parity (Lopez de Prado), implemented natively on scipy (v4; v3 used
PyPortfolioOpt, which does not install on Python 3.14). HRP
clusters your top-N picks by correlation structure and allocates more
weight to less-correlated, lower-volatility clusters — without needing
you to supply expected-return forecasts (unlike Markowitz/Black-
Litterman, which are notoriously sensitive to small errors in those
forecasts). Since the model's predicted scores are a reasonable
*ranking* signal but not a precise *expected-return* number, HRP is the
more honest match for what the model actually gives you.

**Where it's wired in:** `umushroom/allocation.py`'s `generate_allocation()`
now takes a `weighting: Literal["equal", "hrp"]` parameter (default
`"equal"`, so nothing changes unless you opt in). `umushroom/cli.py` exposes
it as `--weighting hrp`. If HRP fails for any reason (e.g. too little
history), it logs a warning and falls back to equal-weight rather than
crashing the whole run.

**To use it:** `umushroom --config config.yaml --weighting hrp`

**Tests:** `tests/test_portfolio_construction.py` — covers weights
summing to 1, no negative weights, no-lookahead (same discipline as
`features.py`), and rejection of single-ticker/insufficient-history inputs.

---

### 1d. Bootstrap Sharpe confidence interval

**New file:** `umushroom/robustness.py`

**What it does:** a single Sharpe ratio point estimate from a handful
of backtest periods has wide sampling uncertainty — this runs a
circular block bootstrap (resampling contiguous *blocks* of returns,
not individual returns, to preserve autocorrelation) to produce a 95%
confidence interval around the Sharpe ratio from the purged walk-forward
backtest. If the interval straddles zero, the report says so explicitly
— that's the case where you cannot rule out "no real edge."

**Where it's wired in:** `umushroom/cli.py`, immediately after the
purged walk-forward backtest prints, using its new `period_returns` field
(see below):
```python
ci = bootstrap_sharpe(
    pwf_result.period_returns,
    periods_per_year=252 / cfg.rebalance_days,
)
print(ci.summary())
```

**Changed file:** `umushroom/purged_backtest.py` — `PurgedWalkForwardResult`
gained a `period_returns: pd.Series` field: every individual rebalance
-period strategy return across all folds, date-indexed. This is the raw
series the bootstrap resamples from (the existing `overall_sharpe` field
is just the point estimate computed from fold-level aggregates).

**Tests:** `tests/test_robustness.py` — covers CI ordering, a clearly
positive-drift series giving a positive point estimate, a zero-drift
series giving a CI that straddles zero, reproducibility with a fixed
seed, and input validation (too few observations, bad block size, bad
confidence level).

---

### 1e. Important: the simple backtest is in-sample, not a real estimate

While reviewing this, I found that the original (non-purged) `backtest.py`
has a real leakage bug: it trains the model on your **entire** price
history and then evaluates it across that same history. The model has
already seen the return patterns it's being "tested" against — those
numbers are inflated and should never be used to judge real performance.

This is now labeled loudly rather than fixed-by-removal, since it's
still useful as a fast sanity check ("does the pipeline run end to end,
do picks look directionally sane") — but `BacktestResult.caveat` and the
CLI's section header now say explicitly **IN-SAMPLE, NOT A REAL BACKTEST**.
**Always use the purged walk-forward numbers (and the bootstrap CI
around them) as your actual performance claim** — never the simple
backtest's Sharpe/win-rate.

---

### 1f. Summary of every file touched

```
umushroom/purged_backtest.py          NEW — purged walk-forward engine
umushroom/prediction_log.py           NEW — logging + next-run accuracy evaluation
umushroom/portfolio_construction.py   NEW — equal-weight and HRP weighting
umushroom/robustness.py               NEW — bootstrap Sharpe confidence interval
umushroom/allocation.py               CHANGED — `all_scores` + `weighting` param
umushroom/config.py                   CHANGED — 5 new validated config fields
umushroom/backtest.py                 CHANGED — caveat now warns IN-SAMPLE explicitly
umushroom/cli.py                      CHANGED — wires everything into main(),
                                                 --weighting flag, 3 skip flags
config.yaml                           CHANGED — new config sections with defaults
requirements.txt / pyproject.toml     CHANGED — added scipy (PyPortfolioOpt removed in v4)
tests/test_purged_backtest.py         NEW (+ 1 test for period_returns)
tests/test_prediction_log.py          NEW
tests/test_portfolio_construction.py  NEW
tests/test_robustness.py              NEW
tests/test_allocation.py              CHANGED — covers all_scores
tests/test_config.py                  CHANGED — covers new fields' validation
```

(v3 summary. For v4 changes see Part 0.)

---

## Part 2 — Opening and running this in PyCharm

### Step 1: Unzip and open the project

1. Unzip the delivered archive somewhere, e.g. `~/umushroom_pkg/`.
2. In PyCharm: **File → Open…** → select the `umushroom_pkg` folder →
   **Open**. Say **"This Window"** if asked, not "New Window", so it
   opens as its own project root.

### Step 2: Create the virtual environment

PyCharm can do this for you automatically:

1. **File → Settings** (PyCharm on Windows/Linux) or **PyCharm → Settings**
   (macOS) → **Project: umushroom_pkg → Python Interpreter**.
2. Click the gear icon → **Add Interpreter → Add Local Interpreter**.
3. Choose **Virtualenv Environment → New**, location defaults to
   `umushroom_pkg/.venv` — leave it, click **OK**.
4. PyCharm creates the venv and switches the project to use it.

### Step 3: Install dependencies

Open PyCharm's terminal (**View → Tool Windows → Terminal**, or
`Alt+F12`) — it opens already activated inside the venv you just made:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

The `-e .` installs the package itself in "editable" mode, so
`import umushroom` works from anywhere in the project, including test
files, without path hacks.

### Step 4: Mark `tests` and `umushroom` as source roots (usually automatic)

PyCharm normally detects this from `pyproject.toml`, but if imports show
red/unresolved:
1. Right-click the `umushroom_pkg` folder (project root) in the project
   tree → **Mark Directory as → Sources Root**.
2. Right-click `tests/` → it should already show as a normal folder;
   pytest discovers it via `pyproject.toml`'s `[tool.pytest.ini_options]`
   `testpaths = ["tests"]`, no manual marking needed.

### Step 5: Set up pytest as the test runner

1. **Settings → Tools → Python Integrated Tools**.
2. Under **Testing → Default test runner**, choose **pytest**.
3. Click **OK**.

Now right-clicking any file under `tests/`, or the `tests/` folder
itself, shows **Run 'pytest in tests'** — do that to run the whole suite
with a green/red result tree, and coverage will print in the Run panel
(coverage settings come from `pyproject.toml`'s `addopts`).

### Step 6: Create a Run Configuration for the CLI

This is what lets you click a single green ▶ button to run the whole
pipeline, instead of typing the command each time.

1. **Run → Edit Configurations… → + → Python**.
2. Fill in:
   - **Name:** `Run UMushroom Pipeline`
   - **Module name** (not "Script path"): `umushroom.cli`
   - **Parameters:** `--config config.yaml --output umushroom_allocation.csv`
   - **Working directory:** the project root (`umushroom_pkg`) — this
     matters, since `config.yaml` and the `.cache/` folder are resolved
     relative to wherever you run from.
   - **Python interpreter:** should already show the venv from Step 2.
3. Click **OK**.

Now the dropdown next to the green ▶ in the top toolbar shows **"Run
UMushroom Pipeline"** — click ▶ to run it, or the bug icon to debug it
with breakpoints.

**To add the morning routine as its own one-click config:** duplicate
that configuration (right-click it → **Copy Configuration**), rename to
`Morning Check`, and change Parameters to add `--skip-purged-backtest`
if you want the faster daily check without the heavier purged backtest
re-running every single morning:
```
--config config.yaml --skip-purged-backtest --output umushroom_allocation.csv
```

### Step 7: Running it day-to-day

- **Each evening after market close:** Run → "Run UMushroom Pipeline".
  This fetches fresh data (or uses the cache if run twice same day),
  trains, runs both backtests, prints today's allocation, and logs
  today's predictions.
- **Each morning (or next time you run it):** the *same* run
  automatically prints the "Previous Run's Prediction Accuracy" section
  first, scoring yesterday's predictions against today's prices — you
  don't need a separate command for this, it's built into every run
  after the first.

### Step 8 (optional): Enable ruff/mypy inline in the editor

PyCharm doesn't bundle ruff/mypy by default, but both work well via
**File Watchers** or the **Ruff** / **Mypy** plugins:

1. **Settings → Plugins → Marketplace**, search **"Ruff"**, install, restart.
2. **Settings → Tools → Ruff**, point it at the venv's `ruff` (usually
   auto-detected once installed via `requirements-dev.txt`), enable
   "Run ruff on save" if you want inline squiggles as you type.

This is optional — the CI workflow (`.github/workflows/ci.yml`) already
runs ruff + mypy + pytest automatically on every push if you push this
to GitHub, so the inline plugin is just a convenience, not a requirement.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'umushroom'` | Run `pip install -e .` again inside the PyCharm terminal (make sure the terminal shows the venv prompt, e.g. `(.venv)`) |
| Tests not discovered | Settings → Tools → Python Integrated Tools → Default test runner → **pytest** (Step 5) |
| `config.yaml` not found when running via the green ▶ | Check the Run Configuration's **Working directory** is the project root, not a subfolder |
| yfinance network errors in the sandbox/CI but works locally | Expected — the test suite never calls yfinance (synthetic fixtures only); only the live CLI run needs real network access |
| `ModuleNotFoundError: No module named 'pypfopt'` | You are running v3 code. Delete the old folder and do a clean v4 unzip (Part 0) |
| `WARNING: ... of the universe is listed on SIX` | You are using a Swiss-only ticker list. Use the v4 `config.yaml` |
| `Dropping N ticker(s) with <90% price coverage` | Recent listings or renamed tickers. Harmless; remove them from `config.yaml` to silence it |
| `zsh: command not found: #` | Paste one command per line, without trailing `# comments` |
