# UMushroom Strategy Engine

A factor-model portfolio construction system for the UMushroom investment
contest. Built as a small, testable package rather than a notebook script:
typed config, retried/validated data fetching, a leakage-safe ML model,
a walk-forward backtest, and a balance-constrained allocation generator —
all covered by an offline, network-free test suite.

## Why it's structured this way

UMushroom prices trades at the **previous day's close** (no intraday
execution, no fees, no dividends — see platform onboarding docs). That means:
- There's no edge in intraday/live data — this system is designed to run
  **once per day**, after market close.
- The one thing that actually matters for a contest is **not silently doing
  the wrong thing with real(ish) money** — hence the emphasis below on
  validation, typed errors, and tests over the pipeline's logic rather than
  just "does it run."

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

## Configure

Edit `config.yaml`: your ticker universe, starting balance, portfolio size,
and model/backtest parameters. Every field is validated at load time —
e.g. `top_n_holdings` exceeding your ticker count raises immediately with
a specific `ConfigError`, instead of failing confusingly three steps later.

Any field can be overridden via environment variable without touching the
file, e.g.:
```bash
UMUSHROOM_STARTING_BALANCE=50000 umushroom --config config.yaml
```

## Run

```bash
umushroom --config config.yaml --weighting hrp --output umushroom_allocation.csv
```

This will:
1. Fetch (or reuse a cached) 5-year daily history for a **multi-market
   universe** (FTSE 100 + US large caps + SMI by default), drop today's
   still-trading bar, and convert every price into your account currency
   (London pence -> pounds, USD/CHF -> GBP)
2. Score yesterday's logged predictions against today's prices
3. Build 9 published factors (12-1 momentum, residual momentum, 52-week
   high, 1-month and 1-week reversal, low volatility, low beta, RSI),
   **ranked within each date and within each exchange** -- no lookahead,
   tested explicitly
4. Train a Ridge model on the **cross-sectional rank of forward returns**,
   with date-based, purged cross-validation that reports IC
5. Run the purged & embargoed walk-forward backtest and print the
   skill statistics: IC, IC t-stat, hit rate, active return, top-minus-
   bottom quintile spread, and a plain VERDICT line
6. Bootstrap a Sharpe confidence interval and regress out-of-sample
   returns on the equal-weight universe for a **headline alpha + t-stat**
   (add `--fama-french` for a Fama-French 3-factor alpha too)
7. Print and save today's orders: ticker, listing currency, local price,
   base-currency price, share count -- ready to key into UMushroom

Useful flags:
- `--weighting hrp` -- Hierarchical Risk Parity weights (native scipy)
- `--fama-french` -- also download Ken French factors and report FF alpha
- `--force-refresh` -- bypass the data cache
- `--skip-backtest` / `--skip-purged-backtest` -- faster daily run
- `--model-out path` -- persist the fitted model (joblib + JSON metadata)

### Sleeves (v4.2)

The balance is split into **sleeves**, each with its own model, backtest
and verdict, configured under `sleeves:` in `config.yaml`:

| Sleeve | Weight | Picks | Universe |
|---|---|---|---|
| equity | 80% | top 10, max 3 per sector | 587 UMushroom-listed London, Swiss and US stocks; ranked within exchange, 5-day horizon, 9 price factors |
| etf | 20% | `auto`: top 8 (max 3 per asset class) when the evidence holds, else the 13 core ETFs equal weight | every ETF UMushroom lists that Yahoo can price (2,047 on 2026-10-12; leveraged/inverse and crypto ETPs excluded), ranked within asset class on 13 trend factors, 10-day horizon |
| crypto | off | -- | 18 large coins (kept in config, commented out) |

Assets are only ever ranked against their own sleeve, so crypto's much
higher volatility never distorts stock rankings. A sleeve whose VERDICT
shows no skill can be cut by lowering its weight.

### The ETF sleeve (v4.19)

The ETF sleeve is the stock model's twin for ETFs:

- **Universe.** `umushroom_etf_catalogue.json` is UMushroom's own ETF
  catalogue (2,261 ETFs: its public ETF sitemap plus three core ETFs the
  sitemap omits), scraped once by `python -m umushroom.catalogue`.
  `umushroom_etf_universe.csv` maps every ISIN to a Yahoo Finance line and
  records WHY each one is or isn't tradeable (`status`, `via`, name match,
  history length). Rebuild it with `python -m umushroom.etf_universe
  --config config.yaml`, or from GitHub: Actions -> ETF universe.
- **Model.** `feature_set: etf` -- 1/3/6/12-month time-series momentum,
  200-day trend, 50/200 cross, 52-week high, 3-month drawdown, 1-week
  reversal, volatility, volatility regime, beta, RSI -- ranked within
  UMushroom's asset classes (`group_by: asset_class`) so bond ETFs compete
  with bond ETFs. Leveraged/inverse ETPs and crypto ETPs are excluded.
- **Contest window.** `contest_window_days: 30` (Stage 1 = 12 Oct - 22 Nov).
  No sleeve may target a longer horizon, and every sleeve is also judged on
  rolling 30-day windows out of sample (`window.py`): did the picks beat
  holding everything over stretches as long as the contest?
- **`selection: auto`.** The model's picks are traded only while BOTH the
  walk-forward IC is significant (t >= 2) AND the contest-window verdict
  is not FAIL; otherwise the sleeve holds the 13 core ETFs equal weight.
  The website's **ETFs** tab shows which applied, and why, on every run.
- **Horizon.** `--compare-horizon etf:5,10,21` is the one pre-declared
  head-to-head (highest contest-window active t among horizons with IC
  t >= 2; ties to the shorter). On 2026-10-12 it chose 10 days.

Offline, reproducible runs: `UMUSHROOM_PRICE_SNAPSHOT=prices.json.gz`
replays a saved price snapshot instead of calling Yahoo (`data.py`).

### v4.20: data guard, near-duplicate guard, factor-risk weights, learners

The 2026-10-19 run exposed four things; each has a fix and, where it would
change the strategy, a pre-declared head-to-head:

- **Data guard.** Yahoo rate-limits big downloads *per symbol* and returns
  the rest empty; those were cached for 12 hours, so the ETF sleeve traded
  1,039 of ~1,700 ETFs with 5 of 13 core ETFs. `data.py` now re-fetches empty
  symbols in small batches, repairs a damaged cache on the next run, and
  refuses to run (or cache) when more than `max_missing_fraction` (10%) of
  the universe is missing. A refused run is the correct outcome: the website
  keeps the last good run and GitHub emails you.
- **Near-duplicate guard** (`max_pick_correlation`). Two MSCI Korea trackers
  and three USD high-yield funds filled five of the eight ETF slots. The
  guard skips a candidate whose trailing-year return correlation with a name
  already picked exceeds the threshold (causal; same rule in the backtest
  and live). `--compare-dedup etf:0.9,0.95` decides whether to switch it on.
- **Factor-risk weights** (`weighting: factor`, `factor_risk.py`). Factors =
  the sleeve's group averages (asset classes / exchanges); each pick's betas
  and the factors' covariance (their correlation) set weights so that every
  factor contributes its budgeted share of risk (equal by default). The
  walk-forward now scores the configured weighting scheme -- HRP included,
  which it never was before. Weights sit between 0.25x and 2x an equal
  share; groups that barely move (money-market ETFs) are not used as
  factors; a held-all basket stays equal weight.
  `--compare-weighting etf:equal,hrp,factor`.
- **Learners** (`model_type: ridge | mlp | gbm`). A one-hidden-layer neural
  network (date-purged early stopping) and gradient boosting behind the same
  interface, compared by `--compare-model etf:ridge,mlp,gbm`. Why there is no
  LSTM/CNN over raw prices is argued in `model.py`'s docstring: with ~430
  weekly cross-sections and factors that already summarise the trailing year,
  the shallow net *is* the appropriate neural network, and the walk-forward
  decides whether even it earns its place.

Rules are fixed before the numbers: learners -- IC t >= 2, then the highest
contest-window active t (a new learner needs t >= 2 of its own), ties to the
simpler model; weighting and guard -- keep >= 75% of the current alpha AND
do not raise the weekly 95% VaR.

Results on 2026-10-19 (full data, 30-day folds; details in CHANGELOG 4.20.0
and config.yaml): ETFs -- factor weighting adopted (alpha +2.7%/yr vs
-0.4%, VaR unchanged), learner and guard unchanged, and on the complete ETF
model's IC is borderline (t +1.9 to +3.2) with no contest-window edge, so
`auto` holds the core basket; stocks -- HRP adopted (alpha +11.9%/yr, VaR
2.54% vs 3.16%), guard unchanged, and gradient boosting beat ridge by a
wide, significant margin (6-week active t +3.95 vs +2.07) and passed the
7-check validation suite -- adopted (`model_type: gbm`).

### v4.21: the live-run fixes (27 Oct 2026)

The first live week exposed four operational defects (CHANGELOG 4.21.0):
runs that used two-day-old signals, a sell tab that ordered a full turnover
every day, a training set that drifted with the rolling data window, and
an ETF book 72% in silver. Now: the pipeline refuses stale data (exit 3,
the workflow retries at 11:40 UTC), the book is held for a whole rebalance
period (weeks), `history_start` pins the first data date, `auto` needs
three consecutive agreeing runs to switch, and the ETF sleeve is equal
weight with the near-duplicate guard on.

**How to trade it:** on the first run of each week (Monday, or the first
successful run after), enter the orders on the Stocks/ETFs tabs and sell
what the When-to-sell tab lists; on other days, hold. The orders shown
mid-week are for anyone entering then, not a signal to re-trade.

### HTML report

```
python -m umushroom.cli --config config.yaml --weighting equal --report
open umushroom_report.html
```

One file, opens in any browser, no internet needed: today's orders with
the facts behind each pick, the top-ranked stocks, out-of-sample alpha
charts, and a Monte Carlo of the contest (`--mc-weeks 8 --mc-sims 5000`).
Add `--validate` to include the validation badges.

### Reading the result honestly

The number to look at is the **IC t-statistic** in the purged
walk-forward section, not total return. If the VERDICT says there is no
statistically reliable evidence of skill, the honest choice is to hold
the equal-weight universe (or present the strategy as risk-controlled
rather than alpha-generating). A great-looking return with an IC t-stat
near zero is luck.

## Test

```bash
pytest                      # full suite, with coverage report
pytest tests/test_features.py -v   # one module
ruff check umushroom tests  # lint
mypy umushroom               # type check
```

The test suite runs entirely offline against synthetic, seeded price data
(see `tests/conftest.py`) — no network access needed, no yfinance calls in
tests. This is deliberate: the one module that talks to the network
(`umushroom/data.py`) is kept thin and isolated specifically so everything
else is verifiable without it.

Key properties the tests assert, beyond "does it run":
- **Known-answer alpha recovery**: on synthetic data with a planted,
  persistent signal the full pipeline must find a significant
  out-of-sample IC; on pure noise it must report NO skill
  (`tests/test_alpha_recovery.py`) -- the test that catches leakage
- **No label leakage**: corrupting every price after a training cut-off
  leaves every training row unchanged; each fold's last training label
  closes before its test window opens
- **No lookahead**: features computed "as of" day N are bit-identical
  whether or not future data exists in the DataFrame
  (`test_build_features_no_lookahead`)
- **Reproducibility**: same seed + same data → bit-identical predictions
  (`test_same_seed_same_data_gives_identical_predictions`)
- **Balance safety**: an allocation can never deploy more than the
  configured balance, never produces negative shares, never duplicates a
  ticker (`test_allocation.py`)
- **Fail loudly, not silently**: insufficient history, a shrunken universe,
  an overfull top_n, or a malformed config all raise specific typed
  exceptions rather than producing a degraded-but-plausible-looking result

## Architecture

```
umushroom/
  config.py       # Typed, validated StrategyConfig (dataclass + YAML + env)
  exceptions.py   # Typed exception hierarchy
  logging_config.py
  universe.py     # Exchange/currency per ticker, FX symbols, concentration check
  catalogue.py    # v4.19: scrape UMushroom's public ETF catalogue (sitemap + API)
  etf_universe.py # v4.19: ISIN -> Yahoo line mapping, audited universe CSV
  window.py       # v4.19: contest-window (30-day) evaluation + verdict
  data.py         # Fetch w/ retry+backoff, validate, FX-convert, parquet cache
  features.py     # Pure functions: prices -> cross-sectionally ranked factors
  model.py        # RankingModel (ridge | mlp | gbm): fit/predict/save/load, purged date-CV
  factor_risk.py  # v4.20: factor exposures, factor covariance, factor-risk-parity weights
  backtest.py     # Quick IN-SAMPLE sanity check (not a performance claim)
  purged_backtest.py      # Out-of-sample purged/embargoed walk-forward + skill stats
  robustness.py           # Block-bootstrap Sharpe confidence interval
  factor_regression.py    # Alpha vs universe and Fama-French (Newey-West t-stats)
  portfolio_construction.py  # Equal weight and native-scipy HRP
  prediction_log.py       # Next-run accuracy check of logged predictions
  allocation.py   # Rankings -> balance-constrained share order, validated
  cli.py          # Orchestration + argument parsing + exit codes
tests/            # Offline, synthetic-data test suite (pytest)
config.yaml       # Your contest's parameters — edit this, not the code
.github/workflows/ci.yml   # Lint + type-check + test on push/PR
```

## Known limitations (stated explicitly, not hidden)

- **Tradability is not verified automatically.** UMushroom has no public
  API; confirm each ticker in `config.yaml` is available on the platform.
- **Currency.** Prices are converted at Yahoo's daily FX close. If
  UMushroom converts at a different rate or time, sizing differs slightly.
- **Fama-French factors are USD**; the strategy is in your base currency.
  The FX mismatch adds noise to that regression (flagged in its output).
- **Survivorship bias.** The universe is today's index members, so the
  backtest never holds companies that later dropped out. This flatters
  absolute returns; it matters much less for the IC/active-return
  comparison against the same universe.
- **Multiple testing.** Every time you try a new feature set or parameter
  and keep the best, the backtest t-stat overstates real skill. Decide
  the configuration, then run the backtest; require t >= 3 if you iterated.
- Ridge regression on 9 ranked factors is deliberately simple for the
  data size -- see `model.py`'s docstring.


## Website (on your own computer)

```
python -m umushroom.web
```

Opens http://127.0.0.1:8000 with a tab per section: **Overview**, **Stocks**, **When to sell**,
**Tracker** (each pick's price now vs. at prediction), **Graphs**,
**Monte Carlo**, **Risk (VaR)**, **Macro**, **Checks** and **History**, plus a **Run now** button. Add `--daily 07:30` to run automatically every weekday.
Stop it with Ctrl+C. Nothing leaves your computer.


## Sharing the website with your team

```
python -m umushroom.web --team
```

It prints an address and a password. Teammates open the address and type the
password (any username). Your Mac must stay on with that window open. The first
time, macOS may ask whether Python can accept incoming connections: click Allow.

* **Same Wi-Fi:** use the "same Wi-Fi" address it prints.
* **Anywhere (recommended): Tailscale**, a free private network -- the site is
  never on the public internet.
  1. Install Tailscale on your Mac (tailscale.com/download) and sign in.
  2. In the Tailscale admin console: **Machines** -> your Mac -> **...** ->
     **Share** -> share by email (or copy an invite link) for each teammate.
  3. Each teammate installs Tailscale, signs in and accepts the invite.
  4. They open `http://<your Mac's 100.x address>:8000` (shown in the
     Tailscale app) and enter the password.

To change the password, delete `site/team_password.txt` and restart (or set
`UMUSHROOM_PASSWORD`).


## Website without your laptop (GitHub)

GitHub runs the model every weekday morning (~07:20 UK time) in its cloud and
publishes the website, locked with a team password. Teammates just open the
link. Every page is encrypted, so the picks stay private even though GitHub
Pages sites are public.

**Private or public repo?** Pages from a *private* repo needs GitHub Pro --
free for students with the GitHub Student Developer Pack
(education.github.com). With a *public* repo the picks are still encrypted,
but anyone can read your code and config. Use `--private` below if you have
Pro, otherwise `--public`.

One-time setup, in Terminal (needs Homebrew):

```
brew install gh
gh auth login
cd ~/Downloads/umushroom_pkg\ 4
mv website.yml .github/workflows/website.yml
git init -b main
git add .
git commit -m "UMushroom pipeline"
gh repo create umushroom --private --source . --push
python -c "import secrets; print('-'.join(secrets.token_hex(3) for _ in range(4)))"
gh secret set UMUSHROOM_PASSWORD
gh api -X POST "repos/$(gh api user -q .login)/umushroom/pages" -f build_type=workflow
gh workflow run website.yml
```

* The `python -c` line prints a strong password: copy it, then paste it when
  `gh secret set` asks. Give it to your teammates.
* The first run takes a while (it downloads prices and runs the backtests).
  Watch it with `gh run watch`.
* The site: `https://<your GitHub username>.github.io/umushroom/`

Run it any time: on GitHub open **Actions -> Website -> Run workflow** (the
site's "Run now" link goes there). After you change the code:
`git add . && git commit -m "update" && git push`.

Change the password: `gh secret set UMUSHROOM_PASSWORD`, then run the workflow.


## Sector cap (optional)

Test it first (one head-to-head, decision rule fixed in advance):

```
python -m umushroom.cli --config config.yaml --compare-sector-cap 3
```

Only if it prints `VERDICT: SWITCH THE CAP ON`, add `max_per_sector: 3` to
config.yaml.
