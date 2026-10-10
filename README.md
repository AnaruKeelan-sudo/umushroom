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
| equity | 70% | top 25 | 608 UMushroom-listed London, Swiss and US stocks; ranked within sector, 21-day horizon, price + volume factors |
| etf | 20% | all 13, equal weight | 13 ETFs: world, S&P 500, Nasdaq, EM, Euro Stoxx, FTSE 100, gilts, Treasuries, gold, silver, oil, clean energy |
| crypto | 10% | top 4 | 18 large coins (fractional units) |

Assets are only ever ranked against their own sleeve, so crypto's much
higher volatility never distorts stock rankings. A sleeve whose VERDICT
shows no skill can be cut by lowering its weight.

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
  data.py         # Fetch w/ retry+backoff, validate, FX-convert, parquet cache
  features.py     # Pure functions: prices -> cross-sectionally ranked factors
  model.py        # RankingModel: fit/predict/save/load, purged date-CV with IC
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
