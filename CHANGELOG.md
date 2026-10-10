# Changelog

## 4.18.0 (2026-10-10) -- 6-week Monte Carlo + sector cap (opt-in), no crypto
- Crypto sleeve switched OFF in config.yaml (model showed no skill; holding
  all 18 coins only added volatility). Equity 80%, ETFs 20%. The crypto
  section is kept, commented out, to switch back on later.
- Monte Carlo default horizon 8 -> 6 weeks: UMushroom UK Stage 1 runs
  12 Oct - 22 Nov 2026. VaR/ES for the horizon follow.
- New optional sector cap for the equity sleeve (`max_per_sector: N` in
  config.yaml; OFF by default). The model's ranking is unchanged; the cap only
  skips to the next-ranked stock when a sector already has N picks. Used the
  same way in live orders, the walk-forward backtest and the validation suite.
- `--compare-sector-cap N`: one pre-declared head-to-head on identical data
  (no cap vs. max N per sector) reporting alpha, alpha t, Sharpe, weekly
  VaR/ES, worst week and how concentrated the picks were. Rule fixed in
  advance: switch the cap on only if it keeps >= 75% of the alpha AND does not
  raise weekly VaR.
- 17 new tests.

## 4.17.0 (2026-10-09) -- website without your laptop (GitHub)
- New `.github/workflows/website.yml`: every weekday ~07:20 UK time (and on
  demand: Actions -> Website -> Run workflow) GitHub's cloud runs the model and
  publishes the website to GitHub Pages. Your laptop can be off.
- New `umushroom/static_site.py`: builds every tab as a static page and
  ENCRYPTS each one with the team password (AES-256-GCM, PBKDF2-SHA256
  600,000 iterations). The published files are unreadable without the
  password; the browser decrypts locally. "Remember on this device" option.
  Password must be 12+ characters (the encrypted pages are public).
- A failed run still publishes the last good run with a warning banner, and
  marks the workflow red so GitHub emails you.
- Past runs (and the price cache) are kept between runs in the Actions cache;
  the newest 40 runs are kept.
- CI never prints a successful run's log (public repos have public logs).
- 9 new tests (encryption round trip, wrong password, fresh IVs, nothing
  readable in the published files, failure banner, pruning).

## 4.16.0 (2026-10-09) -- Macro tab + team access
- New Macro tab (reporting only -- the model is unchanged):
  * what the economy is doing: global stocks, US 10-year yield, UK gilts,
    Brent oil, US dollar, GBP/USD, gold, VIX -- level, 1M/3M change and where
    each sits in its 1-year range;
  * the portfolio's market beta and its sensitivity to each driver on top of
    the market (from ~2 years of weekly returns), per holding too;
  * stress tests (stocks -10%, yields +0.5pp, oil +20%, pound +5% incl.
    currency translation of USD holdings, VIX +10, ...);
  * currency mix and auto-written talking points for the pitch's macro and
    risk-management slides.
- Team mode: `python -m umushroom.web --team` lets teammates open the website
  from their own computers (same Wi-Fi, or anywhere via Tailscale). Password
  protected (generated once, kept in site/team_password.txt, readable only by
  you; or set UMUSHROOM_PASSWORD); 10 wrong guesses in 10 minutes -> locked
  out; it refuses to open to other computers without a password.
- 16 new tests (known-answer betas, scenarios, currency translation, password
  checks, lockout).

## 4.15.0 (2026-10-09) -- When to sell
- New "When to sell" tab: compares the orders of the run you bought from
  with the latest run and lists every trade, sells first: SELL ALL (dropped
  out of the picks, with its new rank), SELL (trim), BUY NEW, BUY MORE, HOLD
  -- with shares to trade, approximate value, % change since you bought and
  the reason. Changes under 2% of a position are left alone.
- Shows when the next sell/rebalance check is due (rebalance_days trading
  days after the picks you hold), or "Sell / rebalance now" when fresh picks
  are ready. Overview gets a "Next sell / rebalance check" tile.
- Uses ONLY the strategy's backtested exit rule (re-rank every period, sell
  what drops out). No stop-loss/profit target: none was tested.
- Reports now record rebalance_days. New `umushroom/rebalance.py`; 11 new tests.

## 4.14.0 (2026-10-09) -- Value at Risk
- New `umushroom/risk.py`: historical VaR and expected shortfall (ES) at 95%
  and 99%, the Kupiec proportion-of-failures test, and a walk-forward VaR
  backtest (each week's VaR uses only the previous 52 weeks).
- Monte Carlo stats add 99% VaR and 95%/99% ES over the contest horizon.
- New Risk (VaR) tab: 1-week VaR/ES from the real out-of-sample weeks (all
  sleeves at their capital weights, each sleeve as actually held), horizon
  VaR/ES from Monte Carlo, in % and GBP; histogram of weekly returns (never
  trimmed: the worst weeks are the point); "did the VaR hold?" chart;
  per-sleeve table. Overview gets a VaR tile; Checks gets the VaR backtest.
- Older reports without risk data still open (the tab explains why it's empty).
- 13 new tests (textbook normal VaR/ES, Kupiec known answers, a VaR that
  must FAIL when risk jumps, no look-ahead in the backtest).

## 4.13.0 (2026-10-09) -- website tabs + since-prediction tracker
- The website now has a tab per section: Overview, Stocks (orders and
  top-ranked), Tracker, Graphs (performance, alpha, deciles), Monte Carlo,
  Checks (validation), History. Any past run can be opened in every tab.
- NEW Tracker tab: each pick's latest price against its price on the day of
  the prediction -- change %, price now, value then/now, gain, a chart of
  the picks' value since, and best/worst. Pence-safe (both prices come from
  the same Yahoo series). Prices cached 15 min; "Refresh prices" re-fetches.
  A failed download shows a message instead of breaking the page.
- Money axis ticks no longer repeat ("9k 9k 9k") for small ranges.
- 22 new tests (tracker known-answer tests, tabs, caching, failures).

## 4.12.0 (2026-10-09) -- the whole thing as a website on your own computer
- `python -m umushroom.web` starts a local website at http://127.0.0.1:8000
  (opens in your browser). Home = latest report; toolbar with "Run now"
  (runs the full pipeline in the background, shows live progress, reloads
  when done) and "History" (every run with its report, orders CSV, log).
- `--daily 07:30` also runs automatically every weekday at that time while
  the website is open; `--validate` adds the validation suite to each run.
- Runs are kept in ./site/reports. A failed run never replaces the latest
  good report; a run cut off by closing the website is marked "interrupted".
- Safety: listens on 127.0.0.1 only; "Run now" needs a custom header and a
  same-origin request; file names are strictly checked; one run at a time.
- Standard library only (nothing new to install). 13 new tests.

## 4.11.0 (2026-10-08) -- graph errors fixed and guarded by tests
- Histograms: the extreme 0.5% each side was squashed into the end bars,
  drawing fake spikes at both edges. Now left off the axis; the caption
  says what share is hidden (the table still includes it).
- Line charts: a series identical to another (e.g. ETF "ours" = benchmark)
  was drawn underneath and looked missing; it's now merged ("A = B"). The
  main series is drawn on top, and end labels that would overlap are skipped.
- A missing benchmark week no longer breaks the curves (counted as flat).
- Rolling 1-year alpha with under 26 weeks of history shows a note, not an
  empty chart.
- "5% worst case" tile no longer prints a double sign ("--7,779").
- Tick labels: no "-0%", decimals follow the step, no scientific notation.
- Orders: an instrument whose ONE unit costs more than its share of the
  budget (SGLD.L) used to appear as a 0-share order with its money unused.
  It is now dropped with a warning; model sleeves bring in the next-ranked
  name, hold-all sleeves spread the money over the rest.
- "Etf" -> "ETF" in sleeve headings.
- 11 new regression tests.

## 4.10.0 (2026-10-08) -- HTML report with alpha charts and Monte Carlo
- `--report [PATH]` writes one self-contained HTML file (no internet needed):
  headline tiles; today's orders with name, sector, 1/3/12-month returns,
  volatility; the 25 top-ranked stocks; per sleeve out-of-sample growth,
  cumulative alpha, rolling 1-year alpha, alpha bootstrap histogram and
  validation badges; prediction deciles; Monte Carlo fan chart and final-
  value histogram (`--mc-weeks`, `--mc-sims`). Light/dark mode, hover
  readouts, a table view under every chart.
- `montecarlo.py`: block bootstrap of real out-of-sample weeks; strategy and
  benchmark always drawn from the same weeks; alpha bootstrap.
- Hold-everything sleeves are charted as what is actually held, with the
  unused model shown only as a labelled reference.

## 4.9.0 (2026-10-08)
- Validation run: equity 7/7 PASS (placebo 20/20, deciles +0.92, 9/9 years,
  all markets). Crypto FAILED (placebo p=0.48, deciles unordered, no delay
  robustness, BTC 87% of weeks) -> crypto sleeve now holds all 18 coins at
  equal weight, the same rule already applied to ETFs.

## 4.8.0 (2026-10-08) -- fixes from the first validation run (model unchanged)
- Validation result, equity sleeve: 7/7 PASS (placebo 20/20, deciles
  ordered 0.90, IC positive 9/9 years, all 3 markets, gradual delay decay).
- FIX: ETF sleeve crashed when Yahoo had no fresh London ETF closes; the
  hold-all sleeve now sizes on the previous close with a warning (only
  model-picked sleeves exclude stale prices, since there it's a stale signal).
- One failing sleeve no longer stops the other sleeves' orders.
- Ticker checker never suggests OTC/ADR lines (it had proposed RHHBY for
  Roche, a 1/8-share ADR that would size orders ~8x wrong).
- Tickers: FLTR->FLUT (US), PHNX->SDLF.L, TM17->EVPL.L, HELN->HBAN.SW;
  Roche via RO.SW (ROG.SW gone from Yahoo); dropped 6 micro-caps whose
  post-merger Yahoo series can't be verified against UMushroom.
- Orders show a `priced_from` column when a substitute Yahoo symbol is used.

## 4.7.0 (2026-10-08) -- validation suite + missing-ticker fixes (model unchanged)
- NEW `--validate`: placebo/permutation test, decile monotonicity, per-year
  stability, per-market IC, 1- and 2-day execution delay, benchmark data
  sanity, pick concentration -- PASS/WARN/FAIL per check. Known-answer
  tested: passes a planted signal, does not pass pure noise.
- NEW `python -m umushroom.check_tickers`: finds tickers Yahoo can't price
  and searches Yahoo by ISIN/name for the right symbol.
- FIX ticker mapping: single letters at the start of a company name were
  read as share classes (J Sainsbury -> SBRY-J.L). 40 mappings corrected;
  SBRY, MNG, BME, JDW, VZUG, RSGN now load; SPGI and TMUS restored.
- `yahoo_symbols`: CRH, Ferguson, Indivior fetched from their US listings
  (Yahoo dropped the London lines; UMushroom still lists and prices them).

## 4.6.0 (2026-10-08)
- Backtest summary shows the test period and ANNUALISED returns next to
  totals (crypto's 1,379% was ~6 years of compounding = ~54%/yr).
- Stale-price guard: instruments with no actual close on the latest day
  (price carried forward) are flagged and never picked for new orders.

## 4.5.0 (2026-10-08) -- head-to-head result + fixes from the 10-year run
- Head-to-head (rule fixed before looking): the v4.3 stock setup beat the
  v4.4 one on 10 years of data -- alpha +11.1%/yr (t=+2.22), active t=+2.07
  vs +5.1% (t=+1.19). Equity sleeve reverted to exchange-neutral, 5-day,
  top 10, price factors; 10-year + ragged data kept. Sector/volume code
  stays available but is off.
- FIX: ETF sleeve crashed when one ETF had no data today; `selection: all`
  now holds every instrument usable today.
- FIX: untradeable prices are excluded BEFORE picking, so the next-ranked
  name fills the slot (v4.4 left 42k of the 70k stock budget in cash).
- Position cap: no position above 2x equal weight (HRP had put 59% in one
  stock).
- Removed 16 USD-quoted London GDRs (e.g. CEIR.L priced 0.031 vs ~3.1).

## 4.4.0 (2026-10-08) -- better predictions, more data
- Sector-neutral ranking: stocks are ranked within sector-and-exchange
  (falling back to sector, then pooled), using sectors scraped from
  UMushroom for 607/608 stocks. Stops the model making one big sector bet
  (the first real run picked 7 banks out of 10).
- 21-day forecast horizon for stocks (crypto stays 5 days), still
  rebalanced weekly. Backtests now SCORE each period on the return earned
  until the next rebalance, so overlapping 21-day windows can't inflate
  t-statistics.
- 25 stock holdings instead of 10.
- More data: 10-year history; ragged panel (each stock joins once it has a
  year of its own data instead of being dropped); trading volume with two
  new factors (abnormal volume, Amihud illiquidity). Missing volume is
  neutral, not a reason to drop a stock.
- `--compare-baseline`: one pre-declared head-to-head of the v4.3 equity
  setup vs the current one on identical data.
- Fix: legacy get_price_data() broke when the fetcher started returning
  volumes (caught by mypy).

## 4.3.1 (2026-10-08)
- FIX: IGLT.L is quoted in pounds on Yahoo, not pence; it was priced
  100x too low (0.0952 vs UMushroom's 9.48). Override added.
- Guard: stock/ETF orders with a local price below 0.10 are skipped with
  a loud warning (catches any future pence/pound error).
- All 27 orders from the first real run were checked against UMushroom's
  own prices; every other instrument matched.

## 4.3.0 (2026-10-08) -- fixes from the first real run
- Data: stocks/ETFs with a >95% single-day move are now DROPPED (they are
  Yahoo pence/pound flips -- 14 found on the first run, they inflated the
  equal-weight benchmark ~17% in a month and corrupted rankings). Crypto
  is exempt.
- ETF sleeve: holds all 13 ETFs equal-weight (`selection: all`), since
  the first run showed no ETF skill and significantly negative alpha.
  New per-sleeve `selection` and `weighting` settings.
- Removed the PerformanceWarning flood from FX conversion.

## 4.2.0 (2026-10-08)
- Sleeves: the portfolio is now split into separately-modelled sleeves,
  each with its own model, purged backtest, skill verdict, alpha and
  orders: equity 70% (top 10), ETF 20% (top 4 of 13), crypto 10% (top 4
  of 18). Every ETF and coin was confirmed on UMushroom via its search.
- Crypto: Yahoo `XXX-USD` tickers recognised; prices aligned to the stock
  trading calendar (weekend moves land in Monday's return); fractional
  units supported (`fractional: true`), always floored so orders can't
  overspend.
- ETFs: currency overrides for USD-quoted London/Swiss ETF lines.
- Output: one combined order table with a `sleeve` column and per-sleeve
  totals; per-sleeve prediction logs.

## 4.1.0 (2026-10-08)
- Universe rebuilt from UMushroom's own equity list (4,491 stocks scraped
  from the platform): 608 tickers = all 299 UMushroom London stocks, all
  207 Swiss stocks, 102 US large caps. Removes SHEL.L and AHT.L, which
  UMushroom does not list in London.
- Added `umushroom_universe_mapped.csv`: every UMushroom equity with ISIN,
  exchange and its Yahoo Finance ticker.

## 4.0.0 (2026-10-08)
- Universe: multi-market (FTSE 100 + US large caps + SMI) replacing the
  Swiss-only v3 list; exchange-mix logging and concentration warning.
- Currency: all prices converted to `base_currency` (GBP default);
  London pence handled; FX never back-filled.
- Data: drop today's incomplete bar; drop low-coverage tickers instead
  of truncating the universe.
- Features: 9 published factors, cross-sectionally ranked per date,
  exchange-neutral; fast causal feature panel.
- Target: cross-sectional rank (or excess) of forward return.
- Model CV: date-based, purged, reports IC and t-stat.
- Purged backtest: IC t-stat, hit rate, active return t-stat, long-short
  spread, VERDICT line; off-by-one boundary fix.
- New `factor_regression.py`: alpha vs universe and Fama-French with
  Newey-West t-stats; `--fama-french` CLI flag.
- HRP reimplemented natively on scipy; PyPortfolioOpt dependency removed
  (Python 3.14 compatible).
- Tests: 92 -> 161, including known-answer alpha recovery, no-alpha-in-
  noise, label-leakage and end-to-end CLI tests. CI matrix adds 3.13/3.14.
