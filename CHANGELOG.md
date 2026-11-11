# Changelog

## 4.28.0 (2026-11-11) -- why the live picks lagged; five phase models; "fresh" means every sleeve; a state save that cannot brick the site
An end-to-end investigation of "the model is not working" (live rank
correlation negative in 11 of 12 scored cross-sections, picks 1.4% a period
behind the universe), then the defects it turned up.

WHAT WAS WRONG WITH THE RESULTS -- and what was not
- Not a bug in the scoring path. Three independent checks agree:
  1. A point-in-time replay of the live pipeline through its real entry
     point, with the data cut off after each day, for all 136 trading days
     from 1 May to 9 Nov 2026: its IC tracks the published live record at a
     correlation of 0.95 (mean -0.100 against -0.116 on the 12 scored dates),
     so stale or partial data at run time explains almost none of it.
  2. The walk-forward, re-run with its fold grid shifted so that the test
     windows cover the contest weeks out of sample: IC -0.27, -0.17, -0.13,
     -0.05 for the signals of 12, 19, 26 Oct and 2 Nov (live: -0.32, -0.16,
     -0.12, -0.09). On every date both have, replay and walk-forward
     correlate at 0.97 with equal means: the backtest and the live procedure
     are the same thing. No look-ahead, no train/serve difference, and the
     live record is measured correctly (score of day D against D -> D+5).
  3. Factor attribution on the model's own inputs: over the four contest
     weeks 12-1 momentum had a weekly IC of -0.155 and residual momentum
     -0.173 (long run +0.03; lower in only 2.6% and 1.5% of all four-week
     stretches since 2017), with short-term reversal at -0.057 (long run
     +0.03) at the same time. The model buys twelve-month winners that have
     just dipped; from 2 Oct the winners kept being sold. September 2026 had
     been its best month since May (weekly IC +0.10, positive every week).
- It is not a reason to switch the model off or turn it round. Weekly factor
  IC has no memory (autocorrelation between -0.07 and +0.03), and after a
  four-period stretch with a mean IC below -0.03 the walk-forward's NEXT
  period averaged +0.058 against +0.036 for all periods (30 cases). A rule
  that stopped trading after bad weeks would have cost money.
- The shipped walk-forward never scored these weeks: its last test fold ends
  on 2 Oct (a fold needs 21 more days), so the Graphs and Checks tabs kept
  showing a record that stops a week before the contest. The Tracker's live
  record is where they appear. Unchanged in this release; stated here.
- What nobody can deliver: an edge of IC 0.04 is a few tenths of a percent a
  week on ten stocks whose weekly spread is several percent. One or two
  remaining rebalances are close to a coin toss whatever the model.

WHAT WAS BROKEN
- THE PICKS DEPENDED ON AN ACCIDENT (`train_phases`, model.py `phases`). The
  model was fitted on every 5th trading day -- one of five equally valid
  samples of the same history. A one-day shift of that sample changed 6 of
  the 10 live picks of 9 Nov (rank correlation of the scores 0.86); a price
  revision of one part in a billion, far less than Yahoo's daily dividend
  re-adjustment of old prices, changed 3 to 4. Hence the cloud and a laptop
  shared 6 of 10 picks for the same close (same code, same library versions),
  one day's picks shared 4.8 of 10 names with the next day's, and the cloud's
  own 8-year walk-forward alpha, on the same folds and an unchanged
  benchmark, read +21.3%/yr (t 4.15), +18.5% and +14.3% (t 2.96) on three
  consecutive runs of the same code (6, 9, 10 Nov), after +7.1% (t 1.45) and
  +12.8% on 4 and 5 Nov. Now five models, one per phase, each exactly the old model on
  its own sample, predictions averaged: every trading day is used, and 8 to 9
  of 10 picks survive the same revisions (rank correlation 0.993).
  `--compare-train-phases equity`, rule declared before the results (adopt
  only if IC t >= 2, contest-window active t >= 2, IC >= the mean of the five
  single-phase ICs, contest-window active t >= their median):
  the five single phases   IC +0.0386..+0.0418   6-week active t +2.69..+4.25   win 59-78%   VaR95/wk 2.88-3.81%
  ensemble5                IC +0.0433 (t +6.09)  6-week active +3.02% (t +5.14)  win 80%      VaR95/wk 3.08%
  -> ADOPTED (`train_phases: 5`; code default 1 = the old behaviour; the ETF
  sleeve keeps one model). It is one more comparison on the same eight years,
  but an average of the same learner, not a new signal, and it beats every
  one of its own members. On the contest weeks it is the wrong way round too
  (IC -0.14 against -0.16; picks -0.95% a week against -1.41%).
- THE ETF SLEEVE RAN ON YESTERDAY'S CLOSES (`max_stale_fraction`). The
  freshness check asked only whether the latest ROW was recent, and a row
  exists once half the stocks have traded. At ~09:30 UTC Yahoo has the
  stocks' closes of the day before but not the ETFs': on 3, 4, 5, 6 and 10
  Nov, 10 to 12 of the 11-13 ETFs were sized on the close of the day before
  (4% off for one of them on 10 Nov), the whole ETF universe was scored on
  day-old prices, and
  the 11:40 slot reused that download. Mondays were unaffected (Friday's
  closes are complete), so the weekly book was not. A run where more than
  half of any sleeve has no close for the latest day is now "not ready"
  (exit 3), its download is not cached, and a later slot fetches again. When
  Yahoo has the ETFs varies (in by 12:35 UTC on 10 Nov, still missing at
  16:35 on 8 Oct), so there are now five slots, 08:40 to 20:40 UTC; after
  the first success the later ones reuse its download and change nothing.
  The whole run waits, rather than publishing the stocks without the ETFs: a
  book without one of its sleeves would make the NEXT period's trade list
  treat that sleeve as not held. Code default 1.0 = off. Such a
  morning is a wait, not a failure, so the site now says "The prices were not
  complete yet at ... the next scheduled run tries again" instead of "The
  latest run failed", and History lists the run as "not ready".
- YESTERDAY'S DOWNLOAD COULD PASS AS TODAY'S (data._cache_is_fresh). A
  download made while markets are open holds a part-day bar dated that day.
  The same day it is dropped as incomplete; reused after midnight, inside the
  12-hour cache age, it counted as a completed close -- signals and order
  prices on a mid-session snapshot, every market that had not yet opened
  carried forward. Found when an offline run at 00:14 reused the 13:57
  download of the day before (as_of one day ahead, 293 instruments stale);
  the later slots would have put a "Run now" after midnight in that window.
  The cache is now good only for the London day it was fetched on.
- ONE BAD SAVE WOULD HAVE STOPPED THE SITE (v4.27's encrypted state, before
  its first release). The archive was piped without `pipefail` and never
  read back, so a truncated save would have been stored and every later
  build would have stopped at "Cannot decrypt"; and with the secret missing
  the state would have been saved under an EMPTY password. Packing now
  refuses a password under 12 characters, verifies that the archive reads
  back, saves nothing otherwise, never blocks the site from publishing, and
  turns the workflow red ("The run history could not be saved"). The eleven
  branches were simulated with the workflow's own shell steps.
- THE SELL TAB ON THE DAY IT MATTERS. An independent review of this release
  walked the calendar to 23 Nov with the real functions and found three
  older faults, each in the window of the last rebalance:
  1. A run made by hand at the WEEKEND (say after setting ACCOUNT_VALUE, as
     the site asks) opened the period on Friday's close -- and Monday's
     scheduled runs, on the same close but another day, then read "Holding
     period in progress ... nothing to do" with no trade list, on the day the
     team trades. A run on the first run's close now counts as the rebalance
     up to and including that close's trade day (rebalance.cadence); a run a
     day late on the same close still holds.
  2. On the due day, before the new period's run has succeeded (it failed, or
     is waiting for Yahoo), the box read "0 trading days to go ... no sells".
     It now says "A rebalance is due today -- its trade list is not ready
     yet ... do not trade from this page yet".
  3. On a same-day repeat run the list is the first run's, but the note above
     it quoted the LATEST run's sizing ("sized on the account value 90,000"
     over a list sized on 100,000, after the variable was set mid-morning).
     The note is now the list's own, and names the later run and the factor
     to scale by.
  Known and not changed: a holding with no fresh price on the rebalance run
  is kept (HOLD) on top of a fully sized new book, so that list's buys cost
  one position more than its sells raise.
- THE WORKFLOW COULD LOSE THE BOOK. (a) A RE-RUN of a workflow run has the
  same run_id, Actions caches are immutable, and the save step only warns
  when its key exists: "Re-run all jobs" after a failed Monday run would
  have published the book and not saved it, and the next run would have
  taken its own orders for the book the team holds. The attempt is now part
  of the cache key. (b) A state.enc missing because the cache SERVICE failed
  was taken for a first run: the pre-4.27 cache (or nothing) would then be
  saved as the newest state, for good. The cache list is now asked first;
  if a saved state exists, the build stops with "run the workflow again".
  (c) A run that outlasted static_site's 3-hour wait was left running while
  the build died before writing a page. It is now stopped, recorded as
  failed, and the last good site is published (`--run-timeout`, 30 minutes
  inside the job's own limit).
- THE CLOUD'S LIBRARIES ARE PINNED (`constraints-site.txt`): NumPy, pandas,
  scikit-learn, SciPy, pyarrow, joblib, threadpoolctl at the versions of the
  run of 10 Nov, on which the pipeline was checked to give the same orders as
  on the versions it was developed on. yfinance stays unpinned (1.7.0 and
  1.8.0 return identical prices; a Yahoo API change needs its next release).
- COST: a run fits five models wherever it fitted one, so it takes about
  three times as long (the walk-forwards dominate), and `--validate` about
  five times: a manual workflow run with the validation suite gets a
  360-minute limit, scheduled runs keep 150 so a hung run frees the next slot.
- 38 new tests: the ensemble equals the mean of the five single-phase models
  to 1e-12 and takes every k-th date; a fold with phase models never trains
  past the purge gap; the verdict rule; the head-to-head and a live run
  through the CLI; a sleeve without the latest closes is refused, not cached,
  and exits 3 end to end; one market's holiday is not; yesterday's download
  is fetched again; a weekend run still shows Monday its list, the due-day
  box and the sizing note say what is true; a run past its limit is stopped
  and the site still published; and the workflow's own shell steps -- pack,
  unpack, the missing-state check -- are run in every branch
  (tests/test_workflow.py: on CI, with the runner's bash, tar and openssl).

## 4.27.0 (2026-11-05) -- orders sized on the account value, an encrypted Actions cache, an age warning, the whole weekly record
Found in the 5 Nov review; every change below has a test that fails on 4.26.0.
- ORDER SIZING (`account_value`, env `UMUSHROOM_ACCOUNT_VALUE`, in the cloud
  the repository variable ACCOUNT_VALUE). Every run sized its orders on the
  fixed starting_balance, so the Sell tab's targets ignored what the account
  is worth: after a 10% fall the buys cost about 10k more than the sells
  raised; after gains the profit was trimmed into cash and never reinvested,
  unlike the compounding backtest. When set, every sleeve's budget is its
  weight of the account value; unset = the old behaviour. The report records
  `capital` / `capital_source` and the Sell tab says which was used (and, on
  the fixed balance, that the buys can exceed the cash). Amounts like
  "98,500" or "£98,500.25" are accepted; an empty variable means unset.
- ENCRYPTED STATE IN THE ACTIONS CACHE (security). The repository is public
  and its Actions caches can be restored by pull-request workflows; the cache
  held `site/` -- every report in plain HTML/CSV/JSON -- so the picks could be
  read without the team password. The workflow now saves one `state.enc`
  (tar of `site/` and `.cache/`, AES-256 with a PBKDF2 key from the team
  password, 600k iterations) under `umushroom-state-*`, and restores the old
  plaintext `umushroom-data-*` cache only once, to migrate. If the state
  cannot be decrypted (a changed password) the build stops and saves
  nothing, rather than starting empty and burying the history. Delete the old
  `umushroom-data-*` caches (Actions -> Caches) after the first encrypted
  save. `umushroom_report*.html` (the default `--report`, plaintext) is now
  git-ignored.
- AGE WARNING. A build that crashed, timed out or never started (Actions
  outage; GitHub pauses scheduled workflows in a public repo after 60 days
  without commits) left yesterday's site up looking current. The page now
  checks its own close date in the reader's browser and shows "This page may
  be out of date" once two or more weekdays have closed since (normally at
  most one, before the morning run; a market holiday can trigger it a day
  early).
- THE WHOLE WEEKLY RECORD. Pruning kept the newest 40 runs -- two a day,
  failed ones included -- so "Week by week" only ever covered about four
  weeks and its oldest row could score a run the team never held. Every
  period's book (its first successful run) is now kept for good.
- 7 new tests.

## 4.26.0 (2026-11-05) -- the Sell tab never sells on a failed run, keeps its list after midday, and the retry slot really retries
Found in a review of the order path and the website workflow; every case
below has a test that fails on 4.25.1.
- SELL ALL ON A FAILED RUN (bug). `rebalance.plan` turned every holding the
  latest run had no order for into SELL ALL, "dropped out of the picks". A
  sleeve that failed that run (a Yahoo hiccup, too few instruments: the run
  still succeeds, the sleeve "stays in cash") therefore told the team to sell
  the whole ETF book, and a held model pick with no fresh close (a model
  sleeve never picks one) was sold too. The report now records
  `skipped_sleeves` and `stale_last` in its meta, and `plan` HOLDs those
  holdings with the true reason. A sleeve removed from the config is still
  sold; reports written before 4.26 behave as before.
- THE SECOND SLOT HID THE TRADE LIST (bug). Both daily slots publish when the
  data is fresh, and only the first run of a period counted as the
  rebalance, so from about 12:40 UK on a rebalance day the Sell tab said
  "Nothing to do until the next period starts". A later run on the same day
  and the same close is now still the rebalance; its trade list targets the
  period's book (the first run's orders, which the record scores and the
  Tracker follows: `Cadence.book`). A run on a later day still holds, even on
  the same close.
- THE RETRY SLOT REUSED THE STALE DOWNLOAD (bug). The download was cached
  before the freshness check refused it, and within cache_max_age_hours (12)
  the 11:40 slot read that cache and refused again without asking Yahoo. A
  refused download is now deleted from the cache.
- The workflow comment ("the other keeps the last good run") now says what
  the slots actually do.
- 6 new tests.

## 4.25.1 (2026-11-05) -- the repairs list names only what the run kept
- REPAIR REPORT (bug). The Checks tab's "Prices repaired in this run" card
  says the listed instruments "are kept instead of dropped", but it listed
  every repaired series -- including those the >95%-day guard still drops
  for a jump the rules cannot explain. Replaying the 2 Nov download: 44
  series repaired, 13 of them still dropped (COFF.L, CURN.SW, DTRE.L, EGAS.L,
  FTSE.PA, HLTW.L, JRDZ.L, JT13.L, NXTU.L, PRIJ.L, S5EW.L, TNOW.L, UC95.L).
  `MarketData.repairs` (and so the card, the run log and the report payload)
  now holds only repairs to instruments and FX pairs the run kept; the run
  log names the repaired-but-dropped ones on a separate line. Prices,
  orders and the model are unchanged.
- 1 new test (a series with a repairable bad print and a real +140% day is
  dropped and not listed; fails on 4.25.0).

## 4.25.0 (2026-11-03) -- a complete core basket, repaired prices, the Tracker in pounds, the live record
Found by decrypting the published site and replaying the 2-3 Nov runs offline:
- CORE BASKET (bug). The ETF sleeve held 11 of its 13 core ETFs. CSNDX.SW
  (Nasdaq 100) has no SIX prints from 24 Dec to 2 Jan, so its 252-day factors
  are undefined for a year, and a held-all basket only held instruments whose
  every factor was defined: it would have stayed out until January 2027.
  IB01.L (the cash-like ETF) was dropped by the >95%-day guard for two
  placeholder prints of 5.00 on its first days in 2019. Now a held-all basket
  holds every instrument with a usable price today (its model score is shown
  where defined and never logged as a prediction); IB01.L is fixed by the
  repair below. The model's own picks still never include an unscored name.
- PRICE REPAIR (`repair_price_errors`, code default off = old behaviour).
  The guard dropped 53 instruments on 2 Nov -- 13 stocks among them -- for one
  bad day in ten years. `data.repair_price_errors` fixes three kinds of Yahoo
  error first, under rules that never change an instrument's latest price
  (the one its order is sized on): listing artefacts (a >95% jump within the
  first 5 prints), pence/pound flips (day-on-day ratio 80-125x or its
  inverse; rescaled only when the latest print is in the series' usual unit)
  and bad prints (at most 5 prints that come back to within 25% of the last
  good close; replaced by that close, never by a later price). FX series get
  the same repair (JPYGBP=X had two one-day 100x prints, in 2016 and 2020).
  On 2 Nov: 44 series repaired, 31 recovered (7 stocks: BXP.L, ELM.L,
  FXPO.L, INDV.L, ISN.SW, LECN.SW, SAR.L; IB01.L and 22 other ETFs; the yen
  rate), 25 still dropped (real jumps, unadjusted splits, BCG.L's flip still
  open today). The Checks tab lists every repair of the run.
  HEAD-TO-HEAD on the same cached download (rule declared before the numbers:
  adopt unless the repaired stock sleeve's walk-forward IC t falls below 2 or
  its contest-window verdict is worse; the ETF sleeve is reported):
    stocks  off 568 names, IC t +4.71, alpha +10.7%/yr (t 2.09), window PASS
            on  575 names, IC t +5.17, alpha +19.1%/yr (t 3.22), window PASS -> ADOPTED
    ETFs    off 1,592 (12 core), IC t +2.92, alpha +0.1%, WARN -> core basket
            on  1,613 (13 core), IC t +2.95, alpha +0.0%, FAIL -> core basket
  The alpha gain is not a forecast. Attribution: the 7 recovered stocks were
  in the walk-forward's top 10 76 times in 330 weeks and add about 2.3 points
  a year (Sareum SAR.L most of it); the rest is the gbm model reshuffling its
  other picks when its training set changes (the fragility noted in 4.23).
  Today's stock orders change by 4 of 10 names; SAR.L is one of them (it was
  suspended for two weeks and resumed 40% lower, which the model's reversal
  factor favours).
- TRACKER IN POUNDS. The since-prediction values and the week-by-week record
  ignored currency moves, so a USD or CHF holding's value differed from what
  the UMushroom account shows. The FX pairs (USDGBP=X, CHFGBP=X, ...) now come
  in the same download and every value, the book and MSCI World (IWDA.L, the
  USD line) included, is in GBP; prices stay in each listing's currency. A
  holding whose rate cannot be fetched falls back to its local change and the
  page names it.
- THE LIVE RECORD ("Is the model right?", Tracker tab). The old check scored
  the previous run against today's price: one day of a 5-day forecast. Every
  logged cross-section is now scored once its horizon has passed, on the same
  base-currency closes the model trains on: rank correlation (IC) with what
  happened, picks against the universe; overlapping windows are shown but
  counted once. The live weeks are set against the walk-forward's own
  per-period IC ("how often did the backtest have a stretch this bad?") and
  against what the walk-forward did after the same number of lagging weeks
  in a row. On the local log (7 Oct - 2 Nov 2026) the stock model's live IC
  was -0.11, -0.20, -0.24, -0.21 and -0.13: negative in all five cross-
  sections (three independent weeks, mean -0.15, picks behind the universe
  by about 2% a week). In the walk-forward's 330 weeks, active returns were
  negatively autocorrelated (lag-1 -0.14) and the week after three lagging
  weeks beat the universe by +0.64% on average (17 times) against +0.25% for
  all weeks: by the model's own history a losing run is not a reason to
  switch it off. No gate was added; three weeks cannot overturn eight years by
  this repo's rules, and the page now shows the evidence every run.
- The ETF sleeve's prediction log holds the whole ranked universe (it held
  only the ~12 core ETFs bought, so the ETF model's live skill could never be
  measured); cross-sections of fewer than 30 names are not scored.
- CONFIG. A `UMUSHROOM_<FIELD>` environment variable now overrides ANY field,
  as the README always said; before, only keys present in config.yaml could
  be overridden, so a field left at its default silently ignored its variable.
- README: the run steps, schedule, learners and module list brought up to
  date; a v4.25 section.
- 28 new tests (held-all basket with a gap; every repair kind with known
  answers, the latest price never changed, idempotence, crypto exempt, FX;
  the loader on/off; GBP valuation, FX alignment and fallback, the record in
  GBP; live scoring, overlaps, the verdict, payload and page; the ETF log;
  a later run scoring an earlier one; the published Tracker pane; config).

## 4.24.0 (2026-11-02) -- a buy / sell link next to every instrument
- Every table that lists an instrument (today's orders, top-ranked stocks,
  the ETF orders and ranking, the sell tab's trade list, the Tracker's
  holdings) gets a **Trade** column with a "Buy / sell" link to the
  instrument's own page on UMushroom, opened in a new tab. The links are the
  platform's published pages: `/en/equity/<urlName>` from
  umushroom_universe_mapped.csv and `/en/etf/<url_name>` from the ETF
  universe file (both scraped from UMushroom's sitemaps), built by
  `universe.instrument_url`; an instrument the universe files do not know
  simply has no link. Carried in the report payload as `url`, so the
  website and the stored full reports show it too.
- 4 new tests (the URL patterns, the tables and escaping, the payload lookup
  and the Tracker).

## 4.23.0 (2026-11-02) -- nothing on the path to the orders is random
An audit of the model-to-orders path ("check that no allocation is randomly
done"), with the one real finding fixed behind a pre-declared head-to-head:
- VERIFIED: the orders are a deterministic function of the data and the
  config. Two runs on identical data gave identical orders, as did a
  single-threaded run (OMP_NUM_THREADS=1). Ridge, HRP, equal weight, the
  sector cap, the near-duplicate guard and the share sizing use no random
  numbers; the Monte Carlo, placebo and bootstrap use seeded generators and
  touch only the report.
- FOUND: the gradient-boosting learner's bin edges came from a SEEDED random
  subsample of the rows (scikit-learn does this above 200,000 rows; the live
  stock fit has 244,000). With `random_seed` 42, 43, 44 and 45 the 10 live
  picks shared only 7 names and the weights of the shared ones moved by up to
  7 points: 2-3 positions and about a third of the capital depended on an
  arbitrary constant. FIX: `gbm_bins: fixed` snaps the centred-rank features
  to the learner's own 255 levels before fitting (`model.quantize_ranks`), so
  scikit-learn uses those levels as the bins for every seed. Verified: seeds
  42 and 43 now give identical orders. Head-to-head (`--compare-gbm-bins
  equity:fixed,subsample`, rule declared first): fixed IC t +6.09, 6-week
  active t +3.03, win 75%, VaR 3.12% vs subsample +6.21, +2.54, 63%, 3.51%
  -> fixed wins and is the shipped setting (code default stays "subsample").
- HARDENED: ties in the ranking are broken alphabetically by a stable sort
  (`selection.rank_order`) in live orders and both backtests, instead of by
  whatever order an unstable sort left equal scores in.
- 12 new tests (tie-break, same inputs -> same orders, every learner
  reproducible with its seed, the quantiser, seed-independence above the
  200,000-row threshold, save/load, config, the CLI head-to-head).

## 4.22.0 (2026-11-02) -- the dashboard shell
The website is now an app frame instead of a row of tabs (shell.py):
- A sidebar with the sections in groups (Trade / Performance / Context /
  Runs), the six most recent runs with their status, Run now, and a
  system / light / dark switch that is remembered on the device; a sticky
  top bar with the section, the signal date and a search button; on a phone
  the sidebar is a drawer.
- The published site is ONE encrypted page: every section is a pane picked
  by the URL hash (#stocks), so switching is instant and the password is
  asked once. The old per-section files are contentless redirects into it,
  so bookmarks keep working. Full reports keep their own page with a slim
  "back to the dashboard" bar. The local website keeps a URL per section
  (and its run picker and Run now) inside the same frame, with the browser's
  cross-document view transition where supported.
- Cmd/Ctrl-K (or "/") opens a palette that finds any ticker, name or section
  on the page and jumps to the row, highlighting it. Every table sorts by a
  click on its header. Tables keep scrolling sideways inside their own
  container; nothing on the page scrolls sideways.
- No library, no font download, nothing fetched from outside: the frame is
  plain HTML, CSS and vanilla JavaScript on the report's own colour tokens,
  so the private site stays private and works offline.
- Tests updated for the one-page site (panes, redirects, the palette).

## 4.21.0 (2026-10-27) -- how the first live week went wrong, and the fixes
Eight days of live runs (20-27 Oct) were reconstructed from the published
site. The week's return was about -1.4% holding the first book for five
days (the validated rule) and +0.4% following the daily orders, against
+0.5% for the equal-weight stock universe and -0.7% for world equities; the
signal itself had a 2-sigma-bad week (its 10-16 picks -3.8% vs +0.2% for
the rest). Four defects in HOW it ran, all fixed here:
- STALE SIGNALS. The 06:20 UTC schedule ran before Yahoo published the
  previous day's European closes: every Tue-Fri run built signals on the
  close two days back (as_of lagged the run by two days; this morning 14 of
  18 holdings had no Monday price at 07:10 UTC). `data.check_freshness`
  now refuses data that does not reach the last trading day (exit code 3 =
  "not ready", `require_fresh_data: true`); the workflow runs at 08:40 and
  11:40 UTC and treats 3 as "keep the last good run, retry later", not as
  a failure. UMushroom fills at the previous close whenever the orders are
  entered, so a later run costs nothing.
- DAILY CHURN. The sell tab compared each run with the previous day's and
  ordered a full turnover every morning (27 Oct: sell 16, buy 13); the
  strategy was validated holding picks for 5 trading days. `rebalance.py`
  now defines periods on a fixed trading-day grid (weeks for 5 days): the
  first run of a period is the book, later runs say HOLD with the date of
  the next rebalance, and the orders page says which period its book is
  for. The local site's picker still allows any comparison.
- DRIFTING TRAINING SET. The rolling 10-year window starts one day later
  each day, moving every training-sample date (every 5th day) and every
  fold boundary: the 26 and 27 Oct runs shared 2 of 10 picks on IDENTICAL
  data, and the ETF walk-forward IC t jittered 1.9 <-> 3.1 day to day,
  flipping `auto` between core and model on 4 of 6 days. `history_start`
  (period "start:YYYY-MM-DD") fixes the first date; `auto_confirm_runs: 3`
  adds hysteresis (a ledger in .cache/selection_<sleeve>.csv; the raw rule
  must agree on 3 consecutive runs before the state switches).
- 72% SILVER. The 27 Oct ETF book put PHAG 25%, SLVR 22%, ISLN 25% -- the
  factor weights loading one commodity exposure through three near-identical
  lines. ETF sleeve back to `weighting: equal` (factor's edge was alpha t
  0.54) and the near-duplicate guard ON at 0.90: its head-to-head showed
  higher alpha and VaR 2.88% vs 2.86%, a tie that the rule now reads as a
  tie (`selection.VAR_TOLERANCE` = 0.05pp).
- `auto_trade_on: pass` -- an `auto` sleeve trades the model's picks only
  on a contest-window PASS (active t >= 2 AND win rate > 50%). Under the
  fixed history start the ETF picks had alpha -0.4%/yr, 6-week active t
  -0.83, win 51% (WARN) with every weighting, and the v4.19 "not FAIL" gate
  still traded them. The ETF sleeve now holds the core basket.
- Headline numbers under the fixed start (2016-10-03 .. 2026-10-26, 21-day
  folds): stocks (gbm, HRP) IC t +4.88, alpha +9.4%/yr (t 1.70), 6-week
  active +1.84% (t +2.83), 70.6% of windows won -- PASS. ETFs (ridge,
  guard 0.90) IC t +2.91, alpha -0.4%, 6-week active -0.47% (t -0.83),
  51% -- WARN -> core basket. Two consecutive runs on the same data now
  give identical orders.
- TRACKER READ "0 / 21 UP" (2026-11-02). The published Tracker tab scored
  the run that had just been published against the latest close -- which
  IS that run's prediction close -- so every holding showed +0.00% and the
  tiles said "Up / down 0 / 21", as if every pick had failed. It also
  tracked whatever ran that morning, not the book the team holds.
  `tracker.py`: `TrackerResult.newer` says whether any close after the
  prediction exists (the page now says "no close newer than <date> yet"
  instead of 0 / N); `track(until=...)` scores a book to the close the next
  book was bought at; `record()` scores every book of the weekly rule
  (`rebalance.books`) with one download, against IWDA.L (MSCI World) over
  the same closes. The static site's Tracker shows the book you hold
  (`web.tracked_run`: this period's first run, or the latest run when it
  opens a new period) and a "Week by week" table under it; the local site
  keeps its run picker and gains the same table.
- 29 new tests (freshness, fixed-start fetch and cache key, cadence known
  answers, books, sell-tab hold/rebalance, hysteresis, ledger, exit code 3
  in the CLI and the static site, VaR tolerance, the PASS gate, config,
  shipped config, the tracker's `until`/`newer`/record and the static
  site's held book, record and "no close yet").

## 4.20.0 (2026-10-19) -- data guard, near-duplicate guard, factor-risk weights, learner head-to-head
Root-cause fixes for the ETF sleeve's bad 2026-10-19 run, each behind a
pre-declared rule; the live strategy is unchanged until a rule says switch.
- FIX (data.py): Yahoo rate-limited 780 of 2,281 symbols in the bulk download
  (HTTP 429 per symbol, no exception) and the empty columns were CACHED for
  12 hours: the ETF sleeve ran on 1,039 of ~1,700 instruments and only 5 of
  its 13 core ETFs, the stock sleeve on 381 of 587. Now: symbols that come
  back empty are re-fetched in batches of 50 with a growing pause (3 rounds);
  a cached frame with empty columns is repaired in place on the next run; and
  a download still missing more than `max_missing_fraction` (10%) of the
  universe is refused and never cached (`UMUSHROOM_MAX_MISSING_FRACTION` for a
  deliberately degraded research run).
- FIX (data.py): a missing FX pair drops only the instruments that need it
  instead of aborting the run (carried over from the uncommitted 10-19 fix).
- Near-duplicate guard `max_pick_correlation` (selection.py): the 10-19 ETF
  picks held two MSCI Korea trackers (1-year return correlation 0.99) and
  three USD high-yield funds (0.6-0.92) -- eight slots, about four bets. A
  candidate whose trailing-252-day return correlation with a name already
  picked exceeds the threshold is skipped for the next-ranked name; causal,
  applied identically in the walk-forward and in live orders; near-duplicates
  are used last only if the universe cannot fill top_n. Off by default;
  `--compare-dedup SLEEVE:0.9,0.95` is the head-to-head (rule: keep >= 75%
  of alpha and do not raise weekly VaR).
- `weighting: factor` (new factor_risk.py): factor-risk parity. Factors are
  the sleeve's group averages (asset classes for ETFs, exchanges for stocks);
  each pick's betas and residual variance come from a trailing-year OLS; the
  factor COVARIANCE (the factors' correlation, not just their variances)
  prices every exposure; weights make the factor risk contributions match
  the budgets (equal by default; Roncalli & Weisang 2016), long-only, capped
  at 2x equal weight like every scheme and floored at 0.25x so every pick
  stays investable. A group whose average moves less than 0.05% a day (the
  money-market ETFs) is not a factor: as a regressor it blew betas up to
  +7.7 in the first live run. Falls back to equal weight with a warning
  when the book is exposed to fewer than two factors. Exposures and
  contributions are printed and shown on the website. A held-all basket
  (`selection: all`, or `auto` holding the core ETFs) is ALWAYS equal
  weight -- that is what the benchmark, the report's held curve and the
  Monte Carlo assume; hrp/factor apply to the model's picks.
- The purged walk-forward now scores the CONFIGURED weighting scheme
  (`weight_fn`), capped exactly as live orders are. Before 4.20 HRP could be
  traded but was never backtested. `--compare-weighting SLEEVE:equal,hrp,factor`.
- `model_type: ridge | mlp | gbm` (model.py) behind the unchanged
  RankingModel interface: a one-hidden-layer neural network (32 units) with
  early stopping judged on the latest tenth of the training DATES (purged, no
  random row split) and histogram gradient boosting with fixed rounds. Every
  hyper-parameter is a module constant fixed before any result was seen.
  Permutation importance (drop in training IC) replaces coefficients for the
  non-linear learners. `--compare-model SLEEVE:ridge,mlp,gbm` (rule: IC t >= 2,
  then the highest contest-window active t; ties to the simpler model). Why
  no LSTM/CNN: see model.py's docstring.
- One `_walk_forward_kwargs` builder feeds the headline backtest, the
  contest-window evaluation, every head-to-head and the validation suite, so
  the learner, caps, guard and weighting can never differ between them.
- cap_weights moved to portfolio_construction.py (shared) and no longer
  returns NaN when the uncapped names all have zero weight.
- Website: ETFs tab shows the learner, weighting scheme and guard in force;
  Risk tab shows the factor-risk decomposition of each sleeve sized by it.
- Learner rule: a learner other than the current one must also show a
  contest-window active t >= 2 (window.py's PASS bar) to be adopted -- added
  after the first ETF comparison produced a "winner" at t = +0.34 vs -0.18.
  `cli.skill_verdict` is a pure, unit-tested function.
- Top-level `weighting:` sets the stock sleeve's scheme from config (the
  GitHub run passes no --weighting flag).
- 58 new tests (batched re-fetch, cache repair, guard thresholds, twin guard
  in backtest and live, closed-form factor-parity answers, causality,
  learner round-trips and date-purged early stopping, config, CLI flags,
  the learner rule).

RESULTS of the pre-declared head-to-heads on the repaired data
(2026-10-19, 568 stocks / 1,590 ETFs, 30-day test folds, ridge unless stated):
- Data repair: 780 empty symbols re-fetched in one 26-second round; the
  stock sleeve went from 381 to 568 instruments, the ETF sleeve from 1,039
  (5 core) to 1,590 (12 core).
- ETF sleeve on the FULL universe: walk-forward IC t +1.93 (21-day folds) /
  +3.21 (30-day folds), active return ~0, alpha -0.4%/yr, 6-week win 56%:
  `auto` holds the core basket. The IC t of +2.71 reported on the damaged
  cache was a property of the missing 40%, not of the model.
- ETF weighting (final, with the factor-vol and weight floors): factor
  +2.7%/yr (t 0.54), win 62%, VaR 2.85% vs equal -0.4%, 56%, 2.86% ->
  ADOPTED (`weighting: factor`): keeps the alpha, does not raise VaR; hrp
  -0.9%, VaR 1.98% lost alpha. (The first run, before the floors, showed
  +9.1% -- partly the degenerate exposures the floors now prevent.)
- ETF learner: mlp 6-week t +0.34, ridge -0.18, gbm -0.30 -> no reliable
  edge, ridge kept. ETF guard 0.90/0.95: no improvement -> off.
- ETF validation suite (informational, run with gbm): placebo FAIL (active
  -0.08%/period beat 5/20 shuffled runs, p = 0.76), deciles ordered but
  +0.09%/period top-minus-bottom, 8/9 years, markets mixed, 488 names
  picked. The ranking is faintly real and does not beat holding every ETF:
  `auto` holding the core basket is the right outcome, not a failure.
- Stock learner: gbm 6-week active t +3.95, alpha +24.2%/yr (t 3.92), win
  71%, VaR 3.99% vs ridge +2.07, +9.2% (t 1.91), 69%, 3.16% -> rule says gbm.
  Validation suite with gbm: 7/7 PASS (placebo beat 20/20 shuffled runs,
  p = 0.048; deciles ordered, rank corr 0.89; IC positive 9/9 years; LSE,
  SIX and US all positive; active +0.44%/period at delay 0, +0.26% at a
  1-day delay; 526 names picked, max 10%) -> ADOPTED (`model_type: gbm`).
- Stock weighting: hrp +11.9%/yr (t 2.65), VaR 2.54% vs equal +9.2%, 3.16%
  -> ADOPTED (`weighting: hrp`); factor +6.4%, VaR 3.66% (too coarse for
  stocks). Stock guard: VaR up 0.05-0.07pp -> off.

## 4.19.0 (2026-10-12) -- ETF model over the whole UMushroom ETF universe
- ETF sleeve is now a model sleeve: `feature_set: etf` (13 trend /
  asset-class momentum factors), ranked within UMushroom's asset classes,
  at most 3 picks per class, over every ETF UMushroom lists that Yahoo can
  price (2,047 of 2,261; leveraged/inverse and crypto ETPs excluded). The 13
  ETFs traded so far stay in as `core_tickers`.
- New `umushroom/catalogue.py`: scrapes UMushroom's public ETF sitemap +
  per-ETF API record into `umushroom_etf_catalogue.json` (2,261 ETFs; three
  core ETFs the sitemap omits are fetched by slug).
- New `umushroom/etf_universe.py`: maps each ISIN to a Yahoo Finance line
  (venue preference for a GBP account, already-traded lines kept, ticker
  guesses only with a matching fund name, every line verified for currency,
  history and freshness) and writes `umushroom_etf_universe.csv` with a
  status and reason per ETF. `--config` keeps the config's own lines.
- New `umushroom/window.py` + `contest_window_days: 30`: every sleeve's
  horizon must fit inside Stage 1, and every sleeve is judged on rolling
  30-day out-of-sample windows (win rate vs. benchmark, active t on
  non-overlapping windows, PASS/WARN/FAIL). The evaluation runs on test
  folds exactly one contest window long so windows never straddle a
  purge/embargo gap.
- `selection: auto`: trade the model only while IC t >= 2 AND the window
  verdict is not FAIL; else hold the core basket. The decision and its
  reason are printed and shown on the website.
- `--compare-horizon SLEEVE:5,10,21`: pre-declared horizon head-to-head
  judged by the contest-window rule. Result 2026-10-12 (ETFs): 5d IC t
  +3.33 / 6-week win 57%; 10d IC t +4.06 / 49% (winner, adopted); 21d no
  skill. Under `auto` the sleeve therefore holds the core basket today.
- Website: new **ETFs** tab (orders with UMushroom ticker and currency,
  top-25 ETF ranking with asset class, universe build status, the
  contest-window evidence with verdict badge, the selection rule in force);
  ETF tile on Overview.
- `UMUSHROOM_PRICE_SNAPSHOT`: replay a saved price snapshot instead of
  Yahoo for reproducible offline runs (`data.SnapshotFetcher`).
- New workflow `.github/workflows/etf-universe.yml` rebuilds and commits
  the universe on demand.
- Data: `.MU`, `.LU`, `.IL` suffixes recognised. Per-sleeve horizons may
  exceed `purged_purge_days` (the purge is lifted to the horizon).
- 60 new tests (window, universe mapping, catalogue scraper, ETF features
  incl. planted-signal recovery, config, snapshot replay, ETF sleeve
  end-to-end, website tabs).
- CI fix: `pytest` (the bare entry point GitHub Actions runs) did not put
  the repo root on `sys.path`, so the 6 tests that share helpers via
  `from tests.test_web import ...` failed with `No module named 'tests'`
  on every CI run since 4.18 (`python -m pytest`, used locally, hid it).
  `pythonpath = ["."]` in the pytest config makes both entry points agree.

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
