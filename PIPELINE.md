# Data pipeline: CIQ earnings-call transcripts + I/B/E/S earnings surprise

The pipeline builds the dataset the LLM work starts from: **one row per earnings call**, with the transcript text as it was available at the time, matched to its I/B/E/S earnings announcement (actual EPS, the analyst consensus before the release, the earnings surprise / SUE, and the trading day the market reacts on).

It runs on WRDS through [doit](https://pydoit.org). The tasks are in `dodo.py`, the code is in `src/llm_based_earnings_surprise_ntam/pipeline/`, and the settings are in `config.yaml`. The data checks behind each decision are in `notebooks/ibes_transcripts_link_eda.ipynb` (HTML version: `docs/ibes_transcripts_link_eda.html`).

---

## 1. How to run it

**One-time setup**
1. You need a WRDS account with access to Capital IQ Transcripts, I/B/E/S, Compustat and CRSP. Save your credentials in a `.pgpass` file; the first `wrds.Connection()` in Python offers to create it.
2. Install the environment from the repo root: `uv sync`

**Run**
```bash
uv run doit                                      # everything, using config.yaml (default: Q4 2025, US firms)
uv run doit start=2024-01-01 end=2024-12-31      # choose the call-date period (inclusive)
uv run doit universe=global                      # all countries instead of US only
uv run doit config=my_config.yaml                # use a different config file
```
`start`, `end` and `universe` on the command line override `config.yaml`. To change anything else, edit `config.yaml` (each option is commented there).

**Useful commands**
```bash
uv run doit list                       # list the tasks
uv run doit info merge                 # explain why a task would (or would not) rerun
uv run doit forget transcripts:2025-12 # re-pull a month that was still incomplete when first pulled
uv run doit clean                      # delete the outputs of the current period
```
The run is incremental. Each task reruns only when its inputs or its part of the config change. Monthly transcript pulls are cached in `data/raw` and reused when you change the period. One quarter of US calls takes about 30 seconds.

**Outputs** (in `data/`, which is not tracked by git)

| File | Contents |
|---|---|
| `processed/earnings_calls_ibes_<start>_<end>_<universe>.parquet` | **The final dataset** (one row per call) |
| `processed/..._report.json` | Match statistics for the run, plus the config used |
| `interim/calls_*.parquet`, `interim/transcript_text_*.parquet` | Cleaned transcript metadata and text |
| `interim/ibes_announcements_*.parquet` | IBES announcements with actual EPS and consensus |
| `interim/ibes_links_*.parquet` | Candidate IBES tickers for each call |
| `interim/matched_*.parquet` | Calls matched to IBES, with event date (before SUE) |
| `interim/prices_*.parquet` | Close on the trading day before each event date |
| `raw/ciq_transcripts_<universe>_<rule>_<YYYY-MM>.parquet` | Monthly WRDS pulls (cache) |

Read the final file with `pd.read_parquet(path)`. Filter on `matched` and/or `sue.notna()` for the estimation sample.

---

## 2. The process

```
  CIQ transcripts (WRDS)           I/B/E/S (WRDS)                   Link tables (WRDS)
          |                              |                                 |
  transcripts:YYYY-MM              ibes                             links
  earnings calls, one version      announcements, actual EPS,       CIQ company -> candidate
  per call, full text              pre-release consensus            IBES tickers (4 routes)
          |                              |                                 |
  transcripts_clean                      |                                 |
  period, length, duplicates             |                                 |
          \______________________________|_________________________________/
                                         |
                                       merge
                          match call <-> announcement, event date
                                         |
                                      prices
                        close on the trading day before the event date
                                         |
                                       final
                          SUE, within-quarter ranks, transcript text
                                         |
                         earnings_calls_ibes_<period>.parquet
```

### Step 1. Transcripts (`transcripts:YYYY-MM`, `transcripts_clean`)

**Source:** S&P Capital IQ Transcripts on WRDS (`ciq_transcripts`).

| Decision | What we do | Why |
|---|---|---|
| Which events | **Earnings calls only** (CIQ event type 48) | Other event types (M&A calls, investor days, conferences) are out of scope. |
| Which countries | `universe: us` keeps US-headquartered companies (CIQ company country). `global` keeps all. | The paper's sample is US. Outside the US, coverage of the surprise is much thinner. |
| Which version of the transcript | **Point-in-time (`version_rule: pit`): the first version created** for each call | A call has 2-4 versions (Preliminary, then Final edits, sometimes weeks later). A backtest must use the text that existed at the time. In Q4 2025 the chosen version appeared a median of about 1.3 hours after the call, and 98% within a day. `version_rule: final` picks the cleanest late version instead. |
| Text | Full text plus separate **prepared remarks** and **Q&A** sections. Each speaker turn is prefixed with name and role ("Jane Doe (Executive): ...") | The paper uses the full transcript. The sections allow separate prompts or features. |
| Period | Keep calls with call date in `[start, end]` | Set in config / on the command line. |
| Short transcripts | **Drop** if under 500 words (`min_words`) | Empty or truncated text. |
| Duplicate events | **Drop**: when CIQ lists the same company and date under two event IDs, keep the longer transcript | About 1% of calls are duplicated in CIQ. |
| Late transcripts | **Kept** by default. `max_lag_days: N` drops calls whose chosen version appeared more than N days after the call. | Old history was often backfilled years later. Whether to drop it depends on the backtest. |

### Step 2. I/B/E/S (`ibes`)

**Source:** I/B/E/S on WRDS: the US file, plus the international file for `global`. All tables are the **unadjusted** versions (`actu_`, `statsumu_`, `detu_`, `stopu_`), so actual and consensus are on the same per-share basis on the announcement date. Split-adjusted values are rounded and distort small surprises.

The defaults follow Koijen & Levy: the **mean** consensus **right before the release**, scaled by the **price just before the announcement**.

| Decision | What we do | Why |
|---|---|---|
| Unit of observation | One row per **announcement** (IBES ticker x announcement date) | A release can carry several actuals: a Q4 release also reports the annual figure. |
| Which actual | **Quarterly** EPS if present, else semi-annual, else annual | The consensus and SUE are quarterly. Semi-annual reporters (mostly Europe and Asia) are kept, but get no SUE. |
| Measure | IBES "street" EPS | It is on the same basis analysts forecast, so actual and consensus are comparable (not GAAP). |
| Consensus | Default `detail`: built from **individual analyst estimates** for the quarter. For each analyst, take the latest estimate that was (1) **made before the release** (an earlier day, or the release day before the release time), (2) **made or last confirmed within 90 days** (`detail_window_days`), and (3) **not stopped** by the broker before the release. Then take the mean (`consensus_stat`). Option `summary`: the last monthly IBES consensus snapshot before the release day. | The paper uses the consensus right before the release. The monthly snapshot is a median of about 20 days old and misses late revisions; in Q4 2025, using it instead changes beat/miss for about 9.5% of events. The detail rule drops estimates nobody has confirmed for 90+ days, which the monthly snapshot keeps, so it covers slightly fewer events (mostly 1-2-analyst firms). |
| Consensus currency | Dropped if the estimates are in a different currency than the actual | Not comparable. |

### Step 3. Linking companies (`links`)

CIQ has no IBES identifier, so each call gets **candidate IBES tickers** from four routes, each checked as of the call date:

| Route | Path | Mainly finds |
|---|---|---|
| CUSIP | CIQ CUSIP (first 8 characters) to IBES CUSIP | US and Canada (finds about 98% of US calls) |
| CCM | CIQ, then Compustat gvkey, then CRSP permno, then the WRDS IBES-CRSP link | US-listed stocks |
| Compustat `ibtic` | gvkey to Compustat's IBES ticker field | US and non-US |
| SEDOL (`global` only) | gvkey to Compustat SEDOL, matched to the IBES international identifier | Non-US |

When two routes both find a ticker, they agree 94-98% of the time. About 99% of calls get at least one candidate. The announcement date decides which candidate is correct (Step 4).

### Step 4. Matching, price and final dataset (`merge`, `prices`, `final`)

| Decision | What we do | Why |
|---|---|---|
| Match window | Accept an announcement if **call date minus announcement date is between -1 and +3 days** | 70% of calls are on the announcement day and 25% the next day (release after the close, call the next morning). The -1 / +2 / +3 days absorb time-zone differences. |
| Several candidates | Prefer: closest date (day-after over day-before), then the IBES record for the company's home market (home listing over a US ADR, or the reverse for US firms), then the strongest route | 0.4% of US matches have a tie; about 10% globally (ADRs). |
| One call per announcement | If two calls match the same announcement, keep the closer one, then the longer transcript | Leftover CIQ duplicates. |
| Unmatched calls | **Kept**, with `matched = False` and empty IBES fields | Lets you see coverage, and keeps the text for uses that do not need IBES. |
| Event date | The **release day if released by 16:00 ET on a trading day, otherwise the next trading day** (NYSE calendar). US companies only. | The paper's rule. This is the day of the return you would explain. It equals the call date only about 64% of the time. |
| Price | **Close on the trading day before the event date** (the t-1 close), from Compustat daily prices (`comp.secd`), primary US share line, unadjusted. If unavailable, fall back to the last IBES price snapshot. `price_source` records which was used. `ibes.price_source: ibes` uses the IBES price throughout. | The paper scales by the price just before the announcement. For a release after the close this is that day's close; for a pre-market release, the previous day's close. In Q4 2025, 99.8% of SUEs use the t-1 close. |
| SUE | `(actual - consensus) / price` | The paper's surprise; the baseline the LLM signal must beat. |
| SUE left missing | No consensus, price below $1 (`min_price`), or the price is in a different currency than the actual | Penny-stock prices make SUE explode. |
| Transcript availability | `transcript_by_event_close`: was the chosen version created by 16:00 ET on the event date? | 90% are. Filter on it if the text must be available by the close. |
| SUE rank | `sue_pct`: percentile of SUE within the event quarter | The paper ranks surprises within the quarter. |

---

## 3. What the data looks like (Q4 2025, US, default config)

**Flow of calls**

| Stage | Calls |
|---|---|
| US earnings calls in CIQ, Oct-Dec 2025 (one version each) | 3,009 |
| after dropping transcripts < 500 words | 3,008 |
| after dropping duplicate CIQ events | 2,998 |
| after keeping one call per IBES announcement | **2,995** (final rows) |
| matched to an IBES announcement | 2,864 (95.6%) |
| with a SUE | 2,602 (86.9%) |

**Why some rows have no SUE:** 131 calls have no IBES announcement in the window. Among matched calls, the gaps overlap, so the counts don't add up: 210 have no consensus (no analyst estimate made or confirmed in the 90 days before the release), 107 have no reported actual, 73 have a price below $1, and 2 have no price.

**Typical values:** a transcript has a median of 6.3k words. A consensus has a median of 5 analysts, and the most recent estimate in it was made a median of 24 days before the release (older estimates count if they were confirmed since). The chosen transcript appeared a median of about 1.3 hours after the call, and 98% within a day. The event date is the call date 64% of the time and the next trading day 35% of the time.

**Columns of the final file (57)**

| Group | Columns |
|---|---|
| Call identifiers | `keydevid` (CIQ event), `transcriptid` (chosen version), `companyid`, `gvkey` (Compustat), `companyname`, `headline`, `country` |
| Call timing | `call_date`, `call_time_utc`, `call_datetime_utc` |
| Transcript version | `presentation_type` (Preliminary / Final), `collection_type`, `created_utc`, `transcript_lag_days`, `n_versions`, `first_created_utc` |
| Transcript text | `full_text`, `presentation_text`, `qa_text` (set `output.include_text: false` to drop them) |
| Transcript stats | `n_words`, `n_words_presentation`, `n_words_qa`, `n_components`, `n_questions`, `n_analysts`, `audio_sec` |
| Link | `matched`, `ticker` (IBES), `link_route`, `routes`, `ibes_file` (US / Intl), `match_gap_days`, `ticker_tie` |
| Announcement | `anndats`, `anntims` (ET), `pdicity` (QTR / SAN / ANN), `pends` (fiscal period end) |
| Earnings | `actual`, `curr_act`, `consensus` (the configured statistic), `consensus_mean`, `consensus_median`, `consensus_std`, `n_estimates`, `consensus_date` (most recent estimate, or the snapshot date for `summary`), `consensus_age_days` |
| Surprise | `surprise` (actual - consensus), `price`, `price_date`, `price_source` (t-1 close or IBES snapshot), `price_ibes`, `price_ibes_date`, `sue`, `sue_pct`, `event_quarter` |
| Event | `event_date`, `transcript_by_event_close` |

A quarter of US calls with text is about 120 MB as parquet.

---

## 4. Known limitations

- **IBES on WRDS lags by about 7 weeks** (monthly refresh; currently through 2026-08-20). Calls after that date cannot match, and the run prints a warning. This suits historical work but not the paper's live design.
- **The latest month can be incomplete.** CIQ keeps adding transcripts for recent calls. Re-pull with `doit forget transcripts:YYYY-MM`.
- **Some misses are data disagreements, not bugs.** Examples: an IBES date set by a preliminary release a week before the call (Skyworks, Q4 2025), or a company whose identifiers changed after a merger (fuboTV). We do not fuzzy-match on names or tickers, to avoid wrong matches.
- **Non-US (`universe: global`):** about 90% of calls match, but only about half get a SUE (semi-annual reporters, thin analyst coverage), and `event_date` is left empty (NYSE calendar only).
- **The consensus vendor differs from the paper.** The paper's main source is Benzinga; IBES is the paper's own robustness source. The consensus rule here is as close as IBES allows, but small differences remain (which analysts count, and how long an estimate stays valid).
- **Returns are not included yet.** Abnormal returns on `event_date` (e.g. from Compustat daily prices, as used for the SUE price) are the next step.

---

## 5. Extension: event-date abnormal return and explanatory-power baseline check

Two tasks added on top of the pipeline above (`returns`, `sample`). They produce the dependent variable the LLM work needs — the abnormal return around the announcement — and check that the dataset is aligned before any LLM money is spent. This supersedes the last bullet of section 4 ("Returns are not included yet").

Nothing in the existing pipeline code changed: the steps are two new modules (`pipeline/returns.py`, `pipeline/sample.py`) wired in through `dodo.py`, with their settings in the new `returns:` and `sample:` sections of `config.yaml`.

### How to run it

`doit` now runs through `sample` by default, so the usual command does everything:

```bash
uv run doit                      # transcripts -> ... -> final -> returns -> sample
uv run doit returns              # just the returns step
uv run doit sample               # just the sample step (re-runs `final` first if needed)
```

Both steps are incremental like the rest: `returns` reruns when `matched`/`prices` or the `returns:` config change; `sample` reruns when the returns file or the `sample:` config change, **or when the columns it adds are missing from the final parquet** (which happens whenever `final` rewrites it).

### Step 5. Abnormal return (`returns`)

**Source:** CRSP daily v2 (`crsp.dsf_v2`), column `dlyret` — total return, dividends included.

| Decision | What we do | Why |
|---|---|---|
| Return table | `crsp.dsf_v2`, not `crsp.dsf` | The classic CRSP daily files (`crsp.dsf`, `crsp.dsi`, `crsp.dsix`) **stop at 2024-12-31** on WRDS and cannot cover recent quarters. The v2 file runs to 2025-12-31. |
| Market proxy | **VTI** (CRSP permno 90878), `returns.market_permno` | The ETF Koijen & Levy use. CRSP's own value-weighted index (`vwretd`) would be the usual substitute, but its daily file ends in 2024. |
| Abnormal return | `r_abn = ret − mkt_ret` on `event_date` | The paper's market-adjusted return, a one-day window. Using CRSP's `ret` rather than differencing the unadjusted Compustat closes keeps dividends and splits handled. |
| gvkey → permno | `crsp.ccmxpf_lnkhist`, primary links only (`linktype LU/LC`, `linkprim P/C`), valid **on the event date**; `P` preferred, lowest permno breaks ties | Same link the `ccm` route uses, but it is re-derived here so `links.py` is untouched. `n_permno_candidates` records how many permnos the call had. |
| Which calls | Matched calls with an `event_date` and a `gvkey` | Without an event date there is no day to measure, and non-US calls have no event date (section 4). |
| Sanity check | Compares `close / price − 1` (event-day close over the t−1 close already used for the SUE) with `dlyret`, and prints how many differ by more than 1pp | Catches a misaligned event date, a wrong permno, or a split/dividend on the event day. |

Output: `interim/returns_<period>.parquet` — `keydevid, permno, event_date, ret, mkt_ret, r_abn, n_permno_candidates`.

### Step 6. Estimation sample and baseline check (`sample`)

Adds seven columns **to the existing final parquet, in place** (row order and existing columns untouched; a rerun replaces these columns rather than duplicating them):

| Column | Meaning |
|---|---|
| `permno`, `ret`, `mkt_ret`, `r_abn` | from the returns step |
| `in_sample` | passes every filter below — the rows to estimate on |
| `r_abn_pct` | percentile rank of `r_abn` within `event_quarter`, over `in_sample` rows only |
| `sue_pct_sample` | percentile rank of `sue` within `event_quarter`, **re-ranked over `in_sample` rows** (the pipeline's own `sue_pct` is ranked over all rows of the file, so it is not the right regressor once rows are dropped) |

Sample filters, printed as a chain so you can see where rows go:

| Filter | Setting |
|---|---|
| `matched` | fixed |
| `sue` present | fixed |
| `r_abn` present | fixed |
| `transcript_by_event_close` | `sample.require_transcript_by_event_close` (default `false`) |

**The baseline check.** The task then runs the surprise-only regression that the LLM has to beat:

```
PRank(r_abn) = a + b * PRank(sue)
```

and prints `N`, `b`, its t-statistic and the R². Koijen & Levy report **b = 0.219 and R² = 5.0%** for this regression (Table 2, column 1). The R² is a single check on the event date, the ticker match, the consensus and the returns at once: if any of them is misaligned, it collapses toward zero.

When the R² falls below `sample.baseline_r2_warn` (default 0.05) the task prints a **WARNING**. Treat it as "look before going further", not as a hard failure — a value a little under 5% is expected, since this sample is not the paper's (I/B/E/S instead of Benzinga, a different market proxy, and about 2,600 calls against their 1,849).

### Current run (Q4 2025, US, default config)

**Where the rows go.** The two steps print two different chains, so their counts do not line up row for row:

| Rows | Filter | Step | Note |
|---|---|---|---|
| 2,995 | all calls in the final file | — | unmatched calls included |
| 2,865 | `matched` | `sample` | 95.6% of calls. `matched` means the call was successfully paired with an IBES earnings announcement (step 4). |
| 2,864 | `matched` & `event_date` & `gvkey` | `returns` | the one call dropped here has no `gvkey`, so there is no security to price. No US call lacks an event date; that only bites with `universe: global`. |
| 2,827 | of those, CRSP had a return | `returns` | 98.7%; a permno was found for 98.8% |
| 2,603 | `matched` & `sue` | `sample` | a SUE also needs a consensus and a price >= `ibes.min_price` |
| **2,586** | **`in_sample`** — also has `r_abn` | `sample` | the 17 calls lost here have a SUE but no permno (firms absent from CRSP: recent IPOs, foreign private issuers, trusts) |

The two chains overlap but are not nested: `r_abn` exists for 2,827 matched calls, more than the 2,603 with a SUE, because a return only needs a permno while a SUE also needs a consensus and a valid price.

| Other checks | |
|---|---|
| Price cross-check beyond 1pp | 2 of 2,826 |
| Abnormal return | mean 0.07%, SD 11.0% (paper's Table 1: SD 11%) |
| **Baseline regression** | b = 0.217 (t = 11.3), **R² = 4.7%** on N = 2,586 |

The slope is within 0.002 of the paper's 0.219 and the R² within 0.3pp of its 5.0%, on a different quarter and a different consensus vendor — the dataset lines up.

### Known limitations of the extension

- **CRSP daily v2 ends 2025-12-31** on WRDS. Calls with a later event date get no return; the task warns when that happens.
- **Non-US calls get no return**, because they have no `event_date` (NYSE calendar only, section 4).
- **One-day window only.** The project's longer horizons (1, 4, 12 weeks) would need cumulative returns, which these tasks do not compute.
- **VTI is an ETF, not an index.** Its return includes a tracking difference and its own dividend timing. The effect on a one-day market-adjusted return is small, but it is not identical to a value-weighted market return.
- **The `price` column is still the t−1 close** used for the SUE; the event-day close is not stored as a column, only used inside the cross-check.
