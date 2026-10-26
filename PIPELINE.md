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
| `interim/ibes_announcements_*.parquet` | IBES announcements with consensus and SUE |
| `interim/ibes_links_*.parquet` | Candidate IBES tickers for each call |
| `raw/ciq_transcripts_<universe>_<rule>_<YYYY-MM>.parquet` | Monthly WRDS pulls (cache) |

Read the final file with `pd.read_parquet(path)`. Filter on `matched` and/or `sue.notna()` for the estimation sample.

---

## 2. The process

```
  CIQ transcripts (WRDS)           I/B/E/S (WRDS)                   Link tables (WRDS)
          |                              |                                 |
  transcripts:YYYY-MM              ibes                             links
  earnings calls, one version      announcements, actual EPS,       CIQ company -> candidate
  per call, full text              consensus, price, SUE            IBES tickers (4 routes)
          |                              |                                 |
  transcripts_clean                      |                                 |
  period, length, duplicates             |                                 |
          \______________________________|_________________________________/
                                         |
                                       merge
                       match call <-> announcement, event date, ranks
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

**Source:** I/B/E/S on WRDS: the US file, plus the international file for `global`. All tables are the **unadjusted** versions (`actu_`, `statsumu_`, `detu_`, `actpsumu_`), so actual, consensus and price are all on the same per-share basis on the announcement date. Split-adjusted values are rounded and distort small surprises.

| Decision | What we do | Why |
|---|---|---|
| Unit of observation | One row per **announcement** (IBES ticker x announcement date) | A release can carry several actuals: a Q4 release also reports the annual figure. |
| Which actual | **Quarterly** EPS if present, else semi-annual, else annual | The consensus and SUE are quarterly. Semi-annual reporters (mostly Europe and Asia) are kept, but get no SUE. |
| Measure | IBES "street" EPS | It is on the same basis analysts forecast, so actual and consensus are comparable (not GAAP). |
| Consensus | Default `summary`: the **last monthly IBES consensus before the announcement date** (current quarter, `fpi = 6`). Option `detail`: rebuilt from individual analysts, taking each analyst's latest estimate within the last 90 days | The summary is IBES's standard. It is a median of about 19 days old at the release. The detail version is fresher but covers fewer analysts. The two SUEs have a rank correlation of 0.91. |
| Price | Last IBES price snapshot before the announcement (within 45 days) | Scales the surprise. |
| SUE | `(actual - consensus) / price`, with the median consensus by default (`consensus_stat`) | Standard definition; the baseline the LLM signal must beat. |
| SUE left missing | Price below $1 (`min_price`), currencies of actual / consensus / price differ, or no consensus | Penny-stock prices make SUE explode. Mixed currencies are not comparable. |

### Step 3. Linking companies (`links`)

CIQ has no IBES identifier, so each call gets **candidate IBES tickers** from four routes, each checked as of the call date:

| Route | Path | Mainly finds |
|---|---|---|
| CUSIP | CIQ CUSIP (first 8 characters) to IBES CUSIP | US and Canada (finds about 98% of US calls) |
| CCM | CIQ, then Compustat gvkey, then CRSP permno, then the WRDS IBES-CRSP link | US-listed stocks |
| Compustat `ibtic` | gvkey to Compustat's IBES ticker field | US and non-US |
| SEDOL (`global` only) | gvkey to Compustat SEDOL, matched to the IBES international identifier | Non-US |

When two routes both find a ticker, they agree 94-98% of the time. About 99% of calls get at least one candidate. The announcement date decides which candidate is correct (Step 4).

### Step 4. Matching and final dataset (`merge`)

| Decision | What we do | Why |
|---|---|---|
| Match window | Accept an announcement if **call date minus announcement date is between -1 and +3 days** | 70% of calls are on the announcement day and 25% the next day (release after the close, call the next morning). The -1 / +2 / +3 days absorb time-zone differences. |
| Several candidates | Prefer: closest date (day-after over day-before), then the IBES record for the company's home market (home listing over a US ADR, or the reverse for US firms), then the strongest route | 0.4% of US matches have a tie; about 10% globally (ADRs). |
| One call per announcement | If two calls match the same announcement, keep the closer one, then the longer transcript | Leftover CIQ duplicates. |
| Unmatched calls | **Kept**, with `matched = False` and empty IBES fields | Lets you see coverage, and keeps the text for uses that do not need IBES. |
| Event date | The **release day if released by 16:00 ET on a trading day, otherwise the next trading day** (NYSE calendar). US companies only. | The paper's rule. This is the day of the return you would explain. It equals the call date only about 64% of the time. |
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
| with a SUE | 2,685 (89.6%) |

**Why some rows have no SUE:** 131 calls have no IBES announcement in the window. Among matched calls, the gaps overlap, so the counts don't add up: 119 have no consensus (not covered by analysts that quarter), 107 have no reported actual, 106 have no price, and 47 have a price below $1.

**Typical values:** a transcript has a median of 6.3k words. A consensus has a median of 6 analysts and is a median of 19 days old at the release. The chosen transcript appeared a median of about 1.3 hours after the call, and 98% within a day. The event date is the call date 64% of the time and the next trading day 35% of the time.

**Columns of the final file (53)**

| Group | Columns |
|---|---|
| Call identifiers | `keydevid` (CIQ event), `transcriptid` (chosen version), `companyid`, `companyname`, `headline`, `country` |
| Call timing | `call_date`, `call_time_utc`, `call_datetime_utc` |
| Transcript version | `presentation_type` (Preliminary / Final), `collection_type`, `created_utc`, `transcript_lag_days`, `n_versions`, `first_created_utc` |
| Transcript text | `full_text`, `presentation_text`, `qa_text` (set `output.include_text: false` to drop them) |
| Transcript stats | `n_words`, `n_words_presentation`, `n_words_qa`, `n_components`, `n_questions`, `n_analysts`, `audio_sec` |
| Link | `matched`, `ticker` (IBES), `link_route`, `routes`, `ibes_file` (US / Intl), `match_gap_days`, `ticker_tie` |
| Announcement | `anndats`, `anntims` (ET), `pdicity` (QTR / SAN / ANN), `pends` (fiscal period end) |
| Earnings | `actual`, `curr_act`, `consensus`, `consensus_median`, `consensus_mean`, `consensus_std`, `n_estimates`, `consensus_date`, `consensus_age_days` |
| Surprise | `price`, `price_date`, `surprise` (actual - consensus), `sue`, `sue_pct`, `event_quarter` |
| Event | `event_date`, `transcript_by_event_close` |

A quarter of US calls with text is about 120 MB as parquet.

---

## 4. Known limitations

- **IBES on WRDS lags by about 7 weeks** (monthly refresh; currently through 2026-08-20). Calls after that date cannot match, and the run prints a warning. This suits historical work but not the paper's live design.
- **The latest month can be incomplete.** CIQ keeps adding transcripts for recent calls. Re-pull with `doit forget transcripts:YYYY-MM`.
- **Some misses are data disagreements, not bugs.** Examples: an IBES date set by a preliminary release a week before the call (Skyworks, Q4 2025), or a company whose identifiers changed after a merger (fuboTV). We do not fuzzy-match on names or tickers, to avoid wrong matches.
- **Non-US (`universe: global`):** about 90% of calls match, but only about half get a SUE (semi-annual reporters, thin analyst coverage), and `event_date` is left empty (NYSE calendar only).
- **The `detail` consensus is a simple rebuild.** It uses a fixed 90-day window and does not yet use IBES's stopped-estimate file.
- **Returns are not included yet.** Abnormal returns on `event_date` (CRSP or another price source) are the next step.
