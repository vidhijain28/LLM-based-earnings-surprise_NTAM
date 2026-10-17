# Data Dictionary — Earnings Announcement Event Panel

**Issue:** #2 · **Status:** Proposed implementation schema grounded in Koijen & Levy (2026).  
**Important:** Field names and types below are **our design**, not a schema supplied by the authors. `Required` refers to replication need, not provider availability.

## A. Identity and event timing

| Field | Type | Definition / rules | Source | Required |
|---|---|---|---|---|
| `event_id` | string | Unique immutable event key; stable security ID + fiscal period + call timestamp | Constructed | Yes |
| `security_id` | string | Stable firm/security identifier; choose and document mapping | Identifier mapping | Yes |
| `ticker_at_event` | string | Historical ticker at announcement, not current ticker | Event/calendar | Yes |
| `fiscal_period` | string | Reported fiscal quarter/year; avoids mixing reports | Event/calendar | Yes |
| `expected_event_date` | date | Expected date from pre-event calendar snapshot | Benzinga | Yes |
| `calendar_snapshot_at` | timestamp UTC | When expected event information was recorded | Ingestion | Yes |
| `event_datetime_et` | timestamp ET | Earnings-event timestamp for Equation 5; confirm which event time | Benzinga / source validation | Yes |
| `call_datetime_et` | timestamp ET | Conference call start, retained separately | Transcript metadata | Yes |
| `trading_event_date` | date | Same day if trading day and event time <= 16:00 ET; else next trading day | Derived Eq. 5 | Yes |
| `reporting_week` | integer | Paper's week-of-season grouping; verify start dates | Derived | For Table 1 |
| `ff12_industry` | categorical | Fama–French 12 industry category using contemporaneous classification | Historical industry mapping | Yes |
| `sample_split` | enum | `early_train` / `late_eval` based on median event date | Derived §3.3.3 | Yes |

## B. Analyst estimates and stock prices

| Field | Type | Definition / rules | Source | Required |
|---|---|---|---|---|
| `reported_eps` | float | EPS reported for matching fiscal period and EPS basis | Benzinga | Yes |
| `reported_eps_available_at` | timestamp UTC | First availability of actual EPS | Provider metadata | Yes |
| `consensus_eps_mean` | float | Mean analyst EPS forecast formed strictly before release | Benzinga | Yes |
| `consensus_asof` | timestamp UTC | Timestamp of the consensus snapshot | Benzinga | Yes |
| `analyst_coverage` | integer | Count of covering analysts; Table 1 descriptive characteristic | Benzinga / estimates | For Table 1 |
| `forecast_dispersion` | float | Cross-analyst forecast dispersion; exact definition/units to verify | Estimates | For Table 1 |
| `pre_event_close` | float | Pre-announcement stock price `p_(i,t-)` used to scale EPS surprise; document precise date | Massive | Yes |
| `stock_total_return` | float | Dividend-inclusive close-to-close return t-1 to t | Massive | Yes |
| `vti_total_return` | float | VTI dividend-inclusive close-to-close return t-1 to t | Massive | Yes |
| `stock_return_start` | date | Close date t-1 | Exchange calendar | Yes |
| `stock_return_end` | date | Close date t | Exchange calendar | Yes |

## C. Transcripts and LLM provenance

| Field | Type | Definition / rules | Source | Required |
|---|---|---|---|---|
| `transcript_id` | string | Unique provider transcript ID | S&P Capital IQ | Yes |
| `transcript_text_path` | string | Secure pointer to raw transcript; no copyrighted text in public Git | Internal storage | Yes |
| `transcript_available_at` | timestamp UTC | First timestamp transcript became available to system | Provider/ingestion | Yes |
| `transcript_version` | string | Provider version / revision ID | Provider metadata | Yes |
| `transcript_text_hash` | string | Hash for provenance and duplicate detection | Derived | Recommended |
| `llm_prediction` | float | Scalar model prediction, aligned to event | LLM output | For LLM stage |
| `llm_model_version` | string | Exact model name/version | LLM logs | For LLM stage |
| `prompt_version` | string | Frozen prompt/rulebook version | LLM logs | For LLM stage |
| `prediction_submitted_at` | timestamp UTC | Time prediction was generated and frozen | LLM logs | For LLM stage |

## D. Derived variables and evaluation

| Field | Type | Formula / definition | Paper |
|---|---|---|---|
| `abnormal_return` | float | `stock_total_return - vti_total_return` | Eq. 6, pp. 9–10 |
| `sue_raw` | float | `(reported_eps - consensus_eps_mean) / pre_event_close` | Eq. 7, p. 10 |
| `abnormal_return_rank` | float | Ascending rank of `abnormal_return` within evaluation quarter / valid N | Eq. 8, p. 10 |
| `sue_rank` | float | Ascending rank of `sue_raw` within evaluation quarter / valid N | Eq. 8, p. 10 |
| `baseline_residual` | float | Training-only residual from regression of ranked abnormal return on ranked controls | Eq. 9, p. 16 |
| `prediction_residual_error` | float | `baseline_residual - llm_prediction` on training set for prompt objective | Eq. 10, p. 16 |

## E. Data rules

1. Store timestamps with explicit timezone; convert to ET only for event-date logic.
2. Do not use consensus updates after earnings release, or transcript revisions published after model prediction.
3. EPS actual and consensus must have compatible accounting bases and fiscal periods.
4. Preserve both **raw** and **ranked** surprise and abnormal return.
5. Percentile ranks follow `rank / N_q`, with 1 lowest; tie convention is unspecified by the paper and must be documented.
6. Treat winsorization of descriptive statistics in Table 1 as distinct from regression preprocessing until verified.
7. Preserve historical security identifiers, missingness flags, source version, and extraction date.
8. Do not include realized return or post-event information in LLM inputs.

## F. Minimal merge keys

Suggested primary key: `event_id`. Join earnings calendar and EPS by historical security ID plus fiscal period; join transcripts by issuer and call/event timestamp with manual review of ambiguous matches; join stock prices by historical security ID and `trading_event_date`. Never join solely on current ticker.

## G. Unresolved definitions to confirm

- Exact announcement/call time source for Equation 5.
- Pre-announcement price date for SUE when the release is after-hours.
- EPS basis (GAAP/adjusted) and analyst coverage/dispersion definitions.
- Tied percentile ranks, missing-data universe, and sample inclusion rules.
- Whether regression inputs are winsorized or only descriptive Table 1 variables.

**References:** Paper §§2.3.1–2.3.4 (pp. 9–11), §3.1 (pp. 11–12), §3.3.3 (p. 15), §3.4.1 (p. 16), Table 1 (p. 31).
