# Data Requirements — Koijen & Levy (2026) Replication

**Issue:** #2 · **Status:** Specification extracted; access not verified  
**Source:** Koijen & Levy, *Assessing the Benefits of Optimized Agentic AI Systems for Asset Pricing*, NBER WP 35431 (July 2026). Page numbers below are printed paper pages, not PDF viewer pages.

## 1. Replication scope and sample

- **Benchmark:** Explain contemporaneous abnormal returns around earnings announcements, rather than forecast long-horizon post-earnings drift (Sections 2.2–2.3).
- **Period:** 2025 Q4 (October–December 2025); **1,849 earnings events**, across **12 Fama–French industries** and **six reporting weeks** (Section 3.1; Table 1, p. 31).
- **Universe:** The paper specifies the observed earnings-event sample and FF12 coverage, but does **not** state a simple fixed membership rule such as S&P 500 or Russell 1000. Do not substitute one without documenting the deviation.
- **Optimization split:** Median announcement date: **early 1,032 observations** (optimization/training), **late 817** (out-of-sample evaluation), Section 3.3.3, p. 15. Obtain the precise cutoff and boundary treatment from the original event calendar or reproduce and document them.
- **Primary benchmark:** Percentile-ranked abnormal return regressed on percentile-ranked standardized earnings surprise, R² ~5.0% (Section 3.3.1, Table 2). LLM prediction + surprise is evaluated in a cross-sectional regression (Equation 3).

## 2. Original data providers

| Dataset | Original provider | Required contents | Paper reference |
|---|---|---|---|
| Expected earnings events | Benzinga | Upcoming event calendar; firm/ticker; expected event date; actual release date/time if supplied | §3.1, p. 11 |
| EPS expectations and reported actuals | Benzinga | Pre-announcement **mean** analyst consensus EPS, reported EPS, their as-of timestamps and compatible EPS basis | §§2.3.2, 3.1, pp. 10–11 |
| Earnings-call transcripts | S&P Capital IQ | Full text, firm, call time, transcript availability/publication timestamp; version | §3.1, p. 11 |
| Stock and market returns | Massive (formerly Polygon.io) | Dividend-inclusive daily stock total returns and VTI daily total returns / data to calculate them | §§2.3.1, 3.1, pp. 9–11 |
| Industry labels | Fama–French 12 industries | Industry mapping per event, ideally from historical SIC codes | §3.1; Table 1 |
| Robustness alternatives | I/B/E/S; S&P Capital IQ | EPS actuals/consensus from I/B/E/S; pricing from Capital IQ | §3.1, pp. 11–12 |

**Do not assume provider access.** University/NTAM holdings, licensing, historical point-in-time availability, and redistribution restrictions remain unverified. Factiva, CRSP, Bloomberg, Compustat etc. may be evaluated as *alternatives*, but they are not the paper's named primary providers.

## 3. Exact target and transformations

1. **Trading event date:** for a call on calendar date `d` at time `t_c`, set event trading date `t=d` when `d` is a trading day and `t_c <= 16:00 ET`; otherwise use the next trading day `τ(d)` (Equation 5, p. 9). **Clarify whether `t_c` is earnings-release or call time** in the source event metadata; the paper's wording refers to the earnings event/call.
2. **One-day return:** use dividend-inclusive stock total return from close `t-1` to close `t` and equivalent VTI market return. `abnormal_return = stock_total_return - VTI_total_return` (Equation 6, pp. 9–10).
3. **Surprise:** `SUE = (reported_EPS - pre_release_mean_consensus_EPS) / pre_announcement_stock_price` (Equation 7, p. 10). Check EPS basis and split adjustments.
4. **Ranks:** within evaluation quarter `q`, `percentile_rank(x) = ascending_rank(x) / N_q`, where rank 1 is lowest (Equation 8, p. 10). Define tie handling in implementation notes because not specified in the text.
5. **Outliers:** Table 1 notes continuous variables winsorized at 1st and 99th percentiles for its summary statistics (p. 31). **Do not assume** this applies to every regression variable; confirm from tables/code.
6. **Evaluation:** regress ranked abnormal return on ranked SUE and, where applicable, the scalar LLM prediction; calculate R² and incremental R² versus surprise-only model (Equations 3, 9–11; pp. 8, 16). The paper reports 99% bootstrap CIs for main results (§3.3); bootstrap implementation details require confirmation.

## 4. Point-in-time and leakage requirements

- Preserve **scheduled event calendar snapshots** collected *before* announcements, not only retrospective actual dates.
- Store consensus **as-of time strictly before the earnings release**; avoid revised consensus values.
- Record earnings event, call, transcript publication/availability, model submission, and price observation timestamps separately in ET or timezone-aware UTC.
- Only allow text available by prediction time; do not use corrected transcripts, later news, post-event price moves, or realized returns as model inputs.
- Freeze prompts/context before each evaluated event. Training on early-quarter realized outcomes may inform later-quarter prompts, but late-quarter outcomes must not feed back into optimization (Section 2.3.4; §3.4.1).
- **Historical recreation is not identical to the paper's real-time OOS benchmark:** modern LLM weights may know Q4 2025 outcomes. Report this limitation prominently and, if possible, run an additional genuinely prospective evaluation.

## 5. Proposed reproducible event panel

**Unit:** one company earnings event, uniquely keyed by a stable security ID plus event timestamp / fiscal period. Proposed schema: `event_id`, `security_id`, `ticker_at_event`, `event_datetime_et`, `trading_event_date`, `transcript_available_at`, `fiscal_period`, `ff12_industry`, `reported_eps`, `consensus_eps`, `consensus_asof`, `price_pre_event`, `stock_total_return`, `vti_total_return`, `abnormal_return`, `sue_raw`, `abnormal_return_rank`, `sue_rank`, `split`, `transcript_id`, `source_versions`.

Keep raw provider files separate from standardized event panel and LLM outputs. Maintain an event-level audit trail and QA log.

## 6. Quality-control checks

- One record per distinct event; no duplicate transcripts or misaligned fiscal periods.
- Match coverage against **1,849 events** and FF12 industry/week totals in Table 1; log omissions and why.
- Confirm close-to-close date alignment for after-hours/non-trading-day events.
- Verify dividend/split adjustments, VTI matching dates, EPS unit/basis, stale consensus, and historical ticker changes.
- Recalculate baseline 5.0% R² on available matched observations, documenting sample deviations rather than forcing a match.
- Store all missing-data exclusions, timestamps, and source provenance.

## 7. Known unknowns / questions for NTAM

1. Do we have licensed **historical, timestamped** Benzinga, Capital IQ transcripts, and Massive data for Q4 2025?
2. Can NTAM or the university supply original event IDs and historical ticker/security mapping?
3. Can we recover the original **1,849-event universe** and the exact early/late split?
4. What was the exact event timestamp used to implement Equation 5, and how are calls spanning the close treated?
5. Are full transcripts available at the same timestamps as the original benchmark, including revisions?
6. How should ties, missing observations, rank universes, and bootstrap resampling be handled?
7. What LLM model versions and point-in-time protections can be reproduced today?

## 8. Paper references

- §2.2, Equation 3 (p. 8): evaluation regression and incremental R².
- §2.3.1–2.3.3, Equations 5–8 (pp. 9–10): timing, abnormal return, SUE, percentile ranks.
- §2.3.4 (p. 11): real-time evaluation protocol.
- §3.1–3.3 (pp. 11–15): providers, sample, results, train/test split.
- §3.4.1, Equations 9–11 (p. 16): optimization/evaluation split.
- Table 1 (p. 31): sample statistics and industry/week distribution.
