# Data Access and Replication Feasibility Checklist

**Issue:** #2   
**Scope:** Q4 2025 event-level replication of Koijen & Levy (2026), 1,849 earnings announcements.

## 1. Provider/access inventory

| Priority | Data | Original paper provider | Potential alternative | Access status | Owner / next action |
|---|---|---|---|---|---|
| P0 | Expected earnings calendar + historical snapshots | Benzinga | NTAM internal calendars / approved historical feed | **Unverified** | Ask NTAM for point-in-time event calendar and metadata |
| P0 | Pre-event EPS consensus and reported actual EPS | Benzinga | I/B/E/S (paper's robustness alternative); approved Bloomberg/FactSet data | **Unverified** | Check licensing, historical timestamps, and EPS comparability |
| P0 | Full earnings-call transcripts + availability timestamps | S&P Capital IQ | Approved transcript vendor / Factiva if sufficient coverage | **Unverified** | Verify Q4 2025 transcripts, versions, and timestamp history |
| P0 | Daily stock total returns and VTI | Massive / Polygon.io | Capital IQ (paper's robustness alternative); CRSP or Bloomberg | **Unverified** | Confirm dividend-inclusive adjusted return history and identifiers |
| P1 | Historical security ID and FF12 industry mapping | Not specified as standalone provider | CRSP/Compustat historical SIC or licensed mapping | **Unverified** | Agree on point-in-time mapping and industry classification |
| P1 | Analyst coverage and dispersion | Benzinga / estimate metadata | I/B/E/S or comparable | **Unverified** | Check whether Table 1 statistics can be reproduced |
| P1 | Original prediction logs / benchmark event IDs | Authors / SDK if available | Recompute predictions | **Unverified** | Check authors' released resources; request event roster if possible |

**Interpretation:** 'Unverified' means we have not confirmed access, not that a source is unavailable. Original vendors are identified in §3.1 of the paper; alternatives are potential options unless explicitly labeled as paper robustness sources.

## 2. Access questions for NTAM / university

- [ ] Confirm institutional accounts/licenses for Benzinga, S&P Capital IQ, Massive, I/B/E/S, and alternative feeds.
- [ ] Confirm permission to retrieve historical Q4 2025 records programmatically and retain derived data.
- [ ] Confirm whether transcript text may be stored internally and whether excerpts can appear in GitHub.
- [ ] Confirm whether provider data can be used with approved external LLM APIs.
- [ ] Obtain API documentation, rate limits, request quotas, and costs.
- [ ] Determine whether source metadata exposes first-available timestamps and revisions.
- [ ] Identify a secure data storage location and access policy; do **not** commit raw licensed data or API keys to GitHub.

## 3. Small-sample proof of access (before bulk collection)

- [ ] Retrieve a handful of Q4 2025 events across industries and before/after 16:00 ET.
- [ ] Verify each event has a historical calendar entry, matching fiscal period, EPS actual, and **pre-release** consensus.
- [ ] Retrieve the original call transcript and record its first availability time.
- [ ] Retrieve stock and VTI total returns on the correct trading event date.
- [ ] Calculate `SUE`, `abnormal_return`, and confirm event-time logic by hand.
- [ ] Record field-level source, missing values, timezones, and adjustments.
- [ ] Decide whether the full 1,849-event roster is reconstructible.

## 4. Full sample readiness gates

- [ ] **Gate A — Coverage:** Event roster and event counts reconcile to paper Table 1 (1,849; FF12; six weeks), or differences are explicitly documented.
- [ ] **Gate B — Point in time:** Pre-event expectations and transcripts have usable as-of timestamps; no post-event leakage.
- [ ] **Gate C — Returns:** Stock/VTI daily total returns are matched correctly; after-hours/weekend mapping tested.
- [ ] **Gate D — Baseline:** Surprise-only regression using quarterly percentile ranks is reproduced or deviations explained (paper R² 5.0%).
- [ ] **Gate E — OOS:** Median-date early/late split yields paper's 1,032 / 817 or a documented deviation.
- [ ] **Gate F — Provenance:** Data dictionary, licensing, access logs, and source/version metadata are complete.

## 5. Replication risks and mitigation

| Risk | Consequence | Mitigation |
|---|---|---|
| Original historical point-in-time snapshots unavailable | Consensus or transcript look-ahead | Request archives; flag retrospective proxy as weaker than real-time benchmark |
| Historical event universe differs by vendor | Sample and R² mismatch | Reconcile to Table 1; keep inclusion/exclusion log |
| Different EPS definition or corporate-action handling | Biased SUE | Match fiscal period, EPS basis and split-adjusted units |
| Transcript revision and publication delay | Information not available when claimed | Store first-available timestamp and raw version hash |
| Modern LLM trained on Q4 2025 outcomes | Contamination even with clean event data | Explicitly classify as historical replication; add genuinely prospective evaluation |
| Licensed content committed to public repository | Contract/privacy/compliance issues | Keep only schemas, code, synthetic samples and derived results in Git |

## 6. Replication decision log (fill as you verify)

| Decision | Status | Evidence / link | Owner | Date |
|---|---|---|---|---|
| Original Benzinga access | Pending | | | |
| Original Capital IQ transcript access | Pending | | | |
| Massive/Polygon pricing access | Pending | | | |
| Historical event roster recoverable | Pending | | | |
| Agreed fallback providers | Pending | | | |
| Point-in-time timestamps available | Pending | | | |
| Legal/NTAM approval for LLM inputs | Pending | | | |

## 7. Issue #2 completion criteria

- [ ] Each **required** dataset has a confirmed source or explicitly documented blocker.
- [ ] Dataset fields and transformations are captured in `data_dictionary.md`.
- [ ] A sample event can be joined from transcript through SUE to abnormal return.
- [ ] The team agrees whether it is pursuing **exact real-time replication**, **retrospective approximation**, or **prospective re-evaluation**.
- [ ] Data access and legal/usage constraints are documented before Issue #3 bulk collection.

**Paper references:** §§2.3.1–2.3.4, 3.1, 3.3.3; Table 1. The paper names Benzinga, S&P Capital IQ and Massive as primary providers; I/B/E/S and Capital IQ pricing as robustness alternatives.
