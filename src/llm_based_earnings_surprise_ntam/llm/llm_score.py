"""Score earnings calls with a local LLM (prediction 0-1, as in Koijen & Levy).

Run from the repo root:
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --n 30 --source full_text   # pilot on full transcripts
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --n 30 --source summary     # pilot on summaries
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --source qa_text            # all calls, Q&A only
Model and default source come from the `llm` section of config.yaml (override with --model / --source).
On Midway, run_llm.sh splits the calls across GPU jobs with --shard / --nshards.

Prompt experiments on the dev set (llm_devset; LLM_README.md section 4b):
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --ids --variant base                  # dev300.csv
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --ids --variant base --temperature 0.7 --samples 5
--variant picks the prompt from PROMPTS below and is saved in the `variant` column. --samples N scores each call
N times (seed 0..N-1, so it is reproducible): `prediction` is the average, `predictions` / `reasons_all` keep
every run.
A run is model x source x variant x temperature x samples; calls already scored by the same run are skipped.
"""
import argparse, json, statistics, time

import pandas as pd
import ollama

from .llm_common import (MERGED, SUMMARIES, SCORES, MODEL, SOURCE, NUM_CTX, DEV_IDS, RUN_DEFAULTS, anonymize, clean,
                         require_ollama, select_events, load_done, load_all, shard_path, save_append, read_ids,
                         with_run_defaults)

SOURCES = ["full_text", "presentation_text", "qa_text", "summary"]

SYSTEM = """You are an equity analyst predicting the stock market's reaction to an earnings call.
Use only the information provided. Do not use outside knowledge about this company
or how its stock actually moved."""

USER_TEMPLATE = """Your task is to predict how this stock's one-day abnormal return (stock minus market)
will rank against all other companies reporting earnings this quarter.

Be mindful that markets price news relative to expectations. Strong results that were
already expected are neutral, not positive. Forward guidance, margins, one-time items and
management's tone in the Q&A often matter more than the headline numbers.

Steps:
1. Identify the key news: results, guidance changes, one-time items, risks.
2. Judge whether each item is better or worse than investors likely expected.
3. Combine these into a single prediction.

Return ONLY a JSON object:
- "prediction": number from 0 to 1 (0 = worst reaction this quarter, 0.5 = neutral, 1 = best)
- "reasons": list of up to 3 short strings

Earnings call:
<call>
{text}
</call>"""

# --- Prompt variants for dev-set experiments (LLM_README.md section 4b) --------------------------------------
# `--variant NAME` scores with PROMPTS[NAME]. "base" is the prompt of the full Q4 2025 run: don't edit it (its
# 2,584 scores are reused). A new variant = base + one change, so each idea is measured on its own.

RECURRING_FOCUS = """Focus on the quality of earnings. Separate results from the recurring, core business
(organic revenue growth, margins excluding one-time items, guidance for the ongoing business) from one-time
items (asset sales, tax benefits, impairments, restructuring, litigation, calendar or FX effects, pull-forward
of demand). Markets reward beats and guidance raises that come from recurring operations and discount those
that come from one-time items; a miss caused only by a one-time charge matters less than a miss in the core
business.

"""

REPORTING_DATE = """Reporting timing: the company released these results {lag_days} days after its fiscal
quarter ended. US companies usually report about 4 to 6 weeks after the quarter ends. Historically, companies
that report later than usual have tended to have worse news, and early reporters better news.

"""

def _add_before(template, anchor, text):
    assert template.count(anchor) == 1, anchor
    return template.replace(anchor, text + anchor)

PROMPTS = {
    "base": USER_TEMPLATE,
    "recurring_focus": _add_before(USER_TEMPLATE, "Steps:", RECURRING_FOCUS),
    "reporting_date": _add_before(USER_TEMPLATE, "Earnings call:\n<call>", REPORTING_DATE),
    "temp": USER_TEMPLATE,   # label used for the temperature runs (base prompt)
}

def prompt_fields(r):
    """Per-call values the variant prompts can use: {lag_days} = release date (IBES anndats) minus fiscal
    quarter end (pends), known on the release day."""
    lag = (r.anndats - r.pends).days if pd.notna(r.anndats) and pd.notna(r.pends) else "an unknown number of"
    return {"lag_days": lag}

def score(text, retries=3, model=MODEL, temperature=0.0, seed=None, template=USER_TEMPLATE, fields=None):
    content = template.format(text=text, **(fields or {}))
    options = {"temperature": temperature, "num_ctx": NUM_CTX}
    if seed is not None:
        options["seed"] = seed
    for attempt in range(retries):
        try:
            resp = ollama.chat(
                model=model, think=False, format="json",
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": content}],
                options=options,
            )
            raw = clean(resp["message"]["content"])
            out = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            out["prediction"] = min(max(float(out["prediction"]), 0.0), 1.0)
            return out
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return {"prediction": None, "reasons": ["failed"]}

def score_samples(text, model, temperature, samples, template=USER_TEMPLATE, fields=None):
    """Score one call `samples` times. Returns (average prediction or None, each run's prediction, each run's
    reasons). At temperature > 0 run k uses seed k, so a rerun gives the same draws."""
    runs = [score(text, model=model, temperature=temperature, seed=k if temperature > 0 else None,
                  template=template, fields=fields)
            for k in range(samples)]
    preds = [r.get("prediction") for r in runs]
    ok = [p for p in preds if p is not None]
    return (statistics.fmean(ok) if ok else None), preds, [r.get("reasons", []) for r in runs]

def run(merged=MERGED, summaries=SUMMARIES, out=SCORES, model=MODEL, source=SOURCE, n=None,
        shard=None, nshards=1, variant=RUN_DEFAULTS["variant"], ids=None,
        temperature=RUN_DEFAULTS["temperature"], samples=RUN_DEFAULTS["samples"]):
    """Score the in-sample calls not yet scored by this run (model x source x variant x temperature x samples)
    in `out` or its shard files. Resumable. With ids (a csv such as dev300.csv), only those calls.
    With shard, only this job's share is done and written to the job's own shard file."""
    if source not in SOURCES:
        raise SystemExit(f"source must be one of {SOURCES}, got {source!r}")
    if variant not in PROMPTS:
        raise SystemExit(f"unknown --variant {variant!r}; prompts in llm_score.py: {', '.join(PROMPTS)}")
    if samples < 1:
        raise SystemExit("--samples must be at least 1")
    if samples > 1 and temperature == 0:
        print("WARNING: at temperature 0 every sample gives the same answer; use --temperature > 0 with --samples")
    df = select_events(pd.read_parquet(merged), n, shard=shard, nshards=nshards,
                       ids=None if ids is None else read_ids(ids))
    if source == "summary":
        summ = load_all(summaries)     # main file + shard files of jobs still running
        if summ.empty:
            raise SystemExit("No summaries yet: run `uv run python -m llm_based_earnings_surprise_ntam.llm.llm_summarize` first")
        df = df.merge(summ[["transcriptid", "summary"]].drop_duplicates("transcriptid"), on="transcriptid")

    seen = with_run_defaults(load_all(out))
    if not seen.empty:
        mine = seen[(seen["source"] == source) & (seen["model"] == model) & (seen["variant"] == variant)
                    & (seen["temperature"] == float(temperature)) & (seen["samples"] == samples)]
        df = df[~df["transcriptid"].isin(mine["transcriptid"])]
    out_path = shard_path(out, shard)
    done = load_done(out_path)
    print(f"{len(df)} calls to score with {model} on {source} "
          f"(variant {variant}, temperature {temperature}, {samples} sample{'s' if samples > 1 else ''})")
    if df.empty:
        return
    require_ollama(model)

    rows = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        raw = getattr(r, source)
        # mask summaries too: the model sometimes writes the year or name back in
        text = anonymize(raw, r.companyname, r.ticker)
        t0 = time.time()
        pred, preds, reasons = score_samples(text, model, temperature, samples,
                                             template=PROMPTS[variant], fields=prompt_fields(r))
        ok = [p for p in preds if p is not None]
        first = next((rs for p, rs in zip(preds, reasons) if p is not None), reasons[0])
        rows.append({"transcriptid": r.transcriptid, "keydevid": r.keydevid,
                     "ticker": r.ticker, "event_date": r.event_date,
                     "prediction": pred,                                  # average over the samples
                     "reasons": json.dumps(first),                        # first successful sample's reasons
                     "model": model, "source": source,
                     "seconds": round(time.time() - t0, 1),
                     "variant": variant, "temperature": float(temperature), "samples": samples,
                     "predictions": json.dumps(preds),                    # every sample (null = failed)
                     "reasons_all": json.dumps(reasons),
                     "prediction_sd": statistics.stdev(ok) if len(ok) > 1 else None})
        runs = f" from {preds}" if samples > 1 else ""
        print(f"[{i}/{len(df)}] {r.ticker} {str(r.event_date)[:10]} -> "
              f"{pred if pred is None else round(pred, 3)}{runs} ({rows[-1]['seconds']}s)")
        if len(rows) >= 10:
            done, rows = save_append(out_path, done, rows), []
    if rows:
        save_append(out_path, done, rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="pilot size (early-quarter calls)")
    ap.add_argument("--source", default=SOURCE, choices=SOURCES, help="default: llm.source in config.yaml")
    ap.add_argument("--model", default=MODEL, help="Ollama model (default: llm.model in config.yaml)")
    ap.add_argument("--shard", type=int, help="this job's number (0 .. nshards-1)")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--variant", default=RUN_DEFAULTS["variant"], choices=list(PROMPTS),
                    help="prompt to score with (PROMPTS in llm_score.py), saved in the `variant` column")
    ap.add_argument("--ids", nargs="?", const=DEV_IDS, metavar="CSV",
                    help=f"only score the calls in this csv (`--ids` alone: {DEV_IDS.name})")
    ap.add_argument("--temperature", type=float, default=RUN_DEFAULTS["temperature"])
    ap.add_argument("--samples", type=int, default=RUN_DEFAULTS["samples"],
                    help="score each call N times; saves every run and the average")
    args = ap.parse_args()
    if args.ids is not None and args.n:
        ap.error("use --ids or --n, not both")
    run(model=args.model, source=args.source, n=args.n, shard=args.shard, nshards=args.nshards,
        variant=args.variant, ids=args.ids, temperature=args.temperature, samples=args.samples)

if __name__ == "__main__":
    main()
