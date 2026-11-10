"""Score earnings calls with a local LLM (prediction 0-1, as in Koijen & Levy).

Run from the repo root:
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --n 30 --source full_text   # pilot on full transcripts
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --n 30 --source summary     # pilot on summaries
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --source qa_text            # all calls, Q&A only
Model and default source come from the `llm` section of config.yaml (override with --model / --source).
"""
import argparse, json, time

import pandas as pd
import ollama

from .llm_common import (MERGED, SUMMARIES, SCORES, MODEL, SOURCE, anonymize, clean, require_ollama,
                            select_events, load_done, save_append)

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

def score(text, retries=3, model=MODEL):
    for attempt in range(retries):
        try:
            resp = ollama.chat(
                model=model, think=False, format="json",
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": USER_TEMPLATE.format(text=text)}],
                options={"temperature": 0, "num_ctx": 32768},
            )
            raw = clean(resp["message"]["content"])
            out = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            out["prediction"] = min(max(float(out["prediction"]), 0.0), 1.0)
            return out
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return {"prediction": None, "reasons": ["failed"]}

def run(merged=MERGED, summaries=SUMMARIES, out=SCORES, model=MODEL, source=SOURCE, n=None):
    """Score the in-sample calls not yet scored by (model, source) in `out`. Resumable."""
    if source not in SOURCES:
        raise SystemExit(f"source must be one of {SOURCES}, got {source!r}")
    df = select_events(pd.read_parquet(merged), n)
    if source == "summary":
        if not summaries.exists():
            raise SystemExit("No summaries yet: run `uv run python -m llm_based_earnings_surprise_ntam.llm.llm_summarize` first")
        df = df.merge(pd.read_parquet(summaries)[["transcriptid", "summary"]], on="transcriptid")

    done = load_done(out)
    if not done.empty:
        mine = done[(done["source"] == source) & (done["model"] == model)]
        df = df[~df["transcriptid"].isin(mine["transcriptid"])]
    print(f"{len(df)} calls to score with {model} on {source}")
    if df.empty:
        return
    require_ollama(model)

    rows = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        raw = getattr(r, source)
        text = raw if source == "summary" else anonymize(raw, r.companyname, r.ticker)
        t0 = time.time()
        res = score(text, model=model)
        rows.append({"transcriptid": r.transcriptid, "keydevid": r.keydevid,
                     "ticker": r.ticker, "event_date": r.event_date,
                     "prediction": res.get("prediction"),
                     "reasons": json.dumps(res.get("reasons", [])),
                     "model": model, "source": source,
                     "seconds": round(time.time() - t0, 1)})
        print(f"[{i}/{len(df)}] {r.ticker} {str(r.event_date)[:10]} -> "
              f"{res.get('prediction')} ({rows[-1]['seconds']}s)")
        if len(rows) >= 10:
            done, rows = save_append(out, done, rows), []
    if rows:
        save_append(out, done, rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="pilot size (early-quarter calls)")
    ap.add_argument("--source", default=SOURCE, choices=SOURCES, help="default: llm.source in config.yaml")
    ap.add_argument("--model", default=MODEL, help="Ollama model (default: llm.model in config.yaml)")
    args = ap.parse_args()
    run(model=args.model, source=args.source, n=args.n)

if __name__ == "__main__":
    main()