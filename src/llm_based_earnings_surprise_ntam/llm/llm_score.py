"""Score earnings calls with an LLM (prediction 0-1, as in Koijen & Levy).

Run from the repo root (local Ollama, free):
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --n 30                    # pilot on full transcripts
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --n 30 --source summary   # pilot on summaries
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --source qa_text          # all calls, Q&A only

Through OpenRouter (paid; see LLM_README.md section 9):
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --backend openrouter \
      --model google/gemini-2.5-flash-lite --source summary --n 30 --max-cost 1

--with-numbers adds the release's EPS, consensus and surprise (I/B/E/S) above the text.
--summary-model picks whose summaries to score (default: the scoring model's own, the paper's Table 3).
"""
import argparse, json, time

import pandas as pd

from .llm_common import (MERGED, SUMMARIES, SCORES, RUN, add_llm_args, configure, chat, add_usage,
                         over_budget, spent_note, run_parallel, chunk_size, anonymize, earnings_numbers,
                         select_events, load_done, save_append)

SOURCES = ["full_text", "presentation_text", "qa_text", "summary"]

SYSTEM = """You are an equity analyst predicting the stock market's reaction to an earnings call.
Use only the information provided. Do not use outside knowledge about this company
or how its stock actually moved."""

# {numbers} is "" unless --with-numbers, so the default prompt is unchanged
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

{numbers}Earnings call:
<call>
{text}
</call>"""

MAX_TOKENS = 600   # the JSON answer is ~100 tokens; the cap only stops runaway replies


def score(text, numbers="", retries=3):
    """Returns (result dict, usage summed over attempts)."""
    usage = {}
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TEMPLATE.format(text=text, numbers=numbers)}]
    for attempt in range(retries):
        try:
            raw, u = chat(messages, max_tokens=MAX_TOKENS, json_mode=True)
            add_usage(usage, u)
            out = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            out["prediction"] = min(max(float(out["prediction"]), 0.0), 1.0)
            return out, usage
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return {"prediction": None, "reasons": ["failed"]}, usage


def already_scored(done, source, model, variant, summary_model):
    """transcriptids done for this exact setup; failed calls (no prediction) are retried on the next run.
    Rows from older runs have no variant / summary_model column: they count as 'plain', scored on the
    scoring model's own summaries."""
    if done.empty:
        return set()
    variant_col = done["variant"].fillna("plain") if "variant" in done else pd.Series("plain", index=done.index)
    sm_col = done["summary_model"] if "summary_model" in done else pd.Series(pd.NA, index=done.index)
    sm_col = sm_col.fillna(done["model"])
    mask = ((done["source"] == source) & (done["model"] == model) & (variant_col == variant)
            & done["prediction"].notna())
    if source == "summary":
        mask &= sm_col == summary_model
    return set(done.loc[mask, "transcriptid"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="pilot size (early-quarter calls)")
    ap.add_argument("--source", default="full_text", choices=SOURCES)
    ap.add_argument("--with-numbers", action="store_true", help="add EPS, consensus and surprise to the prompt")
    ap.add_argument("--summary-model", help="whose summaries to score (default: the scoring model)")
    add_llm_args(ap)
    args = ap.parse_args()
    configure(args)
    model = RUN["model"]
    variant = "numbers" if args.with_numbers else "plain"
    summary_model = (args.summary_model or model) if args.source == "summary" else None

    df = select_events(pd.read_parquet(MERGED), args.n)
    if args.source == "summary":
        if not SUMMARIES.exists():
            raise SystemExit("No summaries yet: run llm_summarize first (or llm_summarize --import-merged)")
        summ = pd.read_parquet(SUMMARIES)
        summ = summ[summ["model"] == summary_model].drop_duplicates("transcriptid", keep="last")
        if summ.empty:
            have = sorted(pd.read_parquet(SUMMARIES)["model"].unique())
            raise SystemExit(f"No summaries by {summary_model!r}. Available: {have}. Use --summary-model.")
        df = df.merge(summ[["transcriptid", "summary"]], on="transcriptid")

    done = load_done(SCORES)
    df = df[~df["transcriptid"].isin(already_scored(done, args.source, model, variant, summary_model))]
    print(f"{len(df)} calls to score with {model} on {args.source}"
          + (f" (summaries by {summary_model})" if summary_model else "") + f", prompt: {variant}")

    def one(r):
        raw = getattr(r, args.source)
        text = raw if args.source == "summary" else anonymize(raw, r.companyname, r.ticker)
        t0 = time.time()
        res, usage = score(text, earnings_numbers(r) if args.with_numbers else "")
        row = {"transcriptid": r.transcriptid, "keydevid": r.keydevid,
               "ticker": r.ticker, "event_date": r.event_date,
               "prediction": res.get("prediction"),
               "reasons": json.dumps(res.get("reasons", [])),
               "model": model, "source": args.source, "summary_model": summary_model, "variant": variant,
               "backend": RUN["backend"], "seconds": round(time.time() - t0, 1), **usage}
        print(f"  {r.ticker} {str(r.event_date)[:10]} -> {res.get('prediction')} ({row['seconds']}s)")
        return row

    n_done = 0
    for start in range(0, len(df), chunk_size()):
        if over_budget():
            print(f"Budget of ${RUN['max_cost']:.2f} reached; stopping. Rerun to continue where it stopped.")
            break
        rows = run_parallel(list(df.iloc[start:start + chunk_size()].itertuples(index=False)), one)
        done = save_append(SCORES, done, rows)
        n_done += len(rows)
        print(f"[{n_done}/{len(df)}] saved. {spent_note()}")
    print(f"Done: {n_done} calls scored. {spent_note()}")


if __name__ == "__main__":
    main()
