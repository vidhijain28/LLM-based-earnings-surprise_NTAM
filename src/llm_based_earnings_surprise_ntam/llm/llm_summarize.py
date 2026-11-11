"""Summarize transcripts (~28x shorter, as in Koijen & Levy) with an LLM.

Run from the repo root (local Ollama, free):
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_summarize --n 30    # pilot (same calls as llm_score --n 30)
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_summarize           # all in-sample calls (resumable)

Through OpenRouter (paid; see LLM_README.md section 9):
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_summarize --backend openrouter       --model google/gemini-2.5-flash-lite --n 30 --max-cost 1

Reuse summaries from an earlier scored panel (llm_merged_<period>.parquet; no model calls):
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_summarize --import-merged
"""
import argparse, time

import pandas as pd

from .llm_common import (MERGED, SUMMARIES, LLM_MERGED, RUN, add_llm_args, configure, chat, add_usage,
                         over_budget, spent_note, run_parallel, chunk_size, anonymize,
                         select_events, load_done, save_append)

COMPRESSION = 28
MIN_WORDS = 150

SYSTEM = """You are an equity research analyst who writes compact, factual summaries of
earnings calls. Your summaries are read by another analyst who must judge how the stock
will react relative to what investors expected before the call. Use only the transcript.
Do not add outside knowledge, and do not predict or comment on the stock price."""

USER_TEMPLATE = """Summarize this earnings call in about {target_words} words (hard limit {max_words}).
That is roughly 1/{compression} of the transcript, so every sentence must carry information.

Keep exact numbers (growth rates, margins in %, guidance ranges, EPS, dollar amounts) and
always say what they are compared with (prior guidance, last year, last quarter, consensus
if management mentions it). Drop boilerplate: safe-harbor statements, operator lines,
greetings, thank-yous, and repeated figures.

Use these sections, in this order, as short bullet points:

RESULTS: headline revenue, EPS and key operating metrics, and whether management says they
were above, in line with, or below their own expectations.

GUIDANCE: every change to forward guidance. State the old and new figure and the direction
(RAISED / LOWERED / REAFFIRMED / INITIATED / WITHDRAWN). Include long-term targets and any
guidance management declined to give. If no guidance was discussed, write "None given".

ONE-TIME ITEMS: non-recurring gains or charges, impairments, restructuring, litigation,
tax benefits, acquisitions/divestitures, calendar or FX effects, and pull-forward of demand.
Say whether they flatter or depress the reported numbers and by how much if stated.

MARGINS: gross, operating and segment margins, their direction, and the stated drivers
(pricing, input costs, mix, wages, investment spend). Flag margins helped by one-time items.

TONE AND CAUTION: management's cautious or hedging language, quoted briefly in their own
words (e.g. "fell short", "macro pressure", "softer", "uncertain", "too early to say"),
plus any confident claims. Note risks they raised and any change in tone vs. prior plans.

Q&A: the main topics analysts pressed on, any new information revealed only in the Q&A,
and questions management dodged, deflected or answered vaguely. Note pushback or concern
from analysts.

Transcript:
<transcript>
{transcript}
</transcript>"""

SHORTEN = """That summary is {n_words} words. Rewrite it in at most {target_words} words with the
same sections. Cut minor metrics and segment detail first; keep every guidance change,
one-time item, margin direction, cautious quote and key Q&A point."""

def target_length(text):
    target = max(MIN_WORDS, round(len(text.split()) / COMPRESSION))
    return target, round(target * 1.3)

def summarize(transcript, retries=3):
    """Returns (summary or None, usage summed over calls, number of passes)."""
    target_words, max_words = target_length(transcript)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TEMPLATE.format(
                    target_words=target_words, max_words=max_words,
                    compression=COMPRESSION, transcript=transcript)}]
    usage = {}
    for attempt in range(retries):
        try:
            summary, u = chat(messages, max_tokens=max_words * 3)
            add_usage(usage, u)
            if not summary:
                raise ValueError("empty response")
            passes = 1
            if len(summary.split()) > max_words:          # one shortening pass
                shorter, u = chat(messages + [
                    {"role": "assistant", "content": summary},
                    {"role": "user", "content": SHORTEN.format(
                        n_words=len(summary.split()), target_words=target_words)},
                ], max_tokens=max_words * 3)
                add_usage(usage, u)
                summary, passes = shorter or summary, 2
            return summary, usage, passes
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return None, usage, 0


def import_merged(path=LLM_MERGED):
    """Copy the summaries saved in a scored panel (llm_merged_<period>.parquet) into llm_summaries.parquet,
    tagged with the model that wrote them, so other models can score them (--summary-model)."""
    if not path.exists():
        raise SystemExit(f"{path} not found")
    m = pd.read_parquet(path, columns=["transcriptid", "summary", "sum_words_in", "sum_words_out",
                                       "sum_passes", "sum_model", "sum_seconds"])
    m = m.dropna(subset=["summary"]).rename(columns={
        "sum_words_in": "words_in", "sum_words_out": "words_out", "sum_passes": "passes",
        "sum_model": "model", "sum_seconds": "seconds"})
    done = load_done(SUMMARIES)
    if not done.empty:
        have = set(zip(done["transcriptid"], done["model"]))
        m = m[[k not in have for k in zip(m["transcriptid"], m["model"])]]
    save_append(SUMMARIES, done, m.assign(backend="imported").to_dict("records"))
    print(f"Imported {len(m):,} summaries from {path.name} "
          f"(models: {sorted(m['model'].unique().tolist())}) into {SUMMARIES.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="pilot size (early-quarter calls)")
    ap.add_argument("--import-merged", action="store_true",
                    help=f"copy summaries from {LLM_MERGED.name} instead of calling a model")
    add_llm_args(ap)
    args = ap.parse_args()
    if args.import_merged:
        return import_merged()
    configure(args)
    model = RUN["model"]

    df = select_events(pd.read_parquet(MERGED), args.n)
    done = load_done(SUMMARIES)
    if not done.empty:
        df = df[~df["transcriptid"].isin(done.loc[done["model"] == model, "transcriptid"])]
    print(f"{len(df)} calls to summarize with {model}")

    def one(r):
        text = anonymize(r.full_text, r.companyname, r.ticker)   # summarize anonymized text
        t0 = time.time()
        summary, usage, passes = summarize(text)
        if summary is None:
            print(f"  {r.ticker} failed, will retry next run")
            return None
        n_in, n_out = len(text.split()), len(summary.split())
        print(f"  {r.ticker}: {n_out} words ({n_in / max(1, n_out):.0f}x shorter, {time.time() - t0:.1f}s)")
        return {"transcriptid": r.transcriptid, "summary": summary, "words_in": n_in, "words_out": n_out,
                "passes": passes, "model": model, "backend": RUN["backend"],
                "seconds": round(time.time() - t0, 1), **usage}

    n_done = 0
    for start in range(0, len(df), chunk_size()):
        if over_budget():
            print(f"Budget of ${RUN['max_cost']:.2f} reached; stopping. Rerun to continue where it stopped.")
            break
        rows = [x for x in run_parallel(list(df.iloc[start:start + chunk_size()].itertuples(index=False)), one) if x]
        done = save_append(SUMMARIES, done, rows)
        n_done += len(rows)
        print(f"[{min(start + chunk_size(), len(df))}/{len(df)}] {n_done} saved. {spent_note()}")
    print(f"Done: {n_done} summaries. {spent_note()}")


if __name__ == "__main__":
    main()
