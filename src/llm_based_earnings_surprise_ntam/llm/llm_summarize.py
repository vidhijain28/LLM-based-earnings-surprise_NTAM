"""Summarize transcripts (~28x shorter, as in Koijen & Levy) with a local LLM.

Run from the repo root:
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_summarize --n 30    # pilot (same calls as llm_score --n 30)
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_summarize           # all in-sample calls (resumable)
"""
import argparse, time

import pandas as pd
import ollama

from .llm_common import (MERGED, SUMMARIES, MODEL, anonymize, clean,
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

def chat(messages, max_words):
    resp = ollama.chat(
        model=MODEL, messages=messages, think=False,
        options={"temperature": 0, "num_ctx": 32768, "num_predict": max_words * 3},
    )
    return clean(resp["message"]["content"])

def summarize(transcript, retries=3):
    target_words, max_words = target_length(transcript)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TEMPLATE.format(
                    target_words=target_words, max_words=max_words,
                    compression=COMPRESSION, transcript=transcript)}]
    for attempt in range(retries):
        try:
            summary = chat(messages, max_words)
            if not summary:
                raise ValueError("empty response")
            if len(summary.split()) > max_words:          # one shortening pass
                summary = chat(messages + [
                    {"role": "assistant", "content": summary},
                    {"role": "user", "content": SHORTEN.format(
                        n_words=len(summary.split()), target_words=target_words)},
                ], max_words) or summary
            return summary
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="pilot size (early-quarter calls)")
    args = ap.parse_args()

    df = select_events(pd.read_parquet(MERGED), args.n)
    done = load_done(SUMMARIES)
    if not done.empty:
        df = df[~df["transcriptid"].isin(done["transcriptid"])]
    print(f"{len(df)} calls to summarize with {MODEL}")

    rows = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        text = anonymize(r.full_text, r.companyname, r.ticker)   # summarize anonymized text
        t0 = time.time()
        summary = summarize(text)
        if summary is None:
            print(f"[{i}/{len(df)}] {r.ticker} failed, will retry next run")
            continue
        n_in, n_out = len(text.split()), len(summary.split())
        rows.append({"transcriptid": r.transcriptid, "summary": summary,
                     "words_in": n_in, "words_out": n_out, "model": MODEL,
                     "seconds": round(time.time() - t0, 1)})
        print(f"[{i}/{len(df)}] {r.ticker}: {n_out} words "
              f"({n_in / max(1, n_out):.0f}x shorter, {rows[-1]['seconds']}s)")
        if len(rows) >= 10:
            done, rows = save_append(SUMMARIES, done, rows), []
    if rows:
        save_append(SUMMARIES, done, rows)

if __name__ == "__main__":
    main()