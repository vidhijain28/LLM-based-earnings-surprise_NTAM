"""Summarize transcripts (~28x shorter, as in Koijen & Levy) with a local LLM.

Run from the repo root:
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_summarize --n 30    # pilot (same calls as llm_score --n 30)
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_summarize           # all in-sample calls (resumable)
"""
import argparse, re, time

import pandas as pd
import ollama

from .llm_common import (MERGED, SUMMARIES, MODEL, NUM_CTX, anonymize, clean,
                         select_events, load_done, load_all, shard_path, save_append)

COMPRESSION = 28
MIN_WORDS = 150
# LLMs can't count words but can count bullets: turn the word target into bullets per section
SECTIONS = ["RESULTS", "GUIDANCE", "ONE-TIME ITEMS", "MARGINS", "TONE AND CAUTION", "Q&A"]
WEIGHTS = [0.20, 0.20, 0.10, 0.15, 0.15, 0.20]
BULLET_WORDS = 18       # max words per bullet (told to the model)
WORDS_PER_BULLET = 16   # typical bullet length, used to size the budget

SYSTEM = """You are an equity research analyst who writes compact, factual summaries of
earnings calls. Your summaries are read by another analyst who must judge how the stock
will react relative to what investors expected before the call. Use only the transcript.
Do not add outside knowledge, and do not predict or comment on the stock price."""

USER_TEMPLATE = """Summarize this earnings call in at most {n_bullets} bullet points in total (about
{target_words} words, hard limit {max_words}). That is roughly 1/{compression} of the transcript,
so every bullet must carry information.

Keep exact numbers (growth rates, margins in %, guidance ranges, EPS, dollar amounts) and
always say what they are compared with (prior guidance, last year, last quarter, consensus
if management mentions it). Drop boilerplate: safe-harbor statements, operator lines,
greetings, thank-yous, and repeated figures.

Use these sections, in this order. The number in brackets is the MAXIMUM number of bullets
for that section (fewer is fine). Each bullet is one line of at most {bullet_words} words. No
sub-bullets, no introduction, no closing remarks. Write the headings without the brackets.

RESULTS [{b0}]: headline revenue, EPS and key operating metrics, and whether management says they
were above, in line with, or below their own expectations.

GUIDANCE [{b1}]: every change to forward guidance. State the old and new figure and the direction
(RAISED / LOWERED / REAFFIRMED / INITIATED / WITHDRAWN). Include long-term targets and any
guidance management declined to give. If no guidance was discussed, write "None given".

ONE-TIME ITEMS [{b2}]: non-recurring gains or charges, impairments, restructuring, litigation,
tax benefits, acquisitions/divestitures, calendar or FX effects, and pull-forward of demand.
Say whether they flatter or depress the reported numbers and by how much if stated.

MARGINS [{b3}]: gross, operating and segment margins, their direction, and the stated drivers
(pricing, input costs, mix, wages, investment spend). Flag margins helped by one-time items.

TONE AND CAUTION [{b4}]: management's cautious or hedging language, quoted briefly in their own
words (e.g. "fell short", "macro pressure", "softer", "uncertain", "too early to say"),
plus any confident claims. Note risks they raised and any change in tone vs. prior plans.

Q&A [{b5}]: the main topics analysts pressed on, any new information revealed only in the Q&A,
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

def bullet_budget(target_words):
    """Bullets per section, e.g. 259 words -> [3, 3, 2, 2, 3, 3] (16 bullets)."""
    n = max(len(SECTIONS), round(target_words / WORDS_PER_BULLET))
    raw = [w * n for w in WEIGHTS]
    b = [max(1, int(x)) for x in raw]
    by_remainder = sorted(range(len(raw)), key=lambda i: raw[i] - b[i], reverse=True)
    for i in by_remainder[:max(0, n - sum(b))]:
        b[i] += 1
    return b

def chat(messages, max_words):
    resp = ollama.chat(
        model=MODEL, messages=messages, think=False,
        options={"temperature": 0, "num_ctx": NUM_CTX, "num_predict": max_words * 3},
    )
    return clean(resp["message"]["content"])

def summarize(transcript, retries=3):
    target_words, max_words = target_length(transcript)
    budget = bullet_budget(target_words)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TEMPLATE.format(
                    target_words=target_words, max_words=max_words, n_bullets=sum(budget),
                    bullet_words=BULLET_WORDS, compression=COMPRESSION, transcript=transcript,
                    **{f"b{i}": k for i, k in enumerate(budget)})}]
    for attempt in range(retries):
        try:
            summary, passes = chat(messages, max_words), 1
            if not summary:
                raise ValueError("empty response")
            if len(summary.split()) > max_words:          # one shortening pass (fallback)
                passes = 2
                summary = chat(messages + [
                    {"role": "assistant", "content": summary},
                    {"role": "user", "content": SHORTEN.format(
                        n_words=len(summary.split()), target_words=target_words)},
                ], max_words) or summary
            # drop the bullet budgets the model copies into headings, e.g. "RESULTS [3]"
            return re.sub(r" ?\[\d+\]", "", summary), passes
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return None, 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="pilot size (early-quarter calls)")
    ap.add_argument("--shard", type=int, help="this job's number (0 .. nshards-1)")
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    df = select_events(pd.read_parquet(MERGED), args.n, shard=args.shard, nshards=args.nshards)
    out_path = shard_path(SUMMARIES, args.shard)
    seen = load_all(SUMMARIES)
    if not seen.empty:   # skip calls already summarized by THIS model
        df = df[~df["transcriptid"].isin(seen.loc[seen["model"] == MODEL, "transcriptid"])]
    done = load_done(out_path)
    print(f"{len(df)} calls to summarize with {MODEL}")

    rows = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        text = anonymize(r.full_text, r.companyname, r.ticker)   # summarize anonymized text
        t0 = time.time()
        summary, passes = summarize(text)
        if summary is None:
            print(f"[{i}/{len(df)}] {r.ticker} failed, will retry next run")
            continue
        n_in, n_out = len(text.split()), len(summary.split())
        rows.append({"transcriptid": r.transcriptid, "summary": summary,
                     "words_in": n_in, "words_out": n_out, "passes": passes, "model": MODEL,
                     "seconds": round(time.time() - t0, 1)})
        print(f"[{i}/{len(df)}] {r.ticker}: {n_out} words "
              f"({n_in / max(1, n_out):.0f}x shorter, {passes} pass{'es' if passes > 1 else ''}, "
              f"{rows[-1]['seconds']}s)")
        if len(rows) >= 10:
            done, rows = save_append(out_path, done, rows), []
    if rows:
        save_append(out_path, done, rows)

if __name__ == "__main__":
    main()
