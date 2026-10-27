# Setup (once): install Ollama from ollama.com, then in terminal: ollama pull qwen3:8b
# Usage: python summarize.py
# Reads every transcripts/*.txt and writes summarized_transcripts/<same name>.txt.
# Transcripts that already have a summary are skipped, so only new ones are processed.

import os, re, glob, time
import ollama

MODEL = "qwen3:8b"            # use "qwen3:4b" if your Mac has 8 GB RAM
TRANSCRIPT_FOLDER = "transcripts"
SUMMARY_FOLDER = "summarized_transcripts"
COMPRESSION = 28              # paper: summaries are ~28x shorter than the full transcript
MIN_WORDS = 150               # floor so very short transcripts still get a usable summary

# ---------- 1. The prompt ----------
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

# ---------- 2. Summarize one transcript ----------
def target_length(text):
    n_words = len(text.split())
    target = max(MIN_WORDS, round(n_words / COMPRESSION))
    return target, round(target * 1.3)

def clean(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)  # qwen3 may emit reasoning
    return raw.strip()

def chat(messages, max_words):
    resp = ollama.chat(
        model=MODEL,
        messages=messages,
        think=False,
        options={"temperature": 0, "num_ctx": 32768,   # room for long transcripts
                 "num_predict": max_words * 3},         # token cap ~ word limit
    )
    return clean(resp["message"]["content"])

def summarize_transcript(transcript, retries=3):
    target_words, max_words = target_length(transcript)
    prompt = USER_TEMPLATE.format(target_words=target_words, max_words=max_words,
                                  compression=COMPRESSION, transcript=transcript)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
    for attempt in range(retries):
        try:
            summary = chat(messages, max_words)
            if not summary:
                raise ValueError("empty response")
            # one shortening pass if the model overshot the hard limit
            if len(summary.split()) > max_words:
                summary = chat(messages + [
                    {"role": "assistant", "content": summary},
                    {"role": "user", "content": SHORTEN.format(
                        n_words=len(summary.split()), target_words=target_words)},
                ], max_words) or summary
            return summary
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return None

# ---------- 3. Run over new transcripts only ----------
# File names must look like: AAPL_2025-10-30.txt
def run_folder(src=TRANSCRIPT_FOLDER, dst=SUMMARY_FOLDER):
    os.makedirs(dst, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(src, "*.txt")))
    todo = [p for p in paths if not os.path.exists(os.path.join(dst, os.path.basename(p)))]
    print(f"{len(paths)} transcripts, {len(paths) - len(todo)} already summarized, "
          f"{len(todo)} to do")

    for path in todo:
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            text = f.read()

        print(f"Summarizing {name} ({len(text.split()):,} words) ...")
        start = time.time()
        summary = summarize_transcript(text)
        if summary is None:
            print(f"  failed, will retry next run")
            continue

        # write to a temp file first so a crash never leaves a half-written "done" summary
        out_path = os.path.join(dst, name)
        with open(out_path + ".tmp", "w", encoding="utf-8") as f:
            f.write(summary + "\n")
        os.replace(out_path + ".tmp", out_path)

        ratio = len(text.split()) / max(1, len(summary.split()))
        print(f"  {len(summary.split())} words ({ratio:.0f}x shorter), "
              f"{time.time() - start:.0f}s -> {out_path}")

# ---------- 4. Run ----------
if __name__ == "__main__":
    run_folder()
