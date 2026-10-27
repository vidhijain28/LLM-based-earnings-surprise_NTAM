# Setup (once): install Ollama from ollama.com, then in terminal: ollama pull qwen3:8b
# In a notebook cell: %pip install ollama pandas

import os, re, json, time, glob
import pandas as pd
import ollama

MODEL = "qwen3:8b"            # use "qwen3:4b" if your Mac has 8 GB RAM
TRANSCRIPT_FOLDER = "transcripts"
OUTPUT_CSV = "llm_scores.csv"

# ---------- 1. The prompt ----------
SYSTEM = """You are an equity analyst predicting the stock market's reaction to an earnings call.
Use only the information in the transcript. Do not use outside knowledge about this company
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

Transcript:
<transcript>
{transcript}
</transcript>"""

# ---------- 2. Hide identity (reduces look-ahead bias) ----------
def anonymize(text, company_name=None, ticker=None):
    if company_name:
        text = re.sub(re.escape(company_name), "the Company", text, flags=re.I)
    if ticker:
        text = re.sub(rf"\b{re.escape(ticker)}\b", "the Company", text)
    text = re.sub(r"\b(19|20)\d{2}\b", "[YEAR]", text)   # hide years
    return text

# ---------- 3. Score one transcript ----------
def score_transcript(transcript, retries=3):
    for attempt in range(retries):
        try:
            resp = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER_TEMPLATE.format(transcript=transcript)},
                ],
                format="json",
                options={"temperature": 0, "num_ctx": 32768},  # room for long transcripts
            )
            raw = resp["message"]["content"]
            return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return {"surprise_score": None, "error": "failed"}

# ---------- 4. Run over a folder of .txt transcripts ----------
# File names must look like: AAPL_2025-10-30.txt
def run_folder(folder=TRANSCRIPT_FOLDER, out_csv=OUTPUT_CSV):
    os.makedirs(folder, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not paths:
        print(f"No .txt files found in '{folder}/'. Add transcripts named like AAPL_2025-10-30.txt")
        return None

    rows = []
    for path in paths:
        name = os.path.basename(path).replace(".txt", "")
        try:
            ticker, date = name.split("_", 1)
        except ValueError:
            print(f"Skipping {name}: file name must be TICKER_YYYY-MM-DD.txt")
            continue

        with open(path, encoding="utf-8") as f:
            text = f.read()
        text = anonymize(text, ticker=ticker)   # add company_name="Apple" etc. if you have it

        print(f"Scoring {ticker} {date} ({len(text):,} characters) ...")
        start = time.time()
        result = score_transcript(text)
        result.update({"ticker": ticker, "date": date,
                       "seconds": round(time.time() - start, 1)})
        rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    cols = [c for c in ["ticker", "date", "surprise_score", "guidance_tone",
                        "confidence", "seconds"] if c in df.columns]
    print(df[cols])
    print(f"\nSaved to {out_csv}")
    return df

# ---------- 5. Run ----------
df = run_folder()