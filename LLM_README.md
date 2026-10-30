# LLM step: summarize, score, evaluate

Reads the pipeline's final dataset (see `PIPELINE.md`), asks a local LLM to summarize each earnings call and predict
its stock reaction, then checks how much of the abnormal return the prediction explains, following Koijen & Levy.

The code is in `src/llm_based_earnings_surprise_ntam/llm/`. Run every command from the repo root.

---

## 1. Before you start

1. **Run the pipeline** so the final dataset exists:
   ```bash
   uv run doit
   ```
2. **Install the environment** (once): `uv sync`
3. **Start Ollama** (open the Ollama app, or run `ollama serve` in another terminal) and download the model:
   ```bash
   ollama pull qwen3:8b
   ```
   The model is set in `llm/llm_common.py` (`MODEL`). Use `qwen3:4b` on a Mac with 8 GB RAM.

---

## 2. Pilot run (check everything works)

`--n 30` picks 30 calls from the first half of the quarter. The pick is random but fixed (seed 0), so the
summarizer and the scorer always use the same calls.

```bash
uv run python -m llm_based_earnings_surprise_ntam.llm.llm_summarize --n 30
uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --n 30 --source summary
uv run python -m llm_based_earnings_surprise_ntam.llm.llm_eval
```

Read a few summaries before the full run:
```python
import pandas as pd
pd.read_parquet("data/processed/llm_summaries.parquet").summary.head(3).tolist()
```

---

## 3. Full run (all in-sample calls)

Leave out `--n` to process every call in the estimation sample (`in_sample == True`, 2,584 calls for Q4 2025 US).
`caffeinate -i` keeps the Mac awake while the command runs.

```bash
caffeinate -i uv run python -m llm_based_earnings_surprise_ntam.llm.llm_summarize
caffeinate -i uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --source summary
uv run python -m llm_based_earnings_surprise_ntam.llm.llm_eval
```

On a MacBook Air with `qwen3:8b`, one summary took about 30 seconds and one score about 4 seconds, so the full
summarize run takes roughly a day. You can stop with Ctrl+C at any time and rerun the same command: it skips
calls already saved (it saves every 10 calls).

---

## 4. Scoring other text

`--source` chooses what the scorer reads:

| `--source` | Text |
|---|---|
| `full_text` (default) | the whole transcript |
| `presentation_text` | prepared remarks only |
| `qa_text` | Q&A only |
| `summary` | the LLM summary (run `llm_summarize` first) |

```bash
uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --n 30 --source full_text
uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --n 30 --source qa_text
```

Scores for each source and model are kept side by side. `llm_eval` reports each combination separately:
R² of the surprise alone, the LLM prediction alone, and both together.

---

## 5. Anonymization

Before the LLM sees a transcript, `anonymize()` in `llm/llm_common.py` hides who the call is about. This lowers the chance that
the model recognizes the company and recalls how its stock moved (look-ahead bias). It runs on every source except
`summary`, which is already written from anonymized text. The rules are in `llm/rp_anonymizer.py`:

| What | Replaced with | How |
|---|---|---|
| Speaker names | `[EXECUTIVE 1]`, `[ANALYST 2]`, ... | Names from the `Name (Role):` lines, numbered in order of first appearance. The same label is used wherever the name appears, also mid-sentence. |
| Speaker first / last names | the speaker's label | e.g. "Thanks, Maria" -> "Thanks, [EXECUTIVE 1]". Skips parts shorter than 4 letters and common words like "Mark" or "Will". |
| Company name | `[COMPANY]` | Full legal name, plus the short name without "The" and suffixes (Inc., Corp., Holdings, ...). |
| Ticker | `[TICKER]` | Whole-word matches only. |
| Years (1900-2099) | `[YEAR]` | Done in `llm_common.py`. |

Example:
```text
Before: Maria Lopez (Executive): At Example, revenue grew 5% and EXMP raised guidance for 2026.
        John Smith (Analyst): Maria, how should we think about margins at Example Corp.?
After:  [EXECUTIVE 1] (Executive): At [COMPANY], revenue grew 5% and [TICKER] raised guidance for [YEAR].
        [ANALYST 1] (Analyst): [EXECUTIVE 1], how should we think about margins at [COMPANY]?
```

Limits:
- Short company names that are also ordinary words (e.g. "Target", "Block") are replaced only when written
  Capitalized or in UPPER CASE, so the lower-case word "target" stays.
- Product names, brands and places are not removed, so a well-known company can still be recognizable.
- Operator lines are kept as they are.

To check one transcript yourself (`stats["leaks"]` counts names the rules missed, which should be close to 0):
```python
from llm_based_earnings_surprise_ntam.llm.rp_anonymizer import anonymize
text_anon, stats = anonymize(text, company="Example Corp.", ticker="EXMP")
```

---

## 6. Outputs

Both files are in `data/processed/` (not tracked by git) and join to the pipeline dataset on `transcriptid`.

| File | One row per | Columns |
|---|---|---|
| `llm_summaries.parquet` | call | `transcriptid, summary, words_in, words_out, model, seconds` |
| `llm_scores.parquet` | call x source x model | `transcriptid, keydevid, ticker, event_date, prediction, reasons, model, source, seconds` |

```python
import pandas as pd
df = pd.read_parquet("data/processed/earnings_calls_ibes_20251001_20251231_us.parquet")
scores = pd.read_parquet("data/processed/llm_scores.parquet")
both = df.merge(scores[["transcriptid", "model", "source", "prediction", "reasons"]], on="transcriptid")
```

---

## 7. Things to know

- **Changed a prompt or the anonymizer?** Delete the old output first, or calls already done are skipped and keep
  the old result: `rm data/processed/llm_summaries.parquet` (or `llm_scores.parquet`).
- **Text is anonymized** before the LLM sees it: speaker names, company name, ticker and years are replaced
  (see section 5).
- **The data period follows `config.yaml`**, the same as the pipeline. Change the period there and rerun
  `uv run doit` first.
- **`llm_eval` with very few calls** prints warnings and `nan` R². That is expected: a regression needs many calls.
- **`ModuleNotFoundError: src...`** means an old command. The module name starts with
  `llm_based_earnings_surprise_ntam.llm.`, never `src.`.
- **Connection error** means Ollama is not running.
