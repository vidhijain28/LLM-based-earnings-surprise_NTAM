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
   The model and the text the scorer reads are set in `config.yaml` (`llm: model, source`); `--model` and `--source`
   override them for one command. The full Q4 2025 run used `qwen3:30b-a3b-instruct-2507-q4_K_M` on summaries (the
   config default), on Midway GPUs (section 8). Use `qwen3:8b` for a laptop pilot, `qwen3:4b` on a Mac with 8 GB RAM.

---

## 2. Pilot run (check everything works)

`--n 30` picks 30 calls from the first half of the quarter. The pick is random but fixed (seed 0), so the
summarizer and the scorer always use the same calls.

```bash
uv run python -m llm_based_earnings_surprise_ntam.llm.llm_summarize --n 30 --model qwen3:8b
uv run python -m llm_based_earnings_surprise_ntam.llm.llm_score --n 30 --source summary --model qwen3:8b
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

**Or let `doit` do it.** `uv run doit` runs the pipeline, then `llm_summarize`, `llm_score`, `llm_merge`,
and `llm_eval` with the model and source in `config.yaml`. The LLM
tasks only send calls that are not done yet, so once everything is scored the whole run takes seconds and needs no
Ollama. `doit clean` never deletes the summaries or scores. On Midway the LLM part runs as GPU jobs instead
(section 8).

---

## 4. Scoring other text

`--source` chooses what the scorer reads (default: `llm.source` in `config.yaml`, which is `summary`):

| `--source` | Text |
|---|---|
| `full_text` | the whole transcript |
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
the model recognizes the company and recalls how its stock moved (look-ahead bias). It runs on every source,
`summary` included: summaries are written from anonymized text, but the model sometimes writes a year back in
("recovery in 2025"), so they are masked again before scoring. The rules are in `llm/rp_anonymizer.py`:

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
| `llm_summaries.parquet` | call | `transcriptid, summary, words_in, words_out, passes, model, seconds` (`passes`: 2 when the shortening round was needed) |
| `llm_scores.parquet` | call x source x model | `transcriptid, keydevid, ticker, event_date, prediction, reasons, model, source, seconds` |
| `llm_merged_<tag>.parquet` | in-sample call | the pipeline's columns (no transcript text) + summary + the configured run's `prediction` (`llm_merge`) |
| `llm_eval_<tag>.txt` | | day-0 R² of the surprise, the LLM score and both, per model and source (`llm_eval`) |

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
- **"Ollama is not running"** means what it says: open the Ollama app or run `ollama serve`.
- **Testing a change?** `LLM_TAG=test` before a command writes to `llm_summaries_test.parquet` /
  `llm_scores_test.parquet` instead, so experiments stay out of the real results.
- **Bullet budget.** The summary prompt gives each section a maximum number of bullets (e.g. "RESULTS: up to 3"),
  sized from the word target. Models copy the numbers into the headings ("RESULTS [3]"); the code removes them.
  It was added on a wrong diagnosis (the slow runs were caused by the reasoning model, section 8.4). With the
  instruct model summaries land near the target length, but the instruct model was never tested with the earlier
  prompt without bullet budgets, so it is not known whether the budget helps.

---

## 8. Running on Midway (GPU)

The full Q4 2025 run was made on the RCC Midway3 cluster. `run_llm.sh` starts 4 GPU jobs; each one starts its own
Ollama server on its GPU node, then summarizes and scores a quarter of the calls. The whole run takes about
2 hours 15 minutes. Run every command below on a Midway3 login node.

### 8.1 One-time setup

Already done in the shared folder `/project/finm36000/ntam_llm` (`$NTAM`): `ollama-0.24.0/` (Ollama),
`models/` (model files), `uv-cache/`, `repo/` (this repository). To set it up again:

```bash
export NTAM=/project/finm36000/ntam_llm

# Ollama 0.24.0 (not a newer one, see 8.4)
mkdir -p $NTAM/ollama-0.24.0 && cd $NTAM/ollama-0.24.0
curl -sL -o ollama.tar.zst https://github.com/ollama/ollama/releases/download/v0.24.0/ollama-linux-amd64.tar.zst
tar --use-compress-program=zstd -xf ollama.tar.zst && rm ollama.tar.zst

# the model (19 GB) into the shared folder
export OLLAMA_MODELS=$NTAM/models OLLAMA_HOST=127.0.0.1:11500
$NTAM/ollama-0.24.0/bin/ollama serve > /dev/null 2>&1 &
sleep 5
$NTAM/ollama-0.24.0/bin/ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
kill %1

# the Python environment (the jobs use it as is: uv run --no-sync)
cd $NTAM/repo && UV_CACHE_DIR=$NTAM/uv-cache uv sync
```

The final dataset must be in `data/processed/`: build it on Midway (next step), or copy it from a laptop with
`scp`. It is licensed data, so it is never in git.

### 8.2 How to run

```bash
cd /project/finm36000/ntam_llm/repo
uv run doit sample
sbatch run_llm.sh
squeue -u $USER
uv run python -m llm_based_earnings_surprise_ntam.llm.llm_combine
uv run doit
```

1. `uv run doit sample` builds the dataset (needs a WRDS login on Midway).
2. `sbatch run_llm.sh` submits the 4 GPU jobs and prints a job number. The jobs keep running when you log out.
3. `squeue -u $USER` lists your jobs: `PD` waiting for a GPU, `R` running. Wait until it lists none.
4. `llm_combine` joins the 4 jobs' shard files into `llm_summaries.parquet` and `llm_scores.parquet`. Run it only
   after all 4 jobs have finished.
5. `uv run doit` runs `llm_merge`, `llm_eval` and the daily returns. The LLM tasks find every call done, so they
   take seconds and need no GPU.

If the dataset was copied from a laptop instead of built on Midway, `doit` has no record of the earlier steps and
would rebuild everything from WRDS. Run the last two LLM steps directly instead:
```bash
uv run python -m llm_based_earnings_surprise_ntam.llm.llm_merge
uv run python -m llm_based_earnings_surprise_ntam.llm.llm_eval
```

**While the jobs run** (`<job>` is the number `sbatch` printed):
```bash
tail -f logs/llm_<job>_0.out
grep "inference compute" logs/ollama_<job>.log
sacct -j <job> -X --format=JobID,Elapsed,State,AllocTRES%45
```
- `logs/llm_<job>_0.out` is job 0's log (`_1`, `_2`, `_3` for the others), one line per call, e.g.
  `[12/646] XYZ: 206 words (32x shorter, 1 pass, 11.2s)`. `2 passes` means the shortening round was needed.
- The `ollama` log must say `library=CUDA` and `Quadro RTX 6000`: the model is on the GPU.
- `sacct` shows each job's time and state, and `gres/gpu=1` when it had a GPU.

**Stopping and restarting.** Each job saves every 10 calls. Cancel with `scancel <job>`. If a job stops early (time
limit, node failure, cancelled), submit `run_llm.sh` again: calls already done are skipped.

**Pilot.** `PILOT=10 LLM_TAG=test sbatch run_llm.sh` runs 10 calls in total, split across the 4 jobs, and writes
`llm_summaries_test.parquet` / `llm_scores_test.parquet`, so the real files are not touched. Combine and evaluate
with the same tag: `LLM_TAG=test uv run python -m llm_based_earnings_surprise_ntam.llm.llm_combine`.

### 8.3 Settings

Set before `sbatch`, e.g. `LLM_MODEL=qwen3:8b sbatch run_llm.sh`. No code changes needed.

| Variable | Default | What |
|---|---|---|
| `LLM_MODEL` | `qwen3:30b-a3b-instruct-2507-q4_K_M` | Ollama model; overrides `llm.model` in `config.yaml` |
| `LLM_NUM_CTX` | `24576` | context window in tokens, set for the RTX 6000's 24 GB (laptop default: 32768) |
| `NSHARDS` | `4` | number of jobs the calls are split into; change `--array` to match: `NSHARDS=8 sbatch --array=0-7 run_llm.sh` |
| `PILOT` | not set | test on this many calls in total |
| `LLM_TAG` | not set | write to `llm_*_<tag>.parquet`, so tests stay out of the real results |
| `NTAM` | `/project/finm36000/ntam_llm` | shared folder with Ollama, the model files and the repo |

Each job gets one Quadro RTX 6000 (24 GB), 4 CPUs, 32 GB of memory and up to 6 hours, on account `finm36000`
(the `#SBATCH` lines at the top of `run_llm.sh`).

### 8.4 Why these settings

| Setting | What we use | Why |
|---|---|---|
| Ollama version | 0.24.0 | Newer Ollama (0.40.15 was installed) needs NVIDIA driver 550 or newer. Midway's GPU nodes have 535, so Ollama quietly ran the model on the CPU, which would take hours for 30 calls ([ollama#16449](https://github.com/ollama/ollama/issues/16449)). With 0.24.0 all 49 layers run on the GPU. |
| GPU check | the job stops unless Ollama reports `library=CUDA` | So a job can never run on the CPU unnoticed again. |
| Model | `qwen3:30b-a3b-instruct-2507-q4_K_M` | The biggest gain in speed and quality. Plain `qwen3:30b-a3b` is now Qwen's reasoning-only model: it ignores `think=False` and writes about 350 words of "let me think..." before each answer. The instruct model answers directly, so it writes about a third as many words: about 30 s down to 11-13 s per summary, and summaries that are actual summaries. |
| One round per summary | the "shorten" round is only a fallback | With the reasoning model, the reasoning was counted as part of the summary, so every call looked too long and the model wrote about 500 words a second time. With the instruct model 98% of calls finish in one round. |
| Reasoning cleanup | `clean()` in `llm_common.py` also removes text that ends with a lone `</think>` | Reasoning models get the opening `<think>` from the chat template, so only the closing tag appears in the output. Before this fix the reasoning was saved as the summary: 3 of 10 pilot calls had no real summary at all. |
| 4 parallel jobs | `--array=0-3`, calls split by `transcriptid % 4` | Each job does a quarter of the calls and writes its own `*_shard<k>.parquet`, so jobs never write to the same file. `llm_combine` joins them at the end. |
| One port per job | `OLLAMA_HOST` port from the job number | Two jobs on the same node would otherwise share one Ollama server. |
| Live progress | `PYTHONUNBUFFERED=1` | Without it Python holds back its output and the log looks empty. |

### 8.5 The Q4 2025 run

- **Calls:** all 2,584 in-sample calls, summarized and then scored on the summaries with
  `qwen3:30b-a3b-instruct-2507-q4_K_M`. No failed calls.
- **Time:** 4 jobs of 2 h 08 min to 2 h 24 min, one Quadro RTX 6000 each. A summary took a median 11 s, a score 1 s.
- **Summaries:** 2,529 of 2,584 (98%) in one round. Median 206 words, 32 times shorter than the transcript
  (target: 28 times, at least 150 words).
- **Day-0 R²** (`llm_eval`, N = 2,584): surprise alone 4.7%, LLM alone 7.2%, both 10.2%. With both in the
  regression the LLM score has a t-statistic of 12.5 (robust standard errors). Its correlation with the surprise
  rank is 0.17, so it carries information the surprise does not.
- **Coarse scores:** only 14 distinct values, and 76% of calls are above 0.5. Sorting into fifths needs a
  tie-break rule.
