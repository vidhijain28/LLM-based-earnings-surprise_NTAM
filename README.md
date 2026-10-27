# LLM-based-earnings-surprise_NTAM
Develop a more robust estimate of the surprise element in an earnings release using large language models and fundamental data. Then build a short-term alpha signal that can be incorporated as a screen or as a targeted alpha factor. 

## Data pipeline: CIQ transcripts + I/B/E/S surprise

One row per earnings call: point-in-time transcript text matched to its I/B/E/S announcement (actual, consensus,
SUE, event date). Quick start:

```bash
uv sync
uv run doit                                    # period / universe from config.yaml (default: Q4 2025, US)
uv run doit start=2024-01-01 end=2024-12-31    # another period
```

**Where the data goes.** Everything is written under `data/` in the repo root (not tracked by git). File names
carry the period and universe, so runs for different periods sit side by side:

```
data/
├── processed/                                              <- use these
│   ├── earnings_calls_ibes_<start>_<end>_<universe>.parquet        final dataset, one row per call
│   └── earnings_calls_ibes_<start>_<end>_<universe>_report.json    match statistics + config used
├── interim/                                                <- intermediate steps
│   ├── calls_<start>_<end>_<universe>.parquet                      cleaned call metadata
│   ├── transcript_text_<start>_<end>_<universe>.parquet            transcript text
│   ├── ibes_announcements_<start>_<end>_<universe>.parquet         IBES actual, consensus, SUE
│   └── ibes_links_<start>_<end>_<universe>.parquet                 call -> IBES ticker candidates
└── raw/                                                    <- monthly WRDS pulls (cache)
    └── ciq_transcripts_<universe>_<version_rule>_<YYYY-MM>.parquet
```

`<start>` and `<end>` are written as `YYYYMMDD`. For example, the default run (Q4 2025, US) produces
`data/processed/earnings_calls_ibes_20251001_20251231_us.parquet`. To load it:

```python
import pandas as pd
df = pd.read_parquet('data/processed/earnings_calls_ibes_20251001_20251231_us.parquet')
sample = df[df.sue.notna()]   # calls matched to IBES with a surprise
```

The output folders can be changed under `paths:` in `config.yaml`.

See **[PIPELINE.md](PIPELINE.md)** for the process, the data decisions (what is kept and dropped), the output columns
and the known limitations.

## LLM step: summarize, score, evaluate

Reads the final dataset above, has a local LLM (Ollama) summarize each call and predict its stock reaction, and
measures how much of the abnormal return the prediction explains beyond the surprise.

See **[LLM_README.md](LLM_README.md)** for the commands (pilot run, full run, text sources) and the output files.
