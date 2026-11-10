"""Shared settings and helpers for the LLM steps."""
import re
from pathlib import Path

import pandas as pd

from ..pipeline.config import load_config, paths
from .rp_anonymizer import anonymize as rp_anonymize

# same period and universe as the pipeline (config.yaml), whatever folder you run from
_CFG = load_config()
_P = paths(_CFG)
MERGED = Path(_P["final"])
SUMMARIES = MERGED.parent / "llm_summaries.parquet"
SCORES = MERGED.parent / "llm_scores.parquet"
_LLM = _CFG.get("llm") or {}
MODEL = _LLM.get("model", "qwen3:8b")     # config.yaml `llm` section; e.g. qwen3:8b for a laptop pilot
SOURCE = _LLM.get("source", "summary")

def require_ollama(model):
    """Fail early, with a clear message, when the Ollama server or the model is not there."""
    import ollama
    try:
        have = {m.model for m in ollama.list().models}
    except Exception as e:
        raise SystemExit(f"Ollama is not running ({e}). Start it with `ollama serve`, or run on Midway.")
    if model not in have and f"{model}:latest" not in have:
        raise SystemExit(f"Ollama has no model {model!r}: `ollama pull {model}` (or change llm.model in config.yaml)")

def anonymize(text, company=None, ticker=None):
    """Mask speaker names, company name, ticker (rp_anonymizer) and years."""
    out, _ = rp_anonymize(text or "",
                          company=company if isinstance(company, str) else "",
                          ticker=ticker if isinstance(ticker, str) else None)
    return re.sub(r"\b(19|20)\d{2}\b", "[YEAR]", out)

def clean(raw):
    """Strip qwen3 <think> blocks."""
    return re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()

def select_events(df, n=None, seed=0):
    """In-sample events; with n, a fixed random pilot from the first half of the quarter."""
    df = df[df["in_sample"]].copy()
    if n:
        early = df[df["event_date"] <= df["event_date"].median()]
        df = early.sample(n=min(n, len(early)), random_state=seed)
    return df

def load_done(path):
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

def save_append(path, done, rows):
    """Append rows and write safely (temp file, then rename)."""
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    tmp = path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    return out