"""Shared settings and helpers for the LLM steps."""
import os, re
from pathlib import Path

import pandas as pd

from ..pipeline.config import load_config, paths
from .rp_anonymizer import anonymize as rp_anonymize

# same period and universe as the pipeline (config.yaml), whatever folder you run from
_CFG = load_config()
_P = paths(_CFG)
MERGED = Path(_P["final"])
# LLM_TAG=xyz writes llm_summaries_xyz.parquet etc., to test a change without touching the main files
_TAG = f"_{os.environ['LLM_TAG']}" if os.environ.get("LLM_TAG") else ""
SUMMARIES = MERGED.parent / f"llm_summaries{_TAG}.parquet"
SCORES = MERGED.parent / f"llm_scores{_TAG}.parquet"
_LLM = _CFG.get("llm") or {}
# config.yaml `llm` section; e.g. qwen3:8b for a laptop pilot. The env vars override it (run_llm.sh on Midway).
MODEL = os.environ.get("LLM_MODEL") or _LLM.get("model", "qwen3:8b")
SOURCE = _LLM.get("source", "summary")
NUM_CTX = int(os.environ.get("LLM_NUM_CTX") or _LLM.get("num_ctx", 32768))   # Midway RTX 6000: 24576

# Dev set for prompt experiments (llm_devset): random calls from the early half of the quarter only
DEV_IDS = MERGED.parent / "dev300.csv"

# One scoring run = model x source x variant x temperature x samples. Rows written before the last three
# existed are the base run: variant "base", temperature 0, one sample.
RUN_KEYS = ["model", "source", "variant", "temperature", "samples"]
RUN_DEFAULTS = {"variant": "base", "temperature": 0.0, "samples": 1}

def with_run_defaults(scores):
    """Fill variant / temperature / samples on score rows that predate them."""
    scores = scores.copy()
    for col, value in RUN_DEFAULTS.items():
        scores[col] = scores[col].fillna(value) if col in scores else value
    return scores.astype({"temperature": float, "samples": int}) if len(scores) else scores

def read_ids(path):
    """transcriptids listed in a csv with a `transcriptid` column (e.g. dev300.csv)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} not found: make it with `uv run python -m llm_based_earnings_surprise_ntam.llm.llm_devset`")
    return set(pd.read_csv(path, usecols=["transcriptid"])["transcriptid"].astype("int64"))

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
    """Strip qwen3 reasoning: <think>...</think> blocks, and anything before a lone </think>
    (thinking-only models get <think> from the template, so only the closing tag is output)."""
    out = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    return out.split("</think>")[-1].strip()

def select_events(df, n=None, seed=0, shard=None, nshards=1, ids=None):
    """In-sample events; with n, a fixed random pilot from the first half of the quarter;
    with ids, only those transcriptids; with shard, only this job's share of the calls."""
    df = df[df["in_sample"]].copy()
    if n:
        early = df[df["event_date"] <= df["event_date"].median()]
        df = early.sample(n=min(n, len(early)), random_state=seed)
    if ids is not None:
        df = df[df["transcriptid"].astype("int64").isin(ids)]
    if shard is not None:
        df = df[df["transcriptid"].astype("int64") % nshards == shard]
    return df

def shard_path(path, shard=None):
    """llm_scores.parquet -> llm_scores_shard2.parquet for job 2."""
    path = Path(path)
    return path if shard is None else path.with_name(f"{path.stem}_shard{shard}{path.suffix}")

def load_done(path):
    return pd.read_parquet(path) if Path(path).exists() else pd.DataFrame()

def load_all(path):
    """The main file plus any shard files (written by parallel GPU jobs, see run_llm.sh)."""
    path = Path(path)
    files = [path] if path.exists() else []
    files += sorted(f for f in path.parent.glob(f"{path.stem}_shard*{path.suffix}")
                    if ".tmp" not in f.name)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()

def save_append(path, done, rows):
    """Append rows and write safely (temp file, then rename)."""
    path = Path(path)
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    tmp = path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    return out
