"""Shared settings and helpers for the LLM steps."""
import os, re
from pathlib import Path

import pandas as pd

from ..pipeline.config import load_config, paths
from .rp_anonymizer import anonymize as rp_anonymize

# same period and universe as the pipeline (config.yaml), whatever folder you run from
_P = paths(load_config())
MERGED = Path(_P["final"])
# LLM_TAG=xyz writes llm_summaries_xyz.parquet etc., to test a change without touching the main files
_TAG = f"_{os.environ['LLM_TAG']}" if os.environ.get("LLM_TAG") else ""
SUMMARIES = MERGED.parent / f"llm_summaries{_TAG}.parquet"
SCORES = MERGED.parent / f"llm_scores{_TAG}.parquet"
MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")   # Midway: qwen3:30b-a3b-instruct-2507-q4_K_M
NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "32768"))     # Midway: 24576

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

def select_events(df, n=None, seed=0, shard=None, nshards=1):
    """In-sample events; with n, a fixed random pilot from the first half of the quarter;
    with shard, only this job's share of the calls."""
    df = df[df["in_sample"]].copy()
    if n:
        early = df[df["event_date"] <= df["event_date"].median()]
        df = early.sample(n=min(n, len(early)), random_state=seed)
    if shard is not None:
        df = df[df["transcriptid"].astype("int64") % nshards == shard]
    return df

def shard_path(path, shard=None):
    """llm_scores.parquet -> llm_scores_shard2.parquet for job 2."""
    return path if shard is None else path.with_name(f"{path.stem}_shard{shard}{path.suffix}")

def load_done(path):
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

def load_all(path):
    """The main file plus any shard files."""
    files = [path] if path.exists() else []
    files += sorted(f for f in path.parent.glob(f"{path.stem}_shard*{path.suffix}")
                    if ".tmp" not in f.name)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()

def save_append(path, done, rows):
    """Append rows and write safely (temp file, then rename)."""
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    tmp = path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    return out
