"""Shared settings and helpers for the LLM steps.

Two backends, chosen per run with --backend (or $LLM_BACKEND):
  ollama      local model through the Ollama app (default; free)
  openrouter  any hosted model through OpenRouter (paid; key in .env as OPENROUTER_API_KEY)
"""
import os, re, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

from ..pipeline.config import ROOT, load_config, paths
from .rp_anonymizer import anonymize as rp_anonymize

# same period and universe as the pipeline (config.yaml), whatever folder you run from
_CFG = load_config()
_P = paths(_CFG)
MERGED = Path(_P["final"])
SUMMARIES = MERGED.parent / "llm_summaries.parquet"
SCORES = MERGED.parent / "llm_scores.parquet"
LLM_MERGED = MERGED.parent / f"llm_merged_{_CFG['tag']}.parquet"   # scored panel from an earlier run
MODEL = "qwen3:8b"   # Ollama default; "qwen3:4b" if your Mac has 8 GB RAM
NUM_CTX = 32768

OPENROUTER = "https://openrouter.ai/api/v1"

# Settings for this run, filled in by configure()
RUN = {"backend": "ollama", "model": MODEL, "reasoning": None, "max_cost": None, "workers": 1,
       "spent": 0.0, "calls": 0, "price": None}
_lock = threading.Lock()
_client = None


# ---------- command line ----------

def add_llm_args(ap):
    """Backend options shared by llm_summarize and llm_score."""
    ap.add_argument("--backend", choices=["ollama", "openrouter"], default=os.environ.get("LLM_BACKEND", "ollama"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"),
                    help=f"Ollama tag (default {MODEL}) or OpenRouter id, e.g. google/gemini-2.5-flash-lite")
    ap.add_argument("--max-cost", type=float, help="OpenRouter: stop once this many dollars are spent in this run")
    ap.add_argument("--workers", type=int, help="parallel requests (default 1 for ollama, 8 for openrouter)")
    ap.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"],
                    help="OpenRouter reasoning models only (e.g. GPT-5 nano): keep it low, thinking tokens are billed")


def configure(args):
    """Apply the backend options; for OpenRouter, load the key and the model's price."""
    RUN.update(backend=args.backend, reasoning=args.reasoning_effort, max_cost=args.max_cost,
               model=args.model or MODEL, spent=0.0, calls=0)
    if args.backend == "openrouter":
        if not args.model:
            raise SystemExit("--model is required with --backend openrouter (e.g. google/gemini-2.5-flash-lite)")
        _load_dotenv()
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit("OPENROUTER_API_KEY not found: put it in .env at the repo root")
        RUN["price"] = _price(args.model)
        if args.max_cost is None:
            print("WARNING: no --max-cost set; the run is limited only by your OpenRouter credit")
    RUN["workers"] = args.workers or (8 if args.backend == "openrouter" else 1)
    print(f"Backend {RUN['backend']} | model {RUN['model']} | workers {RUN['workers']}"
          + (f" | budget ${RUN['max_cost']:.2f}" if RUN["max_cost"] is not None else ""))


def _load_dotenv(path=ROOT / ".env"):
    """Minimal .env reader (KEY=VALUE lines); variables already set in the shell win."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _http():
    global _client
    if _client is None:
        _client = httpx.Client(base_url=OPENROUTER, timeout=300,
                               headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                                        "X-Title": "LLM-based earnings surprise (NTAM)"})
    return _client


def _price(model):
    """$ per token (prompt, completion) from OpenRouter's public model list; a fallback if a reply has no cost."""
    try:
        for m in httpx.get(f"{OPENROUTER}/models", timeout=60).json()["data"]:
            if m["id"] == model:
                p = m["pricing"]
                print(f"Price: ${float(p['prompt']) * 1e6:.3f} / ${float(p['completion']) * 1e6:.3f} "
                      f"per million input / output tokens")
                return float(p["prompt"]), float(p["completion"])
    except Exception as e:
        print(f"WARNING: could not load prices ({e})")
        return None
    raise SystemExit(f"Model {model!r} not found on OpenRouter: copy the exact id from openrouter.ai/models")


# ---------- one chat call, either backend ----------

def chat(messages, max_tokens=None, json_mode=False):
    """Returns (text, usage). usage has prompt_tokens, completion_tokens and cost ($; 0 for Ollama)."""
    if RUN["backend"] == "ollama":
        import ollama
        opts = {"temperature": 0, "num_ctx": NUM_CTX}
        if max_tokens:
            opts["num_predict"] = max_tokens
        kw = {"format": "json"} if json_mode else {}
        resp = ollama.chat(model=RUN["model"], messages=messages, think=False, options=opts, **kw)
        return clean(resp["message"]["content"]), {"prompt_tokens": resp.get("prompt_eval_count"),
                                                   "completion_tokens": resp.get("eval_count"), "cost": 0.0}

    body = {"model": RUN["model"], "messages": messages, "temperature": 0, "usage": {"include": True}}
    if max_tokens:
        body["max_tokens"] = max_tokens + (4000 if RUN["reasoning"] else 0)   # thinking counts toward the cap
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if RUN["reasoning"]:
        body["reasoning"] = {"effort": RUN["reasoning"], "exclude": True}
    r = _http().post("/chat/completions", json=body)
    if r.status_code != 200:
        raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:300]}")
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"OpenRouter error: {data['error']}")
    u = data.get("usage") or {}
    pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
    cost = u.get("cost")
    if cost is None and RUN["price"]:
        cost = pt * RUN["price"][0] + ct * RUN["price"][1]
    cost = float(cost or 0.0)
    with _lock:
        RUN["spent"] += cost
        RUN["calls"] += 1
    text = data["choices"][0]["message"].get("content") or ""
    return clean(text), {"prompt_tokens": pt, "completion_tokens": ct, "cost": cost}


def add_usage(total, u):
    for k in ("prompt_tokens", "completion_tokens", "cost"):
        total[k] = (total.get(k) or 0) + (u.get(k) or 0)
    return total


def over_budget():
    return RUN["max_cost"] is not None and RUN["spent"] >= RUN["max_cost"]


def spent_note():
    return f"${RUN['spent']:.3f} spent on {RUN['calls']} requests" if RUN["backend"] == "openrouter" else ""


def run_parallel(items, fn):
    """fn over items with RUN['workers'] threads, results in input order."""
    if RUN["workers"] <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(RUN["workers"]) as ex:
        return list(ex.map(fn, items))


def chunk_size():
    """Calls per save: small enough to lose little on a crash, big enough to keep the workers busy."""
    return max(10, 2 * RUN["workers"])


# ---------- text helpers ----------

def anonymize(text, company=None, ticker=None):
    """Mask speaker names, company name, ticker (rp_anonymizer) and years."""
    out, _ = rp_anonymize(text or "",
                          company=company if isinstance(company, str) else "",
                          ticker=ticker if isinstance(ticker, str) else None)
    return re.sub(r"\b(19|20)\d{2}\b", "[YEAR]", out)


def clean(raw):
    """Strip qwen3 <think> blocks."""
    return re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()


def earnings_numbers(r):
    """Header with the release's EPS numbers from I/B/E/S (for llm_score --with-numbers); '' if missing."""
    a, c, n, sue = (getattr(r, k, None) for k in ("actual", "consensus", "n_estimates", "sue"))
    if pd.isna(a) or pd.isna(c):
        return ""
    est = f" (mean of {int(n)} analyst estimates)" if pd.notna(n) else ""
    pct = f" ({float(sue) * 100:+.2f}% of the pre-announcement share price)" if pd.notna(sue) else ""
    return ("Earnings numbers (I/B/E/S):\n"
            f"- Reported EPS: {float(a):.2f}\n"
            f"- Consensus EPS before the release{est}: {float(c):.2f}\n"
            f"- Surprise: {float(a) - float(c):+.2f} per share{pct}\n\n")


# ---------- data helpers ----------

def select_events(df, n=None, seed=0):
    """In-sample events; with n, a fixed random pilot from the first half of the quarter."""
    if "in_sample" not in df.columns:
        raise SystemExit(f"{MERGED.name} has no in_sample column: it was built before the returns/sample steps. "
                         "Rerun the pipeline first: uv run doit")
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
