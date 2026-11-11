"""Summarize, then score the summaries, for every in-sample call with Qwen on a GPU (vLLM batch inference).

Batch version of `llm_summarize` + `llm_score --source summary` for the RCC Midway cluster (see LLM_README.md,
section 8). Same prompts, same anonymization, same output files (`llm_summaries.parquet`, `llm_scores.parquet`);
rows are tagged with the Hugging Face model id (e.g. `Qwen/Qwen3-8B`) so they sit next to the Ollama runs and
`llm_eval` reports them separately.

Run from the repo root on a GPU node (needs `vllm`, Linux + NVIDIA GPU):
  python -m llm_based_earnings_surprise_ntam.llm.llm_vllm --n 30     # pilot (same calls as the Ollama pilot)
  python -m llm_based_earnings_surprise_ntam.llm.llm_vllm            # all in-sample calls

Resumable: calls already summarized / scored with this model are skipped, and results are saved after every
chunk of --chunk calls, so a job killed at its time limit can simply be resubmitted.
"""
import argparse, json, time

import pandas as pd
from vllm import LLM, SamplingParams

from .llm_common import MERGED, SUMMARIES, SCORES, anonymize, clean, select_events, load_done, save_append
from .llm_summarize import (SYSTEM as SUM_SYSTEM, USER_TEMPLATE as SUM_USER, SHORTEN, COMPRESSION,
                            target_length)
from .llm_score import SYSTEM as SCORE_SYSTEM, USER_TEMPLATE as SCORE_USER

PROMPT_MARGIN = 1500      # tokens reserved for the instructions around the transcript
SCORE_MAX_TOKENS = 512


def generate(llm, convs, max_tokens, temperature=0.0):
    """One batched pass over many conversations; thinking off, as in the Ollama runs (think=False)."""
    params = [SamplingParams(temperature=temperature, max_tokens=int(m), seed=0) for m in max_tokens]
    outs = llm.chat(convs, params, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
    return [clean(o.outputs[0].text) for o in outs]


def fit(tok, text, budget):
    """Cut text to `budget` tokens (keeps the start). Returns (text, truncated?)."""
    ids = tok.encode(text, add_special_tokens=False)
    return (text, False) if len(ids) <= budget else (tok.decode(ids[:budget]), True)


def parse(raw):
    """Same parsing as llm_score.score: first {...} block, prediction clipped to [0, 1]."""
    try:
        out = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        out["prediction"] = min(max(float(out["prediction"]), 0.0), 1.0)
        return out
    except Exception:
        return None


def summarize(llm, df, model, max_len, chunk):
    tok = llm.get_tokenizer()
    done = load_done(SUMMARIES)
    if not done.empty:
        df = df[~df.transcriptid.isin(done.loc[done.model == model, "transcriptid"])]
    print(f"{len(df)} calls to summarize with {model}")

    for start in range(0, len(df), chunk):
        part, t0 = df.iloc[start:start + chunk], time.time()
        texts = [anonymize(r.full_text, r.companyname, r.ticker) for r in part.itertuples(index=False)]
        limits = [target_length(t) for t in texts]                   # (target_words, max_words), from full text
        convs, cut = [], []
        for text, (target, mx) in zip(texts, limits):
            text, was_cut = fit(tok, text, max_len - PROMPT_MARGIN - mx * 3)
            cut.append(was_cut)
            convs.append([{"role": "system", "content": SUM_SYSTEM},
                          {"role": "user", "content": SUM_USER.format(
                              target_words=target, max_words=mx, compression=COMPRESSION, transcript=text)}])
        outs = generate(llm, convs, [mx * 3 for _, mx in limits])

        # one shortening pass for summaries over the hard limit (as in llm_summarize)
        long = [i for i, (o, (_, mx)) in enumerate(zip(outs, limits)) if len(o.split()) > mx]
        if long:
            redo = generate(llm, [convs[i] + [
                {"role": "assistant", "content": outs[i]},
                {"role": "user", "content": SHORTEN.format(n_words=len(outs[i].split()),
                                                           target_words=limits[i][0])}] for i in long],
                [limits[i][1] * 3 for i in long])
            for i, o in zip(long, redo):
                outs[i] = o or outs[i]

        secs = round((time.time() - t0) / len(part), 2)
        rows = [{"transcriptid": r.transcriptid, "summary": o, "words_in": len(t.split()),
                 "words_out": len(o.split()), "model": model, "seconds": secs, "truncated": c}
                for r, t, o, c in zip(part.itertuples(index=False), texts, outs, cut) if o]
        done = save_append(SUMMARIES, done, rows)
        print(f"summaries: {start + len(part)}/{len(df)} done ({len(part) - len(rows)} empty, retried next run; "
              f"{len(long)} shortened; {sum(cut)} truncated)")


def score(llm, df, model, chunk):
    summ = load_done(SUMMARIES)
    if summ.empty:
        raise SystemExit("no summaries to score")
    summ = summ[summ.model == model].drop_duplicates("transcriptid", keep="last")
    df = df.merge(summ[["transcriptid", "summary"]], on="transcriptid")
    done = load_done(SCORES)
    if not done.empty:
        mine = done[(done.source == "summary") & (done.model == model)]
        df = df[~df.transcriptid.isin(mine.transcriptid)]
    print(f"{len(df)} summaries to score with {model}")

    for start in range(0, len(df), chunk):
        part, t0 = df.iloc[start:start + chunk], time.time()
        convs = [[{"role": "system", "content": SCORE_SYSTEM},
                  {"role": "user", "content": SCORE_USER.format(text=s, numbers="")}] for s in part.summary]
        res = [parse(o) for o in generate(llm, convs, [SCORE_MAX_TOKENS] * len(convs))]
        bad = [i for i, r in enumerate(res) if r is None]
        if bad:   # greedy decoding would repeat the same output, so retry with a little sampling
            for i, o in zip(bad, generate(llm, [convs[i] for i in bad], [SCORE_MAX_TOKENS] * len(bad), 0.3)):
                res[i] = parse(o)
        secs = round((time.time() - t0) / len(part), 2)
        rows = [{"transcriptid": r.transcriptid, "keydevid": r.keydevid, "ticker": r.ticker,
                 "event_date": r.event_date,
                 "prediction": out["prediction"] if out else None,
                 "reasons": json.dumps(out.get("reasons", []) if out else ["failed"]),
                 "model": model, "source": "summary", "seconds": secs}
                for r, out in zip(part.itertuples(index=False), res)]
        done = save_append(SCORES, done, rows)
        n_fail = sum(out is None for out in res)
        print(f"scores: {start + len(part)}/{len(df)} done ({n_fail} unparseable after retry)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="pilot size (early-quarter calls, same pick as the Ollama scripts)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="Hugging Face model id")
    ap.add_argument("--max-model-len", type=int, default=32768, help="context length (Ollama runs: 32768)")
    ap.add_argument("--chunk", type=int, default=256, help="calls per batch between saves")
    ap.add_argument("--dtype", default="auto", help="'half' on GPUs without bf16 (V100)")
    ap.add_argument("--tp", type=int, default=1, help="number of GPUs (tensor parallel)")
    ap.add_argument("--skip-summarize", action="store_true", help="only score existing summaries")
    args = ap.parse_args()

    df = select_events(pd.read_parquet(MERGED), args.n)
    llm = LLM(model=args.model, max_model_len=args.max_model_len, dtype=args.dtype,
              tensor_parallel_size=args.tp, gpu_memory_utilization=0.90, seed=0)
    if not args.skip_summarize:
        summarize(llm, df, args.model, args.max_model_len, args.chunk)
    score(llm, df, args.model, args.chunk)


if __name__ == "__main__":
    main()
