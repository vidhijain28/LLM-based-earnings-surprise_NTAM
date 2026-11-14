"""Combine the parallel GPU jobs' shard files into llm_summaries.parquet and llm_scores.parquet.
Run only after ALL jobs of run_llm.sh have finished:
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_combine
Then continue as usual: llm_merge (joins with the pipeline output) and llm_eval, or `uv run doit`.
"""
from .llm_common import RUN_KEYS, SUMMARIES, SCORES, load_all, with_run_defaults


def combine(summaries=SUMMARIES, scores=SCORES):
    for path, keys in [(summaries, ["transcriptid"]),               # one summary per call
                       (scores, ["transcriptid"] + RUN_KEYS)]:      # one score per call and run
        df = load_all(path)
        if df.empty:
            print(f"{path.name}: nothing to combine")
            continue
        if path == scores:
            df = with_run_defaults(df)
        df = df.drop_duplicates(keys, keep="last")
        df.to_parquet(path, index=False)
        for f in path.parent.glob(f"{path.stem}_shard*{path.suffix}"):
            f.unlink()
        print(f"{path.name}: {len(df)} rows")


if __name__ == "__main__":
    combine()
