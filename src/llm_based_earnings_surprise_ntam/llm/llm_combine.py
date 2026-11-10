"""Combine the parallel GPU jobs' shard files into llm_summaries.parquet and llm_scores.parquet.
Run only after ALL jobs have finished:
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_combine
"""
from .llm_common import SUMMARIES, SCORES, load_all

for path, keys in [(SUMMARIES, ["transcriptid", "model"]),
                   (SCORES, ["transcriptid", "model", "source"])]:
    df = load_all(path)
    if df.empty:
        print(f"{path.name}: nothing to merge")
        continue
    df = df.drop_duplicates(keys, keep="last")
    df.to_parquet(path, index=False)
    for f in path.parent.glob(f"{path.stem}_shard*{path.suffix}"):
        f.unlink()
    print(f"{path.name}: {len(df)} rows")
