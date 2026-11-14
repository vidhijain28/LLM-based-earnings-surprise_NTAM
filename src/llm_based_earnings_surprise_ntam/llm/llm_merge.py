"""Join the pipeline output (in-sample rows) with the LLM summaries and scores, and save one file.
Run: uv run python -m llm_based_earnings_surprise_ntam.llm.llm_merge     (or `uv run doit llm_merge`)

Writes data/processed/llm_merged_<period tag>.parquet, using the scores of the (model, source) run set in
config.yaml (`llm` section), base variant (temperature 0, one sample): dev-set experiments are left out.
The transcript text columns (*_text) are left out.
"""
import pandas as pd
import pyarrow.parquet as pq

from .llm_common import MERGED, MODEL, RUN_DEFAULTS, RUN_KEYS, SCORES, SOURCE, SUMMARIES, with_run_defaults

OUT = MERGED.parent / MERGED.name.replace("earnings_calls_ibes_", "llm_merged_")


def build(final=MERGED, summaries=SUMMARIES, scores=SCORES, out=OUT, model=MODEL, source=SOURCE,
          variant=RUN_DEFAULTS["variant"], temperature=RUN_DEFAULTS["temperature"], samples=RUN_DEFAULTS["samples"]):
    cols = [c for c in pq.read_schema(final).names if not c.endswith("_text")]
    df = pd.read_parquet(final, columns=cols)
    df = df[df["in_sample"].fillna(False).astype(bool)]

    summ = pd.read_parquet(summaries).rename(columns={
        "words_in": "sum_words_in", "words_out": "sum_words_out", "passes": "sum_passes",
        "model": "sum_model", "seconds": "sum_seconds"})
    allsc = with_run_defaults(pd.read_parquet(scores))
    sc = allsc[(allsc["model"] == model) & (allsc["source"] == source) & (allsc["variant"] == variant)
               & (allsc["temperature"] == float(temperature)) & (allsc["samples"] == samples)]
    if sc.empty:
        raise SystemExit(f"{scores.name} has no scores from {model} on {source}, variant {variant}, "
                         f"temperature {temperature}, {samples} sample(s); runs in the file:\n"
                         f"{allsc[RUN_KEYS].value_counts()}")
    sc = sc[["transcriptid", "prediction", "reasons", "model", "source", "seconds"]].rename(columns={
        "model": "score_model", "source": "score_source", "seconds": "score_seconds"})

    m = (df.merge(summ, on="transcriptid", how="inner", validate="one_to_one")
           .merge(sc, on="transcriptid", how="inner", validate="one_to_one"))
    missing = len(df) - len(m)
    if missing:
        print(f"WARNING: {missing} in-sample events have no summary or score; "
              f"r_abn_pct / sue_pct_sample ranks were computed over all {len(df):,} in-sample events")
    m.to_parquet(out, index=False)
    print(f"LLM merge: wrote {out.name} ({len(m):,} rows x {m.shape[1]} columns; scores from {model} on {source})")


if __name__ == "__main__":
    build()
