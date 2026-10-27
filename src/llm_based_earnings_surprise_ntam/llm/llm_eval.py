"""R² for surprise only vs. LLM vs. both, per model and text source.
Run: uv run python -m llm_based_earnings_surprise_ntam.llm.llm_eval"""
import pandas as pd
import statsmodels.formula.api as smf

from .llm_common import MERGED, SCORES

df = pd.read_parquet(MERGED)
s = pd.read_parquet(SCORES)

for (model, source), g in s.groupby(["model", "source"]):
    m = df[df["in_sample"]].merge(g[["transcriptid", "prediction"]], on="transcriptid")
    m = m.dropna(subset=["prediction", "r_abn_pct", "sue_pct_sample"])
    print(f"\n=== {model} | {source} | N = {len(m)} ===")
    for f in ["r_abn_pct ~ sue_pct_sample",
              "r_abn_pct ~ prediction",
              "r_abn_pct ~ prediction + sue_pct_sample"]:
        res = smf.ols(f, data=m).fit()
        print(f"{f:45s} R² = {res.rsquared:.3f}")