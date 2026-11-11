"""R² for surprise only vs. LLM vs. both, per model, text source, summary author and prompt variant.
Run: uv run python -m llm_based_earnings_surprise_ntam.llm.llm_eval"""
import pandas as pd
import statsmodels.formula.api as smf

from .llm_common import MERGED, SCORES

df = pd.read_parquet(MERGED)
s = pd.read_parquet(SCORES)

# older rows have no variant / summary_model / cost columns
s["variant"] = s["variant"].fillna("plain") if "variant" in s else "plain"
sm = s["summary_model"] if "summary_model" in s else pd.Series(pd.NA, index=s.index)
s["summary_model"] = sm.fillna(s["model"].where(s["source"] == "summary")).fillna("-")
if "cost" not in s:
    s["cost"] = 0.0
# a call retried after a failure has two rows: keep the latest one with a prediction
s = s.dropna(subset=["prediction"]).drop_duplicates(
    ["transcriptid", "model", "source", "summary_model", "variant"], keep="last")

for (model, source, summary_model, variant), g in s.groupby(["model", "source", "summary_model", "variant"]):
    m = df[df["in_sample"]].merge(g[["transcriptid", "prediction"]], on="transcriptid")
    m = m.dropna(subset=["prediction", "r_abn_pct", "sue_pct_sample"])
    by = f" (summaries by {summary_model})" if source == "summary" and summary_model != model else ""
    print(f"\n=== {model} | {source}{by} | prompt: {variant} | N = {len(m)} | cost ${g['cost'].sum():.2f} ===")
    for f in ["r_abn_pct ~ sue_pct_sample",
              "r_abn_pct ~ prediction",
              "r_abn_pct ~ prediction + sue_pct_sample"]:
        res = smf.ols(f, data=m).fit()
        print(f"{f:45s} R² = {res.rsquared:.3f}")
