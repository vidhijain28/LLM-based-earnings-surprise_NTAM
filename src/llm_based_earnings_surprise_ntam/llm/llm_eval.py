"""Day-0 R² (Koijen & Levy): surprise only vs. LLM vs. both, per model and text source.
Run: uv run python -m llm_based_earnings_surprise_ntam.llm.llm_eval     (or `uv run doit llm_eval`)

`evaluate` reports every (model, source) run in the scores file and writes the report to a text file."""
import pandas as pd
import statsmodels.formula.api as smf

from .llm_common import MERGED, SCORES

FORMULAS = ["r_abn_pct ~ sue_pct_sample",
            "r_abn_pct ~ prediction",
            "r_abn_pct ~ prediction + sue_pct_sample"]


def evaluate(final=MERGED, scores=SCORES, out=None):
    df = pd.read_parquet(final, columns=["transcriptid", "in_sample", "r_abn_pct", "sue_pct_sample"])
    df = df[df["in_sample"].fillna(False).astype(bool)]
    s = pd.read_parquet(scores)
    lines = []
    for (model, source), g in s.groupby(["model", "source"]):
        m = df.merge(g[["transcriptid", "prediction"]], on="transcriptid")
        m = m.dropna(subset=["prediction", "r_abn_pct", "sue_pct_sample"])
        lines.append(f"\n=== {model} | {source} | N = {len(m)} ===")
        for f in FORMULAS:
            lines.append(f"{f:45s} R² = {smf.ols(f, data=m).fit().rsquared:.3f}")
    text = "\n".join(lines).strip()
    print(text)
    if out is not None:
        out.write_text(text + "\n")


if __name__ == "__main__":
    evaluate()
