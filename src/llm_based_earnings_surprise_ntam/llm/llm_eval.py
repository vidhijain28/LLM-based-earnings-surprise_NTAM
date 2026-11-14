"""Day-0 R² (Koijen & Levy): surprise only vs. LLM vs. both, per run (model, source, variant, temperature, samples).
Run: uv run python -m llm_based_earnings_surprise_ntam.llm.llm_eval     (or `uv run doit llm_eval`)
     uv run python -m llm_based_earnings_surprise_ntam.llm.llm_eval --ids   # dev set only: compare variants

`evaluate` reports every run in the scores file and writes the report to a text file. With ids, every run is
restricted to those calls, so the base run and the experiments are compared on the same calls."""
import argparse

import pandas as pd
import statsmodels.formula.api as smf

from .llm_common import DEV_IDS, MERGED, RUN_KEYS, SCORES, read_ids, with_run_defaults

FORMULAS = ["r_abn_pct ~ sue_pct_sample",
            "r_abn_pct ~ prediction",
            "r_abn_pct ~ prediction + sue_pct_sample"]


def evaluate(final=MERGED, scores=SCORES, out=None, ids=None):
    df = pd.read_parquet(final, columns=["transcriptid", "in_sample", "r_abn_pct", "sue_pct_sample"])
    df = df[df["in_sample"].fillna(False).astype(bool)]
    if ids is not None:
        df = df[df["transcriptid"].astype("int64").isin(read_ids(ids))]
    s = with_run_defaults(pd.read_parquet(scores))
    lines = [f"calls restricted to {ids}" if ids is not None else ""]
    for (model, source, variant, temperature, samples), g in s.groupby(RUN_KEYS):
        m = df.merge(g[["transcriptid", "prediction"]], on="transcriptid")
        m = m.dropna(subset=["prediction", "r_abn_pct", "sue_pct_sample"])
        if len(m) < 10:
            continue
        lines.append(f"\n=== {model} | {source} | variant {variant} | temperature {temperature:g} | "
                     f"{samples} sample{'s' if samples > 1 else ''} | N = {len(m)} ===")
        for f in FORMULAS:
            lines.append(f"{f:45s} R² = {smf.ols(f, data=m).fit().rsquared:.3f}")
    text = "\n".join(lines).strip()
    print(text)
    if out is not None:
        out.write_text(text + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="?", const=DEV_IDS, metavar="CSV",
                    help=f"only these calls (`--ids` alone: {DEV_IDS.name})")
    evaluate(ids=ap.parse_args().ids)
