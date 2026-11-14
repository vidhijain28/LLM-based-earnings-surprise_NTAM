"""Dev set for prompt experiments: 300 random in-sample calls from the early half of the quarter (seed 0).

Run from the repo root:
  uv run python -m llm_based_earnings_surprise_ntam.llm.llm_devset        # -> data/processed/dev300.csv

Score it with `llm_score --ids` (see LLM_README.md). The late half of the quarter (event dates after the
cut-off printed below) is the hold-out for the final test: don't score or look at it until then.
The pick is deterministic: the same pipeline output always gives the same 300 calls, so rerunning on Midway
gives the same file. If the pipeline output changed, the script refuses to overwrite a different dev set
(earlier experiments were scored on the old one); pass --force to replace it.
"""
import argparse

import pandas as pd

from .llm_common import DEV_IDS, MERGED, select_events

COLS = ["transcriptid", "keydevid", "ticker", "event_date"]


def make(merged=MERGED, out=DEV_IDS, n=300, seed=0, force=False):
    df = pd.read_parquet(merged, columns=COLS + ["in_sample"])
    insample = df[df["in_sample"]]
    cutoff = insample["event_date"].median()
    dev = (select_events(df, n=n, seed=seed)[COLS]          # same rule as the --n pilots
           .sort_values(["event_date", "transcriptid"]).reset_index(drop=True))
    if len(dev) < n:
        raise SystemExit(f"only {len(dev)} calls in the early half, fewer than {n}")

    if out.exists() and not force:
        old = pd.read_csv(out, usecols=["transcriptid"])["transcriptid"]
        if set(old) != set(dev["transcriptid"].astype("int64")):
            raise SystemExit(f"{out.name} exists and holds a different set of calls (the pipeline output changed?). "
                             "Experiments already scored on it would no longer match; pass --force to replace it.")
    dev.to_csv(out, index=False)
    n_early = int((insample["event_date"] <= cutoff).sum())
    print(f"wrote {out} ({len(dev)} calls, seed {seed}, events {dev['event_date'].min():%Y-%m-%d} to "
          f"{dev['event_date'].max():%Y-%m-%d})\n"
          f"dev pool: {n_early:,} in-sample calls with event date <= {cutoff:%Y-%m-%d}; "
          f"hold-out: the {len(insample) - n_early:,} calls after it")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="replace an existing dev set with a different one")
    args = ap.parse_args()
    out = DEV_IDS if args.n == 300 else DEV_IDS.with_name(f"dev{args.n}.csv")
    make(n=args.n, seed=args.seed, out=out, force=args.force)


if __name__ == "__main__":
    main()
