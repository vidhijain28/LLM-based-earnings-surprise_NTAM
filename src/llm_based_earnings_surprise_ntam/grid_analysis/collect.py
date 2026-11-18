"""Combine each run's part files into one table per run, and print how often the model answered in the required format.

    python -m llm_based_earnings_surprise_ntam.grid_analysis.collect --config <llm_runs.yaml> [--runs A B C P]

Writes out/scores_<run>.parquet: one row per call, with the run's fields (prediction and reasons, or the persistence
fields), status, attempts and the raw replies. persistence = recurring share / (recurring + one-off share), so it is
the recurring share whenever the two add up to 1; it is empty when both are 0. Format compliance: the share of calls
whose first reply parsed, and the share that parsed within all retries.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from ..GEPA_Optimization.config import load
from .score import DEFAULT, read_results


def table(results):
    rows = []
    for r in results:
        row = {k: r[k] for k in ('transcriptid', 'keydevid', 'gvkey', 'event_date', 'run', 'spec', 'n_history',
                                 'status', 'attempts', 'seconds')}
        res = r['result'] or {}
        for k, v in res.items():
            row[k] = json.dumps(v) if isinstance(v, list) else v
        row['raw'] = json.dumps(r['raw'])
        rows.append(row)
    df = pd.DataFrame(rows).drop_duplicates('transcriptid', keep='first')
    if 'recurring_share' in df:
        total = df['recurring_share'] + df['one_off_share']
        df['persistence'] = (df['recurring_share'] / total).where(total > 0)
    df['event_date'] = pd.to_datetime(df['event_date'])
    return df.sort_values('transcriptid').reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(DEFAULT))
    ap.add_argument('--runs', nargs='*', help='default: every run with a folder in paths.out')
    args = ap.parse_args()
    cfg = load(args.config)
    out = Path(cfg['paths']['out'])
    runs = args.runs or [r for r in cfg['runs'] if (out / r).is_dir()]
    for run in runs:
        results = read_results(out / run)
        if not results:
            print(f'run {run}: no results in {out / run}')
            continue
        df = table(results)
        path = out / f'scores_{run}.parquet'
        df.to_parquet(path, index=False)
        first = ((df['status'] == 'ok') & (df['attempts'] == 1)).mean()
        ok = (df['status'] == 'ok').mean()
        print(f'run {run}: {len(df)} calls -> {path}; format compliance: first reply {first:.1%}, '
              f'within retries {ok:.1%}; {len(df) - int(ok * len(df) + 0.5)} unparseable')


if __name__ == '__main__':
    main()
