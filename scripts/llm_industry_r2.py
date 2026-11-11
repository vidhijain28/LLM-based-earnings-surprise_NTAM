"""Where does the LLM help on day 0? The LLM's incremental R² by Fama-French 12 industry.

The LLM step reports one incremental R² for the whole quarter (LLM_README.md, Austin's llm_results notebook).
This script splits it by industry (section 7.3's mapping), for every scoring run found.

In each group (all events, then each industry), the day-0 abnormal return, the surprise and the LLM score are
turned into percentile ranks inside that group, and
    R² surprise   rank return on rank surprise
    R² LLM        rank return on rank LLM score
    R² both       rank return on both
    ΔR²           R² both minus R² surprise, with a paired 99% bootstrap interval (events resampled with
                  replacement inside the group, as in section 7.1)

Returns are the pipeline's CRSP day-0 abnormal return (`r_abn`) and the surprise is the pipeline's `sue`, so this
is section 5's regression run by industry. The sample is every scored event matched to the Compustat price panel,
the same events as llm_long_short.py without the transcript-timing filter, since day 0 is not traded. Industry is
the panel's SIC (a download-date snapshot, section 7.3).

Run:
    uv run python scripts/llm_industry_r2.py <folder with the cleaned files> <output folder>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llm_long_short import load_prices, load_scores


def r2(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return 1 - e.var() / y.var()


def pct_rank(x):
    return pd.Series(x).rank(pct=True).to_numpy()


def group_r2(y, s, p, resamples, rng):
    y, s, p = pct_rank(y), pct_rank(s), pct_rank(p)
    out = dict(n=len(y), r2_surprise=r2(y, [s]), r2_llm=r2(y, [p]), r2_both=r2(y, [s, p]))
    out['delta_r2'] = out['r2_both'] - out['r2_surprise']
    if resamples:
        boot = np.empty(resamples)
        for b in range(resamples):
            i = rng.integers(0, len(y), len(y))
            boot[b] = r2(y[i], [s[i], p[i]]) - r2(y[i], [s[i]])
        out['delta_lo'], out['delta_hi'] = np.quantile(boot, [0.005, 0.995])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--resamples', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    ev, *_ = load_prices(args.data_dir)
    sc = load_scores(args.data_dir)
    # the pipeline's CRSP day-0 abnormal return, keyed like the scores
    ret = pd.concat([pd.read_parquet(f, columns=['gvkey', 'event_date', 'r_abn'])
                     for f in sorted(Path(args.data_dir).glob('llm_merged_*.parquet'))], ignore_index=True)
    ret = ret.assign(gvkey_num=pd.to_numeric(ret.gvkey, errors='coerce'),
                     event_date=pd.to_datetime(ret.event_date)).dropna(subset=['r_abn'])
    ret = ret.drop_duplicates(['gvkey_num', 'event_date'])[['gvkey_num', 'event_date', 'r_abn']]

    rng = np.random.default_rng(args.seed)
    rows = []
    for spec, g in sc.groupby('spec'):
        d = ev.merge(g[['gvkey_num', 'event_date', 'score', 'sue']], on=['gvkey_num', 'event_date'])
        d = d.dropna(subset=['score', 'surprise_pct', 'sue']).merge(ret, on=['gvkey_num', 'event_date'])
        dup = d.duplicated(['gvkey_num', 'event_date']).sum()
        print(f'[{spec}] {len(d)} events ({dup} are a second share class of the same company and date)')
        groups = [('ALL', d)] + sorted(d.groupby('industry'), key=lambda t: -len(t[1]))
        for name, gg in groups:
            rows.append(dict(spec=spec, industry=name,
                             **group_r2(gg.r_abn.to_numpy(float), gg.sue.to_numpy(float),
                                        gg.score.to_numpy(float), args.resamples, rng)))
    r = pd.DataFrame(rows)
    r.to_csv(out / 'llm_industry_r2.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_rows', 200, 'display.max_colwidth', 70):
        print(r.round(4).to_string(index=False))
    print(f"\nWritten to {out / 'llm_industry_r2.csv'}")


if __name__ == '__main__':
    main()
