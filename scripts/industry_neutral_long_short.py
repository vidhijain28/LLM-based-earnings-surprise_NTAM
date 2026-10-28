"""Industry-neutral version of the stage gate 1 long-short (PIPELINE.md section 6).

Section 6 sorts every event into one pool, so the top fifth can end up overweight one industry and the spread
can be an industry bet. This script repeats the test three ways and prints them side by side:

  as in section 6     one pool, abnormal return vs VTI
  industry-neutral    fifths formed inside each Fama-French 12 industry, legs pooled afterwards
  industry-adjusted   the daily return is measured against the industry's own equal-weighted return
                      instead of VTI, then sorted in one pool

If the spread survives both, it is not an industry bet. Industry comes from SIC (section 7.3); SIC is the value
at the download date, so it groups events but is not point-in-time.

Run (from the repo root, with the cleaned files outside the repo):
    uv run python scripts/industry_neutral_long_short.py <folder with the cleaned files> <output folder>
    uv run python scripts/industry_neutral_long_short.py ../data/processed out --min-events 100
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llm_based_earnings_surprise_ntam.signals.ff12 import ff12_name
from llm_based_earnings_surprise_ntam.signals.long_short import (WINDOWS, event_paths, long_short, quantile,
                                                                 window_return)

LARGE = 10e9


def load(folder):
    """Events with industry and size, the daily return panel and the market series (as in scripts/long_short_q4_2025.py)."""
    folder = Path(folder)
    ev = pd.read_parquet(folder / 'events_baseline.parquet')
    px = pd.read_parquet(folder / 'prices_clean.parquet',
                         columns=['gvkey', 'iid', 'datadate', 'ret', 'sic', 'fic', 'mcap_lag1', 'tpci',
                                  'prev_prccd', 'primary_pit'])
    mk = pd.read_parquet(folder / 'market_ret.parquet')

    cal = pd.DatetimeIndex(sorted(pd.to_datetime(mk.datadate).unique()))
    market = mk.assign(datadate=pd.to_datetime(mk.datadate)).set_index('datadate').mkt_ret
    px['key'] = px.gvkey.astype(str) + '|' + px.iid.astype(str)
    px['datadate'] = pd.to_datetime(px.datadate)
    returns = px.set_index(['key', 'datadate']).ret.unstack().reindex(columns=cal)

    # The benchmark panel excludes rows the price file keeps but nobody could hold: non-common stock, non-US,
    # sub-$1 shares, secondary lines. Without this the equal-weighted industry mean is dominated by placeholder
    # prices in the file (its daily mean is 1.4pp above VTI; with the filter the gap is 0.00pp).
    inv = (px.tpci.eq('0') & px.fic.eq('USA') & (px.prev_prccd >= 1)
           & px.primary_pit.fillna(False).astype(bool) & px.ret.abs().lt(0.5))
    bench_returns = px[inv].set_index(['key', 'datadate']).ret.unstack().reindex(columns=cal)

    ev['key'] = ev.gvkey.astype(str) + '|' + ev.iid.astype(str)
    ev['event_date'] = pd.to_datetime(ev.event_date)
    ev = ev.merge(px[['key', 'datadate', 'mcap_lag1']].rename(columns={'datadate': 'event_date'}),
                  on=['key', 'event_date'], how='left')
    sec = px.drop_duplicates('key')[['key', 'fic', 'sic']]
    ev = ev.merge(sec, on='key', how='left')
    ev['industry'] = ff12_name(ev['sic'])
    ev['us'] = ev.fic.eq('USA')
    ev['large'] = ev.us & (ev.mcap_lag1 >= LARGE)

    # industry of every security in the panel, for the industry-adjusted benchmark
    panel_industry = pd.Series(ff12_name(sec.set_index('key')['sic']).values, index=sec.key)
    return ev, returns, bench_returns, market, panel_industry


def industry_quantile(signal, industry, n=5, min_events=100):
    """Fifths formed inside each industry. Industries with fewer than `min_events` events are left out (NaN)."""
    out = pd.Series(np.nan, index=signal.index)
    for ind, idx in signal.groupby(industry).groups.items():
        s = signal.loc[idx].dropna()
        if len(s) < min_events:
            continue
        out.loc[s.index] = pd.qcut(s.rank(method='first'), n, labels=False) + 1
    return out


def industry_benchmark(returns, panel_industry, min_names=5):
    """One row per security: its industry's equal-weighted daily return (NaN on days with fewer than `min_names`)."""
    ind = panel_industry.reindex(returns.index)
    grp = returns.groupby(ind)
    mean = grp.mean().where(grp.count() >= min_names)
    keys = ind.dropna()
    out = mean.reindex(keys.to_numpy())
    out.index = keys.index
    return out[~out.index.duplicated()]


def table(stock, mkt, q, dates, signal, label, windows=WINDOWS):
    out = long_short(stock, mkt, q, dates, signal=signal, windows=windows)
    return out.assign(check=label)[['check', 'window', 'N', 'n_long', 'n_short', 'spread', 't', 'rank_r2']]


def run(ev, returns, bench_returns, market, panel_industry, mask=None, label='all events', min_events=100):
    """The three variants for one sample."""
    d = ev if mask is None else ev[mask].copy()
    stock, mkt, _ = event_paths(d.key, d.event_date, returns, market, horizon=60)

    # benchmark path per event: its industry's equal-weighted return on the same days
    ind_rows = industry_benchmark(bench_returns, panel_industry).reindex(columns=returns.columns)
    pos = {dt: i for i, dt in enumerate(returns.columns)}
    bench = np.full_like(stock, np.nan)
    for r, (k, dt) in enumerate(zip(d.key, d.event_date)):
        j = pos.get(pd.Timestamp(dt))
        if j is None or k not in ind_rows.index:
            continue
        seg = ind_rows.loc[k].to_numpy()[j:j + stock.shape[1]]
        bench[r, :len(seg)] = seg

    sig = d.surprise_pct
    q_ind = industry_quantile(sig, d.industry, min_events=min_events)
    used = d.industry[q_ind.notna()].value_counts()
    print(f'{label}: industry-neutral fifths use {len(used)} industries and {int(used.sum())} of {len(d)} events '
          f'(at least {min_events} each): {", ".join(f"{k} {v}" for k, v in used.items())}')
    rows = [table(stock, mkt, quantile(sig), d.event_date, sig, f'{label}: as in section 6'),
            table(stock, mkt, q_ind, d.event_date, sig, f'{label}: industry-neutral fifths'),
            table(stock, bench, quantile(sig), d.event_date, sig, f'{label}: industry-adjusted return')]
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir', help="folder with events_baseline / prices_clean / market_ret parquet")
    ap.add_argument('out_dir')
    ap.add_argument('--min-events', type=int, default=100,
                    help='industry-neutral fifths: skip industries with fewer events (default 100)')
    ap.add_argument('--min-events-sub', type=int, default=50,
                    help='the same threshold for subsamples such as large caps, which are smaller (default 50)')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ev, returns, bench_returns, market, panel_industry = load(args.data_dir)
    chk = industry_benchmark(bench_returns, panel_industry).mean(axis=0) - market.reindex(bench_returns.columns)
    print(f'Industry benchmark check: its daily mean is {chk.mean():+.4%} from VTI on average '
          f'(sd {chk.std():.4%}); large values mean the benchmark is contaminated\n')
    print(f'{len(ev)} events, {int(ev.industry.notna().sum())} with an industry, '
          f'{ev.industry.nunique()} industries\n')
    counts = ev.industry.value_counts()
    print('Events per industry:')
    print(counts.to_string(), '\n')
    res = pd.concat([run(ev, returns, bench_returns, market, panel_industry, None, 'all events', args.min_events),
                     run(ev, returns, bench_returns, market, panel_industry, ev.large, 'US large caps',
                         args.min_events_sub)],
                    ignore_index=True)
    res.to_csv(out / 'industry_neutral_results.csv', index=False)

    # how concentrated are the legs in section 6's one-pool sort?
    q = quantile(ev.surprise_pct)
    conc = (pd.crosstab(ev.industry, q, normalize='columns')[[1, 5]] * 100).round(1)
    conc.columns = ['bottom fifth %', 'top fifth %']
    conc['all events %'] = (ev.industry.value_counts(normalize=True) * 100).round(1)
    conc.sort_values('top fifth %', ascending=False).to_csv(out / 'industry_concentration.csv')
    print('Industry share of each leg, one-pool sort (%):')
    print(conc.sort_values('top fifth %', ascending=False).to_string(), '\n')

    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(res.round(3).to_string(index=False))
    print(f'\nWritten to {out / "industry_neutral_results.csv"}')


if __name__ == '__main__':
    main()
