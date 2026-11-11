"""Stage gate 1 for the LLM score: is it tradable after the announcement day?

Austin's notebook shows the LLM score explains the day-0 reaction (incremental R² about 0.055 over the surprise).
Day 0 cannot be traded by a signal built from the release. This script asks the stage-gate question instead:
after the day-0 close, does a long-short on the LLM score earn anything at 1, 4 and 12 weeks, and does it add
anything to the earnings surprise?

Three checks per signal, as in PIPELINE.md 7.4:
    as in section 6     fifths in one pool, return against VTI
    industry-neutral    fifths inside each Fama-French 12 industry, legs pooled afterwards
    industry-adjusted   return against the industry's own equal-weighted return

Plus size-neutral sorts (market-cap tertiles) and a double sort: inside each surprise quintile, sort by the LLM
score. If the LLM only repeats the surprise, the double-sorted spread collapses. The reverse sort (surprise
inside LLM quintiles) is run too, to see which signal dominates.

Two samples, each sorted into its own fifths (as in PIPELINE.md section 6):
    all events          every event with a score, a surprise and (optionally) a transcript by the day-0 close
    US large caps       US-incorporated, market cap >= $10bn the day before the event

Output columns beyond the spread: Q1..Q5 are the mean abnormal return of each fifth (winsorized, as in the
spread), so Q5 is the long-only leg against the market, `all` is the mean over every event in the sample and
`long_minus_all` = Q5 - all is the long leg against the average event.

Run:
    uv run python scripts/llm_long_short.py <folder with the cleaned files> <output folder> --require-available
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llm_based_earnings_surprise_ntam.signals.ff12 import ff12_name
from llm_based_earnings_surprise_ntam.signals.long_short import (WINDOWS, event_paths, long_short, quantile)

LARGE = 10e9
MIN_TOTAL = {'all events': 200, 'US large caps': 100}   # fewest matched events worth running per sample


def load_prices(folder):
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

    # investable panel for the industry benchmark (the raw panel contains placeholder prices)
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
    ev['large'] = ev.fic.eq('USA') & (ev.mcap_lag1 >= LARGE)
    ev['gvkey_num'] = pd.to_numeric(ev.gvkey, errors='coerce')
    panel_industry = pd.Series(ff12_name(sec.set_index('key')['sic']).values, index=sec.key)
    return ev, returns, bench_returns, market, panel_industry


def load_scores(folder):
    """Every scoring setup found, as one long table: gvkey_num, event_date, spec, score."""
    folder, out = Path(folder), []
    for f in sorted(folder.glob('llm_merged_*.parquet')):
        d = pd.read_parquet(f, columns=['gvkey', 'event_date', 'prediction', 'score_model', 'score_source',
                                        'in_sample', 'transcript_by_event_close', 'sue'])
        d = d[d.in_sample.fillna(False).astype(bool)].dropna(subset=['prediction'])
        out.append(pd.DataFrame({'gvkey_num': pd.to_numeric(d.gvkey, errors='coerce'),
                                 'event_date': pd.to_datetime(d.event_date),
                                 'spec': d.score_model.astype(str) + ' scores ' + d.score_source.astype(str),
                                 'score': d.prediction.astype(float),
                                 'available': d.transcript_by_event_close.fillna(False).astype(bool),
                                 'sue': d.sue.astype(float)}))
    p = folder / 'llm_scores.parquet'
    if p.exists():
        d = pd.read_parquet(p)
        if 'gvkey' not in d.columns:                      # key it through the merged file
            link = pd.concat([pd.read_parquet(f, columns=['keydevid', 'gvkey', 'event_date',
                                                          'transcript_by_event_close', 'sue'])
                              for f in sorted(folder.glob('llm_merged_*.parquet'))], ignore_index=True)
            d = d.merge(link.drop_duplicates('keydevid'), on='keydevid', how='inner', suffixes=('_s', ''))
        d = d.dropna(subset=['prediction'])
        src = d.summary_model if 'summary_model' in d.columns else d.get('source', 'summary')
        out.append(pd.DataFrame({'gvkey_num': pd.to_numeric(d.gvkey, errors='coerce'),
                                 'event_date': pd.to_datetime(d.event_date),
                                 'spec': d.model.astype(str) + ' scores ' + src.astype(str).str.split('/').str[-1],
                                 'score': d.prediction.astype(float),
                                 'available': d.transcript_by_event_close.fillna(False).astype(bool),
                                 'sue': d.sue.astype(float)}))
    s = pd.concat(out, ignore_index=True).dropna(subset=['gvkey_num', 'event_date', 'score'])
    return s.drop_duplicates(['gvkey_num', 'event_date', 'spec'])


def industry_quantile(signal, industry, n=5, min_events=100):
    out = pd.Series(np.nan, index=signal.index)
    for _, idx in signal.groupby(industry).groups.items():
        s = signal.loc[idx].dropna()
        if len(s) >= min_events:
            out.loc[s.index] = pd.qcut(s.rank(method='first'), n, labels=False) + 1
    return out


def conditional_quantile(signal, control, n=5, min_events=100):
    """Fifths of `signal` inside each quintile of `control`: the LLM score holding the surprise roughly fixed."""
    ctrl = quantile(control)
    out = pd.Series(np.nan, index=signal.index)
    for _, idx in signal.groupby(ctrl).groups.items():
        s = signal.loc[idx].dropna()
        if len(s) >= min_events:
            out.loc[s.index] = pd.qcut(s.rank(method='first'), n, labels=False) + 1
    return out


def size_group(mcap, n=3):
    """Tertiles of market capitalization the day before the event (1 = smallest). NaN where cap is missing."""
    out = pd.Series(np.nan, index=mcap.index)
    ok = mcap.notna() & (mcap > 0)
    if ok.sum() >= 150:
        out[ok] = pd.qcut(mcap[ok].rank(method='first'), n, labels=False) + 1
    return out


def industry_benchmark(returns, panel_industry, min_names=5):
    ind = panel_industry.reindex(returns.index)
    grp = returns.groupby(ind)
    mean = grp.mean().where(grp.count() >= min_names)
    keys = ind.dropna()
    out = mean.reindex(keys.to_numpy())
    out.index = keys.index
    return out[~out.index.duplicated()]


def bench_paths(d, ind_rows, returns, width):
    pos = {dt: i for i, dt in enumerate(returns.columns)}
    bench = np.full((len(d), width), np.nan)
    for r, (k, dt) in enumerate(zip(d.key, d.event_date)):
        j = pos.get(pd.Timestamp(dt))
        if j is None or k not in ind_rows.index:
            continue
        seg = ind_rows.loc[k].to_numpy()[j:j + width]
        bench[r, :len(seg)] = seg
    return bench


def sorted_spread(stock, mkt, q, d, signal, windows, what):
    """long_short, or None (with a note) when the sort leaves no top or bottom fifth, which happens when no
    industry or size cell reaches the minimum number of events. A skipped sort is better than a crash or a
    spread computed from a handful of events."""
    q = pd.Series(q)
    if not (q == 1).any() or not (q == 5).any():
        print(f'    skipped: {what} (no cell has enough events for fifths; lower --min-events-sub)')
        return None
    return long_short(stock, mkt, q, d.event_date, signal=signal, windows=windows)


def run_one(d, stock, mkt, bench, sig, label, min_events, min_events_sub, windows):
    """The five sorts of one signal on one sample. `min_events` is the fewest events an industry or size cell
    needs to be sorted into fifths; `min_events_sub` is the same for the size-and-industry cells."""
    rows = []
    size = size_group(d.mcap_lag1)
    for variant, q, m in [('as in section 6', quantile(sig), mkt),
                          ('industry-neutral', industry_quantile(sig, d.industry, min_events=min_events), mkt),
                          ('industry-adjusted', quantile(sig), bench),
                          ('size-neutral', industry_quantile(sig, size, min_events=min_events), mkt),
                          ('size and industry neutral',
                           industry_quantile(sig, d.industry.astype(str) + '|' + size.astype(str),
                                             min_events=min_events_sub), mkt)]:
        t = sorted_spread(stock, m, q, d, sig, windows, f'{variant} ({label})')
        if t is not None:
            rows.append(t.assign(sample=label, variant=variant))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_spec(d, spec, returns, ind_rows, market, min_events, min_events_sub, windows=WINDOWS):
    """Every table for one scoring setup on one sample `d` (already merged with the price events)."""
    res = []
    label = f'LLM: {spec}'
    stock, mkt, _ = event_paths(d.key, d.event_date, returns, market, horizon=60)
    bench = bench_paths(d, ind_rows, returns, stock.shape[1])
    res.append(run_one(d, stock, mkt, bench, d.score, label, min_events, min_events_sub, windows))
    # the LLM inside surprise quintiles
    q = conditional_quantile(d.score, d.surprise_pct, min_events=min_events)
    t = sorted_spread(stock, mkt, q, d, d.score, windows, f'LLM within surprise quintiles ({label})')
    if t is not None:
        res.append(t.assign(sample=label, variant='LLM within surprise quintiles'))
    # and the reverse: the surprise inside LLM quintiles, to see which signal dominates
    q_rev = conditional_quantile(d.surprise_pct, d.score, min_events=min_events)
    t = sorted_spread(stock, mkt, q_rev, d, d.surprise_pct, windows, f'surprise within LLM quintiles ({label})')
    if t is not None:
        res.append(t.assign(sample=label, variant='surprise within LLM quintiles'))
    # the surprise on the same events, for a like-for-like comparison
    for lab, col in [('surprise (monthly snapshot)', 'surprise_pct'), ('surprise (pipeline sue)', 'sue')]:
        res.append(long_short(stock, mkt, quantile(d[col]), d.event_date, signal=d[col], windows=windows)
                   .assign(sample=f'{lab}, same events as {spec}', variant='as in section 6'))
    return pd.concat([x for x in res if len(x)], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--min-events', type=int, default=100,
                    help='fewest events an industry or size cell needs, all-events sample')
    ap.add_argument('--min-events-sub', type=int, default=50,
                    help='the same for the US large-cap sample, which is about a sixth of the size')
    ap.add_argument('--require-available', action='store_true',
                    help='keep only events whose transcript existed by the 16:00 ET close on the event date, '
                         'the only ones where the score could have been formed before the entry price')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ev, returns, bench_returns, market, panel_industry = load_prices(args.data_dir)
    sc = load_scores(args.data_dir)
    ind_rows = industry_benchmark(bench_returns, panel_industry).reindex(columns=returns.columns)
    print(f'{len(ev)} events in the price panel; scoring setups found:')
    print(sc.groupby('spec').size().to_string(), '\n')

    res = []
    for spec, g in sc.groupby('spec'):
        d = ev.merge(g[['gvkey_num', 'event_date', 'score', 'available', 'sue']],
                     on=['gvkey_num', 'event_date'], how='inner')
        d = d.dropna(subset=['score', 'surprise_pct', 'sue']).reset_index(drop=True)
        if args.require_available:
            n0 = len(d); d = d[d.available].reset_index(drop=True)
            print(f'  [{spec}] transcript available by the day-0 close: {len(d)} of {n0}')
        print(f'[{spec}] {len(g)} scored events, {len(d)} matched to the price panel')
        for name, dd, me, mes in [('all events', d, args.min_events, args.min_events_sub),
                                  ('US large caps', d[d.large].reset_index(drop=True),
                                   args.min_events_sub, max(20, args.min_events_sub // 2))]:
            if len(dd) < MIN_TOTAL[name]:
                print(f'  {name}: {len(dd)} events, too few, skipped'); continue
            print(f'  {name}: {len(dd)} events')
            res.append(run_spec(dd, spec, returns, ind_rows, market, me, mes).assign(subsample=name))

    r = pd.concat(res, ignore_index=True)
    r['long_minus_all'] = r['Q5'] - r['all']
    cols = ['sample', 'subsample', 'variant', 'window', 'N', 'n_long', 'n_short',
            'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'all', 'long_minus_all', 'spread', 'se', 't', 'rank_r2']
    r[cols].to_csv(out / 'llm_long_short_results.csv', index=False)
    show = ['sample', 'subsample', 'variant', 'window', 'N', 'Q5', 'long_minus_all', 'spread', 't']
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 400,
                           'display.max_colwidth', 70):
        print('\nTop fifth minus bottom fifth, abnormal return (t clustered by event date); Q5 = long leg only:')
        print(r[show].round(4).to_string(index=False))
    print(f"\nWritten to {out / 'llm_long_short_results.csv'}")


if __name__ == '__main__':
    main()
