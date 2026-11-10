"""Check the daily returns file (pipeline step daily_returns) before using it in the backtest.

Run from the repo root, after `uv run doit daily_returns` (no WRDS login needed, it only reads local files):
    uv run python scripts/check_daily_returns.py
    uv run python scripts/check_daily_returns.py <final parquet> <daily returns parquet>    # other files

Checks:
  1. structure   one row per security and day, VTI calendar without unexpected gaps, one share class per company
  2. coverage    every in-sample call has its company and event date in the file; how many have 60 days
  3. vs CRSP     the Compustat return on each call's event day against the CRSP return in the final parquet
                 (the `returns` step): an independent source for the same stock and day
  4. 60-day CAR  computed with the backtest code (signals/long_short.py) and with a plain loop: must agree
  5. spot check  buy-and-hold over days 1-60 for the largest companies, to compare with adjusted closes on a
                 public site (e.g. Yahoo Finance: adj close on day 60 / adj close on day 0 - 1)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from llm_based_earnings_surprise_ntam.pipeline.config import load_config, path, paths  # noqa: E402
from llm_based_earnings_surprise_ntam.signals.long_short import event_paths, window_return  # noqa: E402

H = 60


def flag(ok):
    return 'OK   ' if ok else 'CHECK'


def main(final_path, daily_path):
    ev = pd.read_parquet(final_path, columns=['transcriptid', 'gvkey', 'ticker', 'event_date', 'in_sample',
                                              'ret', 'mkt_ret'])
    ev = ev[ev.in_sample.fillna(False).astype(bool)].copy()
    ev['event_date'] = pd.to_datetime(ev.event_date).astype('datetime64[ns]')
    d = pd.read_parquet(daily_path)
    d['date'] = pd.to_datetime(d.date).astype('datetime64[ns]')
    print(f'{len(ev):,} in-sample calls | {len(d):,} daily rows in {Path(daily_path).name}\n')

    # 1. structure --------------------------------------------------------------------------------------------
    print('1. STRUCTURE')
    dup = int(d.duplicated(['gvkey', 'iid', 'date']).sum())
    print(f'   {flag(dup == 0)} duplicate security-days: {dup}')
    cal = pd.DatetimeIndex(np.sort(d.loc[d.is_market, 'date'].unique()))
    wk = pd.bdate_range(cal.min(), cal.max())
    gaps = wk.difference(cal)
    print(f'   {flag(len(cal) > 0)} VTI calendar: {len(cal)} trading days, {cal.min().date()} to {cal.max().date()}')
    print(f'   {flag(len(gaps) <= 12)} weekdays without VTI (should be market holidays only): '
          f'{", ".join(str(g.date()) for g in gaps) or "none"}')
    off = int((~d.date.isin(cal)).sum())
    print(f'   {flag(off == 0)} rows on days VTI did not trade: {off}')
    iids = d[~d.is_market].groupby('gvkey').iid.nunique()
    print(f'   {flag((iids == 1).all())} companies with more than one share class: {int((iids > 1).sum())}')
    tick = ev.drop_duplicates('gvkey').set_index('gvkey').ticker
    d = d.sort_values(['gvkey', 'iid', 'date'])
    d['next_ret'] = d.groupby(['gvkey', 'iid']).ret.shift(-1)
    big = d.loc[~d.is_market & (d.ret.abs() > 0.5)]
    two_day = (1 + big.ret) * (1 + big.next_ret.fillna(0)) - 1
    back = big[two_day.abs() < 0.5 * big.ret.abs()]  # most of the move undone the next day
    print(f'   {flag(len(big) < 100)} daily moves beyond +/-50% (section 6 found 57 in a larger panel): {len(big)}, '
          f'of which {len(back)} reverse most of the move the next day (likely bad prices)')
    for r in big.reindex(big.ret.abs().sort_values(ascending=False).index).head(10).itertuples():
        print(f'          {str(tick.get(r.gvkey, "?")):6s} gvkey {r.gvkey} {r.date.date()} {r.ret:+8.1%}   '
              f'next day {r.next_ret:+8.1%}')

    # 2. coverage ---------------------------------------------------------------------------------------------
    print('\n2. COVERAGE')
    have = set(d.loc[~d.is_market, 'gvkey'])
    no_co = ~ev.gvkey.isin(have)
    no_day = ~ev.event_date.isin(cal)
    print(f'   {flag(no_co.mean() < 0.02)} calls whose company is not in the file: {int(no_co.sum())} '
          f'({no_co.mean():.1%})')
    print(f'   {flag(no_day.sum() == 0)} calls whose event date is not a VTI trading day: {int(no_day.sum())}')
    ok = ev[~no_co & ~no_day].copy()
    pos = cal.get_indexer(ok.event_date)
    after = len(cal) - 1 - pos
    full = after >= H
    print(f'   {flag(full.mean() > 0.9)} calls with {H} trading days after the event: {int(full.sum()):,} of '
          f'{len(ok):,} ({full.mean():.0%}); last event with a full window: '
          f'{ok.event_date[full].max().date() if full.any() else "none"}')

    # wide panel, as the backtest uses it
    stock = d[~d.is_market].pivot(index='gvkey', columns='date', values='ret').reindex(columns=cal)
    vti = d[d.is_market].set_index('date').ret.reindex(cal)
    S, M, filled = event_paths(ok.gvkey.to_numpy(), ok.event_date.to_numpy(), stock, vti, horizon=H)
    inside = ~np.isnan(M[:, 1:])
    miss = filled[:, 1:][inside].mean()
    print(f'   {flag(miss < 0.02)} missing stock days inside the windows (filled with VTI, 0 abnormal): '
          f'{miss:.2%} (section 6: about 0.6%)')

    # 3. vs CRSP ----------------------------------------------------------------------------------------------
    print('\n3. AGAINST CRSP (event day, independent source)')
    comp0 = S[:, 0].copy()
    comp0[filled[:, 0]] = np.nan                       # compare real Compustat returns only
    c = pd.DataFrame({'ticker': ok.ticker.to_numpy(), 'date': ok.event_date.dt.date.to_numpy(),
                      'compustat': comp0, 'crsp': ok.ret.to_numpy(dtype=float)}).dropna()
    diff = (c.compustat - c.crsp).abs()
    exact = (diff < 1e-9).mean()
    print(f'   {flag(c.compustat.corr(c.crsp) > 0.98)} stock day-0 returns: {len(c):,} pairs, correlation '
          f'{c.compustat.corr(c.crsp):.6f}, within 0.1 pp: {(diff < 0.001).mean():.1%}, within 1 pp: '
          f'{(diff < 0.01).mean():.1%}')
    print(f'         gaps in bp: median {diff.median() * 1e4:.3f}, 99th pct {diff.quantile(.99) * 1e4:.3f}, '
          f'max {diff.max() * 1e4:.3f} | exactly equal (to 1e-9): {exact:.1%} | gaps above 10 bp: '
          f'{int((diff > 0.001).sum())}')
    if exact > 0.5:
        print('   CHECK many returns are bit-for-bit identical: two independent sources should differ in the last '
              'digits (CRSP stores 6 decimals). Is the daily file really computed from Compustat prices?')
    for r in c.assign(gap=diff).nlargest(5, 'gap').itertuples():
        print(f'          {str(r.ticker):6s} {r.date}  Compustat {r.compustat:+.4%}  CRSP {r.crsp:+.4%}  '
              f'gap {r.gap * 1e4:.2f} bp')
    m = pd.DataFrame({'compustat': M[:, 0], 'crsp': ok.mkt_ret.to_numpy(dtype=float)}).dropna().drop_duplicates()
    mg = (m.compustat - m.crsp).abs()
    print(f'   {flag(mg.max() < 0.001)} VTI on event days: {len(m)} days, largest gap {mg.max() * 1e4:.1f} bp')

    # 4. 60-day CAR two ways ----------------------------------------------------------------------------------
    print(f'\n4. {H}-DAY CAR, TWO WAYS')
    car = window_return(S, M, 1, H)
    R, V, idx = stock.to_numpy(), vti.to_numpy(), stock.index.get_indexer(ok.gvkey)
    loop = np.full(len(ok), np.nan)
    for e, (i, p) in enumerate(zip(idx, pos)):
        if p + H < len(cal):
            s, m_ = R[i, p + 1:p + H + 1], V[p + 1:p + H + 1]
            loop[e] = np.sum(np.where(np.isnan(s), m_, s) - m_)
    both = ~np.isnan(car) & ~np.isnan(loop)
    gap = np.nanmax(np.abs(car[both] - loop[both])) if both.any() else np.nan
    print(f'   {flag(both.any() and gap < 1e-12)} backtest code vs plain loop: {int(both.sum()):,} calls, '
          f'largest difference {gap:.1e}')
    print(f'   {flag(np.array_equal(np.isnan(car), np.isnan(loop)))} same calls left out (window past the data)')
    q = pd.Series(car[both])
    print(f'         {H}-day CAR: mean {q.mean():+.2%}, median {q.median():+.2%}, '
          f'1st-99th pct {q.quantile(.01):+.1%} to {q.quantile(.99):+.1%}')

    # 5. spot checks ------------------------------------------------------------------------------------------
    print(f'\n5. SPOT CHECKS (compare with adjusted closes on a public site)')
    size = d[~d.is_market].groupby('gvkey').mcap_lag1.median()
    top = ok.assign(size=ok.gvkey.map(size).to_numpy(), i=np.arange(len(ok)))[full]
    top = top[top.ticker.notna()].sort_values('size', ascending=False, na_position='last').head(5)
    if top.empty:
        print('   no calls with a ticker and a full window')
    for r in top.itertuples():
        bh = np.prod(1 + S[r.i, 1:H + 1]) - 1
        print(f'   {str(r.ticker):6s} day 0 = {cal[pos[r.i]].date()}, day {H} = {cal[pos[r.i] + H].date()}: '
              f'stock {bh:+.1%} (buy-and-hold, days 1-{H}), filled days {int(filled[r.i, 1:].sum())}')
    print('   Yahoo Finance: (Adj Close on the day-60 date) / (Adj Close on the day-0 date) - 1 should be close.')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        cfg = load_config()
        main(paths(cfg)['final'], path(cfg, 'interim', f"daily_returns_{cfg['tag']}.parquet"))
