"""The grid's inputs for each tradable call: daily abnormal returns, attention, industry and next announcement.

Built from pull.py's raw WRDS files with the cleaning rules of the team's Compustat panel (PIPELINE.md 7.2,
scripts/extend_compustat_prices.py on main):
    drop duplicate rows, rows with a missing or non-positive close and non-USD rows; the trading calendar is the
    days the market ETF (VTI) traded; a return is close to close from the security's previous record, valid only
    if that record is the previous trading day and its close is at least $0.01: the total return (dividend factor
    trfd) where it exists, else the split-adjusted price return. Nothing is imputed.
    The call's security is the company's primary security on the event date: the one with the highest mean
    dollar volume over its previous 20 records (the team's `primary_pit`).
Abnormal return = stock return - VTI return; a missing stock return (no trade, delisted) counts as 0 abnormal.
Attention = dollar volume on the event date / mean dollar volume over the previous `attention_days` trading days.
Next event date = the next I/B/E/S quarterly announcement after the call's, moved to a trading day with the
pipeline's release-time rule (pipeline.merge.event_dates, config.yaml match.release_cutoff_et).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..pipeline.config import load_config
from ..pipeline.merge import event_dates
from ..signals.ff12 import ff12_name


def _secd(raw):
    d = pd.read_parquet(raw / 'comp_secd.parquet')
    d['gvkey'] = d['gvkey'].astype(str).str.strip()
    d['iid'] = d['iid'].astype(str).str.strip()
    d['datadate'] = pd.to_datetime(d['datadate'])
    for c in ['prccd', 'cshtrd', 'cshoc', 'ajexdi', 'trfd', 'div']:
        d[c] = pd.to_numeric(d[c], errors='coerce').astype('float64')
    d['curcdd'] = d['curcdd'].astype('string').str.strip()
    n0 = len(d)
    d = d.drop_duplicates(['gvkey', 'iid', 'datadate'])
    d = d[d['prccd'].notna() & (d['prccd'] > 0)]
    d = d[d['curcdd'].eq('USD').fillna(False)]
    return d, n0


def _derive(d, cal):
    """Daily return, dollar volume and the team's primary-security flag, each from earlier records only."""
    d = d.sort_values(['gvkey', 'iid', 'datadate'], kind='mergesort').reset_index(drop=True)
    t = pd.Series(np.arange(len(cal)), index=cal).reindex(d['datadate']).to_numpy(float)
    same = (d['gvkey'].eq(d['gvkey'].shift(1)) & d['iid'].eq(d['iid'].shift(1))).to_numpy()

    def prev(x):
        x = np.asarray(x, float)
        out = np.r_[np.nan, x[:-1]]
        out[~same] = np.nan
        return out

    adj = (d['prccd'] / d['ajexdi']).to_numpy()
    tr = adj * d['trfd'].to_numpy()
    valid = (prev(t) == t - 1) & (prev(d['prccd']) >= 0.01)
    ret_tr = np.where(valid, tr / prev(tr) - 1, np.nan)
    ret_px = np.where(valid, adj / prev(adj) - 1, np.nan)
    d['t'] = t.astype(int)
    d['ret'] = np.where(np.isfinite(ret_tr), ret_tr, ret_px)
    d['dv'] = (d['prccd'] * d['cshtrd']).to_numpy()
    dv_prev = pd.Series(prev(d['dv']), index=d.index)
    d['adv_lag1'] = dv_prev.groupby([d['gvkey'], d['iid']], sort=False).transform(
        lambda s: s.rolling(20, min_periods=1).mean())
    rank = d.groupby(['gvkey', 'datadate'])['adv_lag1'].rank(method='first', ascending=False)
    only = d.groupby(['gvkey', 'datadate'])['iid'].transform('size').eq(1)
    d['primary_pit'] = rank.eq(1) | only
    return d


def build(cfg, calls):
    """(calls with t0, industry, attention and next-event position; calendar; abnormal returns n x T; info)."""
    raw = Path(cfg['paths']['raw'])
    r = cfg['rules']
    log = json.loads((raw / 'pull_log.json').read_text())
    d, n_raw = _secd(raw)
    mg, mi = log['market']['gvkey'], log['market']['iid']
    is_mkt = d['gvkey'].eq(mg) & d['iid'].eq(mi)
    cal = pd.DatetimeIndex(sorted(d.loc[is_mkt, 'datadate'].unique()))
    d = d[d['datadate'].isin(cal)]
    d = _derive(d, cal)
    T = len(cal)

    mkt = np.full(T, np.nan)
    m = d[d['gvkey'].eq(mg) & d['iid'].eq(mi)]
    mkt[m['t'].to_numpy()] = m['ret'].to_numpy()

    calls = calls.copy()
    calls['gvkey'] = calls['gvkey'].astype(str).str.strip()
    calls['event_date'] = pd.to_datetime(calls['event_date'])
    pos = pd.Series(np.arange(T), index=cal)
    calls['t0'] = calls['event_date'].map(pos)

    # the call's security: the primary one on the event date
    prim = d[d['primary_pit']][['gvkey', 'iid', 'datadate']]
    calls = calls.merge(prim.rename(columns={'datadate': 'event_date'}), on=['gvkey', 'event_date'], how='left')
    lost = {'no_event_day_on_calendar': int(calls['t0'].isna().sum()),
            'no_price_on_event_date': int((calls['t0'].notna() & calls['iid'].isna()).sum())}
    calls = calls[calls['t0'].notna() & calls['iid'].notna()].reset_index(drop=True)
    calls['t0'] = calls['t0'].astype(int)
    n = len(calls)

    # stock returns and dollar volume, one row per call
    key = calls['gvkey'] + '|' + calls['iid']
    d['key'] = d['gvkey'] + '|' + d['iid']
    d = d[d['key'].isin(set(key))]
    row = {k: i for i, k in enumerate(sorted(set(key)))}
    R = np.full((len(row), T), np.nan)
    DV = np.full((len(row), T), np.nan)
    ri = d['key'].map(row).to_numpy()
    R[ri, d['t'].to_numpy()] = d['ret'].to_numpy()
    DV[ri, d['t'].to_numpy()] = d['dv'].to_numpy()
    rows = key.map(row).to_numpy()
    AR = R[rows] - mkt[None, :]
    AR = np.where(np.isfinite(AR), AR, 0.0)

    # attention ratio
    att = np.full(n, np.nan)
    L, need = int(r['attention_days']), int(r['attention_min_days'])
    for i, (k, t0) in enumerate(zip(rows, calls['t0'])):
        if t0 < L:
            continue
        hist = DV[k, t0 - L:t0]
        ok = np.isfinite(hist) & (hist > 0)
        if ok.sum() >= need and np.isfinite(DV[k, t0]):
            att[i] = DV[k, t0] / hist[ok].mean()
    calls['attention'] = att

    # industry
    comp = pd.read_parquet(raw / 'comp_company.parquet')
    comp['gvkey'] = comp['gvkey'].astype(str).str.strip()
    sic = calls['gvkey'].map(comp.drop_duplicates('gvkey').set_index('gvkey')['sic'])
    calls['industry'] = ff12_name(pd.to_numeric(sic, errors='coerce')).astype(object).values

    # next announcement -> next event date -> its position on the calendar
    ib = pd.read_parquet(raw / 'ibes_next.parquet')
    ib['anndats'] = pd.to_datetime(ib['anndats'])
    ib['pends'] = pd.to_datetime(ib['pends'])
    nxt = calls[['ticker', 'anndats', 'pends']].copy()
    nxt['anndats'] = pd.to_datetime(nxt['anndats'])
    nxt['pends'] = pd.to_datetime(nxt['pends'])
    nxt['i'] = np.arange(n)
    cand = nxt.merge(ib, on='ticker', suffixes=('', '_n'))
    cand = cand[(cand['anndats_n'] > cand['anndats']) & (cand['pends_n'] > cand['pends'])]
    cand = cand.sort_values(['i', 'anndats_n']).drop_duplicates('i')
    cand = cand.assign(country='US', anndats=cand['anndats_n'], anntims=cand['anntims'])
    cand['next_event'] = event_dates(cand, load_config()['match']['release_cutoff_et'])
    calls['next_event'] = pd.Series(cand['next_event'].to_numpy(), index=cand['i'].to_numpy()).reindex(range(n))
    calls['t_next'] = calls['next_event'].map(pos)

    info = {'raw_rows': n_raw, 'calendar': [str(cal[0].date()), str(cal[-1].date()), T],
            'calls_in': int(n + sum(lost.values())), 'calls_kept': n, **lost,
            'attention_missing': int(np.isnan(att).sum()), 'industry_missing': int(calls['industry'].isna().sum()),
            'next_event_missing': int(calls['t_next'].isna().sum()),
            'stock_return_missing_share_days_1_60': float(np.mean([
                np.isnan(R[k, t0 + 1:t0 + 61]).mean() for k, t0 in zip(rows, calls['t0'])]))}
    return calls, cal, AR, info
