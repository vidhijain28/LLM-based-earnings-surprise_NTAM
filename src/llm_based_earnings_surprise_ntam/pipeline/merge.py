"""Match each call to its IBES announcement and set the event date (`merge`); then compute the SUE with the
pre-announcement price and write the final dataset (`finalize`).

Match rule (from notebooks/ibes_transcripts_link_eda.ipynb): among the call's candidate tickers, take the IBES
announcement with call_date - anndats inside match.window_days. Ties are broken by
  1. smallest |gap|, preferring the day after the release over the day before;
  2. the IBES file matching the company's domicile (home listing over a US ADR line, and vice versa);
  3. the strongest link route (cusip > ccm > ibtic > sedol), then ticker.
Each IBES announcement is then kept for one call only.

Event date (US companies; Koijen & Levy): the release day if it is an NYSE trading day and the release time
is at or before match.release_cutoff_et, otherwise the next trading day.

SUE = (actual - consensus) / price, price = close on the trading day before the event date (ibes.price_source:
compustat), falling back to the IBES price snapshot when no Compustat close is available.
"""
import json

import exchange_calendars as xcals
import numpy as np
import pandas as pd

IBES_COLS = ['ticker', 'ibes_file', 'pdicity', 'pends', 'anndats', 'anntims', 'actual', 'curr_act', 'consensus',
             'consensus_median', 'consensus_mean', 'consensus_std', 'n_estimates', 'consensus_date',
             'consensus_age_days', 'surprise', 'price_ibes', 'price_ibes_date']


def match_calls(calls, links, ann, window):
    m = links.merge(calls[['keydevid', 'call_date', 'country']], on='keydevid').merge(ann[IBES_COLS], on='ticker')
    m['match_gap_days'] = (m.call_date - m.anndats).dt.days
    m = m[m.match_gap_days.between(*window)].copy()
    m['abs_gap'] = m.match_gap_days.abs()
    m['file_mismatch'] = (m.ibes_file.eq('US') != m.country.eq('US')).astype(int)
    m = m.sort_values(['keydevid', 'abs_gap', 'match_gap_days', 'file_mismatch', 'route_rank', 'ticker'],
                      ascending=[True, True, False, True, True, True])
    best = m.drop_duplicates('keydevid')
    tie = m.merge(best[['keydevid', 'abs_gap', 'match_gap_days']], on=['keydevid', 'abs_gap', 'match_gap_days'])
    best = best.merge(tie.groupby('keydevid').ticker.nunique().gt(1).rename('ticker_tie'), on='keydevid')
    return best.drop(columns=['call_date', 'country', 'abs_gap', 'file_mismatch', 'route_rank'])


def event_dates(df, cutoff):
    """NYSE event date from the IBES release date + time (ET). Missing for non-US companies or unknown times."""
    out = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
    ann = df.anndats.astype('datetime64[ns]')
    t = pd.to_timedelta(df.anntims.astype('string'), errors='coerce')
    ok = df.country.eq('US') & ann.notna() & t.notna() & (t != pd.Timedelta(0))
    if not ok.any():
        return out
    cal = xcals.get_calendar('XNYS', start=ann[ok].min() - pd.Timedelta(days=10),
                             end=ann[ok].max() + pd.Timedelta(days=10))
    sessions = cal.sessions.tz_localize(None).values.astype('datetime64[ns]')
    d = ann[ok].values
    pos = np.searchsorted(sessions, d, side='left')
    on_session = sessions[pos] == d
    late = (t[ok] > pd.Timedelta(cutoff + ':00')).values
    out[ok] = sessions[np.where(on_session & late, pos + 1, pos)]
    return out


def build(cfg, calls_path, links_path, ibes_path, target):
    """Calls + matched IBES announcement + event date (no SUE yet, no text)."""
    calls = pd.read_parquet(calls_path)
    links = pd.read_parquet(links_path)
    ann = pd.read_parquet(ibes_path)
    best = match_calls(calls, links, ann, cfg['match']['window_days'])

    df = calls.merge(best, on='keydevid', how='left')
    df['matched'] = df.ticker.notna()
    df['ticker_tie'] = df.ticker_tie.astype('boolean')
    # one call per IBES announcement (duplicate CIQ events a day apart): smallest gap, then longest transcript
    df['abs_gap'] = df.match_gap_days.abs()
    df = df.sort_values(['abs_gap', 'n_words'], ascending=[True, False])
    dup = df.matched & df.duplicated(['ticker', 'anndats'])
    df = df[~dup].drop(columns='abs_gap')
    df['event_date'] = event_dates(df, cfg['match']['release_cutoff_et'])
    df = df.sort_values(['call_date', 'keydevid']).reset_index(drop=True)
    df.to_parquet(target, index=False)
    print(f'Matched {df.matched.mean():.1%} of {len(df):,} calls ({int(dup.sum())} duplicate calls dropped); '
          f'event date for {df.event_date.notna().mean():.1%}')


def finalize(cfg, matched_path, prices_path, text_path, ibes_path, target, report_path):
    df = pd.read_parquet(matched_path)
    df = df.merge(pd.read_parquet(prices_path), on='keydevid', how='left')
    c = cfg['ibes']

    # pre-announcement price: t-1 close (Compustat) when available and in the actual's currency, else IBES price
    close_ok = df.close_price.notna() & df.close_currency.eq(df.curr_act).fillna(False)
    use_close = close_ok if c['price_source'] == 'compustat' else pd.Series(False, index=df.index)
    df['price'] = df.close_price.where(use_close, df.price_ibes)
    df['price_date'] = df.close_date.where(use_close, df.price_ibes_date)
    df['price_source'] = pd.Series(np.select([use_close, df.price_ibes.notna()], ['compustat_t-1_close', 'ibes_snapshot'],
                                             default=None), index=df.index).astype('string')
    ok = df.surprise.notna() & (df.price >= c['min_price'])
    df['sue'] = (df.surprise / df.price).where(ok).astype(float)

    # was the chosen transcript version available by the event-day close (16:00 ET)?
    close_utc = (df.event_date + pd.Timedelta(hours=16)).dt.tz_localize('America/New_York', nonexistent='NaT',
                                                                         ambiguous='NaT').dt.tz_convert(None)
    df['transcript_by_event_close'] = (df.created_utc <= close_utc).astype('boolean').mask(df.event_date.isna())
    # within-quarter percentile rank of SUE (re-rank after any further sample filtering)
    qtr = df.event_date.fillna(df.anndats).dt.to_period('Q')
    df['event_quarter'] = qtr.astype('string')
    df['sue_pct'] = df.groupby(qtr).sue.rank(pct=True)

    df = df.drop(columns=['close_price', 'close_date', 'close_currency'])
    if cfg['output']['include_text']:
        df = df.merge(pd.read_parquet(text_path).drop(columns='keydevid'), on='transcriptid', how='left')
    df = df.sort_values(['call_date', 'keydevid']).reset_index(drop=True)
    df.to_parquet(target, index=False)

    m = df[df.matched]
    us = df.country.eq('US')
    rate = lambda s: round(float(s.mean()), 4) if len(s) else None
    report = {
        'config': {k: cfg[k] for k in ['period', 'universe', 'transcripts', 'ibes', 'match']},
        'n_calls': int(len(df)),
        'matched_share': rate(df.matched),
        'quarterly_actual_share_of_matched': rate(m.pdicity.eq('QTR')),
        'consensus_share_of_matched': rate(m.consensus.notna()),
        'sue_share': rate(df.sue.notna()),
        'sue_share_us': rate(df[us].sue.notna()) if us.any() else None,
        'price_source_share_of_sue': {k: round(float(v), 4) for k, v in
                                      df[df.sue.notna()].price_source.value_counts(normalize=True).items()},
        'median_n_estimates': float(m.n_estimates.median()),
        'median_consensus_age_days': float(m.consensus_age_days.median()),
        'ticker_tie_share_of_matched': rate(m.ticker_tie),
        'event_date_share': rate(df.event_date.notna()),
        'transcript_by_event_close_share': rate(df.transcript_by_event_close.dropna()),
        'match_gap_days': {str(int(k)): int(v) for k, v in df.match_gap_days.value_counts().sort_index().items()},
        'calls_after_last_ibes_announcement': int((df.call_date > pd.read_parquet(ibes_path, columns=['anndats'])
                                                   .anndats.max()).sum()),
        'output': str(target),
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Final: {report['n_calls']:,} calls, matched {report['matched_share']:.1%}, SUE {report['sue_share']:.1%} "
          f"(price: {report['price_source_share_of_sue']}) -> {target}")
