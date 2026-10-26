"""Match each call to its IBES announcement, set the event date, and write the final dataset.

Match rule (from notebooks/ibes_transcripts_link_eda.ipynb): among the call's candidate tickers, take the IBES
announcement with call_date - anndats inside match.window_days. Ties are broken by
  1. smallest |gap|, preferring the day after the release over the day before;
  2. the IBES file matching the company's domicile (home listing over a US ADR line, and vice versa);
  3. the strongest link route (cusip > ccm > ibtic > sedol), then ticker.
Each IBES announcement is then kept for one call only.

Event date (US companies; Koijen & Levy): the release day if it is an NYSE trading day and the release time
is at or before match.release_cutoff_et, otherwise the next trading day.
"""
import json

import exchange_calendars as xcals
import numpy as np
import pandas as pd

IBES_COLS = ['ticker', 'ibes_file', 'pdicity', 'pends', 'anndats', 'anntims', 'actual', 'curr_act', 'consensus',
             'consensus_median', 'consensus_mean', 'consensus_std', 'n_estimates', 'consensus_date',
             'consensus_age_days', 'price', 'price_date', 'surprise', 'sue']


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


def build(cfg, calls_path, text_path, links_path, ibes_path, target, report_path):
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
    # was the chosen transcript version available by the event-day close (16:00 ET)?
    close_utc = (df.event_date + pd.Timedelta(hours=16)).dt.tz_localize('America/New_York', nonexistent='NaT',
                                                                         ambiguous='NaT').dt.tz_convert(None)
    df['transcript_by_event_close'] = (df.created_utc <= close_utc).astype('boolean').mask(df.event_date.isna())
    # within-quarter percentile rank of SUE (the paper ranks surprises within the quarter's sample)
    qtr = df.event_date.fillna(df.anndats).dt.to_period('Q')
    df['event_quarter'] = qtr.astype('string')
    df['sue_pct'] = df.groupby(qtr).sue.rank(pct=True)

    if cfg['output']['include_text']:
        df = df.merge(pd.read_parquet(text_path).drop(columns='keydevid'), on='transcriptid', how='left')
    df = df.sort_values(['call_date', 'keydevid']).reset_index(drop=True)
    df.to_parquet(target, index=False)

    us = df.country.eq('US')
    report = {
        'config': {k: cfg[k] for k in ['period', 'universe', 'transcripts', 'ibes', 'match']},
        'n_calls': int(len(df)),
        'matched_share': round(float(df.matched.mean()), 4),
        'quarterly_actual_share_of_matched': round(float(df[df.matched].pdicity.eq('QTR').mean()), 4),
        'sue_share': round(float(df.sue.notna().mean()), 4),
        'sue_share_us': round(float(df[us].sue.notna().mean()), 4) if us.any() else None,
        'ticker_tie_share_of_matched': round(float(df[df.matched].ticker_tie.mean()), 4),
        'duplicate_calls_dropped': int(dup.sum()),
        'event_date_share': round(float(df.event_date.notna().mean()), 4),
        'match_gap_days': {str(k): int(v) for k, v in df.match_gap_days.value_counts().sort_index().items()},
        'calls_after_last_ibes_announcement': int((df.call_date > ann.anndats.max()).sum()),
        'output': str(target),
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Merged: {report['n_calls']:,} calls, matched {report['matched_share']:.1%}, SUE {report['sue_share']:.1%}, "
          f"event date {report['event_date_share']:.1%} -> {target}")
