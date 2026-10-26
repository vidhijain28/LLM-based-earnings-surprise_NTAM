"""Consensus (expected) EPS and actual (reported) EPS for quarterly earnings announcements, from I/B/E/S.

Usage:
    import wrds
    from src.ibes_consensus import consensus_eps

    db = wrds.Connection()
    mean_eps, actual = consensus_eps(db, ticker='AAPL', release_date='2025-07-31')
    # release_date = the day the company released its earnings (the event date)
    db.close()

Procedure (consensus_eps):
    1. Pull next-quarter (fpi='6') analyst forecasts from ibes.det_epsus for the ticker and release date.
    2. Keep forecasts entered into I/B/E/S before the release date (up to the day before).
    3. Keep each analyst's latest forecast.
    4. Average them: mean_eps = consensus (forecast); actual = reported EPS, if available.

Other functions:
    fetch_detail             step 1 (raw forecast rows, all columns)
    latest_analyst_forecasts steps 2-3 (one row per analyst per event, all columns)
    latest_snapshot_by_fpi   alternative consensus from the monthly summary table (ibes.statsum_epsus):
                             latest snapshot on or before the announcement, one row per fpi
"""

import pandas as pd

QUARTER_FPI = '6'  # next fiscal quarter
DETAIL_TABLES = ('det_epsus', 'detu_epsus')  # split-adjusted, unadjusted


def latest_snapshot_by_fpi(statsum):
    """For each quarterly announcement, the latest snapshot's rows (one per fpi).

    statsum: rows from ibes.statsum_epsus (needs ticker, statpers, measure, fpi, fpedats, anndats_act).

    Returns the snapshot rows plus event columns:
        event_fpedats  fiscal quarter end being announced
        event_date     announcement date (anndats_act of the fpi='6' row)
    One row per (event, fpi). Announcements with no snapshot on or before the event date are dropped.
    """
    s = statsum.copy()
    for c in ['statpers', 'fpedats', 'anndats_act']:
        s[c] = pd.to_datetime(s[c], errors='coerce')
    s['fpi'] = s.fpi.astype(str)
    s = s[s.measure.eq('EPS')]

    # Events: announced quarters, taken from the next-quarter rows
    q = s[s.fpi.eq(QUARTER_FPI) & s.anndats_act.notna()]

    # Latest snapshot on or before the announcement that still forecasts this quarter as fpi='6'
    q = q[q.statpers <= q.anndats_act]
    snap_dates = (q.groupby(['ticker', 'fpedats', 'anndats_act'], as_index=False).statpers.max()
                   .rename(columns={'fpedats': 'event_fpedats', 'anndats_act': 'event_date'}))

    # All fpi rows of that snapshot
    out = snap_dates.merge(s, on=['ticker', 'statpers'], how='inner')
    cols = ['ticker', 'event_fpedats', 'event_date', 'statpers', 'fpi', 'fpedats']
    out = out[cols + [c for c in out.columns if c not in cols]]
    return out.sort_values(['ticker', 'event_date', 'fpi']).reset_index(drop=True)


def fetch_detail(db, events, table='det_epsus'):
    """Next-quarter (fpi='6') analyst forecasts from an I/B/E/S detail table for the given announcements.

    db:     an open wrds.Connection
    events: list of (ticker, release_date) pairs, e.g. [('AAPL', '2025-07-31'), ('MSFT', '2025-07-30')].
            ticker is the I/B/E/S ticker (the `ticker` column, not `oftic`);
            release_date ('YYYY-MM-DD') must equal anndats_act exactly.
    table:  'det_epsus' (split-adjusted) or 'detu_epsus' (unadjusted)
    """
    if table not in DETAIL_TABLES:
        raise ValueError(f'table must be one of {DETAIL_TABLES}, got {table!r}')
    events = list(dict.fromkeys((str(t), str(pd.Timestamp(d).date())) for t, d in events))
    if not events:
        raise ValueError('events is empty')
    ev_list = ','.join("('" + t.replace("'", "''") + f"', date '{d}')" for t, d in events)
    return db.raw_sql(f'''
        with ev(ticker, anndats_act) as (values {ev_list})
        select d.*
        from ibes.{table} d
        join ev on ev.ticker = d.ticker and ev.anndats_act = d.anndats_act
        where d.fpi = '{QUARTER_FPI}'
    ''')


def latest_analyst_forecasts(det):
    """Each analyst's latest valid forecast per quarterly announcement. All original columns kept.

    det: rows from ibes.det_epsus / detu_epsus (e.g. from `fetch_detail`).

    For each event (ticker, fpedats, anndats_act), a forecast is kept if:
        - it is a next-quarter forecast (fpi='6') of EPS, i.e. made after the previous quarter was reported,
        - it was entered into I/B/E/S before the announcement date (actdats < anndats_act),
        - it is the analyst's (estimator, analys) latest such forecast, by entry date/time.
    """
    d = det.copy()
    act = pd.to_datetime(d.actdats, errors='coerce')
    ann_act = pd.to_datetime(d.anndats_act, errors='coerce')
    entered = pd.to_datetime(d.actdats.astype(str) + ' ' + d.acttims.astype(str), errors='coerce')

    keep = (d.measure.eq('EPS') & d.fpi.astype(str).eq(QUARTER_FPI)
            & ann_act.notna() & act.notna() & (act < ann_act))
    keep = keep.fillna(False).astype(bool)  # WRDS returns nullable dtypes
    d = d[keep].assign(_entered=entered[keep])

    event_analyst = ['ticker', 'fpedats', 'anndats_act', 'estimator', 'analys']
    d = d.sort_values(event_analyst + ['_entered', 'anndats', 'anntims'], kind='mergesort', na_position='first')
    d = d.drop_duplicates(event_analyst, keep='last').drop(columns='_entered')
    return d.sort_values(['ticker', 'anndats_act', 'actdats'], kind='mergesort').reset_index(drop=True)


def consensus_eps(db, ticker, release_date, table='det_epsus'):
    """Consensus (forecast) EPS and actual EPS for one earnings announcement.

    db:         an open wrds.Connection
    ticker:     I/B/E/S ticker, e.g. 'AAPL'
    release_date: earnings release date ('YYYY-MM-DD'), i.e. the event date, not the quarter-end date;
                  must equal I/B/E/S anndats_act
    table:      'det_epsus' (split-adjusted) or 'detu_epsus' (unadjusted)

    Returns (mean_eps, actual). Either is None if not available.
    """
    fc = latest_analyst_forecasts(fetch_detail(db, [(ticker, release_date)], table=table))
    if fc.empty:
        return None, None
    if fc.fpedats.nunique() > 1:
        raise ValueError(f'{ticker} reported more than one quarter on {release_date}')
    actual = fc.actual.dropna()
    return float(fc.value.mean()), (float(actual.iloc[0]) if len(actual) else None)
