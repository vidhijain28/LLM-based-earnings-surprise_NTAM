"""Daily returns for the long-short backtest: days 0..60 after each in-sample call (PIPELINE.md sections 6 and 7.2).

CRSP daily v2 ends too early for 4- and 12-week windows on recent quarters, so this uses Compustat daily
(comp.secd), the primary US issue of each company (comp.company.priusa, as in the prices step) plus VTI as the
market. Rules of section 7.2:
    - drop rows with a missing or non-positive close and non-USD rows; keep the days VTI traded
    - close-to-close total return (dividend factor trfd) where it exists on both days, else split-adjusted price
    - a return needs the previous record to be the previous trading day and a previous close of at least $0.01
    - nothing is imputed (the backtest fills missing days with the market return)
Output (long format): gvkey, iid, date, ret, mcap_lag1, fic, is_market.
"""
import numpy as np
import pandas as pd

from .config import connect

MARKET = ('143851', '01')  # VTI in Compustat (gvkey, iid)
CALENDAR_DAYS_AFTER = 100  # covers 60 trading days after the last event date

SQL = """
    select s.gvkey, s.iid, s.datadate as date, s.prccd, s.ajexdi, s.trfd, s.cshoc, s.curcdd, s.fic
    from comp.secd s join comp.company c on c.gvkey = s.gvkey
    where s.datadate between %(s)s and %(e)s
      and ((s.gvkey = any(%(g)s) and s.iid = c.priusa) or (s.gvkey = %(mg)s and s.iid = %(mi)s))
"""
OUT_COLS = ['gvkey', 'iid', 'date', 'ret', 'mcap_lag1', 'fic', 'is_market']


def build(raw):
    """Compustat daily rows -> daily returns on VTI's trading calendar (rules in the module docstring)."""
    px = raw[(raw.prccd > 0) & raw.curcdd.eq('USD')].copy()
    px['date'] = px.date.astype('datetime64[ns]')
    px = px.drop_duplicates(['gvkey', 'iid', 'date'])  # one row per security and day (as in section 7.2)
    px['is_market'] = px.gvkey.eq(MARKET[0]) & px.iid.eq(MARKET[1])
    cal = pd.DatetimeIndex(sorted(px.loc[px.is_market, 'date'].unique()))
    if cal.empty:
        raise SystemExit(f'Daily returns: no VTI rows (gvkey {MARKET[0]}, iid {MARKET[1]}) in comp.secd for the '
                         'period, so there is no trading calendar. Check MARKET in pipeline/daily_returns.py.')
    px = px[px.date.isin(cal)].sort_values(['gvkey', 'iid', 'date'])
    px['adj'] = px.prccd / px.ajexdi
    px['tri'] = px.adj * px.trfd
    g = px.groupby(['gvkey', 'iid'], sort=False)
    day = pd.Series(np.arange(len(cal)), index=cal)
    consecutive = (px.date.map(day) - g.date.shift().map(day)) == 1
    r_tot, r_px = px.tri / g.tri.shift() - 1, px.adj / g.adj.shift() - 1
    px['ret'] = np.where(consecutive & (g.prccd.shift() >= 0.01), r_tot.fillna(r_px), np.nan)
    px['mcap_lag1'] = g.prccd.shift() * g.cshoc.shift()
    return px[OUT_COLS].reset_index(drop=True), float(r_tot.notna().mean())


def pull(cfg, final_path, target):
    ev = pd.read_parquet(final_path, columns=['gvkey', 'event_date', 'in_sample'])
    ev = ev[ev.in_sample.fillna(False).astype(bool) & ev.gvkey.notna() & ev.event_date.notna()]
    if ev.empty:
        pd.DataFrame(columns=OUT_COLS).to_parquet(target, index=False)
        print('Daily returns: no in-sample calls')
        return
    dates = ev.event_date.astype('datetime64[ns]')
    lo = (dates.min() - pd.Timedelta(days=10)).date()
    hi = min(dates.max() + pd.Timedelta(days=CALENDAR_DAYS_AFTER), pd.Timestamp.today().normalize()).date()
    db = connect(cfg)
    try:
        raw = db.raw_sql(SQL, params={'s': lo, 'e': hi, 'g': sorted(ev.gvkey.astype(str).unique()),
                                      'mg': MARKET[0], 'mi': MARKET[1]}, date_cols=['date'])
    finally:
        db.close()
    out, total_share = build(raw)
    out.to_parquet(target, index=False)
    print(f'Daily returns: {out.loc[~out.is_market, "gvkey"].nunique():,} companies, {out.date.nunique()} trading '
          f'days ({out.date.min().date()} to {out.date.max().date()}), {total_share:.0%} total-return days '
          '(Compustat comp.secd)')
    cal = np.sort(out.loc[out.is_market, 'date'].unique())
    short = int((len(cal) - np.searchsorted(cal, dates.to_numpy(), side='right') < 60).sum())
    if short:
        print(f'NOTE: {short:,} calls have fewer than 60 trading days of prices after the event date; their '
              '12-week window is left out of the backtest')
