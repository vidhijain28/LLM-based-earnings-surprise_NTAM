"""Pre-announcement price: the close on the last trading day before the event date (the t-1 close).

Source: Compustat daily securities (comp.secd), unadjusted close `prccd` of the company's primary US issue
(comp.company.priusa), so it is on the same per-share basis as the unadjusted IBES EPS. For calls without an event
date (unknown release time) the cut-off is the announcement date instead. Companies without a primary US issue
(most non-US firms) get no close here; the final step then falls back to the IBES price.
"""
import pandas as pd

from .config import connect

SQL = """
    select s.gvkey, s.datadate as close_date, s.prccd as close_price, s.curcdd as close_currency
    from comp.secd s join comp.company c on c.gvkey = s.gvkey and s.iid = c.priusa
    where s.gvkey = any(%(g)s) and s.datadate between %(s)s and %(e)s and s.prccd > 0
"""


def pull(cfg, matched_path, target):
    m = pd.read_parquet(matched_path, columns=['keydevid', 'gvkey', 'anndats', 'event_date', 'matched'])
    m = m[m.matched & m.gvkey.notna()].copy()
    m['cutoff'] = m.event_date.fillna(m.anndats).astype('datetime64[ns]')
    out_cols = ['keydevid', 'close_price', 'close_date', 'close_currency']
    if m.empty:
        pd.DataFrame(columns=out_cols).to_parquet(target, index=False)
        return
    db = connect(cfg)
    try:
        px = db.raw_sql(SQL, params={'g': m.gvkey.unique().tolist(),
                                     's': (m.cutoff.min() - pd.Timedelta(days=15)).date(), 'e': m.cutoff.max().date()},
                        date_cols=['close_date'])
    finally:
        db.close()
    px['close_date'] = px.close_date.astype('datetime64[ns]')
    out = pd.merge_asof(m.sort_values('cutoff'), px.sort_values('close_date'), left_on='cutoff', right_on='close_date',
                        by='gvkey', allow_exact_matches=False, direction='backward', tolerance=pd.Timedelta(days=10))
    out[out_cols].to_parquet(target, index=False)
    print(f'Prices: t-1 close for {out.close_price.notna().mean():.1%} of {len(m):,} matched calls (Compustat comp.secd)')
