"""Event-date abnormal return for each matched call (extension; see PIPELINE.md section 5).

    r_abn = stock return - market return on event_date

Both come from CRSP daily v2 (`crsp.dsf_v2`, column `dlyret`: total return, dividends included). The market
proxy is VTI (permno 90878), the ETF used by Koijen & Levy; CRSP's own value-weighted index files
(`crsp.dsi`, `crsp.dsix`) stop at 2024-12-31 and cannot cover recent quarters.

gvkey -> permno comes from the CRSP/Compustat link (`crsp.ccmxpf_lnkhist`, primary links only), evaluated on
the event date. Calls without a match, without an event date or without a gvkey get no return.
"""
import pandas as pd

from .config import connect

MARKET_PERMNO = 90878  # VTI (Vanguard Total Stock Market ETF)

LINK_SQL = """
    select gvkey, lpermno::bigint as permno, linkprim, linkdt, linkenddt
    from crsp.ccmxpf_lnkhist
    where linktype in ('LU', 'LC') and linkprim in ('P', 'C') and lpermno is not null
      and gvkey = any(%(g)s)
"""
RET_SQL = """
    select permno::bigint as permno, dlycaldt as date, dlyret as ret, dlyclose as close
    from crsp.dsf_v2
    where permno = any(%(p)s) and dlycaldt between %(s)s and %(e)s
"""
OUT_COLS = ['keydevid', 'permno', 'event_date', 'ret', 'mkt_ret', 'r_abn', 'n_permno_candidates']


def _pick_permno(calls, link):
    """One permno per call: link valid on the event date, primary ('P') preferred, lowest permno to break ties."""
    lk = calls.merge(link, on='gvkey')
    ok = ((lk.linkdt.isna() | (lk.event_date >= lk.linkdt))
          & (lk.linkenddt.isna() | (lk.event_date <= lk.linkenddt)))
    lk = lk[ok.fillna(False)].copy()
    n_cand = lk.groupby('keydevid').permno.nunique().rename('n_permno_candidates')
    lk['prim_rank'] = lk.linkprim.map({'P': 0, 'C': 1}).fillna(2)
    lk = lk.sort_values(['keydevid', 'prim_rank', 'permno'], kind='mergesort').drop_duplicates('keydevid')
    return lk[['keydevid', 'event_date', 'permno']].merge(n_cand, on='keydevid', how='left')


def pull(cfg, matched_path, prices_path, target):
    mkt_permno = int((cfg.get('returns') or {}).get('market_permno', MARKET_PERMNO))
    m = pd.read_parquet(matched_path, columns=['keydevid', 'gvkey', 'event_date', 'matched'])
    calls = m[m.matched.fillna(False) & m.event_date.notna() & m.gvkey.notna()].copy()
    calls['event_date'] = calls.event_date.astype('datetime64[ns]')
    if calls.empty:
        pd.DataFrame(columns=OUT_COLS).to_parquet(target, index=False)
        print('Returns: no matched calls with an event date')
        return

    lo, hi = calls.event_date.min(), calls.event_date.max()
    db = connect(cfg)
    try:
        link = db.raw_sql(LINK_SQL, params={'g': sorted(calls.gvkey.unique())},
                          date_cols=['linkdt', 'linkenddt'])
        px = _pick_permno(calls, link)
        permnos = sorted({int(p) for p in px.permno.dropna()} | {mkt_permno})
        ret = db.raw_sql(RET_SQL, params={'p': permnos, 's': lo.date(), 'e': hi.date()}, date_cols=['date'])
        crsp_end = db.raw_sql('select max(dlycaldt) as d from crsp.dsf_v2', date_cols=['d']).d.iloc[0]
    finally:
        db.close()

    ret['date'] = ret.date.astype('datetime64[ns]')
    mkt = ret[ret.permno.eq(mkt_permno)][['date', 'ret']].rename(columns={'ret': 'mkt_ret'})
    stock = ret[ret.permno.ne(mkt_permno)] if len(permnos) > 1 else ret

    out = (px.merge(stock[['permno', 'date', 'ret', 'close']], left_on=['permno', 'event_date'],
                    right_on=['permno', 'date'], how='left')
             .merge(mkt, left_on='event_date', right_on='date', how='left', suffixes=('', '_mkt')))
    out['r_abn'] = out.ret - out.mkt_ret
    out = calls[['keydevid']].merge(out, on='keydevid', how='left')

    # cross-check against the t-1 close already used for the SUE denominator
    prev = pd.read_parquet(prices_path, columns=['keydevid', 'close_price'])
    chk = out.merge(prev, on='keydevid', how='left')
    implied = chk.close / chk.close_price - 1
    gap = (implied - chk.ret).abs()
    n_chk, n_off = int(gap.notna().sum()), int((gap > 0.01).sum())

    out[OUT_COLS].to_parquet(target, index=False)
    print(f'Returns: r_abn for {out.r_abn.notna().mean():.1%} of {len(calls):,} matched calls with an event date '
          f'(permno for {out.permno.notna().mean():.1%}; market proxy permno {mkt_permno}; '
          f'CRSP daily ends {pd.Timestamp(crsp_end).date()}). '
          f'Price cross-check: {n_off:,} of {n_chk:,} differ from close/close_t-1 by more than 1pp')
    if pd.Timestamp(crsp_end) < hi:
        print(f'WARNING: event dates run to {hi.date()} but CRSP daily ends {pd.Timestamp(crsp_end).date()}; '
              'later calls get no return.')
