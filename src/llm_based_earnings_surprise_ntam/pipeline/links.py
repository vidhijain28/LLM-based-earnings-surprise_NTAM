"""Candidate IBES tickers for each call. CIQ carries no IBES identifier, so we combine four routes
(see notebooks/ibes_transcripts_link_eda.ipynb), each evaluated as of the call date where links are dated:

  A. CUSIP   ciq.wrds_cusip (first 8 chars) -> ibes.id.cusip
  B. CCM     ciq.wrds_gvkey -> crsp.ccmxpf_lnkhist -> permno -> wrdsapps_link_crsp_ibes.ibcrsphist
  C. ibtic   gvkey -> comp.security / comp.g_security.ibtic
  D. SEDOL   gvkey -> Compustat SEDOL -> IBES international cusip (2-letter country code + first 6 chars of SEDOL)

Which candidate is right is decided later by the announcement date (merge step).
"""
import pandas as pd

from .config import connect

ROUTES = ['cusip', 'ccm', 'ibtic', 'sedol']  # order = preference when breaking ties


def _dated(df, start, end, on='call_date'):
    return (df[start].isna() | (df[on] >= df[start])) & (df[end].isna() | (df[on] <= df[end]))


def build(cfg, calls_path, target):
    calls = pd.read_parquet(calls_path, columns=['keydevid', 'companyid', 'call_date'])
    ids = [int(i) for i in calls.companyid.dropna().unique()]
    glob = cfg['universe'] == 'global'
    db = connect(cfg)
    try:
        gv = db.raw_sql('select companyid::bigint as companyid, gvkey, startdate, enddate, primaryflag '
                        'from ciq.wrds_gvkey where companyid = any(%(ids)s)', params={'ids': ids},
                        date_cols=['startdate', 'enddate'])
        cus = db.raw_sql('select companyid::bigint as companyid, cusip, startdate, enddate '
                         'from ciq.wrds_cusip where companyid = any(%(ids)s)', params={'ids': ids},
                         date_cols=['startdate', 'enddate'])
        iid = db.raw_sql('select distinct ticker, cusip, usfirm from ibes.id where cusip is not null')
        ccm = db.raw_sql("""select gvkey, lpermno::bigint as permno, linkdt, linkenddt from crsp.ccmxpf_lnkhist
                            where linktype in ('LU', 'LC') and linkprim in ('P', 'C') and lpermno is not null""",
                         date_cols=['linkdt', 'linkenddt'])
        ibc = db.raw_sql('select ticker, permno::bigint as permno, sdate, edate from wrdsapps_link_crsp_ibes.ibcrsphist '
                         'where permno is not null', date_cols=['sdate', 'edate'])
        sec = db.raw_sql('select gvkey, ibtic, sedol from comp.security'
                         + (' union all select gvkey, ibtic, sedol from comp.g_security' if glob else ''))
    finally:
        db.close()

    # date-aware primary gvkey per call
    g = calls.merge(gv, on='companyid')
    g = g[_dated(g, 'startdate', 'enddate')].sort_values('primaryflag', ascending=False).drop_duplicates('keydevid')
    g = g[['keydevid', 'call_date', 'gvkey']]
    # IBES-CRSP links still open when the table was last updated are cut at that date: treat as open-ended
    ibc.loc[ibc.edate.eq(ibc.edate.max()), 'edate'] = pd.NaT

    found = {}
    a = calls.merge(cus, on='companyid')
    a = a[_dated(a, 'startdate', 'enddate')].assign(cusip8=lambda d: d.cusip.str[:8])
    found['cusip'] = a.merge(iid[['cusip', 'ticker']].drop_duplicates(), left_on='cusip8', right_on='cusip')
    b = g.merge(ccm, on='gvkey')
    b = b[_dated(b, 'linkdt', 'linkenddt')].merge(ibc, on='permno')
    found['ccm'] = b[_dated(b, 'sdate', 'edate')]
    found['ibtic'] = g.merge(sec.dropna(subset=['ibtic'])[['gvkey', 'ibtic']].drop_duplicates(), on='gvkey') \
                      .rename(columns={'ibtic': 'ticker'})
    if glob:
        intl = iid[iid.usfirm.eq(0)].assign(sed6=lambda d: d.cusip.str[2:8])[['sed6', 'ticker']].drop_duplicates()
        sed = sec.dropna(subset=['sedol']).assign(sed6=lambda d: d.sedol.str[:6])[['gvkey', 'sed6']].drop_duplicates()
        found['sedol'] = g.merge(sed, on='gvkey').merge(intl, on='sed6')

    cand = pd.concat([d[['keydevid', 'ticker']].drop_duplicates().assign(route=k) for k, d in found.items()])
    cand['route_rank'] = cand.route.map({r: i for i, r in enumerate(ROUTES)})
    cand = (cand.sort_values('route_rank').groupby(['keydevid', 'ticker'], as_index=False)
                .agg(route_rank=('route_rank', 'min'), routes=('route', '+'.join)))
    cand['link_route'] = cand.route_rank.map(dict(enumerate(ROUTES)))
    # the call's date-aware gvkey travels with the candidates (used for Compustat prices)
    cand = cand.merge(g[['keydevid', 'gvkey']], on='keydevid', how='left')
    cand.to_parquet(target, index=False)
    n = calls.keydevid.isin(cand.keydevid).mean()
    print(f'Links: {len(cand):,} (call, IBES ticker) candidates; {n:.1%} of {len(calls):,} calls have at least one '
          f'(routes: {", ".join(f"{k} {d.keydevid.nunique():,}" for k, d in found.items())})')
