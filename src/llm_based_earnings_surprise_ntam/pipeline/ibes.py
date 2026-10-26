"""I/B/E/S: one row per earnings announcement (ticker x date) with actual EPS and the pre-release consensus.

All tables are the unadjusted versions (actu_, statsumu_, detu_, actpsumu_), so actual and consensus are on the
same per-share basis on the announcement date. Consensus is only attached to quarterly actuals (fpi = '6').

Consensus (config ibes.consensus):
  detail  - built from analyst-level estimates (detu), as close as IBES allows to "right before the release":
            each analyst's latest estimate for the quarter that was
              * made before the release (earlier day, or the release day before the release time),
              * made or last confirmed (revdats) within ibes.detail_window_days of the release, and
              * not stopped by the broker before the release (stopu file).
  summary - the last monthly IBES consensus snapshot (statsumu) before the announcement date.

surprise = actual - consensus. The SUE (surprise / pre-announcement price) is computed in the final step, where the
t-1 closing price is known; the IBES price kept here (price_ibes) is the fallback.
"""
import pandas as pd

from .config import connect

ACT_SQL = """
    select ticker, pends, pdicity, anndats, anntims, value as actual, curr_act
    from ibes.actu_{f} where measure = 'EPS' and anndats between %(s)s and %(e)s
"""
SUMMARY_SQL = """
    select ticker, statpers, fpedats, numest, meanest, medest, stdev, curcode
    from ibes.statsumu_{f}
    where measure = 'EPS' and fiscalp = 'QTR' and fpi = '6' and statpers between %(s)s and %(e)s
"""
DETAIL_SQL = """
    with a as (select distinct ticker, pends, anndats, anntims from ibes.actu_{f}
               where measure = 'EPS' and pdicity = 'QTR' and anndats between %(s)s and %(e)s),
    d as (select distinct on (a.ticker, a.pends, a.anndats, coalesce(x.analys, -x.estimator))
                 a.ticker, a.pends, a.anndats, x.value, x.curr, x.anndats as est_date
          from a join ibes.detu_{f} x
            on x.ticker = a.ticker and x.fpedats = a.pends and x.measure = 'EPS' and x.fpi = '6'
           and (x.anndats < a.anndats or (x.anndats = a.anndats and x.anntims::time < a.anntims::time))
           and greatest(x.anndats, x.revdats) >= a.anndats - %(w)s
           and not exists (select 1 from ibes.stopu_{f} st
                           where st.ticker = x.ticker and st.estimator = x.estimator and st.fpedats = x.fpedats
                             and st.measure = 'EPS' and st.pdicity = 'Q'
                             and st.astpdats >= x.anndats and st.astpdats < a.anndats)
          order by a.ticker, a.pends, a.anndats, coalesce(x.analys, -x.estimator), x.anndats desc, x.anntims desc)
    select ticker, pends, anndats, count(*) as numest, avg(value) as meanest,
           percentile_cont(0.5) within group (order by value) as medest, stddev_samp(value) as stdev,
           max(est_date) as statpers, min(curr) as curcode, count(distinct curr) as n_curr
    from d group by 1, 2, 3
"""
PRICE_SQL = """
    select ticker, statpers, price, prdays, curr_price
    from ibes.actpsumu_{f} where measure = 'EPS' and statpers between %(s)s and %(e)s
"""
FILES = {'us': {'epsus': 'US'}, 'global': {'epsus': 'US', 'epsint': 'Intl'}}
P_RANK = {'QTR': 0, 'SAN': 1, 'ANN': 2}


def _pull(db, sql, files, params, date_cols):
    return pd.concat([db.raw_sql(sql.format(f=f), params=params, date_cols=date_cols).assign(ibes_file=label)
                      for f, label in files.items()], ignore_index=True)


def pull(cfg, target):
    c = cfg['ibes']
    files = FILES[cfg['universe']]
    lo, hi = cfg['match']['window_days']
    # announcements that can match a call in the period: anndats in [start - hi, end - lo] (plus slack)
    s = pd.Timestamp(cfg['period']['start']) - pd.Timedelta(days=hi + 5)
    e = pd.Timestamp(cfg['period']['end']) - pd.Timedelta(days=lo - 5)
    look = pd.Timedelta(days=200)  # summary snapshots can precede the release by a few months
    db = connect(cfg)
    try:
        act = _pull(db, ACT_SQL, files, {'s': s.date(), 'e': e.date()}, ['pends', 'anndats'])
        if c['consensus'] == 'summary':
            cons = _pull(db, SUMMARY_SQL, files, {'s': (s - look).date(), 'e': e.date()}, ['statpers', 'fpedats'])
        else:
            cons = _pull(db, DETAIL_SQL, files, {'s': s.date(), 'e': e.date(), 'w': c['detail_window_days']},
                         ['pends', 'anndats', 'statpers'])
            cons.loc[cons.n_curr > 1, 'curcode'] = 'MIXED'  # estimates in different currencies: unusable
            cons = cons.drop(columns='n_curr')
        px = _pull(db, PRICE_SQL, files, {'s': (s - pd.Timedelta(days=60)).date(), 'e': e.date()}, ['statpers', 'prdays'])
        latest = db.raw_sql('select max(anndats) as d from ibes.actu_epsus', date_cols=['d']).d.iloc[0]
    finally:
        db.close()

    # One row per announcement; quarterly record preferred (a Q4 release also carries the annual actual)
    act = act.sort_values('anndats').drop_duplicates(['ticker', 'pdicity', 'pends'], keep='last')
    act['p_rank'] = act.pdicity.map(P_RANK)
    ann = (act.sort_values(['ticker', 'anndats', 'p_rank', 'pends'], ascending=[True, True, True, False])
              .drop_duplicates(['ticker', 'anndats']).drop(columns='p_rank'))

    # Consensus for the announced quarter, formed before the release
    q = ann[ann.pdicity.eq('QTR')][['ticker', 'pends', 'anndats']]
    if c['consensus'] == 'summary':
        cs = q.merge(cons.drop(columns='ibes_file'), left_on=['ticker', 'pends'], right_on=['ticker', 'fpedats'])
        cs = cs[cs.statpers < cs.anndats].sort_values('statpers').drop_duplicates(['ticker', 'anndats'], keep='last')
        cs = cs.drop(columns='fpedats')
    else:
        cs = q.merge(cons.drop(columns='ibes_file'), on=['ticker', 'pends', 'anndats'])
    ann = ann.merge(cs.drop(columns='pends'), on=['ticker', 'anndats'], how='left')
    ann = ann.rename(columns={'statpers': 'consensus_date', 'numest': 'n_estimates', 'meanest': 'consensus_mean',
                              'medest': 'consensus_median', 'stdev': 'consensus_std'})
    # a consensus in another currency than the actual is not comparable
    ccy_bad = ann.curcode.notna() & ann.curcode.ne(ann.curr_act).fillna(True)
    ann.loc[ccy_bad, ['consensus_mean', 'consensus_median', 'consensus_std']] = pd.NA
    ann['consensus'] = ann[f"consensus_{c['consensus_stat']}"]
    ann['consensus_age_days'] = (ann.anndats - ann.consensus_date).dt.days
    ann['surprise'] = ann.actual - ann.consensus

    # Fallback price: last IBES price snapshot before the announcement (within 45 days), same currency as actual
    px = px.drop(columns='ibes_file').dropna(subset=['price']).drop_duplicates(['ticker', 'statpers'])
    ann = pd.merge_asof(ann.sort_values('anndats'), px.sort_values('statpers').rename(columns={'statpers': 'price_snapshot'}),
                        left_on='anndats', right_on='price_snapshot', by='ticker', allow_exact_matches=False,
                        direction='backward', tolerance=pd.Timedelta(days=45))
    ok = ann.curr_price.eq(ann.curr_act).fillna(False)
    ann['price_ibes'] = ann.price.where(ok)
    ann['price_ibes_date'] = ann.prdays.where(ok)
    ann = ann.drop(columns=['price', 'prdays', 'price_snapshot', 'curcode', 'curr_price']).sort_values(['anndats', 'ticker'])
    ann.to_parquet(target, index=False)

    qq = ann[ann.pdicity.eq('QTR')]
    print(f"IBES ({c['consensus']} consensus): {len(ann):,} announcements {s.date()}..{e.date()} ({len(qq):,} quarterly; "
          f'consensus for {qq.consensus.notna().mean():.1%}, median {qq.n_estimates.median():.0f} estimates, '
          f'median age {qq.consensus_age_days.median():.0f} days). Latest IBES announcement on WRDS: {latest.date()}')
    if latest < pd.Timestamp(cfg['period']['end']):
        print(f"WARNING: IBES on WRDS ends {latest.date()}, before period end {cfg['period']['end']}; later calls cannot match.")
