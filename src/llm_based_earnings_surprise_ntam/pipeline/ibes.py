"""I/B/E/S: one row per earnings announcement (ticker x date) with actual, pre-release consensus, price and SUE.

All tables are the unadjusted versions (actu_, statsumu_, detu_, actpsumu_), so actual, consensus and price are on
the same per-share basis on the announcement date.

    SUE = (actual - consensus) / price,   price = last IBES price before the announcement

SUE is only computed for quarterly actuals (the consensus is the current-quarter forecast, fpi = '6').
"""
import numpy as np
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
    with a as (select distinct ticker, pends, anndats from ibes.actu_{f}
               where measure = 'EPS' and pdicity = 'QTR' and anndats between %(s)s and %(e)s),
    d as (select distinct on (a.ticker, a.pends, a.anndats, x.analys)
                 a.ticker, a.pends, a.anndats, x.value, x.anndats as est_date
          from a join ibes.detu_{f} x
            on x.ticker = a.ticker and x.fpedats = a.pends and x.measure = 'EPS' and x.fpi = '6'
           and x.anndats < a.anndats and x.anndats >= a.anndats - %(w)s
          order by a.ticker, a.pends, a.anndats, x.analys, x.anndats desc, x.anntims desc)
    select ticker, pends, anndats, count(*) as numest, avg(value) as meanest,
           percentile_cont(0.5) within group (order by value) as medest, stddev_samp(value) as stdev,
           max(est_date) as statpers
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
    look = pd.Timedelta(days=max(c['detail_window_days'], 200))  # consensus and price come from before the release
    db = connect(cfg)
    try:
        act = _pull(db, ACT_SQL, files, {'s': s.date(), 'e': e.date()}, ['pends', 'anndats'])
        if c['consensus'] == 'summary':
            cons = _pull(db, SUMMARY_SQL, files, {'s': (s - look).date(), 'e': e.date()}, ['statpers', 'fpedats'])
        else:
            cons = _pull(db, DETAIL_SQL, files, {'s': s.date(), 'e': e.date(), 'w': c['detail_window_days']},
                         ['pends', 'anndats', 'statpers'])
        px = _pull(db, PRICE_SQL, files, {'s': (s - pd.Timedelta(days=60)).date(), 'e': e.date()}, ['statpers', 'prdays'])
        latest = db.raw_sql('select max(anndats) as d from ibes.actu_epsus', date_cols=['d']).d.iloc[0]
    finally:
        db.close()

    # One row per announcement; quarterly record preferred (a Q4 release also carries the annual actual)
    act = act.sort_values('anndats').drop_duplicates(['ticker', 'pdicity', 'pends'], keep='last')
    act['p_rank'] = act.pdicity.map(P_RANK)
    ann = (act.sort_values(['ticker', 'anndats', 'p_rank', 'pends'], ascending=[True, True, True, False])
              .drop_duplicates(['ticker', 'anndats']).drop(columns='p_rank'))

    # Consensus for the announced quarter, formed strictly before the announcement date
    q = ann[ann.pdicity.eq('QTR')][['ticker', 'pends', 'anndats']]
    if c['consensus'] == 'summary':
        cs = q.merge(cons.drop(columns='ibes_file'), left_on=['ticker', 'pends'], right_on=['ticker', 'fpedats'])
        cs = cs[cs.statpers < cs.anndats].sort_values('statpers').drop_duplicates(['ticker', 'anndats'], keep='last')
        cs = cs.drop(columns='fpedats')
    else:
        cs = q.merge(cons.drop(columns='ibes_file'), on=['ticker', 'pends', 'anndats']).assign(curcode=pd.NA)
    ann = ann.merge(cs.drop(columns='pends'), on=['ticker', 'anndats'], how='left')
    ann = ann.rename(columns={'statpers': 'consensus_date', 'numest': 'n_estimates', 'meanest': 'consensus_mean',
                              'medest': 'consensus_median', 'stdev': 'consensus_std'})
    ann['consensus'] = ann[f"consensus_{c['consensus_stat']}"]
    ann['consensus_age_days'] = (ann.anndats - ann.consensus_date).dt.days

    # Price: last IBES price snapshot before the announcement (within 45 days)
    px = px.drop(columns='ibes_file').dropna(subset=['price']).drop_duplicates(['ticker', 'statpers'])
    ann = pd.merge_asof(ann.sort_values('anndats'), px.sort_values('statpers').rename(columns={'statpers': 'price_snapshot'}),
                        left_on='anndats', right_on='price_snapshot', by='ticker', allow_exact_matches=False,
                        direction='backward', tolerance=pd.Timedelta(days=45))
    ann = ann.rename(columns={'prdays': 'price_date'}).drop(columns='price_snapshot')

    ccy_ok = ann.curr_act.eq(ann.curr_price).fillna(False) & (ann.curcode.isna() | ann.curcode.eq(ann.curr_act).fillna(False))
    ann['surprise'] = ann.actual - ann.consensus
    ok = ccy_ok & (ann.price >= c['min_price']) & ann.surprise.notna()
    ann['sue'] = np.where(ok, ann.surprise / ann.price, np.nan)
    ann = ann.drop(columns=['curcode', 'curr_price']).sort_values(['anndats', 'ticker'])
    ann.to_parquet(target, index=False)

    qq = ann[ann.pdicity.eq('QTR')]
    print(f'IBES: {len(ann):,} announcements {s.date()}..{e.date()} ({len(qq):,} quarterly; '
          f'consensus for {qq.consensus.notna().mean():.1%}, SUE for {qq.sue.notna().mean():.1%}). '
          f'Latest IBES announcement on WRDS: {latest.date()}')
    if latest < pd.Timestamp(cfg['period']['end']):
        print(f"WARNING: IBES on WRDS ends {latest.date()}, before period end {cfg['period']['end']}; later calls cannot match.")
