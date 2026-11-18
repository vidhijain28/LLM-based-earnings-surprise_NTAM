"""Download, untouched, the WRDS data the grid needs for the calls it trades (Duo approval needed).

    python -m llm_based_earnings_surprise_ntam.grid_analysis.pull [--config grid.yaml]

Writes to paths.raw (licensed data: keep it under the git-ignored data/):
    comp_secd.parquet       Compustat daily rows (comp.secd) of the calls' companies and of the market ETF,
                            from pull.prices_start to pull.prices_end, every security of each company
    comp_security.parquet   security headers (comp.security) of those companies
    comp_company.parquet    company headers (comp.company): SIC for the FF12 industries
    ibes_next.parquet       I/B/E/S quarterly EPS announcements (ibes.actu_epsus) of the calls' tickers from
                            pull.ibes_start on, for each call's next announcement
    pull_log.json           what was pulled, when, and how many rows
The columns are those of the team's Compustat panel (PIPELINE.md 7.2, scripts/extend_compustat_prices.py).
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..GEPA_Optimization.config import load
from ..pipeline.config import connect, load_config

DEFAULT = Path(__file__).with_name('grid.yaml')
SECD_COLS = ['gvkey', 'iid', 'datadate', 'curcdd', 'prccd', 'cshtrd', 'cshoc', 'ajexdi', 'trfd', 'div']
SECURITY_COLS = ['gvkey', 'iid', 'tic', 'cusip', 'tpci', 'exchg', 'secstat']
COMPANY_COLS = ['gvkey', 'conm', 'fic', 'sic']


def tradable_calls(cfg):
    """The calls of `calls.quarter` the grid trades: in sample and, if required, with a transcript by the close."""
    df = pd.read_parquet(cfg['paths']['calls'])
    df = df[df['in_sample'].astype(bool) & (df['event_quarter'] == cfg['calls']['quarter'])]
    if cfg['calls']['transcript_by_close']:
        df = df[df['transcript_by_event_close'].fillna(False).astype(bool)]
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(DEFAULT))
    args = ap.parse_args()
    cfg = load(args.config)
    p = cfg['pull']
    out = Path(cfg['paths']['raw'])
    out.mkdir(parents=True, exist_ok=True)

    calls = tradable_calls(cfg)
    gvkeys = sorted(calls['gvkey'].dropna().astype(str).unique())
    tickers = sorted(calls['ticker'].dropna().astype(str).unique())
    print(f'{len(calls)} calls, {len(gvkeys)} companies, {len(tickers)} I/B/E/S tickers', flush=True)

    print('connecting to WRDS (approve the Duo push)...', flush=True)
    db = connect(load_config())
    log = {'pulled_at': datetime.now().isoformat(timespec='seconds'), 'settings': p, 'files': {}}
    try:
        mkt = db.raw_sql('select gvkey, iid from comp.security where tic = %(t)s', params={'t': p['market_tic']})
        if len(mkt) != 1:
            raise SystemExit(f"expected one Compustat security with tic {p['market_tic']}, found {len(mkt)}")
        log['market'] = {'tic': p['market_tic'], 'gvkey': str(mkt.gvkey.iloc[0]), 'iid': str(mkt.iid.iloc[0])}
        keys = gvkeys + [str(mkt.gvkey.iloc[0])]

        jobs = {
            'comp_secd.parquet': (f"select {', '.join(SECD_COLS)} from comp.secd "
                                  "where gvkey = any(%(g)s) and datadate between %(s)s and %(e)s",
                                  {'g': keys, 's': p['prices_start'], 'e': p['prices_end']}),
            'comp_security.parquet': (f"select {', '.join(SECURITY_COLS)} from comp.security where gvkey = any(%(g)s)",
                                      {'g': keys}),
            'comp_company.parquet': (f"select {', '.join(COMPANY_COLS)} from comp.company where gvkey = any(%(g)s)",
                                     {'g': keys}),
            'ibes_next.parquet': ("select ticker, pends, pdicity, anndats, anntims from ibes.actu_epsus "
                                  "where measure = 'EPS' and pdicity = 'QTR' and ticker = any(%(t)s) "
                                  "and anndats >= %(s)s",
                                  {'t': tickers, 's': p['ibes_start']}),
        }
        for name, (sql, params) in jobs.items():
            t0 = datetime.now()
            df = db.raw_sql(sql, params=params)
            df.to_parquet(out / name, index=False)
            log['files'][name] = {'rows': len(df)}
            print(f'{name}: {len(df):,} rows, {(datetime.now() - t0).seconds}s', flush=True)
    finally:
        db.close()
    (out / 'pull_log.json').write_text(json.dumps(log, indent=2, default=str))


if __name__ == '__main__':
    main()
