"""Extend Hazel's cleaned Compustat price panel (PIPELINE.md section 7.2) past 2026-01-30, so the 12-week CARs
of section 6 cover every Q4 2025 event instead of only the early reporters.

Two steps, from the repo root. The data are licensed WRDS data, so every folder below is outside the repo:

    uv run python scripts/extend_compustat_prices.py pull ../data/raw/comp_secd
    uv run python scripts/extend_compustat_prices.py build ../data/processed ../data/raw/comp_secd \
        ../data/processed_extended

`pull` logs in to WRDS (Duo approval) and downloads, untouched: the Compustat daily security file (comp.secd,
all currencies) from 2025-12-01 to 2026-04-30, one parquet per month (months already on disk are skipped), and
the security and company headers (comp.security, comp.company) for the attribute columns.

`build` cleans the download with section 7.2's rules and writes a copy of Hazel's folder whose price panel and
VTI series run to the last downloaded day. Her rows are kept exactly as they are; only the days after her last
day are added.
    events_baseline.parquet   unchanged copy
    prices_clean.parquet      Hazel's rows, then the new days (same 32 columns and types)
    market_ret.parquet        VTI, extended
    cleaning_summary.json     Hazel's summary plus an `extension` block with the checks below
    README.txt                what changed

The rules: drop duplicate rows, rows with a missing or non-positive close, and non-USD rows; the trading calendar
is the days VTI traded. A return is close to close from the security's previous record, valid only if that
record is the previous trading day and its close is at least $0.01: the total return (dividend factor) where
the factor exists on both days, else the split-adjusted price return. Nothing is imputed. Helper columns use only
earlier records: prev_prccd and mcap_lag1 (the previous record), adv_lag1 (mean dollar volume of the previous 20
records), primary_pit (the company's security with the highest adv_lag1 that day; a company's only security is
always primary).

Checks, printed and saved in the summary:
  1. rules     the rules rebuild every derived column of Hazel's panel from its own base columns
  2. overlap   on the days both cover (Dec 2025 to Jan 2026), the new download gives Hazel's rows and returns
  3. truncation   rebuilding with every row after a cutoff removed changes nothing up to the cutoff, while a
               deliberately leaky column does change (so the test can see leakage)

A new day's return always uses two records of the new download, never one of Hazel's, so a split or dividend
restatement since her export cannot leak across the seam. Company and security attributes (tic, conm, sic, fic,
...) are snapshots: securities in Hazel's panel keep hers, so each security's attributes stay constant; the few
new securities get the headers downloaded here.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['gvkey', 'iid']
ROW_COLS = ['gvkey', 'iid', 'datadate', 'curcdd', 'prccd', 'cshtrd', 'cshoc', 'ajexdi', 'trfd', 'div']
SECURITY_COLS = ['gvkey', 'iid', 'tic', 'cusip', 'tpci', 'exchg', 'secstat']
COMPANY_COLS = ['gvkey', 'conm', 'cik', 'fic', 'sic', 'naics', 'ggroup']
ATTRS = ['tic', 'conm', 'cusip', 'cik', 'tpci', 'fic', 'exchg', 'secstat', 'sic', 'naics', 'ggroup']
HIST = KEYS + ['datadate', 'prccd', 'cshtrd', 'cshoc', 'ajexdi', 'trfd', 'div', 'curcdd']
FLOAT_DERIVED = ['prev_prccd', 'adj_px', 'tr_price', 'ret_tr', 'ret_px', 'ret', 'adv_lag1', 'mcap_lag1']
EXACT_DERIVED = ['ret_src', 'px_floor_t', 'primary_pit']
BASE_FLOATS = ['prccd', 'cshtrd', 'cshoc', 'ajexdi', 'trfd', 'div']


# --------------------------------------------------------------------------- pull

def pull(args):
    from llm_based_earnings_surprise_ntam.pipeline.config import connect, load_config

    out = Path(args.raw_dir)
    guard(out)
    out.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    todo = [m for m in pd.period_range(start, end, freq='M') if not (out / f'comp_secd_{m}.parquet').exists()]
    headers = {'comp_security.parquet': ('comp.security', SECURITY_COLS),
               'comp_company.parquet': ('comp.company', COMPANY_COLS)}
    todo_h = {f: v for f, v in headers.items() if not (out / f).exists()}
    if not todo and not todo_h:
        print(f'everything from {start.date()} to {end.date()} is already in {out}')
        return

    print('connecting to WRDS (approve the Duo push)...', flush=True)
    db = connect(load_config())
    have = db.raw_sql("select table_name, column_name from information_schema.columns "
                      "where table_schema = 'comp' and table_name in ('secd', 'security', 'company')")
    have = set(zip(have.table_name.str.lower(), have.column_name.str.lower()))
    need = [('secd', c) for c in ROW_COLS] + [('security', c) for c in SECURITY_COLS] + \
           [('company', c) for c in COMPANY_COLS]
    if not have:
        print('(the column list is not visible to this account; skipping the column check)')
    elif missing := [f'comp.{t}.{c}' for t, c in need if (t, c) not in have]:
        db.close()
        sys.exit(f'missing on WRDS: {missing}')

    logf = out / 'pull_log.json'
    log = json.loads(logf.read_text()) if logf.exists() else {'months': {}, 'headers': {}}
    for m in todo:
        lo, hi = max(m.start_time, start).date(), min(m.end_time.normalize(), end).date()
        t0 = datetime.now()
        df = db.raw_sql(f"select {', '.join(ROW_COLS)} from comp.secd where datadate between '{lo}' and '{hi}'")
        df.to_parquet(out / f'comp_secd_{m}.parquet', index=False)
        days = pd.to_datetime(df.datadate)
        log['months'][str(m)] = {'pulled_at': datetime.now().isoformat(timespec='seconds'), 'rows': len(df),
                                 'days': int(days.nunique()), 'first': str(days.min().date()),
                                 'last': str(days.max().date())}
        print(f'{m}: {len(df):,} rows, {days.nunique()} days ({days.min().date()} to {days.max().date()}), '
              f'{(datetime.now() - t0).seconds}s', flush=True)
    for f, (table, cols) in todo_h.items():
        df = db.raw_sql(f"select {', '.join(cols)} from {table}")
        df.to_parquet(out / f, index=False)
        log['headers'][f] = {'pulled_at': datetime.now().isoformat(timespec='seconds'), 'rows': len(df)}
        print(f'{table}: {len(df):,} rows', flush=True)
    db.close()
    logf.write_text(json.dumps(log, indent=2))


# --------------------------------------------------------------------------- build

def guard(path):
    """Licensed data may live outside the repo, or in its git-ignored data/ folder, nowhere else."""
    p, data = Path(path).resolve(), ROOT / 'data'
    if (p == ROOT or ROOT in p.parents) and not (p == data or data in p.parents):
        sys.exit(f'{p} is inside the public repo: licensed data must stay outside it (or under data/)')


def normalize(df):
    """WRDS types to the types of Hazel's panel."""
    df = df.copy()
    df['gvkey'] = pd.to_numeric(df.gvkey).astype('int64')
    if 'iid' in df:
        df['iid'] = df.iid.astype(str).str.strip()
    if 'datadate' in df:
        df['datadate'] = pd.to_datetime(df.datadate).astype('datetime64[us]')
    for c in [c for c in BASE_FLOATS + ['cik', 'sic', 'naics', 'ggroup'] if c in df]:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    if 'exchg' in df:
        df['exchg'] = pd.to_numeric(df.exchg, errors='coerce').astype('Int64').astype('string')
    for c in [c for c in ['curcdd', 'tic', 'conm', 'cusip', 'tpci', 'fic', 'secstat'] if c in df]:
        df[c] = df[c].astype('string').str.strip()
    return df


def derive(df, cal):
    """Section 7.2's derived columns, each from the security's earlier records only. Keeps every input column."""
    d = df.sort_values(['gvkey', 'iid', 'datadate'], kind='mergesort').reset_index(drop=True)
    t = pd.Series(pd.Series(np.arange(len(cal)), index=cal).reindex(d.datadate).to_numpy(float), index=d.index)
    if t.isna().any():
        raise ValueError('rows off the trading calendar')
    same = d.gvkey.eq(d.gvkey.shift(1)) & d.iid.eq(d.iid.shift(1))

    def prev(s):
        return s.shift(1).where(same)

    d['prev_prccd'] = prev(d.prccd)
    d['adj_px'] = d.prccd / d.ajexdi
    d['tr_price'] = d.adj_px * d.trfd
    valid = prev(t).eq(t - 1) & d.prev_prccd.ge(0.01)
    d['ret_tr'] = (d.tr_price / prev(d.tr_price) - 1).where(valid)
    d['ret_px'] = (d.adj_px / prev(d.adj_px) - 1).where(valid)
    d['ret'] = d.ret_tr.where(d.ret_tr.notna(), d.ret_px)
    d['ret_src'] = np.select([d.ret_tr.notna(), d.ret_px.notna()], ['total_return', 'price_only'], 'invalid')
    d['ret_src'] = d.ret_src.astype(object)
    d['px_floor_t'] = d.prccd.lt(0.01)
    d['mcap_lag1'] = d.prev_prccd * prev(d.cshoc)
    dv = prev(d.prccd * d.cshtrd)
    adv = dv.groupby([d.gvkey, d.iid], sort=False).rolling(20, min_periods=1).mean()
    d['adv_lag1'] = adv.reset_index(level=[0, 1], drop=True)
    rank = d.groupby(['gvkey', 'datadate']).adv_lag1.rank(method='first', ascending=False)
    only = d.groupby(['gvkey', 'datadate']).iid.transform('size').eq(1)
    d['primary_pit'] = (rank.eq(1) | only).astype('boolean')
    return d


def same_days(a, b):
    """Two calendars hold the same days (whatever their datetime units)."""
    return pd.DatetimeIndex(a).as_unit('ns').equals(pd.DatetimeIndex(b).as_unit('ns'))


def match(a, b, exact=False):
    """Share of rows where two columns agree (NaN equals NaN)."""
    if exact:
        a, b = a.astype(object).where(a.notna(), None), b.astype(object).where(b.notna(), None)
        return float(np.mean([x == y for x, y in zip(a, b)]))
    return float(np.isclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True).mean())


def compare(mine, theirs, floats, exact):
    m = mine.merge(theirs, on=KEYS + ['datadate'], how='outer', suffixes=('', '_h'), indicator=True)
    both = m[m._merge.eq('both')]
    out = {'rows_both': len(both), 'rows_only_new': int(m._merge.eq('left_only').sum()),
           'rows_only_hazel': int(m._merge.eq('right_only').sum())}
    out.update({c: match(both[c], both[c + '_h']) for c in floats})
    out.update({c: match(both[c], both[c + '_h'], exact=True) for c in exact})
    return out, both


def build(args):
    hz_dir, raw_dir, out = Path(args.hazel_dir), Path(args.raw_dir), Path(args.out_dir)
    guard(out)
    print('reading...', flush=True)
    hz = pd.read_parquet(hz_dir / 'prices_clean.parquet')
    hz_mk = pd.read_parquet(hz_dir / 'market_ret.parquet')
    files = sorted(raw_dir.glob('comp_secd_*.parquet'))
    if not files:
        sys.exit(f'no comp_secd_*.parquet in {raw_dir}: run `pull` first')
    raw = normalize(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))
    head = normalize(pd.read_parquet(raw_dir / 'comp_security.parquet')).merge(
        normalize(pd.read_parquet(raw_dir / 'comp_company.parquet')), on='gvkey', how='left')[KEYS + ATTRS]
    head = head.drop_duplicates(KEYS)

    # 1. Hazel's cleaning steps, in her order
    n_raw = len(raw)
    dup = raw.duplicated(KEYS + ['datadate'])
    raw = raw[~dup]
    bad = raw.prccd.isna() | raw.prccd.le(0)
    raw = raw[~bad]
    non_usd = ~raw.curcdd.eq('USD').fillna(False)
    clean = raw[~non_usd].copy()
    drops = {'dropped_duplicates': int(dup.sum()), 'dropped_bad_price': int(bad.sum()),
             'dropped_non_usd': int(non_usd.sum())}

    # 2. calendar: Hazel's days, then the days VTI traded after her last day
    vti = hz.loc[hz.tic.eq('VTI'), KEYS].drop_duplicates()
    if len(vti) != 1:
        sys.exit("can't find VTI in Hazel's panel")
    vg, vi = vti.iloc[0]
    seam, first = hz.datadate.max(), clean.datadate.min()
    cal_h = pd.DatetimeIndex(sorted(pd.to_datetime(hz_mk.datadate)))
    vti_days = pd.DatetimeIndex(sorted(clean.loc[clean.gvkey.eq(vg) & clean.iid.eq(vi), 'datadate']))
    calendar_agrees = same_days(vti_days[vti_days <= seam], cal_h[cal_h >= first])
    cal = cal_h.union(vti_days[vti_days > seam])
    off = ~clean.datadate.isin(cal)
    drops['dropped_not_vti_day'] = int(off.sum())
    clean = clean[~off]

    # 3. derived columns on one history: Hazel's rows before the download starts, then the download
    print('deriving...', flush=True)
    hist = pd.concat([hz.loc[hz.datadate < first, HIST].assign(src='hazel'), clean[HIST].assign(src='new')],
                     ignore_index=True)
    d = derive(hist, cal)
    mine = d[d.src.eq('new')]
    new = mine[mine.datadate > seam].copy()

    # check 1: the rules rebuild Hazel's derived columns
    print('check 1: rules on her panel...', flush=True)
    rb = derive(hz[HIST], cal_h)
    rules, _ = compare(rb[KEYS + ['datadate'] + FLOAT_DERIVED + EXACT_DERIVED],
                       hz[KEYS + ['datadate'] + FLOAT_DERIVED + EXACT_DERIVED], FLOAT_DERIVED, EXACT_DERIVED)
    del rb

    # check 2: the overlap. The first downloaded day is left out: its previous record is one of Hazel's.
    print('check 2: overlap...', flush=True)
    cols = KEYS + ['datadate'] + BASE_FLOATS + FLOAT_DERIVED + EXACT_DERIVED
    ovl_days = cal[(cal > first) & (cal <= seam)]
    overlap, both = compare(mine.loc[mine.datadate.isin(ovl_days), cols], hz.loc[hz.datadate.isin(ovl_days), cols],
                            BASE_FLOATS + FLOAT_DERIVED, EXACT_DERIVED)
    diff = (both.ret - both.ret_h).abs()
    overlap.update({
        'days': f'{ovl_days.min().date()} to {ovl_days.max().date()} ({len(ovl_days)} trading days)',
        'calendar_agrees': bool(calendar_agrees),
        'ret_rows_off_by_more_than_1e-6': int((diff > 1e-6).sum() + (both.ret.isna() != both.ret_h.isna()).sum()),
        'ret_max_abs_diff': float(diff.max()),
        'vti_ret': match(both.loc[both.gvkey.eq(vg) & both.iid.eq(vi), 'ret'],
                         both.loc[both.gvkey.eq(vg) & both.iid.eq(vi), 'ret_h']),
    })
    a = hz.drop_duplicates(KEYS)[KEYS + ATTRS].merge(head, on=KEYS, suffixes=('', '_h'))
    overlap['attributes_same_as_headers'] = {c: match(a[c], a[c + '_h'], exact=True) for c in ATTRS}

    # check 3: truncation (look-ahead) test at the middle new day
    print('check 3: truncation...', flush=True)
    new_days = cal[cal > seam]
    cut = new_days[len(new_days) // 2]
    tr = derive(hist[hist.datadate <= cut], cal[cal <= cut])
    keep = (d.datadate > seam) & (d.datadate <= cut)
    full, part = d[keep].reset_index(drop=True), tr[tr.datadate > seam].reset_index(drop=True)
    same_rows = full[KEYS + ['datadate']].equals(part[KEYS + ['datadate']])
    mism = {c: int(round((1 - match(full[c], part[c], exact=c in EXACT_DERIVED)) * len(full)))
            for c in FLOAT_DERIVED + EXACT_DERIVED}

    def leak(x):  # next record's close: uses the future, so the test must catch it
        nxt = x.gvkey.eq(x.gvkey.shift(-1)) & x.iid.eq(x.iid.shift(-1))
        return x.prccd.shift(-1).where(nxt)

    control = int(round((1 - match(leak(d)[keep].reset_index(drop=True),
                                   leak(tr)[tr.datadate > seam].reset_index(drop=True))) * len(full)))
    truncation = {'cutoff': str(cut.date()), 'rows': len(full), 'same_rows': bool(same_rows), 'mismatches': mism,
                  'negative_control': control, 'passed': bool(same_rows and not any(mism.values()) and control > 0)}
    del tr

    # 4. assemble: attributes from Hazel's panel, else the downloaded headers
    attrs = hz.drop_duplicates(KEYS)[KEYS + ATTRS]
    known = new.set_index(KEYS).index.isin(attrs.set_index(KEYS).index)
    n_unknown = int(new.loc[~known, KEYS].drop_duplicates().shape[0])
    new = pd.concat([new[known].merge(attrs, on=KEYS, how='left'), new[~known].merge(head, on=KEYS, how='left')])
    new = new.sort_values(KEYS + ['datadate'], kind='mergesort')[hz.columns].astype(hz.dtypes.to_dict())
    ext = pd.concat([hz, new], ignore_index=True)
    if ext.duplicated(KEYS + ['datadate']).any():
        sys.exit('duplicate rows after the merge')
    v = new.loc[new.gvkey.eq(vg) & new.iid.eq(vi), ['datadate', 'ret']].rename(columns={'ret': 'mkt_ret'})
    mk = pd.concat([hz_mk, v.assign(mkt_source='VTI').astype(hz_mk.dtypes.to_dict())], ignore_index=True)
    if not same_days(mk.datadate, cal) or mk.mkt_ret.iloc[1:].isna().any():
        sys.exit('the VTI series does not cover the calendar')

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hz_dir / 'events_baseline.parquet', out / 'events_baseline.parquet')
    ext.to_parquet(out / 'prices_clean.parquet', index=False)
    mk.to_parquet(out / 'market_ret.parquet', index=False)

    q = new.ret.quantile([0.001, 0.01, 0.5, 0.99, 0.999])
    ext_summary = {
        'source': 'WRDS comp.secd (+ comp.security, comp.company headers)', 'raw_files': [f.name for f in files],
        'pull_log': json.loads((raw_dir / 'pull_log.json').read_text()) if (raw_dir / 'pull_log.json').exists()
        else None,
        'download': {'rows': n_raw, 'first': str(first.date()), 'last': str(clean.datadate.max().date())},
        'drops': drops,
        'seam': str(seam.date()),
        'added': {'rows': len(new), 'trading_days': len(new_days), 'first': str(new_days.min().date()),
                  'last': str(new_days.max().date()), 'securities': int(new.groupby(KEYS).ngroups),
                  'securities_not_in_hazel_panel': n_unknown,
                  'ret_src': new.ret_src.value_counts().to_dict(),
                  'abs_ret_gt_50pct': int(new.ret.abs().gt(0.5).sum()),
                  'ret_quantiles': {f'{100 * k:g}%': round(float(x), 4) for k, x in q.items()}},
        'panel': {'rows': len(ext), 'date_min': str(ext.datadate.min().date()),
                  'date_max': str(ext.datadate.max().date()), 'trading_days': len(cal)},
        'check_rules_rebuild_hazel_panel': rules,
        'check_overlap_with_hazel_export': overlap,
        'check_truncation': truncation,
    }
    summary = json.loads((hz_dir / 'cleaning_summary.json').read_text())
    summary['extension'] = ext_summary
    (out / 'cleaning_summary.json').write_text(json.dumps(summary, indent=2, default=str))
    (out / 'README.txt').write_text(README.format(seam=seam.date(), last=new_days.max().date(), first=first.date(),
                                                  days=len(cal), rows=len(ext)))
    print(json.dumps(ext_summary, indent=2, default=str))
    print(f'\nwrote {out}')


README = """Hazel's cleaned files (see ../processed/README.txt), with the price panel and the VTI series extended past
{seam} to {last} ({days} trading days, {rows:,} rows) by scripts/extend_compustat_prices.py in the team repo.

Hazel's rows are unchanged. The days after {seam} come from a new WRDS download of the Compustat daily security
file (comp.secd, from {first}) cleaned with the same rules (PIPELINE.md section 7.2). events_baseline.parquet
is an unchanged copy. cleaning_summary.json is Hazel's summary plus an "extension" block with the checks: the
rules rebuild her derived columns, the new download matches her export where they overlap, and a truncation
(look-ahead) test.

Licensed WRDS data, covered by the project NDA: share inside the team only, never commit.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pull', help='download comp.secd and its headers from WRDS')
    p.add_argument('raw_dir')
    p.add_argument('--start', default='2025-12-01')
    p.add_argument('--end', default='2026-04-30')
    b = sub.add_parser('build', help="clean the download and extend Hazel's folder")
    b.add_argument('hazel_dir', help="folder with Hazel's events_baseline / prices_clean / market_ret parquet")
    b.add_argument('raw_dir', help='folder written by `pull`')
    b.add_argument('out_dir', help='where to write the extended copy (outside the repo)')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    pull(args) if args.cmd == 'pull' else build(args)


if __name__ == '__main__':
    main()
