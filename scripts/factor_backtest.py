"""Factor backtest of the earnings signals (SUE and the LLM score) in calendar time, as NTAM asked for it.

The event study of PIPELINE.md section 6 asks how far a stock drifts after its announcement. This asks how a
portfolio that trades the signal would have done, day by day, and how fast the signal decays when the portfolio
is only rebalanced monthly. It follows the standard CRSP-style protocol that NTAM asked for, with Compustat
stand-ins where CRSP (which ends on 2025-12-31 on WRDS) can't cover the Q4 2025 holding periods:

    universe      US common stocks (CRSP shrcd 10/11 -> Compustat tpci "0" and fic "USA"), primary issue, on
                  NYSE, AMEX or Nasdaq (exchcd 1/2/3 -> Compustat exchg 11/12/14)
    breakpoints   NYSE stocks only, so microcaps don't drive the sorts, from announcements made so far
    weights       market cap capped at the NYSE 80th percentile (headline; Jensen, Kelly and Pedersen 2023), with
                  plain value weights and equal weights as checks; a leg trades only with >= 30 names (--min-names)
    timing        calls are used only if the transcript existed by the day-0 close, the entry (--all-calls: keep all)
    delistings    Shumway: a stock that stops trading for a performance reason gets -30% (NYSE/AMEX) or -55%
                  (Nasdaq) on its first missing day
    tests         Newey-West t-statistics, per-event spreads clustered by event date (also by reporting week),
                  Fama-MacBeth slopes, alphas on Fama-French 5 + momentum, t > 3 flag

Holding schemes: enter the day after the announcement and hold days 1-5, 6-20, 21-60 (plus 1-20, 1-60); or
rebalance monthly from each month's announcers and hold month 1, 2 or 3 (and months 1-3 overlapping), also
conditioned on 12-1 momentum (momentum-neutral and by tercile).

Two steps, from the repo root (the data are licensed, so the data folders stay outside the repo):

    uv run python scripts/factor_backtest.py pull ../data/raw/factor_inputs
    uv run python scripts/factor_backtest.py run ../data/processed_extended ../data/raw/factor_inputs <out folder> \
        --pipeline data/processed/earnings_calls_ibes_20251001_20251231_us.parquet \
        --llm data/processed/llm_scores.parquet --llm-qwen data/processed/llm_merged_20251001_20251231_us.parquet

`pull` downloads the Fama-French daily factors (public, Ken French's library) and, from WRDS (Duo approval),
Compustat daily prices from 2024-10 to 2025-11 for 12-1 momentum plus each security's deletion date and reason
for the delisting adjustment. `run` writes factor_summary.csv (every portfolio test), per_event_spreads.csv,
per_event_by_week.csv, fama_macbeth.csv, factor_daily_returns.csv, run_info.json and the charts.
"""
import argparse
import io
import json
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FF_URLS = {
    'ff5': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip',
    'mom': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip',
}
SECD_COLS = ['gvkey', 'iid', 'datadate', 'curcdd', 'prccd', 'ajexdi', 'trfd']


def guard(path):
    """Licensed data may live outside the repo, or in its git-ignored data/ folder, nowhere else."""
    p, data = Path(path).resolve(), ROOT / 'data'
    if (p == ROOT or ROOT in p.parents) and not (p == data or data in p.parents):
        sys.exit(f'{p} is inside the public repo: licensed data must stay outside it (or under data/)')


# --------------------------------------------------------------------------- pull

def french_daily(url):
    """One of Ken French's daily CSV zips -> DataFrame indexed by date, in decimals."""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research script)'})
    with urllib.request.urlopen(req, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode('latin-1').splitlines()
    head = next(i for i, ln in enumerate(text) if ln.strip() and ln.split(',')[0].strip() == ''
                and len(ln.split(',')) > 1)
    cols = [c.strip().lower().replace('-', '') for c in text[head].split(',')[1:]]
    rows = [ln.split(',') for ln in text[head + 1:] if ln[:8].strip().isdigit() and len(ln.split(',')) > 1]
    df = pd.DataFrame([[float(x) for x in r[1:]] for r in rows], columns=cols,
                      index=pd.to_datetime([r[0].strip() for r in rows], format='%Y%m%d'))
    return df / 100


def pull(args):
    out = Path(args.raw_dir)
    guard(out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = out / 'pull_log.json'
    log = json.loads(log_f.read_text()) if log_f.exists() else {}

    ff_f = out / 'ff_daily.csv'
    if not ff_f.exists():
        ff = french_daily(FF_URLS['ff5']).join(french_daily(FF_URLS['mom']), how='inner')
        ff = ff.rename(columns={'mom': 'umd'})
        ff.index.name = 'date'
        ff.to_csv(ff_f)
        log['ff_daily'] = {'pulled_at': datetime.now().isoformat(timespec='seconds'), 'rows': len(ff),
                           'first': str(ff.index.min().date()), 'last': str(ff.index.max().date()),
                           'columns': list(ff.columns), 'source': FF_URLS}
        print(f'Fama-French daily: {len(ff):,} days, {ff.index.min().date()} to {ff.index.max().date()}, '
              f'columns {list(ff.columns)}', flush=True)

    months = [m for m in pd.period_range(args.start, args.end, freq='M')
              if not (out / f'comp_secd_{m}.parquet').exists()]
    need_del = not (out / 'comp_deletions.parquet').exists()
    if months or need_del:
        from llm_based_earnings_surprise_ntam.pipeline.config import connect, load_config
        print('connecting to WRDS (approve the Duo push)...', flush=True)
        db = connect(load_config())
        for m in months:
            lo, hi = m.start_time.date(), m.end_time.date()
            df = db.raw_sql(f"select {', '.join(SECD_COLS)} from comp.secd "
                            f"where datadate between '{lo}' and '{hi}'")
            df.to_parquet(out / f'comp_secd_{m}.parquet', index=False)
            log.setdefault('comp_secd', {})[str(m)] = {'pulled_at': datetime.now().isoformat(timespec='seconds'),
                                                      'rows': len(df)}
            print(f'comp.secd {m}: {len(df):,} rows', flush=True)
        if need_del:
            sec = db.raw_sql('select gvkey, iid, dldtei, dlrsni, secstat, exchg from comp.security')
            co = db.raw_sql('select gvkey, dldte, dlrsn from comp.company')
            sec.merge(co, on='gvkey', how='left').to_parquet(out / 'comp_deletions.parquet', index=False)
            log['comp_deletions'] = {'pulled_at': datetime.now().isoformat(timespec='seconds'), 'rows': len(sec)}
            print(f'comp.security + comp.company deletion fields: {len(sec):,} securities', flush=True)
        db.close()
    log_f.write_text(json.dumps(log, indent=2, default=str))
    print(f'done: {out}')


# --------------------------------------------------------------------------- run

EXCHANGES = {'11': 'NYSE', '12': 'AMEX', '14': 'Nasdaq'}
PERF_DELIST = {'02', '03'}  # Compustat deletion reasons treated as performance-related: bankruptcy, liquidation
SHUMWAY = {'11': -0.30, '12': -0.30, '14': -0.55}
WINDOWS = {'days 1-5': (1, 5), 'days 6-20': (6, 20), 'days 21-60': (21, 60), 'days 1-20': (1, 20),
           'days 1-60': (1, 60)}
FACTORS = ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd']
NW_LAGS = 5
MIN_NAMES = 30  # a leg with fewer names on a day is not traded that day (--min-names)
MIN_NYSE = 50   # point-in-time breakpoints need this many NYSE announcements so far
RNG = np.random.default_rng(0)


class Panel:
    """Daily returns and market caps of the extended Compustat panel on VTI's trading calendar."""

    def __init__(self, data_dir, raw_dir, shumway=True):
        data_dir = Path(data_dir)
        px = pd.read_parquet(data_dir / 'prices_clean.parquet',
                             columns=['gvkey', 'iid', 'datadate', 'ret', 'mcap_lag1', 'primary_pit', 'tpci', 'fic',
                                      'exchg'])
        mk = pd.read_parquet(data_dir / 'market_ret.parquet')
        mk['datadate'] = pd.to_datetime(mk.datadate).astype('datetime64[ns]')
        self.cal = pd.DatetimeIndex(mk.datadate.sort_values())
        self.market = mk.set_index('datadate').mkt_ret.reindex(self.cal).fillna(0.0).to_numpy()
        px['key'] = px.gvkey.astype(str) + '|' + px.iid.astype(str)
        px['datadate'] = pd.to_datetime(px.datadate).astype('datetime64[ns]')
        self.keys = pd.Index(np.sort(px.key.unique()))
        r, c = self.keys.get_indexer(px.key), self.cal.get_indexer(px.datadate)
        shape = (len(self.keys), len(self.cal))
        self.R = np.full(shape, np.nan)
        self.R[r, c] = px.ret.to_numpy(float)
        M = np.full(shape, np.nan)
        M[r, c] = px.mcap_lag1.to_numpy(float)
        self.M = pd.DataFrame(M).ffill(axis=1).to_numpy()  # last known market cap (close x shares, day before)
        have = np.zeros(shape, bool)
        have[r, c] = True
        self.last = np.where(have.any(axis=1), shape[1] - 1 - np.argmax(have[:, ::-1], axis=1), -1)
        a = px.drop_duplicates('key').set_index('key').reindex(self.keys)
        self.exchg = a.exchg.astype('string').fillna('').to_numpy(str)
        self.universe = (a.tpci.eq('0') & a.fic.eq('USA') & a.exchg.isin(list(EXCHANGES))).fillna(False).to_numpy(bool)
        self.primary = px.loc[px.primary_pit.fillna(False).astype(bool), ['key', 'gvkey', 'datadate']]
        # capped value weights (Jensen, Kelly and Pedersen 2023): a stock's weight is its market cap, capped at the
        # 80th percentile of NYSE universe stocks that day, so one mega-cap can't be most of a leg
        nyse = self.universe & (self.exchg == '11')
        self.cap80 = np.nanpercentile(np.where(nyse[:, None], self.M, np.nan), 80, axis=0)
        # a missing day earns the market (no trade, bad print, or after the stock stops trading)
        self.F = np.where(np.isnan(self.R), self.market[None, :], self.R)
        self.shumway = None
        self.perf = np.zeros(shape[0], bool)
        if shumway:
            perf = performance_delisted(raw_dir, self.keys)
            if perf is not None:
                hit = perf & (self.last >= 0) & (self.last < shape[1] - 1)
                for i in np.where(hit)[0]:
                    self.F[i, self.last[i] + 1] = SHUMWAY.get(self.exchg[i], -0.30)
                self.shumway = int(hit.sum())
                self.perf = hit


def performance_delisted(raw_dir, keys):
    """Securities Compustat deleted for bankruptcy or liquidation (security level, else company level)."""
    f = Path(raw_dir) / 'comp_deletions.parquet'
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    d['key'] = pd.to_numeric(d.gvkey).astype('int64').astype(str) + '|' + d.iid.astype(str).str.strip()
    reason = d.dlrsni.where(d.dlrsni.notna(), d.dlrsn).astype('string').str.strip().str.zfill(2)
    when = pd.to_datetime(d.dldtei.where(d.dldtei.notna(), d.dldte), errors='coerce')
    return keys.isin(d.loc[reason.isin(PERF_DELIST) & (when >= '2025-09-01'), 'key'])


def momentum(raw_dir):
    """12-1 momentum by security and formation month m: total-return index at the end of m-1 over m-12, minus 1.
    Both ends come from the same download, so restated split factors cancel."""
    files = sorted(Path(raw_dir).glob('comp_secd_*.parquet'))
    if not files:
        return None
    parts = []
    for f in files:
        df = pd.read_parquet(f)
        for c in ('prccd', 'ajexdi', 'trfd'):
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
        df = df[df.curcdd.astype(str).eq('USD') & df.prccd.gt(0) & df.ajexdi.gt(0)].copy()
        df['key'] = pd.to_numeric(df.gvkey).astype('int64').astype(str) + '|' + df.iid.astype(str).str.strip()
        df['datadate'] = pd.to_datetime(df.datadate)
        df['tri'] = df.prccd / df.ajexdi * df.trfd.fillna(1.0)
        df = df.sort_values('datadate').groupby('key').tail(1)
        parts.append(df.assign(month=df.datadate.dt.to_period('M'))[['key', 'month', 'tri']])
    tri = pd.concat(parts).set_index(['key', 'month']).tri.unstack('month')
    out = {m: tri[m - 1] / tri[m - 12] - 1 for m in pd.period_range('2025-10', '2025-12', freq='M')
           if (m - 1) in tri.columns and (m - 12) in tri.columns}
    return pd.DataFrame(out) if out else None


def jitter(x):
    """Random tie-breaking (fixed seed) for scores with few distinct values, far below their resolution."""
    x = np.asarray(x, float)
    scale = np.nanstd(x) if np.nanstd(x) > 0 else 1.0
    return x + RNG.uniform(0, 1e-7, len(x)) * scale


def load_events(panel, data_dir, pipeline, llm, llm_qwen, mom, by_close=True):
    """Events per sample, keyed to the primary Compustat issue on the event date, inside the protocol universe."""
    prim = panel.primary.rename(columns={'datadate': 'event_date'})
    ev = pd.read_parquet(Path(data_dir) / 'events_baseline.parquet',
                         columns=['gvkey', 'iid', 'event_date', 'surprise_pct', 'ibes_ticker', 'announce_date'])
    ev['key'] = ev.gvkey.astype(str) + '|' + ev.iid.astype(str)
    ev['event_date'] = pd.to_datetime(ev.event_date).astype('datetime64[ns]')
    ev = ev.merge(prim[['key', 'event_date']], on=['key', 'event_date']).rename(
        columns={'surprise_pct': 'sue', 'announce_date': 'ann_date'})
    # the I/B/E/S ticker and announcement date only serve to find each company's next announcement (decay step)
    samples = {'I/B/E/S events': ev[['key', 'event_date', 'sue', 'ibes_ticker', 'ann_date']]}
    if pipeline:
        d = pd.read_parquet(pipeline, columns=['keydevid', 'gvkey', 'event_date', 'sue', 'in_sample',
                                               'transcript_by_event_close', 'ticker', 'anndats'])
        d = d.rename(columns={'ticker': 'ibes_ticker', 'anndats': 'ann_date'})
        d = d[d.in_sample.fillna(False).astype(bool)].copy()
        if by_close:  # the LLM can only score a call whose transcript exists by the entry (day-0 close)
            d = d[d.transcript_by_event_close.fillna(False).astype(bool)].copy()
        d = d.drop(columns='transcript_by_event_close')
        d['keydevid'] = pd.to_numeric(d.keydevid).astype('int64')
        d['gvkey'] = pd.to_numeric(d.gvkey).astype('int64')
        d['event_date'] = pd.to_datetime(d.event_date).astype('datetime64[ns]')
        d = d.merge(prim, on=['gvkey', 'event_date']).drop_duplicates('keydevid')
        if llm:
            s = pd.read_parquet(llm)
            own = s[s.summary_model.eq(s.model)]  # scored its own summaries: the best run in Austin's notebook
            d = d.merge(own[['keydevid', 'prediction']].rename(columns={'prediction': 'llm_gemini'}),
                        on='keydevid', how='left')
        if llm_qwen:
            q = pd.read_parquet(llm_qwen, columns=['keydevid', 'prediction'])
            q['keydevid'] = pd.to_numeric(q.keydevid).astype('int64')
            d = d.merge(q.drop_duplicates('keydevid').rename(columns={'prediction': 'llm_qwen'}), on='keydevid',
                        how='left')
        samples['pipeline calls'] = d.drop(columns=['gvkey', 'in_sample'])
    out = {}
    for name, d in samples.items():
        d = d.copy()
        d['i'] = panel.keys.get_indexer(d.key)
        d['pos0'] = panel.cal.get_indexer(d.event_date)
        d = d[(d.i >= 0) & (d.pos0 >= 0)]
        d = d[panel.universe[d.i.to_numpy()]].copy()
        d['nyse'] = panel.exchg[d.i.to_numpy()] == '11'
        d['month'] = d.event_date.dt.to_period('M')
        if mom is not None:
            stacked = mom.stack()
            d['mom'] = stacked.reindex(pd.MultiIndex.from_arrays([d.key, d.month])).to_numpy()
        else:
            d['mom'] = np.nan
        for c in ('sue', 'llm_gemini', 'llm_qwen', 'mom'):
            if c in d:
                d[c + '_j'] = jitter(d[c])
        out[name] = d.sort_values(['event_date', 'key']).reset_index(drop=True)
    return out


def bins(x, ref, n):
    """Quantile 1..n of x, with breakpoints from the reference values (the NYSE stocks)."""
    ref = ref[~np.isnan(ref)]
    if len(ref) < 2 * n:
        return np.full(len(x), np.nan)
    bp = np.quantile(ref, np.arange(1, n) / n)
    return np.where(np.isnan(x), np.nan, np.searchsorted(bp, x, side='right') + 1.0)


def values(g, signal):
    """The signal for a formation set. 'llm_resid' is the LLM score's rank net of the SUE rank (OLS within g),
    the part of the LLM score that the surprise doesn't explain."""
    if signal != 'llm_resid':
        return g[signal + '_j'].to_numpy()
    ok = (g.llm_gemini.notna() & g.sue.notna()).to_numpy()
    out = np.full(len(g), np.nan)
    if ok.sum() < 30:
        return out
    rl, rs = g.llm_gemini_j.rank(pct=True).to_numpy(), g.sue_j.rank(pct=True).to_numpy()
    b = np.polyfit(rs[ok], rl[ok], 1)
    out[ok] = rl[ok] - np.polyval(b, rs[ok])
    return out


def split_legs(g, x, cond, n=5):
    """(long index, short index) pairs: top and bottom quintile on NYSE breakpoints, within momentum terciles if
    cond is 'neutral' (one pair per tercile) or 1/2/3 (that tercile only)."""
    nyse = g.nyse.to_numpy()
    if cond is None:
        parts = [np.ones(len(g), bool)]
    else:
        m = g.mom_j.to_numpy()
        t = bins(m, m[nyse], 3)
        parts = [t == j for j in (1, 2, 3)] if cond == 'neutral' else [t == cond]
    pairs = []
    for p in parts:
        q = np.full(len(g), np.nan)
        q[p] = bins(x[p], x[p & nyse], n)
        pairs.append((np.where(q == n)[0], np.where(q == 1)[0]))
    return pairs


def monthly_books(d, signal, panel, k, cond=None):
    """Form at each month-end from that month's announcers; hold through month m+k (buy and hold in the month)."""
    per = panel.cal.to_period('M')
    books, lags = [], []
    for m, g in d.groupby('month'):
        held = np.where(per == m + k)[0]
        formed = np.where(per == m)[0]
        if not len(held) or not len(formed):
            continue
        x = values(g, signal)
        keep = ~np.isnan(x)
        g, x = g[keep], x[keep]
        if len(g) < 50:
            continue
        a, b = held[0], held[-1]
        idx = g.i.to_numpy()
        for lo, sh in split_legs(g, x, cond):
            books.append(([(i, a, b) for i in idx[lo]], [(i, a, b) for i in idx[sh]]))
        lags += list(formed[-1] - g.pos0.to_numpy())
    return books, lags


def event_books(d, signal, panel, a, b, cond=None):
    """Enter each stock at the close of day a-1 after its event and hold through day b; fifths from NYSE
    breakpoints over the announcements made so far (what a live portfolio would know)."""
    longs, shorts = [], []
    dates = d.event_date.to_numpy()
    for day in np.unique(dates):
        past = d[dates <= day]
        x = values(past, signal)
        ok = ~np.isnan(x)
        if (ok & past.nyse.to_numpy()).sum() < MIN_NYSE:
            continue
        today = (past.event_date.to_numpy() == day) & ok
        for lo, sh in split_legs(past, x, cond):
            for sel, side in ((lo, longs), (sh, shorts)):
                sel = sel[today[sel]]
                side += [(i, p + a, p + b) for i, p in zip(past.i.to_numpy()[sel], past.pos0.to_numpy()[sel])]
    return [(longs, shorts)], []


def leg(panel, positions, weight):
    """Daily return of one leg: buy-and-hold positions weighted at entry by market cap ('VW'), market cap capped
    at the NYSE 80th percentile ('VWcap') or equally ('EW'). Also the effective number of names (1 / sum of
    squared weights) and the largest weight, by day."""
    T = len(panel.cal)
    num, den, den2, vmax, cnt = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    for i, a, b in positions:
        b = min(b, T - 1)
        if a > b:
            continue
        m = panel.M[i, a]
        w0 = 1.0 if weight == 'EW' else (min(m, panel.cap80[a]) if weight == 'VWcap' else m)
        if not np.isfinite(w0) or w0 <= 0:
            continue
        r = panel.F[i, a:b + 1]
        v = w0 * np.concatenate(([1.0], np.cumprod(1 + r)[:-1]))
        num[a:b + 1] += v * r
        den[a:b + 1] += v
        den2[a:b + 1] += v * v
        vmax[a:b + 1] = np.maximum(vmax[a:b + 1], v)
        cnt[a:b + 1] += 1
    live = (den > 0) & (cnt >= MIN_NAMES)
    safe = np.where(den > 0, den, 1)
    ret = np.where(live, num / safe, np.nan)
    eff = np.where(live, den ** 2 / np.where(den2 > 0, den2, 1), np.nan)
    top = np.where(live, vmax / safe, np.nan)
    return ret, cnt, eff, top


def book_returns(panel, books, weight):
    """Long minus short, averaged over the books (formation months, momentum terciles) active that day, plus the
    names, effective names and largest weight per leg."""
    T = len(panel.cal)
    tot, nb = np.zeros(T), np.zeros(T)
    agg = {k: np.zeros(T) for k in ('nl', 'ns', 'el', 'es', 'tl', 'ts')}
    for longs, shorts in books:
        lr, lc, le, lt = leg(panel, longs, weight)
        sr, sc, se, st = leg(panel, shorts, weight)
        ok = ~np.isnan(lr) & ~np.isnan(sr)
        tot[ok] += lr[ok] - sr[ok]
        nb[ok] += 1
        for k, v in (('nl', lc), ('ns', sc), ('el', le), ('es', se), ('tl', lt), ('ts', st)):
            agg[k][ok] += v[ok]
    on = nb > 0
    ls = np.where(on, tot / np.maximum(nb, 1), np.nan)
    conc = {k: float(np.mean(v[on] / nb[on])) if on.any() else np.nan for k, v in agg.items()}
    return ls, conc


def stats(ls, panel, ff):
    import statsmodels.api as sm
    s = pd.Series(ls, index=panel.cal).dropna()
    if len(s) < 15:
        return {'days': len(s)}
    fit = sm.OLS(s.to_numpy(), np.ones((len(s), 1))).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    cum = (1 + s).cumprod()
    out = {'days': len(s), 'first': s.index[0].date(), 'last': s.index[-1].date(), 'mean_bp': 1e4 * s.mean(),
           't': float(fit.tvalues[0]), 'ann_ret': 252 * s.mean(), 'ann_vol': np.sqrt(252) * s.std(),
           'sharpe': np.sqrt(252) * s.mean() / s.std() if s.std() > 0 else np.nan,
           'cum_ret': cum.iloc[-1] - 1, 'max_dd': (cum / cum.cummax() - 1).min()}
    f = ff.reindex(s.index).dropna()
    if len(f) >= 15:
        y = s.loc[f.index].to_numpy()
        fit = sm.OLS(y, sm.add_constant(f[FACTORS].to_numpy())).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
        out.update({'alpha_bp': 1e4 * fit.params[0], 't_alpha': float(fit.tvalues[0]), 'r2_ff': fit.rsquared,
                    **{f'beta_{c}': fit.params[j + 1] for j, c in enumerate(FACTORS)}})
    out['passes_t3'] = abs(out['t']) >= 3
    return out


def fama_macbeth(d, panel, regressors, a, b):
    """Daily cross-sectional OLS of returns on standardized ranks of the regressors plus size, over stocks in days
    a..b after their event; mean slope (bp per 1 SD) with a Newey-West t over days."""
    import statsmodels.api as sm
    idx, p0 = d.i.to_numpy(), d.pos0.to_numpy()
    cols = {r: d[r + '_j'].to_numpy() for r in regressors}
    slopes = []
    for t in range(len(panel.cal)):
        act = (p0 + a <= t) & (t <= p0 + b)
        if act.sum() < 50:
            continue
        y = panel.R[idx[act], t]
        X = [pd.Series(v[act]).rank(pct=True).to_numpy() for v in cols.values()]
        X.append(np.log(panel.M[idx[act], t]))
        X = np.column_stack(X)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() < 50:
            continue
        Z = (X[ok] - X[ok].mean(0)) / X[ok].std(0)
        beta = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), Z]), y[ok], rcond=None)[0]
        slopes.append(beta[1:])
    S = np.array(slopes)
    names = list(regressors) + ['log size']
    res = {}
    for j, nm in enumerate(names):
        fit = sm.OLS(S[:, j], np.ones((len(S), 1))).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
        res[nm] = (1e4 * S[:, j].mean(), float(fit.tvalues[0]))
    return res, len(S)


def per_event(d, signal, panel, a, b):
    """The event-study view of the same positions: each event's CAR vs VTI over days a..b (winsorized at 1% and
    99%), top fifth minus bottom fifth, t clustered by event date. Every event counts once, where the calendar-time
    portfolio counts every day once."""
    import statsmodels.api as sm
    (longs, shorts), = event_books(d, signal, panel, a, b)[0]
    T = len(panel.cal)
    rows = []
    for side, pos in ((1.0, longs), (0.0, shorts)):
        for i, x, y in pos:
            y = min(y, T - 1)
            if x <= y:
                rows.append((side, (panel.F[i, x:y + 1] - panel.market[x:y + 1]).sum(), x - a))
    if len(rows) < 50:
        return {}
    side, car, day = (np.array(v) for v in zip(*rows))
    car = np.clip(car, *np.quantile(car, [0.01, 0.99]))
    fit = sm.OLS(car, sm.add_constant(side)).fit(cov_type='cluster', cov_kwds={'groups': day})
    return {'spread_pct': 100 * fit.params[1], 't_clustered': float(fit.tvalues[1]), 'se_pct': 100 * fit.bse[1],
            'n_long': int(side.sum()), 'n_short': int((1 - side).sum()), 'event_dates': int(len(np.unique(day)))}


def per_event_by_week(d, signal, panel, a, b, min_side=10):
    """The per-event spread within each reporting week (weeks with at least min_side events per leg), and the
    spread over all the other weeks, to see whether one week drives the result."""
    (longs, shorts), = event_books(d, signal, panel, a, b)[0]
    T = len(panel.cal)
    week_of = dict(zip(d.pos0, d.event_date.dt.to_period('W-SUN')))
    rows = [(side, (panel.F[i, x:min(y, T - 1) + 1] - panel.market[x:min(y, T - 1) + 1]).sum(), week_of[x - a])
            for side, pos in ((1, longs), (0, shorts)) for i, x, y in pos if x <= min(y, T - 1)]
    if not rows:
        return []
    t = pd.DataFrame(rows, columns=['side', 'car', 'week'])
    t['car'] = t.car.clip(*t.car.quantile([0.01, 0.99]))
    g = t.groupby(['week', 'side']).car.agg(['mean', 'size']).unstack('side')
    g = g[(g['size'][1] >= min_side) & (g['size'][0] >= min_side)]
    out = []
    for w in g.index:
        rest = t[t.week != w]
        out.append({'week_of': str(w.start_time.date()), 'spread_pct': 100 * (g['mean'][1][w] - g['mean'][0][w]),
                    'n_long': int(g['size'][1][w]), 'n_short': int(g['size'][0][w]),
                    'spread_without_this_week_pct': 100 * (rest[rest.side == 1].car.mean()
                                                           - rest[rest.side == 0].car.mean())})
    return out


SIGNALS = {'I/B/E/S events': {'SUE': 'sue'},
           'pipeline calls': {'SUE': 'sue', 'LLM (Gemini)': 'llm_gemini', 'LLM (Qwen)': 'llm_qwen',
                              'LLM beyond SUE': 'llm_resid'}}


def run(args):
    global MIN_NAMES
    MIN_NAMES = args.min_names
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = Panel(args.data_dir, args.raw_dir, shumway=not args.no_shumway)
    ff = pd.read_csv(Path(args.raw_dir) / 'ff_daily.csv', index_col=0, parse_dates=True)
    ff.index = ff.index.astype('datetime64[ns]')
    mom = momentum(args.raw_dir)
    samples = load_events(panel, args.data_dir, args.pipeline, args.llm, args.llm_qwen, mom,
                          by_close=not args.all_calls)
    print(f"panel {len(panel.keys):,} securities x {len(panel.cal)} days ({panel.cal[0].date()} to "
          f"{panel.cal[-1].date()}); universe {int(panel.universe.sum()):,} US common stocks on NYSE/AMEX/Nasdaq; "
          f"momentum {'yes' if mom is not None else 'NOT available (run pull)'}; Shumway "
          f"{panel.shumway if panel.shumway is not None else 'NOT available (run pull)'} securities")
    for name, d in samples.items():
        print(f"  {name}: {len(d):,} events in the universe ({int(d.nyse.sum()):,} NYSE); by month "
              f"{d.groupby('month').size().to_dict()}")

    rows, daily, lag_note = [], {}, {}

    def record(sample, label, scheme, horizon, weight, cond, made):
        books, lags = made
        ls, c = book_returns(panel, books, weight)
        name = f'{sample} | {label} | {scheme}, {horizon} | {weight} | {cond or "unconditional"}'
        daily[name] = ls
        rows.append({'sample': sample, 'signal': label, 'scheme': scheme, 'horizon': horizon, 'weight': weight,
                     'momentum': cond or 'unconditional', 'names_long': c['nl'], 'names_short': c['ns'],
                     'eff_names_long': c['el'], 'eff_names_short': c['es'], 'top_weight_long': c['tl'],
                     'top_weight_short': c['ts'], **stats(ls, panel, ff)})
        if lags:
            lag_note[(sample, label)] = (int(np.median(lags)), int(np.percentile(lags, 10)),
                                         int(np.percentile(lags, 90)))

    for sample, sigs in SIGNALS.items():
        d = samples.get(sample)
        if d is None:
            continue
        for label, sig in sigs.items():
            need = ['llm_gemini', 'sue'] if sig == 'llm_resid' else [sig]
            if any(c not in d or d[c].notna().sum() < 100 for c in need):
                continue
            print(f'  running {sample} / {label}', flush=True)
            ev_books = {h: event_books(d, sig, panel, a, b) for h, (a, b) in WINDOWS.items()}
            mo_books = {k: monthly_books(d, sig, panel, k) for k in (1, 2, 3)}
            mo_books['1-3'] = (mo_books[1][0] + mo_books[2][0] + mo_books[3][0], mo_books[1][1])
            for w in ('VWcap', 'VW', 'EW'):
                for h in WINDOWS:
                    record(sample, label, 'event-time', h, w, None, ev_books[h])
                for k in (1, 2, 3, '1-3'):
                    record(sample, label, 'monthly', f'month{"s" if k == "1-3" else ""} {k}', w, None, mo_books[k])
            if mom is not None:
                neutral = monthly_books(d, sig, panel, 1, 'neutral')
                terciles = {nm: monthly_books(d, sig, panel, 1, j)
                            for j, nm in ((1, 'low momentum'), (2, 'mid momentum'), (3, 'high momentum'))}
                for w in ('VWcap', 'VW'):
                    record(sample, label, 'monthly', 'month 1', w, 'momentum-neutral', neutral)
                    for nm, made in terciles.items():
                        record(sample, label, 'monthly', 'month 1', w, nm, made)
                record(sample, label, 'event-time', 'days 1-5', 'VWcap', 'momentum-neutral',
                       event_books(d, sig, panel, 1, 5, 'neutral'))

    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'factor_summary.csv', index=False)
    pe_rows = []
    for sample, sigs in SIGNALS.items():
        d = samples.get(sample)
        if d is None:
            continue
        for label, sig in sigs.items():
            need = ['llm_gemini', 'sue'] if sig == 'llm_resid' else [sig]
            if any(c not in d or d[c].notna().sum() < 100 for c in need):
                continue
            for h, (a, b) in (('days 1-5', (1, 5)), ('days 6-20', (6, 20)), ('days 21-60', (21, 60))):
                pe_rows.append({'sample': sample, 'signal': label, 'window': h, **per_event(d, sig, panel, a, b)})
    per_ev = pd.DataFrame(pe_rows)
    per_ev.to_csv(out / 'per_event_spreads.csv', index=False)
    wk_rows = []
    d = samples.get('pipeline calls')
    if d is not None:
        for label, sig in SIGNALS['pipeline calls'].items():
            need = ['llm_gemini', 'sue'] if sig == 'llm_resid' else [sig]
            if any(c not in d or d[c].notna().sum() < 100 for c in need):
                continue
            for h, (a, b) in (('days 1-5', (1, 5)), ('days 6-20', (6, 20)), ('days 21-60', (21, 60))):
                wk_rows += [{'sample': 'pipeline calls', 'signal': label, 'window': h, **r}
                            for r in per_event_by_week(d, sig, panel, a, b)]
    pd.DataFrame(wk_rows).to_csv(out / 'per_event_by_week.csv', index=False)
    pd.DataFrame(daily, index=panel.cal).to_csv(out / 'factor_daily_returns.csv', index_label='date')

    fm_rows = []
    ctrl = [[], ['mom']] if mom is not None else [[]]
    specs = (('I/B/E/S events', [['sue'] + c for c in ctrl]),
             ('pipeline calls', [r + c for r in (['sue'], ['llm_gemini'], ['sue', 'llm_gemini']) for c in ctrl]))
    for sample, regs in specs:
        d = samples.get(sample)
        if d is None:
            continue
        for reg in regs:
            if any(c not in d for c in reg):
                continue
            dd = d.dropna(subset=reg)
            for h, (a, b) in (('days 1-5', (1, 5)), ('days 6-20', (6, 20)), ('days 21-60', (21, 60))):
                res, n = fama_macbeth(dd, panel, reg, a, b)
                for nm, (bp, t) in res.items():
                    fm_rows.append({'sample': sample, 'model': ' + '.join(reg) + ' + log size', 'window': h,
                                    'regressor': nm, 'slope_bp_per_sd_per_day': bp, 't_nw': t, 'days': n,
                                    'passes_t3': abs(t) >= 3})
    fm = pd.DataFrame(fm_rows)
    fm.to_csv(out / 'fama_macbeth.csv', index=False)
    touched = {}
    for name, d in samples.items():
        i, p0 = d.i.to_numpy(), d.pos0.to_numpy()
        gone = panel.last[i] + 1
        touched[name] = int((panel.perf[i] & (gone >= p0 + 1) & (gone <= p0 + 60)).sum())
    meta = {'min_names_per_leg': MIN_NAMES, 'calls_transcript_by_day0_close_only': not args.all_calls,
            'universe_securities': int(panel.universe.sum()),
            'momentum': mom is not None,
            'shumway_securities': panel.shumway, 'shumway_events_hit_within_60_days': touched,
            'momentum_coverage': {k: float(v.mom.notna().mean()) for k, v in samples.items()},
            'monthly_entry_lag_days (median, p10, p90)':
                {f'{k[0]} / {k[1]}': v for k, v in lag_note.items()},
            'events': {k: int(len(v)) for k, v in samples.items()}}
    (out / 'run_info.json').write_text(json.dumps(meta, indent=2, default=str))
    charts(summary, daily, panel, out, per_ev)
    show = summary[summary.momentum.eq('unconditional')]
    cols = ['sample', 'signal', 'scheme', 'horizon', 'weight', 'days', 'mean_bp', 't', 'sharpe', 'alpha_bp',
            't_alpha', 'names_long', 'eff_names_long', 'top_weight_long']
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 500):
        print(show[cols].round(2).to_string(index=False))
        cond = summary[summary.momentum.ne('unconditional')]
        if len(cond):
            print(cond[['sample', 'signal', 'scheme', 'horizon', 'weight', 'momentum', 'days', 'mean_bp', 't',
                        'names_long', 'eff_names_long']].round(2).to_string(index=False))
        print(fm.round(2).to_string(index=False))
    print(json.dumps(meta, indent=1, default=str))
    print(f'wrote {out}')


def plot_cumulative(ax, cal, ret, color, ls='-', label=None, dots=False, lw=1.8):
    """Cumulative return in %: solid on the days the portfolio trades, dotted across the days it holds nothing,
    where the cumulative return can't change (a question from NTAM on Nov 11). Returns the trading days."""
    raw = pd.Series(np.asarray(ret, float), index=cal)
    on = raw.notna().to_numpy()
    cum = 100 * ((1 + raw.fillna(0.0)).cumprod() - 1)
    ax.plot(cal, cum.where(~on | ~np.append(on[1:], False)), ':', color=color, linewidth=0.9, alpha=0.8)
    ax.plot(cal, cum.where(on), ls, color=color, linewidth=lw, label=label)
    if dots:
        ax.plot(cal[on], cum[on], 'o', color=color, markersize=2.4)
    return list(raw.dropna().index)


def charts(summary, daily, panel, out, per_ev=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Segoe UI', 'DejaVu Sans']
    colors = {'SUE': '#2463a6', 'LLM (Gemini)': '#eb6834', 'LLM beyond SUE': '#7a5195', 'LLM (Qwen)': '#9c9a92'}
    s = summary[summary['sample'].eq('pipeline calls') & summary.weight.eq('VWcap')
                & summary.momentum.eq('unconditional')]
    if s.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200, sharey=True)
    for ax, (scheme, hs, title) in zip(axes, [('event-time', ['days 1-5', 'days 6-20', 'days 21-60'],
                                               'Enter the day after the announcement'),
                                              ('monthly', ['month 1', 'month 2', 'month 3'],
                                               'Rebalanced monthly (month after formation)')]):
        sigs = [g for g in ('SUE', 'LLM (Gemini)', 'LLM beyond SUE') if g in set(s.signal)]
        width = 0.8 / max(len(sigs), 1)
        for j, g in enumerate(sigs):
            r = s[s.signal.eq(g) & s.scheme.eq(scheme)].set_index('horizon').reindex(hs)
            x = np.arange(len(hs)) + (j - (len(sigs) - 1) / 2) * width
            se = (r.mean_bp / r.t).abs()
            ax.bar(x, r.mean_bp, width, color=colors[g], label=g)
            ax.errorbar(x, r.mean_bp, yerr=1.96 * se, fmt='none', ecolor='#52514e', elinewidth=1, capsize=2)
        ax.axhline(0, color='#c3c2b7', linewidth=1)
        ax.set_xticks(range(len(hs)), hs)
        ax.set_title(title, fontsize=11)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('Long-short, capped value weights, bp per day')
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Signal decay: where the long-short earns its return (calls, Q4 2025)',
                 fontsize=12, fontweight='bold')
    fig.text(0.01, 0.01, 'Bars: mean daily return of top minus bottom fifth (NYSE breakpoints, US common stocks, '
             'weights capped at the NYSE 80th percentile market cap). Lines: 95% intervals, Newey-West.',
             fontsize=7.5, color='#898781')
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out / 'decay_chart.png', facecolor='white')
    plt.close(fig)

    if per_ev is not None and len(per_ev):
        pe = per_ev[per_ev['sample'].eq('pipeline calls')]
        hs = ['days 1-5', 'days 6-20', 'days 21-60']
        sigs = [g for g in ('SUE', 'LLM (Gemini)', 'LLM beyond SUE') if g in set(pe.signal)]
        width = 0.8 / max(len(sigs), 1)
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
        for j, g in enumerate(sigs):
            r = pe[pe.signal.eq(g)].set_index('window').reindex(hs)
            x = np.arange(len(hs)) + (j - (len(sigs) - 1) / 2) * width
            ax.bar(x, r.spread_pct, width, color=colors[g], label=g)
            ax.errorbar(x, r.spread_pct, yerr=1.96 * r.se_pct, fmt='none', ecolor='#52514e', elinewidth=1, capsize=2)
            for xi, v, e in zip(x, r.spread_pct, r.se_pct):  # label beyond the end of the error bar
                y = v + (1.96 * e + 0.12) * (1 if v >= 0 else -1)
                ax.text(xi, y, f'{v:+.1f}%'.replace('-', '−'), ha='center', va='bottom' if v >= 0 else 'top',
                        fontsize=8, color='#0b0b0b')
        ax.axhline(0, color='#c3c2b7', linewidth=1)
        ax.set_xticks(range(len(hs)), ['days 1-5\n(week 1)', 'days 6-20\n(weeks 2-4)', 'days 21-60\n(weeks 5-12)'])
        ax.set_ylabel('Top minus bottom fifth, CAR vs VTI, %')
        ax.set_title('Signal decay per event: week one, then flat (SUE) or reversal (LLM)', fontsize=11.5,
                     fontweight='bold')
        ax.legend(frameon=False, fontsize=9)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        fig.text(0.01, 0.01, 'Calls with a transcript by the day-0 close, Q4 2025; US common stocks; NYSE breakpoints '
                 'from earlier announcements. Lines: 95% intervals, clustered by event date.', fontsize=7,
                 color='#898781')
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(out / 'per_event_chart.png', facecolor='white')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    live = []
    for g in ('SUE', 'LLM (Gemini)'):
        for scheme, horizon, ls in (('event-time', 'days 1-5', '-'), ('monthly', 'month 1', '--')):
            name = f'pipeline calls | {g} | {scheme}, {horizon} | VWcap | unconditional'
            if name in daily:
                live += plot_cumulative(ax, panel.cal, daily[name], colors[g], ls,
                                        f'{g}: {"held days 1-5" if scheme == "event-time" else "monthly rebalance"}',
                                        dots=scheme == 'event-time')
    ax.axhline(0, color='#c3c2b7', linewidth=1)
    if live:
        ax.set_xlim(min(live) - pd.Timedelta(days=4), max(live) + pd.Timedelta(days=4))
    fig.text(0.01, 0.01, f'Solid line (dots for days 1-5): days the portfolio trades. Dotted: days it holds nothing, '
             f'because a leg needs at least {MIN_NAMES} names, so the cumulative return can\'t change.', fontsize=7,
             color='#898781')
    ax.set_ylabel('Cumulative long-short return, %')
    ax.set_title('Calendar-time long-short, capped value weights (calls, Q4 2025)', fontsize=12,
                 fontweight='bold')
    ax.legend(frameon=False, fontsize=8.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / 'cumulative_chart.png', facecolor='white')
    plt.close(fig)


# --------------------------------------------------------------------------- decay
# NTAM's feedback of Nov 11-17: how fast the signal's information coefficient (IC) decays and its half-life (their
# proposal #4), the split by the sign of the surprise (#5), and a smoothed, signal-weighted portfolio that fades each
# stock at a half-life and drops it at the company's next announcement (alpha weights and smoothing, and #4).

IC_DAYS = 120            # event days 1..120 after day 0
FIT_DAYS = 90            # the half-life is fitted on days 1-90 (NTAM: the drift ends around day 90)
IC_WINDOWS = {'days 1-5': (1, 5), 'days 6-20': (6, 20), 'days 21-60': (21, 60), 'days 61-90': (61, 90),
              'days 91-120': (91, 120)}
HALF_LIVES = {'1 day': 1.0, '1 week': 5.0, '1 month': 21.0, '1 quarter': 63.0}
EXPIRY_FALLBACK = 63     # trading days held when the next announcement date is unknown
LIVE_WEIGHT = 0.01       # a smoothed position counts as a held name while its decay factor is at least 1%
SMOOTH_WEIGHTS = ('signal', 'signal x capped cap')
GROUPS = ('all', 'positive surprise', 'negative surprise')


def expiry_ages(d, panel, ibes_csv):
    """For each event, the age (trading days after day 0) of the last day it can be held: the day before the
    company's next quarterly announcement in I/B/E/S, or EXPIRY_FALLBACK where that date is unknown."""
    a = pd.read_csv(ibes_csv, usecols=['TICKER', 'FISCALP', 'ANNDATS_ACT'], low_memory=False)
    a = a[a.FISCALP.eq('QTR')].copy()
    a['ANNDATS_ACT'] = pd.to_datetime(a.ANNDATS_ACT, errors='coerce')
    a = a.dropna(subset=['ANNDATS_ACT']).drop_duplicates(['TICKER', 'ANNDATS_ACT'])
    by_ticker = {t: np.sort(g.ANNDATS_ACT.to_numpy('datetime64[ns]')) for t, g in a.groupby('TICKER')}
    ann = pd.to_datetime(d.ann_date, errors='coerce').fillna(d.event_date).to_numpy('datetime64[ns]')
    nxt = np.full(len(d), np.datetime64('NaT', 'ns'))
    for j, (t, x) in enumerate(zip(d.ibes_ticker.to_numpy(), ann)):
        s = by_ticker.get(t)
        if s is not None:
            later = s[s > x + np.timedelta64(20, 'D')]  # a later date within 20 days restates the same release
            if len(later):
                nxt[j] = later[0]
    known = ~np.isnat(nxt)
    filled = np.where(known, nxt, panel.cal[-1].to_datetime64())
    pos_next = panel.cal.searchsorted(pd.DatetimeIndex(filled))  # first trading day on or after it
    age = np.where(known, pos_next - 1 - d.pos0.to_numpy(), EXPIRY_FALLBACK)
    return np.maximum(age, 1), known


def event_ar(d, panel, days=IC_DAYS):
    """Abnormal return vs VTI on event days 1..days, one row per event; NaN after the panel ends. A missing return
    inside the panel earns the market (AR 0), with the Shumway return where it applies, as in the portfolios."""
    T = len(panel.cal)
    pos = d.pos0.to_numpy()[:, None] + np.arange(1, days + 1)[None, :]
    p = np.minimum(pos, T - 1)
    ar = panel.F[d.i.to_numpy()[:, None], p] - panel.market[p]
    return np.where(pos <= T - 1, ar, np.nan)


def next_announcement_car(d, panel, expiry, known):
    """CAR vs VTI over days -1..+1 around each company's next announcement; NaN when unknown or past the data."""
    T = len(panel.cal)
    pn = d.pos0.to_numpy() + expiry + 1  # first trading day on or after the next announcement
    i, out = d.i.to_numpy(), np.full(len(d), np.nan)
    for j in np.where(known & (pn - 1 >= 0) & (pn + 1 <= T - 1))[0]:
        t = np.arange(pn[j] - 1, pn[j] + 2)
        out[j] = float((panel.F[i[j], t] - panel.market[t]).sum())
    return out


def _std_rank(x):
    from scipy.stats import rankdata
    r = rankdata(x)
    sd = r.std()
    return (r - r.mean()) / (sd if sd > 0 else 1.0)


def rank_ic(x, y, groups):
    """Spearman rank correlation of x and y, with a standard error clustered by group (event date): the slope of
    y's standardized rank on x's, with the sandwich variance summed within groups."""
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 30:
        return np.nan, np.nan, n
    u, v = _std_rank(x[ok]), _std_rank(y[ok])
    ic = float(np.mean(u * v))
    s = pd.Series(u * (v - ic * u)).groupby(groups[ok]).sum().to_numpy()
    G = len(s)
    return ic, float(np.sqrt(G / max(G - 1, 1) * np.sum(s ** 2)) / np.sum(u ** 2)), n


def decay_fit(t, ic, se):
    """Weighted least squares fit of IC_k = a * 2^(-(k - 1) / H) over days k; returns (a, H in trading days)."""
    from scipy.optimize import curve_fit
    ok = np.isfinite(ic) & np.isfinite(se) & (se > 0)
    if ok.sum() < 5:
        return np.nan, np.nan
    a0 = float(np.mean(ic[ok][:3])) or 0.01
    try:
        p, _ = curve_fit(lambda k, a, h: a * 0.5 ** ((k - 1) / h), t[ok], ic[ok], p0=(a0, 5.0), sigma=se[ok],
                         absolute_sigma=True, bounds=([-1.0, 0.1], [1.0, 1000.0]), maxfev=10000)
        return float(p[0]), float(p[1])
    except (RuntimeError, ValueError):
        return np.nan, np.nan


def boot_daily_ic(x, AR, groups, boot, seed=0):
    """Cluster bootstrap over event dates of every day's IC (boot x days): resample dates with replacement and
    recompute each IC as a weighted correlation of the same ranks."""
    rng = np.random.default_rng(seed)
    gid, uniq = pd.factorize(groups)
    C = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=boot).astype(float)
    out = np.full((boot, AR.shape[1]), np.nan)
    for k in range(AR.shape[1]):
        ok = np.isfinite(x) & np.isfinite(AR[:, k])
        if ok.sum() < 30:
            continue
        u, v = _std_rank(x[ok]), _std_rank(AR[ok, k])
        W = C[:, gid[ok]]
        sw = W.sum(1)
        mu, mv = W @ u / sw, W @ v / sw
        cuv, cuu, cvv = W @ (u * v) / sw - mu * mv, W @ (u * u) / sw - mu ** 2, W @ (v * v) / sw - mv ** 2
        out[:, k] = cuv / np.sqrt(np.maximum(cuu * cvv, 1e-12))
    return out


def ic_analysis(x, AR, groups, labels, boot):
    """Daily, window and weekly ICs, the summed daily IC with a bootstrap band, and the fitted half-life."""
    rows, K = [], AR.shape[1]
    daily = [rank_ic(x, AR[:, k], groups) for k in range(K)]
    rows += [{**labels, 'kind': 'daily', 'start': k, 'end': k, 'n': n, 'ic': ic, 'se': se}
             for k, (ic, se, n) in enumerate(daily, 1)]
    for w, (a, b) in IC_WINDOWS.items():
        ic, se, n = rank_ic(x, AR[:, a - 1:b].sum(axis=1), groups)
        rows.append({**labels, 'kind': 'window', 'window': w, 'start': a, 'end': b, 'n': n, 'ic': ic, 'se': se})
    for w in range(K // 5):
        ic, se, n = rank_ic(x, AR[:, 5 * w:5 * w + 5].sum(axis=1), groups)
        rows.append({**labels, 'kind': 'weekly', 'start': 5 * w + 1, 'end': 5 * w + 5, 'n': n, 'ic': ic, 'se': se})
    ics, ses = np.array([r[0] for r in daily]), np.array([r[1] for r in daily])
    B = boot_daily_ic(x, AR, groups, boot)
    summed, band = np.nancumsum(ics), np.nancumsum(B, axis=1)
    lo, hi = np.nanpercentile(band, [2.5, 97.5], axis=0)
    # the summed IC is only shown while at least half the events are still held (before most next announcements);
    # later days rest on a few late reporters
    held = np.array([r[2] for r in daily]) >= 0.5 * daily[0][2]
    last = int(np.where(held)[0].max()) + 1 if held.any() else 1
    rows += [{**labels, 'kind': 'summed', 'start': 1, 'end': k + 1, 'n': daily[k][2], 'ic': summed[k],
              'lo': lo[k], 'hi': hi[k], 'half_still_held': bool(held[k])} for k in range(K)]
    t = np.arange(1, FIT_DAYS + 1, dtype=float)
    a, h = decay_fit(t, ics[:FIT_DAYS], ses[:FIT_DAYS])
    hb = np.array([decay_fit(t, B[j, :FIT_DAYS], ses[:FIT_DAYS])[1] for j in range(len(B))])
    hb = hb[np.isfinite(hb)]
    peak = int(np.nanargmax(summed[:min(FIT_DAYS, last)])) + 1
    fit = {**labels, 'n_events': daily[0][2], 'ic_day1_fit': a, 'half_life_days': h,
           'half_life_lo': float(np.percentile(hb, 2.5)) if len(hb) else np.nan,
           'half_life_hi': float(np.percentile(hb, 97.5)) if len(hb) else np.nan, 'boot_fits': int(len(hb)),
           'share_by_day5': 1 - 0.5 ** (5 / h) if np.isfinite(h) else np.nan,
           'share_by_day20': 1 - 0.5 ** (20 / h) if np.isfinite(h) else np.nan,
           'summed_ic_peak_day': peak, 'summed_ic_peak': float(summed[peak - 1]),
           'half_held_until_day': last, 'summed_ic_at_that_day': float(summed[last - 1]),
           'summed_ic_day5': float(summed[4]), 'summed_ic_day20': float(summed[19]),
           'summed_ic_day60': float(summed[59])}
    return rows, fit


def _cluster_se(psi, groups):
    """Standard error from per-observation influence values, summed within groups (event dates)."""
    s = pd.Series(psi).groupby(groups).sum().to_numpy()
    G = len(s)
    return float(np.sqrt(G / max(G - 1, 1) * np.sum(s ** 2)))


def car_paths(d, sig, panel, days=IC_DAYS):
    """Mean CAR vs VTI over days 1..h of the top and bottom fifths (the same NYSE breakpoints from announcements so
    far as in `run`), winsorized at 1%/99% per h, with standard errors clustered by event date; plus days 91-120
    alone, for NTAM's note that the bottom quintile reverses then."""
    (longs, shorts), = event_books(d, sig, panel, 1, days)[0]
    T = len(panel.cal)
    legs = {}
    for name, pos in (('top', longs), ('bottom', shorts)):
        if not pos:
            return pd.DataFrame(), {}
        i = np.array([p[0] for p in pos])
        p0 = np.array([p[1] - 1 for p in pos])
        at = p0[:, None] + np.arange(1, days + 1)[None, :]
        pp = np.minimum(at, T - 1)
        ar = np.where(at <= T - 1, panel.F[i[:, None], pp] - panel.market[pp], np.nan)
        legs[name] = (np.cumsum(ar, axis=1), p0)
    rows = []
    for h in range(days):
        (yt, gt), (yb, gb) = ((legs[k][0][:, h], legs[k][1]) for k in ('top', 'bottom'))
        okt, okb = np.isfinite(yt), np.isfinite(yb)
        if okt.sum() < 10 or okb.sum() < 10:
            continue
        lo, hi = np.quantile(np.concatenate([yt[okt], yb[okb]]), [0.01, 0.99])
        yt, gt, yb, gb = np.clip(yt[okt], lo, hi), gt[okt], np.clip(yb[okb], lo, hi), gb[okb]
        mt, mb = yt.mean(), yb.mean()
        rows.append({'day': h + 1, 'top_pct': 100 * mt, 'top_se': 100 * _cluster_se((yt - mt) / len(yt), gt),
                     'n_top': len(yt), 'bottom_pct': 100 * mb,
                     'bottom_se': 100 * _cluster_se((yb - mb) / len(yb), gb), 'n_bottom': len(yb),
                     'spread_pct': 100 * (mt - mb),
                     'spread_se': 100 * _cluster_se(np.concatenate([(yt - mt) / len(yt), -(yb - mb) / len(yb)]),
                                                    np.concatenate([gt, gb]))})
    late = {}
    if days >= 120:
        for name in ('top', 'bottom'):
            c, g = legs[name]
            y = c[:, 119] - c[:, 89]
            ok = np.isfinite(y)
            if ok.sum() >= 10:
                y, g = np.clip(y[ok], *np.quantile(y[ok], [0.01, 0.99])), g[ok]
                m = y.mean()
                late[name] = (100 * m, 100 * _cluster_se((y - m) / len(y), g), int(ok.sum()))
    return pd.DataFrame(rows), late


def pit_scores(d, sig):
    """Point-in-time score in [-1, 1]: 2 x the event's percentile among the NYSE announcements made so far, minus 1.
    It uses the same information as the NYSE breakpoints of `event_books`."""
    dates, nyse = d.event_date.to_numpy(), d.nyse.to_numpy()
    s = np.full(len(d), np.nan)
    for day in np.unique(dates):
        m = int((dates <= day).sum())  # d is sorted by date: the announcements so far are its first m rows
        x = values(d.iloc[:m], sig)
        ok = ~np.isnan(x)
        ref = np.sort(x[ok & nyse[:m]])
        if len(ref) < MIN_NYSE:
            continue
        today = np.where((dates[:m] == day) & ok)[0]
        pct = (np.searchsorted(ref, x[today], 'left') + np.searchsorted(ref, x[today], 'right')) / (2 * len(ref))
        s[today] = 2 * pct - 1
    return s


def smoothed_book(d, scores, expiry, panel, half_life, weight):
    """Daily return of the smoothed, signal-weighted long-short:

        w_i(t) = s_i x 2^(-(age - 1) / H)   for ages 1..expiry_i (age = trading days since day 0), else 0
               x min(ME_i, NYSE 80th-percentile ME)   for 'signal x capped cap'

    Positive weights form the long leg and negative ones the short leg, each scaled to $1 and rebalanced daily. A
    leg trades only with MIN_NAMES names whose decay factor is at least LIVE_WEIGHT."""
    T = len(panel.cal)
    ev = np.where(~np.isnan(scores))[0]
    i, p0, sc, ex = d.i.to_numpy()[ev], d.pos0.to_numpy()[ev], scores[ev], expiry[ev]
    age = np.arange(T)[None, :] - p0[:, None]
    fac = np.where((age >= 1) & (age <= ex[:, None]), 0.5 ** ((age - 1) / half_life), 0.0)
    w = sc[:, None] * fac
    if weight == 'signal x capped cap':
        cap = np.minimum(panel.M[i], panel.cap80[None, :])
        w = w * np.where(np.isfinite(cap) & (cap > 0), cap, 0.0)
    r = panel.F[i]
    wl, ws = np.where(w > 0, w, 0.0), np.where(w < 0, -w, 0.0)
    L, S = wl.sum(0), ws.sum(0)
    nl, ns = ((w > 0) & (fac >= LIVE_WEIGHT)).sum(0), ((w < 0) & (fac >= LIVE_WEIGHT)).sum(0)
    live = (nl >= MIN_NAMES) & (ns >= MIN_NAMES) & (L > 0) & (S > 0)
    Ls, Ss = np.where(L > 0, L, 1.0), np.where(S > 0, S, 1.0)
    ret = np.where(live, (wl * r).sum(0) / Ls - (ws * r).sum(0) / Ss, np.nan)
    # one-way daily turnover per leg (0 = no trade, 1 = the whole leg replaced), on consecutive live days
    turn = 0.25 * (np.abs(np.diff(wl / Ls, axis=1)).sum(0) + np.abs(np.diff(ws / Ss, axis=1)).sum(0))
    both = live[1:] & live[:-1]
    eff_l = L ** 2 / np.maximum((wl ** 2).sum(0), 1e-300)
    eff_s = S ** 2 / np.maximum((ws ** 2).sum(0), 1e-300)
    c = {'names_long': float(nl[live].mean()) if live.any() else np.nan,
         'names_short': float(ns[live].mean()) if live.any() else np.nan,
         'eff_names_long': float(eff_l[live].mean()) if live.any() else np.nan,
         'eff_names_short': float(eff_s[live].mean()) if live.any() else np.nan,
         'turnover': float(turn[both].mean()) if both.any() else np.nan}
    return ret, c


def decay(args):
    global MIN_NAMES
    MIN_NAMES = args.min_names
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = Panel(args.data_dir, args.raw_dir, shumway=not args.no_shumway)
    ff = pd.read_csv(Path(args.raw_dir) / 'ff_daily.csv', index_col=0, parse_dates=True)
    ff.index = ff.index.astype('datetime64[ns]')
    samples = load_events(panel, args.data_dir, args.pipeline, args.llm, args.llm_qwen, momentum(args.raw_dir),
                          by_close=not args.all_calls)
    qf = out / 'factor_daily_returns.csv'  # the Nov 11 portfolios from `run`, to compare trading days
    quint = pd.read_csv(qf, index_col=0, parse_dates=True).reindex(panel.cal) if qf.exists() else None
    ic_rows, fits, nxt_rows, paths, late_rows, sm_rows, sm_daily, exp_info = [], [], [], [], [], [], {}, {}
    for sample, sigs in SIGNALS.items():
        d = samples.get(sample)
        if d is None:
            continue
        AR_fixed, groups, sue = event_ar(d, panel), d.pos0.to_numpy(), d.sue.to_numpy(float)
        expiry, known = expiry_ages(d, panel, args.ibes)
        # headline ICs use each call's returns only until its company's next announcement (NTAM's #4 expiry);
        # the 'fixed days' rows keep all 120 days, as a check
        AR = np.where(np.arange(1, IC_DAYS + 1)[None, :] <= expiry[:, None], AR_fixed, np.nan)
        nxt_car = next_announcement_car(d, panel, expiry, known)
        exp_info[sample] = {'next_announcement_known': float(known.mean()),
                            'expiry_days_median_p10_p90': [int(np.median(expiry[known])),
                                                           int(np.percentile(expiry[known], 10)),
                                                           int(np.percentile(expiry[known], 90))]}
        masks = {'all': np.ones(len(d), bool), 'positive surprise': sue > 0, 'negative surprise': sue < 0}
        for label, sig in sigs.items():
            need = ['llm_gemini', 'sue'] if sig == 'llm_resid' else [sig]
            if any(c not in d or d[c].notna().sum() < 100 for c in need):
                continue
            print(f'  decay: {sample} / {label}', flush=True)
            x = values(d, sig) if sig == 'llm_resid' else d[sig].to_numpy(float)
            lab = {'sample': sample, 'signal': label}
            for grp in GROUPS:
                m = masks[grp]
                rows, fit = ic_analysis(x[m], AR[m], groups[m],
                                        {**lab, 'group': grp, 'returns': 'until next announcement'}, args.boot)
                ic_rows += rows
                fits.append(fit)
                if grp == 'all':
                    fitted = fit['half_life_days']
                v, se, n = rank_ic(x[m], nxt_car[m], groups[m])
                nxt_rows.append({**lab, 'group': grp, 'n': n, 'ic': v, 'se': se, 't': v / se if se else np.nan})
            rows, fit = ic_analysis(x, AR_fixed, groups, {**lab, 'group': 'all', 'returns': 'fixed days'}, args.boot)
            ic_rows += rows
            fits.append(fit)
            path, late = car_paths(d, sig, panel)
            paths.append(path.assign(sample=sample, signal=label))
            late_rows += [{**lab, 'leg': f'{k} fifth', 'car_days_91_120_pct': mm,
                           't_clustered': mm / se if se else np.nan, 'events': n} for k, (mm, se, n) in late.items()]
            hls = dict(HALF_LIVES, **({'IC half-life': round(fitted, 1)} if np.isfinite(fitted) else {}))
            scores = pit_scores(d, sig)
            qname = f'{sample} | {label} | event-time, days 1-5 | VWcap | unconditional'
            on = quint[qname].notna().to_numpy() if quint is not None and qname in quint else None
            for hname, h in hls.items():
                for w in SMOOTH_WEIGHTS:
                    ret, c = smoothed_book(d, scores, expiry, panel, h, w)
                    sm_daily[f'{sample} | {label} | smoothed, half-life {hname} | {w}'] = ret
                    split = {}
                    if on is not None:  # the same book on the days the Nov 11 fifths (days 1-5) trade, and on the rest
                        r = pd.Series(ret)
                        split = {'mean_bp_fifths_days': 1e4 * r[on].mean(), 'days_fifths': int(r[on].notna().sum()),
                                 'mean_bp_other_days': 1e4 * r[~on].mean(), 'days_other': int(r[~on].notna().sum())}
                    sm_rows.append({**lab, 'half_life': hname, 'half_life_days': h, 'weight': w, **c,
                                    **stats(ret, panel, ff), **split})
    ic = pd.DataFrame(ic_rows)
    fits = pd.DataFrame(fits)
    paths = pd.concat(paths, ignore_index=True)
    sm = pd.DataFrame(sm_rows)
    nxt = pd.DataFrame(nxt_rows)
    ic.to_csv(out / 'ic_decay.csv', index=False)
    fits.to_csv(out / 'half_life.csv', index=False)
    nxt.to_csv(out / 'next_announcement_ic.csv', index=False)
    paths.to_csv(out / 'car_paths.csv', index=False)
    pd.DataFrame(late_rows).to_csv(out / 'car_days_91_120.csv', index=False)
    sm.to_csv(out / 'smoothed_summary.csv', index=False)
    pd.DataFrame(sm_daily, index=panel.cal).to_csv(out / 'smoothed_daily_returns.csv', index_label='date')
    meta = {'ic_days': IC_DAYS, 'fit_days': FIT_DAYS, 'bootstrap_draws': args.boot, 'half_lives': HALF_LIVES,
            'expiry_fallback_days': EXPIRY_FALLBACK, 'live_weight': LIVE_WEIGHT, 'min_names_per_leg': MIN_NAMES,
            'panel_last_day': str(panel.cal[-1].date()), 'calls_transcript_by_day0_close_only': not args.all_calls,
            'events': {k: int(len(v)) for k, v in samples.items()}, 'expiry': exp_info}
    (out / 'decay_info.json').write_text(json.dumps(meta, indent=2, default=str))
    decay_charts(out, ic, fits, paths, sm_daily, panel, quint, exp_info.get('pipeline calls'))
    if (out / 'factor_summary.csv').exists() and quint is not None:  # redraw `run`'s charts from its saved output
        charts(pd.read_csv(out / 'factor_summary.csv'), {c: quint[c].to_numpy() for c in quint.columns}, panel, out,
               pd.read_csv(out / 'per_event_spreads.csv'))
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 500):
        print(fits.round(3).to_string(index=False))
        print(ic[ic.kind.eq('window')][['sample', 'signal', 'group', 'returns', 'window', 'n', 'ic', 'se']].round(4)
              .to_string(index=False))
        print(nxt.round(4).to_string(index=False))
        print(pd.DataFrame(late_rows).round(2).to_string(index=False))
        print(sm[['sample', 'signal', 'half_life', 'weight', 'days', 'mean_bp', 't', 'sharpe', 'alpha_bp', 't_alpha',
                  'names_long', 'eff_names_long', 'turnover'] + list(split)].round(2).to_string(index=False))
    print(json.dumps(meta, indent=1, default=str))
    print(f'wrote {out}')


def decay_charts(out, ic, fits, paths, sm_daily, panel, quint=None, expiry=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Segoe UI', 'DejaVu Sans']
    colors = {'SUE': '#2463a6', 'LLM (Gemini)': '#eb6834', 'LLM beyond SUE': '#7a5195', 'LLM (Qwen)': '#9c9a92'}
    sample, sigs = 'pipeline calls', ('SUE', 'LLM (Gemini)')
    head = 'until next announcement'
    ic = ic[ic.returns.eq(head)]
    fits = fits[fits.returns.eq(head)]
    nxt = expiry['expiry_days_median_p10_p90'] if expiry else None

    def rows(signal, kind, group='all'):
        return ic[ic['sample'].eq(sample) & ic.signal.eq(signal) & ic.group.eq(group) & ic.kind.eq(kind)]

    def tidy(ax, marks=(5, 20, 60, 90), next_band=True):
        ax.axhline(0, color='#c3c2b7', linewidth=1)
        for x in marks:
            ax.axvline(x + 0.5, color='#c3c2b7', linewidth=0.8, linestyle=':')
        if next_band and nxt:  # where most companies' next announcement falls (10th to 90th percentile)
            ax.axvspan(nxt[1] + 1, nxt[2] + 1, color='#f1efe8', zorder=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    # 1. daily IC with the fitted decay, and the summed IC
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=200)
    for g in sigs:
        dly, f = rows(g, 'daily'), fits[fits['sample'].eq(sample) & fits.signal.eq(g) & fits.group.eq('all')]
        if dly.empty or f.empty:
            continue
        f = f.iloc[0]
        dly = dly[dly.n >= 0.5 * dly.n.iloc[0]]  # days on which at least half the calls are still held
        a1.scatter(dly.start, dly.ic, s=7, color=colors[g], alpha=0.45, linewidths=0)
        if np.isfinite(f.half_life_days):
            k = np.arange(1, int(dly.start.max()) + 1)
            top = 'no upper bound' if f.half_life_hi >= 999 else f'{f.half_life_hi:.1f}'
            a1.plot(k, f.ic_day1_fit * 0.5 ** ((k - 1) / f.half_life_days), color=colors[g], linewidth=2,
                    label=f'{g}: half-life {f.half_life_days:.1f} days ({f.half_life_lo:.1f} to {top})')
        s = rows(g, 'summed')
        s = s[s.half_still_held.astype(bool)]
        a2.plot(s.end, s.ic, color=colors[g], linewidth=1.8, label=g)
        a2.fill_between(s.end, s.lo, s.hi, color=colors[g], alpha=0.15, linewidth=0)
    for ax in (a1, a2):
        tidy(ax)
        ax.set_xlabel('Trading days after the announcement (day 0)')
    a1.set_ylabel('Daily IC (rank correlation)')
    a1.set_title('Each day\'s IC, and the fitted exponential decay', fontsize=10.5)
    a2.set_ylabel('Summed daily IC, days 1 to h')
    a2.set_title('Summed IC: where the signal stops paying', fontsize=10.5)
    a1.legend(frameon=False, fontsize=7.8)
    a2.legend(frameon=False, fontsize=8.5)
    fig.suptitle('IC decay of the earnings signals (calls, Q4 2025)', fontsize=12, fontweight='bold')
    fig.text(0.01, 0.012, 'IC: Spearman correlation across calls between the score and that day\'s return vs VTI, using '
             'each call only until its company\'s next announcement (shaded: where 80% of those fall). Fit: IC_k = a × '
             '2^(−(k−1)/H),\nweighted least squares on days 1–90; brackets: 95% bootstrap interval (resampling event '
             'dates). Band: 95% bootstrap. Dotted lines: our Nov 11 windows end on days 5, 20 and 60, and day 90.',
             fontsize=6.6, color='#898781')
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out / 'ic_decay_chart.png', facecolor='white')
    plt.close(fig)

    # 2. summed IC within positive and within negative surprises
    tone = {'positive surprise': '#1b7f4b', 'negative surprise': '#b23a48'}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=200, sharey=True)
    for ax, g in zip(axes, sigs):
        for grp, col in tone.items():
            s = rows(g, 'summed', grp)
            s = s[s.half_still_held.astype(bool)]
            if s.empty:
                continue
            n = int(s.n.iloc[0])
            ax.plot(s.end, s.ic, color=col, linewidth=1.8, label=f'{grp}s ({n:,} calls)')
            ax.fill_between(s.end, s.lo, s.hi, color=col, alpha=0.13, linewidth=0)
        tidy(ax)
        ax.set_title(f'{g}: summed daily IC, by the sign of the surprise', fontsize=10.5)
        ax.set_xlabel('Trading days after the announcement (day 0)')
        ax.legend(frameon=False, fontsize=8.5)
    axes[0].set_ylabel('Summed daily IC, days 1 to h')
    fig.text(0.01, 0.012, 'Within each group, the IC ranks calls by the score. Positive (negative) surprise: actual EPS '
             'above (below) the consensus. Bands: 95% bootstrap over event dates.', fontsize=6.8, color='#898781')
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / 'ic_sign_chart.png', facecolor='white')
    plt.close(fig)

    # 3. top and bottom fifth CAR out to day 120
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=200, sharey=True)
    for ax, g in zip(axes, sigs):
        p = paths[paths['sample'].eq(sample) & paths.signal.eq(g)]
        if p.empty:
            continue
        for leg, col, lab in (('top', '#1b7f4b', 'top fifth'), ('bottom', '#b23a48', 'bottom fifth')):
            ax.plot(p.day, p[f'{leg}_pct'], color=col, linewidth=1.8, label=lab)
            ax.fill_between(p.day, p[f'{leg}_pct'] - 1.96 * p[f'{leg}_se'], p[f'{leg}_pct'] + 1.96 * p[f'{leg}_se'],
                            color=col, alpha=0.13, linewidth=0)
        ax.axvspan(90.5, 120.5, color='#f1efe8', zorder=0)
        if nxt:
            ax.axvline(nxt[0] + 1, color='#898781', linewidth=0.9, linestyle='--')
            ax.text(nxt[0] + 2, ax.get_ylim()[0], ' median next\n announcement', fontsize=7, color='#898781',
                    va='bottom')
        tidy(ax, marks=(), next_band=False)
        n1, n2 = int(p.n_top.iloc[0] + p.n_bottom.iloc[0]), int(p.n_top.iloc[-1] + p.n_bottom.iloc[-1])
        ax.set_title(f'{g}: CAR vs VTI by fifth ({n1:,} calls; {n2:,} reach day 120)', fontsize=10)
        ax.set_xlabel('Trading days after the announcement (day 0)')
        ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    axes[0].set_ylabel('Mean CAR from day 1, %')
    fig.text(0.01, 0.012, 'Fifths from NYSE breakpoints over earlier announcements, as in the Nov 11 dashboard. Shaded: '
             'days 91–120, where NTAM\'s note puts the bottom fifth\'s reversal. Bands: 95%, clustered by event date. '
             'Prices end Apr 30, 2026.', fontsize=6.6, color='#898781')
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / 'car_paths_chart.png', facecolor='white')
    plt.close(fig)

    # 4. smoothed portfolios vs the quintile days 1-5 portfolio, cumulative, with no-trading days dotted
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=200, sharey=True)
    styles = (('1 week', '#2463a6'), ('1 month', '#7a5195'), ('IC half-life', '#eb6834'))
    live = []
    for ax, g in zip(axes, sigs):
        f = fits[fits['sample'].eq(sample) & fits.signal.eq(g) & fits.group.eq('all')]
        hl = f'{f.iloc[0].half_life_days:.1f}' if len(f) else '?'
        series = [(f'smoothed, half-life {h if h != "IC half-life" else f"from the IC ({hl} days)"}',
                   sm_daily.get(f'{sample} | {g} | smoothed, half-life {h} | signal'), col) for h, col in styles]
        if quint is not None:
            name = f'{sample} | {g} | event-time, days 1-5 | VWcap | unconditional'
            if name in quint:
                series.append(('fifths, held days 1-5 (Nov 11)', quint[name].to_numpy(), '#52514e'))
        for lab, ret, col in series:
            if ret is not None:
                live += plot_cumulative(ax, panel.cal, ret, col, '-', lab, lw=1.6)
        tidy(ax, marks=(), next_band=False)
        ax.set_title(f'{g}: cumulative long-short', fontsize=10.5)
        ax.legend(frameon=False, fontsize=7.8)
    if live:
        for ax in axes:
            ax.set_xlim(pd.Timestamp(min(live)) - pd.Timedelta(days=4), pd.Timestamp(max(live)) + pd.Timedelta(days=4))
    axes[0].set_ylabel('Cumulative return, %')
    fig.text(0.01, 0.012, 'Smoothed: each call weighted by its point-in-time score, faded at the half-life and dropped '
             'the day before the company\'s next announcement; signal weights. Solid: trading days; dotted: no '
             'position.', fontsize=6.6, color='#898781')
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / 'smoothed_cumulative_chart.png', facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pull', help='Fama-French factors (public) and Compustat history + deletions (WRDS)')
    p.add_argument('raw_dir')
    p.add_argument('--start', default='2024-10')
    p.add_argument('--end', default='2025-11')
    r = sub.add_parser('run', help='build the factor portfolios and write the dashboard tables and charts')
    dc = sub.add_parser('decay', help='IC decay, half-life, the sign split and smoothed portfolios (Nov 18)')
    for q in (r, dc):
        q.add_argument('data_dir', help='the extended price folder (events_baseline, prices_clean, market_ret)')
        q.add_argument('raw_dir', help='folder written by `pull`')
        q.add_argument('out_dir')
        q.add_argument('--pipeline', help="the pipeline's final parquet (calls with transcripts)")
        q.add_argument('--llm', help="Austin's llm_scores.parquet (Gemini 2.5 Flash-Lite)")
        q.add_argument('--llm-qwen', help="Sean's llm_merged parquet (Qwen3-30B)")
        q.add_argument('--no-shumway', action='store_true')
        q.add_argument('--all-calls', action='store_true',
                       help="keep calls whose transcript came out after the day-0 close (default: drop them, since "
                            "the LLM couldn't have scored them by the entry)")
        q.add_argument('--min-names', type=int, default=30,
                       help='a leg trades only on days it holds at least this many names (default 30)')
    dc.add_argument('--ibes', default=str(ROOT.parent / 'data' / 'raw' / 'ibes_summary_stats_raw.csv'),
                    help='I/B/E/S summary statistics, for each company\'s next announcement date')
    dc.add_argument('--boot', type=int, default=500, help='bootstrap draws for the half-life and IC bands')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    {'pull': pull, 'run': run, 'decay': decay}[args.cmd](args)


if __name__ == '__main__':
    main()
