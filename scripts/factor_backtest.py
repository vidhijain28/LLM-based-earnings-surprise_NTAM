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
                         columns=['gvkey', 'iid', 'event_date', 'surprise_pct'])
    ev['key'] = ev.gvkey.astype(str) + '|' + ev.iid.astype(str)
    ev['event_date'] = pd.to_datetime(ev.event_date).astype('datetime64[ns]')
    ev = ev.merge(prim[['key', 'event_date']], on=['key', 'event_date']).rename(columns={'surprise_pct': 'sue'})
    samples = {'I/B/E/S events': ev[['key', 'event_date', 'sue']]}
    if pipeline:
        d = pd.read_parquet(pipeline, columns=['keydevid', 'gvkey', 'event_date', 'sue', 'in_sample',
                                               'transcript_by_event_close'])
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
    for g, style in (('SUE', '-'), ('LLM (Gemini)', '-')):
        for scheme, horizon, ls in (('event-time', 'days 1-5', '-'), ('monthly', 'month 1', '--')):
            name = f'pipeline calls | {g} | {scheme}, {horizon} | VWcap | unconditional'
            if name in daily:
                raw = pd.Series(daily[name], index=panel.cal)
                live += list(raw.dropna().index)
                r = raw.fillna(0.0)
                ax.plot(panel.cal, 100 * ((1 + r).cumprod() - 1), ls, color=colors[g], linewidth=1.8,
                        label=f'{g}: {"held days 1-5" if scheme == "event-time" else "monthly rebalance"}')
    ax.axhline(0, color='#c3c2b7', linewidth=1)
    if live:
        ax.set_xlim(min(live) - pd.Timedelta(days=4), max(live) + pd.Timedelta(days=4))
    fig.text(0.01, 0.01, f'Flat stretches: no trading that day (each leg needs at least {MIN_NAMES} names). '
             'One quarter of announcements, so the portfolios only trade for a few weeks.', fontsize=7,
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pull', help='Fama-French factors (public) and Compustat history + deletions (WRDS)')
    p.add_argument('raw_dir')
    p.add_argument('--start', default='2024-10')
    p.add_argument('--end', default='2025-11')
    r = sub.add_parser('run', help='build the factor portfolios and write the dashboard tables and charts')
    r.add_argument('data_dir', help='the extended price folder (events_baseline, prices_clean, market_ret)')
    r.add_argument('raw_dir', help='folder written by `pull`')
    r.add_argument('out_dir')
    r.add_argument('--pipeline', help="the pipeline's final parquet (calls with transcripts)")
    r.add_argument('--llm', help="Austin's llm_scores.parquet (Gemini 2.5 Flash-Lite)")
    r.add_argument('--llm-qwen', help="Sean's llm_merged parquet (Qwen3-30B)")
    r.add_argument('--no-shumway', action='store_true')
    r.add_argument('--all-calls', action='store_true',
                   help="keep calls whose transcript came out after the day-0 close (default: drop them, since the "
                        "LLM couldn't have scored them by the entry)")
    r.add_argument('--min-names', type=int, default=30,
                   help='a leg trades only on days it holds at least this many names (default 30)')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    if args.cmd == 'pull':
        pull(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
