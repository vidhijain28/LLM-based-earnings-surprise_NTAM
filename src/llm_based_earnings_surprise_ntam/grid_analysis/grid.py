"""Run the grid: every configuration of grid.yaml as a daily calendar-time long-short portfolio, and its Sharpe.

    python -m llm_based_earnings_surprise_ntam.grid_analysis.grid [--config grid.yaml]

A configuration = one level of each setting in grid.yaml `grid`. For each one:
    1. signal     the scoring version's LLM score, or its narrative surprise (signals.py), optionally x persistence
    2. legs       point-in-time fifths or tenths, over all calls so far or inside the call's FF12 industry
    3. weights    1 per call, or rules.attention_weight for the lowest third of attention so far
    4. holding    enter at the day-0 close; fixed: weight 1 on days 1..h; decay: weight 0.5^((d-1)/half-life) on
                  day d, last day the one before the next event date (rules.next_unknown_days if unknown)
    5. daily return of a leg = sum(weight x abnormal return) / capital; fixed holds: capital = the day's sum of
                  open weights (equal weights, fully invested); decay: capital = the leg's largest daily sum of
                  open weights (not rescaled: faded weight sits in cash). A leg with fewer than
                  rules.min_names open names that day is not traded (0). Day return = long leg - short leg.
    6. Sharpe     mean / sd of the daily returns on the configuration's active days (a position open) x sqrt(252)
For N_eff and the correlations every configuration uses one common window, with 0 on days it holds nothing.

Writes to paths.out: configs.parquet (one row per configuration: settings, Sharpe and other statistics),
daily_returns.parquet (configurations x days), summary.json (sample, variance decomposition, N_eff, deflated
Sharpe, permutation null if permutations.runs > 0).
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..GEPA_Optimization.config import load
from . import panel, stats
from .pull import tradable_calls
from .signals import Days, legs

DEFAULT = Path(__file__).with_name('grid.yaml')
FACTORS = ['version', 'signal', 'persistence', 'industry', 'sort', 'attention', 'holding']


def load_inputs(cfg):
    """Tradable calls with every scoring version's score and run P's persistence, sorted by event date, plus the panel."""
    calls = tradable_calls(cfg)
    calls = calls.rename(columns={'prediction': 'score_0'})
    out = Path(cfg['paths']['scores'])
    for ver in cfg['grid']['version']:
        if ver == '0':
            continue
        s = pd.read_parquet(out / f'scores_{ver}.parquet', columns=['transcriptid', 'status', 'prediction'])
        s = s[s['status'] == 'ok'].rename(columns={'prediction': f'score_{ver}'})
        calls = calls.merge(s[['transcriptid', f'score_{ver}']], on='transcriptid', how='left')
    p = pd.read_parquet(out / 'scores_P.parquet', columns=['transcriptid', 'status', 'persistence'])
    calls = calls.merge(p[p['status'] == 'ok'][['transcriptid', 'persistence']], on='transcriptid', how='left')

    calls, cal, AR, info = panel.build(cfg, calls)
    order = np.lexsort((calls['transcriptid'].to_numpy(), calls['event_date'].to_numpy()))
    calls, AR = calls.iloc[order].reset_index(drop=True), AR[order]
    # keep the calendar from the first event day on
    first = int(calls['t0'].min())
    cal, AR = cal[first:], AR[:, first:]
    calls['t0'] -= first
    calls['t_next'] = calls['t_next'] - first
    info['missing_scores'] = {c: int(calls[c].isna().sum())
                              for c in [f'score_{a}' for a in cfg['grid']['version']] + ['persistence']}
    return calls, cal, AR, info


def holdings(cfg, calls, T):
    """{holding: (n x T weights, kind)}: the weight each call's position has on each day."""
    out = {}
    t0 = calls['t0'].to_numpy()
    tn = calls['t_next'].to_numpy()
    unknown = int(cfg['rules']['next_unknown_days'])
    for name, h in cfg['grid']['holding'].items():
        H = np.zeros((len(calls), T))
        for i, a in enumerate(t0):
            if h['kind'] == 'fixed':
                H[i, a + 1:min(a + 1 + int(h['days']), T)] = 1.0
            else:
                last = int(tn[i]) - 1 if np.isfinite(tn[i]) else a + unknown
                last = min(last, T - 1)
                d = np.arange(1, last - a + 1)
                H[i, a + 1:last + 1] = 0.5 ** ((d - 1) / float(h['half_life']))
        out[name] = (H, h['kind'])
    return out


def books(cfg, calls, days, llm):
    """(keys, long weights K x n, short weights K x n) for every configuration except the holding.

    `llm` = {'score_<version>': scores, 'persistence': ...}; the permutation null passes shuffled copies."""
    r, g = cfg['rules'], cfg['grid']
    pct_sue = days.pct(calls['sue'])
    pct_d0 = days.pct(calls['r_abn'])
    att_pct = days.pct(calls['attention'])
    att_w = np.where(np.isfinite(att_pct) & (att_pct < r['attention_low_share']), r['attention_weight'], 1.0)
    keys, WL, WS = [], [], []
    for ver in g['version']:
        score = np.asarray(llm[f'score_{ver}'], float)
        base = {'llm': score, 'narrative': days.narrative(days.pct(score), pct_sue, pct_d0)}
        for sig, pers in itertools.product(g['signal'], g['persistence']):
            x = base[sig]
            if pers == 'weighted':
                x = (days.pct(x) - 0.5) * np.asarray(llm['persistence'], float)
            for ind in g['industry']:
                pc = days.pct(x) if ind == 'none' else days.pct_in_groups(x, calls['industry'],
                                                                          int(r['min_industry_calls']))
                for sort, q in g['sort'].items():
                    side = legs(pc, q)
                    for att in g['attention']:
                        w = att_w if att == 'low' else np.ones(len(side))
                        keys.append({'version': ver, 'signal': sig, 'persistence': pers, 'industry': ind,
                                     'sort': sort, 'attention': att})
                        WL.append((side == 1) * w)
                        WS.append((side == -1) * w)
    return keys, np.array(WL), np.array(WS)


def portfolio(WL, WS, H, kind, AR, min_names):
    """Daily returns (K x T), active-day mask, and open names per day in each leg. A leg trades on a day only
    if it holds at least `min_names` open names; otherwise it counts 0 that day."""
    RH = H * AR
    open_ = H > 0
    out = []
    for W in (WL, WS):
        num, den = W @ RH, W @ H
        names = (W > 0).astype(float) @ open_
        trades = names >= min_names
        den = np.where(trades, den, 0.0)
        if kind == 'fixed':
            cap = den                                     # equal weights, fully invested each day
        else:
            cap = np.broadcast_to(den.max(axis=1, keepdims=True), den.shape)   # decay: not rescaled
        leg = np.divide(num, cap, out=np.zeros_like(num), where=trades & (cap > 0))
        out.append((leg, trades, names))
    (leg_l, tr_l, names_l), (leg_s, tr_s, names_s) = out
    return leg_l - leg_s, tr_l | tr_s, names_l, names_s


def run_grid(cfg, calls, AR, hold, llm):
    """All configurations: (settings table, daily returns K_total x T, active mask)."""
    days = Days(calls['event_date'], int(cfg['rules']['min_calls']))
    keys, WL, WS = books(cfg, calls, days, llm)
    rows, rets, acts = [], [], []
    for name, (H, kind) in hold.items():
        ret, active, nl, ns = portfolio(WL, WS, H, kind, AR, int(cfg['rules']['min_names']))
        for k, key in enumerate(keys):
            a = active[k]
            rows.append({**key, 'holding': name, 'active_days': int(a.sum()),
                         'long_calls': int((WL[k] > 0).sum()), 'short_calls': int((WS[k] > 0).sum()),
                         'open_long_avg': float(nl[k][a].mean()) if a.any() else 0.0,
                         'open_short_avg': float(ns[k][a].mean()) if a.any() else 0.0})
        rets.append(ret)
        acts.append(active)
    R, A = np.vstack(rets), np.vstack(acts)
    table = pd.DataFrame(rows)
    table = pd.concat([table, pd.DataFrame(stats.sharpe_table(R, A, cfg['rules']['annualize']))], axis=1)
    return table, R, A


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(DEFAULT))
    args = ap.parse_args()
    cfg = load(args.config)
    out = Path(cfg['paths']['out'])
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    calls, cal, AR, info = load_inputs(cfg)
    hold = holdings(cfg, calls, len(cal))
    llm = {c: calls[c].to_numpy() for c in calls if c.startswith('score_') or c == 'persistence'}
    table, R, A = run_grid(cfg, calls, AR, hold, llm)
    print(f'{len(table)} configurations on {len(calls)} calls, {len(cal)} days, {time.time() - t0:.0f}s', flush=True)

    # one common window for every configuration: first to last day any configuration holds a position
    live = np.where(A.any(axis=0))[0]
    win = slice(live[0], live[-1] + 1)
    Rw = np.where(A, R, 0.0)[:, win]
    dates = cal[win]

    summary = {'sample': info, 'window': [str(dates[0].date()), str(dates[-1].date()), len(dates)],
               'traded_calls_first_day': str(calls['event_date'].iloc[int(cfg['rules']['min_calls']) - 1].date()),
               'decomposition': stats.decompose(table, FACTORS, 'sharpe'),
               'audit': stats.audit(table, R, A, Rw, cfg['rules']['annualize'])}

    runs = int(cfg['permutations']['runs'])
    if runs > 0:
        null = stats.permutation_null(cfg, calls, AR, hold, llm, runs, int(cfg['permutations']['seed']), run_grid)
        best = float(table['sharpe'].max())
        summary['permutation'] = {'runs': runs, 'null_max': null, 'best': best,
                                  'p_value': (1 + sum(v >= best for v in null)) / (runs + 1),
                                  'implied_n': stats.implied_n(float(np.mean(null)), table, cfg['rules']['annualize'])}

    table.to_parquet(out / 'configs.parquet', index=False)
    pd.DataFrame(Rw, columns=[d.strftime('%Y-%m-%d') for d in dates]).assign(config=range(len(table))) \
        .to_parquet(out / 'daily_returns.parquet', index=False)
    calls[['transcriptid', 'gvkey', 'event_date', 'industry', 'attention', 't0', 't_next']] \
        .to_parquet(out / 'calls_used.parquet', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ('window', 'traded_calls_first_day')}, default=str))
    print(f"best Sharpe {table['sharpe'].max():.2f}, median {table['sharpe'].median():.2f}; "
          f"N_eff {summary['audit']['n_eff']:.1f}; wrote {out}", flush=True)


if __name__ == '__main__':
    main()
