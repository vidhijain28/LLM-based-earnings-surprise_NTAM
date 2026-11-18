"""How fast does the LLM long-short decay, and what would a monthly rebalance get?

Two event-time views of the same fifths as llm_long_short.py (one pool, return against VTI, entry at the day-0
close, transcript available by that close):

    path    top minus bottom fifth, CAR from day 1 to day t, for t = 1 ... 60
    stale   the 20-day CAR a holder gets when the signal is already k days old (days k+1 .. k+20). A monthly
            rebalance holds signals of every age from 0 to about 20 days, so this is a rough proxy for it

For each stale window, the spread is also computed inside each reporting week (weeks with at least 10 events per
leg), to see whether a result comes from one week. Windows that run past the panel's last day drop the event, so
long windows cover only earlier reporters: read N.

Run:
    uv run python scripts/llm_decay.py <folder with the cleaned files> <output folder>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llm_long_short import load_prices, load_scores
from llm_based_earnings_surprise_ntam.signals.long_short import _spread, event_paths, quantile, window_return

PATH_DAYS = [1, 2, 3, 5, 10, 15, 20, 30, 40, 60]
STALE_AGES = [0, 5, 10, 15, 20, 40]


def spread(stock, mkt, q, dates, a, b):
    r = window_return(stock, mkt, a, b)
    ok, _, s, se, *_ = _spread(r, q, dates, 5, None, 0.01)
    return int(ok.sum()), s, s / se


def by_week(stock, mkt, q, dates, a, b, min_leg=10):
    r = window_return(stock, mkt, a, b)
    wk = pd.to_datetime(dates).dt.to_period('W').to_numpy()
    q = np.asarray(q)
    out = []
    for w in sorted(set(wk)):
        m = (wk == w) & ~np.isnan(r)
        top, bot = m & (q == 5), m & (q == 1)
        if top.sum() >= min_leg and bot.sum() >= min_leg:
            out.append((str(w.start_time.date()), int(top.sum()), int(bot.sum()), r[top].mean() - r[bot].mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('out_dir')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    ev, returns, _, market, _ = load_prices(args.data_dir)
    sc = load_scores(args.data_dir)
    rows, weeks = [], []
    for spec, g in sc.groupby('spec'):
        d = ev.merge(g[['gvkey_num', 'event_date', 'score', 'available', 'sue']], on=['gvkey_num', 'event_date'])
        d = d.dropna(subset=['score', 'surprise_pct', 'sue'])
        d = d[d.available].reset_index(drop=True)
        for sub, dd in [('all events', d), ('US large caps', d[d.large].reset_index(drop=True))]:
            stock, mkt, _ = event_paths(dd.key, dd.event_date, returns, market, horizon=60)
            for sig_name, sig in [('LLM', dd.score), ('surprise (pipeline sue)', dd.sue)]:
                q = quantile(sig)
                base = dict(spec=spec, subsample=sub, signal=sig_name)
                for t in PATH_DAYS:
                    n, s, tt = spread(stock, mkt, q, dd.event_date, 1, t)
                    rows.append(dict(base, view='path', start=1, end=t, N=n, spread=s, t=tt))
                for k in STALE_AGES:
                    n, s, tt = spread(stock, mkt, q, dd.event_date, k + 1, k + 20)
                    rows.append(dict(base, view='stale', start=k + 1, end=k + 20, N=n, spread=s, t=tt))
                    for w, nl, ns, s_w in by_week(stock, mkt, q, dd.event_date, k + 1, k + 20):
                        weeks.append(dict(base, start=k + 1, end=k + 20, week=w, n_long=nl, n_short=ns, spread=s_w))
    r, w = pd.DataFrame(rows), pd.DataFrame(weeks)
    r.to_csv(out / 'llm_decay.csv', index=False)
    w.to_csv(out / 'llm_decay_by_week.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_rows', 500, 'display.max_colwidth', 70):
        print(r.round(4).to_string(index=False))
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
