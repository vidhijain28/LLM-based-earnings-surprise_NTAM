"""Event-time long-short returns for an earnings-announcement signal (NTAM stage gate 1).

Day 0 is the event date: the first trading day investors can act on the release (16:00 ET rule). Events are
sorted into quantiles of a signal (the earnings surprise now, the LLM score later), and a portfolio that is long
the top quantile and short the bottom one is formed at the day-0 close. Its return over days 1..h after the
event, net of the market, is the cumulative abnormal return (CAR) the stage gate asks for at 1, 4 and 12 weeks
(h = 5, 20, 60 trading days). Day 0 itself is reported as the announcement reaction; a signal built from the
release cannot trade it.

Inputs are plain pandas objects, so the same code runs on Hazel's Compustat files now and on CRSP returns from
the pipeline later: security keys and event dates, a wide daily-return panel (one row per security, one column
per trading day) and a market-return series on the same calendar.
"""
import numpy as np
import pandas as pd

WINDOWS = {'day 0': (0, 0), '1 week': (1, 5), '4 weeks': (1, 20), '12 weeks': (1, 60)}


def event_paths(keys, event_dates, returns, market, horizon=60):
    """Stock and market returns on days 0..horizon after each event, arrays of shape (n_events, horizon + 1).

    Days past the end of the panel are NaN: the window is incomplete and is left out of that horizon. A missing
    stock return inside the panel (no trade, invalid price, delisting) is replaced by the market return, i.e. the
    money sits in the market that day and earns no abnormal return. `filled` flags those days.
    """
    cal = pd.DatetimeIndex(returns.columns)
    row = pd.Index(returns.index).get_indexer(keys)
    col = cal.get_indexer(pd.DatetimeIndex(event_dates))
    if (row < 0).any() or (col < 0).any():
        raise ValueError('every event needs a row in `returns` and an event date on its calendar')
    R, M = returns.to_numpy(dtype=float), market.reindex(cal).to_numpy(dtype=float)
    n, T = len(row), len(cal)
    stock, mkt = np.full((n, horizon + 1), np.nan), np.full((n, horizon + 1), np.nan)
    filled = np.zeros((n, horizon + 1), dtype=bool)
    for e in range(n):
        k = min(horizon, T - 1 - col[e])
        r, m = R[row[e], col[e]:col[e] + k + 1], M[col[e]:col[e] + k + 1]
        filled[e, :k + 1] = np.isnan(r)
        stock[e, :k + 1], mkt[e, :k + 1] = np.where(np.isnan(r), m, r), m
    return stock, mkt, filled


def window_return(stock, mkt, start, end, compound=False):
    """Abnormal return over days start..end: summed daily abnormal returns (CAR) or buy-and-hold (BHAR).
    NaN where the window runs past the data."""
    s, m = stock[:, start:end + 1], mkt[:, start:end + 1]
    out = np.prod(1 + s, axis=1) - np.prod(1 + m, axis=1) if compound else (s - m).sum(axis=1)
    return np.where(np.isnan(stock[:, end]), np.nan, out)


def quantile(signal, n=5):
    """Quantile 1 (lowest) .. n (highest) of `signal` within the sample, ties broken by position."""
    signal = pd.Series(signal)
    return pd.Series(pd.qcut(signal.rank(method='first'), n, labels=False) + 1, index=signal.index)


def quantile_pit(signal, dates, n=5, min_history=250):
    """Quantile against the signals of events on earlier dates only, which is what a live signal could use.

    The breakpoints come from every earlier event in the sample. Events with fewer than `min_history` earlier
    events get NaN.
    """
    s = pd.Series(signal, dtype=float)
    d = pd.Series(pd.DatetimeIndex(dates), index=s.index)
    out = pd.Series(np.nan, index=s.index)
    for day in np.sort(d.unique()):
        past = s[d < day].dropna()
        if len(past) < min_history:
            continue
        today = (d == day) & s.notna()
        out[today] = np.searchsorted(np.quantile(past, np.arange(1, n) / n), s[today], side='right') + 1
    return out


def ols_cluster(y, X, groups, weights=None):
    """OLS (WLS with `weights`) coefficients and standard errors clustered by `groups`, with the usual
    small-sample factor G/(G-1) * (N-1)/(N-K)."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    if weights is not None:
        sw = np.sqrt(np.asarray(weights, float))
        y, X = y * sw, X * sw[:, None]
    inv = np.linalg.inv(X.T @ X)
    beta = inv @ X.T @ y
    g = pd.factorize(np.asarray(groups))[0]
    G, (n, k) = g.max() + 1, X.shape
    score = np.zeros((G, k))
    np.add.at(score, g, X * (y - X @ beta)[:, None])
    V = inv @ score.T @ score @ inv * G / (G - 1) * (n - 1) / (n - k)
    return beta, np.sqrt(np.diag(V))


def _spread(r, q, groups, n, weights, winsor):
    """Winsorize r over the events with a quantile, then the top-minus-bottom mean and its clustered SE."""
    q = np.asarray(q, float)
    ok = ~np.isnan(r) & ~np.isnan(q)
    if winsor:
        r = np.where(ok, np.clip(r, *np.nanquantile(np.where(ok, r, np.nan), [winsor, 1 - winsor])), np.nan)
    w = None if weights is None else np.asarray(weights, float)
    legs = ok & np.isin(q, [1, n])
    X = np.column_stack([np.ones(legs.sum()), q[legs] == n])
    beta, se = ols_cluster(r[legs], X, np.asarray(groups)[legs], None if w is None else w[legs])
    means = {f'Q{k}': np.average(r[ok & (q == k)], weights=None if w is None else w[ok & (q == k)])
             for k in range(1, n + 1)}
    means['all'] = np.average(r[ok], weights=None if w is None else w[ok])
    return ok, means, beta[1], se[1], int((legs & (q == n)).sum()), int((legs & (q == 1)).sum())


def _rank_r2(r, signal, ok):
    """R-squared of PRank(r) on PRank(signal) over `ok` rows (one regressor, so R2 = corr^2), as in the paper."""
    y = pd.Series(r[ok]).rank(pct=True)
    x = pd.Series(np.asarray(signal, float)[ok]).rank(pct=True)
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def long_short(stock, mkt, q, groups, signal=None, n=5, windows=WINDOWS, compound=False, weights=None,
               winsor=0.01):
    """One row per window: events, quantile means, top-minus-bottom spread, its t-stat (clustered by `groups`)
    and, if `signal` is given, the rank R-squared of the window return on the signal."""
    rows = []
    for name, (a, b) in windows.items():
        r = window_return(stock, mkt, a, b, compound)
        ok, means, s, se, n_long, n_short = _spread(r, q, groups, n, weights, winsor)
        rows.append({'window': name, 'days': f'{a}-{b}', 'N': int(ok.sum()), 'n_long': n_long, 'n_short': n_short,
                     **means, 'spread': s, 'se': se, 't': s / se,
                     'rank_r2': np.nan if signal is None else _rank_r2(r, signal, ok)})
    return pd.DataFrame(rows)


def spread_path(stock, mkt, q, groups, n=5, start=1, horizon=60, weights=None, winsor=0.01):
    """Cumulative top-minus-bottom abnormal return from day `start` through each day t, with clustered SEs.
    Row t uses the events whose window reaches day t."""
    rows = [{'day': start - 1, 'N': np.nan, 'spread': 0.0, 'se': 0.0}]
    for t in range(start, horizon + 1):
        r = window_return(stock, mkt, start, t)
        if np.isnan(r).all():
            break
        ok, _, s, se, _, _ = _spread(r, q, groups, n, weights, winsor)
        rows.append({'day': t, 'N': int(ok.sum()), 'spread': s, 'se': se})
    return pd.DataFrame(rows)
