"""Point-in-time signals, legs and weights: each call is compared only with the calls that came in up to its own
event date (no look-ahead).

    pct           a call's mid-rank percentile among the calls so far (equal values share their mean rank)
    narrative     residual of the LLM pct on the SUE pct and the day-0 return pct, from an OLS on the calls so far
    persistence   (pct of the signal - 0.5) x run P's persistence, then ranked again
    legs          long if pct >= 1 - 1/q, short if pct < 1/q (q = 5 fifths, 10 tenths); with FF12-neutral the pct
                  is taken inside the call's industry, or inside one pooled group while the industry has fewer
                  than rules.min_industry_calls calls so far
    attention     weight rules.attention_weight for calls in the lowest third of attention so far, else 1
A day trades only once rules.min_calls calls have come in; earlier calls get no signal.
"""
import numpy as np

POOL = '__pooled__'


def _midrank_pct(ref, x):
    """Mid-rank percentile of each x among the sorted reference values (x itself is in ref)."""
    lo = np.searchsorted(ref, x, side='left')
    hi = np.searchsorted(ref, x, side='right')
    return (lo + 0.5 * (hi - lo)) / len(ref)


class Days:
    """The calls grouped by event date, in order; calls must be sorted by event date."""

    def __init__(self, event_date, min_calls):
        d = np.asarray(event_date, dtype='datetime64[ns]')
        if (np.diff(d.astype('int64')) < 0).any():
            raise ValueError('calls must be sorted by event date')
        self.n = len(d)
        _, start = np.unique(d, return_index=True)
        self.bounds = list(zip(start, list(start[1:]) + [self.n]))   # [a, b) of each day
        self.min_calls = min_calls

    def pct(self, x):
        """Point-in-time pct of x (NaN where x is NaN or too few calls have come in)."""
        x = np.asarray(x, float)
        out = np.full(self.n, np.nan)
        for a, b in self.bounds:
            if b < self.min_calls:
                continue
            ref = np.sort(x[:b][np.isfinite(x[:b])])
            today = np.arange(a, b)[np.isfinite(x[a:b])]
            out[today] = _midrank_pct(ref, x[today])
        return out

    def pct_in_groups(self, x, industry, min_industry):
        """FF12-neutral pct: inside the call's industry, or against all calls so far of currently small ones."""
        x = np.asarray(x, float)
        ind = np.array([POOL if (v is None or v != v) else v for v in industry], dtype=object)
        out = np.full(self.n, np.nan)
        for a, b in self.bounds:
            if b < self.min_calls:
                continue
            names, counts = np.unique(ind[:b], return_counts=True)
            big = {k for k, c in zip(names, counts) if c >= min_industry and k != POOL}
            grp = np.array([v if v in big else POOL for v in ind[:b]], dtype=object)
            past = np.isfinite(x[:b])
            today = np.arange(a, b)[np.isfinite(x[a:b])]
            for k in set(grp[today]):
                sel = past & (grp == k)
                idx = today[grp[today] == k]
                out[idx] = _midrank_pct(np.sort(x[:b][sel]), x[idx])
        return out

    def narrative(self, y, x1, x2):
        """Residual of y on [1, x1, x2] from an OLS over the calls so far (all three finite)."""
        y, x1, x2 = (np.asarray(v, float) for v in (y, x1, x2))
        out = np.full(self.n, np.nan)
        ok = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
        X = np.column_stack([np.ones(self.n), x1, x2])
        for a, b in self.bounds:
            sel = ok[:b]
            if sel.sum() < self.min_calls:
                continue
            beta, *_ = np.linalg.lstsq(X[:b][sel], y[:b][sel], rcond=None)
            today = np.arange(a, b)[ok[a:b]]
            out[today] = y[today] - X[today] @ beta
        return out


def legs(pct, q):
    """+1 long, -1 short, 0 not traded."""
    p = np.asarray(pct, float)
    out = np.zeros(len(p))
    out[p >= 1 - 1 / q] = 1
    out[p < 1 / q] = -1
    out[~np.isfinite(p)] = 0
    return out
