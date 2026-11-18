"""Statistics on the grid's results, following the logic of Khanna's grid-is-the-alpha (harness_audit.py):

    Sharpe            mean / sd of a configuration's daily returns on its active days, annualized by sqrt(252)
    decomposition     balanced sum-of-squares split of the Sharpes: each setting's main effect, each pair's
                      interaction, and the rest (higher-order interactions)
    N_eff             participation ratio (sum lambda)^2 / sum lambda^2 of the eigenvalues of the configurations'
                      correlation matrix (common window, 0 when flat); Li and Ji (2005) as a second estimate
    expected max      Bailey and Lopez de Prado (2014) eq. (1): the best Sharpe a grid of N independent trials
                      reaches under the null, with the variance of the configurations' Sharpes; at N = 1, N_eff
                      and the number of configurations
    deflated Sharpe   eq. (2): probability that the best configuration's Sharpe beats that expected maximum
    permutation       the LLM outputs shuffled within each entry day (one block per call: every scoring version's score and
                      the persistence move together), the full grid rerun, its best Sharpe recorded
"""
import itertools
import math

import numpy as np
from scipy import stats as st

GAMMA = 0.5772156649015329


def sharpe_table(R, A, ann):
    rows = []
    for r, a in zip(R, A):
        x = r[a]
        sd = x.std(ddof=1) if len(x) > 1 else np.nan
        rows.append({'sharpe': float(x.mean() / sd * math.sqrt(ann)) if sd and sd > 0 else np.nan,
                     'ann_return': float(x.mean() * ann) if len(x) else np.nan,
                     'ann_vol': float(sd * math.sqrt(ann)) if sd == sd else np.nan,
                     'cum_return': float(np.prod(1 + x) - 1) if len(x) else np.nan,
                     'hit_rate': float((x > 0).mean()) if len(x) else np.nan})
    return rows


def decompose(table, factors, col):
    """Shares of the total sum of squares of `col`: main effects, two-way interactions, higher order."""
    y = table[col].to_numpy(float)
    if np.isnan(y).any():
        raise ValueError(f'{int(np.isnan(y).sum())} configurations have no {col}; the split needs every cell')
    mu = y.mean()
    total = float(((y - mu) ** 2).sum())
    main = {}
    for f in factors:
        m = table.groupby(f)[col].transform('mean').to_numpy()
        main[f] = float(((m - mu) ** 2).sum())
    pairs = {}
    for f, g in itertools.combinations(factors, 2):
        cell = table.groupby([f, g])[col].transform('mean').to_numpy()
        mf = table.groupby(f)[col].transform('mean').to_numpy()
        mg = table.groupby(g)[col].transform('mean').to_numpy()
        pairs[f'{f} x {g}'] = float(((cell - mf - mg + mu) ** 2).sum())
    rest = total - sum(main.values()) - sum(pairs.values())
    share = (lambda v: v / total) if total > 0 else (lambda v: 0.0)
    return {'total_ss': total, 'main': {k: share(v) for k, v in main.items()},
            'pairs': {k: share(v) for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])},
            'higher_order': share(rest),
            'level_means': {f: table.groupby(f)[col].mean().to_dict() for f in factors}}


def eigenvalues(Rw):
    Z = Rw - Rw.mean(axis=1, keepdims=True)
    sd = Z.std(axis=1)
    Z = Z[sd > 0] / sd[sd > 0][:, None]
    n, T = Z.shape
    lam = np.clip(np.linalg.eigvalsh((Z.T @ Z) / T), 0.0, None)[::-1]
    return lam * (n / lam.sum())


def n_eff(lam):
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def n_eff_li_ji(lam):
    return float(sum(1.0 for x in lam if x >= 1) + sum(x - math.floor(x) for x in lam if x < 1))


def expected_max_sr(var_sr, n):
    """Eq. (1) per observation under a zero mean; 0 at N = 1 (the expected maximum of one trial is its mean)."""
    if n <= 1:
        return 0.0
    return math.sqrt(max(var_sr, 0.0)) * ((1 - GAMMA) * st.norm.ppf(1 - 1 / n)
                                          + GAMMA * st.norm.ppf(1 - 1 / (n * math.e)))


def deflated_sharpe(sr, var_sr, n, T, skew, kurt):
    """Eq. (2); `sr` per observation, `kurt` non-excess."""
    sr0 = expected_max_sr(var_sr, n)
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    return float(st.norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom))


def _var_sr(table, ann):
    return float((table['sharpe'] / math.sqrt(ann)).var(ddof=1))


def implied_n(measured_max_annual, table, ann):
    """The N at which eq. (1) gives a measured best-of-grid Sharpe (e.g. the permutation null's mean maximum)."""
    v = _var_sr(table, ann)
    lo, hi = 1.0000001, 1e12
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if expected_max_sr(v, mid) * math.sqrt(ann) < measured_max_annual else (lo, mid)
    return float(math.sqrt(lo * hi))


def audit(table, R, A, Rw, ann):
    lam = eigenvalues(Rw)
    ne, nl = n_eff(lam), n_eff_li_ji(lam)
    best = int(table['sharpe'].idxmax())
    x = R[best][A[best]]
    sr = float(table['sharpe'].iloc[best]) / math.sqrt(ann)
    v = _var_sr(table, ann)
    skew, kurt = float(st.skew(x)), float(st.kurtosis(x, fisher=False))
    n_cfg = float(len(table))
    levels = {'N = 1': 1.0, 'N = N_eff': ne, 'N = configurations': n_cfg}
    return {'configurations': len(table), 'best_config': best,
            'sharpe_best': float(table['sharpe'].max()), 'sharpe_median': float(table['sharpe'].median()),
            'sharpe_sd': float(table['sharpe'].std(ddof=1)),
            'share_positive': float((table['sharpe'] > 0).mean()),
            'selection_premium': float(table['sharpe'].max() - table['sharpe'].median()),
            'n_eff': ne, 'n_eff_li_ji': nl, 'eigen_top10_share': float(lam[:10].sum() / lam.sum()),
            'best_active_days': int(A[best].sum()), 'best_skew': skew, 'best_kurtosis': kurt,
            'expected_max_sharpe': {k: expected_max_sr(v, n) * math.sqrt(ann) for k, n in levels.items()},
            'deflated_sharpe': {k: deflated_sharpe(sr, v, n, int(A[best].sum()), skew, kurt)
                                for k, n in levels.items()}}


def permutation_null(cfg, calls, AR, hold, llm, runs, seed, run_grid):
    """Best Sharpe of the full grid on `runs` within-day shuffles of the LLM outputs (all scoring versions + persistence
    together, the same permutation for every version)."""
    rng = np.random.default_rng(seed)
    days = calls['event_date'].to_numpy()
    groups = [np.where(days == d)[0] for d in np.unique(days)]
    out = []
    for k in range(runs):
        perm = np.arange(len(calls))
        for g in groups:
            perm[g] = rng.permutation(g)
        shuffled = {c: np.asarray(v)[perm] for c, v in llm.items()}
        table, _, _ = run_grid(cfg, calls, AR, hold, shuffled)
        out.append(float(table['sharpe'].max()))
        print(f'  permutation {k + 1}/{runs}: best Sharpe {out[-1]:.2f}', flush=True)
    return out
