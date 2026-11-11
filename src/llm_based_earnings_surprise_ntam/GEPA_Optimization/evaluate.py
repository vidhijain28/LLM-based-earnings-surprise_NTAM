"""The day-0 regressions of the paper (Tables 2-3), with paired bootstrap intervals.

    y = PRank(abnormal return), s = PRank(surprise), g = the prompt's prediction (0-1, not re-ranked)
    surprise only:  y ~ s          LLM + surprise:  y ~ g + s          ΔR² = R²(g + s) - R²(s)

Every prompt is evaluated on the same calls and the same bootstrap draws, so the gain of the optimized prompt over the
seed prompt has a paired interval: the direct test of whether optimization helped.
"""
import numpy as np


def r2(y, *xs):
    """R² of the OLS of y on xs, with an intercept."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in xs])
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    d = y - y.mean()
    return float(1 - e @ e / (d @ d))


def predictions(results):
    """Scorer results -> array; an unreadable reply counts as the prompt's average prediction (no information)."""
    g = np.array([np.nan if r['prediction'] is None else r['prediction'] for r in results], float)
    return np.where(np.isnan(g), np.nanmean(g) if (~np.isnan(g)).any() else 0.5, g)


def _ci(x):
    lo, hi = np.percentile(x, [0.5, 99.5])
    return [float(lo), float(hi)]


def compare(y, s, preds, bootstrap, seed, base='seed'):
    """R² of the surprise alone, then for each prompt: alone, with the surprise, ΔR² (99% CI), and the paired gain over
    `base`. `preds` maps a prompt name to its predictions on the same calls."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    preds = {k: np.asarray(v, float) for k, v in preds.items()}
    n = len(y)
    out = {'calls': n, 'surprise_r2': r2(y, s), 'prompts': {}}
    if n < 10:
        return out
    idx = np.random.default_rng(seed).integers(0, n, size=(int(bootstrap), n))
    base_r2 = np.array([r2(y[i], s[i]) for i in idx])
    gains = {}
    for name, g in preds.items():
        both = r2(y, g, s)
        draws = np.array([r2(y[i], g[i], s[i]) for i in idx]) - base_r2
        gains[name] = draws
        out['prompts'][name] = {'r2_alone': r2(y, g), 'r2_with_surprise': both, 'delta_r2': both - out['surprise_r2'],
                                'delta_r2_ci99': _ci(draws), 'mean_prediction': float(g.mean()),
                                'sd_prediction': float(g.std())}
    for name in preds:
        if name != base and base in preds:
            diff = gains[name] - gains[base]
            p = out['prompts'][name]
            p['gain_over_' + base] = p['delta_r2'] - out['prompts'][base]['delta_r2']
            p['gain_ci99'] = _ci(diff)
            p['gain_share_positive'] = float((diff > 0).mean())
    return out


def table(title, res):
    """A compare() result as text lines."""
    lines = [title, f'  calls: {res["calls"]:,}   surprise only: R² = {res["surprise_r2"]:.3f}']
    for name, p in res['prompts'].items():
        lo, hi = p['delta_r2_ci99']
        line = (f'  {name:<10} alone {p["r2_alone"]:.3f} | with surprise {p["r2_with_surprise"]:.3f} | '
                f'ΔR² {p["delta_r2"]:+.3f} [{lo:+.3f}, {hi:+.3f}] | mean prediction {p["mean_prediction"]:.3f}')
        lines.append(line)
        for key in [k for k in p if k.startswith('gain_over_')]:
            glo, ghi = p['gain_ci99']
            lines.append(f'  {"":<10} gain over {key[10:]}: {p[key]:+.3f} [{glo:+.3f}, {ghi:+.3f}], '
                         f'positive in {p["gain_share_positive"]:.0%} of resamples')
    return lines
