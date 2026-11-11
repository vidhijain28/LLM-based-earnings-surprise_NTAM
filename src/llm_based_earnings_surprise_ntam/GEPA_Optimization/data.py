"""The input file and the point-in-time training sets built from it.

The input is the LLM step's merged file (one row per in-sample call). A training set uses only the calls in its window:
return and surprise ranks, the surprise-only regression and its residuals are all recomputed inside the window, so
nothing about later calls leaks into the targets. The final evaluation uses the file's full-quarter ranks instead
(`r_abn_pct`, `sue_pct_sample`), as the paper and the team's results do.
"""
import numpy as np
import pandas as pd

COLUMNS = ['transcriptid', 'event_date', 'event_quarter', 'summary', 'r_abn', 'sue',
           'r_abn_pct', 'sue_pct_sample', 'prediction', 'in_sample']


def load(path):
    df = pd.read_parquet(path, columns=COLUMNS)
    df = df[df['in_sample'].fillna(False).astype(bool)]
    n = len(df)
    df = df.dropna(subset=['summary', 'r_abn', 'sue', 'event_date', 'r_abn_pct', 'sue_pct_sample'])
    if len(df) < n:
        print(f'Data: dropped {n - len(df)} in-sample calls with no summary, return or surprise')
    df['transcriptid'] = df['transcriptid'].astype('int64')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['week'] = df['event_date'].dt.to_period('W-SUN').dt.start_time      # Monday of the reporting week
    return df.sort_values(['event_date', 'transcriptid']).reset_index(drop=True)


def cutoff(df, value):
    """The static split date: the median event date (the paper's rule), or a given date."""
    return df['event_date'].median() if value == 'median' else pd.Timestamp(value)


def records(df):
    """Calls as plain dicts, the form GEPA passes around."""
    keep = [c for c in ('transcriptid', 'summary', 'y', 's', 'y_base', 'u') if c in df.columns]
    out = df[keep].to_dict('records')
    for r, d in zip(out, df['event_date']):
        r['transcriptid'] = int(r['transcriptid'])
        r['event_date'] = f'{d:%Y-%m-%d}'
    return out


def training_set(window, selection_share, selection_max):
    """Targets from `window` alone, then split by date: older calls for minibatches, newest for selection.

    y = return rank, s = surprise rank (within quarter, among the window's calls); y_base = the surprise-only fit
    a + theta*s; u = y - y_base, the part of the return the surprise doesn't explain: what the prompt is trained on.
    """
    w = window.copy()
    q = w.groupby('event_quarter')
    w['y'] = q['r_abn'].rank(pct=True)
    w['s'] = q['sue'].rank(pct=True)
    theta, a = np.polyfit(w['s'], w['y'], 1)
    w['y_base'] = a + theta * w['s']
    w['u'] = w['y'] - w['y_base']

    k = min(int(selection_max), int(round(selection_share * len(w))))
    if not 0 < k < len(w):
        raise SystemExit(f'Training window of {len(w)} calls is too small to split ({k} for selection)')
    train, val = w.iloc[:-k], w.iloc[-k:]          # `window` is sorted by date
    info = {'calls': len(w), 'first_date': f'{w.event_date.min():%Y-%m-%d}', 'last_date': f'{w.event_date.max():%Y-%m-%d}',
            'baseline_a': float(a), 'baseline_theta': float(theta),
            'baseline_r2': float(np.corrcoef(w['s'], w['y'])[0, 1] ** 2),
            'train_calls': len(train), 'selection_calls': len(val),
            'selection_first_date': f'{val.event_date.min():%Y-%m-%d}'}
    return records(train), records(val), info
