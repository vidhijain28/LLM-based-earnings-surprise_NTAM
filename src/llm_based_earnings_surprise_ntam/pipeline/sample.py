"""Estimation sample for the LLM step (extension; see PIPELINE.md section 5).

Joins the event-date returns onto the final dataset, flags the estimation sample, re-ranks the surprise and
the abnormal return within the event quarter on those rows, and runs the surprise-only baseline regression

    PRank(r_abn) = a + b * PRank(sue)

whose R-squared is about 5% in Koijen & Levy. A much lower value means something upstream is misaligned
(event date, ticker match, price basis or returns), so the task prints a warning.

The columns are added to the existing processed parquet in place; existing columns and row order are
untouched, and rerunning replaces the added columns rather than duplicating them.
"""
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

NEW_COLS = ['permno', 'ret', 'mkt_ret', 'r_abn', 'in_sample', 'r_abn_pct', 'sue_pct_sample']
BASELINE_R2_WARN = 0.05
NEEDED = ['keydevid', 'matched', 'sue', 'event_quarter', 'transcript_by_event_close']


def has_columns(final_path):
    """doit uptodate check: are the extension columns already in the parquet?"""
    try:
        names = pq.read_schema(final_path).names
    except (OSError, pa.ArrowInvalid):
        return False
    return all(c in names for c in NEW_COLS)


def _ols(y, x):
    """Slope, R-squared and t-statistic of y on x with an intercept (single regressor: R2 = corr^2)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    n = len(y)
    if n < 3 or np.std(x) == 0:
        return np.nan, np.nan, np.nan
    r = float(np.corrcoef(x, y)[0, 1])
    slope = r * y.std(ddof=1) / x.std(ddof=1)
    r2 = r ** 2
    t = r * np.sqrt((n - 2) / max(1 - r2, 1e-12))
    return slope, r2, t


def build(cfg, final_path, returns_path):
    c = cfg.get('sample') or {}
    need_text_ready = bool(c.get('require_transcript_by_event_close', False))
    warn_at = float(c.get('baseline_r2_warn', BASELINE_R2_WARN))

    tbl = pq.read_table(final_path)
    base = tbl.select([c_ for c_ in NEEDED if c_ in tbl.column_names]).to_pandas()
    rets = pd.read_parquet(returns_path)
    df = base.merge(rets.drop(columns=['event_date', 'n_permno_candidates'], errors='ignore'),
                    on='keydevid', how='left')

    log = {'rows': len(df)}
    keep = df.matched.fillna(False).astype(bool)
    log['matched'] = int(keep.sum())
    keep &= df.sue.notna()
    log['with SUE'] = int(keep.sum())
    keep &= df.r_abn.notna()
    log['with return'] = int(keep.sum())
    if need_text_ready:
        keep &= df.transcript_by_event_close.fillna(False).astype(bool)
        log['transcript by event close'] = int(keep.sum())
    df['in_sample'] = keep

    # within-quarter percentile ranks, over the estimation sample only
    qtr = df.event_quarter.where(keep)
    df['r_abn_pct'] = df.r_abn.where(keep).groupby(qtr).rank(pct=True)
    df['sue_pct_sample'] = df.sue.where(keep).groupby(qtr).rank(pct=True)

    s = df[keep]
    slope, r2, t = _ols(s.r_abn_pct, s.sue_pct_sample)

    # replace, rather than duplicate, the added columns; row order of the parquet is preserved
    tbl = tbl.drop_columns([col for col in NEW_COLS if col in tbl.column_names])
    for col in NEW_COLS:
        tbl = tbl.append_column(col, pa.array(df[col].to_numpy()))
    tmp = f'{final_path}.tmp'
    pq.write_table(tbl, tmp, compression='snappy')
    os.replace(tmp, final_path)

    print('Sample: ' + ' -> '.join(f'{k} {v:,}' for k, v in log.items()))
    print(f'Baseline (surprise only): PRank(r_abn) = a + b*PRank(sue), N={len(s):,}, '
          f'b={slope:.3f} (t={t:.1f}), R2={r2:.1%}')
    if not (r2 >= warn_at):
        print(f'WARNING: surprise-only R2 of {r2:.1%} is below {warn_at:.1%}. Koijen & Levy report 5.0%; a lower '
              'value usually means something upstream is misaligned (event date, ticker match, price basis or '
              'returns). Check these before relying on the LLM results.')
    print(f'Added columns {", ".join(NEW_COLS)} -> {final_path}')
