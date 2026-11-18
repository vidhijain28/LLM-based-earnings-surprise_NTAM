"""Each call's earlier calls, and the two history pieces built from them: their summaries and one line of numbers.

An earlier call is a call of the same company (gvkey) in one of the `history_quarters` event quarters before the call's
own, announced before it. When the company has several calls in that quarter, the latest one is used. Every piece
lists the quarters oldest first ("3 quarters ago", "2 quarters ago", "Last quarter"); a quarter without a call says
"not available". Nothing after the call is used: the earlier calls are announced before it, and each one's surprise
rank is within its own quarter.
"""
import math

import pandas as pd


def earlier_calls(df, calls, n_quarters):
    """{transcriptid: [row or None, ...]}, oldest first: element i is the call n_quarters - i quarters before."""
    df = df.assign(_q=pd.PeriodIndex(df['event_quarter'], freq='Q'))
    by_company = {g: grp.sort_values(['event_date', 'transcriptid']) for g, grp in df.groupby('gvkey')}
    out = {}
    for c in calls.itertuples(index=False):
        quarter = pd.Period(c.event_quarter, freq='Q')
        mine = by_company.get(c.gvkey)
        rows = []
        for k in range(n_quarters, 0, -1):
            m = None if mine is None else mine[(mine['_q'] == quarter - k) & (mine['event_date'] < c.event_date)]
            rows.append(None if m is None or m.empty else m.iloc[-1])
        out[c.transcriptid] = rows
    return out


def label(k):
    return 'Last quarter' if k == 1 else f'{k} quarters ago'


def _money(x):
    return f'-${-x:.2f}' if x < 0 else f'${x:.2f}'


def _ordinal(n):
    suffix = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def summaries(rows):
    """The earlier calls' summaries, each under its label."""
    if all(r is None for r in rows):
        return 'No earlier calls available.'
    n = len(rows)
    return '\n\n'.join(f'{label(n - i)}:\n' + ('not available' if r is None else str(r['summary']).strip())
                       for i, r in enumerate(rows))


def numbers(rows):
    """One line: EPS vs consensus, beat/miss and the surprise's percentile rank in its quarter, per earlier quarter."""
    n = len(rows)
    parts, beats, known = [], 0, 0
    for i, r in enumerate(rows):
        if r is None:
            parts.append(f'{label(n - i)}: not available.')
            continue
        s = float(r['surprise'])
        word = 'beat' if s > 0 else 'miss' if s < 0 else 'in line'
        pct = min(100, max(1, math.ceil(100 * float(r['sue_pct_sample']))))
        parts.append(f'{label(n - i)}: EPS {_money(float(r["actual"]))} vs consensus {_money(float(r["consensus"]))}, '
                     f'{word} (surprise in the {_ordinal(pct)} percentile of reporting firms).')
        known += 1
        beats += s > 0
    if not known:
        return 'Earlier results: none available.'
    tail = (f'Beat in {beats} of the last {n} quarters.' if known == n
            else f'Beat in {beats} of the {known} earlier quarters available.')
    return 'Earlier results (oldest first): ' + ' '.join(parts) + ' ' + tail


PIECES = {'summaries': summaries, 'numbers': numbers}
