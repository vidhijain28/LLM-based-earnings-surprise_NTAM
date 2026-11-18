"""One self-contained HTML page with the grid's results, for presenting (works offline, charts have hover).

    python -m llm_based_earnings_surprise_ntam.grid_analysis.report [--config grid.yaml]

Reads grid.py's outputs in paths.out and writes paths.report.
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline

from ..GEPA_Optimization.config import load
from .grid import FACTORS

DEFAULT = Path(__file__).with_name('grid.yaml')

# reference palette (light): surfaces, ink, categorical slots, diverging blue <-> gray <-> red
SURFACE, PAGE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#f9f9f7', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
BLUE, RED = '#2a78d6', '#e34948'
DIVERGING = [[0.0, '#e34948'], [0.5, '#f0efec'], [1.0, '#2a78d6']]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

NAMES = {'version': 'Scoring version', 'signal': 'Signal (T1)', 'persistence': 'Persistence (T9)', 'industry': 'Industry (T4)',
         'sort': 'Sort (T2)', 'attention': 'Attention (T7)', 'holding': 'Holding (T5)'}
LEVELS = {
    'version': {'0': 'No history', 'A': 'Past summaries', 'B': 'Past numbers', 'C': 'Summaries + numbers'},
    'signal': {'llm': 'LLM score', 'narrative': 'Narrative surprise'},
    'persistence': {'none': 'Not used', 'weighted': '× persistence'},
    'industry': {'none': 'All calls', 'ff12': 'FF12-neutral'},
    'sort': {'fifths': 'Fifths', 'tenths': 'Tenths'},
    'attention': {'none': 'None', 'low': 'Low attention ×2'},
    'holding': {'fixed_5': 'Fixed 5d', 'fixed_20': 'Fixed 20d', 'fixed_60': 'Fixed 60d',
                'decay_10': 'Decay 10d', 'decay_20': 'Decay 20d', 'decay_45': 'Decay 45d'},
}


def label(f, v):
    return LEVELS[f].get(v, v)


def style(fig, height, **kw):
    layout = dict(height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                  font=dict(family=FONT, color=INK2, size=13), margin=dict(l=60, r=30, t=30, b=50),
                  hoverlabel=dict(bgcolor='#ffffff', bordercolor=AXIS, font=dict(family=FONT, color=INK)),
                  showlegend=False)
    fig.update_layout(**{**layout, **kw})
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    return fig


def to_div(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={'displaylogo': False, 'responsive': True,
                               'modeBarButtonsToRemove': ['lasso2d', 'select2d']})


def config_text(row):
    return ' · '.join(label(f, row[f]) for f in FACTORS)


# ------------------------------------------------------------------ charts

def chart_distribution(t, audit):
    s = t['sharpe']
    fig = go.Figure(go.Histogram(x=s, xbins=dict(size=1), marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
                                 hovertemplate='Sharpe %{x}<br>%{y} configurations<extra></extra>'))
    marks = [('median', float(s.median()), INK2),
             ('mean', float(s.mean()), SERIES[1]),
             ('luck bar at N_eff', audit['expected_max_sharpe']['N = N_eff'], MUTED),
             ('luck bar at 768', audit['expected_max_sharpe']['N = configurations'], MUTED),
             ('best', float(s.max()), RED)]
    for i, (name, x, c) in enumerate(marks):
        fig.add_vline(x=x, line=dict(color=c, width=2,
                                     dash='dot' if 'luck' in name else ('dash' if name == 'mean' else 'solid')))
        fig.add_annotation(x=x, y=1.0 - 0.09 * i, yref='paper', text=f'{name} {x:.1f}', showarrow=False,
                           xanchor='left', xshift=4, font=dict(color=INK2, size=12), bgcolor=SURFACE)
    fig.update_xaxes(title='Annualized Sharpe')
    fig.update_yaxes(title='Configurations')
    return style(fig, 360, bargap=0.04)


def chart_by_holding(t):
    order = list(LEVELS['holding'])
    fig = go.Figure()
    for h in order:
        d = t[t['holding'] == h]
        fig.add_trace(go.Box(y=d['sharpe'], name=label('holding', h), boxpoints='all', jitter=0.5, pointpos=0,
                             marker=dict(color=BLUE, size=5, opacity=0.45), line=dict(color=INK2, width=1.5),
                             fillcolor='rgba(0,0,0,0)', customdata=d.apply(config_text, axis=1),
                             hovertemplate='Sharpe %{y:.2f}<br>%{customdata}<extra></extra>'))
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.update_yaxes(title='Annualized Sharpe')
    return style(fig, 380)


def chart_decomposition(dec):
    items = [(NAMES[k], v, 'main effect') for k, v in dec['main'].items()]
    items += [(' × '.join(NAMES[x].split(' (')[0] for x in k.split(' x ')), v, 'interaction')
              for k, v in list(dec['pairs'].items())[:5]]
    items.append(('All higher-order interactions', dec['higher_order'], 'interaction'))
    pairs_rest = sum(list(dec['pairs'].values())[5:])
    items.append(('Other two-way interactions', pairs_rest, 'interaction'))
    items.sort(key=lambda x: x[1])
    fig = go.Figure(go.Bar(
        y=[i[0] for i in items], x=[100 * i[1] for i in items], orientation='h',
        marker=dict(color=[BLUE if i[2] == 'main effect' else '#86b6ef' for i in items],
                    line=dict(color=SURFACE, width=2)),
        text=[f'{100 * i[1]:.1f}%' for i in items], textposition='outside', textfont=dict(color=INK2),
        customdata=[i[2] for i in items], hovertemplate='%{y}: %{x:.1f}% (%{customdata})<extra></extra>'))
    fig.update_xaxes(title='Share of the spread of the 768 Sharpes (%)', range=[0, 100 * max(i[1] for i in items) * 1.2])
    return style(fig, 440, margin=dict(l=250, r=40, t=20, b=50))


def chart_main_effects(t):
    """One horizontal bar per level, grouped by setting: mean Sharpe of the configurations using that level."""
    ys, xs, groups = [], [], []
    for f in FACTORS:
        m = t.groupby(f)['sharpe'].mean().reindex(list(LEVELS[f]))
        for v, x in m.items():
            ys.append(f'{NAMES[f].split(" (")[0]} · {label(f, v)}')
            xs.append(float(x))
            groups.append(NAMES[f])
    fig = go.Figure(go.Bar(y=ys, x=xs, orientation='h',
                           marker=dict(color=[BLUE if v >= 0 else RED for v in xs], line=dict(color=SURFACE, width=2)),
                           text=[f'{round(v, 2) + 0.0:+.2f}' for v in xs], textposition='outside', textfont=dict(color=INK2, size=12),
                           hovertemplate='%{y}<br>mean Sharpe %{x:.2f}<extra></extra>'))
    fig.add_vline(x=float(t['sharpe'].mean()), line=dict(color=MUTED, width=1, dash='dot'))
    # a hairline between settings
    k = 0
    for f in FACTORS[:-1]:
        k += len(LEVELS[f])
        fig.add_hline(y=k - 0.5, line=dict(color=GRID, width=1))
    lim = max(abs(min(xs)), max(xs)) * 1.25
    fig.update_yaxes(autorange='reversed', gridcolor=SURFACE)
    fig.update_xaxes(title='Mean Sharpe of the configurations with this level', range=[-lim, lim])
    return style(fig, 40 + 26 * len(ys), margin=dict(l=260, r=40, t=10, b=50))


def heatmap(t, rows, title_rows):
    hold = list(LEVELS['holding'])
    z, ylab = [], []
    for f in rows:
        for v in LEVELS[f]:
            z.append([t[(t[f] == v) & (t['holding'] == h)]['sharpe'].mean() for h in hold])
            ylab.append(f'{NAMES[f].split(" (")[0]}: {label(f, v)}' if title_rows else label(f, v))
    z = np.array(z)
    lim = float(np.nanmax(np.abs(z)))
    fig = go.Figure(go.Heatmap(z=z, x=[label('holding', h) for h in hold], y=ylab, zmin=-lim, zmax=lim,
                               colorscale=DIVERGING, xgap=2, ygap=2,
                               text=[[f'{v:.2f}' for v in r] for r in z], texttemplate='%{text}',
                               textfont=dict(color=INK, size=12),
                               colorbar=dict(title='mean Sharpe', thickness=10, outlinewidth=0),
                               hovertemplate='%{y} · %{x}<br>mean Sharpe %{z:.2f}<extra></extra>'))
    fig.update_yaxes(autorange='reversed', gridcolor=SURFACE)
    fig.update_xaxes(side='top', gridcolor=SURFACE)
    return style(fig, 70 + 34 * len(ylab), margin=dict(l=260, r=30, t=60, b=20))


def chart_paths(t, R, dates, picks):
    fig = go.Figure()
    for k, (name, idx) in enumerate(picks):
        cum = np.cumsum(R[idx]) * 100
        fig.add_trace(go.Scatter(x=dates, y=cum, mode='lines', line=dict(color=SERIES[k], width=2), name=name,
                                 hovertemplate=name + '<br>%{x|%b %d}: %{y:.1f}%<extra></extra>'))
        fig.add_annotation(x=dates[-1], y=cum[-1], text=f'{name} {cum[-1]:+.1f}%', showarrow=False, xanchor='left',
                           xshift=6, font=dict(color=INK2, size=12))
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.update_yaxes(title='Cumulative return (%)')
    fig.update_xaxes(range=[dates[0], dates[-1]])
    style(fig, 400, margin=dict(l=60, r=330, t=20, b=40), showlegend=True,
          legend=dict(orientation='h', y=-0.12, font=dict(color=INK2)))
    fig.update_layout(hovermode='x unified')
    return fig


# ------------------------------------------------------------------ page

def table_html(df, cols, fmt):
    head = ''.join(f'<th>{html.escape(c)}</th>' for c in cols)
    body = ''
    for _, r in df.iterrows():
        body += '<tr>' + ''.join(f'<td>{html.escape(fmt.get(c, str)(r[c]))}</td>' for c in cols) + '</tr>'
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def tile(value, name, note=''):
    return (f'<div class="tile"><div class="v">{html.escape(value)}</div><div class="n">{html.escape(name)}</div>'
            f'<div class="s">{html.escape(note)}</div></div>')


IDEA = {   # what each choice decides, in plain words, in the order the steps are applied
    'version': ('Which LLM score', 'What extra information the LLM got when it read the call: nothing, the '
                'summaries of past calls, the past results in numbers, or both.'),
    'signal': ('What we rank', 'Either the LLM score itself, or only the part of it that the earnings numbers and '
               'the first-day price move do not already explain.'),
    'persistence': ('Adjust for one-off news', 'We ask the LLM how much of the result is recurring and how much is '
                    'one-off. The recurring share is a persistence score from 0 to 1, which can shrink one-off '
                    'news.'),
    'industry': ('Compare within industry?', 'Rank each company against all companies, or only against companies '
                 'in the same industry.'),
    'sort': ('Who we buy and sell', 'Buy the best-ranked companies and sell the worst-ranked: the top and bottom '
             '20% (fifths) or 10% (tenths).'),
    'attention': ('How much to bet on each', 'Bet the same on every stock, or bet double on stocks that few '
                  'investors traded on the earnings day.'),
    'holding': ('How long we hold', 'Keep each position for a fixed number of days, or shrink it step by step until '
                'the company reports again.'),
}


def M(body):
    """A displayed formula in MathML (rendered natively by the browser, works offline)."""
    return f'<math display="block">{body}</math>'


def R(x):
    # rank(x): rank from 0 (worst) to 1 (best) among the companies so far
    return f'<mtext>rank</mtext><mo>(</mo>{x}<mo>)</mo>'


R0 = '<msub><mi>r</mi><mn>0</mn></msub>'
MATH = {
    'predicted': M('<mtext>predicted</mtext><mo>=</mo><msub><mi>b</mi><mn>0</mn></msub><mo>+</mo>'
                   '<msub><mi>b</mi><mn>1</mn></msub>' + R('<mtext>SUE</mtext>') + '<mo>+</mo>'
                   '<msub><mi>b</mi><mn>2</mn></msub>' + R(R0)),
    'narrative': M('<mtext>narrative</mtext><mo>=</mo>' + R('<mtext>LLM</mtext>') + '<mo>−</mo>'
                   '<mtext>predicted</mtext>'),
    'persistence': M('<mi>s</mi><mo>=</mo><mo>(</mo>' + R('<mtext>signal</mtext>') + '<mo>−</mo><mn>0.5</mn>'
                     '<mo>)</mo><mo>×</mo><mtext>persistence</mtext>'),
    'sort': M('<mtext>long if </mtext>' + R('<mi>s</mi>') + '<mo>≥</mo><mn>1</mn><mo>−</mo>'
              '<mfrac><mn>1</mn><mi>q</mi></mfrac><mo>,</mo><mspace width="1.2em"/><mtext>short if </mtext>'
              + R('<mi>s</mi>') + '<mo>&lt;</mo><mfrac><mn>1</mn><mi>q</mi></mfrac>'),
    'attention': M('<mtext>attention</mtext><mo>=</mo><mfrac><mtext>dollars traded on the earnings day</mtext>'
                   '<mtext>average dollars traded per day over the past year</mtext></mfrac>'),
    'fixed': M('<msub><mi>w</mi><mi>d</mi></msub><mo>=</mo><mn>1</mn><mspace width="0.6em"/><mtext>for</mtext><mspace width="0.4em"/>'
               '<mi>d</mi><mo>=</mo><mn>1</mn><mo>,</mo><mo>…</mo><mo>,</mo><mi>h</mi>'),
    'decay': M('<msub><mi>w</mi><mi>d</mi></msub><mo>=</mo><msup><mn>0.5</mn><mrow><mo>(</mo><mi>d</mi><mo>−</mo>'
               '<mn>1</mn><mo>)</mo><mo>/</mo><mi>H</mi></mrow></msup>'),
    'sharpe': M('<mtext>Sharpe</mtext><mo>=</mo><mfrac><mover><mi>r</mi><mo>¯</mo></mover><msub><mi>σ</mi>'
                '<mi>r</mi></msub></mfrac><mo>×</mo><msqrt><mn>252</mn></msqrt>'),
}

DETAIL = {   # under each card: formula, small example, and why we test it (HTML, plain English)
    'version': [('Example', 'What the LLM reads besides the latest call summary:<br>'
                 '<b>No history</b>: nothing<br>'
                 '<b>Past summaries</b>: the summaries of the last three calls<br>'
                 '<b>Past numbers</b>: one line: “Earlier results: … EPS $1.39 vs consensus $1.33, beat … Beat in 2 of the '
                 'last 3 quarters.”<br>'
                 '<b>Summaries + numbers</b>: both'),
                ('Why', 'Does knowing the company\'s recent history help the LLM?')],
    'signal': [('Formula', MATH['predicted'] + MATH['narrative'] + 'rank() = rank from 0 (worst) to 1 (best) among '
                'the companies so far; r<sub>0</sub> = the stock\'s move on the earnings day. “predicted” is the LLM '
                'rank that the earnings numbers and the price move alone would suggest; the b\'s are refitted each '
                'day on the companies so far.'),
               ('Example', 'Two companies both get an LLM score of 0.80. One beat expectations by a lot and its '
                'stock jumped 6%: little is left (+0.02). The other was in line and its stock did not move: a lot '
                'is left (+0.25), so it ranks much higher.'),
               ('Why', 'Shows whether the LLM sees something the numbers do not.')],
    'persistence': [('Formula', MATH['persistence'] + 'persistence = the recurring share (0 to 1). Subtracting 0.5 '
                     'makes the middle mean “no news”, so one-off news moves to the middle, not to the sell side.'),
                    ('Example', 'One LLM reply (a real Q4 call; reasons shortened):'
                     '<pre class="json">{\n'
                     '  "recurring": {"share": 0.9,\n'
                     '    "reason": "Growth from organic expansion and cross-selling; guidance raised"},\n'
                     '  "one_off":   {"share": 0.1,\n'
                     '    "reason": "Minimal one-time items; no material charges or benefits"},\n'
                     '  "ai_capex":  {"cited": false,\n'
                     '    "reason": "No mention of AI or capital investment"}\n'
                     '}</pre>Persistence = the recurring share = 0.9.'),
                    ('Why', 'A one-time gain, such as selling a building, should not keep moving the stock.')],
    'industry': [('Example', 'Without it, the buy list might be mostly tech stocks, so we would partly be betting '
                  'on tech rather than on the LLM.'),
                 ('Why', 'Makes sure we test picking stocks, not betting on industries.')],
    'sort': [
             ('Example', 'With 2,349 companies, fifths buy about 470 and sell about 470; tenths about 235 each.'),
             ('Why', 'Tenths keep only the strongest cases: a cleaner signal, but fewer stocks, so more noise.')],
    'attention': [('Formula', MATH['attention'] + 'The lowest third gets double weight.'),
                  ('Example', 'A stock that usually trades $50m a day trades only $25m on its earnings day: '
                   'investors were not paying attention, so it gets double weight.'),
                  ('Why', 'When few people are watching, prices catch up slowly, so there may be more to gain.')],
    'holding': [('Formula', 'Fixed:' + MATH['fixed'] + 'Fading:' + MATH['decay'] + 'w<sub>d</sub> = size of the '
                 'position on day d; H = half-life, the number of days until it is half its size.'),
                ('Example', 'Buy $100 of a stock:<br>'
                 '<b>Fixed 20 days</b>: $100 on every day from day 1 to day 20, then sold.<br>'
                 '<b>Fading, half-life 20 days</b>: $100 on day 1 → $50 on day 21 → $25 on day 41 → sold when the '
                 'company reports again.'),
                ('Why', 'News may fade slowly rather than stop mattering on a set day.')],
}


def config_overview(rules, summary):
    limits = [
        f"<b>One quarter only</b> (calls from Oct to Dec 2025, about {summary['window'][2]} trading days), so every "
        "result is noisy.",
        f"<b>No look-ahead:</b> each company is ranked only against the companies that reported up to the same day, "
        f"and trading starts once {rules['min_calls']} companies have reported ({summary['traded_calls_first_day']}).",
        f"<b>A side (buys or sells) only trades on days it holds at least {rules['min_names']} stocks</b>.",
        "<b>Returns are measured against the whole market</b> (the VTI index fund).",
        "<b>No trading costs and no trimming of extreme returns</b>.",
    ]
    def card(i, f):
        chips = ''.join(f'<span class="chip">{html.escape(v)}</span>' for v in LEVELS[f].values())
        more = ''.join(f'<div class="kv"><span class="tag">{k}</span><span>{v}</span></div>' for k, v in DETAIL[f])
        return (f'<div class="step"><div class="head"><span class="badge">{i}</span>'
                f'<span class="name">{html.escape(NAMES[f])}</span></div>'
                f'<div class="decides">{html.escape(IDEA[f][0])}</div>'
                f'<div class="idea">{html.escape(IDEA[f][1])}</div><div class="chips">{chips}</div>'
                f'<details><summary>{" · ".join(k for k, _ in DETAIL[f]).capitalize()}</summary>{more}</details></div>')
    cards = ''.join(card(i, f) for i, f in enumerate(FACTORS, start=1))
    n = int(np.prod([len(LEVELS[f]) for f in FACTORS]))
    return f"""<section><h2>What a configuration is</h2>
<p class="how">Each strategy we test is a set of 7 choices, made in this order. We try every combination:
{' × '.join(str(len(LEVELS[f])) for f in FACTORS)} = {n} strategies. Only choice 1 needs new LLM work; the other
six reuse the same scores. Click a card for a formula, an example and why we test it.</p>
<div class="steps">{cards}</div>
<h3 class="lim">Main assumptions and limits</h3>
<ul class="find">{''.join(f'<li>{x}</li>' for x in limits)}</ul></section>"""


METRICS = [   # the measures the results are reported in, one plain sentence each (no results)
    ('Sharpe ratio', 'Return per unit of risk: the average daily gain divided by how much the daily gains swing, '
     'scaled to a year.' + MATH['sharpe']),
    ('Variance decomposition', 'How much each choice matters in explaining why some strategies do better than '
     'others.'),
    ('N_eff', 'How many truly different strategies we tested, since many of them behave almost the same.'),
    ('Luck bar (expected best Sharpe)', 'The best Sharpe you would expect from luck alone after trying that many '
     'strategies.'),
    ('Deflated Sharpe', 'The chance that the best Sharpe is above the luck bar.'),
    ('Permutation p-value', 'How often random, meaningless scores do as well as the real LLM scores.'),
]


def metric_guide():
    rows = ''.join(f'<tr><td><b>{html.escape(m)}</b></td><td>{d}</td></tr>' for m, d in METRICS)
    return f"""<section><h2>The metrics we use</h2>
<table class="guide"><thead><tr><th style="width:28%">Metric</th><th>What it means</th></tr></thead>
<tbody>{rows}</tbody></table></section>"""


def _f2(x):
    return f'{round(x, 2) + 0.0:.2f}'


def compare_html(t, lm):
    """What each choice does to the average Sharpe, and where that depends on the holding period."""
    cases = [  # (setting, from level, to level, what the change is, tag if it raises, tag if it lowers)
        ('version', '0', 'A', 'Giving the LLM the summaries of past calls', 'history helps', 'history does not help'),
        ('version', '0', 'B', 'Giving the LLM the past results in numbers', 'history helps', 'history does not help'),
        ('version', '0', 'C', 'Giving the LLM both', 'history helps', 'history does not help'),
        ('persistence', 'none', 'weighted', 'Adjusting for one-off news', 'persistence helps',
         'persistence does not help'),
        ('attention', 'none', 'low', 'Double weight on stocks few investors watched', 'focusing on ignored stocks helps',
         'focusing on ignored stocks does not help'),
        ('industry', 'none', 'ff12', 'Comparing companies only within their industry', 'comparing within industry helps',
         'comparing within industry does not help'),
        ('sort', 'tenths', 'fifths', 'Buying and selling the top and bottom 20% instead of 10%',
         'wider lists work better', 'narrower lists work better'),
    ]
    items = ''
    for f, off, on, text, up, down in cases:
        a, b = lm[f][off], lm[f][on]
        items += (f'<li>{text}: <b>{"raises" if b > a else "lowers"}</b> the average Sharpe from {_f2(a)} to '
                  f'{_f2(b)} <i>({up if b > a else down})</i>.</li>')
    hm = lm['holding']
    others = [v for k, v in hm.items() if k != 'fixed_5']
    items += (f'<li>Holding longer than 5 days: the average Sharpe goes from {_f2(hm["fixed_5"])} to between '
              f'{_f2(min(others))} and {_f2(max(others))} <i>('
              + ('the signal fades within about a week' if hm['fixed_5'] > max(others) else
                 'longer holds can do as well') + ')</i>.</li>')

    # where the effect flips with the holding period
    flips = ''
    for f, off, on, text in [('persistence', 'none', 'weighted', 'Adjusting for one-off news'),
                             ('version', '0', 'C', 'Giving the LLM summaries + numbers'),
                             ('signal', 'llm', 'narrative', 'Using the narrative surprise'),
                             ('attention', 'none', 'low', 'Double weight on ignored stocks')]:
        cell = t.groupby(['holding', f])['sharpe'].mean().unstack()
        d = (cell[on] - cell[off])
        lo, hi = d.idxmin(), d.idxmax()
        if d[lo] < -0.2 and d[hi] > 0.2:      # a real flip, not noise around zero
            flips += (f'<li>{text}: lowers the Sharpe for {label("holding", lo).lower()} holds '
                      f'({_f2(cell.loc[lo, off])} → {_f2(cell.loc[lo, on])}) but raises it for '
                      f'{label("holding", hi).lower()} holds ({_f2(cell.loc[hi, off])} → {_f2(cell.loc[hi, on])}) '
                      f'<i>(it depends on how long you hold)</i>.</li>')
        else:
            word = 'raises or barely changes' if d.mean() > 0 else 'lowers or barely changes'
            flips += (f'<li>{text}: {word} the Sharpe for every holding period; the biggest change is for '
                      f'{label("holding", lo if abs(d[lo]) > abs(d[hi]) else hi).lower()} holds '
                      f'({_f2(d[lo] if abs(d[lo]) > abs(d[hi]) else d[hi])}) <i>(no real change of direction)</i>.</li>')
    return (f'<b>What each choice does on average</b> (average Sharpe before → after the change):'
            f'<ul class="subs">{items}</ul>'
            f'<b>Where it depends on the holding period</b>:<ul class="subs">{flips}</ul>')


def pick(t, **kw):
    m = np.ones(len(t), bool)
    for k, v in kw.items():
        m &= (t[k] == v).to_numpy()
    return int(np.where(m)[0][0])


def build(cfg):
    out = Path(cfg['paths']['out'])
    t = pd.read_parquet(out / 'configs.parquet')
    dr = pd.read_parquet(out / 'daily_returns.parquet')
    R = dr.drop(columns='config').to_numpy()
    dates = pd.to_datetime(dr.drop(columns='config').columns)
    s = json.loads((out / 'summary.json').read_text())
    a, dec, smp = s['audit'], s['decomposition'], s['sample']
    best = t.loc[a['best_config']]
    med_idx = int((t['sharpe'] - t['sharpe'].median()).abs().idxmin())
    base_kw = dict(signal='llm', persistence='none', industry='none', sort='fifths', attention='none',
                   holding='fixed_5')
    picks = [('Best configuration', a['best_config']), ('Median configuration', med_idx),
             ('Plain, no history, 5-day hold', pick(t, version='0', **base_kw)),
             ('Plain, summaries + numbers, 5-day hold', pick(t, version='C', **base_kw))]
    lm = dec['level_means']
    prow = pd.DataFrame([{'Line': name, **{NAMES[f].split(' (')[0]: label(f, t.loc[i, f]) for f in FACTORS},
                          'Sharpe': f"{t.loc[i, 'sharpe']:.2f}"} for name, i in picks])
    paths_table = table_html(prow, list(prow.columns), {})
    ver_means = {k: lm['version'][k] for k in LEVELS['version']}
    hold_means = lm['holding']
    best_hold = max(hold_means, key=hold_means.get)
    perm = s.get('permutation')

    # headline sentences, from the numbers
    findings = [
        f"<b>The holding period drives the results.</b> It explains {100 * dec['main']['holding']:.0f}% of the "
        f"spread of the Sharpes on its own; {label('holding', best_hold)} configurations average "
        f"{hold_means[best_hold]:.1f}, every other holding averages between {min(hold_means.values()):.1f} and "
        f"{sorted(hold_means.values())[-2]:.1f}. Interactions add {100 * (dec['higher_order'] + sum(dec['pairs'].values())):.0f}%.",
        f"<b>History {'helps' if min(ver_means['A'], ver_means['B'], ver_means['C']) > ver_means['0'] else 'does not clearly help'} on average.</b> Mean Sharpe by scoring version: " + ', '.join(f"{label('version', k).lower()} {v:.2f}" for k, v in ver_means.items()) + "; "
        f"but the scoring version explains only "
        f"{100 * dec['main']['version']:.0f}% of the spread, far less than the holding period.",
        (f"<b>The value is mostly in the raw LLM score.</b> Strategies that rank companies on the <b>raw LLM "
         f"score</b> earn an average Sharpe of {_f2(lm['signal']['llm'])}; strategies that rank on the <b>narrative "
         f"surprise</b> earn {_f2(lm['signal']['narrative'])}. SUE and the day-0 return explain only a little of the "
         f"LLM score, but that little is the part that makes money; the rest of the score adds no tradable value on "
         f"average."
         if lm['signal']['llm'] > lm['signal']['narrative'] else
         f"<b>The narrative surprise does at least as well as the raw LLM score.</b> Strategies that rank on the "
         f"narrative surprise earn an average Sharpe of {_f2(lm['signal']['narrative'])}, vs "
         f"{_f2(lm['signal']['llm'])} for the raw score: the words add value of their own."),
        compare_html(t, lm),
    ]

    top = t.sort_values('sharpe', ascending=False).head(12).copy()
    bottom = t.sort_values('sharpe').head(5).copy()
    for d in (top, bottom):
        for f in FACTORS:
            d[f] = d[f].map(lambda v, f=f: label(f, v))
    cols = FACTORS + ['sharpe', 'ann_return', 'active_days', 'open_long_avg', 'open_short_avg']
    fmt = {'sharpe': lambda v: f'{v:.2f}', 'ann_return': lambda v: f'{100 * v:.0f}%',
           'active_days': lambda v: f'{int(v)}', 'open_long_avg': lambda v: f'{v:.0f}',
           'open_short_avg': lambda v: f'{v:.0f}'}
    pretty = {f: NAMES[f] for f in FACTORS}

    def tbl(d):
        d = d.rename(columns={**pretty, 'sharpe': 'Sharpe', 'ann_return': 'Ann. return', 'active_days': 'Days traded',
                              'open_long_avg': 'Avg longs', 'open_short_avg': 'Avg shorts'})
        c = [pretty[f] for f in FACTORS] + ['Sharpe', 'Ann. return', 'Days traded', 'Avg longs', 'Avg shorts']
        f2 = {'Sharpe': fmt['sharpe'], 'Ann. return': fmt['ann_return'], 'Days traded': fmt['active_days'],
              'Avg longs': fmt['open_long_avg'], 'Avg shorts': fmt['open_short_avg']}
        return table_html(d, c, f2)


    plotly_js = plotly.offline.get_plotlyjs()
    css = f"""
    body {{ margin:0; background:{PAGE}; color:{INK}; font-family:{FONT}; }}
    main {{ max-width:1180px; margin:0 auto; padding:32px 28px 60px; }}
    h1 {{ font-size:28px; margin:0 0 4px; }} h2 {{ font-size:20px; margin:0 0 6px; }}
    .sub {{ color:{INK2}; margin:0 0 22px; }}
    section {{ background:{SURFACE}; border:1px solid rgba(11,11,11,0.10); border-radius:10px; padding:20px 22px;
              margin:0 0 18px; }}
    .how {{ color:{INK2}; margin:0 0 10px; font-size:14px; }}
    .tiles {{ display:grid; grid-template-columns:repeat(6, 1fr); gap:12px; margin:0 0 18px; }}
    .tile {{ background:{SURFACE}; border:1px solid rgba(11,11,11,0.10); border-radius:10px; padding:14px 16px; }}
    .tile .v {{ font-size:28px; font-weight:600; color:{INK}; font-variant-numeric:tabular-nums; }}
    .tile .n {{ font-size:13px; color:{INK2}; margin-top:2px; }} .tile .s {{ font-size:12px; color:{MUTED}; }}
    ul.find li {{ margin:0 0 8px; line-height:1.45; }}
    ul.subs {{ margin:6px 0 0; }} ul.subs li {{ margin:0 0 4px; }}
    table {{ border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }}
    th {{ text-align:left; color:{INK2}; font-weight:600; border-bottom:1px solid {AXIS}; padding:6px 8px; }}
    td {{ border-bottom:1px solid {GRID}; padding:5px 8px; color:{INK}; }}
    .note {{ color:{INK2}; font-size:14px; }} code {{ background:#f0efec; padding:1px 4px; border-radius:4px; }}
    .two {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    table.guide td {{ vertical-align:top; line-height:1.4; padding:8px; }} table.guide td.num {{ white-space:nowrap;
              font-size:15px; font-weight:600; }}
    .steps {{ display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:22px; align-items:start; margin:6px 0 4px; }}
    .step {{ position:relative; border:1px solid rgba(11,11,11,0.10); border-radius:10px; padding:12px 14px;
             background:#ffffff; }}
    .step:not(:nth-child(4n)):not(:last-child)::after {{ content:'→'; position:absolute; right:-18px; top:14px;
             color:{MUTED}; font-size:16px; }}
    .step .head {{ display:flex; align-items:center; gap:8px; }}
    .badge {{ display:inline-flex; width:22px; height:22px; border-radius:11px; background:{BLUE}; color:#fff;
              font-size:12px; font-weight:600; align-items:center; justify-content:center; }}
    .step .name {{ font-weight:600; font-size:15px; }}
    .step .decides {{ color:{INK}; font-size:13.5px; margin:6px 0 2px; font-weight:500; }}
    .step .idea {{ color:{INK2}; font-size:12.5px; line-height:1.4; }}
    .chips {{ display:flex; flex-wrap:wrap; gap:5px; margin:9px 0 6px; }}
    .chip {{ background:#f0efec; border-radius:999px; padding:2px 9px; font-size:12px; color:{INK}; }}
    .step details summary {{ cursor:pointer; color:{BLUE}; font-size:12.5px; margin-top:4px; }}
    .step details[open] summary {{ margin-bottom:6px; }}
    math[display=block] {{ margin:4px 0 6px; font-size:14px; color:{INK}; }}
    .step:has(details[open]) {{ grid-column:span 2; }}
    pre.json {{ background:#f0efec; border-radius:6px; padding:8px 10px; margin:4px 0 6px; font-size:11.5px;
               line-height:1.35; white-space:pre-wrap; color:{INK}; }}
    .kv {{ font-size:12.5px; color:{INK2}; line-height:1.4; margin:0 0 8px; }}
    .tag {{ display:block; font-weight:600; color:{INK}; font-size:11.5px; text-transform:uppercase;
            letter-spacing:0.03em; margin-bottom:1px; }}
    @media (max-width: 900px) {{ .steps {{ grid-template-columns:repeat(2, minmax(0, 1fr)); }}
             .step::after {{ display:none; }} }}
    h3.lim {{ font-size:16px; margin:20px 0 6px; }}
    .say {{ background:#f0efec; border-radius:8px; padding:10px 14px; margin:14px 0 8px; line-height:1.45; }}
    @media print {{ section {{ break-inside:avoid; }} }}
    """
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Grid analysis: Q4 2025 LLM earnings signal</title><style>{css}</style>
<script>{plotly_js}</script></head><body><main>
<h1>Grid analysis: is the LLM earnings signal tradable?</h1>
<p class="sub">Q4 2025 US earnings calls · {len(t)} configurations (4 scoring versions × 192 trading settings) · daily
calendar-time long-short portfolios · {smp['calls_kept']:,} calls · generated from grid.yaml</p>

{config_overview(cfg['rules'], s)}

{metric_guide()}

<div class="tiles">
{tile(str(len(t)), 'configurations', '4 scoring versions × 192 settings')}
{tile(f"{a['sharpe_median']:.2f}", 'median Sharpe', f"{a['share_positive'] * 100:.0f}% of configurations > 0")}
{tile(f"{a['sharpe_best']:.1f}", 'best Sharpe', f"{a['best_active_days']} days traded")}
{tile(f"{100 * dec['main']['holding']:.0f}%", 'of the spread from holding', 'the biggest single setting')}
{tile(f"{a['n_eff']:.1f}", 'N_eff (participation)', f"Li–Ji {a['n_eff_li_ji']:.0f}")}
{tile('not run' if not perm else f"{perm['p_value']:.3f}", 'permutation p-value', '1,000 shuffles planned')}
</div>

<section><h2>Key findings</h2><ul class="find">{''.join(f'<li>{x}</li>' for x in findings)}</ul></section>

<section><h2>1 · The Sharpe of every configuration</h2>
<p class="how">Each bar counts configurations by their annualized Sharpe. Gray line: median; dashed orange line: mean; red line: best. Dotted lines: the best Sharpe luck alone
is expected to reach with N_eff and with 768 independent tries (Bailey &amp; López de Prado eq. 1).</p>
{to_div(chart_distribution(t, a))}</section>

<section><h2>2 · Sharpe by holding period</h2>
<p class="how">Each dot is one configuration (128 per holding); hover for its settings. Box = middle half.</p>
{to_div(chart_by_holding(t))}</section>

<section><h2>3 · What drives the spread: variance decomposition</h2>
<p class="how">Share of the total spread (sum of squares) of the 768 Sharpes explained by each setting alone (dark)
and by combinations of settings (light).</p>
{to_div(chart_decomposition(dec))}</section>

<section><h2>4 · The average effect of each setting</h2>
<p class="how">Mean Sharpe of all configurations that use a level; the dotted line is the mean of all 768. Decay = half-life of the position's weight.</p>
{to_div(chart_main_effects(t))}</section>

<section><h2>5 · Does it depend on the holding period?</h2>
<p class="how">Mean Sharpe for each level × holding (blue positive, red negative). This is where the
interactions sit.</p>
{to_div(heatmap(t, ['version', 'signal', 'persistence', 'industry', 'sort', 'attention'], True))}</section>

<section><h2>6 · Return paths</h2>
<p class="how">Cumulative daily long-short return over the common window (0 on days a configuration holds
nothing).</p>
{to_div(chart_paths(t, R, dates, picks))}
{paths_table}</section>

<section><h2>7 · Best and worst configurations</h2>
<p class="how">Top 12</p>{tbl(top)}<p class="how" style="margin-top:16px">Bottom 5</p>{tbl(bottom)}</section>

</main></body></html>"""
    return page


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(DEFAULT))
    args = ap.parse_args()
    cfg = load(args.config)
    path = Path(cfg['paths']['report'])
    path.write_text(build(cfg), encoding='utf-8')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
