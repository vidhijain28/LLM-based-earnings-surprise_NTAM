"""Stage gate 1 first look: earnings-surprise long-short CARs at 1, 4 and 12 weeks, Q4 2025.

Runs on Hazel's cleaned files (events_baseline.parquet, prices_clean.parquet, market_ret.parquet), which are
licensed WRDS data and stay outside this public repo. From the repo root:

    uv run python scripts/long_short_q4_2025.py ../data/processed ../meetings/2026-10-28_week2 \
        --pipeline data/processed/earnings_calls_ibes_20251001_20251231_us.parquet

`--pipeline` (optional) adds the same test on the pipeline's in-sample calls, the calls the LLM scores, with the
pipeline's analyst-level SUE as the signal. Writes to the output folder:
    long_short_results.csv     main table plus robustness checks (one row per check and window)
    long_short_large_cap_stress.csv   is the large-cap 1-week spread carried by a few names or one week?
    long_short_path.csv        the charts' numbers: cumulative spread and standard error by day
    long_short_chart_all.png   chart: all events, days 0-20
    long_short_chart.png       chart: all events vs large caps, days 0-20 (PIPELINE.md, section 6)
    long_short_chart_12w.png   backup chart: companies that reported by Nov 3, days 0-60

Signal: the surprise as a percent of the pre-announcement price, sorted into fifths within the sample. Returns:
Compustat daily (total return where a dividend factor exists) minus VTI. The prices end on 2026-01-30, so the
12-week window exists only for events up to 2025-11-03. Compustat market caps are wrong for some ADRs (share
counts at the local-share level), so size cuts use US-incorporated firms only and nothing is value-weighted.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from llm_based_earnings_surprise_ntam.signals.long_short import (  # noqa: E402
    event_paths, long_short, quantile, quantile_pit, spread_path, window_return)

INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE = '#2463a6', '#eb6834'  # validated pair (dataviz palette check: CVD dE 22, contrast >= 3:1)
SPLIT = {'day 1': (1, 1), 'days 2-5': (2, 5)}
LARGE = 10e9


def load(folder):
    folder = Path(folder)
    ev = pd.read_parquet(folder / 'events_baseline.parquet')
    px = pd.read_parquet(folder / 'prices_clean.parquet',
                         columns=['gvkey', 'iid', 'datadate', 'ret', 'mcap_lag1', 'fic'])
    mk = pd.read_parquet(folder / 'market_ret.parquet')
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(mk.datadate).unique()))
    market = mk.assign(datadate=pd.to_datetime(mk.datadate)).set_index('datadate').mkt_ret
    px['key'] = px.gvkey.astype(str) + '|' + px.iid.astype(str)
    px['datadate'] = pd.to_datetime(px.datadate)
    returns = px.set_index(['key', 'datadate']).ret.unstack().reindex(columns=cal)
    ev['key'] = ev.gvkey.astype(str) + '|' + ev.iid.astype(str)
    ev['event_date'] = pd.to_datetime(ev.event_date)
    ev = ev.merge(px[['key', 'datadate', 'mcap_lag1']].rename(columns={'datadate': 'event_date'}),
                  on=['key', 'event_date'], how='left')
    ev = ev.merge(px.drop_duplicates('key')[['key', 'fic']], on='key', how='left')
    ev['us'] = ev.fic.eq('USA')
    ev['large'] = ev.us & (ev.mcap_lag1 >= LARGE)
    return ev, returns, market


def load_pipeline(path, ev_hazel, px_path):
    """The pipeline's in-sample calls (the calls the LLM scores), keyed to the primary Compustat line on the
    event date so they can use the same daily returns. Signal: the pipeline's SUE (analyst-level consensus)."""
    cols = ['keydevid', 'gvkey', 'event_date', 'sue', 'in_sample']
    d = pd.read_parquet(path, columns=cols)
    d = d[d.in_sample.fillna(False).astype(bool)].copy()
    d['gvkey'] = d.gvkey.astype(str).str.zfill(6)
    d['event_date'] = pd.to_datetime(d.event_date)
    px = pd.read_parquet(px_path, columns=['gvkey', 'iid', 'datadate', 'primary_pit', 'mcap_lag1', 'fic'])
    px = px[px.primary_pit.fillna(False).astype(bool)].copy()
    px['key'] = px.gvkey.astype(str) + '|' + px.iid.astype(str)  # same key as the return panel in load()
    px['gvkey'] = px.gvkey.astype(str).str.zfill(6)
    px['datadate'] = pd.to_datetime(px.datadate)
    d = d.merge(px.drop(columns='iid').rename(columns={'datadate': 'event_date'}), on=['gvkey', 'event_date'],
                how='inner')
    d['large'] = d.fic.eq('USA') & (d.mcap_lag1 >= LARGE)
    return d.drop_duplicates('keydevid').reset_index(drop=True)


def run_checks(ev, stock, mkt):
    """Main specification first, then one change at a time. Subsamples are re-sorted into fifths."""
    sig, dates = ev.surprise_pct, ev.event_date
    main = dict(q=quantile(sig), groups=dates, signal=sig)
    sub = lambda mask: dict(main, q=quantile(sig.where(mask)))  # noqa: E731
    checks = [
        ('main', main),
        ('days 1 and 2-5 separately', dict(main, windows=SPLIT)),
        ('tradable cut-offs (earlier events only)', dict(main, q=quantile_pit(sig, dates))),
        ('buy-and-hold returns', dict(main, compound=True)),
        ('no winsorizing', dict(main, winsor=0)),
        ('clustered by reporting week', dict(main, groups=ev.week)),
        ('deciles (top minus bottom tenth)', dict(main, q=quantile(sig, 10), n=10)),
        ('price >= $5', sub(ev.p_pre >= 5)),
        ('2+ analysts', sub(~ev.single_analyst.astype(bool))),
        ('US-incorporated', sub(ev.us)),
        ('US, market cap >= $2bn', sub(ev.us & (ev.mcap_lag1 >= 2e9))),
        ('US large caps, market cap >= $10bn', sub(ev.large)),
        ('US large caps, days 1 and 2-5 separately', dict(sub(ev.large), windows=SPLIT)),
        ('US large caps, clustered by reporting week', dict(sub(ev.large), groups=ev.week)),
    ]
    return pd.concat([long_short(stock, mkt, **kw).assign(check=label) for label, kw in checks], ignore_index=True)


def stress_test(ev, stock, mkt, mask, start=1, end=5):
    """Is a subsample's spread carried by a few names or one week? Mean vs median, trimmed legs, weekly spreads."""
    q = quantile(ev.surprise_pct.where(mask))
    d = ev.assign(q=q, r=window_return(stock, mkt, start, end))[mask & q.isin([1, 5])]
    top, bot = d[d.q == 5].r.sort_values(), d[d.q == 1].r.sort_values()
    rows = [('mean spread', top.mean() - bot.mean()), ('median spread', top.median() - bot.median())]
    rows += [(f'drop {k} best and {k} worst per side', top.iloc[k:-k].mean() - bot.iloc[k:-k].mean())
             for k in (3, 5, 10)]
    wk = d.groupby(['week', 'q']).r.agg(['mean', 'size']).unstack('q')
    wk = wk[(wk['size'][1] >= 5) & (wk['size'][5] >= 5)]
    rows += [(f'reporting week {w} ({int(wk["size"][5][w])} long / {int(wk["size"][1][w])} short)',
              wk['mean'][5][w] - wk['mean'][1][w]) for w in wk.index]
    return pd.DataFrame(rows, columns=['test', 'spread'])


def pct(v):
    return f'{v:+.1f}%'.replace('-', '−')


def fitted_text(fig, x, y, s, size, **kw):
    """Figure text that shrinks until it fits inside the figure's right edge."""
    while True:
        t = fig.text(x, y, s, fontsize=size, **kw)
        fig.canvas.draw()
        if t.get_window_extent().transformed(fig.transFigure.inverted()).x1 <= 0.985 or size <= 6:
            return t
        t.remove()
        size -= 0.25


def plot(lines, title, subtitle, xmax, xticks, notes, out_png):
    """lines: (path, color, legend label, {day: label offset in points})."""
    plt.rcParams['font.family'] = ['Segoe UI', 'DejaVu Sans']
    fig = plt.figure(figsize=(6.0, 4.8), dpi=250, facecolor='white')
    left = 0.115
    ax = fig.add_axes([left, 0.205, 0.83, 0.535])
    lo, hi = 0.0, 0.0
    for p, color, _, _ in lines:
        p = p[p.day <= xmax]
        x, y, band = p.day.to_numpy(), 100 * p.spread.to_numpy(), 196 * p.se.to_numpy()
        lo, hi = min(lo, (y - band).min()), max(hi, (y + band).max())
        ax.fill_between(x, y - band, y + band, color=color, alpha=0.10, linewidth=0)
        ax.plot(x, y, color=color, linewidth=2, solid_capstyle='round', solid_joinstyle='round')
    for p, color, _, labels in lines:
        for day, dy in labels.items():
            y = 100 * float(p.loc[p.day == day, 'spread'].iloc[0])
            ax.scatter([day], [y], s=46, color=color, edgecolor='white', linewidth=2, zorder=5)
            ax.annotate(pct(y), (day, y), xytext=(0, dy), textcoords='offset points', ha='center',
                        va='bottom' if dy > 0 else 'top', fontsize=10.5, color=INK, fontweight='bold')
    ax.axhline(0, color=AXIS, linewidth=1)
    lo, hi = np.floor(lo), np.ceil(hi)
    step = 1 if hi - lo <= 7 else 2
    ticks = np.arange(step * np.floor(lo / step), hi + 0.01, step)
    ax.set_ylim(ticks[0], max(ticks[-1], hi))
    ax.set_yticks(ticks, [pct(v).replace('.0', '') if v else '0' for v in ticks])
    ax.set_xlim(-0.5, xmax + 0.8)
    ax.set_xticks(list(xticks), list(xticks.values()))
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=10.5, length=0, pad=6)
    ax.set_xlabel('Trading days after the announcement (enter at the day-0 close)', fontsize=10.5, color=INK2,
                  labelpad=8)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(False)

    fitted_text(fig, left, 0.935, title, 14, fontweight='bold', color=INK)
    fitted_text(fig, left, 0.885, subtitle, 11.5, color=INK2)
    x = left
    for _, color, label, _ in (lines if len(lines) > 1 else []):  # one series: the title names it
        fig.lines.append(plt.Line2D([x, x + 0.035], [0.828, 0.828], transform=fig.transFigure, color=color,
                                    linewidth=2.5, solid_capstyle='round'))
        t = fig.text(x + 0.047, 0.828, label, fontsize=10.5, color=INK, va='center')
        fig.canvas.draw()
        x = t.get_window_extent().transformed(fig.transFigure.inverted()).x1 + 0.045
    for note, y in zip(notes, [0.058, 0.025]):
        fitted_text(fig, left, y, note, 8.5, color=MUTED)
    fig.savefig(out_png, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('data_dir', help="folder with Hazel's events_baseline / prices_clean / market_ret parquet")
    parser.add_argument('out_dir', help='where to write the results, the charts and their data')
    parser.add_argument('--pipeline', help="also run on the pipeline's in-sample calls (final parquet from doit)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ev, returns, market = load(args.data_dir)
    stock, mkt, filled = event_paths(ev.key, ev.event_date, returns, market, horizon=60)
    day0_gap = float(np.max(np.abs((stock[:, 0] - mkt[:, 0]) - ev.abn_ret.to_numpy())))
    print(f"{len(ev):,} events ({int(ev.us.sum()):,} US-incorporated, {int(ev.large.sum()):,} US large caps); "
          f"day-0 abnormal return matches Hazel's abn_ret to {day0_gap:.1e}; "
          f"{int(filled[:, 1:].sum()):,} missing daily returns inside windows set to the market")

    res = run_checks(ev, stock, mkt)
    if args.pipeline:
        pl = load_pipeline(args.pipeline, ev, Path(args.data_dir) / 'prices_clean.parquet')
        s_pl, m_pl, _ = event_paths(pl.key, pl.event_date, returns, market, horizon=60)
        print(f'pipeline: {len(pl):,} in-sample calls keyed to Compustat ({int(pl.large.sum()):,} US large caps)')
        res = pd.concat([res] + [
            long_short(s_pl, m_pl, quantile(pl.sue.where(mask)), pl.event_date, signal=pl.sue).assign(check=label)
            for label, mask in [('pipeline calls with transcripts (analyst-level SUE)', pl.sue.notna()),
                                ('pipeline calls, US large caps', pl.large)]], ignore_index=True)
    res.to_csv(out / 'long_short_results.csv', index=False)
    stress = stress_test(ev, stock, mkt, ev.large)
    stress.to_csv(out / 'long_short_large_cap_stress.csv', index=False)
    get = lambda check, window, col='spread': float(  # noqa: E731
        res.loc[(res.check == check) & (res.window == window), col].iloc[0])

    groups, early = ev.event_date, ~np.isnan(stock[:, 60])
    q_all, q_large = quantile(ev.surprise_pct), quantile(ev.surprise_pct.where(ev.large))
    paths = {
        'all events, days 0-20': spread_path(stock, mkt, q_all, groups, horizon=20),
        'US large caps, days 0-20': spread_path(stock, mkt, q_large, groups, horizon=20),
        'all events reported by Nov 3, days 0-60': spread_path(stock[early], mkt[early], q_all[early].to_numpy(),
                                                               groups[early], horizon=60),
        'US large caps reported by Nov 3, days 0-60': spread_path(
            stock[early], mkt[early], q_large[early].to_numpy(), groups[early], horizon=60),
    }
    pd.concat([p.assign(sample=k) for k, p in paths.items()]).to_csv(out / 'long_short_path.csv', index=False)

    n_large = int(ev.large.sum())
    plot([(paths['all events, days 0-20'], BLUE, f'All {len(ev):,} events', {5: 9, 20: 9})],
         'After day 0, about 1% more in week one, then flat',
         f'All {len(ev):,} events: top fifth minus bottom fifth by surprise, vs VTI',
         20, {0: '0', 5: '1 wk', 10: '2 wk', 15: '3 wk', 20: '4 wk'},
         ['Shaded: 95% band, clustered by event date. Day 0 (not tradable): '
          f"{pct(100 * get('main', 'day 0'))}.", "Source: I/B/E/S and Compustat daily via WRDS (Hazel's cleaned "
                                                  'Q4 2025 files).'],
         out / 'long_short_chart_all.png')
    plot([(paths['all events, days 0-20'], BLUE, f'All {len(ev):,} events', {5: -9, 20: -9}),
          (paths['US large caps, days 0-20'], ORANGE, f'{n_large} large caps (US, ≥ $10bn)', {5: 9, 20: 9})],
         'After day 0: about 1% overall, 2–2.5% in large caps',
         'Top fifth minus bottom fifth by surprise, abnormal return vs VTI',
         20, {0: '0', 5: '1 wk', 10: '2 wk', 15: '3 wk', 20: '4 wk'},
         ['Shaded: 95% band, clustered by event date. Day 0 (not tradable): '
          f"{pct(100 * get('main', 'day 0'))} all, {pct(100 * get('US large caps, market cap >= $10bn', 'day 0'))} "
          'large caps.', "Source: I/B/E/S and Compustat daily via WRDS (Hazel's cleaned Q4 2025 files)."],
         out / 'long_short_chart.png')
    n_early_large = int((early & ev.large.to_numpy()).sum())
    plot([(paths['all events reported by Nov 3, days 0-60'], BLUE, f'All {int(early.sum()):,} events', {60: -9}),
          (paths['US large caps reported by Nov 3, days 0-60'], ORANGE, f'{n_early_large} large caps (US, ≥ $10bn)',
           {60: 9})],
         '12 weeks out, the gap is too noisy to call',
         'Companies that reported by Nov 3 (prices end Jan 30)',
         60, {0: '0', 5: '1 wk', 20: '4 wk', 40: '8 wk', 60: '12 wk'},
         ['Shaded: 95% confidence band, clustered by event date. Same signal and returns as the main chart.',
          "Source: I/B/E/S and Compustat daily via WRDS (Hazel's cleaned Q4 2025 files)."],
         out / 'long_short_chart_12w.png')

    pd.set_option('display.width', 220)
    show = res.assign(spread_bp=(1e4 * res.spread).round(0), t=res.t.round(2), r2_pct=(100 * res.rank_r2).round(2))
    print(show[['check', 'window', 'N', 'n_long', 'n_short', 'spread_bp', 't', 'r2_pct']].to_string(index=False))
    print('\nquintile means, bp (main):')
    print((1e4 * res[res.check == 'main'].set_index('window')[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'all']]).round(0)
          .to_string())
    print('\nUS large caps, 1-week spread stress test:')
    print(stress.assign(spread_bp=(1e4 * stress.spread).round(0))[['test', 'spread_bp']].to_string(index=False))
    print(f'\nwrote {out / "long_short_results.csv"}, long_short_large_cap_stress.csv, long_short_path.csv, '
          'long_short_chart_all.png, long_short_chart.png, long_short_chart_12w.png')


if __name__ == '__main__':
    main()
