"""Continuous GEPA: before each reporting week, re-optimize on every call announced earlier, then score that week.

    python -m llm_based_earnings_surprise_ntam.optimize.continuous [--config optimize.yaml] [--run NAME]

Refit at the start (Monday) of each week once `burn_in` calls are known. A refit uses only calls with
event_date < that Monday: targets rebuilt inside that window, GEPA's selection set = its newest calls, and a share of
each minibatch drawn from calls announced since the previous refit. The search starts from the previous week's
prompt (the package carries one prompt). The prompt it picks is frozen and scores that week's calls, which it never
saw. The stacked weekly predictions are the out-of-sample result, compared with the seed prompt on the same calls.

Each refit is saved in its own folder; rerunning the same command skips finished refits and resumes the current
one. Output in runs/continuous/<NAME>/.
"""
import argparse
import json

import pandas as pd

from . import config, data, evaluate, search
from .llm import Ollama, ScoreCache, Scorer


def schedule(df, burn_in, weeks):
    """The refits: every reporting week with at least `burn_in` earlier calls (or the listed weeks among them)."""
    plan = []
    for d in sorted(df['week'].unique()):
        history = int((df['event_date'] < d).sum())
        if history >= burn_in:
            plan.append({'week': pd.Timestamp(d), 'history': history, 'calls': int((df['week'] == d).sum())})
    if weeks != 'all':
        wanted = {pd.Timestamp(w) for w in weeks}
        unknown = wanted - {p['week'] for p in plan}
        if unknown:
            raise SystemExit(f'continuous.weeks: {_dates(sorted(unknown))} are not refit weeks; '
                             f'refit weeks are {_dates(p["week"] for p in plan)}')
        plan = [p for p in plan if p['week'] in wanted]
    return plan


def _dates(days, fmt='%Y-%m-%d'):
    return ', '.join(d.strftime(fmt) for d in days)


def refit(cfg, df, step, previous, seed, client, scorer, folder):
    """One refit: search on calls before the week, pick, score the week. Returns the week's predictions."""
    d = step['week']
    rdir = folder / f'refit_{d:%Y-%m-%d}'
    rdir.mkdir(exist_ok=True)
    done = rdir / 'done.json'
    if done.exists():
        print(f'Week of {d:%Y-%m-%d}: done earlier', flush=True)
        return pd.read_parquet(rdir / 'week_predictions.parquet'), (rdir / 'production_prompt.txt').read_text()

    train, val, info = data.training_set(df[df['event_date'] < d], **cfg['split'])
    new_ids = [i for i, c in enumerate(train) if previous is not None and c['event_date'] >= f'{previous:%Y-%m-%d}']
    info.update(week=f'{d:%Y-%m-%d}', week_calls=step['calls'], new_calls_in_training_set=len(new_ids),
                starts_from='previous week' if previous is not None else 'seed prompt')
    (rdir / 'split.json').write_text(json.dumps(info, indent=1))
    print(f"Week of {d:%Y-%m-%d}: train on {info['calls']:,} earlier calls ({info['train_calls']:,} minibatch, "
          f"{info['selection_calls']:,} selection, {len(new_ids):,} new since the last refit); "
          f"score {step['calls']:,} calls", flush=True)

    c = cfg['continuous']
    sampler = search.RecentFirstSampler(cfg['gepa']['minibatch'], new_ids, c['new_share'], cfg['gepa']['seed'])
    result = search.run(cfg, client, scorer, train, val, seed, rdir, c['budget'], sampler=sampler)
    rows = search.candidates(result, scorer, val)
    chosen = search.choose(rows, float(cfg['objective']['selection_margin']))
    pd.DataFrame(rows).to_csv(rdir / 'candidates.csv', index=False)
    (rdir / 'production_prompt.txt').write_text(chosen['instructions'])

    week = df[df['week'] == d]
    res = scorer.score(chosen['instructions'], data.records(week))
    out = week[['transcriptid', 'event_date', 'week']].assign(
        refit=d, prompt_key=chosen['prompt_key'], prediction=evaluate.predictions(res),
        status=[r['status'] for r in res], reasons=[json.dumps(r['reasons']) for r in res])
    out.to_parquet(rdir / 'week_predictions.parquet', index=False)
    done.write_text(json.dumps({'chosen_candidate': chosen['candidate'], 'prompt_key': chosen['prompt_key'],
                                'selection_r2_start': rows[0]['selection_r2'],
                                'selection_r2_chosen': chosen['selection_r2'], 'candidates': len(rows),
                                'metric_calls_used': result.total_metric_calls}, indent=1))
    print(f"  {len(rows)} candidates; selection-set R² start {rows[0]['selection_r2']:.3f}, "
          f"chosen candidate {chosen['candidate']} {chosen['selection_r2']:.3f}", flush=True)
    return out, chosen['instructions']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default=None, help='optimize.yaml (default: the one next to this file)')
    ap.add_argument('--run', default='default', help='run name; the same name resumes the run')
    args = ap.parse_args()
    cfg = config.load(args.config)
    folder = config.open_run(cfg, 'continuous', args.run)

    df = data.load(cfg['paths']['data'])
    plan = schedule(df, int(cfg['continuous']['burn_in']), cfg['continuous']['weeks'])
    pd.DataFrame(plan).to_csv(folder / 'schedule.csv', index=False)
    print(f'Continuous GEPA, run {folder}: {len(plan)} refits ({_dates((p["week"] for p in plan), "%b %d")})',
          flush=True)

    client = Ollama(cfg)
    client.require(cfg['models']['scorer']['model'], cfg['models']['reflector']['model'])
    scorer = Scorer(cfg, client, ScoreCache(cfg['paths']['cache'], writer=f'continuous-{args.run}'))

    prompt, previous, weeks = cfg['prompts']['seed_instructions'], None, []
    for step in plan:
        week, prompt = refit(cfg, df, step, previous, prompt, client, scorer, folder)
        weeks.append(week)
        previous = step['week']
    preds = pd.concat(weeks, ignore_index=True)

    # out-of-sample result: the weekly prompts vs. the seed prompt on the same calls
    oos = df.merge(preds[['transcriptid', 'refit', 'prompt_key', 'prediction', 'status']].rename(
        columns={'prediction': 'continuous_prediction'}), on='transcriptid', how='inner')
    from_file = bool(cfg['evaluate']['seed_from_file'])
    seed = cfg['prompts']['seed_instructions']
    oos['seed_prediction'] = (oos['prediction'].to_numpy(float) if from_file
                              else evaluate.predictions(scorer.score(seed, data.records(oos))))
    oos.drop(columns=['summary']).to_parquet(folder / 'predictions.parquet', index=False)

    ev, median = cfg['evaluate'], df['event_date'].median()
    lines, report = [], {'refits': len(plan), 'seed_predictions': 'data file' if from_file else 'rescored'}
    for name, part in [('all', oos), ('late_half', oos[oos['event_date'] > median])]:
        res = evaluate.compare(part['r_abn_pct'], part['sue_pct_sample'],
                               {'seed': part['seed_prediction'], 'continuous': part['continuous_prediction']},
                               ev['bootstrap'], ev['seed'])
        report[name] = res
        title = ('Out-of-sample, all refit weeks' if name == 'all'
                 else f'Late half (after {median:%Y-%m-%d}, comparable with the static run)')
        lines += evaluate.table(f'{title}, day-0 regressions:', res) + ['']
    lines.append('By week (R² with the surprise; weeks under 30 calls not shown):')
    report['by_week'] = []
    for d, part in oos.groupby('refit'):
        row = {'week': f'{d:%Y-%m-%d}', 'calls': len(part)}
        if len(part) >= 30:
            row.update(surprise=evaluate.r2(part['r_abn_pct'], part['sue_pct_sample']),
                       seed=evaluate.r2(part['r_abn_pct'], part['seed_prediction'], part['sue_pct_sample']),
                       continuous=evaluate.r2(part['r_abn_pct'], part['continuous_prediction'], part['sue_pct_sample']))
            lines.append(f"  {row['week']}  {len(part):>5,} calls  surprise {row['surprise']:.3f}  "
                         f"seed {row['seed']:.3f}  continuous {row['continuous']:.3f}")
        report['by_week'].append(row)
    (folder / 'report.json').write_text(json.dumps(report, indent=1))
    (folder / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
