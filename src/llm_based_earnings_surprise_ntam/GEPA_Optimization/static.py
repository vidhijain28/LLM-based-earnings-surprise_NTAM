"""Static GEPA (the paper's setup): optimize the prompt on the early half of the quarter, report on the late half.

    python -m llm_based_earnings_surprise_ntam.GEPA_Optimization.static [--config optimize.yaml] [--run NAME]

Early half = event_date <= the median event date. Inside it: targets from the early half only, GEPA's training set
(older calls) and selection set (newest). The prompt GEPA picks then scores the late half, which it never saw, and
is compared with the seed prompt on those same calls. Rerunning the same command resumes: GEPA from its saved state,
every scored call from the cache. Output in runs/static/<NAME>/.
"""
import argparse
import json

import pandas as pd

from . import config, data, evaluate, search
from .llm import Ollama, ScoreCache, Scorer


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default=None, help='optimize.yaml (default: the one next to this file)')
    ap.add_argument('--run', default='default', help='run name; the same name resumes the run')
    args = ap.parse_args()
    cfg = config.load(args.config)
    folder = config.open_run(cfg, 'static', args.run)

    df = data.load(cfg['paths']['data'])
    cut = data.cutoff(df, cfg['static']['cutoff'])
    early, late = df[df['event_date'] <= cut], df[df['event_date'] > cut]
    train, val, info = data.training_set(early, **cfg['split'])
    info.update(cutoff=f'{cut:%Y-%m-%d}', test_calls=len(late))
    (folder / 'split.json').write_text(json.dumps(info, indent=1))
    print(f"Static GEPA, run {folder}\n  train on {info['calls']:,} calls to {info['cutoff']} "
          f"({info['train_calls']:,} for minibatches, {info['selection_calls']:,} for selection from "
          f"{info['selection_first_date']}); test on {len(late):,} later calls", flush=True)

    client = Ollama(cfg)
    client.require(cfg['models']['scorer']['model'], cfg['models']['reflector']['model'])
    scorer = Scorer(cfg, client, ScoreCache(cfg['paths']['cache'], writer=f'static-{args.run}'))

    seed = cfg['prompts']['seed_instructions']
    result = search.run(cfg, client, scorer, train, val, seed, folder, cfg['static']['budget'])
    rows = search.candidates(result, scorer, val)
    chosen = search.choose(rows, float(cfg['objective']['selection_margin']))
    pd.DataFrame(rows).to_csv(folder / 'candidates.csv', index=False)
    (folder / 'production_prompt.txt').write_text(chosen['instructions'])
    print(f"  {len(rows)} candidates; selection-set R² seed {rows[0]['selection_r2']:.3f}, "
          f"chosen candidate {chosen['candidate']} {chosen['selection_r2']:.3f}", flush=True)

    # late half: the chosen prompt vs. the seed prompt on the same calls
    calls = data.records(late)
    from_file = bool(cfg['evaluate']['seed_from_file'])
    seed_g = late['prediction'].to_numpy(float) if from_file else evaluate.predictions(scorer.score(seed, calls))
    if chosen['candidate'] == 0:
        opt_res = None
        opt_g = seed_g
    else:
        opt_res = scorer.score(chosen['instructions'], calls)
        opt_g = evaluate.predictions(opt_res)
    out = late[['transcriptid', 'event_date', 'r_abn_pct', 'sue_pct_sample']].assign(
        seed_prediction=seed_g, prediction=opt_g, prompt_key=chosen['prompt_key'],
        status=[r['status'] for r in opt_res] if opt_res else 'seed')
    out.to_parquet(folder / 'test_predictions.parquet', index=False)

    ev = cfg['evaluate']
    res = evaluate.compare(late['r_abn_pct'], late['sue_pct_sample'], {'seed': seed_g, 'optimized': opt_g},
                           ev['bootstrap'], ev['seed'])
    report = {'split': info, 'chosen_candidate': chosen['candidate'], 'chosen_prompt_key': chosen['prompt_key'],
              'candidates': len(rows), 'metric_calls_used': result.total_metric_calls,
              'seed_predictions': 'data file' if from_file else 'rescored', 'test': res}
    (folder / 'report.json').write_text(json.dumps(report, indent=1))
    lines = evaluate.table(f'Late half ({len(late):,} calls after {info["cutoff"]}), day-0 regressions:', res)
    (folder / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
