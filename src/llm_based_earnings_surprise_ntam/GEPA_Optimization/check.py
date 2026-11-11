"""Before the real runs: re-score a few calls with the seed prompt and compare with the data file's predictions.

    python -m llm_based_earnings_surprise_ntam.GEPA_Optimization.check [--config optimize.yaml]

The file's `prediction` column came from the team's Midway run (same model and prompt, but its own context size and
batching). If this setup reproduces those numbers, the file's seed predictions are a fair "before" for the
comparisons (`evaluate.seed_from_file: true`); if not, set it to false so the seed prompt is re-scored here.
The scores are cached, so the optimization reuses them.
"""
import argparse

import numpy as np

from . import config, data
from .llm import Ollama, ScoreCache, Scorer


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default=None, help='optimize.yaml (default: the one next to this file)')
    cfg = config.load(ap.parse_args().config)
    df = data.load(cfg['paths']['data'])
    sample = df.sample(n=min(int(cfg['check']['n_calls']), len(df)), random_state=int(cfg['gepa']['seed']))

    client = Ollama(cfg)
    client.require(cfg['models']['scorer']['model'])
    scorer = Scorer(cfg, client, ScoreCache(cfg['paths']['cache'], writer='check'))
    res = scorer.score(cfg['prompts']['seed_instructions'], data.records(sample))
    here = np.array([np.nan if r['prediction'] is None else r['prediction'] for r in res], float)
    file = sample['prediction'].to_numpy(float)
    ok = ~np.isnan(here) & ~np.isnan(file)
    gap = np.abs(here[ok] - file[ok])
    print(f'{len(sample)} calls re-scored with the seed prompt ({ok.sum()} readable)')
    print(f'  identical to the data file: {(gap < 1e-9).sum()} of {ok.sum()}')
    print(f'  mean |difference| {gap.mean():.3f}, largest {gap.max():.3f}')
    if ok.sum() > 2:
        print(f'  correlation {np.corrcoef(here[ok], file[ok])[0, 1]:.3f}')
    for t, h, f in zip(sample['transcriptid'], here, file):
        print(f'    {t}: here {h:.2f}  file {f:.2f}')


if __name__ == '__main__':
    main()
