"""Score one run's calls with Ollama and append every result, with the raw replies, to the run's folder.

    python -m llm_based_earnings_surprise_ntam.grid_analysis.score --config <llm_runs.yaml> --run A

Runs: base (run 0), A, B, C, P (llm_runs.yaml). Calls = the in-sample calls of `quarter`. Resumable: calls already in
the run's folder are skipped, so a job that hits its time limit is simply submitted again. Each process appends to its
own file (out/<run>/part-<writer>.jsonl), so jobs for different runs, or a resubmitted job, never share a file.
A call whose reply never parses after all retries is kept as status "unparseable" (with its raw replies) and not
retried. If Ollama stops answering, the run stops without recording that call.
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ..GEPA_Optimization.config import load
from ..GEPA_Optimization.llm import Ollama
from . import history, prompts

DEFAULT = Path(__file__).with_name('llm_runs.yaml')


def load_calls(cfg):
    """(calls of `quarter`, {transcriptid: earlier-call rows}) from the data file's in-sample rows."""
    df = pd.read_parquet(cfg['paths']['data'])
    df = df[df['in_sample'].astype(bool)]
    calls = df[df['event_quarter'] == cfg['quarter']].sort_values('transcriptid').reset_index(drop=True)
    if calls.empty:
        raise SystemExit(f"no in-sample calls with event_quarter {cfg['quarter']} in {cfg['paths']['data']}")
    return calls, history.earlier_calls(df, calls, int(cfg['history_quarters']))


def open_run(cfg, run):
    """out/<run>/, with the run's spec; reopening it resumes the run, but only with the same spec."""
    folder = Path(cfg['paths']['out']) / run
    folder.mkdir(parents=True, exist_ok=True)
    spec = prompts.run_spec(cfg, run)
    now = json.dumps(spec, sort_keys=True, indent=1)
    saved = folder / 'spec.json'
    if saved.exists() and saved.read_text() != now:
        raise SystemExit(f'{folder} was started with a different model, prompt or history (see {saved}). '
                         f'Restore the settings to resume it, or move the folder away to start the run again.')
    saved.write_text(now)
    return folder, spec


def read_results(folder):
    """Every result line in the run's part files (a line cut off by a killed job is skipped)."""
    out = []
    for f in sorted(Path(folder).glob('part-*.jsonl')):
        for line in f.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(DEFAULT))
    ap.add_argument('--run', required=True, help='a run in llm_runs.yaml: base, A, B, C or P')
    args = ap.parse_args()
    cfg = load(args.config)
    if args.run not in cfg['runs']:
        raise SystemExit(f"unknown --run {args.run!r}; runs in {args.config}: {', '.join(cfg['runs'])}")

    calls, earlier = load_calls(cfg)
    folder, spec = open_run(cfg, args.run)
    key = prompts.spec_key(spec)
    done = {r['transcriptid'] for r in read_results(folder)}
    todo = calls[~calls['transcriptid'].isin(done)]
    print(f"run {args.run}: {len(calls)} calls in {cfg['quarter']}, {len(done)} done, {len(todo)} to score "
          f"with {cfg['model']['model']}", flush=True)
    if todo.empty:
        return

    client = Ollama(cfg)
    client.require(cfg['model']['model'])
    writer = os.environ.get('SLURM_JOB_ID') or f'local-{os.getpid()}'
    path = folder / f'part-{writer}.jsonl'
    lock = threading.Lock()
    counter = {'n': 0, 'bad': 0}
    t_start = time.time()

    def score_one(c):
        rows = earlier[c['transcriptid']]
        msgs = prompts.messages(cfg, args.run, str(c['summary']), rows)
        raws, result, status = [], None, 'unparseable'
        t0 = time.time()
        for _ in range(client.retries):
            raw = client.chat(cfg['model'], msgs, json_mode=True)     # raises LLMError if the server is gone
            raws.append(raw)
            try:
                result, status = prompts.parse(spec['kind'], raw), 'ok'
                break
            except (ValueError, KeyError, TypeError):
                continue
        rec = {'transcriptid': int(c['transcriptid']), 'keydevid': int(c['keydevid']), 'gvkey': str(c['gvkey']),
               'event_date': str(c['event_date'])[:10], 'run': args.run, 'spec': key,
               'n_history': sum(r is not None for r in rows), 'status': status, 'attempts': len(raws),
               'raw': raws, 'result': result, 'seconds': round(time.time() - t0, 1)}
        with lock:
            with open(path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            counter['n'] += 1
            counter['bad'] += status != 'ok'
            if counter['n'] % 25 == 0 or counter['n'] == len(todo):
                rate = counter['n'] / max(time.time() - t_start, 1e-9) * 60
                print(f"  {counter['n']}/{len(todo)} scored, {counter['bad']} unparseable, {rate:.1f} calls/min",
                      flush=True)

    with ThreadPoolExecutor(max_workers=client.parallel) as pool:
        list(pool.map(score_one, (r for _, r in todo.iterrows())))
    print(f'run {args.run}: wrote {counter["n"]} results to {path}', flush=True)


if __name__ == '__main__':
    main()
