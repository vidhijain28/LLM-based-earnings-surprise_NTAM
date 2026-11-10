"""optimize.yaml: load it, open a run folder, and print the shell settings run_optimize.sh needs.

    python -m llm_based_earnings_surprise_ntam.optimize.config --shell optimize.yaml   # used by run_optimize.sh
"""
import argparse
import json
import shlex
import shutil
from pathlib import Path

import yaml

DEFAULT = Path(__file__).with_name('optimize.yaml')
# Sections that describe the machine, not the experiment: they may differ when a run is resumed elsewhere.
MACHINE = ('paths', 'ollama')


def load(path=None):
    path = Path(path or DEFAULT)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg['_file'] = str(path.resolve())
    return cfg


def experiment(cfg):
    """The settings that define a run's results (everything but the machine sections)."""
    return {k: v for k, v in cfg.items() if k not in MACHINE and not k.startswith('_')}


def open_run(cfg, kind, name):
    """runs/<kind>/<name>, with a copy of the config. Reopening it resumes the run, but only with the same settings."""
    folder = Path(cfg['paths']['runs']) / kind / name
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder / 'experiment.json'
    now = json.dumps(experiment(cfg), sort_keys=True, default=str, indent=1)
    if saved.exists() and saved.read_text() != now:
        raise SystemExit(f'{folder} was started with different settings (see {saved}). '
                         f'Use a new --run name, or restore the settings to resume it.')
    saved.write_text(now)
    shutil.copy(cfg['_file'], folder / 'optimize.yaml')
    return folder


def shell_exports(cfg):
    o = cfg['ollama']
    return '\n'.join([
        f'export PATH={shlex.quote(str(o["binary_dir"]))}:"$PATH"',
        f'export OLLAMA_MODELS={shlex.quote(str(o["models_dir"]))}',
        f'export OLLAMA_NUM_PARALLEL={int(o["num_parallel"])}',
        'export OLLAMA_NOPRUNE=1',      # never delete model files on startup (the models folder is shared)
    ])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shell', metavar='CONFIG', required=True, help='print export lines for run_optimize.sh')
    print(shell_exports(load(ap.parse_args().shell)))
