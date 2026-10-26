"""Load config.yaml, apply command-line overrides, and derive file paths."""
from pathlib import Path

import pandas as pd
import wrds
import yaml

ROOT = Path(__file__).resolve().parents[3]
UNIVERSES = ('us', 'global')


def load_config(path='config.yaml', start=None, end=None, universe=None):
    path = Path(path)
    cfg = yaml.safe_load((path if path.is_absolute() else ROOT / path).read_text())
    if start is not None:
        cfg['period']['start'] = start
    if end is not None:
        cfg['period']['end'] = end
    if universe is not None:
        cfg['universe'] = universe

    start, end = pd.Timestamp(str(cfg['period']['start'])), pd.Timestamp(str(cfg['period']['end']))
    if end < start:
        raise ValueError(f'period.end ({end.date()}) is before period.start ({start.date()})')
    if cfg['universe'] not in UNIVERSES:
        raise ValueError(f"universe must be one of {UNIVERSES}, got {cfg['universe']!r}")
    if cfg['transcripts']['version_rule'] not in ('pit', 'final'):
        raise ValueError("transcripts.version_rule must be 'pit' or 'final'")
    if cfg['ibes']['consensus'] not in ('summary', 'detail'):
        raise ValueError("ibes.consensus must be 'summary' or 'detail'")

    cfg['period'] = {'start': str(start.date()), 'end': str(end.date())}
    cfg['tag'] = f"{start:%Y%m%d}_{end:%Y%m%d}_{cfg['universe']}"
    cfg['paths'] = {k: str(Path(v) if Path(v).is_absolute() else ROOT / v) for k, v in cfg['paths'].items()}
    return cfg


def months(cfg):
    """Calendar months ('YYYY-MM') touched by the period."""
    return [str(p) for p in pd.period_range(cfg['period']['start'], cfg['period']['end'], freq='M')]


def path(cfg, kind, name):
    p = Path(cfg['paths'][kind]) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def paths(cfg):
    """Every file the pipeline reads or writes, keyed by role."""
    tag, t = cfg['tag'], cfg['transcripts']
    return {
        'raw_transcripts': {m: path(cfg, 'raw', f"ciq_transcripts_{cfg['universe']}_{t['version_rule']}_{m}.parquet")
                            for m in months(cfg)},
        'calls': path(cfg, 'interim', f'calls_{tag}.parquet'),
        'text': path(cfg, 'interim', f'transcript_text_{tag}.parquet'),
        'ibes': path(cfg, 'interim', f'ibes_announcements_{tag}.parquet'),
        'links': path(cfg, 'interim', f'ibes_links_{tag}.parquet'),
        'final': path(cfg, 'processed', f'earnings_calls_ibes_{tag}.parquet'),
        'report': path(cfg, 'processed', f'earnings_calls_ibes_{tag}_report.json'),
    }


def connect(cfg):
    user = cfg.get('wrds_username')
    return wrds.Connection(wrds_username=user) if user else wrds.Connection()
