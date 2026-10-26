"""doit pipeline: CIQ earnings-call transcripts -> IBES surprises -> merged dataset.

    uv run doit                                        # period and universe from config.yaml
    uv run doit start=2024-01-01 end=2024-12-31        # override the period
    uv run doit universe=global                        # all countries, IBES US + international files
    uv run doit config=other.yaml start=2023-01-01     # another config file
    uv run doit list                                   # show tasks;  `doit info merge` shows why a task would run

Tasks (each reruns only when its inputs or the relevant config section change):
    transcripts:YYYY-MM   pull one month of earnings calls from WRDS (chosen version + text), cached in data/raw
    transcripts_clean     combine months, keep the period, drop short / duplicate calls
    ibes                  IBES announcements with actual EPS and pre-release consensus
    links                 candidate IBES tickers for each call (CUSIP, CCM, Compustat ibtic, SEDOL)
    merge                 match calls to announcements, event date
    prices                pre-announcement price: close on the trading day before the event date (Compustat)
    final                 SUE, ranks, transcript text -> final parquet + report
    returns               event-date abnormal return (CRSP daily v2, market proxy VTI)
    sample                join returns onto the final parquet, flag the estimation sample, re-rank, baseline check
Monthly transcript files are reused across runs; to refresh a month that was still incomplete when pulled:
    uv run doit forget transcripts:2025-12 && uv run doit
"""
import json

from doit import get_var
from doit.tools import config_changed

from llm_based_earnings_surprise_ntam.pipeline import ibes, links, merge, prices, returns, sample, transcripts
from llm_based_earnings_surprise_ntam.pipeline.config import load_config, path, paths

DOIT_CONFIG = {'default_tasks': ['sample'], 'verbosity': 2, 'dep_file': '.doit.db'}

CFG = load_config(get_var('config', 'config.yaml'), start=get_var('start', None), end=get_var('end', None),
                  universe=get_var('universe', None))
P = paths(CFG)
P['returns'] = path(CFG, 'interim', f"returns_{CFG['tag']}.parquet")


def _sig(*keys):
    """Config signature for uptodate checks: only the sections a task actually uses."""
    return json.dumps({k: CFG[k] for k in keys}, sort_keys=True, default=str)


def task_transcripts():
    """Pull one month of CIQ earnings calls (chosen version + full text)."""
    for month, target in P['raw_transcripts'].items():
        yield {
            'name': month,
            'actions': [(transcripts.pull_month, [CFG, month, target])],
            'targets': [target],
            'uptodate': [True],  # file name already encodes universe + version rule; rerun via `doit forget`
            'clean': True,
        }


def task_transcripts_clean():
    """Combine monthly pulls, restrict to the period, drop short and duplicate calls."""
    chunks = list(P['raw_transcripts'].values())
    return {
        'actions': [(transcripts.clean, [CFG, chunks, P['calls'], P['text']])],
        'file_dep': chunks,
        'targets': [P['calls'], P['text']],
        'uptodate': [config_changed(_sig('period', 'transcripts'))],
        'clean': True,
    }


def task_ibes():
    """IBES announcements in the period with actual EPS and the pre-release consensus."""
    return {
        'actions': [(ibes.pull, [CFG, P['ibes']])],
        'targets': [P['ibes']],
        'uptodate': [config_changed(_sig('period', 'universe', 'ibes', 'match'))],
        'clean': True,
    }


def task_links():
    """Candidate IBES tickers for each call."""
    return {
        'actions': [(links.build, [CFG, P['calls'], P['links']])],
        'file_dep': [P['calls']],
        'targets': [P['links']],
        'clean': True,
    }


def task_merge():
    """Match calls to IBES announcements and set the event date."""
    return {
        'actions': [(merge.build, [CFG, P['calls'], P['links'], P['ibes'], P['matched']])],
        'file_dep': [P['calls'], P['links'], P['ibes']],
        'targets': [P['matched']],
        'uptodate': [config_changed(_sig('match'))],
        'clean': True,
    }


def task_prices():
    """Close on the trading day before the event date, for the SUE denominator."""
    return {
        'actions': [(prices.pull, [CFG, P['matched'], P['prices']])],
        'file_dep': [P['matched']],
        'targets': [P['prices']],
        'clean': True,
    }


def task_final():
    """SUE with the pre-announcement price, within-quarter ranks, transcript text; final parquet + report."""
    return {
        'actions': [(merge.finalize, [CFG, P['matched'], P['prices'], P['text'], P['ibes'], P['final'], P['report']])],
        'file_dep': [P['matched'], P['prices'], P['text'], P['ibes']],
        'targets': [P['final'], P['report']],
        'uptodate': [config_changed(_sig('ibes', 'output'))],
        'clean': True,
    }


def task_returns():
    """Event-date abnormal return: stock return minus market return (CRSP daily v2, market proxy VTI)."""
    return {
        'actions': [(returns.pull, [CFG, P['matched'], P['prices'], P['returns']])],
        'file_dep': [P['matched'], P['prices']],
        'targets': [P['returns']],
        'uptodate': [config_changed(_sig('returns'))],
        'clean': True,
    }


def task_sample():
    """Add returns, the estimation-sample flag and within-quarter ranks to the final parquet; baseline check."""
    return {
        'actions': [(sample.build, [CFG, P['final'], P['returns']])],
        'file_dep': [P['returns']],
        'task_dep': ['final'],  # the columns are added to final's output, so it must be written first
        'uptodate': [config_changed(_sig('sample')), lambda: sample.has_columns(P['final'])],
    }
