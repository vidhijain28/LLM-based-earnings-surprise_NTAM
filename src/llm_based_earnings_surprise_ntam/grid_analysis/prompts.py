"""A run's messages for one call, its identity (everything that determines a result except the call), and the parsers."""
import hashlib
import json
import math

from ..GEPA_Optimization.llm import parse_score
from .history import PIECES


def history_text(cfg, pieces, rows):
    """The run's history pieces for one call, each followed by a blank line; empty for a run without history."""
    out = ''
    for name in pieces:
        out += cfg['history'][name].replace('{' + name + '}', PIECES[name](rows)) + '\n\n'
    return out


def messages(cfg, run, summary, rows):
    spec = cfg['runs'][run]
    p = cfg['prompts'][spec['prompt']]
    user = p['user'].replace('{history}', history_text(cfg, spec['history'], rows)).replace('{text}', summary)
    return [{'role': 'system', 'content': p['system']}, {'role': 'user', 'content': user}]


def run_spec(cfg, run):
    """Model, settings, prompt and history templates of a run; a run folder only ever holds results of one spec."""
    spec = cfg['runs'][run]
    return {'run': run, 'quarter': cfg['quarter'], 'history_quarters': cfg['history_quarters'],
            'model': cfg['model'], 'num_ctx': cfg['ollama']['num_ctx'],
            'prompt': cfg['prompts'][spec['prompt']], 'kind': spec['prompt'],
            'history': {name: cfg['history'][name] for name in spec['history']}, 'history_order': spec['history']}


def spec_key(spec):
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def _share(obj, name):
    part = obj[name]
    v = float(part['share'])
    if math.isnan(v):
        raise ValueError(f'{name} share is NaN')
    return min(max(v, 0.0), 1.0), str(part.get('reason', ''))


def parse_persistence(raw):
    """P's JSON reply -> its fields; raises if it is not usable."""
    obj = json.loads(raw[raw.find('{'): raw.rfind('}') + 1])
    rec, rec_reason = _share(obj, 'recurring')
    one, one_reason = _share(obj, 'one_off')
    cited = obj['ai_capex']['cited']
    if not isinstance(cited, bool):
        raise ValueError('ai_capex.cited is not true/false')
    return {'recurring_share': rec, 'recurring_reason': rec_reason, 'one_off_share': one, 'one_off_reason': one_reason,
            'ai_capex_cited': cited, 'ai_capex_reason': str(obj['ai_capex'].get('reason', ''))}


def parse(kind, raw):
    if kind == 'reaction':
        p, reasons = parse_score(raw)
        return {'prediction': p, 'reasons': reasons}
    return parse_persistence(raw)
