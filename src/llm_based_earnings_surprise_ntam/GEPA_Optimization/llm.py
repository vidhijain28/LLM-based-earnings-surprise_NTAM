"""Talking to Ollama: the HTTP client, the scored-call cache, the scorer and the reflector.

Uses Ollama's REST API directly (no extra package). The scorer sends exactly what the LLM step's scorer sends:
the fixed system prompt, then the instructions, the fixed output format and the call's summary, with temperature 0
and JSON output. Every result is cached on disk by (prompt, call), so a resumed or repeated search never pays twice.
"""
import hashlib
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class LLMError(RuntimeError):
    """Ollama did not answer after all retries: stop the run (it resumes from the cache and GEPA's saved state)."""


class Ollama:
    def __init__(self, cfg):
        o = cfg['ollama']
        host = o.get('host') or os.environ.get('OLLAMA_HOST') or '127.0.0.1:11434'
        self.host = (host if host.startswith(('http://', 'https://')) else f'http://{host}').rstrip('/')
        self.num_ctx = int(o['num_ctx'])
        self.parallel = int(o['num_parallel'])
        self.timeout = float(o['timeout_seconds'])
        self.retries = int(o['retries'])

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.host + path, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def require(self, *models):
        """Fail early, with a clear message, when the server or a model is missing."""
        try:
            tags = self._request('/api/tags')
        except (urllib.error.URLError, OSError) as e:
            raise SystemExit(f'Ollama is not answering at {self.host} ({e}). Run through run_optimize.sh on Midway.')
        have = {m.get(k) for m in tags.get('models', []) for k in ('name', 'model')}
        missing = [m for m in models if m not in have and f'{m}:latest' not in have]
        if missing:
            raise SystemExit(f'Ollama at {self.host} has no model {missing}; models it has: {sorted(have - {None})}')

    def chat(self, role, messages, json_mode=False):
        """One chat completion for a role from optimize.yaml (`models.scorer` / `models.reflector`)."""
        options = {'temperature': role['temperature'], 'num_ctx': self.num_ctx}
        if role.get('seed') is not None:
            options['seed'] = int(role['seed'])
        payload = {'model': role['model'], 'messages': messages, 'stream': False,
                   'think': bool(role.get('think', False)), 'options': options}
        if json_mode:
            payload['format'] = 'json'
        last = None
        for attempt in range(self.retries):
            try:
                return clean(self._request('/api/chat', payload)['message']['content'])
            except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
                last = e
                time.sleep(2 ** attempt)
        raise LLMError(f'Ollama failed {self.retries} times at {self.host}: {last}')


def clean(raw):
    """Strip qwen3 reasoning: <think>...</think> blocks, and anything before a lone </think>."""
    out = re.sub(r'<think>.*?</think>', '', raw or '', flags=re.S)
    return out.split('</think>')[-1].strip()


def parse_score(raw):
    """The scorer's JSON reply -> (prediction clamped to [0, 1], reasons); raises if it is not usable."""
    obj = json.loads(raw[raw.find('{'): raw.rfind('}') + 1])
    p = float(obj['prediction'])
    if math.isnan(p):
        raise ValueError('prediction is NaN')
    reasons = obj.get('reasons', [])
    reasons = [str(r) for r in reasons] if isinstance(reasons, list) else [str(reasons)]
    return min(max(p, 0.0), 1.0), reasons


class ScoreCache:
    """Every scored (prompt, call) pair, kept on disk.

    Each process appends to its own file (scores-<writer>.jsonl), so jobs running at the same time never write to the
    same file; every process reads all files at start. The prompts themselves are stored once in prompts/<key>.json.
    """

    def __init__(self, folder, writer):
        self.folder = Path(folder)
        (self.folder / 'prompts').mkdir(parents=True, exist_ok=True)
        self.entries = {}
        for f in sorted(self.folder.glob('scores-*.jsonl')):
            for line in f.read_text().splitlines():
                try:
                    e = json.loads(line)
                except ValueError:          # a line cut off by a killed job
                    continue
                self.entries[(e['prompt'], e['transcriptid'])] = e['result']
        self.path = self.folder / f'scores-{writer}.jsonl'
        self.lock = threading.Lock()

    def get(self, prompt, transcriptid):
        return self.entries.get((prompt, transcriptid))

    def put(self, prompt, transcriptid, result):
        with self.lock:
            with open(self.path, 'a') as f:
                f.write(json.dumps({'prompt': prompt, 'transcriptid': transcriptid, 'result': result}) + '\n')
            self.entries[(prompt, transcriptid)] = result

    def save_prompt(self, key, spec):
        path = self.folder / 'prompts' / f'{key}.json'
        if not path.exists():
            path.write_text(json.dumps(spec, indent=1))


class Scorer:
    """Instructions + calls -> one result per call: {'prediction', 'reasons', 'status'} (status 'ok' or 'unparseable')."""

    def __init__(self, cfg, client, cache):
        p = cfg['prompts']
        self.system, self.output_format = p['system'], p['output_format']
        self.role = cfg['models']['scorer']
        self.client, self.cache = client, cache
        self._keys = {}

    def user_message(self, instructions, summary):
        return instructions.strip() + '\n\n' + self.output_format.replace('{text}', summary)

    def prompt_key(self, instructions):
        """Identifies everything that determines a result except the call: model, settings and the full prompt."""
        if instructions not in self._keys:
            spec = {'model': self.role['model'], 'temperature': self.role['temperature'],
                    'think': bool(self.role.get('think', False)), 'num_ctx': self.client.num_ctx,
                    'system': self.system, 'user_template': self.user_message(instructions, '{text}')}
            key = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
            self.cache.save_prompt(key, spec)
            self._keys[instructions] = key
        return self._keys[instructions]

    def _score_one(self, key, instructions, call):
        messages = [{'role': 'system', 'content': self.system},
                    {'role': 'user', 'content': self.user_message(instructions, call['summary'])}]
        raw = ''
        for _ in range(self.client.retries):
            raw = self.client.chat(self.role, messages, json_mode=True)    # raises LLMError if the server is gone
            try:
                p, reasons = parse_score(raw)
                result = {'prediction': p, 'reasons': reasons, 'status': 'ok'}
                break
            except (ValueError, KeyError, TypeError):
                continue
        else:
            result = {'prediction': None, 'reasons': [], 'status': 'unparseable', 'raw': raw[:500]}
        self.cache.put(key, call['transcriptid'], result)
        return result

    def score(self, instructions, calls):
        key = self.prompt_key(instructions)
        out = [self.cache.get(key, c['transcriptid']) for c in calls]
        todo = [i for i, r in enumerate(out) if r is None]
        if todo:
            with ThreadPoolExecutor(max_workers=self.client.parallel) as pool:
                for i, r in zip(todo, pool.map(lambda i: self._score_one(key, instructions, calls[i]), todo)):
                    out[i] = r
        return out


def reflector(cfg, client):
    """GEPA's reflection model: a function from the reflection prompt to the model's reply."""
    role = dict(cfg['models']['reflector'], seed=cfg['gepa']['seed'])

    def lm(prompt):
        messages = prompt if isinstance(prompt, list) else [{'role': 'user', 'content': prompt}]
        return client.chat(role, messages)
    return lm
