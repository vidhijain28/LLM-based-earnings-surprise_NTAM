"""One GEPA search with the `gepa` package: our adapter (score + feedback), the minibatch sampler, the log of every
proposal, and the pick of the prompt that goes forward.

The loss is the centered squared error against the residual u of the surprise-only regression, per call:

    score = -((u - mean u) - (g - mean g))^2        (means over the calls being scored)

Centering removes the bias term: the 0-1 prediction g averages about 0.65 while u averages 0, so the plain squared
error is ~80% bias and GEPA would mostly learn to lower every prediction. What is left rewards ranking the calls the
way the residual does. The prompt that goes forward is picked by R² of y on g and s on the selection set.
"""
import json
import random

import gepa
from gepa.core.adapter import EvaluationBatch
import numpy as np

from .evaluate import predictions, r2
from .llm import reflector

COMPONENT = 'instructions'


class ResidualAdapter:
    """GEPA adapter: runs a prompt on calls, scores it against the residual, and writes the feedback for reflection."""

    propose_new_texts = None    # use GEPA's own reflection prompt

    def __init__(self, scorer, band):
        self.scorer, self.band = scorer, band

    def evaluate(self, batch, candidate, capture_traces=False):
        results = self.scorer.score(candidate[COMPONENT], batch)
        g = np.array([np.nan if r['prediction'] is None else r['prediction'] for r in results], float)
        ok = ~np.isnan(g)
        g_mean = float(g[ok].mean()) if ok.any() else 0.5
        gc = np.where(ok, g - g_mean, 0.0)              # an unreadable reply = an average prediction
        u = np.array([c['u'] for c in batch], float)
        uc = u - u.mean()
        err = gc - uc
        outputs = [{'transcriptid': c['transcriptid'], **r} for c, r in zip(batch, results)]
        traces = None
        if capture_traces:
            traces = [{'call': c, 'result': r, 'g_mean': g_mean, 'gc': float(a), 'uc': float(b), 'err': float(e)}
                      for c, r, a, b, e in zip(batch, results, gc, uc, err)]
        return EvaluationBatch(outputs=outputs, scores=[float(-e * e) for e in err], trajectories=traces)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = [{'Inputs': {'Earnings call summary': t['call']['summary']},
                    'Generated Outputs': {'prediction': t['result']['prediction'], 'reasons': t['result']['reasons']},
                    'Feedback': feedback(t, self.band)} for t in eval_batch.trajectories]
        return {name: records for name in components_to_update}


def feedback(t, band):
    """What the reflector reads about one call: the outcome, what the surprise explained, and how the prediction did."""
    c, r = t['call'], t['result']
    text = (f"The stock's realized one-day abnormal return ranked {c['y']:.2f} among the calls known so far "
            f"(0 = worst, 1 = best). The earnings surprise alone (surprise rank {c['s']:.2f}) implied {c['y_base']:.2f}, "
            f"so the stock did {c['u']:+.2f} versus what the surprise explains. Relative to the other calls in this "
            f"batch that is {t['uc']:+.2f}, and that is what the prediction is judged on.")
    if r['status'] != 'ok':
        return text + ' The response was not the required JSON object, so it counted as an average prediction.'
    e = t['err']
    verdict = 'about right' if abs(e) <= band else (f'too optimistic by {e:.2f}' if e > 0 else f'too pessimistic by {-e:.2f}')
    return text + (f" The prediction was {r['prediction']:.2f}, {t['gc']:+.2f} versus the average prediction in this "
                   f"batch ({t['g_mean']:.2f}). Verdict: {verdict}.")


class RecentFirstSampler:
    """Minibatches with a fixed share of calls announced since the previous refit (continuous version).

    Draws depend only on (seed, iteration), so a resumed search draws the same minibatches.
    """

    def __init__(self, size, new_ids, share, seed):
        self.size, self.new, self.share, self.seed = int(size), set(new_ids), float(share), seed
        self._iteration, self._calls = None, 0

    def next_minibatch_ids(self, loader, state):
        if state.i == self._iteration:
            self._calls += 1
        else:
            self._iteration, self._calls = state.i, 0
        ids = list(loader.all_ids())
        new = [i for i in ids if i in self.new]
        old = [i for i in ids if i not in self.new]
        rng = random.Random(f'{self.seed}-{state.i}-{self._calls}')
        k_new = min(len(new), round(self.share * self.size))
        k_old = min(len(old), self.size - k_new)
        k_new = min(len(new), self.size - k_old)
        return rng.sample(new, k_new) + rng.sample(old, k_old)


class ProposalLog:
    """GEPA callback: one JSON line per round (parent, minibatch, the proposed prompt, scores, accepted or rejected),
    rejected proposals included, plus a one-line progress message. After a resume the interrupted round may repeat."""

    def __init__(self, path, trainset):
        self.path, self.trainset, self.cur = path, trainset, {}

    def on_iteration_start(self, ev):
        self.cur = {'round': ev['iteration']}

    def on_candidate_selected(self, ev):
        self.cur.update(parent=ev['candidate_idx'], parent_selection_score=ev['score'])

    def on_minibatch_sampled(self, ev):
        self.cur['minibatch'] = [self.trainset[i]['transcriptid'] for i in ev['minibatch_ids']]

    def on_proposal_end(self, ev):
        self.cur['proposed'] = ev['new_instructions'].get(COMPONENT)
        self.cur['reflector_reply'] = ev['raw_lm_outputs'].get(COMPONENT)

    def on_candidate_accepted(self, ev):
        self.cur.update(outcome='accepted', candidate=ev['new_candidate_idx'], minibatch_score=ev['new_score'])

    def on_candidate_rejected(self, ev):
        self.cur.update(outcome='rejected', parent_minibatch_score=ev['old_score'],
                        minibatch_score=ev['new_score'], reason=ev['reason'])

    def on_budget_updated(self, ev):
        self.cur['calls_used'] = ev.get('metric_calls_used')

    def on_iteration_end(self, ev):
        if not self.cur:
            return
        with open(self.path, 'a') as f:
            f.write(json.dumps(self.cur, default=str) + '\n')
        print(f"  round {self.cur['round']}: parent {self.cur.get('parent')} -> {self.cur.get('outcome', 'no proposal')}"
              f"{' as candidate ' + str(self.cur['candidate']) if 'candidate' in self.cur else ''}", flush=True)
        self.cur = {}


class FileLogger:
    """GEPA's detailed log (every per-call score) to gepa/run_log.txt only, so the job log stays readable."""

    def __init__(self, path):
        self.path = path

    def log(self, *args, **kwargs):
        with open(self.path, 'a') as f:
            print(*args, file=f, **kwargs)


def run(cfg, client, scorer, train, val, seed_instructions, folder, budget, sampler=None):
    """Run (or resume) GEPA in folder/gepa; returns gepa's result with every accepted candidate."""
    g = cfg['gepa']
    (folder / 'gepa').mkdir(exist_ok=True)
    kwargs = dict(seed_candidate={COMPONENT: seed_instructions}, trainset=train, valset=val,
                  adapter=ResidualAdapter(scorer, float(cfg['objective']['feedback_band'])),
                  reflection_lm=reflector(cfg, client), candidate_selection_strategy=g['candidate_selection'],
                  skip_perfect_score=False, max_metric_calls=int(budget), run_dir=str(folder / 'gepa'),
                  logger=FileLogger(folder / 'gepa' / 'run_log.txt'),
                  callbacks=[ProposalLog(folder / 'proposals.jsonl', train)], seed=int(g['seed']),
                  use_merge=False, track_best_outputs=False, raise_on_exception=True)
    if sampler is None:
        kwargs['reflection_minibatch_size'] = int(g['minibatch'])
    else:
        kwargs['batch_sampler'] = sampler
    return gepa.optimize(**kwargs)


def candidates(result, scorer, val):
    """Every accepted candidate with its selection-set scores (all cached: GEPA scored each one on the full set)."""
    y, s = [c['y'] for c in val], [c['s'] for c in val]
    rows = []
    for i, cand in enumerate(result.candidates):
        g = predictions(scorer.score(cand[COMPONENT], val))
        rows.append({'candidate': i, 'parents': json.dumps(result.parents[i]),
                     'prompt_key': scorer.prompt_key(cand[COMPONENT]),
                     'selection_mean_score': float(result.val_aggregate_scores[i]),
                     'selection_r2': r2(y, g, s), 'selection_mean_prediction': float(g.mean()),
                     'instructions': cand[COMPONENT]})
    return rows


def choose(rows, margin):
    """The candidate with the best selection-set R², if it beats the starting prompt (candidate 0) by `margin`."""
    best = max(rows, key=lambda r: r['selection_r2'])
    return best if best['candidate'] != 0 and best['selection_r2'] >= rows[0]['selection_r2'] + margin else rows[0]
