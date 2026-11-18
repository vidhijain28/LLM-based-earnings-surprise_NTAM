# GEPA prompt optimization of the LLM earnings-call scorer

## Question
Can an automatic prompt optimizer (GEPA) rewrite the scoring instructions so the LLM's score explains more of the abnormal return beyond the earnings surprise?

- **Baseline:** Sean's prompt, Qwen3-30B (local, Ollama on a Midway GPU)
- **Data:** 2,584 US earnings calls, Oct–Dec 2025
- **Metric:** regress rank(abnormal return) on the LLM score + rank(surprise)
  - ΔR² = R²(LLM + surprise) − R²(surprise only)

## How GEPA works (one round)
1. Pick a prompt from the pool.
2. Score 5 calls with that prompt.
3. Generate feedback on how each prediction differs from the truth.
4. An LLM (the *reflector*) uses the feedback to rewrite the prompt.
5. Keep the new prompt only if it beats the old one on the same 5 calls.
6. All kept prompts stay in the pool. At the end, **final pick:** the best R² on the held-out **selection set**, if
   it beats the starting prompt by at least 0.005.

## Two versions

| | Static | Continuous |
|---|---|---|
| Optimize | once, on calls up to Nov 5 | every week (9 refits, Oct 27 – Dec 22), on calls before that week |
| Prompt between weeks | — | carried forward and improved |
| Out-of-sample test | 1,141 calls after Nov 5 | each week's calls, scored by a prompt optimized only on calls from before that week |
| Budget | 3,000 scoring calls | 1,500 per refit |
| Run time | 2 h 20 m | 11 h 48 m |

## Results
Out-of-sample. The same 1,141 late-half calls are used for every row.

| Prompt | R² (LLM + surprise) | ΔR² [99% CI] | Gain vs Sean's prompt [99% CI] |
|---|---|---|---|
| Surprise only | 0.039 | — | — |
| Sean's prompt | 0.096 | +0.057 [+0.029, +0.097] | — |
| Static GEPA | 0.086 | +0.047 [+0.022, +0.081] | −0.011 [−0.031, +0.006] (not significant) |
| Continuous GEPA | 0.065 | +0.026 [+0.006, +0.054] | −0.032 [−0.058, −0.008] (significantly worse) |

**Neither optimized prompt beat the original prompt.**

## Why it fell short

**1. The feedback leaked into the prompt.**
The feedback told the reflector the batch-average prediction (e.g. 0.79). The reflector wrote it into the prompt as a
rule:

> *"The average prediction in a batch is 0.79 — this is a hard-coded benchmark. Your prediction must be within ±0.10
> of this value."*

So the LLM gave almost every call the same score: 1,314 of 2,218 continuous scores are exactly 0.79. Static got a
milder rule (*"stay within ±0.05 of the average"*), so its scores were squeezed but did not collapse.

**2. The loss does not punish giving every call the same score.**
Giving every call the same score earns a "neutral" loss, never a bad one, so GEPA kept such prompts.

**3. The validation sets were small.**
Static used 300 calls from just 2 days. Continuous used only 110 calls at its first refit. The 0.79 prompt won there
by luck, then lost out of sample.

**4. The prompts grew long.**
They went from 93 words to 1,300–2,400 words, mostly narrow rules learned from a few calls.

## Next steps
1. Remove numbers from the feedback: say only "ranked too high / too low versus the other calls".
2. Reward prompts for telling winners from losers, so giving every call the same score gets no credit.
3. Reject candidates whose scores barely vary.
4. Use larger selection sets, spread over more days.
5. Use another prompt optimization method (ACE, Agentic Context Engineering, the paper's second method: it builds a
   rulebook through small edits instead of rewriting the whole prompt each round).

## Takeaway
Sean's prompt remains the best (ΔR² ≈ +0.057). The optimizer's failure has an identifiable cause (feedback
design), not GEPA itself. The pipeline is built and runs end to end on Midway, so a rerun with the fixes is cheap.
