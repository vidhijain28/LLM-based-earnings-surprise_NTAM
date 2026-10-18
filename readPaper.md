# 📄 Reading Notes: Koijen & Levy (2026)

> **Paper:** Koijen, R. S. J., & Levy, B. (2026). *Assessing the Benefits of Optimized Agentic AI Systems for Asset Pricing.*
>
> *Proxy yourself as a standpoint of author:*

**Contents**
1. [What do I want to know?](#what-do-i-want-to-know)
2. [Why?](#why)
3. [What data do I need?](#what-data-do-i-need)
4. [How can I achieve my goal?](#how-can-i-achieve-my-goal)

---

## What do I want to know?

> [!NOTE]
> The paper aims to *explain* the stock's reaction to an earnings announcement (does the AI read the news the way the market does?), **not** to predict future returns and trade on them. The AI reads the transcript after the call, when the price may already have started moving, so the result is a test of understanding rather than a trading strategy.

### Main question

**How much of the announcement-window stock return can an optimized AI system explain, using only information available at announcement time?**

Formally, the idiosyncratic return is

$$
\varepsilon = g(\omega, \Omega_{-}) + \nu
$$

where $\omega$ is the release/call content, $\Omega_{-}$ is what investors knew before, $g(\cdot)$ is the unobservable market pricing function, and $\nu$ is noise. The question is how well an LLM's prediction $\hat{g}$ approximates $g$, measured by the **incremental R² ($\Delta R^2$)** over the standard earnings surprise (SUE) in a cross-sectional regression.

### Sub-questions

> [!TIP]
> Read through these sub-questions carefully. They outline what the paper tests, so they will help you better understand the rest of the paper.

1. With just a simple prompt, does an LLM reading the transcript explain the stock's reaction better than the earnings surprise alone?
2. Does the LLM capture new information, or does it just repeat what the earnings surprise already tells us?
3. Can we make the LLM better by improving its prompt or giving it learned rules, and does that improvement hold on new, unseen announcements?
4. Do we need the full transcript, or is a much shorter summary good enough (cheaper and faster)?
5. Do some models work better for some industries, or is one model best everywhere?
6. Can rules learned by a strong, expensive model make a cheaper model perform almost as well?
7. What does the AI learn about how the market reacts to news, and does it make economic sense?

---

## Why?

| Reason | Explanation |
|---|---|
| **"Dark matter" of prices** | 50 years of research only moved explained variation from ~2% (Lev 1989) to ~8% (Hand et al. 2022). Earnings announcements are the cleanest test of how markets price new information. |
| **LLMs can read soft information** | Tone, guidance, Q&A, and causal explanations matter (Tetlock 2007; Loughran-McDonald 2011; Mayew-Venkatachalam 2012), and dictionary methods capture little of it. |
| **Historical backtests are contaminated** | LLMs are trained on past returns (look-ahead bias), and optimization can amplify it by recovering memorized outcomes (Hu et al. 2024). Knowledge-cutoff models are too weak and only give a lower bound. |
| **Markets are reflexive** | As investors adopt AI, the text→return mapping changes, so historical results may not apply today. Only a **real-time, out-of-sample benchmark** measures current capability. The benchmark is designed for construct validity, contamination resistance, ecological validity, discriminative power, and reproducibility. |
| **Interpretability** | Optimized prompts/rulebooks are human-readable rules that can propose new determinants of realized returns for asset pricing theory, which traditional ML cannot. |
| **Collective progress** | Launch an open competition (explainingmarkets.ai), analogous to ImageNet. |

---

## What data do I need?

| Need | Source in paper | Role |
|---|---|---|
| Earnings event calendar (dates/times) | Benzinga | Build the ex-ante information set strictly before each event; assign effective event date (16:00 ET rule) |
| Analyst consensus EPS $\mathbb{E}[e \mid \Omega^a_{-}]$ and actual EPS $e$ | Benzinga (I/B/E/S for robustness) | Earnings surprise $s = (e - \text{consensus}) / \text{pre-announcement price}$ |
| Earnings call transcripts $\omega$ | S&P Capital IQ | Main LLM input, full or summarized; collected in real time |
| Daily stock total returns and market return | Massive (formerly Polygon.io); market proxy = VTI (Capital IQ for robustness) | Abnormal return $r^{abn} = r_i - r_m$, close-to-close, 1-day window |
| Pre-announcement price | Same pricing source | Scales the surprise |
| Industry classification | Fama-French 12 | Industry heterogeneity analysis |
| Analyst coverage and forecast dispersion | Estimates data | Descriptives; ex-ante uncertainty proxy |
| LLM access | Gemini 2.5 Flash-Lite, GPT-5 nano, Opus 4.5 (+ Haiku 4.5, Sonnet 4.5); no web search | Generate predictions $\hat{g}$; training cutoff must precede sample |

**Sample:** 1,849 earnings announcements in Q4 2025 (Oct–Dec). Split at the median date: Early (1,032) = training for GEPA/ACE; Late (817) = out-of-sample evaluation. Returns and surprises are converted to within-quarter percentile ranks.

**Data the authors say they did not use yet:** prior calls of the same firm, peer firms' calls, and pre-call information about what analysts are focused on.

---

## How can I achieve my goal?

### 1. What is my overall approach?

Build a **real-time, out-of-sample benchmark**: the AI makes its predictions live, and they are checked later against what actually happened.

1. **Before the call:** lock the AI system (model + prompt/rulebook). It cannot be changed for this event.
2. **Right after the call:** the AI reads the transcript and gives one number, $\hat{g}$ between 0 and 1. This is its guess of how this stock's 1-day abnormal return will compare with all other stocks reporting that quarter (0 = worst reaction, 0.5 = typical, 1 = best).
3. **After the event day closes:** the actual 1-day abnormal return becomes known.
4. **End of the quarter:** rank all realized abnormal returns from 0 to 1, then measure how well the AI's guesses line up with these realized ranks (regression R²).

> [!IMPORTANT]
> **Why this avoids look-ahead bias:** the guess is locked in before the return exists, so the AI cannot have seen the answer in its training data or anywhere else.

### 2. What steps do I follow, in order?

#### Step 1 — Collect events
Keep a calendar of upcoming earnings announcements with the date, call time, and the analyst consensus EPS formed *before* the release. This defines the sample and the information investors had beforehand.

#### Step 2 — Set the event date $t$
The event date is the first trading day on which investors can act on the news. For a call at time $t_c$ on calendar date $d$:

$$
t =
\begin{cases}
d & \text{if } t_c \le \text{16:00 ET and } d \text{ is a trading day} \\
\tau(d) & \text{if } t_c > \text{16:00 ET or } d \text{ is not a trading day}
\end{cases}
$$

where $\tau(d)$ is the next trading day after $d$. Example: a call at 17:00 ET on Tuesday → $t$ = Wednesday.

#### Step 3 — Compute the abnormal return
The stock's return minus the market's return on the event day, close-to-close from $t-1$ to $t$:

$$
r^{abn}_{i,t} = r_{i,t} - r_{m,t}
$$

where $r_{i,t}$ is stock $i$'s total return (including dividends) and $r_{m,t}$ is the market return (proxied by VTI). This removes market-wide moves and keeps the firm-specific reaction.

#### Step 4 — Compute the earnings surprise
How much actual EPS beat or missed the consensus, scaled by the pre-announcement price so firms are comparable:

$$
s_{i,t} = \frac{e_{i,t} - \mathbb{E}[e_{i,t} \mid \Omega^a_{t-}]}{p_{i,t-}}
$$

where $e_{i,t}$ is actual EPS, $\mathbb{E}[e_{i,t} \mid \Omega^a_{t-}]$ is the mean analyst forecast before the release, and $p_{i,t-}$ is the stock price before the announcement.

#### Step 5 — Convert to percentile ranks
Within each quarter $q$, rank every abnormal return and every surprise from most negative to most positive:

$$
\text{PRank}(x_{i,t}) = \frac{\text{rank}(x_{i,t})}{N_q}
$$

where $N_q$ is the number of events in the quarter, so values run from $1/N_q$ (worst) to $1$ (best). This limits the influence of outliers. From here on, $r^{abn}$ and $s$ mean their ranked versions.

#### Step 6 — Get the LLM prediction
Give the LLM the transcript (full or summarized) and the earnings numbers. It returns one number $\hat{g}(\omega_{i,t}, \Omega_{t-}) \in [0, 1]$: its guess of where this stock's abnormal return will rank among all stocks reporting that quarter.

#### Step 7 — Run the regressions
Compare two cross-sectional regressions:

$$
\text{Baseline:} \quad r^{abn}_{i,t} = \alpha + \theta \, s_{i,t} + u_{i,t}
$$

$$
\text{With LLM:} \quad r^{abn}_{i,t} = \alpha + \gamma \, \hat{g}(\omega_{i,t}, \Omega_{t-}) + \theta \, s_{i,t} + u_{i,t}
$$

The gain from the LLM is $\Delta R^2 = R^2_{\text{With LLM}} - R^2_{\text{Baseline}}$.

| Symbol | Meaning |
|---|---|
| $r^{abn}_{i,t}$ | Ranked (0–1) abnormal return of stock $i$ on event date $t$; the variable being explained |
| $s_{i,t}$ | Ranked (0–1) earnings surprise of stock $i$ at event $t$ |
| $\hat{g}(\omega_{i,t}, \Omega_{t-})$ | The LLM's prediction (0–1) |
| $\omega_{i,t}$ | What was released at the event: transcript and earnings numbers |
| $\Omega_{t-}$ | What investors knew before the release |
| $\alpha$ | Intercept (constant) |
| $\theta$ | Coefficient on the surprise: how much the surprise moves the return rank |
| $\gamma$ | Coefficient on the LLM prediction: how much the LLM's guess moves the return rank |
| $u_{i,t}$ | Residual: the part of the return rank neither variable explains |
| $R^2$ | Share of the variation in $r^{abn}_{i,t}$ across stocks that the regression explains |
| $\Delta R^2$ | Extra share explained by adding the LLM on top of the surprise |

### 3. What tools do I use?

| Category | Tools |
|---|---|
| **Data** | Benzinga (calendar, consensus, actual EPS), S&P Capital IQ (transcripts), Massive / Polygon.io (prices, VTI). Robustness: I/B/E/S and Capital IQ prices. |
| **LLMs** | Gemini 2.5 Flash-Lite, GPT-5 nano, Opus 4.5; Haiku 4.5 and Sonnet 4.5 to test cheaper models. |
| **Prompt/context optimization** | GEPA (Agrawal et al. 2026) and ACE (Zhang et al. 2026). |
| **Statistics** | OLS regressions with 1,000-sample bootstrap 99% confidence intervals. |

### 4. How do I make sure there is no look-ahead bias?

- Predictions are made **in real time**, before returns are observed.
- Models have **training cutoffs before the sample** (e.g., Opus 4.5 cutoff Aug 2025; sample is Oct–Dec 2025).
- **No web search** or external information at prediction time.
- Optimization (improving the prompt by learning from past mistakes; see section 5) uses **only the early half** of the quarter (training). The improved prompt is then frozen and results are reported on the **late half** (testing), on announcements it has never seen.

### 5. How do I improve the LLM beyond a simple prompt?

#### What is optimization?
The LLM's answers depend heavily on the instructions it is given (the prompt). Optimization means **automatically improving those instructions** using past examples where we already know the actual return. The model itself is not retrained; only the text we give it changes.

#### Why use it?
With a simple prompt, the LLM makes systematic mistakes. For example, it predicts a stock will go up whenever results are strong, even when the market already expected them ("good news is already priced in"). Optimization finds and corrects these mistakes, so the LLM reads earnings news more like the market does.

#### What does it do, step by step?
1. **Summarize transcripts first.** Optimization needs many LLM calls, so shorter inputs make it affordable (Opus 4.5 summaries are ~28× shorter, with only a small loss in R²).
2. **Set the target.** On the training half, regress return on surprise and keep the residuals: the part of the return the surprise cannot explain. The LLM is optimized to predict these residuals, so it focuses on new information.
3. **Learn from mistakes.** Run the LLM on training examples, compare its predictions to the targets, have an LLM explain *why* it was wrong, and rewrite the prompt or rules.
4. **Repeat** until the budget is used up, then freeze the best version and test it on the late half.

#### What tools are used?
Both are *gradient-free*: they improve the prompt by having an LLM read and rewrite text, not by changing model weights.

| | **GEPA** (Genetic-Pareto prompt optimization) | **ACE** (Agentic Context Engineering) |
|---|---|---|
| **Reference** | Agrawal et al. (2026) | Zhang et al. (2026) |
| **How it works** | Keeps a **pool of candidate prompts**, like a population in evolution. Picks a prompt, tests it on a small batch, and has an LLM read the failures and **rewrite the prompt** ("reflective mutation"). Keeps the new prompt only if it does better. | Builds a **rulebook** of if-then rules that is added to the prompt. Three roles: the **Generator** makes predictions using the rulebook; the **Reflector** studies what went right or wrong; the **Curator** adds, updates, or removes rules. Makes **small edits** each round instead of rewriting everything, so useful rules are not lost. |
| **Output** | **One improved prompt**. In the paper it mainly fixed bias ("good is priced in") and added sector-specific tips. | **A rulebook**, e.g. "if record margins come from one-time gains and a major segment is shrinking, predict a low rank." It gave the biggest gain in the paper and also works when given to cheaper models. |

### 6. How do I evaluate the result?

#### Main regression
Across all announcements in the evaluation sample, regress the ranked abnormal return on the LLM prediction and the ranked surprise:

$$
\text{PRank}(r^{abn}_{i,t}) = \alpha + \gamma \, \hat{g}(\omega_{i,t}, \Omega_{t-}) + \theta \, \text{PRank}(s_{i,t}) + u_{i,t}
$$

(Symbols are defined in the table under step 7 of section 2.) A positive, significant $\gamma$ means the LLM's guess lines up with how the stock actually reacted.

#### Key metric: extra R² ($\Delta R^2$)
How much more of the return variation is explained once the LLM is added on top of the surprise:

$$
\Delta R^2 = R^2_{\text{surprise} + \hat{g}} - R^2_{\text{surprise only}}
$$

The surprise-only $R^2$ is the baseline to beat. $\Delta R^2$ is not the LLM's own $R^2$ minus the baseline; it only counts what the LLM adds *beyond* the surprise, since the two partly overlap.

#### Confidence intervals
Report **99% bootstrap confidence intervals** for each $R^2$:
1. Resample the announcements with replacement (1,000 times).
2. Re-run the regression on each resample and record $R^2$.
3. Take the 0.5th and 99.5th percentiles as the interval.

If the intervals of two models or prompts overlap a lot, the difference between them may just be noise.

#### Robustness checks
Make sure the result is not driven by one slice of the data:

| Check | Question |
|---|---|
| **By industry** (Fama-French 12) | Does the LLM help in every industry, or only some? Which model is best in each? |
| **Early vs. late announcements** | Is performance similar in both halves of the quarter? This also checks that the late (test) half is not systematically easier or harder than the early (training) half. |
| **Full transcripts vs. summaries** | How much is lost by using shorter inputs? |
| **Data source** | Results should not change much when switching providers (e.g., I/B/E/S instead of Benzinga). |
