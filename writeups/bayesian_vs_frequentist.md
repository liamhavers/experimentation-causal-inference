# Bayesian vs. Frequentist A/B Analysis

**Notebook:** `notebooks/07_bayesian_vs_frequentist.ipynb`
**Code:** `src/bayesian_ab.py`
**Dataset:** Hillstrom MineThatData — the **Womens E-Mail vs. No E-Mail**
arms, `visit` outcome. The same comparison Week 1 ran a two-proportion
z-test on and Week 5's bandit couldn't cleanly call.

## Why this matters for a product DS role

Weeks 1-6 are entirely frequentist — z-tests, confidence intervals, IPW,
2SLS. Plenty of product orgs (and interviewers) run Bayesian A/B testing
instead, and "why would you pick one over the other" is a standard
question. The useful answer isn't a philosophy lecture; it's knowing that
for a well-powered binary-metric test the two frameworks give *the same
interval*, and being able to name the three situations where they actually
diverge: small samples, an informative prior, and optional stopping. This
notebook builds the Bayesian analysis from scratch alongside the
frequentist one on a test whose answer is already known, so the comparison
is concrete.

## 1. The two reads of the same test

**Frequentist** (`frequentist_two_proportion`): pooled-variance
two-proportion z-test for the p-value, unpooled Wald SE for the 95% CI on
the difference — the Week 1 convention.

**Bayesian** (`src/bayesian_ab.py`, from scratch): the Beta-Bernoulli
conjugate model. Prior `θ ~ Beta(a, b)`; observe `s` visits in `n`
customers; posterior is exactly `Beta(a + s, b + n - s)`. No MCMC. From the
two arms' posteriors:

- **credible interval** on each rate and on the lift — a direct probability
  statement about the parameter;
- **`P(treatment > control)`** — posterior probability the treatment arm is
  genuinely better;
- **expected loss** of each ship decision —
  `E[max(θ_control − θ_treatment, 0)]` for shipping treatment, and the
  mirror image for shipping control. The expected amount of visit rate you
  forgo by shipping that arm, averaged only over the part of the posterior
  where it's the wrong call. Ship when the smaller expected loss drops
  below a pre-set "threshold of caring."

`P(T>C)` and the lift interval have closed forms for Beta-Bernoulli; the
module estimates them by Monte Carlo from the posteriors (200k draws,
reported MC standard error) so the same code would work for a non-conjugate
metric.

## 2. Full sample: the numbers coincide, the claims don't

| | Frequentist | Bayesian (flat `Beta(1,1)` prior) |
|---|---|---|
| Point estimate | +4.52pp | +4.52pp (posterior mean) |
| 95% interval | **[+3.89pp, +5.16pp]** (confidence) | **[+3.89pp, +5.16pp]** (credible) |
| "Significance" number | p ≈ 0 (z ≈ 13.9) | P(T > C) ≈ 1.00 |
| Interval means | 95% of intervals built this way cover the truth | 95% posterior probability the truth is in *this* interval |
| Significance number means | P(data this extreme \| no effect) | P(treatment better \| this data) |

With ~21,300 observations per arm and a flat prior, the posterior is driven
almost entirely by the likelihood, so it lands exactly where the
frequentist sampling distribution does — **the intervals match to four
decimal places** (`writeups/figures/24_bayes_posteriors.png`). What differs
is what you may say: the credible interval licenses "there's a 95%
probability the true lift is between 3.89 and 5.16 points," which is the
sentence people usually want and the confidence interval does not support.
At this sample size, framework choice is a communication decision, not a
statistical one.

## 3. Small samples: same interval, different decision

Shrink each arm to **n = 150** (one fixed subsample; visit rates 11.3% vs.
15.3%). The Week 1 calculator says n = 150/arm can only reliably detect a
~10pp effect — this test is underpowered for the true ~4.5pp effect by
design, which is the point.

| | Result | Summary |
|---|---|---|
| **Frequentist** | diff +4.0pp, p = 0.31, 95% CI [−3.7pp, +11.7pp] | "Inconclusive — cannot reject no effect." Correct, not actionable. |
| **Bayesian** | P(T > C) = 0.84; E[loss \| ship treatment] ≈ **0.3pp**; E[loss \| ship control] ≈ **4.3pp** | With a 0.5pp threshold of caring, shipping treatment is already safe enough; standing pat is not. |

The frequentist test answers "can I reject the no-difference hypothesis at
this sample size" — no. The Bayesian decision framing answers "which arm do
I ship, and what do I lose if I'm wrong" — ship treatment, expect to lose
very little. The second is the question a product decision actually poses.
(`writeups/figures/25_bayes_vs_freq_smalln.png`.)

Two honest caveats. **(1)** This is one subsample; `P(T>C)` ranges ~0.75-0.95
across seeds — small samples are unstable in either framework. **(2)**
Nothing here is *intrinsically* Bayesian — a frequentist can build a
loss-based decision rule too. But the posterior makes "probability
treatment is better" and "expected cost of being wrong" fall out directly,
whereas the p-value has to be argued around.

## 4. Prior sensitivity: when the prior actually matters

Four priors from flat to opinionated, each applied to the n = 150 subsample
and the full sample (`writeups/figures/26_prior_sensitivity.png`):

| Prior | n = 150 posterior lift | Full-sample posterior lift |
|---|---:|---:|
| `Beta(1,1)` flat | +3.9pp (P=0.84) | +4.5pp (P=1.00) |
| `Beta(0.5,0.5)` Jeffreys | +4.0pp (P=0.85) | +4.5pp (P=1.00) |
| `Beta(2,16)` weakly informative (~18 pseudo-obs) | +3.6pp (P=0.84) | +4.5pp (P=1.00) |
| `Beta(33,267)` tight, skeptical, ~11% (~300 pseudo-obs) | **+1.3pp (P=0.73)** | +4.5pp (P=1.00) |

**Full sample:** all four priors agree to three decimals — 21,000
observations swamp even a 300-count prior. "You can get any answer with the
prior" is false at scale.

**n = 150:** flat, Jeffreys, and weakly-informative priors still agree
closely. Only the tight prior — deliberately worth 2x the data and centred
on "no effect" — visibly pulls the posterior toward zero. The honest rule:
a prior changes the conclusion only when it is **both genuinely informative
and competing with a small dataset**, and in that case you should have to
defend where it came from.

## 5. Peeking, Bayesian edition — a callback to Week 1

Week 1: peeking at a frequentist test 20 times and stopping at the first
`p < 0.05` inflated the false-positive rate from 5% to ~24%. Does a Bayesian
posterior threshold fix it? Same design — no true effect, 2,000 sims, 20
looks, stop at first `P(T > C) > 0.95` or `< 0.05`
(`writeups/figures/27_bayesian_peeking.png`):

| | P(ship null treatment) |
|---|---:|
| Single look at the end | **4.8%** |
| Peek at 20 looks, stop early | **21%** |

**It does not fix it** — same qualitative failure, same reason (20
correlated chances to cross a line beat one). The nuance for an interview:

- The posterior *itself* isn't invalidated by peeking — it's always just
  the posterior given the data so far, no correction needed to *state a
  belief*.
- But a **decision rule** on a posterior threshold has frequentist
  operating characteristics, and those degrade under optional stopping
  exactly as above. To control the long-run rate of shipping
  nothing-effects you still need a design built for sequential looks
  (calibrated Bayesian sequential thresholds, or an always-valid / mSPRT
  approach).
- With a *real* +4.5pp effect, both the fixed and the peeking rule ship the
  treatment ~100% of the time — early stopping costs you on the nulls, not
  the true positives, which is what makes it both tempting and dangerous.

## 6. When to use which

**Bayesian framing when:**
- The question is "what do we ship and what's the risk," not "is the effect
  distinguishable from zero" — `P(T > C)` and expected loss map onto a
  decision without translation.
- Samples are thin *and* you have a defensible informative prior (a solid
  historical baseline, a meta-analytic estimate) — it buys real precision.
- You want honest mid-flight progress reporting — provided you don't
  confuse "I can state my current belief" with "I have controlled my error
  rate."

**Frequentist framing when:**
- The result must be defensible to people who'll ask "what prior, and why" —
  a flat-prior analysis at scale gives the same interval anyway, so there's
  little to gain and an argument to have.
- You need a pre-registered, auditable guarantee on the false-positive rate.
- It's the house standard and the numbers coincide (as here) — matching the
  org's default beats the philosophical upgrade.

**Practical reality:** at Hillstrom's full sample the two frameworks produce
the same interval to four decimal places, so the choice is about
communication and decision framing. They genuinely diverge when data is
scarce, when a real prior exists, or when someone wants to stop early —
knowing which of those you're in is the actual skill.

## What's next

This is the first of the optional Phase 7 stretch items. See
`writeups/one_pager.md` for where it sits in the whole project and
`CLAUDE.md` for the remaining stretch items (Streamlit power calculator,
interference/network-effects note).
