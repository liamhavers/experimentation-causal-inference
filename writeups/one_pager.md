# Experimentation & Causal Inference Toolkit — Summary

**A portfolio project demonstrating the experiment design, causal inference,
and metrics judgment a Product Data Scientist role requires.**

## The gap this closes

Three prior projects (fraud detection, NLP, sentiment analysis) show applied
ML and engineering skill but read as fraud/risk and NLP work. Product DS
interviews (fintech, consulting, big tech) skew toward experiment design,
causal inference, and metrics judgment — this project demonstrates that
directly, end to end, on public datasets, with every technique paired to a
plain-English writeup built to survive interview follow-up questions.

## What's built (all 6 core weeks complete)

**Week 1 — Power analysis & experiment design** (Hillstrom, 64K rows).
Built a two-proportion-test sample-size/MDE calculator from first
principles, cross-checked against `statsmodels`. Used it retrospectively to
confirm a real email A/B test was well-powered for the effect it found
(achievable MDE 7.9% vs. observed 72.1% lift). Simulated 3,000 repeated-
peeking experiments with **no true effect** and showed stopping at the
first significant look inflates the false positive rate from a nominal
5% to **24.3%**.

**Week 2 — CUPED & variance reduction** (Hillstrom). Implemented CUPED by
hand — single- and multi-covariate — matching the theoretical `1 - corr²`
variance reduction exactly. Checked real covariate strength *before*
applying it: found Hillstrom's pre-experiment covariates weak for `visit`
and negligible for `conversion`/`spend`, an honest finding rather than
blind application. A labeled synthetic strong-covariate example showed the
method's full effect: **36% variance reduction**, translated into an
equivalent traffic saving via the Week 1 calculator.

**Week 3 — Heterogeneous treatment effects & uplift modelling** (Criteo,
~14M rows). Built S-, T-, and X-learner meta-learners, evaluated with a
Qini curve — found the simplest model (S-learner) narrowly topped the
leaderboard despite its known shrinkage bias, a concrete lesson in checking
the metric rather than assuming the fancier model wins. Turned the winning
model's predictions into a **cost-based targeting policy**, evaluated via
inverse-propensity weighting on real held-out outcomes (not the model's own
predictions): targeting **37% more profitable than blanket rollout while
treating only 12.5% of users**, with a cost-sensitivity sweep confirming
the conclusion holds across a wide range of assumed costs.

**Week 4 — Metrics framework & guardrails.** A worked, runnable simulation
of a checkout-flow test where the primary metric (checkout completion)
improves significantly while a guardrail (delivery-failure rate) degrades
significantly — diagnosed down to the exact subgroup causing it (users with
a stale saved address), then priced both metrics into a common $ unit to
show the naive "ship it" call was actually a **net loss of ~$422K per
million sessions**. Plus three more runnable failure-mode demos: Simpson's
paradox (a pooled result that reverses sign vs. every subgroup), novelty
effects (a full-length cumulative readout still overstating long-run impact
~3x), and network/spillover contamination (a shared-resource constraint
found — and reported honestly — to *compress* rather than inflate the
measured effect, against the naive assumption).

**Week 5 — Bandits vs. fixed-horizon A/B testing** (Hillstrom). Implemented
epsilon-greedy and Thompson sampling from scratch, run against Week 1's
actual fixed-split design on the same 64,000-email budget, with every
simulated reward drawn from the real recorded outcomes. Thompson sampling
cut cumulative regret **~97.5%** versus the fixed split (56.8 vs. 2,303
foregone visits — worth ~2,246 extra visits at zero extra spend), but its
own sample sizes couldn't reproduce Week 1's result: the same
Womens-vs-No-Email comparison that was significant at **p<0.001** under
the fixed split came out **non-significant (p=0.32)** under Thompson
sampling's own much smaller per-arm samples — a concrete demonstration,
not just an assertion, of what a bandit's efficiency gain costs a
stakeholder report.

**Week 6 — Quasi-experimental methods** (Criteo + Hillstrom). Added after
the original plan, since Weeks 1-5 all used real RCTs and most real
product decisions don't come with one. Two validated demonstrations: (1)
**instrumental variables** on a genuine natural experiment already inside
the Criteo data — `treatment` as a randomized instrument for `exposure`
(one-sided noncompliance, not constructed for the occasion) — recovered a
LATE of **0.287** on `visit` via the Wald estimator, exactly matching a
`linearmodels` IV2SLS cross-check, and showed the naive as-treated
comparison **overstates the true effect by ~32%**; (2) **propensity score
matching, IPW, and regression adjustment validated against a known
answer** — deliberately confounded a resampled version of the Hillstrom
RCT (real outcomes, biased retention only) and confirmed all three methods
recover the already-known true ATE closely (IPW within **0.5%**) from a
sample where the naive comparison was **56.5% too high**.

**Phase 7 — Bayesian vs. frequentist** (Hillstrom; optional stretch, in
progress). Built the Beta-Bernoulli Bayesian A/B analysis from scratch —
credible intervals, `P(treatment > control)`, decision-theoretic expected
loss — next to the Week 1 z-test on the same comparison. At full sample the
credible and confidence intervals **match to four decimal places**: the
framework choice is about what you may claim, not the answer. They diverge
where it matters — at n=150/arm the z-test is inconclusive (p=0.31) while
the Bayesian expected-loss framing still gives a defensible ship call; the
prior only moves the posterior when it's both informative and fighting a
small sample; and a `P(T>C) > 0.95` early-stop rule inflates the false-win
rate from ~5% to ~21% under peeking — Week 1's trap in Bayesian clothing.

## Why this matters

Every result above ties a statistical technique to a business number or a
decision: traffic saved, profit gained, or a launch call overturned. That's
the actual job — not building a model, but knowing which number to trust,
how much traffic it costs to trust it, and what it's worth once you do.

## What's next

The core 6-week build is complete and the Bayesian-vs-frequentist stretch
item is done. Remaining Phase 7 extensions in `CLAUDE.md`: a Streamlit app
wrapping the power calculator, and a short note on interference/network
effects.

---
*Full repo, code, and writeups: see the [README](../README.md).*
