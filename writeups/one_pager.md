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

## What's built (Weeks 1-4 of 5, in progress)

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

## Why this matters

Every result above ties a statistical technique to a business number or a
decision: traffic saved, profit gained, or a launch call overturned. That's
the actual job — not building a model, but knowing which number to trust,
how much traffic it costs to trust it, and what it's worth once you do.

## What's next

Week 5 adds a multi-armed bandit on the same Hillstrom dataset used in
Week 1, comparing cumulative regret against the fixed-horizon A/B test and
writing up when each approach is the right call.

---
*Full repo, code, and writeups: see the [README](../README.md).*
