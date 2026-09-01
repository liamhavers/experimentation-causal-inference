# Quasi-Experimental Methods

**Notebook:** `notebooks/06_quasi_experimental_methods.ipynb`
**Code:** `src/instrumental_variables.py`, `src/propensity_matching.py`

## Why this matters for a product DS role

Every dataset in Weeks 1-5 was a real randomized experiment — which makes
for clean statistics, but most real product decisions don't come with an
RCT attached. A feature that's already fully rolled out, a pricing change
that applied to everyone at once, a policy cutoff, a self-selected opt-in —
all common, all unrandomized. Quasi-experimental methods are what's left
when random assignment isn't available, and knowing how to reach for them
*and* how to check whether they actually worked is a distinct skill from
running a clean A/B test. This week builds two such scenarios, each with a
way to check the answer rather than just trust the method.

## Part A — Instrumental Variables (Criteo)

### The setup

Criteo's `treatment` column is a randomized instrument: 85% of users were
*eligible* to see a retargeting ad. `exposure` records whether the ad
actually rendered — and **exposure is only ever 1 when treatment is 1**
(confirmed directly: 0 counterexamples in 14M rows). That's one-sided
noncompliance, the cleanest possible case for IV: the instrument is known
randomized (Week 3 confirmed R²≈0 for treatment regressed on all
features), and monotonicity holds automatically since exposure is
structurally impossible under control. Only **3.6%** of eligible users
were ever actually exposed — most "treated" users never saw the ad at all.

### Three answers to "what's the effect of the ad?"

| | `visit` | `conversion` |
|---|---:|---:|
| **ITT** (effect of assignment) | 0.0103 | 0.0012 |
| **Naive as-treated** (exposed vs. unexposed) | 0.3792 | 0.0525 |
| **LATE** (IV-corrected, Wald) | **0.2870** | **0.0320** |

Three genuinely different, individually correct answers to three
genuinely different questions:

- **ITT** answers "what's the effect of running this campaign as
  deployed" — small, because it's diluted across the ~96% of eligible
  users who never actually saw an ad.
- **Naive as-treated** is wrong as an answer to "what's the effect of
  exposure." Exposure isn't random even within the treatment group — it
  requires browsing a page where the ad network can serve the impression,
  which is itself correlated with exactly the browsing intent that
  predicts a visit, independent of the ad. Comparing exposed to unexposed
  users compares two systematically different kinds of people, not two
  outcomes of the same kind of person.
- **LATE** answers "what's the effect of exposure, among the users whose
  exposure the campaign eligibility actually controls" (compliers), using
  only the randomized variation in exposure rather than all of it.

**The naive estimate overstates the true exposure effect on `visit` by
~32%** (0.379 vs. 0.287) — a real, checkable number, not a hypothetical
warning about what could go wrong.

### Instrument strength and cross-checks

First-stage F-statistic: **78,392** — an instrument about as strong as
exists in practice (rule of thumb: F < 10 is "weak"), unsurprising since
`treatment` is a literal randomized eligibility switch driving `exposure`
directly. Not every real instrument is this clean; checking this before
trusting an IV estimate matters precisely because weaker, more contestable
instruments (a policy threshold, a distance-to-something, a lottery) are
the norm outside a case this well-behaved.

The Wald ratio (0.28700) matches the manual two-stage-least-squares
second-stage coefficient (0.28700) exactly — the two are mathematically
identical for a single binary instrument and a single endogenous
regressor. `linearmodels.iv.IV2SLS` confirms the same point estimate
(0.2870) and adds a proper standard error (0.0040) and confidence interval
(0.279, 0.295) that the manual version can't provide, since it doesn't
account for the first stage itself being estimated.

## Part B — Propensity Score Methods, Validated Against a Known Truth (Hillstrom)

### Why validate at all

Unconfoundedness — the assumption that treatment is as-good-as-random once
you condition on observed covariates — can't be tested from observational
data alone; there's no way to rule out an unmeasured confounder using only
what's in front of you. The standard way causal-inference methodology
validates an estimator anyway: take data where the truth *is* known (a
real RCT), deliberately construct a confounded version of it, and check
whether the estimator recovers the number already known to be right. Used
Week 1's Mens E-Mail vs. No E-Mail comparison — true ATE on `visit`:
**+0.0766**.

### Engineering the confound

Real outcomes throughout — only *which rows are kept* is biased. Retention
probability depends on each unit's own treatment status and covariates
(`recency`, `history`, `newbie`, `mens`), tuned so high-value/low-recency
customers are over-represented in treatment and under-represented in
control (a stand-in for real selection, e.g. an opt-in feature engaged
customers choose more often). Result: 21,108 rows (from 42,613), with the
naive treatment-vs-control comparison on that sample landing at **0.1199**
— a **56.5% overstatement** of the true 0.0766 effect. Covariate imbalance
driving that bias is large and visible: standardized mean differences of
-0.70 (recency), +0.85 (history), +0.34 (mens) — all well past the
conventional |SMD| < 0.1 balance threshold.

### Three corrections, checked against the known truth

| Method | Estimated ATE | Error vs. true | % error |
|---|---:|---:|---:|
| Naive (confounded sample) | 0.1199 | +0.0433 | +56.5% |
| Nearest-neighbor matching | 0.0831 | +0.0065 | +8.5% |
| Inverse propensity weighting | 0.0762 | -0.0004 | **-0.5%** |
| Regression adjustment | 0.0775 | +0.0009 | +1.2% |

All three corrections land close to the true, already-known ATE — IPW
essentially exactly, regression adjustment and matching both within single
digits of percent error, none close to the naive comparison's error.
Matching also achieved good covariate balance directly (SMD dropped from
-0.70/+0.85/+0.34 to 0.009/-0.003/0.009 on the same three covariates,
matching 10,123 of 10,123 treated units within a 0.05 propensity caliper)
— in a genuine observational study where the true ATE *isn't* known in
advance, this before/after balance check, not proximity to a ground truth,
is the actual signal that a propensity method has done its job. Propensity
overlap between arms was checked directly too (histogram of estimated
scores by arm) — reasonable across most of the range, with expected
thinning at the extremes, no region so bare that the methods above should
be expected to fail.

The three methods didn't agree with each other exactly (0.076-0.083 range)
— a realistic and useful finding on its own: no single method is
guaranteed best, and triangulating across two or three is standard
practice precisely because a single estimator can be off even when its
assumptions roughly hold.

## When to reach for which

**Use a real RCT (Weeks 1-5) whenever you can.** Every method in this
notebook exists to answer a question an RCT would answer more simply, and
both halves of this week only "work" because a known-true benchmark was
available to check against — a luxury genuine observational analysis
doesn't have.

**Use instrumental variables when**: treatment itself isn't randomized (or
compliance is imperfect, as here), but something upstream of it *is*
randomized or plausibly as-good-as-random, and that upstream thing
plausibly affects the outcome *only* through treatment (the exclusion
restriction — the hardest IV assumption to defend, and the one most worth
scrutinizing in a real analysis, unlike this week's unusually clean case).

**Use propensity score matching/IPW/regression adjustment when**: there's
no credible instrument, but a rich enough set of observed covariates makes
unconfoundedness plausible — and always report a covariate-balance check
(SMD, overlap histogram), since that's the diagnostic available even when
(as in every real use of this method) the true answer isn't known in
advance to check against.

**The common thread with Weeks 1-5**: every method here rests on an
assumption a real RCT gets for free (random assignment). None of those
assumptions are free to assert — they're worth checking (instrument
strength, covariate balance, overlap) or validating (against a known
answer, where one exists) with the same diligence this project applied to
randomization checks in Week 1, covariate strength in Week 2, and
propensity constancy in Week 3.
