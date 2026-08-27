# Heterogeneous Treatment Effects & Uplift Modelling

**Notebook:** `notebooks/03_uplift_modeling.ipynb`
**Code:** `src/uplift.py`, `src/targeting_policy.py`
**Dataset:** [Criteo Uplift Modeling Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
— ~13.98M rows from a randomized display-advertising retargeting test
(85% treated / 15% control), 12 anonymized features, and two outcomes:
`visit` (4.9% overall) and `conversion` (0.29% overall).

## Why this matters for a product DS role

Weeks 1-2 answered "does this work, on average, and how precisely can I
say so?" That's necessary but incomplete — most product interventions help
some users, do nothing for most, and occasionally annoy a few. A model that
tells you *who* falls into which group turns an average effect into a
targeting decision, which is where uplift modelling connects directly to
revenue: this week ends not with a model comparison but with a dollar
figure for a specific targeting policy versus blanket rollout.

## 1. Confirming this is a clean randomized test

Before trusting any effect estimate, I regressed `treatment` on all 12
features (`statsmodels` OLS): **R² = 0.00037**, against an empirical
treatment rate of exactly **0.8500**. The features carry essentially no
information about who was treated — propensity is constant, so pooling
across arms wherever needed below (the X-learner's blending weight, the
targeting policy's IPW evaluation) is safe to do using that single number
rather than fitting a propensity model that would only be learning noise.
This is the same diligence Week 1 applied to Hillstrom's randomization and
Week 2 applied to CUPED's covariates — check the assumption a technique
depends on before leaning on it.

## 2. Working at scale, deliberately

The full CSV loads in ~27s at 727MB once the 12 features are downcast to
float32 and the four flag columns to int8 (vs. ~1.8GB at default float64) —
worth doing explicitly rather than letting pandas infer dtypes. Model
fitting used a 2,000,000-row simple random sample (which preserves the
85/15 treatment split and both outcome rates automatically, since it's
i.i.d.), split 1.5M train / 500K test. `HistGradientBoostingClassifier`
fits in single-digit seconds at this size — the sample size is a
deliberate runtime/iteration-speed choice, not a memory constraint (the
full 14M rows fits comfortably in the environment's memory and fits in
about a minute per model), and still represents ~30x the row count of the
entire Hillstrom dataset used in Weeks 1-2.

## 3. Three meta-learners on `visit`

| Learner | Mechanism | Fit time (1.5M rows) |
|---|---|---:|
| S-learner | One model, treatment as a feature; `uplift = f(X,1) - f(X,0)` | 4.9s |
| T-learner | Two independent models, one per arm; `uplift = mu1(X) - mu0(X)` | 4.5s |
| X-learner | T-learner's two models impute a per-unit effect for *every* unit, a second-stage model per arm learns that imputed effect, then blends with the propensity | 6.1s |

All three use `HistGradientBoostingClassifier`/`Regressor` — fast at this
scale and a defensible default without adding a heavier dependency for a
first pass.

**Predicted uplift distribution on the held-out test set:**

| | mean | std | min | max |
|---|---:|---:|---:|---:|
| S-learner | 0.0067 | 0.0220 | -0.092 | 0.361 |
| T-learner | 0.0072 | 0.0288 | -0.564 | 0.477 |
| X-learner | 0.0073 | 0.0249 | -0.245 | 0.670 |

The S-learner's spread is visibly narrower than T- or X-learner's — the
textbook shrinkage failure mode, where a tree ensemble under-uses a single
treatment column against 12 other features and regularizes the estimated
effect toward zero.

## 4. Evaluation: Qini coefficient

| Learner | Qini coefficient |
|---|---:|
| S-learner | **1610.7** |
| X-learner | 1560.4 |
| T-learner | 1524.0 |

**The S-learner narrowly won** — the opposite of the naive expectation that
more model flexibility (T-, then X-learner) should always score higher.
This is a genuine, useful finding rather than a bug: with a modest true
heterogeneity signal and an unequal 85/15 split, T-learner's control-arm
model is fit on a much smaller, noisier sample, and that added variance can
outweigh the bias the S-learner accepts by sharing one model (and one
regularization budget) across both arms.

X-learner was carried forward for the decision-relevant sections anyway —
its Qini (1560) is within 3% of the S-learner's, and unlike the S-learner it
doesn't structurally shrink every prediction toward zero, which matters
once predictions feed a per-user cost threshold (Section 6). **The
practical lesson for an interview answer**: check the evaluation metric
rather than assume the more sophisticated learner wins by construction —
here it didn't, on this dataset, by this metric.

## 5. Calibration: does the ranking match reality?

Bucketing the X-learner's test-set predictions and comparing the *actual*
experimentally-measured effect per bucket (`qcut` collapsed 10 requested
bins to 8, since tree-based scores produce many tied predictions,
especially near zero):

| Bin (7 = highest) | n | Predicted uplift | Actual uplift |
|---:|---:|---:|---:|
| 7 | 50,000 | 0.059 | **0.065** |
| 6 | 48,701 | 0.009 | 0.006 |
| 5 | 50,570 | 0.003 | 0.002 |
| 4 | 16,058 | 0.002 | 0.003 |
| 3 | 75,635 | 0.001 | 0.002 |
| 2 | 50,990 | 0.001 | 0.001 |
| 1 | 14,432 | 0.001 | -0.001 |
| 0 | 193,614 | -0.0004 | 0.001 |

Overall ATE on `visit` is ~0.0102 (test set treated 4.84% vs. control
3.80%). The pattern is heavily concentrated at the top: the top bin alone
captures an effect roughly **6x the overall ATE**, while every other bin
sits at or below it, several indistinguishable from zero given bin sample
size. This matches a common real-world uplift shape — most users are close
to "sure things" or "lost causes" whose behavior barely depends on
treatment, and the real heterogeneity lives in a persuadable minority the
model is picking out. It's also exactly the shape that makes a cost-based
policy valuable: if most of the effect concentrates in a small top slice,
treating everyone wastes budget on users the campaign barely moves —
Section 6 turns that into an actual dollar figure.

## 6. Which segments differ? (S- vs T- vs X-learner trade-offs)

Comparing mean feature values between the X-learner's top-10% and
bottom-10%-predicted-uplift segments on the test set shows `f9`, `f6`, and
`f0` separate the two groups most; features are anonymized so this can't be
given a business label here, but the same comparison on named features in a
real deployment turns directly into a segment description ("high-uplift
users tend to have low recent engagement but high historical value," etc.).

**S- vs T- vs X-learner, summarized:**

- **S-learner** — cheapest, least variance, but structurally biased toward
  under-estimating heterogeneity (shrinkage); can still win on ranking
  quality (Qini) when the true signal is modest, as it did here.
- **T-learner** — no shrinkage bias, but each arm's model only sees its own
  data; suffers most when arms are unequal in size, as seen in its lowest
  Qini score here (its control model has only ~225K training rows vs. the
  treated model's ~1.275M).
- **X-learner** — designed specifically to fix the T-learner's
  unequal-arm-size weakness by letting both arms inform every prediction;
  didn't top the Qini leaderboard here, but came within 3% of the winner
  while avoiding the S-learner's shrinkage, making it the more defensible
  choice once predictions drive an actual per-user decision.

None of the three is a universal winner — the right choice depends on
arm balance, signal strength, and what the predictions are used for
downstream, which is the actual judgment call a product DS is making, not
just "pick the model with the highest metric."

## 7. From model to decision: a cost-based targeting policy

Fitted a fresh T-learner on `conversion` (the revenue-relevant outcome —
visits are free to the advertiser, conversions carry revenue) and applied a
targeting rule: **treat user *i* iff `predicted_uplift_i * value_per_conversion
> cost_per_treatment`.**

**Illustrative business assumptions** (Criteo attaches no currency to this
dataset): `value_per_conversion = $50` (a plausible average order value for
retail retargeting) and `cost_per_treatment = $0.05` (roughly the cost of
the ad exposures one user receives during the campaign, in line with
typical programmatic display CPMs). These are asserted, not measured — see
the sensitivity check below for how much the conclusion depends on them.

Policy value was estimated using **inverse propensity weighting (IPW)** on
the real, held-out experimental outcomes — not the model's own
predictions. Since treatment was randomly assigned (Section 1), a policy's
value can be estimated directly: for each user, use their actual outcome
only if their actual treatment happened to match what the policy would
have assigned, reweighted by the inverse probability of that match
(Horvitz-Thompson estimator). This grades the *policy*, using the
experiment as an unbiased holdout, regardless of whether the underlying
uplift model is any good.

| Policy | % treated | Expected profit / user | Incremental vs. blanket |
|---|---:|---:|---:|
| **Model-targeted** | **12.5%** | **$0.1495** | **+$0.0404** |
| Blanket (treat all) | 100% | $0.1092 | — |
| None (treat no one) | 0% | $0.1073 | -$0.0018 |

At an audience of 1,000,000 users, the targeted policy nets **~$149,530**
in expected profit versus **~$109,176** for a blanket rollout — **37% more
profit while treating only 12.5% of users**, because most users' predicted
incremental conversion probability doesn't clear the $0.05 cost at $50/
conversion. Blanket rollout barely beats doing nothing at all ($109,176 vs.
$107,333) — nearly all of blanket's ad spend is going to users who were
converting anyway or were never going to convert regardless.

**Sensitivity check**: sweeping `cost_per_treatment` from near-zero to
$0.35 shows the targeted policy's profit stays comparatively flat (roughly
$130K-$155K per million users across the whole range) while blanket
rollout's profit declines linearly and turns negative above roughly
$0.13/treatment — the assumed $0.05 cost isn't a knife-edge assumption the
conclusion depends on; targeting keeps winning across a wide range of
plausible costs, and its advantage over blanket grows as treatment gets
more expensive.

## What's next

Week 4 shifts from "which users to treat" to "which metrics to trust" —
a metrics framework covering north star and guardrail metrics, and a
worked example of a guardrail catching a metric that looks good but hides
a real problem.
