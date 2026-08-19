# Power Analysis & Experiment Design

**Notebook:** `notebooks/01_power_analysis.ipynb`
**Code:** `src/power_analysis.py`
**Dataset:** Kevin Hillstrom's MineThatData E-Mail Analytics dataset (64,000
customers, randomized 1/3 Men's e-mail / 1/3 Women's e-mail / 1/3 control,
outcomes tracked over the following two weeks).

## Why this matters for a product DS role

Before a product data scientist touches a model, they're usually the one
answering: *is this test even worth running, and how long will it take?*
That's a power analysis question, not a modelling question — and getting it
wrong in either direction is expensive. Underpowered tests burn traffic and
calendar time to produce an inconclusive result; ignoring the "peeking"
problem produces false positives that erode stakeholder trust in the testing
process itself. This notebook builds the calculator from first principles and
then uses simulation to make the peeking problem concrete rather than
theoretical.

## 1. Randomization check

Before trusting any effect estimate, I checked that the three arms are
actually balanced on pre-experiment covariates (`recency`, `history`,
`mens`, `womens`, `newbie` — all measured *before* the campaign). A one-way
ANOVA per covariate found no significant imbalance across arms (all
p-values > 0.6). This is a five-minute check that would catch a broken
randomization or corrupted assignment pipeline before it wastes an entire
analysis — cheap insurance worth doing on every real test, not just this one.

## 2. The sample size / MDE calculator

`src/power_analysis.py` implements the standard two-proportion z-test power
formula for a two-sided test:

```
n = (z_(1-alpha/2) * sqrt(2 * p_bar * (1 - p_bar)) + z_(1-beta) * sqrt(p1*(1-p1) + p2*(1-p2)))^2
    / mde^2
```

where `p_bar` is the pooled rate under the null and `p1`, `p2` are the
control and treatment rates. It exposes the same relationship in three
directions, because in practice you're rarely solving for just one of them:

- `required_sample_size(baseline_rate, mde_absolute, alpha, power)` — how
  many users per arm to hit a target MDE.
- `minimum_detectable_effect(baseline_rate, n_per_variant, alpha, power)` —
  what MDE a fixed sample size can actually detect (the question that
  matters when traffic or timeline is the real constraint, not the effect
  size).
- `SampleSizeResult.runtime_days(daily_traffic, traffic_split)` — turns a
  required sample size into calendar days given expected traffic.

I cross-checked `required_sample_size` against
`statsmodels.stats.power.NormalIndPower` (which uses Cohen's h / an arcsine
transform rather than the raw-proportion approximation used here) — the two
agree to within 0.2% on a representative case (baseline 10%, MDE +2pp:
3,841 vs. statsmodels' 3,835 per arm), which is the expected level of
agreement between two textbook-standard approximations.

### Applied to the Hillstrom control visit rate (10.62%)

| Relative lift | Absolute MDE | n per arm | n total | Runtime @ 5,000 users/day |
|---:|---:|---:|---:|---:|
| 5%  | 0.53pp | 54,024 | 108,048 | 21.6 days |
| 10% | 1.06pp | 13,794 | 27,588  | 5.5 days |
| 15% | 1.59pp | 6,257  | 12,514  | 2.5 days |
| 20% | 2.12pp | 3,591  | 7,182   | 1.4 days |
| 30% | 3.19pp | 1,658  | 3,316   | 0.7 days |
| 50% | 5.31pp | 641    | 1,282   | 0.3 days |

Required sample size grows roughly with `1/MDE²`, so halving the effect you
want to detect roughly quadruples the sample needed — the sharp knee below
~10% relative lift in the sample-size-vs-MDE curve is why product teams
generally don't chase very small effects without either a lot of traffic or
a variance-reduction technique like CUPED (Week 2).

### Retrospective check: was the real Hillstrom test well-powered?

The actual experiment ran ~21,300 customers per arm. Plugging that into
`minimum_detectable_effect` against the observed 10.62% control rate gives an
achievable MDE of **0.84 percentage points (7.9% relative)** at alpha=0.05,
power=0.80. The Men's campaign's actual effect was **+7.66 percentage points
(72.1% relative)** — nearly 10x the MDE the sample size was built to detect,
confirmed directly with a two-proportion z-test (z=22.5, p≈0). That's the
signature of a comfortably well-powered test: the result isn't a lucky draw
from an underpowered study, and (with hindsight) the sample size was
arguably larger than strictly necessary for an effect this large — a
real-world example of why prospective MDE choice matters, not just running
the biggest test you can afford.

## 3. The peeking problem

A p-value threshold controls the false positive rate at `alpha` only under
the assumption that you looked at the data **once**, at a pre-committed
sample size. Checking a dashboard daily and stopping the moment the test
"turns green" breaks that assumption — even when there is truly no effect.

`simulate_peeking()` makes this concrete: 3,000 simulated experiments,
**no true effect in any of them**, each checked at 20 evenly spaced points as
the sample accumulates from 0 to 4,000 users per arm (baseline rate = the
observed 10.62% Hillstrom control rate).

- **Fixed-horizon** (look only at the final checkpoint): false positive rate
  = **4.9%** — matches the nominal alpha of 5%, as it should.
- **Peeking** (stop at the first checkpoint where p < 0.05): false positive
  rate = **24.3%** — roughly 5x the nominal rate.

Each *individual* look is well-calibrated (~5% of experiments reject at any
single checkpoint — see the per-look plot). The inflation comes from
taking 20 *different* chances to reject across the life of the test: the
probability that *at least one* of 20 correlated-but-imperfectly-correlated
looks turns up significant by chance is much higher than the probability
that any one look does. It's the same underlying logic as a multiple
comparisons problem, just spread across time instead of across metrics.

**Practical takeaways:**

- Decide upfront whether a test is fixed-horizon (commit to the sample size,
  don't act on interim reads) or genuinely sequential (use a method built
  for repeated looks — group-sequential designs with an alpha-spending
  function such as O'Brien-Fleming, or an always-valid sequential testing
  approach).
- "We'll peek and stop early if it's already significant" is not a free
  shortcut — at 20 looks it costs roughly 5x the nominal false positive
  rate, and the cost scales with how often you check.

## What's next

Week 2 (CUPED) picks up directly from the sample-size-vs-MDE trade-off
above: instead of just buying a smaller MDE with more traffic, CUPED buys it
by reducing outcome variance using a pre-experiment covariate — the same
mechanism that made the randomization check in Section 1 possible in the
first place (there being usable pre-period covariates at all).
