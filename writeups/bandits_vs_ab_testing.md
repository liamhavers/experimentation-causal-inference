# Bandits vs. Fixed-Horizon A/B Testing

**Notebook:** `notebooks/05_bandits.ipynb`
**Code:** `src/bandits.py`
**Dataset:** Hillstrom MineThatData — the same dataset and the same three
arms (No E-Mail / Womens E-Mail / Mens E-Mail) used for the fixed-horizon
test in Week 1.

## Why this matters for a product DS role

Weeks 1-4 all assumed the test design itself — a fixed split, decided once
— was the right call. It usually is, but not always: an always-on campaign
or a revenue-sensitive rollout can afford to *adapt* traffic toward the
winning option as evidence accumulates, rather than holding a locked split
for the test's full duration. Knowing when to reach for which is itself a
judgment call a product DS is expected to make, not just execute a fixed
test because that's the default.

## 1. Setup

Week 1 established (and confirmed significant): No E-Mail 10.6% visit
rate, Womens E-Mail 15.1%, Mens E-Mail 18.3%. That test ran all ~64,000
customers through a locked 1/3 split — the right design for a clean,
one-off causal question. This week asks a different question: if the goal
during the test itself were to maximize total visits generated — the
situation for an ongoing campaign rather than a one-time research question
— how much is a fixed split actually leaving on the table?

Every simulated pull draws its reward via bootstrap resampling from that
arm's real recorded `visit` outcomes in Hillstrom — distributionally
identical to Bernoulli at the arm's empirical rate, but grounded in the
literal same numbers Week 1 used. Ran three strategies over the same
64,000-round budget, averaged over 200 independent simulations:

- **Fixed 1/3 split** — Week 1's actual design, replicated exactly.
- **Epsilon-greedy (ε=0.1)** — explore a random arm 10% of the time,
  otherwise exploit the current best.
- **Thompson sampling** — Beta-Bernoulli posterior per arm, pull whichever
  arm's random posterior draw is highest each round.

## 2. Cumulative regret

Regret is measured against the *true* per-arm rates — the standard,
low-variance way to score a bandit, since it isolates allocation quality
from the extra noise of realized 0/1 outcomes.

| Strategy | Final mean cumulative regret (foregone visits) |
|---|---:|
| Fixed 1/3 split | 2,303.0 |
| Epsilon-greedy (0.1) | 268.2 |
| Thompson sampling | **56.8** |

Fixed-split regret grows **linearly** for the entire 64,000-round test —
by construction, it never stops sending a third of traffic to the two
already-known-worse arms. Both bandits' regret grows **sub-linearly**,
flattening out as they concentrate traffic on Mens E-Mail. Thompson
sampling beats epsilon-greedy, the typical result in the literature: a
fixed exploration rate doesn't shrink once the posterior is confident, so
epsilon-greedy keeps paying a constant 10%-of-traffic exploration tax that
Thompson sampling mostly stops paying once it's confident.

**Traffic allocation at the end of the run:**

| Strategy | No E-Mail | Womens E-Mail | Mens E-Mail |
|---|---:|---:|---:|
| Fixed 1/3 split | 33.3% | 33.3% | 33.3% |
| Epsilon-greedy | 3.5% | 4.7% | 91.8% |
| Thompson sampling | 0.4% | 1.8% | **97.8%** |

**Business number**: Thompson sampling would have generated **~2,246
additional visits** over the same 64,000-email budget compared to the
fixed 1/3 split actually used — purely from allocating the same emails
better, no extra volume required.

## 3. The trade-off the regret number doesn't show

This is the part that's easy to skip past while celebrating the regret
reduction above. Fewer emails sent to the underperforming arms doesn't
just cut waste — it leaves less data to *confirm*, with a defensible
confidence interval, how much worse those arms actually were. Comparing
Womens vs. No-Email under each design's own realized sample sizes:

| Design | n (No-Email) | n (Womens) | Estimated diff | 95% CI width | p-value |
|---|---:|---:|---:|---:|---:|
| Fixed split | 21,333 | 21,339 | +0.0396 | 0.0128 | **<0.001** |
| Thompson sampling | 260 | 1,176 | +0.0227 | 0.0899 | **0.32** |

The fixed split's ~21,300-per-arm design gives an unambiguous,
publication-grade confidence interval on the Womens-vs-No-Email gap.
Thompson sampling — *precisely because* it succeeded at minimizing
regret by mostly stopping sending Womens and No-Email — leaves a
confidence interval **7x wider**, one that doesn't even exclude zero. Both
designs are working exactly as intended; they're just optimized for
different jobs, and this is the concrete cost of the one Thompson sampling
was built for.

## 4. When to use which

**Use a bandit when:**
- The test is an always-on or long-running allocation decision (which
  subject line, which recommendation, which price) rather than a one-off
  research question — the "test" and the "production traffic" are the
  same thing, so wasted traffic on a known-worse arm is real, ongoing
  opportunity cost, not just a research cost.
- The number of arms or the rate of change is too high for a human to keep
  re-running fixed tests (e.g. dozens of creative variants, prices that
  should adapt to shifting conditions).
- A directionally-correct, continuously-improving answer is worth more
  than a publication-grade confidence interval on every pairwise
  comparison.

**Use a fixed-horizon A/B test when:**
- The result needs to be defensible to a stakeholder, a board, or a
  regulator — "we're 95% confident the effect is between X and Y" is a
  sentence a bandit's own data usually can't support on the
  underperforming arms, as Section 3 shows directly.
- The decision is a one-time launch call, not an ongoing allocation — once
  the test ends, the value of further-optimized traffic during the test
  itself is much smaller than the value of a trustworthy answer.
- Multiple stakeholders need to independently audit or replicate the
  result — a fixed, pre-registered design is far easier to reason about
  than a sequential, outcome-dependent one.

**The honest middle ground**: this isn't strictly binary. A test can start
as a fixed-horizon design to get a clean read, then switch to bandit-style
allocation for the ongoing production traffic once the causal question is
answered — using the fixed test to *decide*, and the bandit to *exploit*
the decision at scale. That's a common pattern in practice, and a
reasonable answer if asked to pick just one: run the science as a fixed
test, run the business as a bandit.

## What's next

This closes the core five-week build (power analysis → CUPED → uplift
modelling → metrics framework → bandits). See
[writeups/one_pager.md](one_pager.md) for the portfolio-ready summary of
the whole project, and the optional Phase 6 stretch items in
[CLAUDE.md](../CLAUDE.md) for further extensions.
