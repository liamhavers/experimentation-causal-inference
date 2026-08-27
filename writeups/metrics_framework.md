# Metrics Framework

**Notebook:** `notebooks/04_guardrail_simulation.ipynb`
**Code:** `src/guardrail_simulation.py`

## Why this matters for a product DS role

Weeks 1-3 built the statistical machinery — power, variance reduction,
heterogeneous effects. None of it protects you from measuring the wrong
thing. A metrics framework is the layer that decides *what* gets measured
and *what counts as a problem*, before any test runs — and this week's
worked examples are all versions of the same interview question: "tell me
about a time a metric lied to you."

## 1. The scenario

A hypothetical e-commerce checkout flow. The team is testing **"Quick
Checkout"**: for returning users with a saved shipping address, remove the
address-confirmation screen and go straight from cart to payment.

**North star (for this funnel): checkout completion rate** — orders placed
divided by checkout sessions started. It's the metric the test is designed
to move, it's directly and immediately attributable to the change (no long
lag, no multi-step causal chain to a downstream business metric), and it's
the natural unit for comparing this test against any other checkout-funnel
idea competing for the same roadmap slot.

**Guardrail: delivery-failure rate among orders placed.** Chosen for a
specific, mechanistic reason, not as a generic "make sure nothing breaks"
catch-all: removing the confirmation step removes a chance to catch a
stale saved address before it ships. A guardrail is most useful when it's
targeted at a plausible, specific failure mode of *this* change, not a
scattershot list of every metric the team tracks — a good guardrail set is
short enough that someone can actually hold it in their head while making
the ship call.

**A reasonable second guardrail for a real launch** would be checkout page
load latency (removing a step should help, not hurt, latency — a
regression there would itself be a signal something's implemented wrong).
Not simulated here, to keep the worked example focused on one clean
mechanism.

## 2. The guardrail violation: worked example

Simulated 200,000 users per arm (`src/guardrail_simulation.py:simulate_checkout_experiment`).
Quick Checkout genuinely lifts completion; independently, ~10% of users
have a stale saved address nobody knows about, and skipping confirmation
means only *their* orders in the treatment arm ship to the wrong place.

| Metric | Control | Treatment | z | p-value |
|---|---:|---:|---:|---:|
| Checkout completion rate | 61.9% | 63.8% | 12.26 | ≈0 |
| Delivery-failure rate (among orders) | 0.99% | 4.86% | 57.40 | ≈0 |

Both moves are real and both are highly significant at n=200K/arm — this
isn't a case of an underpowered guardrail throwing a noisy false alarm
(Week 1's power calculator cuts both ways: it's what makes this guardrail
signal trustworthy rather than dismissible). A launch decision now has two
significant, conflicting signals.

**Diagnosis.** Splitting the guardrail rate by the (normally hidden)
stale-address flag shows the regression isn't spread across all treatment
users — it's entirely concentrated in the ~10% with a stale address
(0.39 failure rate there vs. ~0.01 everywhere else). This is the real
payoff of tracking a guardrail at all: it doesn't just flag a problem, it
points at the mechanism, which turns directly into a fix (silently
re-validate the saved address against a shipping API, rather than removing
the check outright) instead of a flat no-ship verdict.

**The decision.** Two significant metrics moving in opposite directions
can't be resolved by eyeballing percentages — they need a common unit.
Converting both to profit per checkout session (illustrative: $24 profit
per order, $35 cost per guardrail failure — return shipping, reshipping,
support handling, typical goodwill compensation) nets them directly:

| | Control | Treatment |
|---|---:|---:|
| Profit per session | $14.6450 | $14.2229 |

**Net impact: -$0.42 per session, or roughly -$422,000 per million checkout
sessions.** The conversion lift ($0.48/session in profit at these
assumptions) doesn't cover the cost of the failed deliveries it causes
($0.88/session). **Decision: hold, don't ship as-is** — but the diagnosis
means the recommendation isn't just "no," it's "re-test with silent
address revalidation instead of removing the check."

This is deliberately built to be uncomfortable: the primary metric wasn't
wrong, exactly — it just wasn't the whole story, and shipping on it alone
would have been a real, quantifiable, avoidable loss.

## 3. Common failure modes

Three more mechanisms, each demonstrated with a runnable simulation rather
than just described.

### Simpson's paradox

Simulated a phased rollout (treatment share climbing from 10% to 70% of
traffic over 14 days) running alongside an unrelated traffic-mix shift
(desktop share also climbing over the same days, e.g. a marketing push).
True effect is **negative** on both desktop (-0.0107) and mobile (-0.0041)
— but the naive pooled comparison shows **+0.0104**, a full sign reversal.

Nothing about the statistics is wrong; a confounder (device mix) is
correlated with both the metric (desktop converts far better regardless of
treatment) and the exposure (treatment share happened to climb alongside
desktop share). **In a real test this is almost always a symptom of an
allocation problem** — a ramping rollout, a mid-test change in traffic
sources, a cohort-gated feature — not a statistical curiosity. The fix is
the same diligence as the Week 1 randomization check, applied
*continuously*: verify the treatment/control split and key segment mix
stay stable across the whole test window, and report key results
stratified by any segment that plausibly shifted, not just pooled.

### Novelty effects

Simulated a treatment effect starting at a 5pp lift and decaying
exponentially toward a ~0.6pp steady state as users habituate.

- Day-3 cumulative estimate: **+0.045** — a **~7-8x** overstatement of the
  true steady-state lift (+0.006).
- Full 21-day cumulative estimate: **+0.019** — still a **~3x**
  overstatement.

The second number is the sharper, less obvious point. Running the test
longer helps less than it sounds like it should, because a *cumulative*
estimate averages in every day seen so far — the high-novelty early days
stay diluted into the average forever, never fully gone. "Run it longer"
only fixes novelty bias if the readout is the **trailing/marginal** effect
(e.g. the last few days only), not the cumulative-to-date number most
dashboards default to showing. This compounds with (but is distinct from)
Week 1's peeking problem: a team peeking early is also disproportionately
likely to be looking at the inflated novelty period.

### Network / spillover contamination

Simulated treatment and control users arriving in random order and
competing for one shared, finite inventory pool (e.g. a promo that lifts
treatment purchase intent, drawing down stock control is also trying to
buy from) — a SUTVA violation, where one unit's assignment affects
another's outcome.

| | Oracle (unlimited inventory) | Contaminated (shared inventory) |
|---|---:|---:|
| Measured effect | +0.103 | +0.034 |

**Worth stating plainly, because it cuts against the common assumption:**
the contaminated effect here is *smaller* than the true effect, not
larger. It's tempting to assume interference always inflates a measured
effect ("treatment steals outcomes from control"), but with a hard shared
capacity constraint and randomly interleaved arrivals, once inventory runs
out *neither* arm can convert further — both get capped toward the same
ceiling, compressing the visible gap rather than widening it. **The
takeaway isn't "spillover biases high" or "biases low" — it's that any
shared resource, budget, or population between arms is worth checking for,
because the direction of the bias depends on the specific mechanism, not
on a general rule.**

## 4. A decision checklist

Distilled from the four examples above, roughly in the order a launch
review should actually apply them:

1. **Before the test**: does this change have an obvious, specific failure
   channel (like the stale-address mechanism)? Name a guardrail for it —
   not a generic list, a targeted one.
2. **During the test**: is the treatment/control split and the mix of any
   plausibly-relevant segment (device, traffic source, cohort) stable
   across the whole window? A ramping rollout or a mid-test traffic change
   is a Simpson's-paradox risk, not just an operational detail.
3. **At readout**: is the effect still moving? Compare the trailing/recent
   effect to the cumulative-to-date effect — a gap between them is a
   novelty (or reverse-novelty) signal, and the cumulative number alone
   will understate how much further it might still shift.
4. **At readout**: did the guardrail move too, and is it well-powered
   enough that "it didn't move" is actually informative rather than just
   underpowered? (Week 1's calculator, applied to the guardrail metric,
   not just the primary one.)
5. **Before shipping**: if the primary metric and a guardrail disagree,
   convert both to a common unit (revenue, cost, or another shared
   currency) rather than debating percentages against each other. If they
   can't be shipped independently, the real question is the net, not
   which one "wins."
6. **If something regressed**: segment before concluding the whole
   population is affected. A regression concentrated in one subgroup is a
   fixable mechanism; the same regression spread evenly is a much harder
   "the change itself is bad" conclusion.

None of this replaces the statistical tools from Weeks 1-3 — it's the
judgment layer that decides which numbers from those tools actually matter
for a given decision, which is most of what separates a working notebook
from a defensible product recommendation.
