# Interference & Network Effects

**Notebook:** `notebooks/08_interference_network_effects.ipynb`
**Code:** `src/interference.py`
**Type:** short note — a SUTVA-violation demo plus the two designs that
handle it.

## Why this matters for a product DS role

Every test in Weeks 1-6 assumed **SUTVA**: unit `i`'s outcome depends only
on `i`'s own treatment assignment. That is false whenever units influence
each other — social products (referrals, feeds, messaging), marketplaces
(shared inventory, drivers, ad auctions), anything with network effects. A
naive A/B test then measures something other than the launch effect, and
"how would you test a feature that spreads between users" is a standard
interview question for exactly those companies. This note makes the failure
concrete and shows the two standard fixes — **cluster randomization** and
**switchback tests** — working on simulated data, each with its real cost.

Week 4 already has a worked *marketplace* example
(`simulate_shared_inventory_spillover`: a finite shared inventory that made
treatment and control compete, and — reported honestly — *compressed* the
measured effect rather than inflating it). This note adds the
**social-graph** case.

## 1. The setup

`make_clustered_network` builds a stochastic block model — 50 clusters of
48 units, dense within a cluster, very sparse across (**96.7% of edges stay
inside one cluster**). Outcome model:

```
p_i = baseline + direct·T_i + spillover·(fraction of i's neighbours treated)   [+ per-cluster shift]
```

with `baseline = 0.20`, `direct = 0.04`, `spillover = 0.07`. The spillover
term is the SUTVA violation: a control unit surrounded by treated
neighbours still does better.

The estimand a launch decision needs is the **global average treatment
effect** — everyone treated vs. nobody treated. Neighbour-treated fraction
is 1 vs. 0 in those two worlds, so `GATE = direct + spillover = 0.11`.

## 2. Individual randomization is biased under spillover

Assign each unit independently 50/50, take the naive difference in means,
repeat 400×:

| | value |
|---|---:|
| mean estimate | **+0.039** |
| true global effect | +0.110 |
| direct effect alone | +0.040 |
| bias vs. GATE | **−0.071 (−65%)** |

The estimate collapses onto the **direct effect alone** and misses roughly
two-thirds of the true impact. At a 50/50 unit-level split every unit —
treated or control — has about half its neighbours treated, so the
spillover term is nearly equal in both arms and cancels in the difference.
The control group is contaminated by construction, and a denser graph makes
it worse (`writeups/figures/29_individual_randomization_bias.png`).

## 3. Cluster randomization: near-unbiased, at a variance cost

Randomize **whole clusters**. A treated unit now sits in a mostly-treated
neighbourhood and a control unit in a mostly-control one, so the spillover
term stops cancelling and the estimate recovers almost all of
`direct + spillover`.

| Design | mean estimate | sampling SD | bias vs. GATE |
|---|---:|---:|---:|
| Individual | +0.039 | 0.018 | −0.071 |
| Cluster (identical clusters) | +0.107 | 0.018 | −0.003 |
| Cluster (heterogeneous clusters, shift sd = 6pp) | **+0.108** | **0.022** | −0.002 |

Costs:

- **Variance / effective sample size.** The unit of randomization is the
  cluster (50), not the user (2,400). With heterogeneous clusters the
  sampling SD here is ~1.3× the individual design's. That multiplier is the
  **design effect `1 + (m − 1)·ICC`** — with large clusters or strong
  within-cluster correlation it routinely reaches 5-20×, so you need far
  more total users for the same precision.
- **The SE must be computed at the cluster level.** Treating 2,400 units as
  independent reported an SE of 0.018 against an honest cluster-level SE of
  0.028 — badly over-confident.

(`writeups/figures/30_cluster_vs_individual.png`.)

## 4. Switchback tests for the marketplace case

When units genuinely can't be separated — one shared pricing / matching
pool — you separate treatment and control **in time**, switching the whole
system between arms over blocks of periods. `simulate_switchback` models the
two frictions (480 periods, 400 simulations per setting):

| Block length | Carryover bias (no washout) | Bias with 1-period washout | True sampling SD | Naive IID SE |
|---:|---:|---:|---:|---:|
| 1 | **−0.050** (−50%) | n/a — washout discards everything | 0.0048 | 0.0051 |
| 2 | −0.025 | −0.000 | 0.0059 | 0.0050 |
| 4 | −0.012 | +0.000 | 0.0065 | 0.0048 |
| 8 | −0.007 | −0.000 | 0.0076 | 0.0047 |
| 24 | −0.002 | +0.000 | 0.0078 | 0.0047 |

- **Settling / carryover.** The first period after a switch is only
  half-transitioned. Short blocks are almost all first-periods, so the
  measured effect is dragged toward zero — ~−50% at block length 1.
- **Washout** (drop the first period of each block) removes the bias, but
  is unaffordable at block length 1 (nothing survives).
- **Longer blocks cost precision.** Fewer, longer blocks = fewer
  independent randomisation units, so the true sampling SD *grows* with
  block length. The naive IID SE stays flat and so understates the real
  uncertainty by ~35% at block length 12; a block-level SE tracks the
  truth.
- **Sweet spot:** a block long enough that one washout period is a small
  fraction of it, short enough to keep enough blocks for precision and to
  limit exposure to time-of-day / day-of-week trends.

(`writeups/figures/31_switchback_tradeoffs.png`.)

## 5. Decision guidance

| Symptom | Likely interference | Design fix | Main cost |
|---|---|---|---|
| Treated users influence friends; control "leaks up" | Social-graph | Cluster randomization at a level containing most edges (city, school, org) | Effective n = #clusters; design effect inflates variance; cluster-level SE required |
| One shared pool (inventory, drivers, auction, prices) | Marketplace / resource | Switchback over time blocks, or cluster by geo/market | Carryover; needs washout; fewer effective units; time-trend exposure |
| Effect bigger in a small pilot than full rollout | Saturation-dependent spillover | Two-stage randomization (cluster saturation, then units within) | Complex analysis; needs many clusters |

**When it's fine to ignore interference:** weak or no ties between units
(most B2C purchase-conversion tests), very low treatment share (a 1%
holdout barely perturbs the network), or a purely individual mechanism (a
checkout-page layout). But the Week 4 shared-inventory result cuts the
other way — interference there *compressed* the effect — so the safe
assumption is "unknown sign," not "this is conservative."

**Detecting it before it burns you:**

- Run at two treatment saturations (e.g. 10% and 50% of each cluster) — if
  the per-unit effect moves with saturation, spillover is real.
- Check for a dose-response: does a control unit's outcome rise with its
  number of treated neighbours?
- Run a cluster-randomized and an individually-randomized arm on the same
  feature — the gap between them *is* the spillover.

## When a real RCT is still the right call

Same caveat as Week 6: none of this is a reason to prefer a quasi-design
when a clean one is available. Cluster randomization and switchbacks *are*
randomized experiments — they just move the unit of randomization to
contain the interference. Reach for them when interference is real and
first-order; keep the simple unit-level A/B test (Weeks 1-5) when ties are
weak, saturation is low, or the mechanism is individual.

## What this adds

Weeks 1-6 assumed SUTVA. This note shows an individually randomized test on
a connected network recovering only the direct effect — missing two-thirds
of the true impact — and the two standard fixes working on the same data,
each with its price: cluster randomization pays in variance (the design
effect), and switchback trades unit interference for temporal interference.
Completes the optional Phase 7 stretch items. See `writeups/one_pager.md`
for where it sits in the whole project.
