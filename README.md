# Experimentation & Causal Inference Toolkit

A hands-on toolkit covering the core statistical judgment of a product data scientist:
power analysis and experiment design, variance reduction (CUPED), heterogeneous
treatment effect / uplift modelling, metrics framework design, and adaptive
allocation (bandits).

Built as a portfolio project for entry/beginner-level Product Data Scientist roles
(fintech, consulting, big tech).

## Motivating Gap

> The real gap isn't ML ability — it's that none of my existing projects would give
> an interviewer a reason to trust my experiment design or causal inference judgment.

Three existing repos (credit card fraud detection, a job-market NER pipeline, and a
financial sentiment/alpha project) demonstrate applied ML and engineering skill, but
they read as fraud/risk and NLP work. Product DS interviews at companies like Stripe
or Checkout.com skew heavily toward experiment design, causal inference, and metrics
judgment rather than pure model-building — this project exists to close that gap
directly, and to be fluently *explainable* in an interview, not just working code.

## Phase 0 Decisions (locked)

These decisions were made deliberately up front, to keep the project focused rather
than open-ended.

- **Two datasets, two different jobs**: the Hillstrom MineThatData Email Analytics
  dataset (small, clean) carries the fundamentals — power analysis, sequential
  testing, CUPED, and later the bandit comparison — precisely because it's simple
  enough that the *statistics* stay the visible subject, not the data wrangling. The
  Criteo Uplift dataset (large-scale) is reserved for heterogeneous treatment effects
  / uplift modelling, where realistic scale is the point.
- **Explainability over cleverness**: every notebook is written to be walked through
  line by line in an interview. Each technique gets working code *and* a markdown
  writeup explaining the why, not just the what.
- **Manual implementation where it aids understanding, libraries otherwise**: CUPED
  is implemented by hand (covariate adjustment via correlation with a pre-period
  outcome is more valuable to demonstrate directly than a black-box call), while
  power analysis, sequential testing, and uplift modelling lean on
  statsmodels/scipy/econml/causalml — well-known, defensible tools over custom
  reimplementations.
- **The targeting policy is a core deliverable, not an afterthought**: Week 3 doesn't
  stop at comparing uplift models — it turns model output into an actual targeting
  policy (treat where uplift exceeds cost of treatment) with an estimated
  incremental profit vs. blanket rollout, because that's the step that connects a
  model to a business decision.
- **The guardrail simulation must be a worked, runnable example**: Week 4 deliberately
  constructs a scenario where a primary metric improves while a guardrail (latency,
  churn, support tickets) degrades, and shows the framework catching it — a strong,
  concrete answer to "tell me about a time a metric lied to you," not just a written
  scenario.

## Goals

- Demonstrate power analysis / experiment design judgment (sample size, MDE, the
  peeking problem)
- Demonstrate CUPED variance reduction, implemented and explained from first
  principles
- Demonstrate heterogeneous treatment effects / uplift modelling at realistic scale,
  including a cost-based targeting policy
- Demonstrate metrics framework thinking — north star, guardrails, common failure
  modes (Simpson's paradox, novelty effects, network/spillover contamination)
- Demonstrate judgment on when to use a bandit vs. a fixed-horizon A/B test
- Pair every technique with a plain-English writeup that holds up under interview
  follow-up questions

## Repo Structure

```
experimentation-causal-inference/
├── CLAUDE.md
├── README.md
├── data/                  # raw + processed datasets (gitignored)
├── notebooks/
│   ├── 01_power_analysis.ipynb
│   ├── 02_cuped.ipynb
│   ├── 03_uplift_modeling.ipynb       # includes cost-based targeting policy
│   ├── 04_guardrail_simulation.ipynb
│   └── 05_bandits.ipynb
├── src/
│   ├── power_analysis.py
│   ├── cuped.py
│   ├── uplift.py
│   ├── targeting_policy.py
│   ├── guardrail_simulation.py
│   └── bandits.py
├── writeups/
│   ├── power_analysis.md
│   ├── cuped.md
│   ├── uplift_modeling.md
│   ├── metrics_framework.md
│   └── bandits_vs_ab_testing.md
└── requirements.txt
```

## Project Plan

### Phase 1 — Power Analysis & Experiment Design (Week 1)
- [x] Repo scaffolding, environment setup
- [x] EDA on the Hillstrom dataset
- [x] Power / MDE calculator: baseline rate + traffic + desired effect size → sample
      size & runtime
- [x] Sequential testing / peeking-problem demo (why early stopping inflates false
      positives)
- [x] Output: `notebooks/01_power_analysis.ipynb` + writeup

### Phase 2 — CUPED & Variance Reduction (Week 2)
- [x] Implement CUPED by hand using pre-experiment covariates on the Hillstrom data
- [x] Show variance reduction numerically and visually (CI width before/after)
- [x] Output: `notebooks/02_cuped.ipynb` + writeup with before/after CI plot

### Phase 3 — Heterogeneous Treatment Effects / Uplift Modelling (Week 3)
- [x] Switch to the Criteo Uplift dataset
- [x] Implement a T-learner or X-learner (stretch: causal forest via econml/causalml)
- [x] Identify and visualize segments with differential treatment response
- [x] Cost-based targeting policy: treat where uplift exceeds cost of treatment,
      estimate incremental revenue/profit vs. blanket rollout
- [x] Output: `notebooks/03_uplift_modeling.ipynb` + writeup covering both the model
      comparison and the targeting policy / profit estimate

### Phase 4 — Metrics Framework, Guardrails & Packaging (Week 4)
- [x] Define a hypothetical product scenario (checkout flow or onboarding funnel)
- [x] Metrics framework doc: north star metric, guardrail metrics, common failure
      modes
- [x] Guardrail violation simulation: a worked, runnable example where the primary
      metric improves but a guardrail degrades
- [x] Polish all notebooks/plots for portfolio and interview use
- [x] Output: `notebooks/04_guardrail_simulation.ipynb`,
      `writeups/metrics_framework.md`, final README, one-page summary doc

### Phase 5 — Bandits & Adaptive Allocation (Week 5)
- [x] Implement a multi-armed bandit (Thompson sampling or epsilon-greedy) on the
      same Hillstrom dataset used in Phase 1
- [x] Compare cumulative regret / results against the fixed-horizon A/B test
- [x] Output: `notebooks/05_bandits.ipynb` + writeup on when to use a bandit vs. a
      fixed A/B test

### Optional Phase 6 — Further Stretch
- [ ] Streamlit app wrapping the power calculator
- [ ] Blog-style writeup of CUPED + uplift findings for LinkedIn/portfolio
- [ ] Bayesian vs. frequentist comparison on the Hillstrom test
- [ ] Short note on interference/network effects (cluster randomization, switchback
      tests)

## Tech Stack

- **Language**: Python
- **Data/stats**: pandas, numpy, scipy, statsmodels, scikit-learn
- **Causal inference**: econml, causalml
- **Datasets**: Hillstrom MineThatData Email Analytics (fundamentals, small/clean),
  Criteo Uplift Modeling Dataset (heterogeneous treatment effects, realistic scale)
- **Notebooks**: Jupyter
- **Visualization**: matplotlib, seaborn — portfolio-quality, labeled and titled,
  exportable as PNG

## Key Design Principles

1. **Clarity and explainability over cleverness** — every notebook reads like
   something to walk an interviewer through line by line.
2. **Code and writeup, paired, always** — a technique isn't done until both the
   implementation and the plain-English *why* exist.
3. **Defensible libraries by default, manual implementation where it teaches
   something** — statsmodels/scipy/econml/causalml unless hand-rolling the logic
   (e.g. CUPED) builds real understanding.
4. **Business framing is the finish line** — uplift modelling ends in a targeting
   policy and a profit estimate, not just a model comparison.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Datasets are gitignored (see `data/raw/`). To fetch the Hillstrom dataset used
in Phases 1, 2, and 5:

```bash
curl -L -o data/raw/hillstrom.csv.gz \
  https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz
gunzip data/raw/hillstrom.csv.gz
```

To fetch the Criteo Uplift dataset used in Phase 3 (~311MB compressed,
~3.2GB extracted):

```bash
curl -L -o data/raw/criteo-uplift.csv.gz \
  https://criteostorage.blob.core.windows.net/criteo-research-datasets/criteo-uplift-v2.1.csv.gz
gunzip data/raw/criteo-uplift.csv.gz
```

## Status

✅ Core build complete — Phases 1-5 (power analysis, CUPED, uplift modelling,
metrics framework & guardrails, bandits) done. See checkboxes above for
progress and [CLAUDE.md](CLAUDE.md) for the full build plan and working
conventions. Optional Phase 6 stretch items remain open. A one-page
portfolio summary of the whole project is at
[writeups/one_pager.md](writeups/one_pager.md).

**Phase 1 summary:** Built a two-proportion-test power/MDE calculator
(`src/power_analysis.py`, cross-checked against `statsmodels`), used it to show
the real Hillstrom email test was well-powered for the effect it found, and ran
a from-scratch simulation showing that "peeking" at a fixed-horizon test and
stopping at the first significant read inflates the false positive rate from a
nominal 5% to ~24% — a concrete demonstration of experiment design judgment,
not just a working calculator. See `notebooks/01_power_analysis.ipynb` and
`writeups/power_analysis.md`.

**Phase 2 summary:** Implemented CUPED by hand (`src/cuped.py`, both
single- and multi-covariate versions, cross-checked against the theoretical
`1 - corr²`/`1 - R²` variance reduction), checked real pre-treatment covariate
strength before applying it — finding Hillstrom's covariates are weak for
`visit` and negligible for `conversion`/`spend`, an honest data-driven call
rather than blind application — and used a labeled synthetic example to show
the ~36% variance reduction CUPED delivers with a strong covariate, then tied
that reduction back to the Week 1 calculator as an equivalent traffic/MDE
saving. See `notebooks/02_cuped.ipynb` and `writeups/cuped.md`.

**Phase 3 summary:** Switched to the Criteo Uplift dataset (~14M rows) and
implemented S-, T-, and X-learner uplift meta-learners (`src/uplift.py`),
confirming clean randomization first (R²≈0 for treatment regressed on all
features). Evaluated with a Qini curve and found the simplest model
(S-learner) narrowly topped the leaderboard despite its known shrinkage
bias — a concrete lesson in checking the metric rather than assuming the
fancier learner wins — then built a cost-based targeting policy
(`src/targeting_policy.py`) on the X-learner, evaluated via inverse-
propensity weighting on real held-out outcomes, showing a targeted policy
nets 37% more profit than blanket rollout while treating only 12.5% of
users. See `notebooks/03_uplift_modeling.ipynb` and
`writeups/uplift_modeling.md`.

**Phase 4 summary:** Defined a checkout-flow scenario ("Quick Checkout")
with a north star (checkout completion) and a targeted guardrail (delivery-
failure rate), then built a worked, runnable simulation
(`src/guardrail_simulation.py`) where the primary metric improves
significantly while the guardrail degrades significantly — diagnosed the
regression down to a specific subgroup, and showed the $-based decision
flips the naive "ship it" call to a net loss of ~$422K per million
sessions. Also built compact runnable demos of Simpson's paradox (a pooled
result that reverses sign vs. every subgroup), novelty effects (a real
effect that decays, so even a full-length cumulative readout overstates
long-run impact ~3x), and network/spillover contamination (found — and
reported honestly — that a shared-inventory constraint compressed rather
than inflated the measured effect). See `notebooks/04_guardrail_simulation.ipynb`
and `writeups/metrics_framework.md`.

**Phase 5 summary:** Implemented epsilon-greedy and Thompson sampling
bandits from scratch (`src/bandits.py`) and ran them against Week 1's
actual fixed-split design on the same 64,000-email Hillstrom budget, with
every simulated reward drawn from the real recorded outcomes. Thompson
sampling cut cumulative regret by ~97.5% vs. the fixed split (56.8 vs.
2,303 foregone visits) — worth ~2,246 extra visits on the same budget with
zero extra spend. But its own data couldn't reproduce Week 1's clean
result: the same Womens-vs-No-Email comparison that was significant at
p<0.001 under the fixed split's ~21,300-per-arm sample came out
non-significant (p=0.32, CI crossing zero) under Thompson sampling's own
~260-to-1,176-pull sample sizes — a concrete demonstration of exactly what
a bandit's efficiency gain costs a stakeholder report. See
`notebooks/05_bandits.ipynb` and `writeups/bandits_vs_ab_testing.md`.
