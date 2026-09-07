# Experimentation & Causal Inference Toolkit

## Project purpose

This is a portfolio project built to close a specific gap in my job search for
**entry/beginner-level Product Data Scientist roles** (fintech, consulting, big tech).

Context: I currently have three GitHub projects (fraud detection with XGBoost/LightGBM,
a job-market NER pipeline, and an in-progress financial sentiment project) that
demonstrate strong applied ML and engineering skill, but read as fraud/risk and NLP
work — not product analytics. Product DS interviews (e.g. Stripe, Checkout.com-style
roles) skew heavily toward experiment design, causal inference, and metrics judgment
rather than pure model-building. This project exists to directly demonstrate:

- Power analysis / experiment design judgment
- CUPED variance reduction
- Heterogeneous treatment effects / uplift modelling
- Metrics framework thinking (north star, guardrails, common failure modes)

The write-up/explanation quality matters as much as the code — the goal is to be able
to *explain* these concepts fluently in an interview, not just have working notebooks.

## My background (for tone/calibration)

- Technical Data Analyst (SO) at HMRC, ~2 years experience
- MSc Data Science (Merit), BSc Mathematics (2:1), Nottingham Trent University
- Comfortable with Python (pandas, sklearn), building toward causal inference fluency
- Existing GitHub: github.com/liamhavers

## Datasets

1. **Hillstrom MineThatData Email Analytics Dataset** — used for power analysis,
   sequential testing, and CUPED (Weeks 1–2). Small, clean, good for fundamentals.
   Reused in Week 5 (bandits) and Week 6 (propensity matching validated against
   the known RCT effect).
2. **Criteo Uplift Modeling Dataset** — used for heterogeneous treatment effects /
   uplift modelling at realistic scale (Week 3). Large — be mindful of memory/runtime.
   Reused in Week 6: its `treatment`/`exposure` columns are a real one-sided
   noncompliance instrumental-variables setup, not something built for the
   occasion.

## Build plan (6 weeks)

### Week 1 — Power Analysis & Experiment Design
- Repo scaffolding, environment setup (pandas, scipy, statsmodels, numpy, matplotlib)
- EDA on Hillstrom dataset
- Power/MDE calculator: baseline rate + traffic + desired effect size → sample size & runtime
- Sequential testing / peeking-problem demo (why early stopping inflates false positives)
- Output: `notebooks/01_power_analysis.ipynb` + short markdown writeup

### Week 2 — CUPED & Variance Reduction
- Implement CUPED using pre-experiment covariates on Hillstrom data
- Show variance reduction numerically and visually (CI width before/after)
- Writeup: why CUPED works (covariate adjustment via correlation with pre-period outcome)
- Output: `notebooks/02_cuped.ipynb` + writeup with before/after CI plot

### Week 3 — Heterogeneous Treatment Effects / Uplift Modelling
- Switch to Criteo Uplift dataset
- Implement T-learner or X-learner (stretch: causal forest via econml or causalml)
- Identify and visualize segments with differential treatment response
- Writeup: trade-offs between S-learner, T-learner, X-learner
- **Cost-based decision layer**: turn uplift model output into an actual targeting
  policy (treat users where uplift > cost of treatment), and estimate incremental
  revenue/profit from targeting vs. blanket rollout. This is the step that connects
  the model to a business decision — treat it as a core deliverable of the week,
  not an afterthought.
- Output: `notebooks/03_uplift_modeling.ipynb` + writeup covering both the model
  comparison and the targeting policy / profit estimate

### Week 4 — Metrics Framework, Guardrails & Packaging
- Define a hypothetical product scenario (checkout flow or onboarding funnel)
- Write metrics framework doc: north star metric, guardrail metrics, common failure
  modes (Simpson's paradox, novelty effects, network/spillover contamination)
- **Guardrail metric violation simulation**: deliberately construct a scenario where
  the primary metric improves but a guardrail (e.g. latency, churn, support tickets)
  degrades, and show how the framework catches it. This should be a worked, runnable
  example, not just a written scenario — it's a strong answer to "tell me about a
  time a metric lied to you."
- Clean up notebooks, write strong top-level README (problem → approach → results →
  business framing)
- Polish all plots for portfolio/interview use
- Output: `notebooks/04_guardrail_simulation.ipynb`, `writeups/metrics_framework.md`,
  final README, polished repo, one-page summary doc

### Week 5 — Bandits & Adaptive Allocation
- Implement a simple multi-armed bandit (Thompson sampling or epsilon-greedy) on the
  same Hillstrom dataset used for the Week 1 fixed-horizon A/B test
- Compare cumulative regret / results against the fixed-horizon approach
- Writeup: when to use a bandit vs. a fixed A/B test (bandits for revenue-sensitive
  ramps and continuous optimization, fixed A/B for clean causal reads needed for
  reporting/stakeholder trust)
- Output: `notebooks/05_bandits.ipynb` + writeup

### Week 6 — Quasi-Experimental Methods
Added after the original 5-week plan, once Weeks 1-5 were complete: both prior
datasets are RCTs, and quasi-experimental methods exist specifically for when
random assignment *isn't* available — so this week deliberately builds two
scenarios where that gap is real (Criteo) or engineered (Hillstrom), each with
a way to check the answer against a known truth rather than trusting the
method on faith.
- **Instrumental variables** on Criteo's real `treatment` (instrument) →
  `exposure` (endogenous, one-sided noncompliance — exposure is only ever 1
  when treatment=1) → `visit`/`conversion` (outcome). Recover the LATE via
  the Wald estimator and cross-check with 2SLS (`linearmodels`), and contrast
  against the naive as-treated (exposed vs. unexposed) estimate to show why
  the naive comparison is confounded — no synthetic construction needed, this
  is a genuine natural experiment already sitting in data fetched in Week 3.
- **Propensity score matching / IPW validated against a known RCT effect**
  on Hillstrom (Mens E-Mail vs. No E-Mail, true ATE known from Week 1).
  Deliberately construct a confounded observational-style subsample by
  biasing *retention* (not outcomes — real recorded outcomes throughout) on
  real pre-treatment covariates, show the naive comparison on that biased
  sample is wrong, then recover the true effect with PSM, IPW, and
  regression adjustment — the standard way causal-inference methodologists
  validate an estimator (LaLonde-style benchmarking) before trusting it on
  data where the truth isn't known.
- Writeup: IV and PSM/IPW assumptions (relevance, exclusion restriction,
  monotonicity; unconfoundedness, overlap), and when a quasi-experimental
  method is the right call vs. when a real RCT (Weeks 1-5) is achievable and
  should be preferred.
- Output: `notebooks/06_quasi_experimental_methods.ipynb` + writeup

### Optional Week 7 (further stretch)
- Streamlit app wrapping the power calculator
- Blog-style writeup of CUPED + uplift findings for LinkedIn/portfolio site
- Bayesian vs. frequentist comparison on the Hillstrom test
- Short note on interference/network effects (cluster randomization, switchback tests)
- Short recorded walkthrough (Loom-style) of the CUPED and uplift sections

## Repo structure (target)

```
experimentation-causal-toolkit/
├── CLAUDE.md
├── README.md
├── data/                  # raw + processed datasets (gitignored if large)
├── notebooks/
│   ├── 01_power_analysis.ipynb
│   ├── 02_cuped.ipynb
│   ├── 03_uplift_modeling.ipynb       # includes cost-based targeting policy
│   ├── 04_guardrail_simulation.ipynb
│   ├── 05_bandits.ipynb
│   └── 06_quasi_experimental_methods.ipynb
├── src/
│   ├── power_analysis.py
│   ├── cuped.py
│   ├── uplift.py
│   ├── targeting_policy.py
│   ├── guardrail_simulation.py
│   ├── bandits.py
│   ├── instrumental_variables.py
│   └── propensity_matching.py
├── writeups/
│   ├── power_analysis.md
│   ├── cuped.md
│   ├── uplift_modeling.md
│   ├── metrics_framework.md
│   ├── bandits_vs_ab_testing.md
│   └── quasi_experimental_methods.md
└── requirements.txt
```

## Working conventions for Claude Code

- Prioritize **clarity and explainability** over cleverness — every notebook should
  read like something I could walk an interviewer through line by line.
- Each technique (power analysis, CUPED, uplift) needs both working code AND a
  markdown writeup explaining the *why*, not just the *what*.
- Keep statistical explanations precise but accessible — I have a strong maths
  background (BSc Maths) but am still building causal inference fluency specifically.
- Favor well-known, defensible libraries (statsmodels, scipy, econml/causalml) over
  custom implementations, except where implementing from scratch aids understanding
  (e.g. a manual CUPED implementation is more valuable here than a black-box library call).
- Plots should be portfolio-quality: labeled, titled, exportable as PNG for use in
  a CV/portfolio site or interview deck.
- After each week's work, help me draft a 2-3 sentence plain-English summary of what
  was built and why it matters for a product DS role — I'll use these in cover letters
  and interview prep.

## Current status

Week 1 (power analysis & experiment design) complete: repo scaffolded, Hillstrom
dataset fetched into `data/raw/` (gitignored, see README for the fetch command),
`src/power_analysis.py` implements the sample-size/MDE calculator plus a peeking
simulation, `notebooks/01_power_analysis.ipynb` runs EDA + a randomization
balance check + the calculator + the peeking demo end to end, and
`writeups/power_analysis.md` has the full writeup.

Week 2 (CUPED & variance reduction) complete: `src/cuped.py` implements CUPED
by hand (single- and multi-covariate), `notebooks/02_cuped.ipynb` checks real
covariate strength on Hillstrom before applying it (weak for `visit`,
negligible for `conversion`/`spend`), demonstrates the raw result honestly,
then uses a labeled synthetic strong-covariate example to show the method's
full effect, and ties variance reduction back to the Week 1 power calculator.
`writeups/cuped.md` has the full writeup.

Week 3 (heterogeneous treatment effects / uplift modelling) complete:
switched to the Criteo Uplift dataset (~14M rows, fetched into `data/raw/`,
see README for the fetch command), `src/uplift.py` implements S-, T-, and
X-learner meta-learners plus Qini-curve and per-decile evaluation,
`src/targeting_policy.py` implements a cost-based targeting policy with
IPW-based offline policy evaluation. `notebooks/03_uplift_modeling.ipynb`
confirms clean randomization, compares the three learners (S-learner
narrowly won on Qini despite its shrinkage bias — noted honestly rather
than assumed away), visualizes response heterogeneity, and estimates the
targeting policy's profit vs. blanket rollout (+37% profit at 12.5% of
users treated, with a cost-sensitivity sweep). `writeups/uplift_modeling.md`
has the full writeup.

Week 4 (metrics framework & guardrails) complete: `src/guardrail_simulation.py`
implements four simulations — a checkout-flow "Quick Checkout" scenario
where a primary metric (checkout completion) improves while a guardrail
(delivery-failure rate) degrades, diagnosed to a specific subgroup and
priced into a net-negative $ decision (~$422K loss/million sessions); plus
Simpson's paradox (a ramping-rollout confound reversing the pooled sign),
novelty effects (a decaying true effect that even a full-length cumulative
readout overstates ~3x), and shared-inventory spillover (found to compress
rather than inflate the measured effect — reported as found, not forced to
match the initial hypothesis). `notebooks/04_guardrail_simulation.ipynb`
runs all four end to end. `writeups/metrics_framework.md` has the north
star/guardrail framework, failure-mode writeups, and a decision checklist.
`writeups/one_pager.md` is a portfolio-ready summary of the project.

Week 5 (bandits vs. fixed-horizon A/B testing) complete: `src/bandits.py`
implements epsilon-greedy and Thompson sampling from scratch, vectorized
across simulations, with rewards drawn via bootstrap resampling from
Hillstrom's real recorded `visit` outcomes. `notebooks/05_bandits.ipynb`
runs both against Week 1's actual fixed-split design on the same
64,000-email budget: Thompson sampling cuts cumulative regret ~97.5%
(56.8 vs. 2,303 foregone visits, worth ~2,246 extra visits at zero extra
spend), but its own sample sizes can't reproduce Week 1's significant
Womens-vs-No-Email result (p=0.32 vs. p<0.001 under the fixed split) — a
concrete demonstration of the bandit/fixed-test trade-off rather than an
assertion of it. `writeups/bandits_vs_ab_testing.md` has the full writeup,
including a decision guide for when to use which. This completes the core
5-week build.

Week 6 (quasi-experimental methods) complete, added after the original
5-week plan since Weeks 1-5 all used real RCTs: `src/instrumental_variables.py`
implements the Wald estimator, a first-stage-strength (weak-instrument)
diagnostic, and a manual 2SLS for illustration; `src/propensity_matching.py`
implements propensity estimation, standardized-mean-difference balance
checks, nearest-neighbor matching, IPW, and regression adjustment. Part A
of `notebooks/06_quasi_experimental_methods.ipynb` uses Criteo's real
`treatment`/`exposure` columns (genuine one-sided noncompliance, not
constructed) as an instrument, recovering a LATE of 0.287 on `visit`
(exact match to `linearmodels`' IV2SLS) against a naive as-treated estimate
that overstates the true effect by ~32%. Part B deliberately confounds a
resampled version of the Hillstrom RCT (real outcomes, biased retention)
and validates matching/IPW/regression adjustment against the already-known
true ATE (0.0766) — all three recover it closely (IPW within 0.5%) where
the naive comparison is 56.5% too high. `writeups/quasi_experimental_methods.md`
has the full writeup, including assumption checklists and a decision guide
for IV vs. propensity methods vs. just running an RCT.
`writeups/one_pager.md` reflects all 6 weeks.

Week 7 (optional stretch) started. Scope decided with the user: Bayesian
vs. frequentist first, Streamlit power calculator second, interference/
network-effects note third; the blog-style CUPED+uplift writeup was
dropped.

Week 7a (Bayesian vs. frequentist) complete: `src/bayesian_ab.py`
implements the Beta-Bernoulli conjugate analysis from scratch — posterior
update, credible intervals, Monte-Carlo `P(treatment > control)` with
reported MC error, posterior lift summary, decision-theoretic expected
loss, a frequentist two-proportion wrapper, and a vectorized Bayesian
optional-stopping simulator. `notebooks/07_bayesian_vs_frequentist.ipynb`
runs the same Hillstrom Womens-vs-No-Email `visit` comparison both ways:
at full sample the credible interval and the Wald CI match to four decimal
places (the divergence is only in what each may claim); at n=150/arm the
z-test is inconclusive (p=0.31) while the Bayesian expected-loss framing
still supports a ship decision; a prior-sensitivity sweep shows the prior
only moves the posterior when it is both informative and competing with a
small sample; and a Bayesian peeking demo (`P(T>C) > 0.95` stop-early
rule) inflates the false-win rate ~5% -> ~21%, mirroring Week 1's
frequentist peeking result. Figures 24-27. `writeups/bayesian_vs_frequentist.md`
has the full writeup; README and one_pager updated.

Week 7b (Streamlit power calculator) complete: `app/power_calculator.py`
is a Streamlit UI over `src/power_analysis.py` (imported unchanged, no
logic duplicated). Three tabs — sample size given a target MDE (with
daily-traffic runtime and a live sample-size-vs-MDE curve), minimum
detectable effect for a fixed sample (direct n or daily-traffic x days,
with an MDE-vs-n curve), and an interactive peeking simulator wrapping
`simulate_peeking` (fixed-horizon vs. peeking FPR, inflation factor,
per-look calibration chart). Sidebar has shared alpha/power/baseline
inputs and a one-click Hillstrom preset. `streamlit` added to
requirements.txt. Smoke-tested headless via `streamlit.testing.v1.AppTest`
(no exceptions across all input paths); numbers reconcile with the Week 1
writeup (e.g. Hillstrom preset + 10% relative MDE -> 13,789/arm vs. the
notebook's 13,794).

Week 7c (interference / network-effects note) complete:
`src/interference.py` implements a stochastic-block-model network builder,
a spillover outcome model (`p_i` depends on the treated fraction of `i`'s
neighbours, plus an optional per-cluster shift), individual- vs.
cluster-randomization estimators, and a switchback simulator with
settling/carryover, an AR(1) period noise, and an optional washout.
`notebooks/08_interference_network_effects.ipynb` shows: individual
randomization on a connected network recovers only the direct effect
(+0.039 vs. a true global effect of 0.11, -65% bias); cluster
randomization recovers it near-unbiased (+0.108) but with ~1.3x the
sampling SD and a unit-level SE (0.018) that badly understates the honest
cluster-level SE (0.028) — the design effect; and a switchback block-length
sweep where carryover bias falls from -50% (block length 1, no washout
possible) to near zero with a one-period washout at length >= 2, while the
true sampling SD rises with block length and the naive IID SE stays flat
(understating it ~35% by length 12). Figures 28-31.
`writeups/interference_network_effects.md` is the note; it cross-references
Week 4's `simulate_shared_inventory_spillover` as the worked
marketplace-resource example. README and one_pager updated.

All optional Phase 7 stretch items are now done (Bayesian vs. frequentist,
Streamlit power calculator, interference note); the blog-style writeup was
dropped by user decision. The project is complete unless further work is
requested.
