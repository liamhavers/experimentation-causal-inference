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
2. **Criteo Uplift Modeling Dataset** — used for heterogeneous treatment effects /
   uplift modelling at realistic scale (Week 3). Large — be mindful of memory/runtime.

## Build plan (4 weeks)

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

### Optional Week 6 (further stretch)
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

Not yet started — this file is the initial scope. Update this section as weeks are completed.
