# Experimentation & Causal Inference Toolkit

A hands-on toolkit covering the core statistical toolkit of a product data scientist:
power analysis and experiment design, variance reduction (CUPED), heterogeneous
treatment effect / uplift modelling, and metrics framework design (north star,
guardrails, common failure modes).

## Why this project

Built to demonstrate experiment design and causal inference judgment for product
data science roles — skills that sit alongside, but are distinct from, general
applied ML work.

## Structure

- `notebooks/` — one notebook per technique, written to be walked through line by line
- `src/` — reusable implementations backing the notebooks
- `writeups/` — plain-English explanations of *why* each technique works, not just *what* it does
- `data/` — raw and processed datasets (Hillstrom MineThatData, Criteo Uplift)

## Roadmap

| Week | Topic | Status |
|------|-------|--------|
| 1 | Power analysis & experiment design | Not started |
| 2 | CUPED & variance reduction | Not started |
| 3 | Heterogeneous treatment effects / uplift modelling | Not started |
| 4 | Metrics framework, guardrails & packaging | Not started |
| 5 | Bandits & adaptive allocation | Not started |

See [CLAUDE.md](CLAUDE.md) for the full build plan.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
