# CUPED & Variance Reduction

**Notebook:** `notebooks/02_cuped.ipynb`
**Code:** `src/cuped.py`
**Dataset:** Hillstrom MineThatData (same as Week 1), comparing the **No
E-Mail** control arm against the **Mens E-Mail** treatment arm.

## Why this matters for a product DS role

Week 1's calculator makes the traffic/MDE trade-off explicit: want a
smaller detectable effect, pay for it with more users or more days. CUPED
is the other lever — instead of buying precision with more traffic, it buys
precision by removing variance the experiment metric already had *before*
treatment was assigned. Both techniques answer the same question ("how do I
get a trustworthy read for less cost?") from different directions, and a
product DS is expected to know when to reach for which.

## 1. How CUPED works

CUPED (Deng et al., 2013) adjusts a metric `Y` using a covariate `X`
measured **before** treatment assignment:

```
Y_cuped = Y - theta * (X - E[X]),    theta = Cov(X, Y) / Var(X)
```

Because `X` is pre-treatment, it's independent of which arm a user landed
in — subtracting a linear function of it cannot bias the treatment-effect
estimate, only remove shared noise both arms already had. At the
variance-minimizing `theta` above:

```
Var(Y_cuped) = Var(Y) * (1 - corr(X, Y)^2)
```

The variance reduction is *exactly* the squared correlation between the
covariate and the outcome. This is the single most useful practical fact
about CUPED: **a weak covariate buys almost nothing**, and it's cheap to
check that correlation empirically before committing to using it.

`src/cuped.py` implements this by hand — `cuped_adjust` for a single
covariate, and `cuped_adjust_multi` for several covariates at once via OLS
(equivalent to regression/ANCOVA adjustment, where the variance reduction
becomes the regression's R²). Both were cross-checked against the
theoretical `1 - corr²` / `1 - R²` formulas on synthetic data with known
correlation and matched to five decimal places.

## 2. Check covariate strength before applying CUPED

The classic CUPED setup uses the *same metric from a previous period* as
the covariate (e.g. last week's conversion rate predicting this week's).
Hillstrom is a single-shot campaign dataset with no such pre-period metric,
so the available pre-treatment covariates are `recency`, `history`,
`mens`/`womens` (past purchase category), and `newbie`.

Regressing each outcome on all five covariates jointly gives an upper bound
on what CUPED can achieve here:

| Outcome | R² (all 5 covariates) |
|---|---:|
| `visit` | 2.43% |
| `conversion` | 0.22% |
| `spend` | 0.12% |

`visit` has the only usable (if modest) signal. `conversion` and `spend`
are essentially unpredictable from anything measured pre-experiment in this
dataset — `spend` in particular is over 98% zeros, so a linear covariate has
almost nothing to grab onto. **This is a genuine, honest finding, not a
modelling failure**: it means CUPED would do close to nothing for
`conversion` or `spend` here, and applying it there would be effort for no
benefit. Checking this before running the adjustment — rather than applying
CUPED reflexively to every metric — is itself the point of this section.

## 3. Single- vs. multi-covariate CUPED on `visit`

| | Variance reduction | 95% CI width | Treatment effect |
|---|---:|---:|---:|
| Raw | — | 0.01327 | 0.0766 |
| CUPED (`recency` only) | 0.64% | 0.01323 | 0.0768 |
| CUPED (5 covariates) | 2.43% | 0.01311 | 0.0765 |

`theta` and `beta` were estimated by pooling both arms — valid because the
covariates are pre-treatment and therefore independent of assignment by
randomization, so pooling doesn't bias the effect estimate, it just uses
the data efficiently. (With a genuine separate pre-period sample available,
estimating `theta` there instead would be the more conservative choice, to
rule out any risk of overfitting noise — not a concern here at n > 20,000.)

The point estimate of the treatment effect barely moves between rows —
CUPED doesn't change *what* you're estimating, only how precisely. Combining
all five weak covariates via `cuped_adjust_multi` captures roughly 4x the
variance reduction of the single best covariate alone (2.43% vs. 0.64%),
the standard argument for combining several weak signals over hunting for
one strong one.

## 4. What CUPED looks like with a strong covariate

To show the mechanism at a realistic "good" correlation — the kind seen
when a genuine same-metric pre-period value is available (session counts,
spend, or engagement scores often correlate with their own next-period
value at r ≈ 0.5–0.7) — the notebook includes a clearly-labeled **synthetic**
example: a simulated covariate and outcome constructed with `corr(X, Y) =
0.6`.

| | Variance reduction | 95% CI width |
|---|---:|---:|
| Raw (synthetic) | — | 0.00306 |
| CUPED (synthetic, corr=0.6) | 36.2% | 0.00244 |

A 36% variance reduction is visibly obvious in the confidence interval —
unlike the real-data case above, where the effect is genuine but small. The
contrast between the two sections is deliberate: it shows both that the
method works exactly as the theory predicts, and that its real-world payoff
is entirely conditional on having a covariate worth using.

## 5. Connecting back to the Week 1 power calculator

A variance reduction of `r` is interchangeable with either of two things a
team might want, using the same `required_sample_size` calculator from
Week 1:

- **Same traffic, tighter MDE** — standard error shrinks by `sqrt(1 - r)`,
  so the smallest reliably-detectable effect shrinks by the same factor at
  a fixed sample size.
- **Same MDE, less traffic** — required sample size to hit a given MDE
  shrinks by a factor of `(1 - r)`, directly cutting test runtime.

| Scenario | Variance reduction | n/arm (no CUPED) | n/arm (with CUPED) | Traffic saved |
|---|---:|---:|---:|---:|
| Real (5-covariate CUPED) | 2.4% | 4,029 | 3,931 | 2.4% |
| Synthetic (strong covariate) | 36.2% | 4,029 | 2,571 | 36.2% |

The weak real covariates here buy a modest but genuinely free ~2% cut in
required traffic — worth taking since CUPED costs nothing beyond a linear
adjustment once the covariate is already in the warehouse. The synthetic
case shows the payoff when a strong, well-chosen covariate exists: over a
third less traffic for the same statistical guarantees, which is why CUPED
is standard practice at companies running high-traffic experiments, where
even a 10-20% runtime cut is worth real calendar time.

## What's next

Week 3 switches datasets (Criteo Uplift) and shifts from "how much traffic
does an average effect need" to "who actually responds" — heterogeneous
treatment effects and a cost-based targeting policy built on top of them.
