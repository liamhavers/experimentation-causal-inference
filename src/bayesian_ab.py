# Bayesian vs. frequentist analysis of a binary-metric A/B test (Week 7).
#
# The frequentist read of a two-arm conversion test is the Week 1 machinery:
# a two-proportion z-test p-value and a Wald confidence interval on the
# difference in rates. This module implements the Bayesian counterpart from
# scratch, using the Beta-Bernoulli conjugate model:
#
#   prior      theta ~ Beta(a, b)
#   likelihood s successes in n trials, s | theta ~ Binomial(n, theta)
#   posterior  theta | data ~ Beta(a + s, b + n - s)
#
# Conjugacy means the posterior is available in closed form with no MCMC.
# From the two arms' posteriors we derive the quantities a Bayesian actually
# reports to a decision-maker:
#
#   - a credible interval on each rate and on the lift (a *direct*
#     probability statement about the parameter, unlike a confidence
#     interval)
#   - P(treatment > control): the posterior probability the treatment arm is
#     genuinely better
#   - expected loss of each ship decision: the decision-theoretic quantity
#     that turns the posterior into an action, and the basis for a
#     principled Bayesian stopping rule
#
# P(treatment > control) and the lift credible interval have exact
# closed forms for the Beta-Bernoulli case, but this module estimates them
# by Monte Carlo from the posteriors: it generalizes unchanged to metrics
# without a conjugate form, and the sampling error is negligible at the
# sample sizes used here (and reported, not hidden).

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class BetaPosterior:
    """A Beta(alpha, beta) posterior over a single arm's conversion rate."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def var(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Equal-tailed credible interval from the posterior quantiles.

        Equal-tailed (not highest-density) for simplicity and because the
        posteriors here are close to symmetric at these sample sizes; the
        two intervals coincide in that regime.
        """
        tail = (1 - level) / 2
        return (
            float(stats.beta.ppf(tail, self.alpha, self.beta)),
            float(stats.beta.ppf(1 - tail, self.alpha, self.beta)),
        )

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.beta(self.alpha, self.beta, size=size)


def update_beta(
    prior_alpha: float, prior_beta: float, n: int, successes: int
) -> BetaPosterior:
    """Beta-Bernoulli posterior update: Beta(a + s, b + n - s)."""
    if successes < 0 or successes > n:
        raise ValueError("successes must be in [0, n]")
    return BetaPosterior(prior_alpha + successes, prior_beta + (n - successes))


def prob_treatment_beats_control(
    control: BetaPosterior,
    treatment: BetaPosterior,
    n_samples: int = 200_000,
    random_state: int | None = None,
) -> dict:
    """Monte Carlo P(theta_treatment > theta_control) from the two posteriors.

    Returns the probability plus its Monte Carlo standard error, so the
    sampling noise in the estimate is visible rather than implied to be
    exact.
    """
    rng = np.random.default_rng(random_state)
    c = control.sample(n_samples, rng)
    t = treatment.sample(n_samples, rng)
    wins = t > c
    p = float(wins.mean())
    mc_se = float(np.sqrt(p * (1 - p) / n_samples))
    return {"prob": p, "mc_se": mc_se, "n_samples": n_samples}


def posterior_lift(
    control: BetaPosterior,
    treatment: BetaPosterior,
    level: float = 0.95,
    n_samples: int = 200_000,
    random_state: int | None = None,
) -> dict:
    """Posterior summary of the treatment effect (treatment - control).

    Reports both the absolute lift (difference in rates) and the relative
    lift (difference as a fraction of the control rate), each with a
    posterior mean and an equal-tailed credible interval.
    """
    rng = np.random.default_rng(random_state)
    c = control.sample(n_samples, rng)
    t = treatment.sample(n_samples, rng)

    abs_lift = t - c
    rel_lift = abs_lift / c
    tail = (1 - level) / 2

    return {
        "abs_mean": float(abs_lift.mean()),
        "abs_ci": (
            float(np.quantile(abs_lift, tail)),
            float(np.quantile(abs_lift, 1 - tail)),
        ),
        "rel_mean": float(rel_lift.mean()),
        "rel_ci": (
            float(np.quantile(rel_lift, tail)),
            float(np.quantile(rel_lift, 1 - tail)),
        ),
        "prob_positive": float((abs_lift > 0).mean()),
        "samples_abs": abs_lift,
    }


def expected_loss(
    control: BetaPosterior,
    treatment: BetaPosterior,
    n_samples: int = 200_000,
    random_state: int | None = None,
) -> dict:
    """Decision-theoretic expected loss of each ship decision.

    If the true rates were known, shipping the worse arm would cost you the
    gap between them. They are not known, so the expected loss integrates
    that regret over the posterior:

        loss(ship treatment) = E[ max(theta_control - theta_treatment, 0) ]
        loss(ship control)   = E[ max(theta_treatment - theta_control, 0) ]

    i.e. the expected amount of conversion rate you forgo by shipping that
    arm, averaging only over the region of the posterior where that choice
    is the wrong one. A standard Bayesian stopping rule is: stop and ship
    when the smaller of these two falls below a pre-set "threshold of
    caring" epsilon (the largest rate difference you're willing to
    accidentally give up).
    """
    rng = np.random.default_rng(random_state)
    c = control.sample(n_samples, rng)
    t = treatment.sample(n_samples, rng)

    loss_ship_treatment = float(np.maximum(c - t, 0.0).mean())
    loss_ship_control = float(np.maximum(t - c, 0.0).mean())
    return {
        "loss_ship_treatment": loss_ship_treatment,
        "loss_ship_control": loss_ship_control,
    }


def frequentist_two_proportion(
    n_control: int,
    conversions_control: int,
    n_treatment: int,
    conversions_treatment: int,
    alpha: float = 0.05,
) -> dict:
    """Frequentist counterpart: two-proportion z-test + Wald CI on the diff.

    Pooled-variance z-test for the p-value (the null-hypothesis test), and
    an unpooled (Wald) standard error for the confidence interval on the
    difference in rates — the same convention used elsewhere in this repo.
    """
    p_c = conversions_control / n_control
    p_t = conversions_treatment / n_treatment
    diff = p_t - p_c

    p_pool = (conversions_control + conversions_treatment) / (n_control + n_treatment)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se_unpooled = np.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return {
        "p_control": float(p_c),
        "p_treatment": float(p_t),
        "diff": float(diff),
        "z": float(z),
        "p_value": float(p_value),
        "ci_low": float(diff - z_crit * se_unpooled),
        "ci_high": float(diff + z_crit * se_unpooled),
    }


def simulate_bayesian_optional_stopping(
    baseline_rate: float,
    n_per_arm_max: int,
    n_checks: int,
    true_effect: float = 0.0,
    prob_threshold: float = 0.95,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_simulations: int = 2000,
    n_posterior_samples: int = 4000,
    random_state: int | None = None,
) -> dict:
    """Peek at a Bayesian A/B test repeatedly and stop when P(T>C) crosses a
    threshold — the Bayesian analogue of the Week 1 frequentist peeking demo.

    For each simulated experiment (no true effect by default), data
    accumulates in both arms and P(treatment > control) is recomputed at
    ``n_checks`` evenly spaced interim looks. "Stop at first look where
    P(T>C) > prob_threshold OR P(T>C) < 1 - prob_threshold" is applied, and
    the false-positive rate of that rule is compared with a single
    fixed-horizon look at the end.

    The point is symmetry with Week 1: a Bayesian posterior probability is
    not a frequentist error rate, and using ``P(T>C) > 0.95`` as a
    ship-early trigger does inflate the rate of wrongly shipping a null
    effect above what a single look would give — the guarantee it offers is
    a different one.
    """
    rng = np.random.default_rng(random_state)
    check_points = np.linspace(
        n_per_arm_max / n_checks, n_per_arm_max, n_checks
    ).astype(int)

    p_c = baseline_rate
    p_t = baseline_rate + true_effect

    control = rng.binomial(1, p_c, size=(n_simulations, n_per_arm_max))
    treatment = rng.binomial(1, p_t, size=(n_simulations, n_per_arm_max))
    control_cumsum = np.cumsum(control, axis=1)
    treatment_cumsum = np.cumsum(treatment, axis=1)

    stopped = np.zeros(n_simulations, dtype=bool)
    peeking_ship_decision = np.zeros(n_simulations, dtype=bool)  # shipped non-control
    peeking_any_stop = np.zeros(n_simulations, dtype=bool)

    for n in check_points:
        sc = control_cumsum[:, n - 1]
        st = treatment_cumsum[:, n - 1]

        # Posterior draws for every simulation at once:
        # (n_simulations, n_posterior_samples)
        c_draw = rng.beta(
            (prior_alpha + sc)[:, None],
            (prior_beta + n - sc)[:, None],
            size=(n_simulations, n_posterior_samples),
        )
        t_draw = rng.beta(
            (prior_alpha + st)[:, None],
            (prior_beta + n - st)[:, None],
            size=(n_simulations, n_posterior_samples),
        )
        prob_t_better = (t_draw > c_draw).mean(axis=1)

        trigger = (
            (prob_t_better > prob_threshold) | (prob_t_better < 1 - prob_threshold)
        ) & ~stopped
        peeking_any_stop |= trigger
        peeking_ship_decision |= trigger & (prob_t_better > prob_threshold)
        stopped |= trigger

    # Fixed-horizon: one look at the final sample size.
    n = check_points[-1]
    sc = control_cumsum[:, n - 1]
    st = treatment_cumsum[:, n - 1]
    c_draw = rng.beta(
        (prior_alpha + sc)[:, None],
        (prior_beta + n - sc)[:, None],
        size=(n_simulations, n_posterior_samples),
    )
    t_draw = rng.beta(
        (prior_alpha + st)[:, None],
        (prior_beta + n - st)[:, None],
        size=(n_simulations, n_posterior_samples),
    )
    prob_t_better_final = (t_draw > c_draw).mean(axis=1)
    fixed_ship = prob_t_better_final > prob_threshold

    return {
        "fixed_horizon_ship_rate": float(fixed_ship.mean()),
        "peeking_ship_rate": float(peeking_ship_decision.mean()),
        "peeking_any_stop_rate": float(peeking_any_stop.mean()),
        "check_points": check_points.tolist(),
        "prob_threshold": prob_threshold,
        "true_effect": true_effect,
        "n_simulations": n_simulations,
    }
