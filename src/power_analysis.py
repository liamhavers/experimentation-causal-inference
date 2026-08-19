# Power / MDE calculator for fixed-horizon A/B tests (Week 1).
#
# Two-sample z-test for a difference in proportions, using the pooled-variance
# formulation. Covers three directions of the same underlying equation:
#   1. sample size given a target MDE
#   2. minimum detectable effect given a fixed sample size
#   3. calendar runtime given daily traffic and a traffic split
#
# Also includes a repeated-significance-testing (peeking) simulator used to
# show why checking a fixed-horizon test early and stopping at the first
# significant p-value inflates the false positive rate above the nominal alpha.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SampleSizeResult:
    baseline_rate: float
    mde_absolute: float
    alpha: float
    power: float
    n_per_variant: int

    @property
    def n_total(self) -> int:
        return 2 * self.n_per_variant

    def runtime_days(self, daily_traffic: int, traffic_split: float = 0.5) -> float:
        """Calendar days to reach n_per_variant in each arm.

        daily_traffic: total users/sessions arriving per day across both arms.
        traffic_split: share of daily_traffic allocated to the smaller arm
            (0.5 for an even A/B split).
        """
        daily_per_variant = daily_traffic * min(traffic_split, 1 - traffic_split)
        return self.n_per_variant / daily_per_variant


def required_sample_size(
    baseline_rate: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> SampleSizeResult:
    """Per-variant sample size for a two-sided two-proportion z-test.

    Uses the standard closed-form approximation (pooled variance under the
    null, unpooled under the alternative):

        n = (z_(1-alpha/2) * sqrt(2*p_bar*(1-p_bar)) + z_(1-beta) * sqrt(p1*(1-p1) + p2*(1-p2)))^2
            / mde^2

    baseline_rate: control conversion rate (p1).
    mde_absolute: smallest true difference in conversion rate you want to
        reliably detect, in absolute terms (e.g. 0.01 for a 1 percentage
        point lift), not relative.
    alpha: two-sided significance level.
    power: desired statistical power (1 - beta).
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be in (0, 1)")
    if mde_absolute <= 0:
        raise ValueError("mde_absolute must be positive")
    if not 0 < baseline_rate + mde_absolute < 1:
        raise ValueError("baseline_rate + mde_absolute must be in (0, 1)")

    p1 = baseline_rate
    p2 = baseline_rate + mde_absolute
    p_bar = (p1 + p2) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)

    numerator = (
        z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_power * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    )
    n = (numerator ** 2) / (mde_absolute ** 2)

    return SampleSizeResult(
        baseline_rate=baseline_rate,
        mde_absolute=mde_absolute,
        alpha=alpha,
        power=power,
        n_per_variant=int(np.ceil(n)),
    )


def minimum_detectable_effect(
    baseline_rate: float,
    n_per_variant: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest absolute effect detectable with a fixed per-variant sample size.

    Inverts required_sample_size by solving the same equation for mde,
    approximating p2 with the baseline rate (mde is small relative to the
    rate for most product-metric use cases, so p1*(1-p1) ~= p2*(1-p2)).
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be in (0, 1)")
    if n_per_variant <= 0:
        raise ValueError("n_per_variant must be positive")

    p = baseline_rate
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)

    # sqrt(2*p*(1-p)) for the pooled term and sqrt(2*p*(1-p)) for the
    # alternative term coincide when p1 ~= p2 ~= p, so they combine linearly.
    mde = (z_alpha + z_power) * np.sqrt(2 * p * (1 - p)) / np.sqrt(n_per_variant)
    return float(mde)


def two_proportion_z_test(
    n1: int, conversions1: int, n2: int, conversions2: int
) -> tuple[float, float]:
    """Two-sided two-proportion z-test. Returns (z_statistic, p_value)."""
    p1 = conversions1 / n1
    p2 = conversions2 / n2
    p_pool = (conversions1 + conversions2) / (n1 + n2)

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0

    z = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p_value)


def simulate_peeking(
    baseline_rate: float,
    n_per_variant_max: int,
    n_checks: int,
    alpha: float = 0.05,
    n_simulations: int = 2000,
    true_effect: float = 0.0,
    random_state: int | None = None,
) -> dict:
    """Simulate repeated significance testing ("peeking") under a fixed alpha.

    For each simulated experiment, generates data for two arms with no
    true effect by default (true_effect=0.0, i.e. the null is true), then
    checks a two-proportion z-test at n_checks evenly spaced points between
    the start and n_per_variant_max, stopping at the first look where
    p < alpha ("stop on significance").

    Returns a dict with:
      - fixed_horizon_false_positive_rate: rejection rate using only the
        final look (the honest fixed-horizon test)
      - peeking_false_positive_rate: rejection rate under "stop at first
        significant look"
      - check_points: the per-variant sample sizes at which looks occurred
      - per_look_reject_rate: marginal rejection rate at each individual look

    Set true_effect > 0 to see the same simulator used for power instead of
    false-positive rate (fraction of experiments that ever/eventually reject).
    """
    rng = np.random.default_rng(random_state)
    check_points = np.linspace(
        n_per_variant_max / n_checks, n_per_variant_max, n_checks
    ).astype(int)

    p_control = baseline_rate
    p_treatment = baseline_rate + true_effect

    control = rng.binomial(1, p_control, size=(n_simulations, n_per_variant_max))
    treatment = rng.binomial(1, p_treatment, size=(n_simulations, n_per_variant_max))

    control_cumsum = np.cumsum(control, axis=1)
    treatment_cumsum = np.cumsum(treatment, axis=1)

    stopped = np.zeros(n_simulations, dtype=bool)
    peeking_rejections = np.zeros(n_simulations, dtype=bool)
    per_look_reject_rate = []

    for n in check_points:
        c1 = control_cumsum[:, n - 1]
        c2 = treatment_cumsum[:, n - 1]
        p1 = c1 / n
        p2 = c2 / n
        p_pool = (c1 + c2) / (2 * n)
        se = np.sqrt(p_pool * (1 - p_pool) * (2 / n))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, (p2 - p1) / se, 0.0)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z)))

        significant_now = (p_values < alpha) & ~stopped
        per_look_reject_rate.append(float(np.mean(p_values < alpha)))

        peeking_rejections |= significant_now
        stopped |= significant_now

    # Fixed-horizon: only the last look counts, no early stopping.
    final_n = check_points[-1]
    c1_final = control_cumsum[:, final_n - 1]
    c2_final = treatment_cumsum[:, final_n - 1]
    p1_final = c1_final / final_n
    p2_final = c2_final / final_n
    p_pool_final = (c1_final + c2_final) / (2 * final_n)
    se_final = np.sqrt(p_pool_final * (1 - p_pool_final) * (2 / final_n))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_final = np.where(se_final > 0, (p2_final - p1_final) / se_final, 0.0)
    p_values_final = 2 * (1 - stats.norm.cdf(np.abs(z_final)))
    fixed_horizon_rejections = p_values_final < alpha

    return {
        "fixed_horizon_false_positive_rate": float(np.mean(fixed_horizon_rejections)),
        "peeking_false_positive_rate": float(np.mean(peeking_rejections)),
        "check_points": check_points.tolist(),
        "per_look_reject_rate": per_look_reject_rate,
        "alpha": alpha,
        "n_simulations": n_simulations,
    }
