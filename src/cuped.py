# CUPED variance reduction using pre-experiment covariates (Week 2).
#
# CUPED (Controlled-experiment Using Pre-Experiment Data, Deng et al. 2013)
# reduces the variance of an experiment metric by subtracting off the part of
# it that's predictable from a covariate measured *before* treatment
# assignment. Because the covariate is pre-treatment, subtracting a linear
# function of it cannot bias the treatment effect estimate — it can only
# remove noise that both arms share for reasons unrelated to treatment.
#
#   Y_cuped = Y - theta * (X - E[X])
#
# theta = Cov(X, Y) / Var(X) is the OLS slope of Y on X. This choice of theta
# minimizes Var(Y_cuped), and at that minimum:
#
#   Var(Y_cuped) = Var(Y) * (1 - corr(X, Y)^2)
#
# so the variance reduction is exactly the squared correlation between the
# covariate and the outcome — a covariate barely correlated with the outcome
# buys almost nothing, and this is worth checking empirically before
# reaching for CUPED, not assuming it always helps.
#
# `cuped_adjust_multi` generalizes this to several covariates at once via
# multivariate OLS (equivalent to regression/ANCOVA adjustment): the
# variance reduction becomes the R^2 of Y regressed on the covariates.

from __future__ import annotations

import numpy as np
from scipy import stats


def compute_theta(x, y) -> float:
    """OLS slope of y on x: Cov(x, y) / Var(x)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))


def cuped_adjust(y, x, theta: float | None = None):
    """Single-covariate CUPED adjustment.

    y: outcome metric (e.g. visit, conversion, spend).
    x: pre-experiment covariate, measured before treatment assignment.
    theta: adjustment coefficient. If None, estimated from the data as
        Cov(x, y) / Var(x). Pass a pre-estimated theta (e.g. from a prior
        period, or from the control arm only) to avoid re-using the same
        data to both fit and evaluate the adjustment.

    Returns (y_adjusted, theta). E[y_adjusted] == E[y] in expectation,
    since x is pre-treatment and E[X - E[X]] = 0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if theta is None:
        theta = compute_theta(x, y)
    y_adjusted = y - theta * (x - x.mean())
    return y_adjusted, theta


def cuped_adjust_multi(y, X, beta=None):
    """Multi-covariate CUPED (generalized CUPED / regression adjustment).

    y: outcome metric.
    X: (n, k) array of pre-experiment covariates.
    beta: adjustment coefficients. If None, estimated via OLS: regress
        (y - mean(y)) on the centered covariates.

    Returns (y_adjusted, beta).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_centered = X - X.mean(axis=0)

    if beta is None:
        beta, *_ = np.linalg.lstsq(X_centered, y - y.mean(), rcond=None)
    else:
        beta = np.asarray(beta, dtype=float)

    y_adjusted = y - X_centered @ beta
    return y_adjusted, beta


def variance_reduction(y, y_adjusted) -> dict:
    """Empirical variance reduction achieved by a CUPED adjustment."""
    var_before = float(np.var(y, ddof=1))
    var_after = float(np.var(y_adjusted, ddof=1))
    return {
        "var_before": var_before,
        "var_after": var_after,
        "relative_reduction": 1 - var_after / var_before,
    }


def diff_in_means_ci(y_control, y_treatment, alpha: float = 0.05) -> dict:
    """Two-sample difference-in-means CI and z-test (Welch-style, unpooled variance).

    Works for both binary metrics (mean of 0/1 = proportion) and continuous
    metrics (e.g. spend). Used to compare CI width and significance before
    vs. after a CUPED adjustment on the *same* underlying data.
    """
    y_control = np.asarray(y_control, dtype=float)
    y_treatment = np.asarray(y_treatment, dtype=float)

    n1, n2 = len(y_control), len(y_treatment)
    m1, m2 = y_control.mean(), y_treatment.mean()
    v1, v2 = y_control.var(ddof=1), y_treatment.var(ddof=1)

    se = float(np.sqrt(v1 / n1 + v2 / n2))
    diff = float(m2 - m1)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    z_stat = diff / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        "diff": diff,
        "se": se,
        "ci_low": diff - z_crit * se,
        "ci_high": diff + z_crit * se,
        "ci_width": 2 * z_crit * se,
        "z": float(z_stat),
        "p_value": float(p_value),
    }
