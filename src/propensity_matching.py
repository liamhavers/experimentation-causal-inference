# Propensity score matching, IPW, and regression adjustment for
# observational (non-randomized) treatment comparisons (Week 6).
#
# All three estimators rely on the same core assumption, which a real RCT
# doesn't need: **unconfoundedness** (a.k.a. ignorability) — that treatment
# assignment is as-good-as-random *conditional on the observed covariates*,
# i.e. there's no unmeasured confounder left over once you've controlled for
# what's in the data. That's an assumption, not something checkable from the
# data alone — which is exactly why this module is validated in the notebook
# against a case where the true effect is known (real randomized-experiment
# data, deliberately re-sampled to be confounded), rather than trusted
# on faith.
#
# A second assumption, **overlap/positivity**, *is* checkable: every unit
# needs a real chance of being in either arm (0 < propensity < 1). Where
# overlap is poor, matching drops unmatched units and IPW inflates variance
# — both a symptom of the same underlying problem, not a bug to route
# around.

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def estimate_propensity(treatment, covariates) -> np.ndarray:
    """Fit P(treatment=1 | covariates) via logistic regression and return
    the fitted propensity score for every unit.
    """
    covariates = np.asarray(covariates, dtype=float)
    treatment = np.asarray(treatment)

    model = LogisticRegression(max_iter=1000)
    model.fit(covariates, treatment)
    return model.predict_proba(covariates)[:, 1]


def standardized_mean_diff(covariate, treatment) -> float:
    """Standardized mean difference (SMD) between arms on one covariate —
    the standard covariate-balance diagnostic, usable both before and after
    matching/weighting (unlike checking the ATE against ground truth, which
    only works when the truth happens to be known). Rule of thumb: |SMD| <
    0.1 is generally considered good balance.
    """
    covariate = np.asarray(covariate, dtype=float)
    treatment = np.asarray(treatment)

    treated, control = covariate[treatment == 1], covariate[treatment == 0]
    pooled_std = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((treated.mean() - control.mean()) / pooled_std)


def nearest_neighbor_match(treatment, propensity, caliper: float | None = 0.05) -> pd.DataFrame:
    """1:1 nearest-neighbor matching on the propensity score, with
    replacement (each control can match to more than one treated unit —
    trades a small amount of efficiency loss for lower bias than matching
    without replacement, the standard choice when the pools are unequal in
    size). `caliper` drops any treated unit whose nearest match is farther
    than that propensity-score distance away, rather than force a bad match.

    Returns a DataFrame with one row per matched treated unit: its index,
    its matched control's index, and the propensity-score distance between
    them.
    """
    treatment = np.asarray(treatment)
    propensity = np.asarray(propensity, dtype=float)

    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    nn = NearestNeighbors(n_neighbors=1).fit(propensity[control_idx].reshape(-1, 1))
    distances, neighbors = nn.kneighbors(propensity[treated_idx].reshape(-1, 1))
    distances = distances.ravel()
    matched_control_idx = control_idx[neighbors.ravel()]

    matches = pd.DataFrame({
        "treated_idx": treated_idx,
        "matched_control_idx": matched_control_idx,
        "distance": distances,
    })

    if caliper is not None:
        matches = matches[matches["distance"] <= caliper].reset_index(drop=True)

    return matches


def matched_ate(outcome, matches: pd.DataFrame) -> dict:
    """ATE on a matched sample: mean of (treated unit's outcome - its
    matched control's outcome), with the paired-difference standard error.
    """
    outcome = np.asarray(outcome, dtype=float)
    diffs = outcome[matches["treated_idx"].values] - outcome[matches["matched_control_idx"].values]

    return {
        "ate": float(diffs.mean()),
        "se": float(diffs.std(ddof=1) / np.sqrt(len(diffs))),
        "n_matched": len(diffs),
    }


def ipw_ate(outcome, treatment, propensity, trim: float | None = 0.01) -> dict:
    """Inverse-propensity-weighted (Horvitz-Thompson) ATE estimator:

        ATE = mean(T*Y/e(X)) - mean((1-T)*Y/(1-e(X)))

    `trim` clips propensity scores to [trim, 1-trim] before weighting —
    without it, a unit with a propensity near 0 or 1 gets an enormous
    weight from a single, often noisy, probability estimate, which can
    dominate the whole estimate. Trimming is a bias/variance trade-off
    (it changes which population the estimate describes, at the boundary),
    worth stating rather than applying silently.
    """
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    propensity = np.asarray(propensity, dtype=float)

    if trim is not None:
        propensity = np.clip(propensity, trim, 1 - trim)

    weighted_treated = treatment * outcome / propensity
    weighted_control = (1 - treatment) * outcome / (1 - propensity)

    return {
        "ate": float(weighted_treated.mean() - weighted_control.mean()),
        "mean_weight_treated": float((treatment / propensity)[treatment == 1].mean()),
        "mean_weight_control": float(((1 - treatment) / (1 - propensity))[treatment == 0].mean()),
    }


def regression_adjustment_ate(outcome, treatment, covariates) -> dict:
    """Simple covariate-adjusted OLS: outcome ~ treatment + covariates.
    The treatment coefficient is the regression-adjustment ATE estimate —
    conceptually the same covariate-adjustment idea as CUPED (Week 2) and
    ANCOVA, applied here to correct for confounding rather than to reduce
    variance in an already-randomized comparison.
    """
    import statsmodels.api as sm

    treatment = np.asarray(treatment, dtype=float).reshape(-1, 1)
    covariates = np.asarray(covariates, dtype=float)
    outcome = np.asarray(outcome, dtype=float)

    X = sm.add_constant(np.hstack([treatment, covariates]))
    model = sm.OLS(outcome, X).fit()

    return {
        "ate": float(model.params[1]),
        "se": float(model.bse[1]),
        "p_value": float(model.pvalues[1]),
    }
