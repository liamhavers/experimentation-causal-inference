# Instrumental variables for imperfect compliance (Week 6).
#
# Built for the Criteo dataset's real `treatment` (randomized ad-eligibility)
# vs. `exposure` (whether the ad was actually rendered) columns — a one-sided
# noncompliance design: exposure is only ever 1 when treatment=1, so nobody
# assigned to control could possibly be exposed. In that setting:
#
#   ITT   = E[Y | Z=1] - E[Y | Z=0]                     (effect of assignment)
#   LATE  = ITT / (E[D | Z=1] - E[D | Z=0])              (Wald estimator)
#
# where Z is the (randomized) instrument, D is the endogenous "actually
# treated" indicator, and Y is the outcome. LATE identifies the average
# effect of D on Y among *compliers* — units that would be exposed if
# assigned treatment (nobody can be an "always-taker" here, since exposure
# is impossible under control) — under three standard assumptions:
#   1. Relevance: Z actually predicts D (checked directly with a first-stage
#      F-statistic below — the "weak instrument" diagnostic).
#   2. Exclusion restriction: Z affects Y only through D, not through any
#      other channel (plausible here: users aren't aware of their assignment,
#      so assignment itself shouldn't change behaviour independent of
#      whether the ad was actually shown).
#   3. Monotonicity: no "defiers" — nobody who'd be exposed under control but
#      not under treatment. Automatically satisfied here, since exposure is
#      structurally impossible under control.

from __future__ import annotations

import numpy as np
import statsmodels.api as sm


def wald_estimator(outcome, instrument, endogenous) -> dict:
    """LATE via the Wald ratio: ITT divided by the first-stage compliance
    effect. Exactly equal to the just-identified 2SLS estimate with a single
    binary instrument and a single endogenous regressor — see
    `two_stage_least_squares` for the regression-based version that
    generalizes to covariates and multiple instruments.
    """
    outcome = np.asarray(outcome, dtype=float)
    instrument = np.asarray(instrument)
    endogenous = np.asarray(endogenous, dtype=float)

    itt = outcome[instrument == 1].mean() - outcome[instrument == 0].mean()
    first_stage = endogenous[instrument == 1].mean() - endogenous[instrument == 0].mean()

    return {
        "itt": float(itt),
        "first_stage_compliance": float(first_stage),
        "late": float(itt / first_stage),
    }


def naive_as_treated_estimate(outcome, endogenous) -> dict:
    """The comparison someone reaches for if they ignore the instrument
    entirely: outcome among the actually-exposed vs. actually-unexposed,
    with no correction for the fact that exposure wasn't randomly assigned.
    Included specifically to contrast against the IV-corrected LATE.
    """
    outcome = np.asarray(outcome, dtype=float)
    endogenous = np.asarray(endogenous)

    exposed_rate = outcome[endogenous == 1].mean()
    unexposed_rate = outcome[endogenous == 0].mean()

    return {
        "exposed_rate": float(exposed_rate),
        "unexposed_rate": float(unexposed_rate),
        "naive_diff": float(exposed_rate - unexposed_rate),
    }


def first_stage_strength(instrument, endogenous) -> dict:
    """F-statistic for the instrument's coefficient in the first-stage
    regression (endogenous ~ instrument) — the standard weak-instrument
    diagnostic. A rule of thumb from the IV literature: F < 10 signals a
    weak instrument, where the Wald/2SLS estimate becomes unreliable (large
    finite-sample bias, misleading confidence intervals) even if the
    instrument is technically statistically significant.
    """
    instrument = np.asarray(instrument, dtype=float)
    endogenous = np.asarray(endogenous, dtype=float)

    X = sm.add_constant(instrument)
    model = sm.OLS(endogenous, X).fit()

    return {
        "first_stage_coefficient": float(model.params[1]),
        "first_stage_f_stat": float(model.fvalue),
        "weak_instrument": bool(model.fvalue < 10),
    }


def two_stage_least_squares_manual(outcome, instrument, endogenous) -> dict:
    """Manual two-stage least squares: regress the endogenous variable on
    the instrument (stage 1), then regress the outcome on the *fitted*
    values from stage 1 (stage 2). The stage-2 coefficient is mathematically
    identical to the Wald ratio in the single-instrument, single-endogenous-
    regressor case — this function exists to make that equivalence visible,
    not as the estimator of record. Its standard errors are not the correct
    2SLS standard errors (they don't account for the first-stage estimation
    uncertainty); use `linearmodels.iv.IV2SLS` for a properly-computed
    standard error and confidence interval.
    """
    instrument = np.asarray(instrument, dtype=float)
    endogenous = np.asarray(endogenous, dtype=float)
    outcome = np.asarray(outcome, dtype=float)

    stage1 = sm.OLS(endogenous, sm.add_constant(instrument)).fit()
    endogenous_hat = stage1.predict(sm.add_constant(instrument))

    stage2 = sm.OLS(outcome, sm.add_constant(endogenous_hat)).fit()

    return {
        "first_stage_coefficient": float(stage1.params[1]),
        "second_stage_coefficient": float(stage2.params[1]),
    }
