# Heterogeneous treatment effect / uplift modelling (Week 3).
#
# Two meta-learners built on top of any scikit-learn-compatible base
# estimator:
#
#   T-learner: fit two outcome models, one per arm, and take the difference
#       of their predictions. Simple, but each model only sees its own arm's
#       data, so it can be noisy when one arm is much smaller than the other
#       (as it is here: Criteo's control arm is ~15% of the data).
#
#   X-learner (Kunzel et al. 2019): reuses the T-learner's two models to
#       impute a per-unit treatment effect for *every* unit (treated units
#       get an imputed effect using the control model's counterfactual
#       prediction, and vice versa), fits a model of that imputed effect on
#       each arm, and blends the two effect models with a propensity weight.
#       This lets the (larger) treated arm contribute to estimating effects
#       even where the control arm is thin, which is exactly the situation
#       an 85/15 treatment split creates.
#
# Also includes the standard uplift-model evaluation tools: a Qini curve
# (cumulative incremental outcomes captured vs. a random-targeting
# baseline) and a per-decile breakdown used to check whether the model's
# ranking is actually monotonic in the real, experimentally-measured
# treatment effect.

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def _default_classifier():
    return HistGradientBoostingClassifier(max_iter=150, random_state=0)


def _default_regressor():
    return HistGradientBoostingRegressor(max_iter=150, random_state=0)


# --------------------------------------------------------------------------
# S-learner
# --------------------------------------------------------------------------

def s_learner_fit(X, y, treatment, base_estimator=None):
    """Fit a single outcome model with treatment included as a feature."""
    base_estimator = base_estimator or _default_classifier()
    X = np.asarray(X)
    y = np.asarray(y)
    treatment = np.asarray(treatment).reshape(-1, 1)
    X_with_t = np.hstack([X, treatment])
    return clone(base_estimator).fit(X_with_t, y)


def s_learner_predict(model, X) -> np.ndarray:
    """Predicted uplift = model(X, T=1) - model(X, T=0).

    The single shared model can end up ignoring the treatment feature
    entirely if it's weak relative to the other features (tree ensembles in
    particular can starve a single low-importance column of splits) — this
    is the classic S-learner failure mode, and worth checking for directly
    rather than assuming the model used treatment at all.
    """
    X = np.asarray(X)
    X1 = np.hstack([X, np.ones((len(X), 1))])
    X0 = np.hstack([X, np.zeros((len(X), 1))])
    return model.predict_proba(X1)[:, 1] - model.predict_proba(X0)[:, 1]


# --------------------------------------------------------------------------
# T-learner
# --------------------------------------------------------------------------

def t_learner_fit(X, y, treatment, base_estimator=None) -> dict:
    """Fit separate outcome models on the treated and control arms."""
    base_estimator = base_estimator or _default_classifier()
    X = np.asarray(X)
    y = np.asarray(y)
    treatment = np.asarray(treatment)

    model_treated = clone(base_estimator).fit(X[treatment == 1], y[treatment == 1])
    model_control = clone(base_estimator).fit(X[treatment == 0], y[treatment == 0])
    return {"model_treated": model_treated, "model_control": model_control}


def t_learner_predict(models: dict, X) -> np.ndarray:
    """Predicted uplift = P(Y=1 | X, T=1) - P(Y=1 | X, T=0)."""
    p1 = models["model_treated"].predict_proba(X)[:, 1]
    p0 = models["model_control"].predict_proba(X)[:, 1]
    return p1 - p0


# --------------------------------------------------------------------------
# X-learner
# --------------------------------------------------------------------------

def x_learner_fit(
    X,
    y,
    treatment,
    outcome_estimator=None,
    effect_estimator=None,
) -> dict:
    """Fit an X-learner: stage-1 outcome models + stage-2 imputed-effect models.

    propensity is intentionally *not* fit here — it's supplied at predict
    time (see x_learner_predict), since in a randomized experiment it's
    typically known (or verified to be effectively constant) rather than
    something to learn from the same data.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    treatment = np.asarray(treatment)
    outcome_estimator = outcome_estimator or _default_classifier()
    effect_estimator = effect_estimator or _default_regressor()

    X_t, y_t = X[treatment == 1], y[treatment == 1]
    X_c, y_c = X[treatment == 0], y[treatment == 0]

    # Stage 1: same as the T-learner.
    mu1 = clone(outcome_estimator).fit(X_t, y_t)
    mu0 = clone(outcome_estimator).fit(X_c, y_c)

    # Stage 2: impute a per-unit effect for each arm using the *other* arm's
    # model as the counterfactual prediction, then model that imputed effect.
    d1 = y_t - mu0.predict_proba(X_t)[:, 1]  # treated units: observed - predicted counterfactual control
    d0 = mu1.predict_proba(X_c)[:, 1] - y_c  # control units: predicted counterfactual treated - observed

    tau1 = clone(effect_estimator).fit(X_t, d1)
    tau0 = clone(effect_estimator).fit(X_c, d0)

    return {"mu1": mu1, "mu0": mu0, "tau1": tau1, "tau0": tau0}


def x_learner_predict(models: dict, X, propensity=0.5) -> np.ndarray:
    """Blend the two stage-2 effect models with a propensity weight.

    tau(x) = g(x) * tau0(x) + (1 - g(x)) * tau1(x), where g(x) = P(T=1 | X).
    propensity can be a scalar (constant propensity, the common case in a
    randomized experiment with a fixed treatment share) or a per-row array.
    """
    tau1_pred = models["tau1"].predict(X)
    tau0_pred = models["tau0"].predict(X)
    g = np.asarray(propensity)
    return g * tau0_pred + (1 - g) * tau1_pred


# --------------------------------------------------------------------------
# Evaluation: Qini curve and per-decile breakdown
# --------------------------------------------------------------------------

def qini_curve(uplift_score, treatment, outcome, n_points: int = 50):
    """Cumulative incremental outcomes captured by targeting the top-k
    fraction of units, ranked by descending predicted uplift.

    At each fraction phi of the population (top phi*n by predicted uplift):

        qini(phi) = Y1(phi) - Y0(phi) * n1(phi) / n0(phi)

    where Y1/Y0 are the summed outcome among treated/control units in that
    slice and n1/n0 are their counts. The n1/n0 re-weighting corrects for
    unequal arm sizes within the slice (needed here: ~85% of units are
    treated overall, so a naive Y1 - Y0 would be biased even under random
    targeting).

    Returns (fractions, qini_values, qini_coefficient) where
    qini_coefficient is the area between the model's curve and the random-
    targeting diagonal (curve from (0, 0) to (1, qini(1))) — the expected
    number of extra positive outcomes captured, integrated across all
    targeting fractions, versus targeting the same fractions at random.
    """
    uplift_score = np.asarray(uplift_score)
    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome)

    order = np.argsort(-uplift_score)
    treatment = treatment[order]
    outcome = outcome[order]
    n = len(outcome)

    cum_n1 = np.cumsum(treatment)
    cum_n0 = np.cumsum(1 - treatment)
    cum_y1 = np.cumsum(outcome * treatment)
    cum_y0 = np.cumsum(outcome * (1 - treatment))

    fractions = np.linspace(1.0 / n_points, 1.0, n_points)
    idx = np.clip((fractions * n).astype(int) - 1, 0, n - 1)

    n1, n0 = cum_n1[idx], cum_n0[idx]
    y1, y0 = cum_y1[idx], cum_y0[idx]
    ratio = np.divide(n1, n0, out=np.zeros_like(n1, dtype=float), where=n0 > 0)
    qini_values = y1 - y0 * ratio

    random_line = fractions * qini_values[-1]
    qini_coefficient = float(np.trapezoid(qini_values - random_line, fractions))

    return fractions, qini_values, qini_coefficient


def uplift_by_decile(uplift_score, treatment, outcome, n_bins: int = 10) -> pd.DataFrame:
    """Actual (experimentally measured) treatment effect within each
    predicted-uplift decile — the standard calibration check for an uplift
    model: does the real effect actually increase across deciles the model
    ranks higher?
    """
    frame = pd.DataFrame({
        "score": np.asarray(uplift_score),
        "treatment": np.asarray(treatment),
        "outcome": np.asarray(outcome),
    })
    frame["decile"] = pd.qcut(frame["score"], n_bins, labels=False, duplicates="drop")

    rows = []
    for decile, group in frame.groupby("decile"):
        treated = group.loc[group["treatment"] == 1, "outcome"]
        control = group.loc[group["treatment"] == 0, "outcome"]
        rows.append({
            "decile": int(decile),
            "n": len(group),
            "n_treated": len(treated),
            "n_control": len(control),
            "mean_predicted_uplift": group["score"].mean(),
            "actual_uplift": treated.mean() - control.mean(),
        })

    return pd.DataFrame(rows).sort_values("decile", ascending=False).reset_index(drop=True)
