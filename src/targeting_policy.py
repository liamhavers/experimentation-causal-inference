# Cost-based targeting policy built on top of an uplift model (Week 3).
#
# Turns an uplift model's predictions into an actual decision rule (treat a
# user only if predicted uplift clears the cost of treating them), then
# estimates what that policy would actually be worth using the real
# experimental data — not the model's own predictions — via inverse
# propensity weighting (IPW).
#
# Why IPW rather than just averaging the model's predicted uplift: the
# predicted uplift is only as good as the model, and grading the policy
# using the same model that built it is circular. Because treatment was
# randomly assigned in the underlying experiment, a policy's true value can
# instead be estimated directly from held-out *observed* outcomes: for each
# user, use their actual outcome only if their actual treatment happened to
# match what the policy would have assigned them, reweighted by the inverse
# probability of that match. This is the Horvitz-Thompson / IPW policy-value
# estimator, and it's unbiased regardless of whether the uplift model is any
# good — it's evaluating the *policy*, using the experiment as a randomized
# holdout.

from __future__ import annotations

import numpy as np


def policy_from_uplift(uplift_pred, value_per_outcome: float, cost_per_treatment: float) -> np.ndarray:
    """Treat a user iff their predicted uplift is worth more than it costs."""
    uplift_pred = np.asarray(uplift_pred)
    return (uplift_pred * value_per_outcome) > cost_per_treatment


def policy_value_ipw(outcome, treatment_actual, treat_decision, propensity) -> float:
    """Horvitz-Thompson estimate of E[outcome] had every user received
    treat_decision(x) rather than their actual (randomly assigned) treatment.

    propensity: P(treatment_actual=1 | X). A scalar is fine when treatment
    was assigned with a fixed, covariate-independent probability (verify
    this empirically before assuming it, the same way Weeks 1-2 checked
    their own assumptions before applying them).
    """
    outcome = np.asarray(outcome, dtype=float)
    treatment_actual = np.asarray(treatment_actual)
    treat_decision = np.asarray(treat_decision)
    propensity = np.asarray(propensity, dtype=float)

    matches = treatment_actual == treat_decision
    p_observed = np.where(treatment_actual == 1, propensity, 1 - propensity)

    contributions = np.where(matches, outcome / p_observed, 0.0)
    return float(contributions.mean())


def evaluate_policy(
    outcome,
    treatment_actual,
    treat_decision,
    propensity,
    value_per_outcome: float,
    cost_per_treatment: float,
) -> dict:
    """Expected revenue/cost/profit per user under a given targeting policy.

    Cost is computed directly (treat_decision is a deterministic function of
    X, so no IPW correction is needed there) — only the outcome needs IPW,
    since an outcome is only observed under the treatment a user actually
    received.
    """
    pct_treated = float(np.mean(treat_decision))
    expected_outcome_rate = policy_value_ipw(outcome, treatment_actual, treat_decision, propensity)
    expected_revenue = value_per_outcome * expected_outcome_rate
    expected_cost = cost_per_treatment * pct_treated

    return {
        "pct_treated": pct_treated,
        "expected_outcome_rate": expected_outcome_rate,
        "expected_revenue_per_user": expected_revenue,
        "expected_cost_per_user": expected_cost,
        "expected_profit_per_user": expected_revenue - expected_cost,
    }


def compare_policies(
    outcome,
    treatment_actual,
    uplift_pred,
    propensity,
    value_per_outcome: float,
    cost_per_treatment: float,
) -> "pd.DataFrame":
    """Compare the uplift-driven targeting policy against the two obvious
    baselines: treat everyone (blanket rollout) and treat no one.
    """
    import pandas as pd

    treatment_actual = np.asarray(treatment_actual)
    n = len(treatment_actual)

    model_decision = policy_from_uplift(uplift_pred, value_per_outcome, cost_per_treatment)
    blanket_decision = np.ones(n, dtype=bool)
    none_decision = np.zeros(n, dtype=bool)

    rows = {
        "Model-targeted": evaluate_policy(outcome, treatment_actual, model_decision, propensity, value_per_outcome, cost_per_treatment),
        "Blanket (treat all)": evaluate_policy(outcome, treatment_actual, blanket_decision, propensity, value_per_outcome, cost_per_treatment),
        "None (treat no one)": evaluate_policy(outcome, treatment_actual, none_decision, propensity, value_per_outcome, cost_per_treatment),
    }
    result = pd.DataFrame(rows).T
    result["incremental_profit_vs_none"] = result["expected_profit_per_user"] - result.loc["None (treat no one)", "expected_profit_per_user"]
    result["incremental_profit_vs_blanket"] = result["expected_profit_per_user"] - result.loc["Blanket (treat all)", "expected_profit_per_user"]
    return result
