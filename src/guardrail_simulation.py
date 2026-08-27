# Simulations for the Week 4 metrics-framework worked examples.
#
# Four self-contained scenarios, each simulating individual-level data so the
# "failure mode" being demonstrated is a property of the data, not something
# hard-coded into a chart:
#
#   simulate_checkout_experiment   - the core deliverable: a primary metric
#       that genuinely improves while a guardrail degrades, and a $-based
#       decision that flips the naive "ship it" call.
#   simulate_simpsons_paradox      - an aggregate result that reverses sign
#       once disaggregated by a confounding subgroup.
#   simulate_novelty_effect        - a treatment effect that's real but
#       decays toward a much smaller steady state, so an early readout
#       overstates the long-run impact.
#   simulate_shared_inventory_spillover - a finite shared resource that lets
#       one arm's behaviour contaminate the other's measured outcome
#       (an SUTVA / interference violation).

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Guardrail violation: "Quick Checkout" scenario
# --------------------------------------------------------------------------

def simulate_checkout_experiment(
    n_per_arm: int,
    control_conversion: float = 0.620,
    treatment_lift: float = 0.020,
    stale_address_rate: float = 0.10,
    guardrail_base_rate: float = 0.010,
    guardrail_stale_treatment_rate: float = 0.40,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Simulate an A/B test of a 'Quick Checkout' flow that skips the
    shipping-address confirmation step for returning users.

    Mechanism: treatment genuinely lifts checkout completion (fewer steps,
    less friction). Independently, some users' saved address is stale
    (`stale_address_rate`) without anyone knowing it at experiment time. In
    the control flow, the confirmation step catches and corrects a stale
    address before the order ships, so control's guardrail rate is just the
    baseline rate regardless of staleness. In treatment, that step is
    skipped — so a stale-address order ships to the wrong place, and only
    *those* users see an elevated failure rate. The guardrail outcome is
    only defined for users who actually placed an order (conversion=1);
    there's no delivery to fail without one.
    """
    rng = np.random.default_rng(random_state)
    n = n_per_arm * 2
    treatment = np.concatenate([np.zeros(n_per_arm, dtype=int), np.ones(n_per_arm, dtype=int)])
    stale_address = rng.binomial(1, stale_address_rate, n)

    conversion_p = np.where(treatment == 1, control_conversion + treatment_lift, control_conversion)
    conversion = rng.binomial(1, conversion_p)

    guardrail_p = np.where(
        (treatment == 1) & (stale_address == 1),
        guardrail_stale_treatment_rate,
        guardrail_base_rate,
    )
    guardrail_failure = np.where(
        conversion == 1,
        rng.binomial(1, guardrail_p).astype(float),
        np.nan,
    )

    return pd.DataFrame({
        "treatment": treatment,
        "stale_address": stale_address,
        "conversion": conversion,
        "guardrail_failure": guardrail_failure,
    })


def business_impact(
    df: pd.DataFrame,
    profit_per_order: float,
    cost_per_guardrail_failure: float,
) -> pd.DataFrame:
    """Translate conversion + guardrail rates into $ profit per checkout
    session, per arm, on a common denominator (checkout sessions, not
    orders) so the two metrics can be netted against each other directly.
    """
    rows = []
    for arm, group in df.groupby("treatment"):
        conversion_rate = group["conversion"].mean()
        orders = group[group["conversion"] == 1]
        guardrail_rate_among_orders = orders["guardrail_failure"].mean()
        failures_per_session = conversion_rate * guardrail_rate_among_orders
        profit_per_session = (
            conversion_rate * profit_per_order
            - failures_per_session * cost_per_guardrail_failure
        )
        rows.append({
            "arm": "Treatment" if arm == 1 else "Control",
            "conversion_rate": conversion_rate,
            "guardrail_rate_among_orders": guardrail_rate_among_orders,
            "failures_per_session": failures_per_session,
            "profit_per_session": profit_per_session,
        })
    return pd.DataFrame(rows).set_index("arm")


# --------------------------------------------------------------------------
# 2. Simpson's paradox: confounded rollout ramp
# --------------------------------------------------------------------------

def simulate_simpsons_paradox(
    n_days: int = 14,
    sessions_per_day: int = 4000,
    treatment_share_start: float = 0.10,
    treatment_share_end: float = 0.70,
    desktop_share_start: float = 0.30,
    desktop_share_end: float = 0.75,
    desktop_base_rate: float = 0.25,
    mobile_base_rate: float = 0.08,
    true_effect_desktop: float = -0.010,
    true_effect_mobile: float = -0.010,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Simulate a phased feature rollout (treatment share ramps up over the
    test) running at the same time as an unrelated traffic-mix shift (e.g. a
    marketing push skews traffic toward desktop over the same days).

    The true effect is negative in both segments. But because both the
    treatment share *and* the desktop share climb together over the test
    window, later days are simultaneously "more treatment" and "more
    desktop" (which converts far better regardless of treatment) — enough
    to flip the naive, pooled comparison's sign. This is why Simpson's
    paradox in a real A/B test is usually a symptom of an allocation problem
    (a ramping rollout, a mid-test traffic-mix change), not a statistical
    curiosity: the fix is the same diligence as the Week 1 randomization
    check, applied over time instead of just at the start.
    """
    rng = np.random.default_rng(random_state)
    rows = []
    for day in range(n_days):
        frac = day / (n_days - 1) if n_days > 1 else 0.0
        treatment_share = treatment_share_start + frac * (treatment_share_end - treatment_share_start)
        desktop_share = desktop_share_start + frac * (desktop_share_end - desktop_share_start)

        treatment = rng.binomial(1, treatment_share, sessions_per_day)
        desktop = rng.binomial(1, desktop_share, sessions_per_day)

        base_rate = np.where(desktop == 1, desktop_base_rate, mobile_base_rate)
        true_effect = np.where(desktop == 1, true_effect_desktop, true_effect_mobile)
        outcome_p = np.clip(base_rate + treatment * true_effect, 0.0, 1.0)
        outcome = rng.binomial(1, outcome_p)

        rows.append(pd.DataFrame({
            "day": day,
            "treatment": treatment,
            "desktop": desktop,
            "outcome": outcome,
        }))

    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# 3. Novelty effect: a real but decaying treatment effect
# --------------------------------------------------------------------------

def simulate_novelty_effect(
    n_days: int = 21,
    sessions_per_arm_per_day: int = 2000,
    control_rate: float = 0.10,
    peak_lift: float = 0.05,
    steady_state_lift: float = 0.005,
    decay_days: float = 5.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Simulate a treatment effect that starts large (curiosity/novelty) and
    decays exponentially toward a much smaller steady state as users
    habituate. Returns day-level counts, not individual rows — this
    simulation is about the *time series* of the effect, not user-level
    heterogeneity.
    """
    rng = np.random.default_rng(random_state)
    rows = []
    for day in range(n_days):
        true_lift = steady_state_lift + (peak_lift - steady_state_lift) * np.exp(-day / decay_days)

        control_conversions = rng.binomial(sessions_per_arm_per_day, control_rate)
        treatment_conversions = rng.binomial(sessions_per_arm_per_day, control_rate + true_lift)

        rows.append({
            "day": day,
            "true_lift": true_lift,
            "control_conversions": control_conversions,
            "treatment_conversions": treatment_conversions,
            "control_n": sessions_per_arm_per_day,
            "treatment_n": sessions_per_arm_per_day,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Network / spillover contamination: shared finite inventory
# --------------------------------------------------------------------------

def simulate_shared_inventory_spillover(
    n_per_arm: int,
    initial_inventory: int,
    control_purchase_intent: float = 0.20,
    treatment_purchase_intent: float = 0.30,
    random_state: int | None = None,
) -> dict:
    """Simulate treatment and control users arriving in random order and
    drawing from one shared, finite inventory pool — e.g. a promo shown to
    the treatment arm that increases purchase intent, competing with
    control for the same limited stock.

    Returns both the *contaminated* measured rates (what a real experiment
    sharing inventory would observe) and an *oracle* comparison run with
    unlimited inventory (each user's purchase depends only on their own
    arm's intent, with no cross-arm interference) — the gap between the two
    is the spillover bias introduced purely by the shared constraint, a
    violation of SUTVA (one unit's assignment affecting another unit's
    outcome).
    """
    rng = np.random.default_rng(random_state)
    n = n_per_arm * 2
    treatment = np.concatenate([np.zeros(n_per_arm, dtype=int), np.ones(n_per_arm, dtype=int)])
    order = rng.permutation(n)
    treatment_shuffled = treatment[order]

    intent_p = np.where(treatment_shuffled == 1, treatment_purchase_intent, control_purchase_intent)
    wants_to_buy = rng.binomial(1, intent_p)

    inventory_remaining = initial_inventory
    purchased_contaminated = np.zeros(n, dtype=int)
    for i in range(n):
        if wants_to_buy[i] == 1 and inventory_remaining > 0:
            purchased_contaminated[i] = 1
            inventory_remaining -= 1

    contaminated_treatment_rate = purchased_contaminated[treatment_shuffled == 1].mean()
    contaminated_control_rate = purchased_contaminated[treatment_shuffled == 0].mean()

    oracle_treatment_rate = wants_to_buy[treatment_shuffled == 1].mean()
    oracle_control_rate = wants_to_buy[treatment_shuffled == 0].mean()

    return {
        "contaminated_treatment_rate": float(contaminated_treatment_rate),
        "contaminated_control_rate": float(contaminated_control_rate),
        "contaminated_effect": float(contaminated_treatment_rate - contaminated_control_rate),
        "oracle_treatment_rate": float(oracle_treatment_rate),
        "oracle_control_rate": float(oracle_control_rate),
        "oracle_effect": float(oracle_treatment_rate - oracle_control_rate),
        "units_sold": int(purchased_contaminated.sum()),
        "initial_inventory": initial_inventory,
    }
