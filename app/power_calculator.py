# Streamlit front-end for the Week 1 power / MDE calculator (Week 7 stretch).
#
# Wraps src/power_analysis.py unchanged — every number shown here comes from
# the same functions the Week 1 notebook uses (required_sample_size,
# minimum_detectable_effect, simulate_peeking). The app is just an
# interactive surface over them: move the sliders, watch the sample size,
# runtime, and the sample-size-vs-MDE curve update, and run the peeking
# simulation without opening the notebook.
#
# Run:  streamlit run app/power_calculator.py

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from power_analysis import (  # noqa: E402
    minimum_detectable_effect,
    required_sample_size,
    simulate_peeking,
)

HILLSTROM_CONTROL_RATE = 0.1062  # observed No E-Mail visit rate, Week 1

st.set_page_config(page_title="Power / MDE Calculator", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------
# Sidebar: parameters shared across all three tabs
# --------------------------------------------------------------------------
st.sidebar.title("Test parameters")

st.session_state.setdefault("baseline_pct", 10.0)
if st.sidebar.button("Load Hillstrom preset", help="Baseline = the real No E-Mail visit rate (10.62%)"):
    st.session_state["baseline_pct"] = HILLSTROM_CONTROL_RATE * 100

baseline_pct = st.sidebar.number_input(
    "Baseline conversion rate (%)",
    min_value=0.01,
    max_value=99.99,
    step=0.5,
    key="baseline_pct",
    help="Control-arm rate for the metric you're testing (p1).",
)
baseline_rate = baseline_pct / 100.0

alpha = st.sidebar.select_slider(
    "Significance level α (two-sided)",
    options=[0.01, 0.02, 0.05, 0.10],
    value=0.05,
)
power = st.sidebar.slider(
    "Power (1 − β)",
    min_value=0.50,
    max_value=0.99,
    value=0.80,
    step=0.01,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "All calculations use `src/power_analysis.py` — the two-proportion "
    "z-test power formula (pooled variance under the null, unpooled under "
    "the alternative). Same code as `notebooks/01_power_analysis.ipynb`."
)


# --------------------------------------------------------------------------
st.title("📊 Power / MDE Calculator")
st.markdown(
    "Interactive front-end for the Week 1 experiment-design toolkit. "
    "Three questions, one underlying equation: **how big a sample**, "
    "**how small an effect**, and **what does peeking cost**."
)

tab_n, tab_mde, tab_peek = st.tabs(
    ["Sample size", "Minimum detectable effect", "Peeking simulator"]
)


# --------------------------------------------------------------------------
# Tab 1 — sample size given a target MDE
# --------------------------------------------------------------------------
with tab_n:
    st.subheader("How many users per arm?")
    st.caption(
        "Pick the smallest effect you need to reliably detect; get the "
        "per-arm sample size, the total, and the calendar runtime."
    )

    left, right = st.columns([1, 1])
    with left:
        effect_mode = st.radio(
            "Specify the minimum detectable effect as",
            ["Relative lift (%)", "Absolute lift (pp)"],
            horizontal=True,
        )
        if effect_mode.startswith("Relative"):
            rel_lift_pct = st.slider(
                "Minimum detectable relative lift (%)",
                min_value=1.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
            )
            mde_absolute = baseline_rate * rel_lift_pct / 100.0
        else:
            mde_pp = st.slider(
                "Minimum detectable absolute lift (percentage points)",
                min_value=0.1,
                max_value=min(50.0, (1 - baseline_rate) * 100 - 0.1),
                value=min(2.0, (1 - baseline_rate) * 100 - 0.1),
                step=0.1,
            )
            mde_absolute = mde_pp / 100.0

        daily_traffic = st.number_input(
            "Total daily traffic (both arms)",
            min_value=1,
            value=5_000,
            step=100,
            key="daily_traffic_tab1",
        )
        split = st.slider(
            "Share of traffic to the smaller arm",
            min_value=0.05,
            max_value=0.50,
            value=0.50,
            step=0.05,
            key="split_tab1",
        )

    treatment_rate = baseline_rate + mde_absolute
    if not 0 < treatment_rate < 1:
        st.error("Baseline + MDE must stay within (0, 100%). Lower the effect size.")
    else:
        res = required_sample_size(baseline_rate, mde_absolute, alpha=alpha, power=power)
        runtime = res.runtime_days(daily_traffic, traffic_split=split)

        with right:
            st.metric("Sample size per arm", f"{res.n_per_variant:,}")
            st.metric("Total sample size", f"{res.n_total:,}")
            st.metric("Runtime", f"{runtime:,.1f} days")
            st.caption(
                f"Detecting {baseline_rate:.2%} → {treatment_rate:.2%} "
                f"(+{mde_absolute * 100:.2f}pp, "
                f"{mde_absolute / baseline_rate:.1%} relative) "
                f"at α={alpha}, power={power:.0%}."
            )

        # sample-size-vs-MDE curve, with the current choice marked
        rel_grid = np.linspace(2, 50, 60)
        n_grid = []
        for r in rel_grid:
            m = baseline_rate * r / 100.0
            if 0 < baseline_rate + m < 1:
                n_grid.append(required_sample_size(baseline_rate, m, alpha=alpha, power=power).n_per_variant)
            else:
                n_grid.append(np.nan)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rel_grid, n_grid, color="#4C72B0", lw=2)
        ax.scatter(
            [mde_absolute / baseline_rate * 100],
            [res.n_per_variant],
            color="#C44E52",
            zorder=5,
            s=60,
            label="your choice",
        )
        ax.set_xlabel("Minimum detectable effect (% relative lift)")
        ax.set_ylabel("Sample size per arm")
        ax.set_title(f"Sample size vs. MDE at {baseline_rate:.2%} baseline")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.caption(
            "Required sample grows roughly with 1/MDE² — halving the effect "
            "you chase roughly quadruples the sample. That knee is why teams "
            "reach for variance reduction (CUPED, Week 2) instead of just "
            "buying more traffic."
        )


# --------------------------------------------------------------------------
# Tab 2 — MDE given a fixed sample size
# --------------------------------------------------------------------------
with tab_mde:
    st.subheader("What's the smallest effect this sample can detect?")
    st.caption(
        "The question that matters when traffic or timeline — not the effect "
        "size — is the real constraint."
    )

    left, right = st.columns([1, 1])
    with left:
        size_mode = st.radio(
            "Specify the available sample as",
            ["Directly (n per arm)", "Daily traffic × runtime"],
            horizontal=True,
        )
        if size_mode.startswith("Directly"):
            n_per_variant = st.number_input(
                "Sample size per arm",
                min_value=10,
                value=20_000,
                step=1_000,
            )
        else:
            dt = st.number_input(
                "Total daily traffic (both arms)", min_value=1, value=5_000, step=100, key="daily_traffic_tab2"
            )
            days = st.number_input("Planned runtime (days)", min_value=1, value=14, step=1, key="days_tab2")
            split2 = st.slider(
                "Share of traffic to the smaller arm",
                min_value=0.05,
                max_value=0.50,
                value=0.50,
                step=0.05,
                key="split_tab2",
            )
            n_per_variant = int(dt * min(split2, 1 - split2) * days)
            st.caption(f"→ {n_per_variant:,} users per arm")

    mde_abs = minimum_detectable_effect(baseline_rate, int(n_per_variant), alpha=alpha, power=power)
    with right:
        st.metric("Detectable absolute lift", f"{mde_abs * 100:.2f} pp")
        st.metric("Detectable relative lift", f"{mde_abs / baseline_rate:.1%}")
        st.caption(
            f"With {int(n_per_variant):,} per arm you can reliably detect a "
            f"move from {baseline_rate:.2%} to "
            f"{baseline_rate + mde_abs:.2%} at α={alpha}, power={power:.0%}."
        )

    n_grid = np.unique(np.geomspace(200, max(2_000, n_per_variant * 3), 60).astype(int))
    mde_grid = [minimum_detectable_effect(baseline_rate, int(n), alpha=alpha, power=power) for n in n_grid]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_grid, np.array(mde_grid) / baseline_rate * 100, color="#55A868", lw=2)
    ax.scatter([n_per_variant], [mde_abs / baseline_rate * 100], color="#C44E52", zorder=5, s=60, label="your sample")
    ax.set_xscale("log")
    ax.set_xlabel("Sample size per arm (log scale)")
    ax.set_ylabel("Detectable relative lift (%)")
    ax.set_title(f"MDE vs. sample size at {baseline_rate:.2%} baseline")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    st.pyplot(fig)


# --------------------------------------------------------------------------
# Tab 3 — the peeking / repeated-significance-testing simulator
# --------------------------------------------------------------------------
with tab_peek:
    st.subheader("What does peeking cost?")
    st.caption(
        "A p-value threshold controls the false positive rate at α only if "
        "you look **once**, at a pre-committed sample size. This simulates "
        "checking repeatedly and stopping at the first significant look — "
        "with **no true effect** in any simulated experiment."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        n_max = st.number_input("Max sample per arm", min_value=200, value=4_000, step=200)
        n_checks = st.slider("Number of interim looks", min_value=2, max_value=50, value=20)
    with c2:
        n_sims = st.select_slider(
            "Simulated experiments",
            options=[500, 1_000, 2_000, 5_000, 10_000],
            value=2_000,
        )
        true_effect_pp = st.slider(
            "True effect (pp) — leave at 0 for the false-positive demo",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
        )
    with c3:
        seed = st.number_input("Random seed", min_value=0, value=0, step=1)
        run = st.button("Run simulation", type="primary")

    @st.cache_data(show_spinner="Simulating…")
    def _run_peeking(baseline_rate, n_max, n_checks, alpha, n_sims, true_effect, seed):
        return simulate_peeking(
            baseline_rate=baseline_rate,
            n_per_variant_max=int(n_max),
            n_checks=int(n_checks),
            alpha=alpha,
            n_simulations=int(n_sims),
            true_effect=true_effect,
            random_state=int(seed),
        )

    if run:
        out = _run_peeking(
            baseline_rate, n_max, n_checks, alpha, n_sims, true_effect_pp / 100.0, seed
        )
        fixed = out["fixed_horizon_false_positive_rate"]
        peek = out["peeking_false_positive_rate"]
        label = "false positive rate" if true_effect_pp == 0 else "rejection rate"

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Fixed-horizon {label}", f"{fixed:.1%}")
        m2.metric(f"Peeking {label}", f"{peek:.1%}", delta=f"{(peek - fixed):.1%}")
        m3.metric("Inflation factor", f"{peek / fixed:.1f}×" if fixed > 0 else "—")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(
            ["Fixed-horizon\n(one look)", f"Peeking\n({n_checks} looks)"],
            [fixed, peek],
            color=["#4C72B0", "#C44E52"],
        )
        if true_effect_pp == 0:
            axes[0].axhline(alpha, color="0.4", ls="--", lw=1, label=f"nominal α = {alpha}")
            axes[0].legend(frameon=False)
        axes[0].set_ylabel(label)
        axes[0].set_title("Cost of stopping at the first significant look")
        for i, v in enumerate([fixed, peek]):
            axes[0].text(i, v + 0.005, f"{v:.1%}", ha="center")

        axes[1].plot(
            out["check_points"],
            out["per_look_reject_rate"],
            marker="o",
            color="#55A868",
        )
        if true_effect_pp == 0:
            axes[1].axhline(alpha, color="0.4", ls="--", lw=1)
        axes[1].set_xlabel("Sample size per arm at the look")
        axes[1].set_ylabel("Marginal reject rate at this look")
        axes[1].set_title("Each individual look is calibrated…")
        axes[1].grid(alpha=0.3)
        st.pyplot(fig)

        if true_effect_pp == 0:
            st.info(
                f"Each single look rejects ~{alpha:.0%} of the time — as it "
                f"should. The inflation to **{peek:.1%}** comes from taking "
                f"{n_checks} correlated chances to cross the line. Same logic "
                "as a multiple-comparisons problem, spread across time. "
                "If you genuinely need interim looks, use a group-sequential "
                "design (O'Brien-Fleming) or an always-valid test — not a "
                "fixed-horizon p-value checked daily."
            )
    else:
        st.caption("Set the parameters and press **Run simulation**.")
