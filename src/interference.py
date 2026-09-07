# Interference / network effects: SUTVA violations and the designs that
# handle them (Week 7 stretch note).
#
# Every test in Weeks 1-6 assumed SUTVA — "stable unit treatment value
# assumption" — one unit's outcome depends only on its *own* treatment
# assignment, not anyone else's. That assumption breaks whenever units
# influence each other:
#
#   - social-graph interference: a treated user's behaviour spills to their
#     untreated friends (referrals, feed content, messaging). An
#     individually randomized A/B test then contaminates the control group
#     and mismeasures the effect.
#   - marketplace / resource interference: treatment and control draw on one
#     shared finite pool (inventory, drivers, ad budget), so helping the
#     treatment arm can hurt control directly. Week 4's
#     `simulate_shared_inventory_spillover` is a worked example of this one.
#
# This module covers the first case and its standard design fix
# (cluster randomization), plus a compact switchback simulation for the
# marketplace case where units can't be separated in space so you separate
# them in time instead.

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# 1. A clustered social network (stochastic block model)
# --------------------------------------------------------------------------

def make_clustered_network(
    n_clusters: int,
    cluster_size: int,
    p_within: float = 0.30,
    p_between: float = 0.002,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Undirected stochastic block model: dense connections inside a cluster,
    sparse connections across clusters — the structure that makes cluster
    randomization work (most of any unit's neighbours share its cluster, so
    most spillover stays within one arm).

    Returns (adjacency, cluster_labels):
      adjacency: (N, N) float array, symmetric, zero diagonal.
      cluster_labels: (N,) int array in [0, n_clusters).
    """
    rng = np.random.default_rng(random_state)
    n = n_clusters * cluster_size
    labels = np.repeat(np.arange(n_clusters), cluster_size)

    same_cluster = labels[:, None] == labels[None, :]
    probs = np.where(same_cluster, p_within, p_between)

    upper = np.triu(rng.random((n, n)) < probs, k=1)
    adj = (upper | upper.T).astype(float)
    return adj, labels


def _neighbour_treated_fraction(adj: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Fraction of each unit's neighbours that are treated (0 if isolated)."""
    degree = adj.sum(axis=1)
    treated_neighbours = adj @ treatment.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(degree > 0, treated_neighbours / degree, 0.0)
    return frac


# --------------------------------------------------------------------------
# 2. Outcome model with spillover
# --------------------------------------------------------------------------

def simulate_network_outcomes(
    adj: np.ndarray,
    treatment: np.ndarray,
    baseline: float,
    direct_effect: float,
    spillover_effect: float,
    cluster_labels: np.ndarray | None = None,
    cluster_effects: np.ndarray | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Binary outcome for each unit:

        p_i = baseline
              + direct_effect   * T_i
              + spillover_effect * (fraction of i's neighbours treated)
              + u_{c(i)}                              # optional cluster shift

    y_i ~ Bernoulli(p_i). The spillover term is what violates SUTVA: unit
    i's outcome depends on its neighbours' assignments, not just its own.

    `cluster_effects` (one value per cluster, indexed by `cluster_labels`)
    adds a persistent per-cluster shift to the rate — real communities
    differ for reasons unrelated to treatment. It leaves an individually
    randomized estimate almost untouched (it averages out across clusters)
    but inflates a cluster-randomized estimate's variance, because whichever
    clusters land in the treatment arm carry their shifts with them. That
    gap is the intra-cluster correlation / design-effect cost.
    """
    rng = np.random.default_rng(random_state)
    frac = _neighbour_treated_fraction(adj, treatment)
    p = baseline + direct_effect * treatment + spillover_effect * frac
    if cluster_effects is not None:
        p = p + cluster_effects[cluster_labels]
    p = np.clip(p, 0.0, 1.0)
    return rng.binomial(1, p)


def true_global_effect(direct_effect: float, spillover_effect: float) -> float:
    """The estimand a launch decision actually cares about: the difference
    between everyone treated and no one treated.

    With everyone treated the neighbour-treated fraction is 1; with no one
    treated it is 0, so the global average treatment effect is simply
    direct_effect + spillover_effect. An individually randomized test that
    only recovers `direct_effect` understates this by the whole spillover
    term.
    """
    return direct_effect + spillover_effect


# --------------------------------------------------------------------------
# 3. Two randomization designs on the same network
# --------------------------------------------------------------------------

def estimate_individual_randomization(
    adj: np.ndarray,
    baseline: float,
    direct_effect: float,
    spillover_effect: float,
    treat_share: float = 0.5,
    n_simulations: int = 200,
    cluster_labels: np.ndarray | None = None,
    cluster_effects: np.ndarray | None = None,
    random_state: int | None = None,
) -> dict:
    """Bernoulli-assign each unit independently, then take the naive
    difference in mean outcome between arms.

    At a 50/50 split every unit — treated or control — has roughly 50% of
    its neighbours treated, so the spillover term nearly cancels in the
    difference and the estimate collapses toward `direct_effect` alone.
    """
    rng = np.random.default_rng(random_state)
    n = adj.shape[0]
    estimates = np.empty(n_simulations)
    for s in range(n_simulations):
        treatment = (rng.random(n) < treat_share).astype(int)
        y = simulate_network_outcomes(
            adj, treatment, baseline, direct_effect, spillover_effect,
            cluster_labels=cluster_labels, cluster_effects=cluster_effects,
            random_state=rng.integers(1 << 32),
        )
        estimates[s] = y[treatment == 1].mean() - y[treatment == 0].mean()
    return {
        "mean_estimate": float(estimates.mean()),
        "se_estimate": float(estimates.std(ddof=1)),
        "estimates": estimates,
    }


def estimate_cluster_randomization(
    adj: np.ndarray,
    cluster_labels: np.ndarray,
    baseline: float,
    direct_effect: float,
    spillover_effect: float,
    treat_share: float = 0.5,
    n_simulations: int = 200,
    cluster_effects: np.ndarray | None = None,
    random_state: int | None = None,
) -> dict:
    """Assign whole clusters to treatment or control, then compare arms.

    Because neighbours mostly share a cluster, a treated unit now sits in a
    mostly-treated neighbourhood and a control unit in a mostly-control one,
    so the spillover term no longer cancels — the estimate picks up almost
    all of `direct_effect + spillover_effect`. The price is variance: the
    effective sample size is the number of *clusters*, not units, so the
    sampling spread of the estimate is much wider.

    Also returns the cluster-level SE (the correct one to report), computed
    from cluster-mean outcomes in a single representative assignment.
    """
    rng = np.random.default_rng(random_state)
    n_clusters = int(cluster_labels.max()) + 1

    estimates = np.empty(n_simulations)
    for s in range(n_simulations):
        treated_clusters = rng.random(n_clusters) < treat_share
        treatment = treated_clusters[cluster_labels].astype(int)
        y = simulate_network_outcomes(
            adj, treatment, baseline, direct_effect, spillover_effect,
            cluster_labels=cluster_labels, cluster_effects=cluster_effects,
            random_state=rng.integers(1 << 32),
        )
        estimates[s] = y[treatment == 1].mean() - y[treatment == 0].mean()

    # Cluster-level SE from one representative assignment: treat each
    # cluster's mean outcome as the unit of analysis (the correct unit when
    # clusters are what was randomized).
    treated_clusters = rng.random(n_clusters) < treat_share
    treatment = treated_clusters[cluster_labels].astype(int)
    y = simulate_network_outcomes(
        adj, treatment, baseline, direct_effect, spillover_effect,
        cluster_labels=cluster_labels, cluster_effects=cluster_effects,
        random_state=rng.integers(1 << 32),
    )
    cluster_means = np.array([y[cluster_labels == c].mean() for c in range(n_clusters)])
    cm_t = cluster_means[treated_clusters]
    cm_c = cluster_means[~treated_clusters]
    cluster_se = float(np.sqrt(
        cm_t.var(ddof=1) / len(cm_t) + cm_c.var(ddof=1) / len(cm_c)
    ))

    return {
        "mean_estimate": float(estimates.mean()),
        "se_estimate": float(estimates.std(ddof=1)),
        "cluster_level_se_single_run": cluster_se,
        "estimates": estimates,
        "n_clusters": int(n_clusters),
    }


# --------------------------------------------------------------------------
# 4. Switchback test for marketplace / temporal interference
# --------------------------------------------------------------------------

def simulate_switchback(
    n_periods: int,
    baseline: float,
    period_effect: float,
    settle_fraction: float = 0.5,
    ar_rho: float = 0.5,
    noise_sd: float = 0.05,
    block_length: int = 1,
    washout: bool = False,
    random_state: int | None = None,
) -> dict:
    """When treatment and control can't be separated across units (they all
    share one marketplace — drivers, inventory, an auction), you separate
    them across time: the whole system is switched between treatment and
    control over successive blocks of `block_length` periods.

    Two real frictions are modelled:

      1. **Settling / carryover.** The first period after a switch is only
         partly transitioned: its effective assignment is a blend
         `(1 - settle_fraction) * A_t + settle_fraction * A_{t-1}`. Short
         blocks are almost all first-periods, so this attenuates the
         measured effect toward zero. `washout=True` drops the first period
         of every block from the analysis — which removes the bias, but is
         only affordable when blocks are long (dropping the first period of
         length-1 blocks discards everything).

      2. **Serial correlation.** Marketplace state persists: the period
         noise follows an AR(1) with coefficient `ar_rho`. Treating periods
         as IID then *understates* the standard error; a block-level SE
         (block means as the unit of analysis) is honest about it, and gets
         wider as blocks get longer and fewer.

    Returns one realized run: the naive effect, its bias vs. the true
    `period_effect`, and the IID vs. block-level SEs.
    """
    rng = np.random.default_rng(random_state)

    n_blocks = int(np.ceil(n_periods / block_length))
    block_assign = (rng.random(n_blocks) < 0.5).astype(int)
    assign = np.repeat(block_assign, block_length)[:n_periods]

    period_idx = np.arange(n_periods)
    is_block_start = (period_idx % block_length) == 0
    prev_assign = np.concatenate([[assign[0]], assign[:-1]])
    switched = is_block_start & (assign != prev_assign)

    effective_a = assign.astype(float)
    effective_a[switched] = (
        (1 - settle_fraction) * assign[switched] + settle_fraction * prev_assign[switched]
    )

    # AR(1) noise
    eps = np.empty(n_periods)
    eps[0] = rng.normal(0, noise_sd)
    innov_sd = noise_sd * np.sqrt(1 - ar_rho ** 2)
    for t in range(1, n_periods):
        eps[t] = ar_rho * eps[t - 1] + rng.normal(0, innov_sd)

    means = baseline + period_effect * effective_a + eps

    keep = ~is_block_start if washout else np.ones(n_periods, dtype=bool)
    if washout and keep.sum() == 0:  # block_length == 1: nothing survives
        return {
            "naive_effect": float("nan"), "true_period_effect": float(period_effect),
            "carryover_bias": float("nan"), "iid_se": float("nan"),
            "block_se": float("nan"), "n_periods": n_periods,
            "block_length": block_length, "washout": washout,
            "note": "washout with block_length=1 discards every period",
        }

    t_mask = keep & (assign == 1)
    c_mask = keep & (assign == 0)
    if t_mask.sum() < 2 or c_mask.sum() < 2:  # a whole arm went unsampled
        return {
            "naive_effect": float("nan"), "true_period_effect": float(period_effect),
            "carryover_bias": float("nan"), "iid_se": float("nan"),
            "block_se": float("nan"), "n_periods": n_periods,
            "block_length": block_length, "washout": washout,
            "note": "too few periods in one arm at this block length",
        }
    naive_effect = means[t_mask].mean() - means[c_mask].mean()

    iid_se = np.sqrt(
        means[t_mask].var(ddof=1) / t_mask.sum()
        + means[c_mask].var(ddof=1) / c_mask.sum()
    )

    block_means, block_lab = [], []
    for b in range(n_blocks):
        sl = slice(b * block_length, (b + 1) * block_length)
        m = means[sl][keep[sl]]
        if m.size:
            block_means.append(m.mean())
            block_lab.append(block_assign[b])
    block_means = np.array(block_means)
    block_lab = np.array(block_lab)
    bm_t, bm_c = block_means[block_lab == 1], block_means[block_lab == 0]
    if len(bm_t) >= 2 and len(bm_c) >= 2:
        block_se = float(np.sqrt(bm_t.var(ddof=1) / len(bm_t) + bm_c.var(ddof=1) / len(bm_c)))
    else:
        block_se = float("nan")  # too few blocks per arm to estimate

    return {
        "naive_effect": float(naive_effect),
        "true_period_effect": float(period_effect),
        "carryover_bias": float(naive_effect - period_effect),
        "iid_se": float(iid_se),
        "block_se": float(block_se),
        "n_periods": n_periods,
        "block_length": block_length,
        "washout": washout,
    }
