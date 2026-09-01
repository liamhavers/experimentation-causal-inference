# Multi-armed bandits vs. a fixed-horizon A/B test, on the Hillstrom dataset
# used in Week 1 (Week 5).
#
# Every simulated "pull" of an arm draws its reward via bootstrap resampling
# from that arm's *real* recorded `visit` outcomes in the Hillstrom dataset,
# rather than an idealized Bernoulli(p) — this is exactly equivalent in
# distribution (resampling a 0/1 array with replacement is Bernoulli at the
# array's empirical mean), but it keeps the simulation literally grounded in
# the same real numbers Week 1 used, rather than a re-derived summary
# statistic.
#
# All three allocation strategies below are vectorized across `n_simulations`
# independent parallel runs (state arrays of shape (n_simulations, n_arms)),
# since a single run of a stochastic bandit is noisy and the standard way to
# report bandit performance is an *expected* regret curve averaged over many
# runs — the per-round loop itself is necessarily sequential (each round's
# choice depends on the outcome of every prior round), so vectorizing across
# simulations rather than rounds is what actually buys the speed.

from __future__ import annotations

import numpy as np


def run_fixed_split(arm_reward_pools, n_rounds: int, n_simulations: int = 1, random_state=None) -> dict:
    """Baseline: a fixed-horizon A/B/n test with an equal split across arms,
    decided once and never adapted — exactly the design actually used for
    the real Hillstrom test in Week 1.
    """
    rng = np.random.default_rng(random_state)
    n_arms = len(arm_reward_pools)

    chosen_arms = rng.integers(0, n_arms, size=(n_rounds, n_simulations))
    rewards = np.empty((n_rounds, n_simulations))
    for arm in range(n_arms):
        mask = chosen_arms == arm
        rewards[mask] = rng.choice(arm_reward_pools[arm], size=int(mask.sum()), replace=True)

    return {"chosen_arms": chosen_arms, "rewards": rewards}


def run_epsilon_greedy(
    arm_reward_pools,
    n_rounds: int,
    epsilon: float = 0.1,
    n_simulations: int = 1,
    random_state=None,
) -> dict:
    """Epsilon-greedy: explore a uniformly random arm with probability
    epsilon, otherwise exploit the arm with the highest sample mean so far.
    """
    rng = np.random.default_rng(random_state)
    n_arms = len(arm_reward_pools)

    counts = np.zeros((n_simulations, n_arms))
    sums = np.zeros((n_simulations, n_arms))
    chosen_arms = np.zeros((n_rounds, n_simulations), dtype=int)
    rewards = np.zeros((n_rounds, n_simulations))

    for t in range(n_rounds):
        means = np.divide(sums, counts, out=np.full_like(sums, np.inf), where=counts > 0)
        greedy_arm = np.argmax(means, axis=1)
        explore = rng.random(n_simulations) < epsilon
        random_arm = rng.integers(0, n_arms, n_simulations)
        arm_t = np.where(explore, random_arm, greedy_arm)

        reward_t = np.empty(n_simulations)
        for arm in range(n_arms):
            mask = arm_t == arm
            n_mask = int(mask.sum())
            if n_mask > 0:
                reward_t[mask] = rng.choice(arm_reward_pools[arm], size=n_mask, replace=True)
                counts[mask, arm] += 1
                sums[mask, arm] += reward_t[mask]

        chosen_arms[t] = arm_t
        rewards[t] = reward_t

    return {"chosen_arms": chosen_arms, "rewards": rewards}


def run_thompson_sampling(
    arm_reward_pools,
    n_rounds: int,
    n_simulations: int = 1,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    random_state=None,
) -> dict:
    """Thompson sampling with a Beta-Bernoulli model: maintain a Beta
    posterior per arm (conjugate for a 0/1 reward), draw one sample per arm
    per round, and pull whichever arm's sample is highest. Naturally
    balances exploration and exploitation through posterior uncertainty,
    with no tunable exploration rate to set.
    """
    rng = np.random.default_rng(random_state)
    n_arms = len(arm_reward_pools)

    alpha = np.full((n_simulations, n_arms), prior_alpha)
    beta_ = np.full((n_simulations, n_arms), prior_beta)
    chosen_arms = np.zeros((n_rounds, n_simulations), dtype=int)
    rewards = np.zeros((n_rounds, n_simulations))

    for t in range(n_rounds):
        samples = rng.beta(alpha, beta_)
        arm_t = np.argmax(samples, axis=1)

        reward_t = np.empty(n_simulations)
        for arm in range(n_arms):
            mask = arm_t == arm
            n_mask = int(mask.sum())
            if n_mask > 0:
                reward_t[mask] = rng.choice(arm_reward_pools[arm], size=n_mask, replace=True)
                alpha[mask, arm] += reward_t[mask]
                beta_[mask, arm] += 1 - reward_t[mask]

        chosen_arms[t] = arm_t
        rewards[t] = reward_t

    return {"chosen_arms": chosen_arms, "rewards": rewards}


def expected_cumulative_regret(chosen_arms: np.ndarray, true_arm_rates) -> np.ndarray:
    """Cumulative regret using the *true* per-arm rates rather than the
    noisy realized rewards — the standard, low-variance way to evaluate a
    bandit algorithm (the realized-reward regret is an unbiased but far
    noisier estimate of the same quantity).

    chosen_arms: (n_rounds, n_simulations) array of arm indices.
    Returns: (n_rounds, n_simulations) array of cumulative regret.
    """
    true_arm_rates = np.asarray(true_arm_rates, dtype=float)
    best_rate = true_arm_rates.max()
    per_round_regret = best_rate - true_arm_rates[chosen_arms]
    return np.cumsum(per_round_regret, axis=0)


def final_arm_pull_counts(chosen_arms: np.ndarray, n_arms: int) -> np.ndarray:
    """Count of pulls per arm per simulation, from a (n_rounds, n_simulations)
    chosen_arms array. Returns (n_simulations, n_arms).
    """
    n_simulations = chosen_arms.shape[1]
    counts = np.zeros((n_simulations, n_arms), dtype=int)
    for arm in range(n_arms):
        counts[:, arm] = (chosen_arms == arm).sum(axis=0)
    return counts
