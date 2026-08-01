"""
Inference Example: Q-Learning
==============================

A classic TABULAR Q-learning implementation for the Weapon-Target
Assignment problem -- a dictionary-based Q-table, not a neural network
(that's what train_dqn.py / inference_example.py do instead).

Why tabular Q-learning needs a FIXED scenario
-----------------------------------------------
A Q-table needs a finite, enumerable set of states. The DQN version's
observation (kill_prob matrix, survival probabilities, etc.) is continuous
and changes every episode (random kill_prob/target_value/weapon_value), so
it can't be indexed by a plain dictionary.

Tabular Q-learning instead solves ONE FIXED problem instance really well:
the kill_prob/target_value/weapon_value are set once at the top of this
file and stay constant across all training episodes. The STATE is just
"which target has each weapon been assigned to so far" (or unassigned),
which is a finite tuple, e.g. (-1, 2, -1, 0, -1, -1) for 6 weapons where
weapon 1 -> target 2 and weapon 3 -> target 0 have been assigned.

This is a useful contrast to the DQN approach:
  - Q-learning (this file): overfits to and solves ONE scenario optimally
    (given enough training episodes to visit/update all relevant states).
  - DQN (train_dqn.py): generalizes across MANY random scenarios using a
    neural network, at the cost of not being provably optimal on any one.

Run:
    python3 "inference_example_q_learning.py"
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from wta_env import WeaponTargetAssignmentEnv


NUM_WEAPONS = 6
NUM_TARGETS = 4

# --------------------------------------------------------------------------
# Fixed scenario. Unlike the DQN example, this MUST stay constant across
# training episodes -- that's what makes the state space enumerable.
#
# For a REPRODUCIBLE scenario, uncomment:
# np.random.seed(7)

KILL_PROB = np.random.uniform(0.3, 0.9, size=(NUM_WEAPONS, NUM_TARGETS))
TARGET_VALUE = np.random.uniform(1.0, 10.0, size=NUM_TARGETS)
WEAPON_VALUE = np.random.uniform(1.0, 10.0, size=NUM_WEAPONS)

# To use YOUR OWN fixed numbers instead, overwrite them here, e.g.:
#
# KILL_PROB = np.array([
#     [0.8, 0.3, 0.5, 0.2],
#     [0.4, 0.7, 0.6, 0.3],
#     [0.6, 0.5, 0.9, 0.4],
#     [0.3, 0.6, 0.4, 0.8],
#     [0.7, 0.4, 0.3, 0.5],
#     [0.5, 0.5, 0.6, 0.6],
# ])
# TARGET_VALUE = np.array([2.0, 8.0, 5.0, 9.0])
# WEAPON_VALUE = np.array([10.0, 1.0, 5.0, 2.0, 7.0, 3.0])
# --------------------------------------------------------------------------


StateKey = tuple  # length NUM_WEAPONS; entry i = target assigned to weapon i, or -1


def state_key_from_assignment(assignment: list[int]) -> StateKey:
    """assignment[i] = target index weapon i is assigned to, or -1 if unassigned."""
    return tuple(assignment)


def valid_actions_for_state(state: StateKey, num_targets: int) -> list[int]:
    """Flattened action indices for every unassigned weapon x every target."""
    actions = []
    for weapon_idx, assigned_target in enumerate(state):
        if assigned_target == -1:
            for target_idx in range(num_targets):
                actions.append(weapon_idx * num_targets + target_idx)
    return actions


def train_q_learning(
    num_weapons: int = NUM_WEAPONS,
    num_targets: int = NUM_TARGETS,
    kill_prob: np.ndarray = None,
    target_value: np.ndarray = None,
    weapon_value: np.ndarray = None,
    num_episodes: int = 8000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_episodes: int = 6000,
) -> dict:
    """
    Train a tabular Q-table on ONE fixed WTA scenario using the classic
    Q-learning update rule:

        Q[s, a] <- Q[s, a] + alpha * (reward + gamma * max_a' Q[s', a'] - Q[s, a])

    Returns
    -------
    Q : dict mapping state_key (tuple) -> np.ndarray of shape
        (num_weapons * num_targets,) of learned Q-values.
        Unvisited (state, action) pairs default to 0.0 (via defaultdict).
    """
    kill_prob = KILL_PROB if kill_prob is None else kill_prob
    target_value = TARGET_VALUE if target_value is None else target_value
    weapon_value = WEAPON_VALUE if weapon_value is None else weapon_value

    env = WeaponTargetAssignmentEnv(num_weapons=num_weapons, num_targets=num_targets)
    n_actions = num_weapons * num_targets

    Q: dict[StateKey, np.ndarray] = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

    for episode in range(1, num_episodes + 1):
        env.reset(options={
            "kill_prob": kill_prob,
            "target_value": target_value,
            "weapon_value": weapon_value,
        })
        assignment = [-1] * num_weapons
        state = state_key_from_assignment(assignment)

        eps = max(eps_end, eps_start - (eps_start - eps_end) * episode / eps_decay_episodes)

        done = False
        while not done:
            valid = valid_actions_for_state(state, num_targets)
            if not valid:
                break

            if random.random() < eps:
                action = random.choice(valid)
            else:
                q_row = Q[state]
                # Restrict argmax to valid actions only.
                best = max(valid, key=lambda a: q_row[a])
                action = best

            weapon_idx, target_idx = divmod(action, num_targets)
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            assignment[weapon_idx] = target_idx
            next_state = state_key_from_assignment(assignment)

            next_valid = valid_actions_for_state(next_state, num_targets)
            next_max_q = max((Q[next_state][a] for a in next_valid), default=0.0)

            # Classic Q-learning (Bellman) update
            td_target = reward + gamma * next_max_q
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state

        if episode % 1000 == 0:
            print(f"Episode {episode:5d}/{num_episodes} | eps={eps:.3f} | "
                  f"Q-table size: {len(Q)} states visited")

    return Q


def get_best_action_q_learning(Q: dict, state: StateKey, num_targets: int) -> int:
    """Greedy action from the Q-table for the given state (no exploration)."""
    valid = valid_actions_for_state(state, num_targets)
    if not valid:
        raise ValueError("No valid actions for this state -- every weapon is already assigned.")
    q_row = Q[state]
    return max(valid, key=lambda a: q_row[a])


def print_q_table(
    Q: dict,
    num_targets: int,
    n: int = 20,
    only_valid_actions: bool = True,
    sort_by_visits: bool = True,
):
    """
    Print a readable view of the learned Q-table.

    Parameters
    ----------
    Q                  : the trained Q-table (dict: state_key -> Q-value array)
    num_targets        : needed to decode flattened actions into (weapon, target)
    n                  : how many states to show (the table can have thousands
                         of entries, so this caps the printout)
    only_valid_actions : if True, only print Q-values for actions that were
                         actually legal in that state (skips already-assigned
                         weapons, which always sit at their default 0.0)
    sort_by_visits     : if True, show states with the most non-zero (i.e.
                         actually learned/updated) Q-values first -- these
                         are the most "informative" states to look at
    """
    if len(Q) == 0:
        print("Q-table is empty -- nothing to display (train first).")
        return

    items = list(Q.items())
    if sort_by_visits:
        items.sort(key=lambda kv: np.count_nonzero(kv[1]), reverse=True)

    items = items[:n]
    print(f"Q-table: {len(Q)} states total (showing {len(items)}):")
    print("=" * 60)

    for state, q_values in items:
        state_str = ", ".join(
            f"w{i}->t{t}" if t != -1 else f"w{i}:unassigned"
            for i, t in enumerate(state)
        )
        print(f"\nState: ({state_str})")

        if only_valid_actions:
            actions_to_show = valid_actions_for_state(state, num_targets)
        else:
            actions_to_show = list(range(len(q_values)))

        if not actions_to_show:
            print("  (terminal state -- no valid actions)")
            continue

        # Sort this state's actions by Q-value, best first
        actions_to_show.sort(key=lambda a: q_values[a], reverse=True)
        for a in actions_to_show:
            weapon_idx, target_idx = divmod(a, num_targets)
            print(f"  weapon {weapon_idx} -> target {target_idx}: Q={q_values[a]:.3f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Training tabular Q-learning on a FIXED scenario "
          f"({NUM_WEAPONS} weapons, {NUM_TARGETS} targets) ...\n")
    print("Kill probability matrix (weapons x targets):")
    print(np.round(KILL_PROB, 2))
    print("Target values:", np.round(TARGET_VALUE, 2))
    print("Weapon values:", np.round(WEAPON_VALUE, 2))
    print()

    Q = train_q_learning(num_episodes=8000)

    # --- Run the learned greedy policy on the SAME fixed scenario ---
    env = WeaponTargetAssignmentEnv(num_weapons=NUM_WEAPONS, num_targets=NUM_TARGETS)
    env.reset(options={
        "kill_prob": KILL_PROB,
        "target_value": TARGET_VALUE,
        "weapon_value": WEAPON_VALUE,
    })
    assignment = [-1] * NUM_WEAPONS
    state = state_key_from_assignment(assignment)

    print("\n=== Learned Q-learning policy ===")
    step = 0
    total_reward = 0.0
    done = False
    info = env._get_info()
    while not done:
        action = get_best_action_q_learning(Q, state, NUM_TARGETS)
        weapon_idx, target_idx = divmod(action, NUM_TARGETS)
        _, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        assignment[weapon_idx] = target_idx
        state = state_key_from_assignment(assignment)

        print(f"Step {step}: assign weapon {weapon_idx} -> target {target_idx} "
              f"| reward={reward:.3f} | objective (lower=better)={info['objective_value']:.3f}")

    print(f"\nQ-learning total_reward={total_reward:.3f}, "
          f"final_objective={info['objective_value']:.3f} (lower is better)")

    # --- Compare against the pure greedy baseline on the same scenario ---
    greedy_env = WeaponTargetAssignmentEnv(num_weapons=NUM_WEAPONS, num_targets=NUM_TARGETS)
    greedy_env.reset(options={
        "kill_prob": KILL_PROB,
        "target_value": TARGET_VALUE,
        "weapon_value": WEAPON_VALUE,
    })
    greedy_result = greedy_env.greedy_solve()

    print("\n=== Greedy baseline (no lookahead) on the same scenario ===")
    for w, t, r in greedy_result["assignments"]:
        print(f"  weapon {w} -> target {t}: reward={r:.3f}")
    print(f"  total_reward={greedy_result['total_reward']}, "
          f"final_objective={greedy_result['final_objective']} (lower is better)")

    print(f"\nQ-learning final_objective={info['objective_value']:.4f} vs "
          f"Greedy final_objective={greedy_result['final_objective']} (lower is better)")

    print()
    print_q_table(Q, NUM_TARGETS, n=10)