"""
Example: using get_best_action() to pick the best action from the
current state of a WeaponTargetAssignmentEnv, using a trained DQN.

This trains a small model (or loads one from disk if you've already run
train_dqn.py and saved wta_dqn.pt), then walks through one episode picking
the greedy best action at every step.

Run:
    python3 inference_example.py
"""

import os

import numpy as np

from wta_env import WeaponTargetAssignmentEnv
from train_dqn import train, get_best_action, get_ranked_actions_per_weapon, verify_all_q_values_found, save_model, load_model


NUM_WEAPONS = 4
NUM_TARGETS = 1

# --------------------------------------------------------------------------
# Scenario setup. By default, KILL_PROB and TARGET_VALUE are randomly
# generated below, sized to match NUM_WEAPONS / NUM_TARGETS.
#
#   - KILL_PROB   : shape (NUM_WEAPONS, NUM_TARGETS), each value in [0, 1] =
#                    probability that weapon i destroys target j
#   - TARGET_VALUE: shape (NUM_TARGETS,) = how valuable/important each target is
#   - WEAPON_VALUE: shape (NUM_WEAPONS,) = each weapon's own importance /
#                    damage potential -- higher means it earns more reward
#                    credit for the same kill probability
#
# For a REPRODUCIBLE scenario (same numbers every run), uncomment:
# np.random.seed(7)

KILL_PROB = np.random.uniform(0.3, 0.9, size=(NUM_WEAPONS, NUM_TARGETS))
TARGET_VALUE = np.random.uniform(1.0, 10.0, size=NUM_TARGETS)
WEAPON_VALUE = np.random.uniform(1.0, 10.0, size=NUM_WEAPONS)

# To use YOUR OWN fixed numbers instead, just overwrite them here, e.g.:
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
# WEAPON_VALUE = np.array([10.0, 1.0, 5.0, 2.0, 7.0, 3.0])  # e.g. weapon 0 is highly capable/important
# --------------------------------------------------------------------------

# Model filename is tied to the problem size, so changing NUM_WEAPONS/NUM_TARGETS
# above always trains/loads a matching checkpoint instead of accidentally
# loading a model shaped for a different size.
MODEL_PATH = f"wta_dqn_{NUM_WEAPONS}w_{NUM_TARGETS}t.pt"

if __name__ == "__main__":

    # 1. Get a trained Q-network: load from disk if available, else train one.
    if os.path.exists(MODEL_PATH):
        print(f"Loading trained model from {MODEL_PATH} ...")
        q_net = load_model(MODEL_PATH)
    else:
        print("No saved model found, training a small one (this may take a bit) ...")
        q_net, _ = train(num_weapons=NUM_WEAPONS, num_targets=NUM_TARGETS, num_episodes=1000)
        save_model(q_net, MODEL_PATH)

    # 2. Set up the environment and load the (random or custom) scenario above
    #    as the current state.
    env = WeaponTargetAssignmentEnv(num_weapons=NUM_WEAPONS, num_targets=NUM_TARGETS)
    obs, info = env.reset(options={
        "kill_prob": KILL_PROB,
        "target_value": TARGET_VALUE,
        "weapon_value": WEAPON_VALUE,
    })

    print("\nStarting a new episode. Kill probability matrix (weapons x targets):")
    print(env.kill_prob.round(2))
    print("Target values:", env.target_value.round(2))
    print("Weapon values:", env.weapon_value.round(2))

    # 3. Repeatedly ask the model for the best action given the current state.
    step = 0
    done = False
    total_reward = 0.0
    while not done:
        action_mask = info["action_mask"]

        # Confirm the network produced a finite Q-value for EVERY one of
        # num_weapons * num_targets actions before trusting its argmax --
        # i.e. make sure all Q values are actually found, nothing missing.
        q_check = verify_all_q_values_found(q_net, obs, env.num_weapons, env.num_targets)
        if not q_check["complete"]:
            print(f"  [WARNING] Step {step + 1}: Q-value grid incomplete! "
                  f"expected {q_check['expected_count']}, got {q_check['actual_count']}, "
                  f"bad indices: {q_check['missing_or_bad_indices']}")
        else:
            print(f"--- Step {step + 1}: verified all {q_check['expected_count']} Q-values present and finite ---")

        # Ground-truth reward the env would give for EVERY possible action
        # right now (not a network estimate -- computed from the reward formula).
        all_rewards = env.get_all_action_rewards()
        print(f"\n--- Step {step + 1}: all action rewards (weapons x targets) ---")
        print(np.round(all_rewards, 2))

        # Same rewards, but sorted best-first as a flat ranked list.
        top_rewards = env.get_ranked_action_rewards(top_k=5)
        print(f"--- Step {step + 1}: top actions by actual reward ---")
        for w, t, r in top_rewards:
            print(f"  weapon {w} -> target {t}: reward={r:.2f}")

        # Same, but excluding targets that already have a weapon assigned to
        # them -- i.e. results where no target repeats.
        top_unique = env.get_ranked_action_rewards(top_k=5, unique_targets=True)
        print(f"--- Step {step + 1}: top actions, no repeated targets ---")
        if top_unique:
            for w, t, r in top_unique:
                print(f"  weapon {w} -> target {t}: reward={r:.2f}")
        else:
            print("  (every target already has a weapon assigned)")

        # For each unassigned weapon, show its best / 2nd-best / 3rd-best
        # target choice by Q-value (not just the single greedy pick).
        ranked = get_ranked_actions_per_weapon(q_net, obs, action_mask, env.num_targets, top_k=3)
        print(f"--- Step {step + 1} Q-value rankings ---")
        for weapon_idx, choices in ranked.items():
            choice_str = ", ".join(
                f"#{rank+1} target {t} (Q={q:.2f})" for rank, (t, q) in enumerate(choices)
            )
            print(f"  Weapon {weapon_idx}: {choice_str}")

        # <<< This is the core call: get the best action from the current state >>>
        action = get_best_action(q_net, obs, action_mask)

        weapon_idx, target_idx = divmod(action, env.num_targets)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        print(f"Step {step}: assign weapon {weapon_idx} -> target {target_idx} "
              f"| reward={reward:.3f} | objective (lower=better)={info['objective_value']:.3f}")

    print(f"\nEpisode finished in {step} steps. "
          f"Total expected damage dealt: {total_reward:.3f}")
    print(f"Final expected surviving target value: {info['objective_value']:.3f}")
    print(f"Total weapon value used: {info['total_weapon_value_used']:.3f}")

    # 4. Compare against a pure greedy baseline on the EXACT same scenario --
    #    greedy always takes the single locally-best action (no lookahead),
    #    while the DQN can learn to hold weapons back for a better long-term
    #    outcome. Lower final_objective is better.
    greedy_env = WeaponTargetAssignmentEnv(num_weapons=NUM_WEAPONS, num_targets=NUM_TARGETS)
    greedy_env.reset(options={
        "kill_prob": KILL_PROB,
        "target_value": TARGET_VALUE,
        "weapon_value": WEAPON_VALUE,
    })
    greedy_result = greedy_env.greedy_solve()

    print("\n=== Greedy baseline on the same scenario (no lookahead) ===")
    for w, t, r in greedy_result["assignments"]:
        print(f"  weapon {w} -> target {t}: reward={r:.3f}")
    print(f"  total_reward={greedy_result['total_reward']}, "
          f"final_objective={greedy_result['final_objective']} (lower is better)")
    print(f"\nDQN final_objective={info['objective_value']:.4f} vs "
          f"Greedy final_objective={greedy_result['final_objective']} (lower is better)")
