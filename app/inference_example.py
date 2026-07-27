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
from train_dqn import train, get_best_action, get_ranked_actions_per_weapon, save_model, load_model


NUM_WEAPONS = 2
NUM_TARGETS = 3

# --------------------------------------------------------------------------
# Scenario setup. By default, KILL_PROB and TARGET_VALUE are randomly
# generated below, sized to match NUM_WEAPONS / NUM_TARGETS.
#
#   - KILL_PROB   : shape (NUM_WEAPONS, NUM_TARGETS), each value in [0, 1] =
#                    probability that weapon i destroys target j
#   - TARGET_VALUE: shape (NUM_TARGETS,) = how valuable/important each target is
#
# For a REPRODUCIBLE scenario (same numbers every run), uncomment:
# np.random.seed(7)

KILL_PROB = np.random.uniform(0.3, 0.9, size=(NUM_WEAPONS, NUM_TARGETS))
TARGET_VALUE = np.random.uniform(1.0, 10.0, size=NUM_TARGETS)

# To use YOUR OWN fixed numbers instead, just overwrite them here, e.g.:
#     T1 T2 T3 ...
# W1
# W2
# W3
# ...

# KILL_PROB = np.array([
#     [0.8, 0.3, 0.5, 0.2],
#     [0.4, 0.7, 0.6, 0.3],
#     [0.6, 0.5, 0.9, 0.4],
#     [0.3, 0.6, 0.4, 0.8],
#     [0.7, 0.4, 0.3, 0.5],
#     [0.5, 0.5, 0.6, 0.6],
# ])
# TARGET_VALUE = np.array([2.0, 8.0, 5.0, 9.0])

KILL_PROB = np.array([
    [0.8, 0.3, 0.5],
    [0.4, 0.7, 0.6]
])
TARGET_VALUE = np.array([1.0, 1.0, 1.0])

# --------------------------------------------------------------------------

# Model filename is tied to the problem size, so changing NUM_WEAPONS/NUM_TARGETS
# above always trains/loads a matching checkpoint instead of accidentally
# loading a model shaped for a different size.
MODEL_PATH = f"wta_dqn_{NUM_WEAPONS}w_{NUM_TARGETS}t.pt"

if __name__ == "__main__":
    assert KILL_PROB.shape == (NUM_WEAPONS, NUM_TARGETS), (
        f"KILL_PROB shape {KILL_PROB.shape} must match "
        f"(NUM_WEAPONS, NUM_TARGETS) = ({NUM_WEAPONS}, {NUM_TARGETS})"
    )
    assert TARGET_VALUE.shape == (NUM_TARGETS,), (
        f"TARGET_VALUE shape {TARGET_VALUE.shape} must match (NUM_TARGETS,) = ({NUM_TARGETS},)"
    )

    # 1. Get a trained Q-network: load from disk if available, else train one.
    if os.path.exists(MODEL_PATH):
        print(f"Loading trained model from {MODEL_PATH} ...")
        q_net = load_model(MODEL_PATH)
    else:
        print("No saved model found, training a small one (this may take a bit) ...")
        q_net, _ = train(num_weapons=NUM_WEAPONS, num_targets=NUM_TARGETS, num_episodes=999)
        save_model(q_net, MODEL_PATH)

    # 2. Set up the environment and load the (random or custom) scenario above
    #    as the current state.
    env = WeaponTargetAssignmentEnv(num_weapons=NUM_WEAPONS, num_targets=NUM_TARGETS)
    obs, info = env.reset(options={"kill_prob": KILL_PROB, "target_value": TARGET_VALUE})

    print("\nStarting a new episode. Kill probability matrix (weapons x targets):")
    print(env.kill_prob.round(2))
    print("Target values:", env.target_value.round(2))

    # 3. Repeatedly ask the model for the best action given the current state.
    step = 0
    done = False
    total_reward = 0.0
    while not done:
        action_mask = info["action_mask"]

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

