"""
Run a full episode by always picking the action with the highest actual
immediate reward (greedy on get_all_action_rewards / get_ranked_action_rewards),
stepping the environment repeatedly until the reward grid is fully exhausted
-- i.e. every entry is nan because every weapon has been assigned.

This does NOT require a trained model; it uses the environment's own
ground-truth reward formula directly.

Run:
    python3 assign_until_complete.py
"""

import numpy as np

from wta_env import WeaponTargetAssignmentEnv

NUM_WEAPONS = 4
NUM_TARGETS = 1

if __name__ == "__main__":
    env = WeaponTargetAssignmentEnv(num_weapons=NUM_WEAPONS, num_targets=NUM_TARGETS)
    obs, info = env.reset(seed=1)

    print("Kill probability matrix (weapons x targets):")
    print(np.round(env.kill_prob, 2))
    print("Target values:", np.round(env.target_value, 2))

    step = 0
    while True:
        rewards = env.get_all_action_rewards()

        if np.all(np.isnan(rewards)):
            print("\nAll entries in get_all_action_rewards() are nan "
                  "-> every weapon has been assigned. Stopping.")
            break

        # Greedy: take the single best (weapon, target) pair by actual reward.
        # unique_targets=False so remaining weapons can still double up on a
        # target once every target already has one (only matters if
        # num_weapons > num_targets).
        best = env.get_ranked_action_rewards(top_k=1, unique_targets=False)
        weapon_idx, target_idx, reward = best[0]

        action = weapon_idx * env.num_targets + target_idx
        obs, actual_reward, terminated, truncated, info = env.step(action)
        step += 1

        print(f"Step {step}: weapon {weapon_idx} -> target {target_idx} "
              f"| reward={actual_reward:.3f} "
              f"| objective (lower=better)={info['objective_value']:.3f}")

        if terminated or truncated:
            print("\nEpisode terminated/truncated by the environment.")
            break

    print("\nFinal get_all_action_rewards() grid (every entry should be nan):")
    print(env.get_all_action_rewards())
    print(f"\nFinal expected surviving target value: {info['objective_value']:.3f}")
