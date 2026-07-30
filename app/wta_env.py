"""
Weapon-Target Assignment (WTA) Environment
===========================================

A Gymnasium-compatible reinforcement-learning environment for the classic
Weapon-Target Assignment optimization problem, intended for training a DQN
(or any discrete-action) agent.

Problem definition
-------------------
- There are `num_weapons` weapons and `num_targets` targets.
- Each weapon `i` has a probability `p[i, j]` of destroying target `j` if
  assigned to it (a "kill probability").
- Each target `j` has a value `v[j]` (its importance / damage potential).
- Each weapon `i` also has a value `w[i]` (ITS importance / damage
  potential -- e.g. a more capable/powerful weapon contributes
  proportionally more expected damage credit when it successfully hits a
  target, mirroring target_value but on the weapon side).
- Assigning multiple weapons to the same target is allowed; kill events are
  assumed independent, so the probability that target `j` survives after a
  set S of weapons is assigned to it is:

      survive(j) = prod_{i in S} (1 - p[i, j])

- The classic WTA objective is to choose an assignment of weapons to targets
  that MINIMIZES the total expected surviving value:

      J = sum_j  v[j] * survive(j)

  This is NP-hard in general, which is exactly why a learned (DQN) policy
  is an interesting approach for it.

Environment framing for RL
---------------------------
- One episode = one full assignment problem instance.
- At each step, the agent picks a (weapon, target) pair to assign one
  still-unassigned weapon to one target. Action space size is
  `num_weapons * num_targets` (flattened) so it works with a standard
  Discrete-action DQN.
- Reward at each step = decrease in expected surviving value caused by
  that assignment, scaled by that weapon's own value/damage potential
  (weapon_value[i]) -- so a more capable weapon earns proportionally more
  credit for the same kill probability, and the agent learns to match
  high-value weapons to high-payoff assignments.
- Choosing an already-assigned weapon is an invalid action: it is penalized
  and does not change state (helps the agent learn action masking even
  without an explicit mask, though a mask is also exposed via `info`).
- Episode ends when every weapon has been assigned.

Observation
-----------
A flat float32 vector containing:
  - weapon_assigned_mask   (num_weapons,)   1.0 if weapon already used
  - kill_prob_matrix       (num_weapons*num_targets,)  flattened p[i,j]
  - target_survival_prob   (num_targets,)   current survive(j) for each target
  - target_value_norm      (num_targets,)   v[j] normalized to [0,1]
  - weapon_value_norm      (num_weapons,)   w[i] normalized to [0,1]

Usage
-----
    from wta_env import WeaponTargetAssignmentEnv

    env = WeaponTargetAssignmentEnv(num_weapons=6, num_targets=4)
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        action = env.action_space.sample()          # replace with agent policy
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class WeaponTargetAssignmentEnv(gym.Env):
    """Gymnasium environment for the Weapon-Target Assignment problem."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_weapons: int = 6,
        num_targets: int = 4,
        kill_prob_range: tuple[float, float] = (0.3, 0.9),
        target_value_range: tuple[float, float] = (1.0, 10.0),
        weapon_value_range: tuple[float, float] = (1.0, 10.0),
        invalid_action_penalty: float = 1.0,
        max_invalid_actions: int = 10,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.num_weapons = num_weapons
        self.num_targets = num_targets
        self.kill_prob_range = kill_prob_range
        self.target_value_range = target_value_range
        self.weapon_value_range = weapon_value_range
        self.invalid_action_penalty = invalid_action_penalty
        self.max_invalid_actions = max_invalid_actions
        self.render_mode = render_mode

        # Action: flattened (weapon_idx, target_idx) pair
        self.action_space = spaces.Discrete(self.num_weapons * self.num_targets)

        obs_dim = (
            self.num_weapons                     # assigned mask
            + self.num_weapons * self.num_targets  # kill prob matrix
            + self.num_targets                   # current survival prob
            + self.num_targets                   # normalized target value
            + self.num_weapons                   # normalized weapon value
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Populated in reset()
        self.kill_prob: np.ndarray | None = None
        self.target_value: np.ndarray | None = None
        self.weapon_value: np.ndarray | None = None
        self.assigned_mask: np.ndarray | None = None
        self.survival_prob: np.ndarray | None = None
        self._invalid_count = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Core Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}

        # Allow injecting a specific problem instance (useful for eval/benchmarks)
        if "kill_prob" in options and "target_value" in options:
            self.kill_prob = np.asarray(options["kill_prob"], dtype=np.float32)
            self.target_value = np.asarray(options["target_value"], dtype=np.float32)
            self.num_weapons, self.num_targets = self.kill_prob.shape

            # weapon_value is optional even in a custom scenario -- default
            # to all-ones (no importance difference between weapons) if not given.
            if "weapon_value" in options:
                self.weapon_value = np.asarray(options["weapon_value"], dtype=np.float32)
            else:
                self.weapon_value = np.ones(self.num_weapons, dtype=np.float32)

            # Guard against bad inputs -- these are the #1 source of nan
            # propagating into rewards, Q-values, and eventually training.
            if np.isnan(self.kill_prob).any():
                raise ValueError("kill_prob contains nan values; every entry must be a real "
                                  "number in [0, 1]. Check the scenario you supplied.")
            if np.isnan(self.target_value).any():
                raise ValueError("target_value contains nan values; every entry must be a "
                                  "real, finite number. Check the scenario you supplied.")
            if np.isnan(self.weapon_value).any():
                raise ValueError("weapon_value contains nan values; every entry must be a "
                                  "real, finite number. Check the scenario you supplied.")
            if (self.kill_prob < 0).any() or (self.kill_prob > 1).any():
                raise ValueError("kill_prob must contain only values in [0, 1].")
            if (self.target_value < 0).any():
                raise ValueError("target_value must contain only non-negative values.")
            if (self.weapon_value < 0).any():
                raise ValueError("weapon_value must contain only non-negative values.")
            if self.weapon_value.shape != (self.num_weapons,):
                raise ValueError(
                    f"weapon_value shape {self.weapon_value.shape} must match "
                    f"(num_weapons,) = ({self.num_weapons},)"
                )
        else:
            lo, hi = self.kill_prob_range
            self.kill_prob = self._rng.uniform(
                lo, hi, size=(self.num_weapons, self.num_targets)
            ).astype(np.float32)

            vlo, vhi = self.target_value_range
            self.target_value = self._rng.uniform(
                vlo, vhi, size=self.num_targets
            ).astype(np.float32)

            wlo, whi = self.weapon_value_range
            self.weapon_value = self._rng.uniform(
                wlo, whi, size=self.num_weapons
            ).astype(np.float32)

        self.assigned_mask = np.zeros(self.num_weapons, dtype=np.float32)
        self.survival_prob = np.ones(self.num_targets, dtype=np.float32)
        self._invalid_count = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        weapon_idx, target_idx = divmod(int(action), self.num_targets)

        terminated = False
        truncated = False

        if self.assigned_mask[weapon_idx] == 1.0:
            # Invalid: weapon already used. Small penalty, state unchanged.
            reward = -self.invalid_action_penalty
            self._invalid_count += 1
            if self._invalid_count >= self.max_invalid_actions:
                truncated = True
        else:
            # Expected surviving value of this target BEFORE the assignment
            prev_expected_survival = self.target_value[target_idx] * self.survival_prob[target_idx]

            # Apply the assignment: target's survival probability shrinks
            p_kill = self.kill_prob[weapon_idx, target_idx]
            self.survival_prob[target_idx] *= (1.0 - p_kill)
            self.assigned_mask[weapon_idx] = 1.0

            new_expected_survival = self.target_value[target_idx] * self.survival_prob[target_idx]

            # Reward = expected damage dealt (reduction in surviving value),
            # scaled by this weapon's own value/damage potential. A more
            # capable/valuable weapon earns proportionally more credit for
            # achieving the same kill probability -- mirroring how
            # target_value scales the importance of the target being hit.
            expected_damage = prev_expected_survival - new_expected_survival
            reward = float(expected_damage * self.weapon_value[weapon_idx])

            if self.assigned_mask.sum() == self.num_weapons:
                terminated = True

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        print("Assigned mask:", self.assigned_mask)
        print("Survival prob:", np.round(self.survival_prob, 3))
        print("Target values:", np.round(self.target_value, 2))
        print("Weapon values:", np.round(self.weapon_value, 2))
        print("Expected surviving value (objective, lower is better):",
              round(float(np.sum(self.target_value * self.survival_prob)), 3))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        value_norm = self.target_value / max(self.target_value.max(), 1e-8)
        weapon_value_norm = self.weapon_value / max(self.weapon_value.max(), 1e-8)
        obs = np.concatenate(
            [
                self.assigned_mask,
                self.kill_prob.flatten(),
                self.survival_prob,
                value_norm,
                weapon_value_norm,
            ]
        ).astype(np.float32)
        return obs

    def _get_info(self) -> dict:
        # Boolean mask over the flattened action space: True = valid action
        action_mask = np.repeat(1.0 - self.assigned_mask, self.num_targets).astype(bool)
        return {
            "action_mask": action_mask,
            "objective_value": float(np.sum(self.target_value * self.survival_prob)),
            "weapons_remaining": int(self.num_weapons - self.assigned_mask.sum()),
            "total_weapon_value_used": float(np.sum(self.assigned_mask * self.weapon_value)),
        }

    def valid_actions(self) -> np.ndarray:
        """Return the array of currently-valid flattened action indices."""
        unassigned_weapons = np.where(self.assigned_mask == 0.0)[0]
        actions = [
            w * self.num_targets + t
            for w in unassigned_weapons
            for t in range(self.num_targets)
        ]
        return np.array(actions, dtype=np.int64)

    def get_all_action_rewards(self) -> np.ndarray:
        """
        Compute the ACTUAL reward the environment would return for every
        possible (weapon, target) action from the CURRENT state -- without
        taking any action or mutating state. This is ground truth from the
        reward formula itself, unlike a network's Q-value estimates.

        Returns
        -------
        rewards : np.ndarray of shape (num_weapons, num_targets)
                  rewards[i, j] = the reward env.step() would return right
                  now if weapon i were assigned to target j.
                  Already-assigned weapons get np.nan (invalid action, so
                  there's no meaningful "reward" for reusing them -- the
                  real env.step() would instead return -invalid_action_penalty).
        """
        # reward(i, j) = weapon_value[i] * value[j] * survival[j] * kill_prob[i, j]
        # (expected damage, scaled by this weapon's own importance/damage potential)
        expected_damage = self.target_value[np.newaxis, :] * self.survival_prob[np.newaxis, :] * self.kill_prob
        rewards = (expected_damage * self.weapon_value[:, np.newaxis]).astype(np.float32)
        rewards[self.assigned_mask == 1.0, :] = np.nan
        return rewards

    def get_ranked_action_rewards(
        self, top_k: int | None = None, unique_targets: bool = False
    ) -> list[tuple[int, int, float]]:
        """
        Sort every possible (weapon, target) action by its actual immediate
        reward, best first. Convenience wrapper around get_all_action_rewards()
        for when you just want a ranked list instead of the raw grid.

        Parameters
        ----------
        top_k          : if given, only return the top_k highest-reward actions.
                          If None, return all valid actions ranked.
        unique_targets : if True, exclude any target that has ALREADY received
                          a weapon (survival_prob < 1.0), so the ranking never
                          suggests assigning a second weapon to the same target
                          -- i.e. a strict one-weapon-per-target result.
                          If every target has already been used, no actions
                          remain and an empty list is returned.

        Returns
        -------
        List of (weapon_idx, target_idx, reward) tuples, sorted descending
        by reward. Already-assigned weapons are always excluded (nan reward).
        """
        rewards = self.get_all_action_rewards()
        if unique_targets:
            already_used_targets = self.survival_prob < 1.0
            rewards = rewards.copy()
            rewards[:, already_used_targets] = np.nan

        flat = rewards.flatten()
        valid_idx = np.where(~np.isnan(flat))[0]  # drop nan (already-assigned weapons / used targets)

        order = valid_idx[np.argsort(-flat[valid_idx])]  # sort valid entries, best first
        if top_k is not None:
            order = order[:top_k]

        ranked = [
            (int(idx // self.num_targets), int(idx % self.num_targets), float(flat[idx]))
            for idx in order
        ]
        return ranked

    def greedy_solve(self) -> dict:
        """
        Solve the environment's CURRENT state to completion using a pure
        greedy heuristic: at every remaining step, exhaustively evaluate
        every (weapon, target) pair's actual reward (get_ranked_action_rewards)
        and take the single best one. No lookahead, no network involved --
        this is a ground-truth baseline for comparing a trained policy against.

        NOTE: this mutates the environment -- it steps it all the way to
        the end, exactly like manually calling env.step() in a loop. Call
        env.reset() again afterward if you need a fresh episode.

        Returns
        -------
        dict with:
          - assignments    : list of (weapon_idx, target_idx, reward) in the
                             order they were taken
          - total_reward   : sum of all rewards collected
          - final_objective: objective_value after the last step (lower = better)
          - steps_taken    : number of assignments made
        """
        assignments = []
        total_reward = 0.0
        info = self._get_info()

        while True:
            ranked = self.get_ranked_action_rewards(top_k=1, unique_targets=False)
            if not ranked:
                break  # no weapons left to assign

            weapon_idx, target_idx, _ = ranked[0]
            action = weapon_idx * self.num_targets + target_idx
            obs, reward, terminated, truncated, info = self.step(action)

            assignments.append((weapon_idx, target_idx, float(reward)))
            total_reward += reward

            if terminated or truncated:
                break

        return {
            "assignments": assignments,
            "total_reward": round(total_reward, 4),
            "final_objective": round(info["objective_value"], 4),
            "steps_taken": len(assignments),
        }


if __name__ == "__main__":
    # Quick smoke test with a random policy
    env = WeaponTargetAssignmentEnv(num_weapons=5, num_targets=4)
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    done = False
    while not done:
        valid = env.valid_actions()
        action = int(np.random.choice(valid))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    env.render()
    print("Episode total reward (total expected damage dealt):", round(total_reward, 3))
