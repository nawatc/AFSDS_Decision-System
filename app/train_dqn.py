"""
DQN training script for the Weapon-Target Assignment environment.

A minimal, dependency-light DQN (PyTorch) with:
  - Q-network (MLP)
  - Replay buffer
  - Target network with periodic sync
  - Epsilon-greedy exploration
  - Action masking (invalid weapon re-assignment is masked out at
    selection time, in addition to being penalized by the environment)

Run:
    python3 train_dqn.py
"""

from __future__ import annotations

import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from wta_env import WeaponTargetAssignmentEnv

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done", "next_mask"])


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def masked_argmax(q_values: torch.Tensor, mask: np.ndarray) -> int:
    """Pick the highest-Q valid action given a boolean validity mask."""
    q = q_values.clone()
    invalid = torch.tensor(~mask, dtype=torch.bool)
    q[invalid] = -1e9
    return int(torch.argmax(q).item())


def get_best_action(
    q_net: QNetwork,
    obs: np.ndarray,
    action_mask: np.ndarray,
    device: str | None = None,
) -> int:
    """
    Given the current environment state, return the best (greedy, no
    exploration) action according to a trained Q-network.

    Parameters
    ----------
    q_net        : a trained QNetwork
    obs          : the current observation, as returned by env.reset()/env.step()
    action_mask  : boolean array from info["action_mask"], True = valid action
    device       : "cpu" or "cuda"; inferred from q_net if not given

    Returns
    -------
    action : int, the flattened (weapon, target) action index to take next.
             Decode it back to (weapon_idx, target_idx) with
             divmod(action, env.num_targets).
    """
    device = device or next(q_net.parameters()).device
    q_net.eval()
    with torch.no_grad():
        state_t = torch.tensor(obs, dtype=torch.float32, device=device)
        q_values = q_net(state_t).cpu()
    action = masked_argmax(q_values, action_mask)
    return action


def get_ranked_actions_per_weapon(
    q_net: QNetwork,
    obs: np.ndarray,
    action_mask: np.ndarray,
    num_targets: int,
    top_k: int = 3,
    device: str | None = None,
) -> dict[int, list[tuple[int, float]]]:
    """
    For each still-unassigned weapon, rank ALL of its possible targets by
    Q-value and return the top `top_k` choices (1st best, 2nd best, 3rd
    best, ...) instead of only the single greedy action.

    Parameters
    ----------
    q_net        : a trained QNetwork
    obs          : the current observation, as returned by env.reset()/env.step()
    action_mask  : boolean array from info["action_mask"], True = valid action
    num_targets  : env.num_targets, needed to reshape the flat Q-values back
                   into a (num_weapons, num_targets) grid
    top_k        : how many ranked choices to return per weapon (default 3)
    device       : "cpu" or "cuda"; inferred from q_net if not given

    Returns
    -------
    dict mapping weapon_idx -> list of (target_idx, q_value) tuples,
    sorted best-first. Only includes weapons that are still unassigned.
    Example: {0: [(2, 4.81), (0, 3.02), (1, 1.55)], ...}
    """
    device = device or next(q_net.parameters()).device
    q_net.eval()
    with torch.no_grad():
        state_t = torch.tensor(obs, dtype=torch.float32, device=device)
        q_values = q_net(state_t).cpu().numpy()

    num_weapons = len(q_values) // num_targets
    q_grid = q_values.reshape(num_weapons, num_targets)
    mask_grid = np.asarray(action_mask).reshape(num_weapons, num_targets)

    results: dict[int, list[tuple[int, float]]] = {}
    for weapon_idx in range(num_weapons):
        if not mask_grid[weapon_idx].any():
            continue  # this weapon is already assigned -> skip it
        q_row = q_grid[weapon_idx]
        k = min(top_k, num_targets)
        ranked_target_idxs = np.argsort(-q_row)[:k]
        results[weapon_idx] = [(int(t), float(q_row[t])) for t in ranked_target_idxs]

    return results


def save_model(q_net: QNetwork, path: str = "wta_dqn.pt"):
    torch.save(
        {
            "state_dict": q_net.state_dict(),
            "obs_dim": int(q_net.net[0].in_features),
            "n_actions": int(q_net.net[-1].out_features),
        },
        path,
    )


def load_model(path: str = "wta_dqn.pt", device: str | None = None) -> QNetwork:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    q_net = QNetwork(checkpoint["obs_dim"], checkpoint["n_actions"]).to(device)
    q_net.load_state_dict(checkpoint["state_dict"])
    q_net.eval()
    return q_net


def train(
    num_weapons: int = 6,
    num_targets: int = 4,
    num_episodes: int = 800,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_episodes: int = 600,
    target_sync_every: int = 10,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    env = WeaponTargetAssignmentEnv(num_weapons=num_weapons, num_targets=num_targets)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = QNetwork(obs_dim, n_actions).to(device)
    target_net = QNetwork(obs_dim, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer()

    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        mask = info["action_mask"]
        eps = max(eps_end, eps_start - (eps_start - eps_end) * episode / eps_decay_episodes)

        done = False
        ep_reward = 0.0

        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32, device=device)

            if random.random() < eps:
                action = int(np.random.choice(np.where(mask)[0]))
            else:
                with torch.no_grad():
                    q_values = q_net(state_t)
                action = masked_argmax(q_values.cpu(), mask)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_mask = info["action_mask"]
            done = terminated or truncated

            buffer.push(obs, action, reward, next_obs, done, next_mask)
            obs = next_obs
            mask = next_mask
            ep_reward += reward

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states = torch.tensor(np.array([b.state for b in batch]), dtype=torch.float32, device=device)
                actions = torch.tensor([b.action for b in batch], dtype=torch.int64, device=device)
                rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=device)
                next_states = torch.tensor(np.array([b.next_state for b in batch]), dtype=torch.float32, device=device)
                dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=device)
                next_masks = np.array([b.next_mask for b in batch])

                q_pred = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(next_states)  # (B, n_actions)
                    next_q_masked = next_q.clone()
                    next_q_masked[torch.tensor(~next_masks, dtype=torch.bool)] = -1e9
                    next_q_max = next_q_masked.max(dim=1).values
                    q_target = rewards + gamma * next_q_max * (1.0 - dones)

                loss = nn.functional.mse_loss(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        episode_rewards.append(ep_reward)

        if episode % target_sync_every == 0:
            target_net.load_state_dict(q_net.state_dict())

        if episode % 50 == 0:
            avg_recent = np.mean(episode_rewards[-50:])
            print(f"Episode {episode:4d} | eps={eps:.3f} | avg reward (last 50)={avg_recent:.3f}")

    return q_net, episode_rewards


def evaluate(q_net: QNetwork, num_weapons=6, num_targets=4, episodes=20, device: str | None = None):
    """Compare the trained policy's objective value against a random baseline."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = WeaponTargetAssignmentEnv(num_weapons=num_weapons, num_targets=num_targets)

    policy_objs, random_objs = [], []

    for ep in range(episodes):
        # Same problem instance for both policies (fair comparison)
        obs, info = env.reset(seed=1000 + ep)
        instance_kill_prob = env.kill_prob.copy()
        instance_target_value = env.target_value.copy()

        # Trained policy
        mask = info["action_mask"]
        done = False
        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                q_values = q_net(state_t)
            action = masked_argmax(q_values.cpu(), mask)
            obs, reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            done = terminated or truncated
        policy_objs.append(info["objective_value"])

        # Random baseline on the SAME instance
        obs, info = env.reset(options={"kill_prob": instance_kill_prob, "target_value": instance_target_value})
        done = False
        while not done:
            valid = env.valid_actions()
            action = int(np.random.choice(valid))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        random_objs.append(info["objective_value"])

    print(f"\nEvaluation over {episodes} instances (lower objective = better, "
          f"objective = expected total surviving target value):")
    print(f"  Trained DQN policy : mean = {np.mean(policy_objs):.3f}")
    print(f"  Random baseline    : mean = {np.mean(random_objs):.3f}")


if __name__ == "__main__":
    trained_q_net, rewards = train(num_episodes=800)
    evaluate(trained_q_net)
    save_model(trained_q_net, "wta_dqn.pt")
    print("\nSaved trained model to wta_dqn.pt")
