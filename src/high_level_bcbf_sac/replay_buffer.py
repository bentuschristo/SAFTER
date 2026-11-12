from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    position: torch.Tensor
    yaw: torch.Tensor
    next_position: torch.Tensor
    next_yaw: torch.Tensor


class VectorReplayBuffer:
    """GPU replay buffer storing the context needed by the analytical CBF layer."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device | str) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs = torch.empty((capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.action = torch.empty((capacity, action_dim), dtype=torch.float32, device=self.device)
        self.reward = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.next_obs = torch.empty((capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.done = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.position = torch.empty((capacity, 2), dtype=torch.float32, device=self.device)
        self.yaw = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.next_position = torch.empty((capacity, 2), dtype=torch.float32, device=self.device)
        self.next_yaw = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.write_index = 0
        self.size = 0

    @torch.no_grad()
    def add_batch(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        position: torch.Tensor,
        yaw: torch.Tensor,
        next_position: torch.Tensor,
        next_yaw: torch.Tensor,
    ) -> None:
        n = int(obs.shape[0])
        if n > self.capacity:
            start = n - self.capacity
            obs, action, reward, next_obs, done = (
                obs[start:], action[start:], reward[start:], next_obs[start:], done[start:]
            )
            position, yaw = position[start:], yaw[start:]
            next_position, next_yaw = next_position[start:], next_yaw[start:]
            n = self.capacity

        idx = (torch.arange(n, device=self.device) + self.write_index) % self.capacity
        self.obs[idx] = obs
        self.action[idx] = action
        self.reward[idx, 0] = reward
        self.next_obs[idx] = next_obs
        self.done[idx, 0] = done.float()
        self.position[idx] = position
        self.yaw[idx, 0] = yaw
        self.next_position[idx] = next_position
        self.next_yaw[idx, 0] = next_yaw
        self.write_index = (self.write_index + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.size < batch_size:
            raise RuntimeError(f"Replay contains {self.size} samples, need {batch_size}.")
        idx = torch.randint(self.size, (batch_size,), device=self.device)
        return ReplayBatch(
            obs=self.obs[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            next_obs=self.next_obs[idx],
            done=self.done[idx],
            position=self.position[idx],
            yaw=self.yaw[idx, 0],
            next_position=self.next_position[idx],
            next_yaw=self.next_yaw[idx, 0],
        )
