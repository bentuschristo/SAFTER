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


class VectorReplayBuffer:
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
    ) -> None:
        n = int(obs.shape[0])
        if n > self.capacity:
            start = n - self.capacity
            obs, action, reward, next_obs, done = (
                obs[start:], action[start:], reward[start:], next_obs[start:], done[start:]
            )
            n = self.capacity

        idx = (torch.arange(n, device=self.device) + self.write_index) % self.capacity
        self.obs[idx] = obs
        self.action[idx] = action
        self.reward[idx, 0] = reward
        self.next_obs[idx] = next_obs
        self.done[idx, 0] = done.float()
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
        )

    def state_dict(self) -> dict[str, object]:
        """Return only initialized replay entries, stored on CPU for portability."""
        return {
            "capacity": self.capacity, "size": self.size, "write_index": self.write_index,
            "obs": self.obs[:self.size].detach().cpu(),
            "action": self.action[:self.size].detach().cpu(),
            "reward": self.reward[:self.size].detach().cpu(),
            "next_obs": self.next_obs[:self.size].detach().cpu(),
            "done": self.done[:self.size].detach().cpu(),
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, object]) -> None:
        saved_capacity = int(state.get("capacity", self.capacity))
        if saved_capacity != self.capacity:
            raise ValueError(f"Replay capacity mismatch: saved={saved_capacity}, current={self.capacity}")
        size = int(state["size"])
        if size > self.capacity:
            raise ValueError(f"Replay size {size} exceeds current capacity {self.capacity}")
        for name in ("obs", "action", "reward", "next_obs", "done"):
            source = state[name]
            target = getattr(self, name)
            if source.shape[1:] != target.shape[1:]:
                raise ValueError(f"Replay {name} shape mismatch: {source.shape} vs {target.shape}")
            target[:size].copy_(source[:size].to(self.device))
        self.size = size
        self.write_index = int(state.get("write_index", size % self.capacity)) % self.capacity
