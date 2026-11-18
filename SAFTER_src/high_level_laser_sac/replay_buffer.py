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
    # Populated only for the critic_sees_nominal_action ablation.
    nominal_action: torch.Tensor | None = None


class VectorReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device | str,
        nominal_action_dim: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.nominal_action_dim = int(nominal_action_dim)
        self.obs = torch.empty((capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.action = torch.empty((capacity, action_dim), dtype=torch.float32, device=self.device)
        self.reward = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.next_obs = torch.empty((capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.done = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.nominal_action = (
            torch.zeros((capacity, self.nominal_action_dim), dtype=torch.float32, device=self.device)
            if self.nominal_action_dim > 0
            else None
        )
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
        nominal_action: torch.Tensor | None = None,
    ) -> None:
        if (nominal_action is not None) != (self.nominal_action is not None):
            raise ValueError(
                "add_batch's nominal_action must be given iff this buffer was "
                "constructed with nominal_action_dim > 0"
            )
        n = int(obs.shape[0])
        if n > self.capacity:
            start = n - self.capacity
            obs, action, reward, next_obs, done = (
                obs[start:], action[start:], reward[start:], next_obs[start:], done[start:]
            )
            if nominal_action is not None:
                nominal_action = nominal_action[start:]
            n = self.capacity

        idx = (torch.arange(n, device=self.device) + self.write_index) % self.capacity
        self.obs[idx] = obs
        self.action[idx] = action
        self.reward[idx, 0] = reward
        self.next_obs[idx] = next_obs
        self.done[idx, 0] = done.float()
        if nominal_action is not None:
            self.nominal_action[idx] = nominal_action
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
            nominal_action=self.nominal_action[idx] if self.nominal_action is not None else None,
        )

    def state_dict(self) -> dict[str, object]:
        """Return only initialized replay entries, stored on CPU for portability."""
        state = {
            "capacity": self.capacity, "size": self.size, "write_index": self.write_index,
            "obs": self.obs[:self.size].detach().cpu(),
            "action": self.action[:self.size].detach().cpu(),
            "reward": self.reward[:self.size].detach().cpu(),
            "next_obs": self.next_obs[:self.size].detach().cpu(),
            "done": self.done[:self.size].detach().cpu(),
        }
        if self.nominal_action is not None:
            state["nominal_action"] = self.nominal_action[:self.size].detach().cpu()
        return state

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
        # Buffers saved without nominal_action keep the zero-initialized value.
        if self.nominal_action is not None and "nominal_action" in state:
            source = state["nominal_action"]
            if source.shape[1:] != self.nominal_action.shape[1:]:
                raise ValueError(
                    f"Replay nominal_action shape mismatch: {source.shape} vs {self.nominal_action.shape}"
                )
            self.nominal_action[:size].copy_(source[:size].to(self.device))
        self.size = size
        self.write_index = int(state.get("write_index", size % self.capacity)) % self.capacity
