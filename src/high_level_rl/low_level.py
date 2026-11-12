from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls


class LowLevelPolicyAdapter:
    def __init__(self, policy, command_obs_start: int = 6) -> None:
        self.policy = policy
        self.command_obs_start = command_obs_start

    def inject_command(self, obs: Any, command: torch.Tensor) -> Any:
        start, stop = self.command_obs_start, self.command_obs_start + 3

        def update(t: torch.Tensor) -> torch.Tensor:
            out = t.clone()
            out[:, start:stop] = command.to(dtype=t.dtype, device=t.device)
            return out

        if isinstance(obs, torch.Tensor):
            return update(obs)
        if type(obs).__module__.startswith("tensordict"):
            out = obs.clone()
            for key in ("actor", "policy"):
                if key in out.keys():
                    out[key] = update(out[key])
                    return out
        if isinstance(obs, dict):
            out = dict(obs)
            for key in ("actor", "policy"):
                if key in out:
                    out[key] = update(out[key])
                    return out
        raise TypeError(f"Unsupported low-level observation type: {type(obs)!r}")

    def __call__(self, obs: Any, command: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.policy(self.inject_command(obs, command))


def load_frozen_low_level_policy(task_id: str, env, checkpoint_file: str, device: str):
    checkpoint = Path(checkpoint_file).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    cfg = load_rl_cfg(task_id)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    return LowLevelPolicyAdapter(policy)
