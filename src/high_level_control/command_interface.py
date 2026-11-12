from __future__ import annotations

from typing import Any
import numpy as np
import torch


class MjlabVelocityCommandInterface:
    """Overwrite the command tensor consumed by mdp.generated_commands('twist').

    The command term in unitree_rl_mjlab is named ``twist``. The exact private
    storage name has changed between mjlab versions, so this adapter resolves the
    term and tries the common command-buffer fields.
    """

    def __init__(self, env: Any, command_name: str = "twist", env_index: int = 0) -> None:
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.command_name = command_name
        self.env_index = env_index
        self.term = self._resolve_term()
        self.buffer_name = self._resolve_buffer_name()
        self.disable_resampling()

    def _resolve_term(self) -> Any:
        manager = self.env.command_manager
        if hasattr(manager, "get_term"):
            try:
                return manager.get_term(self.command_name)
            except Exception:
                pass
        for attr in ("_terms", "terms", "command_terms"):
            mapping = getattr(manager, attr, None)
            if isinstance(mapping, dict) and self.command_name in mapping:
                return mapping[self.command_name]
        raise RuntimeError(
            f"Could not resolve command term '{self.command_name}'. "
            "Inspect env.unwrapped.command_manager and update the adapter."
        )

    def _resolve_buffer_name(self) -> str:
        candidates = (
            "command",
            "command_b",
            "vel_command_b",
            "twist_command",
            "_command",
            "_command_b",
        )
        for name in candidates:
            value = getattr(self.term, name, None)
            if isinstance(value, torch.Tensor) and value.shape[-1] >= 3:
                return name
        raise RuntimeError(
            "Could not find the command tensor on the twist command term. "
            f"Available attributes: {dir(self.term)}"
        )

    def disable_resampling(self) -> None:
        # Prevent the random command generator from replacing external commands.
        for name in ("time_left", "time_left_s", "_time_left", "_time_left_s"):
            value = getattr(self.term, name, None)
            if isinstance(value, torch.Tensor):
                value.fill_(1.0e9)

    def write(self, command: np.ndarray) -> None:
        buffer = getattr(self.term, self.buffer_name)
        cmd = torch.as_tensor(command, dtype=buffer.dtype, device=buffer.device)
        buffer[self.env_index, :3] = cmd[:3]
        self.disable_resampling()
