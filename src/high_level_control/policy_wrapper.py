from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from .command_interface import MjlabVelocityCommandInterface
from .controller import FeedbackLinearizationController
from .keyboard_controller import TerminalKeyboardController
from .logger import ExperimentLogger
from .model import HighLevelKinematicModel
from .safety_filter import IdentitySafetyFilter
from .state_interface import MjlabHighLevelStateInterface
from .trajectories import ReferenceTrajectory


class HighLevelPolicyWrapper:
    """Viewer-compatible wrapper around the frozen low-level policy.

    The low-level actor still runs at the environment rate (50 Hz for the
    current task). The high-level command is recomputed at `high_level_rate_hz`
    and held between updates.
    """

    def __init__(
        self,
        env: Any,
        locomotion_policy: Callable,
        state_interface: MjlabHighLevelStateInterface,
        command_interface: MjlabVelocityCommandInterface,
        model: HighLevelKinematicModel,
        controller: FeedbackLinearizationController,
        trajectory: ReferenceTrajectory,
        safety_filter: Any | None = None,
        keyboard_controller: TerminalKeyboardController | None = None,
        high_level_rate_hz: float = 10.0,
        command_obs_start: int = 6,
        verbose_every: int = 100,
        logger: ExperimentLogger | None = None,
        trajectory_name: str = "unknown",
    ) -> None:
        if high_level_rate_hz <= 0.0:
            raise ValueError("high_level_rate_hz must be positive.")
        self.env = env
        self.locomotion_policy = locomotion_policy
        self.state_interface = state_interface
        self.command_interface = command_interface
        self.model = model
        self.controller = controller
        self.trajectory = trajectory
        self.safety_filter = safety_filter or IdentitySafetyFilter()
        self.keyboard_controller = keyboard_controller
        self.high_level_period = 1.0 / high_level_rate_hz
        self.command_obs_start = command_obs_start
        self.verbose_every = verbose_every
        self.logger = logger
        self.trajectory_name = trajectory_name
        self.last_update_time = -np.inf
        self.last_command = np.zeros(3, dtype=np.float64)
        self.last_nominal = np.zeros(3, dtype=np.float64)
        self.call_count = 0

    def _sim_time(self) -> float:
        raw = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        for path in (("sim", "data", "time"), ("sim", "time"), ("episode_length_buf",)):
            obj = raw
            try:
                for item in path:
                    obj = getattr(obj, item)
                if isinstance(obj, torch.Tensor):
                    return float(obj.flatten()[0].item()) * float(getattr(raw, "step_dt", 1.0))
                return float(obj)
            except Exception:
                continue
        step_dt = float(getattr(raw, "step_dt", 0.02))
        return self.call_count * step_dt

    def _inject_command(self, obs: Any) -> Any:
        """Put the held command into the current actor observation."""
        start = self.command_obs_start
        stop = start + 3

        def inject_tensor(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.ndim != 2 or tensor.shape[1] < stop:
                raise RuntimeError(
                    f"Actor observation has shape {tuple(tensor.shape)}; cannot inject "
                    f"command into columns {start}:{stop}."
                )
            updated = tensor.clone()
            command = torch.as_tensor(
                self.last_command, dtype=updated.dtype, device=updated.device
            ).view(1, 3)
            updated[:, start:stop] = command.expand(updated.shape[0], -1)
            return updated

        if isinstance(obs, torch.Tensor):
            return inject_tensor(obs)

        # TensorDict is used by current mjlab versions. Avoid a hard dependency
        # on a specific TensorDict class by checking its mapping-like interface.
        if type(obs).__module__.startswith("tensordict"):
            updated = obs.clone()
            for key in ("actor", "policy"):
                if key in updated.keys():
                    updated[key] = inject_tensor(updated[key])
                    return updated
            raise KeyError(f"Could not find actor/policy tensor in TensorDict keys: {tuple(updated.keys())}")

        if isinstance(obs, dict):
            updated = dict(obs)
            for key in ("actor", "policy"):
                if key in updated and isinstance(updated[key], torch.Tensor):
                    updated[key] = inject_tensor(updated[key])
                    return updated
            raise KeyError(f"Could not find actor/policy tensor in observation keys: {tuple(obs.keys())}")
        raise TypeError(f"Unsupported observation type: {type(obs)!r}")

    def _update_high_level_command(self, now: float) -> None:
        state = self.state_interface.read()
        pose = self.model.controlled_pose(state.base_pose_world)
        reference_pose = None

        if self.keyboard_controller is not None:
            u_nom = self.keyboard_controller.command()
            mode = "keyboard"
        else:
            ref = self.trajectory.evaluate(now)
            reference_pose = ref.pose
            u_nom = self.controller.compute(pose, ref)
            mode = "trajectory"

        result = self.safety_filter.filter(pose, u_nom)
        self.last_nominal = np.asarray(u_nom, dtype=np.float64).copy()
        self.last_command = np.asarray(result.command, dtype=np.float64).copy()
        self.last_update_time = now

        if self.logger is not None:
            self.logger.log(
                time_s=now,
                mode=mode,
                trajectory=self.trajectory_name,
                pose=pose,
                reference_pose=reference_pose,
                measured_velocity=state.base_velocity_body,
                nominal_command=self.last_nominal,
                applied_command=self.last_command,
                h=result.h,
                cbf_value_nominal=result.cbf_value_nominal,
                cbf_active=result.active,
                cbf_correction_norm=result.correction_norm,
            )

        if self.verbose_every > 0 and self.call_count % self.verbose_every == 0:
            print(
                f"[HL:{mode}] t={now:7.2f} pose=({pose[0]:+.2f},{pose[1]:+.2f},{pose[2]:+.2f}) "
                f"u_nom=({u_nom[0]:+.2f},{u_nom[1]:+.2f},{u_nom[2]:+.2f}) "
                f"u=({self.last_command[0]:+.2f},{self.last_command[1]:+.2f},{self.last_command[2]:+.2f}) "
                f"h={result.h:+.3f} active={result.active}"
            )

    def __call__(self, obs: Any):
        now = self._sim_time()
        if now - self.last_update_time >= self.high_level_period - 1.0e-9:
            self._update_high_level_command(now)

        # Re-write every low-level step because the environment command term may
        # otherwise resample or overwrite its internal command buffer.
        self.command_interface.write(self.last_command)
        policy_obs = self._inject_command(obs)

        self.call_count += 1
        with torch.inference_mode():
            return self.locomotion_policy(policy_obs)
