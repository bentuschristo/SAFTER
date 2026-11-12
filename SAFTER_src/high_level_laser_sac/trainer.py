from __future__ import annotations

import copy
import json
import math
import os
import time
import warnings
from collections import Counter, deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .config import LaserSacConfig
from .networks import LaserSafeSacActor, TwinQCritic
from .procedural_maps import CURRICULUM_WEIGHTS
from .replay_buffer import VectorReplayBuffer


def _format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


class LaserSacTrainer:
    ROLLING_EPISODES = 1000
    EPISODE_METRIC_TAGS = {
        "return": "return_mean",
        "length": "length_mean",
        "success": "success_rate",
        "collision": "collision_rate",
        "timeout": "timeout_rate",
        "curriculum": "curriculum_truncation_rate",
    }
    # Exact tag names used by the v4 SAC-based methods' 100-episode window
    # (Train/mean_episode_return_100, Train/episode_success_rate_100, etc.),
    # reused verbatim here so both trainers' curves overlay as the same chart
    # when their logs are loaded into one TensorBoard instance. timeout/
    # curriculum have no v4 equivalent (v4 doesn't track those outcomes), so
    # they extend the same "episode_..._rate_100" pattern.
    ROLLING_100_TAGS = {
        "return": "mean_episode_return_100",
        "length": "mean_episode_length_100",
        "success": "episode_success_rate_100",
        "collision": "episode_collision_rate_100",
        "timeout": "episode_timeout_rate_100",
        "curriculum": "episode_curriculum_truncation_rate_100",
    }

    def __init__(self, env, cfg: LaserSacConfig, log_dir: Path, device: str) -> None:
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        obs = self.actor_obs(env.get_observations())
        obs_dim = int(obs.shape[-1])
        self.obs_dim = obs_dim
        hp = cfg.sac
        nav = env.navigation_cfg
        self.actor = LaserSafeSacActor(
            obs_dim=obs_dim,
            hidden_dims=hp.hidden_dims,
            action_limits=(nav.action.max_vx, nav.action.max_vy, nav.action.max_wz),
            ray_angles=env.ray_angles,
            range_start=8,
            num_rays=cfg.laser.num_rays,
            max_range=cfg.laser.max_range,
            safety_cfg=cfg.safety,
            method=cfg.method,
            log_std_min=hp.log_std_min,
            log_std_max=hp.log_std_max,
            min_vx=nav.action.min_vx,
        ).to(self.device)
        self.critic = TwinQCritic(obs_dim, 3, hp.hidden_dims).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
        self.log_alpha = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=hp.alpha_lr)
        self.buffer = VectorReplayBuffer(hp.replay_size, obs_dim, 3, self.device)
        self.global_step = 0
        self.vector_step = 0
        self.gradient_step = 0
        self.resumed = False
        self.replay_restored = False
        self.target_vector_steps: int | None = None
        self.curriculum_total_vector_steps: int | None = None
        self.original_curriculum_target: int | None = None
        self.curriculum_replanned = False
        self.latest_replay_path: Path | None = None
        self.cumulative_outcomes = Counter()
        self.rolling = {
            name: deque(maxlen=self.ROLLING_EPISODES) for name in
            ("return", "length", "success", "collision", "timeout", "curriculum")
        }
        # Additional 100-episode window, matching the v4 SAC-based methods'
        # convention, logged alongside (not instead of) the 1000-episode one.
        self.rolling_100 = {name: deque(maxlen=100) for name in self.rolling}
        # Interval buffers contain every episode completed since the previous
        # log.  Unlike a bounded deque, a synchronized vector-env completion
        # batch can never discard its lower-index environments.
        self.interval_episodes = {name: [] for name in self.rolling}
        # Accumulate every optimizer result between log records instead of
        # reporting only the final minibatch update.
        self.interval_updates: dict[str, list[float]] = {}
        self.collision_rolling = {
            name: deque(maxlen=100) for name in (
                "surface_clearance", "laser_min_range", "filter_active", "correction_norm",
                "nominal_command_norm", "safe_command_norm", "h_lse", "projection_feasible",
                "projected_residual", "input_bound_violation",
            )
        }
        self.collision_history_steps = int(cfg.collision_trajectory_history_steps)
        if self.collision_history_steps <= 0:
            raise ValueError("collision_trajectory_history_steps must be positive")
        # Older server checkpoints/configs predate update_every_steps and
        # performed optimizer updates every vector step.
        self.update_every_steps = int(getattr(hp, "update_every_steps", 1))
        if self.update_every_steps <= 0:
            raise ValueError("update_every_steps must be positive")
        self._collision_history: dict[str, torch.Tensor] = {}
        self._collision_history_write = 0
        self._collision_history_samples = 0

        # --- Optional bCBF invariant-ellipse training-time diagnostics ---
        # (2026-09-07). Purely additive: derived only from fields the safety
        # filter already returns (safety.individual_h, .cbf_value_nominal,
        # .correction_norm, .active, .projection_feasible) plus the single
        # new diagnostic-only field safety.lg_aug_sq_norm -- nothing here is
        # read by the RL update or fed back into `command`. Off by default;
        # enabled via train(enable_bcbf_diagnostics=True). Only meaningful
        # for method="bcbf_cf" with terminal_backup_mode="invariant_ellipse"
        # (the legacy mode and the QP variants don't populate
        # lg_aug_sq_norm / don't have the same individual_h column layout).
        self._bcbf_diag_ready = (
            self.cfg.method == "bcbf_cf"
            and getattr(self.actor.safety_filter, "terminal_backup_mode", None) == "invariant_ellipse"
        )
        self._bcbf_diag_enabled = False
        self._reset_bcbf_diag_accumulators()

        config_path = self.log_dir / "config.json"
        if config_path.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_path = self.log_dir / f"resume_config_{stamp}.json"
        config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        (self.log_dir / "logger_schema_v5.json").write_text(json.dumps({
            "schema_version": 5,
            "x_axis": "environment_transitions",
            "num_parallel_environments": int(self.env.num_envs),
            "reward_profile": cfg.reward.profile,
            "high_level_dt_seconds": cfg.navigation.high_level_dt,
            "low_level_dt_seconds": cfg.navigation.low_level_dt,
            "safety_integration_dt_seconds": cfg.safety.integration_dt,
            "backup_rollout_substep_seconds": (
                cfg.safety.backup_horizon_s
                / (cfg.safety.backup_samples - 1)
                / cfg.safety.backup_rollout_substeps
            ),
            "rolling_episode_window": self.ROLLING_EPISODES,
            "interval_definition": "all episodes completed since the previous TensorBoard log",
            "collision_definition": "obstacle collision OR wall/outside collision OR fall",
            "laser_safety_violation_logged": False,
            "raw_episode_file": "episode_metrics.jsonl",
            "tensorboard_groups": {
                "TrainInterval": "unsmoothed interval episodic statistics",
                "TrainRolling1000": "rolling episodic diagnostics",
                "Train": "includes a 100-episode rolling window (mean_episode_return_100, episode_success_rate_100, etc.) using the v4 SAC-based methods' exact tag names, so both overlay as the same chart in TensorBoard",
                "TrainCumulative": "all completed training episodes",
                "SAC": "mean and population standard deviation over optimizer updates in the log interval",
                "CollisionEpisodes": "diagnostics conditional on obstacle-collision episodes",
            },
        }, indent=2), encoding="utf-8")
        self._print_spec(obs_dim)

    @staticmethod
    def actor_obs(obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs
        if hasattr(obs, "get"):
            actor = obs.get("actor")
            if actor is not None:
                return actor
        return obs["actor"]

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _print_spec(self, obs_dim: int) -> None:
        r = self.cfg.reward
        order = "left-to-right" if self.cfg.laser.left_to_right else "right-to-left"
        print("\n+------------------------------------------------------------+")
        print(f"| Laser SAC method={self.cfg.method:<39}|")
        print("+-------------------------------+----------------------------+")
        print(f"| Observation dimension         | {obs_dim:<26}|")
        print(f"| Laser rays                    | {self.cfg.laser.num_rays:<26}|")
        print(f"| Laser FOV deg                 | {self.cfg.laser.fov_deg:<26.1f}|")
        print(f"| Hidden dimensions             | {str(self.cfg.sac.hidden_dims):<26}|")
        print(f"| SAC discount gamma            | {self.cfg.sac.gamma:<26.6f}|")
        print(f"| Optimizer update interval     | {self.update_every_steps:<26}|")
        print(f"| Training reward profile       | {self.cfg.reward.profile:<26}|")
        print(f"| High-level/filter Hz          | {1.0 / self.cfg.navigation.high_level_dt:<26.1f}|")
        print(f"| Low-level command Hz          | {1.0 / self.cfg.navigation.low_level_dt:<26.1f}|")
        print(f"| Safety integration dt [s]     | {self.cfg.safety.integration_dt:<26.3f}|")
        backup_substep = (
            self.cfg.safety.backup_horizon_s
            / (self.cfg.safety.backup_samples - 1)
            / self.cfg.safety.backup_rollout_substeps
        )
        print(f"| Backup rollout substep [s]    | {backup_substep:<26.3f}|")
        print(f"| Laser noise enabled           | {str(self.cfg.laser.enable_lidar_noise):<26}|")
        print(f"| Laser noise preset            | {self.cfg.laser.lidar_noise_preset:<26}|")
        print(f"| Laser index order             | {order:<26}|")
        print(f"| Critic action                 | {'executed [vx, vy, wz]':<26}|")
        print("+-------------------------------+----------------------------+")
        print("\nReward table")
        rows = [
            ("progress", r.progress_weight),
            ("time", -r.time_penalty),
            ("forward_alignment", r.forward_alignment_weight),
            ("reverse_motion", -r.reverse_motion_weight),
            ("yaw_tracking", -r.yaw_tracking_weight),
            ("smoothness", -r.smoothness_weight),
            ("cbf_correction", -r.cbf_correction_weight),
        ]
        if r.profile == "original":
            rows.extend([
                ("goal", r.goal_bonus),
                ("timeout", -r.timeout_penalty),
                ("collision", -r.collision_penalty),
            ])
        else:
            dt = self.cfg.navigation.high_level_dt
            rows.extend([
                ("goal_tight_scale_x_dt", r.seanav_goal_scale * dt),
                ("timeout_scale_x_dt", r.seanav_timeout_scale * dt),
                ("collision_scale_x_dt", r.seanav_collision_scale * dt),
                ("termination_scale_x_dt", r.seanav_termination_scale * dt),
            ])
        for name, weight in rows:
            print(f"  {name:<24} {weight:+.4f}")
        print()

    def _soft_update(self) -> None:
        # See SacHyperparameters.reference_dt: preserves the effective
        # per-second target-tracking rate across different high_level_dt
        # settings rather than applying the archived-10Hz-calibrated tau
        # 5x more (or less) often per second of simulated time than intended.
        dt_scale = float(self.cfg.navigation.high_level_dt) / float(self.cfg.sac.reference_dt)
        tau = 1.0 - (1.0 - self.cfg.sac.tau) ** dt_scale
        with torch.no_grad():
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters(), strict=True):
                target.mul_(1.0 - tau).add_(source, alpha=tau)

    @staticmethod
    def _flatten_collision_sample(
        sample: dict[str, object], prefix: str = ""
    ) -> dict[str, torch.Tensor]:
        """Flatten nested diagnostic mappings into tensor-valued NPZ fields."""
        flattened: dict[str, torch.Tensor] = {}
        for name, value in sample.items():
            field = f"{prefix}_{name}" if prefix else name
            if isinstance(value, dict):
                flattened.update(LaserSacTrainer._flatten_collision_sample(value, field))
            elif torch.is_tensor(value):
                flattened[field] = value
            else:
                raise TypeError(
                    f"Collision history field {field!r} must be a tensor or mapping, "
                    f"got {type(value).__name__}"
                )
        return flattened

    def _append_collision_history(self, sample: dict[str, object], vector_step: int) -> None:
        """Append one vector step without CPU synchronization or disk I/O."""
        slot = self._collision_history_write
        augmented = self._flatten_collision_sample(sample)
        augmented["vector_step"] = torch.full(
            (self.env.num_envs,), vector_step, dtype=torch.long, device=self.device
        )
        for name, value in augmented.items():
            value = value.detach().to(self.device)
            if value.shape[0] != self.env.num_envs:
                raise ValueError(f"Collision history field {name!r} has shape {tuple(value.shape)}")
            if name not in self._collision_history:
                self._collision_history[name] = torch.empty(
                    (self.collision_history_steps, *value.shape),
                    dtype=value.dtype,
                    device=value.device,
                )
            self._collision_history[name][slot].copy_(value)
        self._collision_history_write = (slot + 1) % self.collision_history_steps
        self._collision_history_samples += 1

    def _save_collision_trajectory(
        self,
        *,
        env_id: int,
        episode_length: int,
        map_id: int,
        scenario_id: int,
    ) -> Path:
        """Atomically save the current episode tail in chronological order."""
        count = min(episode_length, self.collision_history_steps, self._collision_history_samples)
        slots = [
            (self._collision_history_write - count + offset) % self.collision_history_steps
            for offset in range(count)
        ]
        arrays = {
            name: values[slots, env_id].detach().cpu().numpy()
            for name, values in self._collision_history.items()
        }
        arrays["history_length"] = np.asarray(count, dtype=np.int64)
        arrays["high_level_dt_seconds"] = np.asarray(
            self.cfg.navigation.high_level_dt, dtype=np.float64
        )
        arrays["laser_ray_angles"] = self.actor.ray_angles.detach().cpu().numpy()
        trajectory_dir = self.log_dir / "collision_trajectories"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"collision_v{self.vector_step:08d}_env{env_id:04d}_"
            f"map{map_id:03d}_scenario{scenario_id:06d}.npz"
        )
        path = trajectory_dir / filename
        temporary = trajectory_dir / f".{filename}.tmp.npz"
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
        return path

    @staticmethod
    def _optional_mean(tensor) -> float | None:
        if tensor is None:
            return None
        return float(tensor.detach().mean())

    def update(self) -> dict[str, float]:
        hp = self.cfg.sac
        batch = self.buffer.sample(hp.batch_size)
        # See SacHyperparameters.reference_dt: preserves the effective
        # per-second discounting (and hence real-time planning horizon)
        # across different high_level_dt settings, rather than silently
        # compounding the archived-10Hz-calibrated gamma once per decision
        # regardless of how much real time that decision now spans.
        dt_scale = float(self.cfg.navigation.high_level_dt) / float(hp.reference_dt)
        gamma = hp.gamma ** dt_scale

        with torch.no_grad():
            next_out = self.actor.sample(batch.next_obs)
            target_q1, target_q2 = self.target_critic(batch.next_obs, next_out.safe_normalized)
            target_v = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_out.log_prob
            target = batch.reward + (1.0 - batch.done) * gamma * target_v

        q1, q2 = self.critic(batch.obs, batch.action)
        with torch.no_grad():
            td_error_q1 = (q1 - target).abs().mean()
            td_error_q2 = (q2 - target).abs().mean()
        critic_loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), hp.gradient_clip_norm)
        self.critic_opt.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        try:
            actor_out = self.actor.sample(batch.obs)
            actor_q1, actor_q2 = self.critic(batch.obs, actor_out.safe_normalized)
            actor_q = torch.minimum(actor_q1, actor_q2)
            actor_loss = (self.alpha.detach() * actor_out.log_prob - actor_q).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), hp.gradient_clip_norm)
            self.actor_opt.step()
        finally:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (actor_out.log_prob.detach() + hp.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update()
        self.gradient_step += 1
        metrics = {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "q": float(actor_q.mean().detach()),
            "q1": float(q1.mean().detach()),
            "q2": float(q2.mean().detach()),
            "target_q": float(torch.minimum(target_q1, target_q2).mean().detach()),
            "target_v": float(target_v.mean().detach()),
            "bellman_target": float(target.mean().detach()),
            "td_error_q1_abs": float(td_error_q1.detach()),
            "td_error_q2_abs": float(td_error_q2.detach()),
            "batch_reward": float(batch.reward.mean().detach()),
            "batch_terminal_fraction": float(batch.done.mean().detach()),
            "entropy_proxy": float((-actor_out.log_prob).mean().detach()),
            "actor_cbf_active": float(actor_out.safety.active.float().mean().detach()),
            "actor_cbf_correction": float(actor_out.safety.correction_norm.mean().detach()),
            "actor_h_lse": float(actor_out.safety.h_lse.mean().detach()),
            "actor_minimum_h": float(actor_out.safety.individual_h.min(dim=-1).values.mean().detach()),
            "actor_lambda": float(actor_out.safety.lambda_value.mean().detach()),
            "actor_nominal_command_norm": float(torch.linalg.norm(actor_out.nominal_command, dim=-1).mean().detach()),
            "actor_safe_command_norm": float(torch.linalg.norm(actor_out.safe_command, dim=-1).mean().detach()),
            "actor_action_delta_norm": float(
                torch.linalg.norm(actor_out.safe_command - actor_out.nominal_command, dim=-1).mean().detach()
            ),
        }
        optional = {
            "actor_h_backup_lse": self._optional_mean(getattr(actor_out.safety, "h_backup_lse", None)),
            "actor_h_input_lse": self._optional_mean(getattr(actor_out.safety, "h_input_lse", None)),
            "actor_input_bound_violation": self._optional_mean(
                getattr(actor_out.safety, "input_bound_violation", None)
            ),
        }
        for name, value in optional.items():
            if value is not None:
                metrics[name] = value
        return metrics

    def _map_bank_metadata(self) -> dict[str, object] | None:
        bank = getattr(self.env, "procedural_map_bank", None)
        return bank.checkpoint_metadata() if bank is not None else None

    @staticmethod
    def _atomic_torch_save(value: object, path: Path) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        torch.save(value, temporary)
        os.replace(temporary, path)

    def save(self, path: Path, save_replay: bool = False, keep_latest_replay_only: bool = True) -> None:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing checkpoint: {path}")
        replay_path = path.with_name(path.name.replace("model_step_", "replay_step_"))
        replay_file = replay_path.name if save_replay else None
        if save_replay:
            if replay_path.exists():
                raise FileExistsError(f"Refusing to overwrite existing replay snapshot: {replay_path}")
            previous_replay = self.latest_replay_path
            self._atomic_torch_save(self.buffer.state_dict(), replay_path)
            self.latest_replay_path = replay_path
        self._atomic_torch_save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_opt.state_dict(),
                "critic_optimizer": self.critic_opt.state_dict(),
                "log_alpha": self.log_alpha.detach(),
                "alpha_optimizer": self.alpha_opt.state_dict(),
                "global_step": self.global_step,
                "vector_step": self.vector_step,
                "gradient_step": self.gradient_step,
                "target_vector_steps": self.target_vector_steps,
                "curriculum_total_vector_steps": self.curriculum_total_vector_steps,
                "config": asdict(self.cfg),
                "map_bank": self._map_bank_metadata(),
                "replay_file": replay_file,
                "cumulative_outcomes": dict(self.cumulative_outcomes),
                "rolling_episode_windows": {name: list(values) for name, values in self.rolling.items()},
                "rolling_episode_windows_100": {name: list(values) for name, values in self.rolling_100.items()},
                "curriculum_state": self.env.curriculum_state_dict()
                if hasattr(self.env, "curriculum_state_dict") else None,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            path,
        )
        # Retire the older large snapshot only after both new files are durable.
        if save_replay and keep_latest_replay_only and previous_replay and previous_replay != replay_path and previous_replay.is_file():
            previous_replay.unlink()

    def _check_checkpoint_compatibility(self, checkpoint: dict, path: Path) -> None:
        saved_cfg = checkpoint.get("config")
        if not isinstance(saved_cfg, dict):
            return
        saved_laser = saved_cfg.get("laser")
        if not isinstance(saved_laser, dict):
            return

        current_laser = asdict(self.cfg)["laser"]
        mismatches: list[str] = []
        fields = ("num_rays", "fov_deg", "max_range", "min_range", "yaw_offset_deg", "left_to_right")
        for field in fields:
            saved_value = saved_laser.get(field)
            if saved_value is None and field == "left_to_right":
                saved_value = False
            elif saved_value is None:
                mismatches.append(f"{field}: checkpoint=<missing>, current={current_laser[field]}")
                continue
            current_value = current_laser[field]
            if isinstance(current_value, float):
                if abs(float(saved_value) - current_value) > 1.0e-6:
                    mismatches.append(f"{field}: checkpoint={saved_value}, current={current_value}")
            elif saved_value != current_value:
                mismatches.append(f"{field}: checkpoint={saved_value}, current={current_value}")

        saved_method = saved_cfg.get("method")
        if saved_method is not None and saved_method != self.cfg.method:
            mismatches.append(f"method: checkpoint={saved_method}, current={self.cfg.method}")
        if saved_laser != current_laser:
            mismatches.append("complete LiDAR configuration differs")
        # Network/optimizer shapes and learning semantics must match exactly on
        # resume. In particular, an original (512,256,128) checkpoint cannot be
        # loaded into the wider (512,512,256) capacity experiment.
        for section in ("navigation", "reward", "safety", "collision_initialization", "sac"):
            saved_section = saved_cfg.get(section)
            current_section = asdict(self.cfg).get(section)
            # Logger-v5 checkpoints created before the SEA-Nav experiment do
            # not contain the newly added reward-profile fields. They are
            # semantically the original profile, so fill only missing keys
            # from the current original configuration before comparison.
            if (
                section == "reward"
                and isinstance(saved_section, dict)
                and isinstance(current_section, dict)
                and current_section.get("profile") == "original"
            ):
                saved_section = {
                    key: saved_section.get(key, value)
                    for key, value in current_section.items()
                }
            # v6 checkpoints created before the invariant-ellipse terminal-set
            # change do not contain terminal_backup_mode/terminal_geom_scale/
            # terminal_rec_scale. Those default to the old legacy behavior
            # (terminal_backup_mode="legacy_inflated_ellipse", scales=1.0), so
            # a checkpoint missing them is semantically unchanged -- fill only
            # missing keys from the current configuration before comparing,
            # same pattern as the reward-profile fill-in above. A checkpoint
            # that DOES record these keys with a different value still
            # correctly reports a mismatch.
            if (
                section == "safety"
                and isinstance(saved_section, dict)
                and isinstance(current_section, dict)
            ):
                saved_section = {
                    key: saved_section.get(key, value)
                    for key, value in current_section.items()
                }
            if saved_section != current_section:
                mismatches.append(f"{section} configuration differs")
        saved_bank = checkpoint.get("map_bank")
        current_bank = self._map_bank_metadata()
        def identity(metadata):
            return None if metadata is None else {k: v for k, v in metadata.items() if k != "path"}
        if identity(saved_bank) != identity(current_bank):
            mismatches.append(f"map-bank/curriculum metadata differs: checkpoint={saved_bank}, current={current_bank}")

        if mismatches:
            details = "\n  - ".join(mismatches)
            raise ValueError(
                f"Laser SAC checkpoint {path} was created with an incompatible training configuration:\n"
                f"  - {details}\n"
                "Use matching LiDAR, safety, method, and SAC-network settings, or train a new checkpoint."
            )

    def load(self, path: Path, load_optimizers: bool = True, requested_target: int | None = None,
             allow_curriculum_replan: bool = False) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._check_checkpoint_compatibility(checkpoint, path)
        # action_scale (old symmetric-only buffer) was replaced by
        # action_low/action_high/action_center/action_half_range; both are
        # derived purely from config, never learned, so a legacy checkpoint's
        # saved value is dropped rather than loaded -- the freshly
        # constructed actor already computed the correct buffers.
        actor_state_dict = dict(checkpoint["actor"])
        actor_state_dict.pop("action_scale", None)
        for buffer_name in ("action_low", "action_high", "action_center", "action_half_range"):
            actor_state_dict.setdefault(buffer_name, self.actor.state_dict()[buffer_name])
        self.actor.load_state_dict(actor_state_dict)
        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])
            self.target_critic.load_state_dict(checkpoint.get("target_critic", checkpoint["critic"]))
        if load_optimizers:
            self.actor_opt.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_opt.load_state_dict(checkpoint["critic_optimizer"])
            self.log_alpha.data.copy_(checkpoint["log_alpha"])
            self.alpha_opt.load_state_dict(checkpoint["alpha_optimizer"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.vector_step = int(checkpoint.get("vector_step", self.global_step // self.env.num_envs))
        expected_global_step = self.vector_step * self.env.num_envs
        if self.global_step != expected_global_step:
            raise ValueError(
                f"Checkpoint counters disagree: global_step={self.global_step}, "
                f"vector_step*num_envs={expected_global_step}"
            )
        self.gradient_step = int(checkpoint.get("gradient_step", 0))
        saved_target = checkpoint.get("curriculum_total_vector_steps", checkpoint.get("target_vector_steps"))
        if saved_target is None:
            raise ValueError("Checkpoint lacks curriculum_total_vector_steps; cannot resume safely")
        self.original_curriculum_target = int(saved_target)
        requested_target = int(requested_target if requested_target is not None else saved_target)
        if requested_target != self.original_curriculum_target and not allow_curriculum_replan:
            raise ValueError(
                f"Checkpoint curriculum target={self.original_curriculum_target}, requested target={requested_target}. "
                "Pass --allow-curriculum-replan to explicitly reinterpret curriculum progress."
            )
        self.curriculum_replanned = requested_target != self.original_curriculum_target
        self.target_vector_steps = requested_target
        self.curriculum_total_vector_steps = requested_target
        if self.curriculum_replanned:
            warnings.warn(
                f"Curriculum replanned from {self.original_curriculum_target} to {requested_target} vector steps; "
                "subsequent progress uses the new denominator.", RuntimeWarning,
            )
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        self.cumulative_outcomes.update(checkpoint.get("cumulative_outcomes", {}))
        for name, values in checkpoint.get("rolling_episode_windows", {}).items():
            if name in self.rolling:
                self.rolling[name].extend(values[-self.ROLLING_EPISODES:])
        # Older checkpoints predate this window; resuming one just starts
        # rolling_100 empty and it refills naturally within 100 episodes.
        for name, values in checkpoint.get("rolling_episode_windows_100", {}).items():
            if name in self.rolling_100:
                self.rolling_100[name].extend(values[-100:])
        curriculum_state = checkpoint.get("curriculum_state")
        if curriculum_state and hasattr(self.env, "load_curriculum_state_dict"):
            self.env.load_curriculum_state_dict(curriculum_state)
        replay_file = checkpoint.get("replay_file")
        replay_path = path.parent / replay_file if replay_file else None
        if replay_path is not None and replay_path.is_file():
            self.buffer.load_state_dict(torch.load(replay_path, map_location="cpu", weights_only=False))
            self.replay_restored = True
            self.latest_replay_path = replay_path
        else:
            warnings.warn(
                "Replay buffer was not restored. Collection will use the loaded actor immediately; "
                "gradient updates will wait for sufficient new replay data.", RuntimeWarning,
            )
        self.resumed = True
        (self.log_dir / f"resume_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json").write_text(
            json.dumps({"source_checkpoint": str(path), "source_replay_file": str(replay_path) if self.replay_restored else None,
                        "original_target": self.original_curriculum_target, "requested_target": requested_target,
                        "vector_step": self.vector_step, "global_step": self.global_step,
                        "map_bank": self._map_bank_metadata(), "curriculum_replanning_allowed": allow_curriculum_replan,
                        "curriculum_replanned": self.curriculum_replanned, "replay_restored": self.replay_restored,
                        "simulator_state": "not checkpointed; resume starts new simulator episodes"}, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _mean(values: deque) -> float:
        return sum(values) / max(len(values), 1)

    @staticmethod
    def _stats(values) -> tuple[float, float]:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return 0.0, 0.0
        return float(array.mean()), float(array.std(ddof=0))

    def _accumulate_update_metrics(self, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            self.interval_updates.setdefault(name, []).append(float(value))

    def _reset_bcbf_diag_accumulators(self) -> None:
        """(Re)zero the GPU-resident bCBF diagnostic accumulators.

        Called once at construction and again after every flush. Every
        accumulator here is a 0-d GPU tensor updated in place with plain
        tensor ops (min/sum/comparison) -- no ``.item()``/``.cpu()`` call
        appears until ``_flush_bcbf_diag`` runs, so accumulating every
        control step adds no extra host<->device synchronization on the hot
        path (see task instructions: "avoid synchronizing GPU tensors every
        control step unnecessarily").
        """
        d = self.device
        self._diag_min_g = torch.tensor(float("inf"), device=d)
        self._diag_min_h_rec = torch.tensor(float("inf"), device=d)
        self._diag_sum_daug = torch.zeros((), device=d)
        self._diag_sum_omega = torch.zeros((), device=d)
        self._diag_sum_correction_norm = torch.zeros((), device=d)
        self._diag_sum_active = torch.zeros((), device=d)
        self._diag_sum_terminal_dominant_frac = torch.zeros((), device=d)
        self._diag_count_e_1e8 = torch.zeros((), dtype=torch.long, device=d)
        self._diag_count_e_1e10 = torch.zeros((), dtype=torch.long, device=d)
        self._diag_count_e_1e12 = torch.zeros((), dtype=torch.long, device=d)
        self._diag_count_daug_1e8 = torch.zeros((), dtype=torch.long, device=d)
        self._diag_count_daug_1e6 = torch.zeros((), dtype=torch.long, device=d)
        self._diag_count_nan_inf = torch.zeros((), dtype=torch.long, device=d)
        self._diag_count_fallback = torch.zeros((), dtype=torch.long, device=d)
        self._diag_env_steps = 0
        # Initial-feasibility check (theorem assumption: h_aug(z0) >= 0 at a
        # fresh reset) and post-filter consistency check (does hard_clip_
        # output actually change the filtered action, and by how much).
        self._diag_sum_h_aug_reset = torch.zeros((), device=d)
        self._diag_count_h_aug_reset_negative = torch.zeros((), dtype=torch.long, device=d)
        self._diag_reset_count = torch.zeros((), dtype=torch.long, device=d)
        self._diag_sum_clip_norm = torch.zeros((), device=d)
        self._diag_count_clip_changed = torch.zeros((), dtype=torch.long, device=d)
        # ZOH-consistent joint box+half-space projection diagnostics
        # (2026-09 hard-clip-consistency fix, Tasks 3/5). All read from
        # LaserSafetyFilterResult fields that are None for older filters, so
        # they simply stay unpopulated (flush guards on `n>0` count) for any
        # non-bcbf_cf method.
        self._diag_count_joint_infeasible = torch.zeros((), dtype=torch.long, device=d)
        self._diag_min_residual_preclip = torch.tensor(float("inf"), device=d)
        self._diag_min_residual_exec = torch.tensor(float("inf"), device=d)
        self._diag_count_residual_exec_below_1e8 = torch.zeros((), dtype=torch.long, device=d)
        self._diag_count_residual_exec_below_1e6 = torch.zeros((), dtype=torch.long, device=d)

    def _accumulate_bcbf_diag(
        self, safety: "LaserSafetyFilterResult", just_reset_mask: torch.Tensor | None = None,
    ) -> None:
        """One step of GPU-only accumulation. See _reset_bcbf_diag_accumulators
        for the no-extra-sync rationale. Column layout matches
        `_backup_terms_and_lie` exactly: individual_h = cat(psi(R),
        h_future(num_samples*R), h_B_geom(R), h_B_rec(R), input_terms(6))."""
        sf = self.actor.safety_filter
        num_rays = self.actor.num_rays
        num_samples = sf.backup_samples - 1
        backup_len = num_rays * (num_samples + 3)
        individual_h = safety.individual_h
        psi = individual_h[:, :num_rays]
        g = individual_h[:, backup_len - 2 * num_rays: backup_len - num_rays]
        h_rec = individual_h[:, backup_len - num_rays: backup_len]
        d_aug = safety.lg_aug_sq_norm

        self._diag_min_g = torch.minimum(self._diag_min_g, g.min())
        self._diag_min_h_rec = torch.minimum(self._diag_min_h_rec, h_rec.min())
        self._diag_sum_daug += d_aug.sum()
        self._diag_count_daug_1e8 += torch.count_nonzero(d_aug < 1.0e-8)
        self._diag_count_daug_1e6 += torch.count_nonzero(d_aug < 1.0e-6)

        # E_i = exp(2 g_i); exp is monotonic so thresholding E_i is equivalent
        # to thresholding g_i directly, avoiding an unnecessary exp() over
        # every (env, ray) pair just to compare against a threshold.
        log_thresholds = (math.log(1.0e-8) / 2.0, math.log(1.0e-10) / 2.0, math.log(1.0e-12) / 2.0)
        self._diag_count_e_1e8 += torch.count_nonzero(g < log_thresholds[0])
        self._diag_count_e_1e10 += torch.count_nonzero(g < log_thresholds[1])
        self._diag_count_e_1e12 += torch.count_nonzero(g < log_thresholds[2])

        self._diag_sum_omega += safety.cbf_value_nominal.sum()
        self._diag_sum_correction_norm += safety.correction_norm.sum()
        self._diag_sum_active += safety.active.float().sum()
        # Cheap per-ray "terminal dominant" proxy: is this ray's terminal
        # branch (min of its own h_B_geom, h_B_rec) tighter than this
        # state's own best current-time HOCBF ray? Deliberately O(R), no
        # sort -- the exhaustive top-3/top-5 LSE-weight ranking used for the
        # offline influence diagnostic (test_bcbf_terminal_invariance.py) is
        # too expensive to run every control step.
        state_best_psi = psi.min(dim=1, keepdim=True).values
        terminal_dominant = torch.minimum(g, h_rec) < state_best_psi
        self._diag_sum_terminal_dominant_frac += terminal_dominant.float().mean(dim=1).sum()

        finite = (
            torch.isfinite(d_aug)
            & torch.isfinite(safety.cbf_value_nominal)
            & torch.all(torch.isfinite(g), dim=1)
            & torch.all(torch.isfinite(h_rec), dim=1)
        )
        self._diag_count_nan_inf += torch.count_nonzero(~finite)
        if safety.projection_feasible is not None:
            self._diag_count_fallback += torch.count_nonzero(~safety.projection_feasible)

        # Initial feasibility: h_aug(z0) at states whose observation THIS
        # step is a fresh post-reset observation (just_reset_mask is the
        # previous step's `done`, since a standard auto-reset vec-env
        # presents the reset observation on the very next step -- see
        # train()'s tracking of self._prev_done for the exact assumption).
        if just_reset_mask is not None and torch.any(just_reset_mask):
            h_aug_reset = safety.h_lse[just_reset_mask]
            self._diag_sum_h_aug_reset += h_aug_reset.sum()
            self._diag_count_h_aug_reset_negative += torch.count_nonzero(h_aug_reset < 0.0)
            self._diag_reset_count += torch.count_nonzero(just_reset_mask)

        # Post-filter consistency: does the optional hard_clip_output step
        # actually change the filtered command, and by how much.
        if safety.command_preclip is not None:
            clip_diff_norm = torch.linalg.norm(safety.command - safety.command_preclip, dim=1)
            self._diag_sum_clip_norm += clip_diff_norm.sum()
            self._diag_count_clip_changed += torch.count_nonzero(clip_diff_norm > 1.0e-9)

        # Joint box+half-space projection diagnostics (Tasks 3/5): empty
        # box/half-space intersections, and the CBF residual at (a) the
        # projected surrogate before the ZOH/hard-clip mapping and (b) the
        # diagnostic-only surrogate that exactly reproduces the ACTUALLY
        # EXECUTED command. Verification only -- never fed back into filter().
        if safety.joint_projection_infeasible is not None:
            self._diag_count_joint_infeasible += torch.count_nonzero(safety.joint_projection_infeasible)
        if safety.projected_cbf_residual is not None:
            self._diag_min_residual_preclip = torch.minimum(
                self._diag_min_residual_preclip, safety.projected_cbf_residual.min()
            )
        if safety.residual_exec is not None:
            self._diag_min_residual_exec = torch.minimum(
                self._diag_min_residual_exec, safety.residual_exec.min()
            )
            self._diag_count_residual_exec_below_1e8 += torch.count_nonzero(safety.residual_exec < -1.0e-8)
            self._diag_count_residual_exec_below_1e6 += torch.count_nonzero(safety.residual_exec < -1.0e-6)

        self._diag_env_steps += individual_h.shape[0]

    def _flush_bcbf_diag(self, global_step: int) -> None:
        """Single sync point: pulls every accumulator to the CPU once and
        writes interval-aggregated scalars, then resets for the next
        interval. Called at the same log_interval cadence as every other
        TensorBoard write in this loop."""
        if self._diag_env_steps == 0:
            return
        n = float(self._diag_env_steps)
        min_g = float(self._diag_min_g)
        w = self.writer
        w.add_scalar(
            "BcbfDiag/min_valid_E_i",
            math.exp(2.0 * min_g) if math.isfinite(min_g) else float("nan"),
            global_step,
        )
        w.add_scalar("BcbfDiag/min_h_B_geom", min_g, global_step)
        w.add_scalar("BcbfDiag/min_h_B_rec", float(self._diag_min_h_rec), global_step)
        w.add_scalar("BcbfDiag/D_aug_mean", float(self._diag_sum_daug) / n, global_step)
        w.add_scalar("BcbfDiag/Omega_mean", float(self._diag_sum_omega) / n, global_step)
        w.add_scalar(
            "BcbfDiag/correction_norm_mean", float(self._diag_sum_correction_norm) / n, global_step
        )
        w.add_scalar("BcbfDiag/intervention_rate", float(self._diag_sum_active) / n, global_step)
        w.add_scalar(
            "BcbfDiag/terminal_dominant_ray_fraction",
            float(self._diag_sum_terminal_dominant_frac) / n,
            global_step,
        )
        w.add_scalar("BcbfDiag/count_E_i_below_1e-8", int(self._diag_count_e_1e8), global_step)
        w.add_scalar("BcbfDiag/count_E_i_below_1e-10", int(self._diag_count_e_1e10), global_step)
        w.add_scalar("BcbfDiag/count_E_i_below_1e-12", int(self._diag_count_e_1e12), global_step)
        w.add_scalar("BcbfDiag/count_D_aug_below_1e-8", int(self._diag_count_daug_1e8), global_step)
        w.add_scalar("BcbfDiag/count_D_aug_below_1e-6", int(self._diag_count_daug_1e6), global_step)
        w.add_scalar("BcbfDiag/nan_inf_events", int(self._diag_count_nan_inf), global_step)
        w.add_scalar(
            "BcbfDiag/loss_of_authority_fallback_events", int(self._diag_count_fallback), global_step
        )

        reset_count = int(self._diag_reset_count)
        if reset_count > 0:
            w.add_scalar(
                "BcbfDiag/initial_h_aug_mean_at_reset",
                float(self._diag_sum_h_aug_reset) / reset_count, global_step,
            )
            w.add_scalar(
                "BcbfDiag/initial_h_aug_negative_fraction",
                float(self._diag_count_h_aug_reset_negative) / reset_count, global_step,
            )
            w.add_scalar(
                "BcbfDiag/initial_h_aug_negative_count",
                int(self._diag_count_h_aug_reset_negative), global_step,
            )
        w.add_scalar("BcbfDiag/postclip_change_norm_mean", float(self._diag_sum_clip_norm) / n, global_step)
        w.add_scalar(
            "BcbfDiag/postclip_changed_fraction", float(self._diag_count_clip_changed) / n, global_step
        )

        w.add_scalar(
            "BcbfDiag/joint_projection_infeasible_count", int(self._diag_count_joint_infeasible), global_step
        )
        w.add_scalar(
            "BcbfDiag/joint_projection_infeasible_fraction",
            float(self._diag_count_joint_infeasible) / n, global_step,
        )
        min_residual_preclip = float(self._diag_min_residual_preclip)
        w.add_scalar(
            "BcbfDiag/min_residual_preclip",
            min_residual_preclip if math.isfinite(min_residual_preclip) else float("nan"),
            global_step,
        )
        min_residual_exec = float(self._diag_min_residual_exec)
        w.add_scalar(
            "BcbfDiag/min_residual_exec",
            min_residual_exec if math.isfinite(min_residual_exec) else float("nan"),
            global_step,
        )
        w.add_scalar(
            "BcbfDiag/count_residual_exec_below_1e-8",
            int(self._diag_count_residual_exec_below_1e8), global_step,
        )
        w.add_scalar(
            "BcbfDiag/count_residual_exec_below_1e-6",
            int(self._diag_count_residual_exec_below_1e6), global_step,
        )

        self._reset_bcbf_diag_accumulators()

    def train(self, vector_steps: int, save_interval: int = 1000, log_interval: int = 10,
              save_replay_buffer: bool = False, replay_save_interval: int = 50_000,
              keep_latest_replay_only: bool = True, enable_bcbf_diagnostics: bool = False) -> None:
        """Train until the absolute target `vector_steps` is reached."""
        if enable_bcbf_diagnostics and not self._bcbf_diag_ready:
            warnings.warn(
                "enable_bcbf_diagnostics=True but method="
                f"{self.cfg.method!r} / terminal_backup_mode="
                f"{getattr(self.actor.safety_filter, 'terminal_backup_mode', None)!r} "
                "-- only bcbf_cf with terminal_backup_mode='invariant_ellipse' is "
                "supported; diagnostics will not be logged this run.",
                stacklevel=2,
            )
        self._bcbf_diag_enabled = bool(enable_bcbf_diagnostics) and self._bcbf_diag_ready
        if self.vector_step >= vector_steps:
            raise ValueError(f"Checkpoint vector_step={self.vector_step} is already at or beyond target={vector_steps}")
        hp = self.cfg.sac
        # learning_starts is calibrated at reference_dt to represent a fixed
        # number of *episode-equivalents* of warm-up data (episode length in
        # steps is episode_duration_s/high_level_dt, so a smaller dt means
        # more steps -- and more raw transitions -- per episode). Left
        # unscaled, a faster high_level_dt would start training on a buffer
        # holding far fewer complete episode trajectories than the archived
        # setup did (e.g. only ~1/5 as many at 5x the decision rate), giving
        # the critic a much less representative sample of how episodes
        # actually end right when learning begins -- the same category of
        # issue replay_size has, just for the warm-up threshold instead of
        # total capacity. Scaling by 1/dt_scale (the inverse of the reward/
        # gamma/tau correction) restores the same episode-equivalent warm-up
        # coverage at any high_level_dt; a no-op at the archived default.
        dt_scale = float(self.cfg.navigation.high_level_dt) / float(hp.reference_dt)
        effective_learning_starts = int(round(hp.learning_starts / dt_scale))
        if self.target_vector_steps is None:
            self.target_vector_steps = vector_steps
            self.curriculum_total_vector_steps = vector_steps
            self.original_curriculum_target = vector_steps
        elif self.target_vector_steps != vector_steps:
            raise ValueError(f"Loaded target={self.target_vector_steps} but train() received {vector_steps}")
        start_vector_step, start_global_step = self.vector_step, self.global_step
        progress = self.vector_step / max(vector_steps, 1)
        if hasattr(self.env, "set_training_progress"):
            self.env.set_training_progress(progress)
        # A resumed process has no live simulator episode to preserve; reset once
        # after restoring sampler state so its first maps match absolute progress.
        obs_dict = self.env.reset() if self.resumed else self.env.get_observations()
        obs = self.actor_obs(obs_dict).to(self.device)
        # Tracks which envs' CURRENT obs is a fresh post-reset observation,
        # for the bCBF diagnostics' initial-feasibility check (h_aug(z0)>=0).
        # True for every env at loop start (both the resumed .reset() and the
        # fresh .get_observations() path start from an initial/reset state);
        # updated to this iteration's `done` after each step_safe call, since
        # a standard auto-reset vec-env presents the reset observation on the
        # very next step.
        self._prev_done = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        start_time = time.perf_counter()
        bank = self._map_bank_metadata() or {}
        print(
            f"[TRAIN START] mode={'resume' if self.resumed else 'fresh'} "
            f"method={self.cfg.method} device={self.device} actor_obs_dim={self.obs_dim} "
            f"vector_step={self.vector_step} target={vector_steps} global_step={self.global_step} "
            f"target_transitions={vector_steps * self.env.num_envs} envs={self.env.num_envs}\n"
            f"[MAP BANK] path={bank.get('path')} hash={bank.get('content_hash')} "
            f"maps={bank.get('num_maps')} scenarios={bank.get('num_scenarios')} "
            f"max_obstacles={bank.get('max_obstacles')} stage={getattr(self.env, 'curriculum_stage', 0) + 1} "
            f"original_curriculum_target={self.original_curriculum_target}\n"
            f"[REPLAY] restored={'yes' if self.replay_restored else 'no'} size={self.buffer.size}"
        )
        last_env_log: dict[str, object] = {}

        for absolute_step in range(self.vector_step + 1, vector_steps + 1):
            progress = (absolute_step - 1) / max(vector_steps, 1)
            if hasattr(self.env, "set_training_progress"):
                self.env.set_training_progress(progress)  # stage-change signal only; no forced reset

            fresh_warmup = not self.resumed and self.buffer.size < effective_learning_starts
            with torch.no_grad():
                if fresh_warmup:
                    nominal_normalized = torch.empty((self.env.num_envs, 3), device=self.device).uniform_(-1, 1)
                    safe_norm, safe_cmd, safety = self.actor.apply_safety(nominal_normalized, obs)
                    nominal_cmd = self.actor.normalized_to_command(nominal_normalized)
                else:
                    out = self.actor.sample(obs)
                    nominal_cmd, safe_norm, safe_cmd, safety = (
                        out.nominal_command, out.safe_normalized, out.safe_command, out.safety
                    )
                if self._bcbf_diag_enabled:
                    self._accumulate_bcbf_diag(safety, just_reset_mask=self._prev_done)

            filter_input_obs = obs
            next_obs_dict, reward, done, extras = self.env.step_safe(nominal_cmd, safe_cmd, safety)
            if self._bcbf_diag_enabled:
                self._prev_done = done.detach()
            next_obs = self.actor_obs(next_obs_dict).to(self.device)
            replay_next_obs = next_obs
            time_outs = extras.get("time_outs")
            if time_outs is not None:
                # Time limits are not MDP terminals: store the pre-reset terminal
                # observation and bootstrap through the timeout transition.
                terminal_obs = self.actor_obs(extras["terminal_observation"]).to(self.device)
                timeout_mask = time_outs.to(self.device).bool().unsqueeze(-1)
                replay_next_obs = torch.where(timeout_mask, terminal_obs, next_obs)
            terminal_mask = done.detach()
            if time_outs is not None:
                terminal_mask = terminal_mask & ~time_outs.to(done.device).bool()
            self.buffer.add_batch(obs, safe_norm.detach(), reward.detach(), replay_next_obs, terminal_mask.detach())
            obs = next_obs
            self.vector_step = absolute_step
            self.global_step += self.env.num_envs
            trajectory_step = extras.get("collision_trajectory_step")
            if trajectory_step is not None:
                # These are the exact normalized ranges from the actor input
                # used to calculate this step's filtered command.
                trajectory_step["filter_input_laser_ranges"] = (
                    filter_input_obs[:, self.actor.range_start : self.actor.range_start + self.actor.num_rays]
                    * self.actor.max_range
                ).detach()
                self._append_collision_history(trajectory_step, absolute_step)

            completed = extras.get("completed_returns")
            if completed is not None and completed.numel():
                fields = {
                    "return": completed, "length": extras["completed_lengths"].float(),
                    "success": extras["completed_success"].float(),
                    "collision": extras["completed_collision"].float(),
                    "timeout": extras["completed_timeouts"].float(),
                    "curriculum": extras["completed_curriculum_truncations"].float(),
                }
                field_values = {name: tensor.cpu().tolist() for name, tensor in fields.items()}
                for name, values in field_values.items():
                    self.rolling[name].extend(values)
                    self.rolling_100[name].extend(values)
                    self.interval_episodes[name].extend(values)
                self.cumulative_outcomes["completed"] += int(completed.numel())
                for name in ("success", "collision", "timeout", "curriculum"):
                    self.cumulative_outcomes[name] += int(fields[name].sum())

                # Append the unsmoothed episode-level source data so every
                # plotted aggregate can be independently reconstructed later.
                episode_path = self.log_dir / "episode_metrics.jsonl"
                env_ids = extras["completed_env_ids"].cpu().tolist()
                map_ids = extras["completed_map_ids"].cpu().tolist()
                scenario_ids = extras["completed_scenario_ids"].cpu().tolist()
                with episode_path.open("a", encoding="utf-8") as episode_file:
                    for index in range(int(completed.numel())):
                        episode_file.write(json.dumps({
                            "vector_step": self.vector_step,
                            "global_step": self.global_step,
                            "env_id": int(env_ids[index]),
                            "map_id": int(map_ids[index]),
                            "scenario_id": int(scenario_ids[index]),
                            "return": float(field_values["return"][index]),
                            "length": int(field_values["length"][index]),
                            "success": bool(field_values["success"][index]),
                            "collision": bool(field_values["collision"][index]),
                            "timeout": bool(field_values["timeout"][index]),
                            "curriculum_truncation": bool(field_values["curriculum"][index]),
                        }, separators=(",", ":")) + "\n")

                collision_mask = extras["completed_obstacle_collision"].bool()
                if bool(collision_mask.any()):
                    collision_values = {
                        "surface_clearance": extras["completed_obstacle_surface_clearance"],
                        "laser_min_range": extras["completed_laser_min_range"],
                        "filter_active": extras["completed_filter_active"].float(),
                        "correction_norm": extras["completed_filter_correction_norm"],
                        "nominal_command_norm": torch.linalg.norm(extras["completed_nominal_command"], dim=-1),
                        "safe_command_norm": torch.linalg.norm(extras["completed_safe_command"], dim=-1),
                        "h_lse": extras["completed_h_lse"],
                        "projection_feasible": extras["completed_projection_feasible"].float(),
                        "projected_residual": extras["completed_projected_cbf_residual"],
                        "input_bound_violation": extras["completed_input_bound_violation"],
                    }
                    for name, tensor in collision_values.items():
                        self.collision_rolling[name].extend(tensor[collision_mask].cpu().tolist())

                    event_path = self.log_dir / "collision_events.jsonl"
                    event_rows = []
                    collision_indices = torch.nonzero(collision_mask, as_tuple=False).flatten()
                    for index in collision_indices.tolist():
                        env_id = int(extras["completed_env_ids"][index])
                        map_id = int(extras["completed_map_ids"][index])
                        scenario_id = int(extras["completed_scenario_ids"][index])
                        trajectory_path = self._save_collision_trajectory(
                            env_id=env_id,
                            episode_length=int(extras["completed_lengths"][index]),
                            map_id=map_id,
                            scenario_id=scenario_id,
                        )
                        nominal = extras["completed_nominal_command"][index].cpu().tolist()
                        safe = extras["completed_safe_command"][index].cpu().tolist()
                        position = extras["completed_terminal_positions"][index].cpu().tolist()
                        event_rows.append({
                            "vector_step": self.vector_step,
                            "global_step": self.global_step,
                            "env_id": env_id,
                            "map_id": map_id,
                            "scenario_id": scenario_id,
                            "trajectory_file": str(trajectory_path.relative_to(self.log_dir)),
                            "trajectory_history_steps": min(
                                int(extras["completed_lengths"][index]),
                                self.collision_history_steps,
                                self._collision_history_samples,
                            ),
                            "position_xy": position,
                            "surface_clearance_m": float(extras["completed_obstacle_surface_clearance"][index]),
                            "laser_min_range_m": float(extras["completed_laser_min_range"][index]),
                            "filter_active": bool(extras["completed_filter_active"][index]),
                            "correction_norm": float(extras["completed_filter_correction_norm"][index]),
                            "nominal_command": nominal,
                            "safe_command": safe,
                            "h_lse": float(extras["completed_h_lse"][index]),
                            "projection_feasible": bool(extras["completed_projection_feasible"][index]),
                            "projected_cbf_residual": float(extras["completed_projected_cbf_residual"][index]),
                            "input_bound_violation": float(extras["completed_input_bound_violation"][index]),
                        })
                    with event_path.open("a", encoding="utf-8") as event_file:
                        for row in event_rows:
                            event_file.write(json.dumps(row, separators=(",", ":")) + "\n")

            if (
                self.buffer.size >= max(effective_learning_starts, hp.batch_size)
                and absolute_step % self.update_every_steps == 0
            ):
                for _ in range(hp.updates_per_step):
                    self._accumulate_update_metrics(self.update())
            last_env_log = extras["log"]

            if absolute_step % log_interval == 0 or absolute_step == vector_steps:
                if self._bcbf_diag_enabled:
                    self._flush_bcbf_diag(self.global_step)
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                run_transitions = self.global_step - start_global_step
                fps = run_transitions / elapsed
                eta = (vector_steps - absolute_step) * self.env.num_envs / max(fps, 1e-6)
                for name, value in last_env_log.items():
                    if name in {
                        "laser_safety_violation_transition_rate",
                        "fall_transition_rate",
                        "wall_collision_transition_rate",
                    }:
                        continue
                    self.writer.add_scalar(f"Episode/{name}", float(value), self.global_step)
                for name, values in self.interval_updates.items():
                    mean, std = self._stats(values)
                    self.writer.add_scalar(f"SAC/{name}_mean", mean, self.global_step)
                    self.writer.add_scalar(f"SAC/{name}_std", std, self.global_step)
                self.writer.add_scalar(
                    "SAC/gradient_updates_interval",
                    max((len(values) for values in self.interval_updates.values()), default=0),
                    self.global_step,
                )
                self.interval_updates.clear()
                train_metrics = {
                    "completed_episodes_total": self.cumulative_outcomes["completed"],
                    "replay_size": self.buffer.size, "gradient_step": self.gradient_step,
                    "fps": fps, "eta_seconds": eta,
                }
                for name, value in train_metrics.items():
                    self.writer.add_scalar(f"Train/{name}", value, self.global_step)
                completed_total = self.cumulative_outcomes["completed"]
                if completed_total:
                    for name in ("success", "collision", "timeout", "curriculum"):
                        tag = self.EPISODE_METRIC_TAGS[name]
                        self.writer.add_scalar(
                            f"TrainCumulative/{tag}",
                            self.cumulative_outcomes[name] / completed_total,
                            self.global_step,
                        )
                rolling_count = len(self.rolling["return"])
                if rolling_count:
                    for name, values in self.rolling.items():
                        mean, std = self._stats(values)
                        tag = self.EPISODE_METRIC_TAGS[name]
                        self.writer.add_scalar(f"TrainRolling1000/{tag}", mean, self.global_step)
                        if name in {"return", "length"}:
                            self.writer.add_scalar(f"TrainRolling1000/{name}_std", std, self.global_step)
                    self.writer.add_scalar("TrainRolling1000/episodes", rolling_count, self.global_step)
                if len(self.rolling_100["return"]):
                    for name, values in self.rolling_100.items():
                        mean, _std = self._stats(values)
                        tag = self.ROLLING_100_TAGS[name]
                        self.writer.add_scalar(f"Train/{tag}", mean, self.global_step)
                interval_count = len(self.interval_episodes["return"])
                if interval_count:
                    for name, values in self.interval_episodes.items():
                        mean, std = self._stats(values)
                        tag = self.EPISODE_METRIC_TAGS[name]
                        self.writer.add_scalar(f"TrainInterval/{tag}", mean, self.global_step)
                        if name in {"return", "length"}:
                            self.writer.add_scalar(f"TrainInterval/{name}_std", std, self.global_step)
                    self.writer.add_scalar("TrainInterval/episodes", interval_count, self.global_step)
                    for values in self.interval_episodes.values():
                        values.clear()
                for name, values in self.collision_rolling.items():
                    self.writer.add_scalar(
                        f"CollisionEpisodes/{name}_last100_obstacle_collisions",
                        self._mean(values), self.global_step,
                    )
                if hasattr(self.env, "curriculum_metrics"):
                    for name, value in self.env.curriculum_metrics().items():
                        self.writer.add_scalar(f"Curriculum/{name}", value, self.global_step)
                print(f"step={absolute_step:6d}/{vector_steps} transitions={self.global_step:10d} "
                      f"eta={_format_seconds(eta)} fps={int(fps):6d} replay={self.buffer.size:8d} "
                      f"return1000={self._mean(self.rolling['return']):+8.3f}")

            if absolute_step % save_interval == 0:
                save_replay = save_replay_buffer and absolute_step % replay_save_interval == 0
                self.save(self.log_dir / f"model_step_{absolute_step}.pt", save_replay=save_replay,
                          keep_latest_replay_only=keep_latest_replay_only)

        final_path = self.log_dir / f"model_step_{self.vector_step}.pt"
        if not final_path.exists():
            self.save(final_path, save_replay=False, keep_latest_replay_only=keep_latest_replay_only)
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        summary = {
            "elapsed_seconds": elapsed, "elapsed_hhmmss": _format_seconds(elapsed),
            "vector_step_start": start_vector_step, "vector_step_end": self.vector_step,
            "target_vector_steps": vector_steps, "global_step_start": start_global_step,
            "global_step_end": self.global_step, "num_envs": self.env.num_envs,
            "run_transitions": self.global_step - start_global_step,
            "cumulative_episode_counts": dict(self.cumulative_outcomes),
            "original_curriculum_target": self.original_curriculum_target,
            "curriculum_replanned": self.curriculum_replanned,
        }
        (self.log_dir / "training_time_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if hasattr(self.env, "map_assignment_counts"):
            quarter_counts = self.env.quarter_difficulty_assignment_counts.cpu()
            quarter_totals = self.env.quarter_assignment_counts.cpu().clamp_min(1).unsqueeze(-1)
            if self.env.procedural_map_bank.difficulty_mode == "mixed":
                requested_weights = CURRICULUM_WEIGHTS
            else:
                one_hot = self.env._requested_difficulty_weights()
                requested_weights = (one_hot,) * 4
            exposure = {
                "map_assignment_counts": self.env.map_assignment_counts.cpu().tolist(),
                "scenario_assignment_counts": self.env.scenario_assignment_counts.cpu().tolist(),
                "difficulty_assignment_counts": self.env.difficulty_assignment_counts.cpu().tolist(),
                "quarter_difficulty_assignment_counts": self.env.quarter_difficulty_assignment_counts.cpu().tolist(),
                "quarter_assignment_counts": self.env.quarter_assignment_counts.cpu().tolist(),
                "quarter_realized_fractions_easy_medium_hard": (quarter_counts / quarter_totals).tolist(),
                "total_assignments": self.env.assignment_count,
                "unique_maps_seen": int((self.env.map_assignment_counts > 0).sum()),
                "unique_scenarios_seen": int((self.env.scenario_assignment_counts > 0).sum()),
                "requested_quarter_probabilities": [list(x) for x in requested_weights],
                "difficulty_mode": self.env.procedural_map_bank.difficulty_mode,
            }
            (self.log_dir / "map_exposure_summary.json").write_text(json.dumps(exposure, indent=2), encoding="utf-8")
        self.writer.flush()
