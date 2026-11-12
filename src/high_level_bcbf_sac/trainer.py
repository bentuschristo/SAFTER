from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .config import SacBcbfCfConfig
from .networks import SafeSacActor, TwinQCritic
from .replay_buffer import VectorReplayBuffer


def _format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


class SacBcbfCfTrainer:
    def __init__(self, env, cfg: SacBcbfCfConfig, log_dir: Path, device: str) -> None:
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        obs = self.actor_obs(env.get_observations())
        obs_dim = int(obs.shape[-1])
        nav = env.navigation_cfg
        hp = cfg.sac

        # All vectorized environments share the same fixed obstacle map.
        centers = env.obstacle_centers[:1]
        radii = env.obstacle_radii[:1]
        mask = env.obstacle_mask[:1]
        self.actor = SafeSacActor(
            obs_dim=obs_dim,
            hidden_dims=hp.hidden_dims,
            arena=nav.arena,
            action_cfg=nav.action,
            safety_cfg=nav.backup_safety,
            obstacle_centers=centers,
            obstacle_radii=radii,
            obstacle_mask=mask,
            log_std_min=hp.log_std_min,
            log_std_max=hp.log_std_max,
            filter_kind=cfg.filter_kind,
        ).to(self.device)
        self.critic = TwinQCritic(obs_dim, 3, hp.hidden_dims).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
        self.log_alpha = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=hp.alpha_lr)
        self.buffer = VectorReplayBuffer(hp.replay_size, obs_dim, 3, self.device)
        self.global_step = 0
        self.gradient_step = 0

        (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
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
        print("\n+------------------------------------------------------------+")
        print("| SAC + differentiable zero-backup bCBF-LSE-CF policy    |")
        print("+-------------------------------+----------------------------+")
        print(f"| Observation dimension         | {obs_dim:<26}|")
        print(f"| Nominal action dimension      | {3:<26}|")
        print(f"| Critic action                 | {'safe [vx, vy, wz]':<26}|")
        print(f"| Safety gradient path          | {'enabled':<26}|")
        print("+-------------------------------+----------------------------+")
        print("\nReward table")
        rows = [
            ("progress", r.progress_weight), ("goal", r.goal_bonus),
            ("time", -r.time_penalty), ("timeout", -r.timeout_penalty),
            ("forward_alignment", r.forward_alignment_weight),
            ("reverse_motion", -r.reverse_motion_weight), ("yaw_tracking", -r.yaw_tracking_weight),
            ("smoothness", -r.smoothness_weight), ("cbf_correction", -r.cbf_correction_weight),
            ("collision", -r.collision_penalty),
        ]
        for name, weight in rows:
            print(f"  {name:<24} {weight:+.4f}")
        print("  fall                    +0.0000 (termination diagnostic only)")
        print()

    def _soft_update(self) -> None:
        tau = self.cfg.sac.tau
        with torch.no_grad():
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters(), strict=True):
                target.mul_(1.0 - tau).add_(source, alpha=tau)

    def update(self) -> dict[str, float]:
        hp = self.cfg.sac
        batch = self.buffer.sample(hp.batch_size)

        with torch.no_grad():
            next_out = self.actor.sample(batch.next_obs, batch.next_position, batch.next_yaw)
            target_q1, target_q2 = self.target_critic(batch.next_obs, next_out.safe_normalized)
            target_v = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_out.log_prob
            target = batch.reward + (1.0 - batch.done) * hp.gamma * target_v

        q1, q2 = self.critic(batch.obs, batch.action)
        critic_loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), hp.gradient_clip_norm)
        self.critic_opt.step()

        actor_out = self.actor.sample(batch.obs, batch.position, batch.yaw)
        actor_q1, actor_q2 = self.critic(batch.obs, actor_out.safe_normalized)
        actor_q = torch.minimum(actor_q1, actor_q2)
        actor_loss = (self.alpha.detach() * actor_out.log_prob - actor_q).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), hp.gradient_clip_norm)
        self.actor_opt.step()

        alpha_loss = -(
            self.log_alpha * (actor_out.log_prob.detach() + hp.target_entropy)
        ).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update()
        self.gradient_step += 1
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "q": float(actor_q.mean().detach()),
            "entropy_proxy": float((-actor_out.log_prob).mean().detach()),
            "actor_cbf_active": float(actor_out.safety.active.float().mean().detach()),
            "actor_cbf_correction": float(actor_out.safety.correction_norm.mean().detach()),
        }

    def save(self, path: Path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_opt.state_dict(),
                "critic_optimizer": self.critic_opt.state_dict(),
                "log_alpha": self.log_alpha.detach(),
                "alpha_optimizer": self.alpha_opt.state_dict(),
                "global_step": self.global_step,
                "gradient_step": self.gradient_step,
                "config": asdict(self.cfg),
            },
            path,
        )

    def load(self, path: Path, load_optimizers: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])
            self.target_critic.load_state_dict(checkpoint.get("target_critic", checkpoint["critic"]))
        if load_optimizers:
            self.actor_opt.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_opt.load_state_dict(checkpoint["critic_optimizer"])
            self.log_alpha.data.copy_(checkpoint["log_alpha"])
            self.alpha_opt.load_state_dict(checkpoint["alpha_optimizer"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.gradient_step = int(checkpoint.get("gradient_step", 0))

    def train(self, vector_steps: int, save_interval: int = 1000, log_interval: int = 10) -> None:
        hp = self.cfg.sac
        obs = self.actor_obs(self.env.get_observations()).to(self.device)
        start_time = time.perf_counter()
        start_global_step = int(self.global_step)
        episode_returns: list[float] = []
        episode_lengths: list[float] = []
        episode_successes: list[float] = []
        episode_collisions: list[float] = []

        for local_step in range(1, vector_steps + 1):
            position, yaw = self.env.safety_context()
            position, yaw = position.to(self.device), yaw.to(self.device)

            if self.buffer.size < hp.learning_starts:
                nominal_normalized = torch.empty((self.env.num_envs, 3), device=self.device).uniform_(-1.0, 1.0)
                safe_norm, safe_cmd, safety = self.actor.apply_safety(
                    nominal_normalized,
                    position,
                    yaw,
                    obs[:, 5:8],
                    obs[:, -4:-1],
                    obs[:, -1:],
                )
                nominal_cmd = self.actor.action_center + nominal_normalized * self.actor.action_half_range
            else:
                with torch.no_grad():
                    out = self.actor.sample(obs, position, yaw)
                nominal_normalized = out.nominal_normalized
                nominal_cmd, safe_norm, safe_cmd, safety = (
                    out.nominal_command, out.safe_normalized, out.safe_command, out.safety
                )

            next_obs_dict, reward, done, extras = self.env.step_safe(nominal_cmd, safe_cmd, safety)
            next_obs = self.actor_obs(next_obs_dict).to(self.device)
            replay_next_obs = next_obs
            next_position, next_yaw = self.env.safety_context()
            time_outs = extras.get("time_outs")
            if time_outs is not None:
                terminal_obs = self.actor_obs(extras["terminal_observation"]).to(self.device)
                timeout_mask = time_outs.to(self.device).bool().unsqueeze(-1)
                replay_next_obs = torch.where(timeout_mask, terminal_obs, next_obs)
                next_position = torch.where(
                    timeout_mask,
                    extras["terminal_position"].to(self.device),
                    next_position.to(self.device),
                )
                next_yaw = torch.where(
                    time_outs.to(self.device).bool(),
                    extras["terminal_yaw"].to(self.device),
                    next_yaw.to(self.device),
                )
            else:
                next_position = next_position.to(self.device)
                next_yaw = next_yaw.to(self.device)
            terminal_mask = done.detach()
            if time_outs is not None:
                terminal_mask = terminal_mask & ~time_outs.to(done.device).bool()

            self.buffer.add_batch(
                obs=obs,
                action=safe_norm.detach(),
                reward=reward.detach(),
                next_obs=replay_next_obs,
                done=terminal_mask.detach(),
                position=position,
                yaw=yaw,
                next_position=next_position,
                next_yaw=next_yaw,
            )
            obs = next_obs
            self.global_step += self.env.num_envs

            completed_returns = extras.get("completed_returns")
            if completed_returns is not None and completed_returns.numel() > 0:
                episode_returns.extend(completed_returns.cpu().tolist())
                episode_lengths.extend(extras["completed_lengths"].float().cpu().tolist())
                episode_successes.extend(extras["completed_success"].float().cpu().tolist())
                episode_collisions.extend(extras["completed_collision"].float().cpu().tolist())

            update_metrics: dict[str, float] = {}
            if self.buffer.size >= max(hp.learning_starts, hp.batch_size):
                for _ in range(hp.updates_per_step):
                    update_metrics = self.update()

            for name, value in extras["log"].items():
                self.writer.add_scalar(f"Episode/{name}", float(value), self.global_step)
            for name, value in update_metrics.items():
                self.writer.add_scalar(f"SAC/{name}", value, self.global_step)

            if local_step % log_interval == 0:
                elapsed = max(time.perf_counter() - start_time, 1.0e-6)
                run_transitions = max(int(self.global_step) - start_global_step, 0)
                fps_float = run_transitions / elapsed
                fps = int(fps_float)
                remaining_transitions = max(vector_steps - local_step, 0) * self.env.num_envs
                eta_seconds = remaining_transitions / max(fps_float, 1.0e-6)
                mean_return = sum(episode_returns[-100:]) / max(len(episode_returns[-100:]), 1)
                success_rate = sum(episode_successes[-100:]) / max(len(episode_successes[-100:]), 1)
                collision_rate = sum(episode_collisions[-100:]) / max(len(episode_collisions[-100:]), 1)
                mean_length = sum(episode_lengths[-100:]) / max(len(episode_lengths[-100:]), 1)
                self.writer.add_scalar("Train/wall_time_seconds", elapsed, self.global_step)
                self.writer.add_scalar("Train/wall_time_hours", elapsed / 3600.0, self.global_step)
                self.writer.add_scalar("Train/fps", fps_float, self.global_step)
                self.writer.add_scalar("Train/eta_seconds", eta_seconds, self.global_step)
                self.writer.add_scalar("Train/run_transitions", run_transitions, self.global_step)
                self.writer.add_scalar("Train/mean_episode_return_100", mean_return, self.global_step)
                self.writer.add_scalar("Train/episode_success_rate_100", success_rate, self.global_step)
                self.writer.add_scalar("Train/episode_collision_rate_100", collision_rate, self.global_step)
                self.writer.add_scalar("Train/mean_episode_length_100", mean_length, self.global_step)
                print(
                    f"step={local_step:6d}/{vector_steps} transitions={self.global_step:10d} "
                    f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta_seconds)} "
                    f"fps={fps:6d} replay={self.buffer.size:8d} "
                    f"return100={mean_return:+8.3f} success100={success_rate:6.3f} "
                    f"collision100={collision_rate:6.3f} alpha={float(self.alpha.detach()):.3f}"
                )

            if local_step % save_interval == 0:
                self.save(self.log_dir / f"model_step_{local_step}.pt")

        self.save(self.log_dir / f"model_step_{vector_steps}.pt")
        total_elapsed = max(time.perf_counter() - start_time, 1.0e-6)
        run_transitions = max(int(self.global_step) - start_global_step, 0)
        summary = {
            "elapsed_seconds": total_elapsed,
            "elapsed_hours": total_elapsed / 3600.0,
            "elapsed_hhmmss": _format_seconds(total_elapsed),
            "run_transitions": run_transitions,
            "global_step_start": start_global_step,
            "global_step_end": int(self.global_step),
            "vector_steps_requested": int(vector_steps),
            "num_envs": int(self.env.num_envs),
            "average_fps": run_transitions / total_elapsed,
        }
        (self.log_dir / "training_time_summary.json").write_text(json.dumps(summary, indent=2))
        self.writer.add_scalar("Train/total_wall_time_seconds", total_elapsed, self.global_step)
        self.writer.add_scalar("Train/total_wall_time_hours", total_elapsed / 3600.0, self.global_step)
        self.writer.flush()
