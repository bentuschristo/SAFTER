"""Play a frozen Unitree Go2 locomotion policy under a modular high-level controller."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from src.high_level_control.command_interface import MjlabVelocityCommandInterface
from src.high_level_control.config import HighLevelConfig
from src.high_level_control.controller import FeedbackLinearizationController
from src.high_level_control.logger import ExperimentLogger
from src.high_level_control.keyboard_controller import (
    KeyboardControllerConfig,
    TerminalKeyboardController,
)
from src.high_level_control.model import HighLevelKinematicModel
from src.high_level_control.policy_wrapper import HighLevelPolicyWrapper
from src.high_level_control.safety_filter import ClosedFormCircleCbf, IdentitySafetyFilter
from src.high_level_control.state_interface import MjlabHighLevelStateInterface
from src.high_level_control.trajectories import make_trajectory


@dataclass
class PlayHighLevelConfig:
    checkpoint_file: str
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    num_envs: int = 1
    no_terminations: bool = True
    high_level: HighLevelConfig = field(default_factory=HighLevelConfig)


def run(task_id: str, cfg: PlayHighLevelConfig) -> None:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = cfg.num_envs
    if cfg.no_terminations:
        env_cfg.terminations = {}
    if "push_robot" in env_cfg.events:
        del env_cfg.events["push_robot"]

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    locomotion_policy = runner.get_inference_policy(device=device)

    state_if = MjlabHighLevelStateInterface(env)
    command_if = MjlabVelocityCommandInterface(env, command_name="twist")
    model = HighLevelKinematicModel(cfg.high_level.model)
    initial_pose = model.controlled_pose(state_if.read().base_pose_world)
    trajectory = make_trajectory(cfg.high_level, initial_pose)
    controller = FeedbackLinearizationController(model, cfg.high_level.controller)

    if cfg.high_level.safety.enabled:
        safety_filter = ClosedFormCircleCbf(
            model=model,
            obstacle=cfg.high_level.obstacle,
            cfg=cfg.high_level.safety,
        )
    else:
        safety_filter = IdentitySafetyFilter()

    keyboard_controller = None
    if cfg.high_level.control_mode == "keyboard":
        c = cfg.high_level.controller
        k = cfg.high_level.keyboard
        keyboard_controller = TerminalKeyboardController(
            KeyboardControllerConfig(
                vx_speed=k.vx_speed,
                vy_speed=k.vy_speed,
                wz_speed=k.wz_speed,
                release_timeout=k.release_timeout,
                min_vx=c.min_vx,
                max_vx=c.max_vx,
                min_vy=c.min_vy,
                max_vy=c.max_vy,
                min_wz=c.min_wz,
                max_wz=c.max_wz,
            )
        )

    logger = None
    if cfg.high_level.logging.enabled:
        logger = ExperimentLogger(
            root_dir=cfg.high_level.logging.root_dir,
            run_name=cfg.high_level.logging.run_name or cfg.high_level.trajectory,
            flush_every=cfg.high_level.logging.flush_every,
            metadata={
                "task_id": task_id,
                "checkpoint": str(checkpoint),
                "device": device,
                "configuration": asdict(cfg),
            },
        )

    policy = HighLevelPolicyWrapper(
        env=env,
        locomotion_policy=locomotion_policy,
        state_interface=state_if,
        command_interface=command_if,
        model=model,
        controller=controller,
        trajectory=trajectory,
        safety_filter=safety_filter,
        keyboard_controller=keyboard_controller,
        high_level_rate_hz=cfg.high_level.high_level_rate_hz,
        command_obs_start=cfg.high_level.command_obs_start,
        logger=logger,
        trajectory_name=cfg.high_level.trajectory,
    )

    if cfg.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        viewer = "native" if has_display else "viser"
    else:
        viewer = cfg.viewer

    try:
        if viewer == "native":
            NativeMujocoViewer(env, policy).run()
        else:
            ViserPlayViewer(env, policy).run()
    finally:
        if keyboard_controller is not None:
            keyboard_controller.close()
        if logger is not None:
            logger.close()
            if cfg.high_level.logging.plot_on_exit:
                try:
                    from src.high_level_control.plotter import plot_log
                    paths = plot_log(logger.csv_path)
                    print(f"[PLOTTER] Saved {len(paths)} plots to: {logger.run_dir / 'plots'}")
                except Exception as exc:
                    print(f"[PLOTTER] Failed: {exc}")
        env.close()


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(
        PlayHighLevelConfig,
        args=remaining,
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    run(chosen_task, cfg)


if __name__ == "__main__":
    main()
