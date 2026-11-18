"""Train the CBF-RL PPO high-level navigation policy.

Reimplements Yang et al., "CBF-RL: Safety Filtering Reinforcement Learning
in Training with Control Barrier Functions", using this repo's elliptical
LiDAR CBF (src.high_level_rl.laser_cbf_filters) with the same footprint and
margin fields as the laser SAC pipeline, trained with rsl_rl PPO.

Collision termination follows the SAC training environment:
--collision-termination-mode direct_contact (default, MuJoCo body-contact
forces) or circle (geometric footprint overlap).

Reward profiles:
  --reward-profile original            : Yang et al. reward, 50 Hz by default
  --reward-profile new_reward_profile  : laser-SAC navigation reward plus the
                                         scaled CBF terms, 10 Hz by default

Ablation modes (Yang et al. Table III / cbf-rl-demo's four shell scripts):
  --use-cbf-action-filtering --use-cbf-reward-penalty     : Dual (the paper's proposed method)
  --use-cbf-action-filtering --no-use-cbf-reward-penalty  : Filter-only
  --no-use-cbf-action-filtering --use-cbf-reward-penalty  : Reward-only
  --no-use-cbf-action-filtering --no-use-cbf-reward-penalty : Nominal

By default the environment uses the fixed 9-obstacle arena in
HighLevelNavigationConfig.arena. Pass --procedural-map-bank to train on a
saved procedural map bank instead, as in SAFTER_scripts/train_SAC_policies.py.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import mujoco
import torch
import tyro

# SAFTER_src is not part of the editable install, so add the repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mjlab
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

from src.high_level_cbf_rl.cbf_rl_navigation_env import CbfRlNavigationEnv
from src.high_level_cbf_rl.config import (
    CBF_RL_HIGH_LEVEL_DT,
    CbfRlConfig,
    CbfRlModeConfig,
    CbfRlRewardProfile,
    default_high_level_hz,
    with_high_level_rate,
    with_reward_profile,
)
from src.high_level_control.command_interface import MjlabVelocityCommandInterface
try:
    from SAFTER_src.high_level_laser_sac.config import CollisionTerminationMode
    from SAFTER_src.high_level_laser_sac.procedural_maps import (
        CURRICULUM_WEIGHTS,
        ProceduralMapBank,
        load_procedural_map_bank,
        resolve_slot_shapes,
    )
except ModuleNotFoundError:
    # Fallback for layouts without SAFTER_src/.
    from src.high_level_laser_sac.config import CollisionTerminationMode
    from src.high_level_laser_sac.procedural_maps import (
        CURRICULUM_WEIGHTS,
        ProceduralMapBank,
        load_procedural_map_bank,
        resolve_slot_shapes,
    )
from src.high_level_rl.low_level import load_frozen_low_level_policy
from SAFTER_scripts.train_SAC_policies import add_navigation_laser_sensor


def make_walls_only_arena_cfg(arena) -> EntityCfg:
    """Physical walls matching the active procedural map bank's arena size."""
    half_x = 0.5 * float(arena.width)
    half_y = 0.5 * float(arena.length)
    half_t = 0.5 * float(arena.wall_thickness)
    half_h = 0.5 * float(arena.wall_height)
    x_center = half_x + half_t
    y_center = half_y + half_t
    wall_rgba = "0.25 0.28 0.32 1"
    wall_common = (
        f'rgba="{wall_rgba}" contype="1" conaffinity="1" '
        'condim="3" friction="0.8 0.05 0.001" group="0"'
    )
    xml = f"""
    <mujoco model="procedural_navigation_walls">
      <worldbody>
        <body name="wall_left" pos="{-x_center} 0 {half_h}">
          <geom name="wall_left_collision" type="box"
                size="{half_t} {half_y + float(arena.wall_thickness)} {half_h}" {wall_common}/>
        </body>
        <body name="wall_right" pos="{x_center} 0 {half_h}">
          <geom name="wall_right_collision" type="box"
                size="{half_t} {half_y + float(arena.wall_thickness)} {half_h}" {wall_common}/>
        </body>
        <body name="wall_bottom" pos="0 {-y_center} {half_h}">
          <geom name="wall_bottom_collision" type="box"
                size="{half_x + float(arena.wall_thickness)} {half_t} {half_h}" {wall_common}/>
        </body>
        <body name="wall_top" pos="0 {y_center} {half_h}">
          <geom name="wall_top_collision" type="box"
                size="{half_x + float(arena.wall_thickness)} {half_t} {half_h}" {wall_common}/>
        </body>
      </worldbody>
    </mujoco>
    """
    return EntityCfg(spec_fn=lambda xml=xml: mujoco.MjSpec.from_string(xml))


def make_procedural_obstacle_slot_entity(slot: int, obstacle_shape: str = "circle") -> EntityCfg:
    """One reusable mocap obstacle whose per-world pose/size change at reset."""
    if obstacle_shape not in ("circle", "square"):
        raise ValueError(f"Unsupported procedural obstacle shape: {obstacle_shape!r}")
    geom_type = "box" if obstacle_shape == "square" else "cylinder"
    initial_size = "0.0001 0.0001 0.0001" if geom_type == "box" else "0.0001 0.0001"
    xml = f"""
    <mujoco model="procedural_obstacle_slot_{slot:02d}">
      <worldbody>
        <body name="procedural_obstacle_slot_{slot:02d}_root" mocap="true" pos="0 0 -100">
          <geom name="procedural_obstacle_slot_{slot:02d}_collision"
                type="{geom_type}" size="{initial_size}"
                rgba="0.85 0.15 0.08 1" contype="1" conaffinity="1"
                condim="3" friction="0.8 0.05 0.001"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return EntityCfg(
        spec_fn=lambda xml=xml: mujoco.MjSpec.from_string(xml),
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
    )


@dataclass
class TrainCbfRlConfig:
    low_level_checkpoint: str
    num_envs: int = 256
    device: str | None = None
    max_iterations: int = 1500
    save_interval: int = 100
    experiment_name: str = "go2_cbf_rl"
    run_name: str = ""
    seed: int = 42
    resume_checkpoint: str | None = None
    use_cbf_action_filtering: bool = True
    use_cbf_reward_penalty: bool = True
    # "original": Yang et al. reward. "new_reward_profile": laser-SAC
    # navigation reward (plus the CBF-RL terms when use_cbf_reward_penalty).
    reward_profile: CbfRlRewardProfile = "original"
    # Policy/LiDAR/filter rate. None: 50 Hz for original, 10 Hz (laser-SAC
    # rate, 5 low-level steps per command) for new_reward_profile.
    high_level_hz: float | None = None
    # Training termination, as in train_SAC_policies.py.
    collision_termination_mode: CollisionTerminationMode = "direct_contact"
    # LiDAR overrides with the same names as train_SAC_policies.py's CLI.
    laser_num_rays: int = 61
    laser_fov_deg: float = 180.0
    laser_max_range: float = 5.0
    enable_lidar_noise: bool = False
    # None: fixed HighLevelNavigationConfig.arena; otherwise a saved procedural map bank.
    procedural_map_bank: str | None = None
    expected_num_maps: int | None = 100
    expected_scenarios_per_map: int | None = 64
    expected_max_obstacles: int | None = 12
    curriculum_recent_window: int = 10_000
    curriculum_mode: Literal["quarters", "seanav_adaptive"] = "quarters"
    config: CbfRlConfig = field(default_factory=CbfRlConfig)


def run(task_id: str, cfg: TrainCbfRlConfig) -> None:
    configure_torch_backends()
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    high_level_hz = cfg.high_level_hz or default_high_level_hz(cfg.reward_profile)
    cfg = replace(
        cfg,
        config=with_high_level_rate(with_reward_profile(cfg.config, cfg.reward_profile), high_level_hz),
    )

    procedural_map_bank: ProceduralMapBank | None = None
    procedural_entity_names: tuple[str, ...] | None = None
    if cfg.procedural_map_bank is not None:
        procedural_map_bank = load_procedural_map_bank(
            cfg.procedural_map_bank,
            expected_num_maps=cfg.expected_num_maps,
            expected_scenarios_per_map=cfg.expected_scenarios_per_map,
            expected_max_obstacles=cfg.expected_max_obstacles,
        )
        procedural_maps = procedural_map_bank.maps
        slot_shapes = resolve_slot_shapes(procedural_maps, procedural_map_bank.max_obstacles)
        procedural_entity_names = tuple(
            f"procedural_obstacle_slot_{slot:02d}" for slot in range(procedural_map_bank.max_obstacles)
        )

    env_cfg = load_env_cfg(task_id, play=False)
    low_agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    # High-level code handles episode endings and reset sampling.
    env_cfg.terminations = {}
    if "push_robot" in env_cfg.events:
        del env_cfg.events["push_robot"]

    navigation_cfg = cfg.config.navigation
    if procedural_map_bank is not None:
        bank_arena = procedural_map_bank.maps[0].arena
        if any(
            (item.arena.width, item.arena.length, item.arena.wall_thickness, item.arena.wall_height)
            != (bank_arena.width, bank_arena.length, bank_arena.wall_thickness, bank_arena.wall_height)
            for item in procedural_map_bank.maps
        ):
            raise ValueError("All maps in one procedural bank must share arena wall geometry")
        env_cfg.scene.entities["navigation_arena"] = make_walls_only_arena_cfg(bank_arena)
        for slot, entity_name in enumerate(procedural_entity_names):
            env_cfg.scene.entities[entity_name] = make_procedural_obstacle_slot_entity(slot, slot_shapes[slot])
        # Obstacle tensors are sized from arena.max_obstacles, which must equal
        # the bank's slot count; obstacle poses come from the mocap slots.
        arena_template = procedural_map_bank.maps[0].arena
        empty_arena_template = replace(arena_template, obstacles=())
        for item in procedural_map_bank.maps[1:]:
            if replace(item.arena, obstacles=()) != empty_arena_template:
                raise ValueError("All procedural maps must use matching arena configuration")
        navigation_cfg = replace(navigation_cfg, arena=empty_arena_template)

    training_laser_cfg = replace(
        cfg.config.laser,
        num_rays=int(cfg.laser_num_rays),
        fov_deg=float(cfg.laser_fov_deg),
        max_range=float(cfg.laser_max_range),
        enable_lidar_noise=bool(cfg.enable_lidar_noise),
    )
    method_cfg = replace(
        cfg.config,
        navigation=navigation_cfg,
        laser=training_laser_cfg,
        contact_collision=replace(
            cfg.config.contact_collision, termination_mode=cfg.collision_termination_mode
        ),
        mode=CbfRlModeConfig(
            use_cbf_action_filtering=cfg.use_cbf_action_filtering,
            use_cbf_reward_penalty=cfg.use_cbf_reward_penalty,
        ),
    )
    add_navigation_laser_sensor(env_cfg, method_cfg)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    if procedural_map_bank is not None:
        # MuJoCo-Warp supports these model fields per environment, so the
        # reusable obstacle slots can preserve each saved map's exact radii.
        raw_env.sim.expand_model_fields(("geom_size", "geom_rbound", "geom_aabb"))
    low_env = RslRlVecEnvWrapper(raw_env, clip_actions=low_agent_cfg.clip_actions)
    low_policy = load_frozen_low_level_policy(task_id, low_env, cfg.low_level_checkpoint, device)
    command_if = MjlabVelocityCommandInterface(low_env, command_name="twist")

    env = CbfRlNavigationEnv(
        low_env,
        low_policy,
        command_if,
        method_cfg,
        procedural_map_bank=procedural_map_bank,
        procedural_map_entity_names=procedural_entity_names,
        curriculum_recent_window=cfg.curriculum_recent_window,
        curriculum_mode=cfg.curriculum_mode,
    )
    if procedural_map_bank is not None:
        env.assert_procedural_geometry_synchronized()

    # Start from the low-level task's PPO defaults, then set CBF-RL hyperparameters.
    agent_cfg = load_rl_cfg(task_id)
    agent_cfg.experiment_name = cfg.experiment_name
    agent_cfg.run_name = cfg.run_name
    agent_cfg.seed = cfg.seed
    agent_cfg.max_iterations = cfg.max_iterations
    agent_cfg.save_interval = cfg.save_interval
    # PPO hyperparameters from the CBF-RL reference implementation; a small MLP
    # suffices for the goal + LiDAR + previous-command observation.
    agent_cfg.num_steps_per_env = 48
    agent_cfg.actor.hidden_dims = (128, 128)
    agent_cfg.critic.hidden_dims = (128, 128)
    agent_cfg.algorithm.gamma = 0.99
    agent_cfg.algorithm.lam = 0.95
    agent_cfg.algorithm.learning_rate = 3.0e-4
    agent_cfg.algorithm.clip_param = 0.2
    agent_cfg.algorithm.num_learning_epochs = 10
    agent_cfg.algorithm.num_mini_batches = 8

    mode_tag = (
        "dual" if (cfg.use_cbf_action_filtering and cfg.use_cbf_reward_penalty) else
        "filter_only" if cfg.use_cbf_action_filtering else
        "reward_only" if cfg.use_cbf_reward_penalty else
        "nominal"
    )
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    variant_tag = ""
    if cfg.reward_profile != "original":
        variant_tag += f"_{cfg.reward_profile}"
    if abs(high_level_hz - 1.0 / CBF_RL_HIGH_LEVEL_DT) > 1.0e-6:
        variant_tag += f"_{high_level_hz:g}hz"
    suffix = f"_{cfg.run_name}" if cfg.run_name else f"_{mode_tag}{variant_tag}_seed{cfg.seed}"
    log_dir = Path("logs/rsl_rl") / cfg.experiment_name / f"{stamp}{suffix}"
    log_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(log_dir / "params" / "navigation.yaml", asdict(method_cfg.navigation))
    # Save the full method config so each run is reproducible.
    dump_yaml(log_dir / "params" / "cbf_rl.yaml", asdict(method_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))
    if procedural_map_bank is not None:
        dump_yaml(
            log_dir / "params" / "procedural_map_bank.yaml",
            {
                "bank": str(Path(cfg.procedural_map_bank).expanduser().resolve()),
                "num_maps": len(procedural_map_bank.maps),
                "num_scenarios": procedural_map_bank.num_scenarios,
                "max_obstacles": procedural_map_bank.max_obstacles,
                "curriculum_mode": cfg.curriculum_mode,
                "quarter_weights_easy_medium_hard": CURRICULUM_WEIGHTS,
            },
        )

    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device)

    if cfg.resume_checkpoint is not None:
        checkpoint = Path(cfg.resume_checkpoint).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"High-level checkpoint not found: {checkpoint}")
        print(f"[CBF-RL] Resuming from {checkpoint}")
        runner.load(str(checkpoint))

    print(
        f"[CBF-RL] mode={mode_tag} reward_profile={method_cfg.reward.profile} "
        f"cbf_reward_scale={env.cbf_reward_scale:g} "
        f"termination={method_cfg.contact_collision.termination_mode} "
        f"margin(footprint_clearance_m)={method_cfg.safety.footprint_clearance_m:.5f}m "
        f"(from safety_clearance={method_cfg.safety.d_safe:.2f}m) "
        f"high_level_hz={1.0 / method_cfg.navigation.high_level_dt:g} num_envs={cfg.num_envs}"
    )
    if procedural_map_bank is not None:
        print(
            f"[CBF-RL] procedural_map_bank={cfg.procedural_map_bank} "
            f"maps={len(procedural_map_bank.maps)} scenarios={procedural_map_bank.num_scenarios} "
            f"max_obstacles={procedural_map_bank.max_obstacles}"
        )
    else:
        print("[CBF-RL] using fixed HighLevelNavigationConfig.arena (no procedural map bank)")
    print(f"[CBF-RL] Logging to: {log_dir}")
    runner.learn(num_learning_iterations=cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(TrainCbfRlConfig, args=remaining, prog=sys.argv[0] + f" {task}", config=mjlab.TYRO_FLAGS)
    run(task, cfg)


if __name__ == "__main__":
    main()
