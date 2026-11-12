"""Train high-level SAC variants with laser-range observations.

Methods:
  vanilla      : ordinary SAC baseline, no safety filter in training path.
  cbf_cf_clip  : direct laser LSE-CBF-CF with command clipping.
  cbf_qp       : direct laser QP with one constraint per valid ray.
  cbf_lse_qp   : direct laser QP with one LSE ray constraint plus six bounds.
  cbf_cd_cf    : laser LSE-CBF-CF with command dynamics and input barriers.
  cbf_lse_cd_qp: QP synthesis of the same one-row LSE-CD aggregate as cbf_cd_cf.
  bcbf_cf      : laser zero-backup LSE-bCBF-CF with command dynamics.
  bcbf_qp      : laser zero-backup QP with one backup LSE row plus six input rows.
  bcbf_lse_cd_qp: QP synthesis of the full bCBF-LSE-CD aggregate used by bcbf_cf.
  bcbf_nonlse_cd_qp: full laser bCBF-CD-QP with every horizon/ray and input row explicit.
  bcbf_lse_cd_qp_cfmatched: same single-row LSE aggregate as bcbf_lse_cd_qp,
    solved with QpthProjector's exact differentiable QP/KKT (no
    straight-through estimator) instead of the training-time active-set
    backend -- an apples-to-apples ablation against bcbf_cf's closed form.
"""
from __future__ import annotations

import sys
import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tyro
import mujoco

import mjlab
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from SAFTER_src.high_level_control.command_interface import MjlabVelocityCommandInterface
from SAFTER_src.high_level_laser_sac.config import (
    CollisionGeometry,
    CollisionTerminationMode,
    LaserSacConfig,
    LaserSacMethod,
    QpBackend,
)
from SAFTER_src.high_level_laser_sac.procedural_maps import (
    resolve_slot_shapes,
    CURRICULUM_WEIGHTS,
    load_procedural_map_bank,
)
from SAFTER_src.high_level_laser_sac.sac_navigation_env import LaserSacNavigationEnv
from SAFTER_src.high_level_laser_sac.trainer import LaserSacTrainer
from SAFTER_src.high_level_rl.low_level import load_frozen_low_level_policy
from SAFTER_src.high_level_rl.raycast_patterns import HorizontalRingPatternCfg
from src.tasks.velocity.config.go2 import obstacle_cfg as physical_arena_cfg


@dataclass
class TrainLaserSacConfig:
    low_level_checkpoint: str
    method: LaserSacMethod = "vanilla"
    num_envs: int = 256
    device: str | None = None
    vector_steps: int = 400_000
    save_interval: int = 1_000
    log_interval: int = 10
    seed: int = 0
    run_name: str = ""
    resume_checkpoint: str | None = None
    enable_lidar_noise: bool = True
    laser_num_rays: int = 61
    laser_fov_deg: float = 180.0
    laser_max_range: float = 5.0
    safety_clearance: float = 0.30
    collision_geometry: CollisionGeometry = "ellipse"
    collision_circle_radius_m: float = 0.30
    collision_termination_mode: CollisionTerminationMode = "seanav_contact"
    collision_margin_m: float = 0.05
    hard_clip_output: bool = True
    qp_backend: QpBackend = "active_set"
    # Original three-hidden-layer SAC MLP. Capacity ablations can override this
    # with a comma-separated CLI value.
    # Comma-separated because the task launcher performs two-stage CLI parsing;
    # a variable-length tuple would lose all but its first value in stage one.
    hidden_dims: str = "512,256,128"
    procedural_map_bank: str = "Maps/Training Maps/handcrafted2"
    save_replay_buffer: bool = False
    replay_save_interval: int = 50_000
    keep_latest_replay_only: bool = True
    allow_curriculum_replan: bool = False
    expected_num_maps: int | None = 5
    expected_scenarios_per_map: int | None = 200
    curriculum_recent_window: int = 10_000
    curriculum_mode: Literal["quarters", "seanav_adaptive"] = "quarters"
    enable_collision_initialization: bool = False
    # Lightweight, off-by-default training-time diagnostics for the
    # invariant-ellipse terminal backup set (min E_i/h_B_geom/h_B_rec,
    # D_aug, Omega, correction norm, intervention rate, terminal-dominant-ray
    # fraction, and near-degenerate/NaN/fallback event counts), aggregated
    # over each log_interval and written under the "BcbfDiag/" TensorBoard
    # tag. Only meaningful for --method bcbf_cf with
    # --config.safety.terminal-backup-mode invariant_ellipse; a warning is
    # printed and nothing is logged otherwise.
    enable_bcbf_diagnostics: bool = False
    collision_initialization_probability: float = 0.80
    collision_undo_time_min_s: float = 2.0
    collision_undo_time_max_s: float = 3.0
    config: LaserSacConfig = field(default_factory=LaserSacConfig)
    # The default deliberately preserves the archived v5 behavior. Only the
    # dedicated SEA-Nav launcher overrides this field.
    training_profile: Literal[
        "original", "seanav_comparable_10hz", "seanav_comparable_50hz"
    ] = "original"


def apply_training_profile(cfg: TrainLaserSacConfig) -> TrainLaserSacConfig:
    """Resolve an isolated timing/reward profile without mutating defaults."""
    if cfg.training_profile == "original":
        return cfg
    if cfg.training_profile not in ("seanav_comparable_10hz", "seanav_comparable_50hz"):
        raise ValueError(f"Unknown training profile {cfg.training_profile!r}")

    if cfg.training_profile == "seanav_comparable_10hz":
        reward = replace(cfg.config.reward, profile="seanav_comparable")
        # Preserve the established hierarchy: policy/LiDAR/filter at 10 Hz,
        # with five frozen low-level controller steps at 50 Hz. SEA-Nav reward
        # scales are multiplied by this 0.10 s policy interval in the env.
        # curriculum_mode is intentionally left as whatever the caller/CLI
        # already set (default "quarters", the static progress-based ramp
        # used elsewhere) rather than forced to "seanav_adaptive" here, so
        # `--curriculum-mode` controls this independently of the reward/timing
        # profile.
        return replace(
            cfg,
            config=replace(cfg.config, reward=reward),
        )

    # SEA-Nav acts every 4 * 0.005 s = 0.02 s. Our frozen low-level policy is
    # already evaluated at 0.02 s, so one low-level step is one policy step.
    navigation = replace(cfg.config.navigation, high_level_dt=0.02)

    # These gains are continuous-time rates [1/s], so they remain 8. The
    # integration interval becomes 0.02 s. A 4 s horizon, 21 samples, and 10
    # substeps also gives an exact 0.02 s internal backup-flow step.
    safety = replace(
        cfg.config.safety,
        integration_dt=0.02,
        backup_rollout_substeps=10,
    )
    # Preserve the existing controlled-comparison 50 Hz profile, including its
    # legacy shaping terms, so old commands and checkpoints remain compatible.
    reward = replace(cfg.config.reward, profile="seanav_comparable")
    return replace(
        cfg,
        config=replace(cfg.config, navigation=navigation, safety=safety, reward=reward),
    )


def make_walls_only_arena_cfg(arena=None) -> EntityCfg:
    """Create physical walls whose inner faces match the active map bank.

    Procedural maps are map-local in every MuJoCo-Warp world.  Building these
    walls from the bank arena (instead of the legacy fixed Go2 arena) keeps
    wall LiDAR returns, workspace termination, and analytical CBF geometry on
    the same coordinates.  The optional fallback preserves old callers.
    """
    if arena is None:
        original_obstacles = physical_arena_cfg.OBSTACLES
        try:
            physical_arena_cfg.OBSTACLES = ()
            xml = physical_arena_cfg._make_arena_xml()
        finally:
            physical_arena_cfg.OBSTACLES = original_obstacles
        return EntityCfg(spec_fn=lambda xml=xml: mujoco.MjSpec.from_string(xml))

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


def add_navigation_laser_sensor(env_cfg, cfg: LaserSacConfig) -> None:
    laser = cfg.laser
    contact = cfg.contact_collision
    managed_sensor_names = {
        laser.sensor_name,
        contact.terminal_sensor_name,
        contact.penalized_sensor_name,
    }
    existing = tuple(
        sensor for sensor in (env_cfg.scene.sensors or ())
        if sensor.name not in managed_sensor_names
    )
    navigation_laser = RayCastSensorCfg(
        name=laser.sensor_name,
        frame=ObjRef(type="site", name=laser.lidar_site_name, entity="robot"),
        ray_alignment="yaw",
        pattern=HorizontalRingPatternCfg(
            num_rays=laser.num_rays,
            fov_deg=laser.fov_deg,
            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,
            yaw_offset_deg=laser.yaw_offset_deg,
            left_to_right=laser.left_to_right,
        ),
        max_distance=laser.max_range,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=laser.debug_vis,
        viz=RayCastSensorCfg.VizCfg(
            show_rays=laser.debug_vis,
            show_normals=False,
            hit_sphere_radius=0.08,
        ),
    )
    extra_sensors: tuple = ()
    if contact.termination_mode == "seanav_contact":
        # These two contact sensors add real per-substep MuJoCo/mjWarp
        # contact-kernel cost, and step_safe()'s per-substep force readout is
        # their only consumer. "circle" mode never reads them, so skip
        # creating them entirely in that mode rather than paying for sensors
        # whose output is always discarded.
        #
        # The Go2 MJCF has no separately named head body. Its front/head
        # collision geoms are base2_collision and base3_collision, while
        # base1_collision is the trunk. This is the MuJoCo equivalent of
        # SEA-Nav's base/head terminal contact set.
        terminal_contact = ContactSensorCfg(
            name=contact.terminal_sensor_name,
            primary=ContactMatch(
                mode="geom",
                pattern=("base1_collision", "base2_collision", "base3_collision"),
                entity="robot",
            ),
            secondary=None,
            fields=("found", "force", "normal", "tangent"),
            reduce="maxforce",
            num_slots=1,
            global_frame=True,
        )
        # Penalize all physical robot collision geoms except the four feet.
        # This includes trunk/front, hip, thigh and calf contacts with
        # terrain, walls, or procedural obstacles.
        nonfoot_contact = ContactSensorCfg(
            name=contact.penalized_sensor_name,
            primary=ContactMatch(
                mode="geom",
                pattern=r".*_collision\d*$",
                entity="robot",
                exclude=tuple(f"{leg}_foot_collision" for leg in ("FR", "FL", "RR", "RL")),
            ),
            secondary=None,
            fields=("found", "force", "normal", "tangent"),
            reduce="maxforce",
            num_slots=1,
            global_frame=True,
        )
        extra_sensors = (terminal_contact, nonfoot_contact)
    env_cfg.scene.sensors = existing + (navigation_laser,) + extra_sensors


def run(task_id: str, cfg: TrainLaserSacConfig) -> None:
    cfg = apply_training_profile(cfg)
    configure_torch_backends()
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    procedural_map_bank = load_procedural_map_bank(
        cfg.procedural_map_bank,
        expected_num_maps=cfg.expected_num_maps,
        expected_scenarios_per_map=cfg.expected_scenarios_per_map,
        # Slot capacity is authoritative bank metadata and is intentionally
        # inferred here. Laser observations do not impose a fixed obstacle
        # count, so users should not have to repeat it on the command line.
        expected_max_obstacles=None,
    )
    procedural_maps = procedural_map_bank.maps
    # Per-slot shape ("circle"/"square"): every map places the same shape in
    # a given physical slot index (resolve_slot_shapes enforces this), so a
    # bank can mix circle and square obstacles across different slots.
    slot_shapes = resolve_slot_shapes(procedural_maps, procedural_map_bank.max_obstacles)

    env_cfg = load_env_cfg(task_id, play=False)
    low_agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = cfg.num_envs
    # Every map in a bank is required to share the arena geometry.  Use that
    # geometry for the physical walls, not the legacy 15 m handcrafted arena.
    bank_arena = procedural_maps[0].arena
    if any(
        (item.arena.width, item.arena.length, item.arena.wall_thickness, item.arena.wall_height)
        != (bank_arena.width, bank_arena.length, bank_arena.wall_thickness, bank_arena.wall_height)
        for item in procedural_maps
    ):
        raise ValueError("All maps in one procedural bank must share arena wall geometry")
    env_cfg.scene.entities["navigation_arena"] = make_walls_only_arena_cfg(bank_arena)
    procedural_entity_names = tuple(
        f"procedural_obstacle_slot_{slot:02d}" for slot in range(procedural_map_bank.max_obstacles)
    )
    for slot, entity_name in enumerate(procedural_entity_names):
        env_cfg.scene.entities[entity_name] = make_procedural_obstacle_slot_entity(slot, slot_shapes[slot])
    env_cfg.terminations = {}
    if "push_robot" in env_cfg.events:
        del env_cfg.events["push_robot"]

    if cfg.laser_max_range <= cfg.config.laser.min_range:
        raise ValueError(
            f"--laser-max-range must exceed the configured minimum range "
            f"({cfg.config.laser.min_range} m), got {cfg.laser_max_range} m"
        )
    if cfg.laser_num_rays <= 0:
        raise ValueError(f"--laser-num-rays must be positive, got {cfg.laser_num_rays}")
    if not (0.0 < cfg.laser_fov_deg <= 360.0):
        raise ValueError(f"--laser-fov-deg must be in (0, 360], got {cfg.laser_fov_deg}")
    if cfg.safety_clearance <= 0.0:
        raise ValueError(f"--safety-clearance must be positive, got {cfg.safety_clearance} m")
    if cfg.collision_margin_m < 0.0:
        raise ValueError(f"--collision-margin-m must be non-negative, got {cfg.collision_margin_m} m")
    if cfg.collision_circle_radius_m <= 0.0:
        raise ValueError(
            f"--collision-circle-radius-m must be positive, got {cfg.collision_circle_radius_m} m"
        )
    if cfg.collision_termination_mode == "circle" and cfg.collision_geometry != "circle":
        raise ValueError(
            "--collision-termination-mode circle requires "
            "--collision-geometry circle"
        )
    if cfg.enable_collision_initialization and cfg.collision_termination_mode != "seanav_contact":
        raise ValueError(
            "SEA-Nav-inspired collision initialization requires "
            "--collision-termination-mode seanav_contact"
        )
    if not (0.0 <= cfg.collision_initialization_probability <= 1.0):
        raise ValueError("--collision-initialization-probability must be in [0, 1]")
    if not (0.0 < cfg.collision_undo_time_min_s <= cfg.collision_undo_time_max_s):
        raise ValueError("Collision undo times must satisfy 0 < min <= max")
    try:
        hidden_dims = tuple(int(value.strip()) for value in cfg.hidden_dims.split(",") if value.strip())
    except ValueError as exc:
        raise ValueError(
            f"--hidden-dims must be comma-separated integers, got {cfg.hidden_dims!r}"
        ) from exc
    if not hidden_dims or any(width <= 0 for width in hidden_dims):
        raise ValueError(f"--hidden-dims must contain positive widths, got {cfg.hidden_dims!r}")

    # Enable bounded, range-dependent LiDAR noise for training. Evaluation
    # constructs the default LaserSacConfig, where noise remains disabled.
    training_laser_cfg = replace(
        cfg.config.laser,
        enable_lidar_noise=cfg.enable_lidar_noise,
        num_rays=int(cfg.laser_num_rays),
        fov_deg=float(cfg.laser_fov_deg),
        max_range=float(cfg.laser_max_range),
    )
    # Preserve the historical CLI meaning: --safety-clearance is the desired
    # front-ray threshold measured from the LiDAR origin.  Convert it to the
    # additional body-footprint margin used by the local point-geometry CBF.
    base_safety = cfg.config.safety
    front_body_distance = max(
        0.5 * float(base_safety.robot_length_m) - float(base_safety.lidar_x_m),
        1.0e-8,
    )
    footprint_clearance = max(0.0, float(cfg.safety_clearance) - front_body_distance)
    training_safety_cfg = replace(
        base_safety,
        d_safe=float(cfg.safety_clearance),
        footprint_clearance_m=footprint_clearance,
        collision_geometry=cfg.collision_geometry,
        collision_circle_radius_m=float(cfg.collision_circle_radius_m),
        collision_margin_m=float(cfg.collision_margin_m),
        hard_clip_output=bool(cfg.hard_clip_output),
        qp_backend=cfg.qp_backend,
    )
    arena_template = procedural_maps[0].arena
    empty_arena_template = replace(arena_template, obstacles=())
    for item in procedural_maps[1:]:
        if replace(item.arena, obstacles=()) != empty_arena_template:
            raise ValueError("All procedural maps must use matching arena configuration")
    # Keep navigation metadata consistent with the configured front-ray
    # threshold.  The local CBF itself uses the derived footprint clearance.
    safety_arena_template = replace(
        empty_arena_template,
        wall_clearance=float(cfg.safety_clearance),
        obstacle_clearance=float(cfg.safety_clearance),
    )
    navigation_cfg = replace(
        cfg.config.navigation,
        arena=safety_arena_template,
    )
    # Vanilla is the no-CBF reward baseline. Keep the counterfactual filter
    # diagnostics, but do not reward or penalize the actor for a correction
    # that is never applied to its command.
    training_reward_cfg = replace(
        cfg.config.reward,
        cbf_correction_weight=(
            0.0 if cfg.method == "vanilla" else cfg.config.reward.cbf_correction_weight
        ),
    )
    training_sac_cfg = replace(
        cfg.config.sac,
        hidden_dims=hidden_dims,
    )
    method_cfg = replace(
        cfg.config,
        method=cfg.method,
        laser=training_laser_cfg,
        safety=training_safety_cfg,
        navigation=navigation_cfg,
        reward=training_reward_cfg,
        sac=training_sac_cfg,
        collision_initialization=replace(
            cfg.config.collision_initialization,
            enabled=bool(cfg.enable_collision_initialization),
            replay_probability=float(cfg.collision_initialization_probability),
            undo_time_min_s=float(cfg.collision_undo_time_min_s),
            undo_time_max_s=float(cfg.collision_undo_time_max_s),
        ),
        contact_collision=replace(
            cfg.config.contact_collision,
            termination_mode=cfg.collision_termination_mode,
        ),
    )
    add_navigation_laser_sensor(env_cfg, method_cfg)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    # MuJoCo-Warp supports these model fields per environment. Expanding them
    # lets twelve reusable cylinders preserve every saved map's exact radii/heights.
    raw_env.sim.expand_model_fields(("geom_size", "geom_rbound", "geom_aabb"))
    low_env = RslRlVecEnvWrapper(raw_env, clip_actions=low_agent_cfg.clip_actions)
    low_policy = load_frozen_low_level_policy(task_id, low_env, cfg.low_level_checkpoint, device)
    command_if = MjlabVelocityCommandInterface(low_env, command_name="twist")

    env = LaserSacNavigationEnv(
        low_env,
        low_policy,
        command_if,
        method_cfg.navigation,
        reward_cfg=method_cfg.reward,
        laser_cfg=method_cfg.laser,
        safety_cfg=method_cfg.safety,
        contact_collision_cfg=method_cfg.contact_collision,
        collision_initialization_cfg=method_cfg.collision_initialization,
        procedural_map_bank=procedural_map_bank,
        procedural_map_entity_names=procedural_entity_names,
        curriculum_recent_window=cfg.curriculum_recent_window,
        curriculum_mode=cfg.curriculum_mode,
    )
    env.assert_procedural_geometry_synchronized()
    shape_counts = {shape: slot_shapes.count(shape) for shape in ("circle", "square")}
    print(
        f"[GEOMETRY SYNC] reusable obstacle poses/sizes match analytical tensors "
        f"(slots: {shape_counts['circle']} circle, {shape_counts['square']} square)"
    )
    expected_actor_obs_dim = 12 + method_cfg.laser.num_rays
    actual_actor_obs_dim = int(env.actor_tensor(env.get_observations()).shape[-1])
    if (
        method_cfg.actor_observation_dim != expected_actor_obs_dim
        or actual_actor_obs_dim != expected_actor_obs_dim
    ):
        raise RuntimeError(
            "Laser actor observation dimension mismatch: "
            f"configured={method_cfg.actor_observation_dim}, "
            f"environment={actual_actor_obs_dim}, expected={expected_actor_obs_dim} "
            f"for {method_cfg.laser.num_rays} rays"
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{cfg.run_name}" if cfg.run_name else f"_{cfg.method}_seed{cfg.seed}"
    checkpoint = Path(cfg.resume_checkpoint).expanduser().resolve() if cfg.resume_checkpoint else None
    log_dir = checkpoint.parent if checkpoint else (
        Path("logs") / "sac_laser" / f"go2_safe_navigation_sac_laser_{cfg.method}" / f"{timestamp}{suffix}"
    )
    trainer = LaserSacTrainer(env, method_cfg, log_dir, device)
    difficulty_counts = {
        difficulty: sum(scenario.difficulty == difficulty for item in procedural_maps for scenario in item.scenarios)
        for difficulty in ("easy", "medium", "hard")
    }
    (log_dir / "procedural_map_curriculum.json").write_text(
        json.dumps(
            {
                **procedural_map_bank.checkpoint_metadata(),
                "num_maps": len(procedural_maps),
                "scenario_difficulty_counts": difficulty_counts,
                "quarter_weights_easy_medium_hard": CURRICULUM_WEIGHTS,
                "difficulty_mode": procedural_map_bank.difficulty_mode,
                "curriculum_mode": cfg.curriculum_mode,
                "collision_termination_mode": method_cfg.contact_collision.termination_mode,
                "collision_initialization": {
                    "enabled": method_cfg.collision_initialization.enabled,
                    "replay_probability": method_cfg.collision_initialization.replay_probability,
                    "undo_time_min_s": method_cfg.collision_initialization.undo_time_min_s,
                    "undo_time_max_s": method_cfg.collision_initialization.undo_time_max_s,
                },
                "physical_scene_architecture": "reusable per-environment mocap obstacle slots",
                "training_profile": cfg.training_profile,
                "reward_profile": method_cfg.reward.profile,
                "high_level_hz": 1.0 / method_cfg.navigation.high_level_dt,
                "low_level_hz": 1.0 / method_cfg.navigation.low_level_dt,
                "safety_integration_dt_s": method_cfg.safety.integration_dt,
                "backup_rollout_substep_s": (
                    method_cfg.safety.backup_horizon_s
                    / (method_cfg.safety.backup_samples - 1)
                    / method_cfg.safety.backup_rollout_substeps
                ),
                "terminal_backup_mode": method_cfg.safety.terminal_backup_mode,
                "terminal_geom_scale": method_cfg.safety.terminal_geom_scale,
                "terminal_rec_scale": method_cfg.safety.terminal_rec_scale,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if checkpoint:
        print(f"[LASER SAC] Resuming from {checkpoint}")
        trainer.load(
            checkpoint, load_optimizers=True, requested_target=cfg.vector_steps,
            allow_curriculum_replan=cfg.allow_curriculum_replan,
        )

    print(
        f"[LASER SAFETY] model={method_cfg.safety.geometry_model} "
        f"footprint=({method_cfg.safety.robot_length_m:.2f}x{method_cfg.safety.robot_width_m:.2f})m "
        f"body_clearance={method_cfg.safety.footprint_clearance_m:.3f}m "
        f"lidar_offset=({method_cfg.safety.lidar_x_m:.3f},{method_cfg.safety.lidar_y_m:.3f})m "
        f"hard_clip_output={method_cfg.safety.hard_clip_output}"
    )
    contact = method_cfg.contact_collision
    print(
        f"[COLLISION TERMINATION] mode={contact.termination_mode}; "
        "MuJoCo contact diagnostics: "
        f"terminal(base/front)>{contact.terminal_force_threshold_n:g}N "
        f"or hard(non-foot)>{contact.hard_force_threshold_n:g}N; "
        f"reward(non-foot)>{contact.penalized_force_threshold_n:g}N"
    )
    print(
        f"[CBF/DIAGNOSTIC PROXY] type={method_cfg.safety.collision_geometry} "
        f"circle_radius={method_cfg.safety.collision_circle_radius_m:.3f}m "
        f"ellipse_margin={method_cfg.safety.collision_margin_m:.3f}m"
    )
    print(
        f"[LASER SAC] method={cfg.method} maps={len(procedural_maps)} scenarios={procedural_map_bank.num_scenarios} "
        f"scenario_difficulty_counts={difficulty_counts} "
        f"physical_scene={procedural_map_bank.max_obstacles}_reusable_slots "
        f"bank={Path(cfg.procedural_map_bank).expanduser().resolve()} logging to {log_dir}"
    )
    ci = method_cfg.collision_initialization
    print(
        f"[TRAINING PROFILE] profile={cfg.training_profile} "
        f"reward={method_cfg.reward.profile} "
        f"high_level={1.0 / method_cfg.navigation.high_level_dt:g}Hz "
        f"low_level={1.0 / method_cfg.navigation.low_level_dt:g}Hz "
        f"curriculum={cfg.curriculum_mode} "
        f"collision_initialization={ci.enabled}"
    )
    if ci.enabled:
        print(
            "[COLLISION INITIALIZATION] "
            f"replay_probability={ci.replay_probability:g} "
            f"undo={ci.undo_time_min_s:g}-{ci.undo_time_max_s:g}s "
            f"early_reset_probability="
            f"{ci.early_reset_probability_min:g}-{ci.early_reset_probability_max:g}"
        )
    trainer.train(
        cfg.vector_steps, cfg.save_interval, cfg.log_interval,
        save_replay_buffer=cfg.save_replay_buffer,
        replay_save_interval=cfg.replay_save_interval,
        keep_latest_replay_only=cfg.keep_latest_replay_only,
        enable_bcbf_diagnostics=cfg.enable_bcbf_diagnostics,
    )
    env.close()


def main(*, training_profile_override: str | None = None) -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(
        TrainLaserSacConfig,
        args=remaining,
        prog=sys.argv[0] + f" {task}",
        config=mjlab.TYRO_FLAGS,
    )
    if training_profile_override is not None:
        cfg = replace(cfg, training_profile=training_profile_override)
    run(task, cfg)


if __name__ == "__main__":
    main()
