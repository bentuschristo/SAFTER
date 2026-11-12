#!/usr/bin/env python3
"""Evaluate laser SAC methods on structured unseen static maps.

Methods:
    vanilla
    cbf_cf_clip
    cbf_cd_cf
    bcbf_cf

This script is the laser-observation analogue of evaluate_unseen_maps_all_methods.py.
It generates structured start/goal scenarios so that the intended obstacle
layout is actually tested. The default evaluation set is the three maps
"low_shift"/"medium_shift"/"high_shift" (loaded from "Maps/Evaluation Maps/"
-- see _load_evaluation_maps below); every other map variant this file used
to carry inline now lives in legacy_unseen_maps.py.

Each map has a start box and a goal box. The sampler rejects cases where
start/goal are too close to obstacles/walls and, by default, rejects cases where
the straight-line start-goal segment is not blocked or near-blocked by at least
one obstacle. This avoids easy episodes where no obstacle lies between start
and goal.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace
from typing import Any, Iterable, Literal

# Allow running as: python scripts/laser_unseen_eval/evaluate_laser_unseen_maps_four_methods.py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from SAFTER_src.high_level_control.command_interface import MjlabVelocityCommandInterface
from SAFTER_src.high_level_laser_sac.config import LaserSacConfig, LaserSacMethod
from SAFTER_src.high_level_laser_sac.networks import LaserSafeSacActor
from SAFTER_src.high_level_laser_sac.sac_navigation_env import LaserSacNavigationEnv
from SAFTER_src.high_level_rl.config import ArenaConfig, CircularObstacle
from SAFTER_src.high_level_rl.low_level import load_frozen_low_level_policy
from SAFTER_src.high_level_rl.raycast_patterns import HorizontalRingPatternCfg
from src.tasks.velocity.config.go2 import obstacle_cfg as physical_arena_cfg

# Register tasks before load_env_cfg().
import mjlab.tasks  # noqa: F401,E402
import src.tasks  # noqa: F401,E402


# -----------------------------------------------------------------------------
# Evaluation maps: low_shift / medium_shift / high_shift.
#
# As of 2026-11, this structured evaluator's default map set was reduced to
# exactly these three maps -- the "low-shift", "medium-shift", "high-shift"
# trio from the reference obstacle-layout image -- loaded from JSON data
# files under "Maps/Evaluation Maps/" rather than carried as inline Python
# literals. They were previously named "slalom" (low_shift), "corridor"
# (medium_shift), and "corridor_derived_highshift_v5" (high_shift) in this
# file's old UNSEEN_MAPS/STRUCTURED_SCENARIO_REGIONS dicts.
#
# Every other map variant this file used to define inline (all the
# exploratory/superseded/OOD/distribution-shift/dynamic-obstacle designs, with
# their original design-rationale comments preserved verbatim) was relocated
# to SAFTER_scripts/laser_unseen_eval/legacy_unseen_maps.py as
# LEGACY_UNSEEN_MAPS/LEGACY_STRUCTURED_SCENARIO_REGIONS -- not deleted, just no
# longer part of this evaluator's default pipeline. That module is NOT
# imported here deliberately; import it directly if you need one of those
# older maps again.
_EVALUATION_MAPS_DIR = ROOT / "Maps" / "Evaluation Maps"
_EVALUATION_MAP_NAMES = ("low_shift", "medium_shift", "high_shift")


def _load_evaluation_maps(maps_dir: Path) -> tuple[
    dict[str, tuple[CircularObstacle, ...]],
    dict[str, dict[str, tuple[float, float, float, float]]],
]:
    """Load the low_shift/medium_shift/high_shift JSON map files into the
    exact UNSEEN_MAPS / STRUCTURED_SCENARIO_REGIONS shapes the rest of this
    file has always expected, so nothing downstream needs to change."""
    unseen_maps: dict[str, tuple[CircularObstacle, ...]] = {}
    regions: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    for name in _EVALUATION_MAP_NAMES:
        path = maps_dir / f"{name}.json"
        data = json.loads(path.read_text())
        obstacles = tuple(
            CircularObstacle(
                x=float(o["x"]),
                y=float(o["y"]),
                radius=float(o["radius"]),
                height=float(o.get("height", 1.0)),
                shape=o.get("shape", "circle"),
            )
            for o in data["obstacles"]
        )
        unseen_maps[data["name"]] = obstacles

        def _box(region: dict[str, float]) -> tuple[float, float, float, float]:
            return (
                float(region["x_min"]),
                float(region["x_max"]),
                float(region["y_min"]),
                float(region["y_max"]),
            )

        regions[data["name"]] = {
            "start": _box(data["start_region"]),
            "goal": _box(data["goal_region"]),
        }
    return unseen_maps, regions


UNSEEN_MAPS: dict[str, tuple[CircularObstacle, ...]]
STRUCTURED_SCENARIO_REGIONS: dict[str, dict[str, tuple[float, float, float, float]]]
UNSEEN_MAPS, STRUCTURED_SCENARIO_REGIONS = _load_evaluation_maps(_EVALUATION_MAPS_DIR)


# -----------------------------------------------------------------------------
# Dynamic (moving) obstacles.
#
# Ported from the trajectory math in the untested/uncommitted prototype
# scripts_v5/laser_unseen_eval/evaluate_laser_dynamic_obstacles_three_methods.py
# (2026-08-10), but wired into THIS evaluator's safety-validated pipeline
# instead of that prototype's older helper (evaluate_laser_unseen_maps_four_methods.py),
# which never set collision.termination_mode (defaults to "seanav_contact"
# but never creates a contact sensor -- would crash or silently mis-detect
# collisions) and reused the old max_obstacles=max(len(obstacles), 9) floor
# bug this session already fixed for the static maps above.
#
# Convention: for any map name present in DYNAMIC_OBSTACLE_COUNTS, the LAST
# N entries of UNSEEN_MAPS[map_name] (N = the registered count) are the
# dynamic ones; everything before that is static. UNSEEN_MAPS stores each
# dynamic obstacle's position at t=0 (matching DynamicObstacleSpec.position_at(0.0)),
# so arena/manifest generation and the one-time compiled-geometry checks
# (verify_compiled_physical_map/verify_parallel_lidar_geometry) work exactly
# as they do for fully-static maps.
@dataclass
class DynamicObstacleSpec:
    radius: float
    trajectory_type: Literal["line", "line_oscillate", "circle", "sine", "waypoints"]
    name: str = "dynamic"

    # Common / line parameters
    p0: tuple[float, float] = (0.0, 0.0)
    speed: float = 0.20
    heading_deg: float = 0.0
    bounce: bool = False

    # Circle parameters
    center: tuple[float, float] = (0.0, 0.0)
    path_radius: float = 2.0
    omega: float | None = None
    phase_deg: float = 0.0

    # Sine parameters
    amplitude: float = 1.0
    frequency_hz: float = 0.10

    # Waypoint parameters
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    loop: bool = True

    def position_at(self, t: float) -> tuple[float, float]:
        if self.trajectory_type == "line":
            heading = math.radians(self.heading_deg)
            dx, dy = math.cos(heading), math.sin(heading)
            x = self.p0[0] + self.speed * t * dx
            y = self.p0[1] + self.speed * t * dy
            return (x, y)

        if self.trajectory_type == "line_oscillate":
            heading = math.radians(self.heading_deg)
            dx, dy = math.cos(heading), math.sin(heading)
            omega = self.speed / max(self.amplitude, 1.0e-9)
            phase = math.radians(self.phase_deg)
            displacement = self.amplitude * math.sin(omega * t + phase)
            x = self.p0[0] + dx * displacement
            y = self.p0[1] + dy * displacement
            return (x, y)

        if self.trajectory_type == "circle":
            omega = self.omega if self.omega is not None else self.speed / max(self.path_radius, 1.0e-9)
            phase = math.radians(self.phase_deg)
            x = self.center[0] + self.path_radius * math.cos(omega * t + phase)
            y = self.center[1] + self.path_radius * math.sin(omega * t + phase)
            return (x, y)

        if self.trajectory_type == "sine":
            heading = math.radians(self.heading_deg)
            forward = (math.cos(heading), math.sin(heading))
            lateral = (-math.sin(heading), math.cos(heading))
            disp = self.amplitude * math.sin(2.0 * math.pi * self.frequency_hz * t)
            x = self.p0[0] + forward[0] * self.speed * t + lateral[0] * disp
            y = self.p0[1] + forward[1] * self.speed * t + lateral[1] * disp
            return (x, y)

        if self.trajectory_type == "waypoints":
            return _position_on_waypoints(self.waypoints, self.speed, t, loop=self.loop)

        raise ValueError(f"Unsupported trajectory_type: {self.trajectory_type}")


def _position_on_waypoints(
    waypoints: Iterable[tuple[float, float]], speed: float, t: float, loop: bool = True
) -> tuple[float, float]:
    pts = list(waypoints)
    if not pts:
        raise ValueError("waypoints trajectory needs at least one waypoint")
    if len(pts) == 1 or speed <= 0:
        return pts[0]

    segments = list(zip(pts, pts[1:] + pts[:1])) if loop else list(zip(pts[:-1], pts[1:]))
    seg_lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segments]
    total_len = sum(seg_lengths)
    if total_len <= 1.0e-9:
        return pts[0]

    dist = speed * t
    dist = dist % total_len if loop else min(dist, total_len)

    accum = 0.0
    for (a, b), length in zip(segments, seg_lengths):
        if dist <= accum + length or length <= 1.0e-9:
            s = 0.0 if length <= 1.0e-9 else (dist - accum) / length
            return (a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]))
        accum += length
    return segments[-1][1]


# Number of trailing UNSEEN_MAPS[name] obstacles that are dynamic, keyed by
# map name. A map absent from this dict is treated as fully static. Populated
# below from len(DYNAMIC_OBSTACLE_SPECS[name]) -- never hardcode these counts
# here; a mismatch with the actual spec tuple length causes obstacle_centers
# slice-assignment shape errors at env build time (hit once already when a
# redesign changed obstacle counts but this dict was left stale).
DYNAMIC_OBSTACLE_COUNTS: dict[str, int] = {}

# The DynamicObstacleSpec for each dynamic obstacle, in the same trailing
# order as DYNAMIC_OBSTACLE_COUNTS/UNSEEN_MAPS.
DYNAMIC_OBSTACLE_SPECS: dict[str, tuple[DynamicObstacleSpec, ...]] = {}


def _dynamic_spec_positions_at_t0(specs: tuple[DynamicObstacleSpec, ...]) -> tuple[tuple[float, float], ...]:
    return tuple(spec.position_at(0.0) for spec in specs)


DEFAULT_LOW ="logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt"
DEFAULT_VANILLA = "logs/sac_laser/go2_safe_navigation_sac_laser_vanilla/2026-07-10_14-13-58_vanilla_laser/model_step_200000.pt"
DEFAULT_CBF_CLIP = "logs/sac_laser/go2_safe_navigation_sac_laser_cbf_cf_clip/2026-07-10_16-53-51_cbf_cf_clip_laser/model_step_200000.pt"
DEFAULT_CBF_CD = "logs/sac_laser/go2_safe_navigation_sac_laser_cbf_cd_cf/2026-07-10_19-49-32_cbf_cd_cf_laser/model_step_200000.pt"
DEFAULT_BCBF = "logs/sac_laser/go2_safe_navigation_sac_laser_bcbf_cf/2026-07-10_22-33-06_bcbf_cf_laser/model_step_200000.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate laser SAC methods on shared unseen map manifests."
    )
    parser.add_argument("task_id", nargs="?", default="Unitree-Go2-Safe-Navigation")
    parser.add_argument(
        "--cases-per-seed", "--cases-per-map", dest="cases_per_seed", type=int, default=20,
        help="Cases generated per map for each evaluation seed (default: 20).",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=10,
        help="Number of consecutive evaluation seeds (default: 10, giving 200 cases/map).",
    )
    parser.add_argument(
        "--seed-stride", type=int, default=1,
        help="Spacing between evaluation seeds (default: 1).",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1,
        help="Evaluate this many cases concurrently on each map.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-duration-s", type=float, default=60.0)
    parser.add_argument("--laser-num-rays", type=int, default=61)
    parser.add_argument("--laser-fov-deg", type=float, default=180.0)
    parser.add_argument("--collision-geometry", choices=("circle", "ellipse"), default="ellipse")
    parser.add_argument("--collision-circle-radius-m", type=float, default=0.30)
    parser.add_argument("--collision-margin-m", type=float, default=0.05)
    parser.add_argument(
        "--collision-termination-mode",
        choices=("circle", "seanav_contact"),
        default="circle",
    )
    parser.add_argument(
        "--terminal-backup-mode",
        choices=("legacy_inflated_ellipse", "invariant_ellipse"),
        default="legacy_inflated_ellipse",
        help="Must match whatever terminal_backup_mode the checkpoint(s) were trained with "
             "(load_actor's safety-config compatibility check will reject a mismatch).",
    )
    parser.add_argument("--terminal-geom-scale", type=float, default=1.0)
    parser.add_argument("--terminal-rec-scale", type=float, default=1.0)
    parser.add_argument(
        "--legacy-projection", action=argparse.BooleanOptionalAction, default=False,
        help="bcbf_cf only: use the pre-2026-09-07 single-half-space projection "
             "(_exact_projection + naive post-hoc clamp) instead of the joint "
             "box+half-space projection. Independent of --terminal-backup-mode. "
             "Must match what the checkpoint was trained with.",
    )
    parser.add_argument(
        "--action-max-vx", type=float, default=0.5,
        help="Must match the checkpoint's training-time navigation.action.max_vx "
             "(load_actor's compatibility check will reject a mismatch).",
    )
    parser.add_argument(
        "--action-min-vx", type=float, default=None,
        help="Must match the checkpoint's training-time navigation.action.min_vx "
             "(load_actor's compatibility check will reject a mismatch). Leave "
             "unset (None) to default to -action-max-vx (symmetric), matching "
             "every checkpoint trained before this field existed.",
    )
    parser.add_argument(
        "--action-max-vy", type=float, default=0.3,
        help="Must match the checkpoint's training-time navigation.action.max_vy.",
    )
    parser.add_argument(
        "--action-max-wz", type=float, default=0.5,
        help="Must match the checkpoint's training-time navigation.action.max_wz.",
    )
    parser.add_argument(
        "--backup-margin", type=float, default=None,
        help="Override safety.backup_margin (bcbf_cf/bcbf_qp/etc. only). Must match "
             "the checkpoint's training-time --config.safety.backup-margin "
             "(load_actor's compatibility check will reject a mismatch). "
             "Leave unset (None) to use the LaserSafetyConfig default (0.30).",
    )
    parser.add_argument(
        "--high-level-dt", type=float, default=None,
        help="Override navigation.high_level_dt (e.g. 0.02 for a 50 Hz checkpoint). "
             "Must match the checkpoint's training-time value (load_actor's "
             "compatibility check will reject a mismatch). Leave unset (None) "
             "for the archived 10 Hz default (0.10).",
    )
    parser.add_argument(
        "--integration-dt", type=float, default=None,
        help="Override safety.integration_dt to match the checkpoint's training-time "
             "value (must agree with --high-level-dt for a 50 Hz checkpoint: 0.02). "
             "Leave unset (None) for the default (0.10).",
    )
    parser.add_argument(
        "--backup-rollout-substeps", type=int, default=None,
        help="Override safety.backup_rollout_substeps to match the checkpoint's "
             "training-time value (10 for a 50 Hz checkpoint). Leave unset (None) "
             "for the default (8).",
    )
    parser.add_argument(
        "--enable-lidar-noise", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--hard-clip-output", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--low-level-checkpoint", default=DEFAULT_LOW)
    parser.add_argument("--vanilla-checkpoint", default=DEFAULT_VANILLA)
    parser.add_argument("--cbf-cf-clip-checkpoint", default=DEFAULT_CBF_CLIP)
    parser.add_argument("--cbf-qp-checkpoint", default=None)
    parser.add_argument("--cbf-lse-qp-checkpoint", default=None)
    parser.add_argument("--cbf-cd-cf-checkpoint", default=DEFAULT_CBF_CD)
    parser.add_argument("--cbf-lse-cd-qp-checkpoint", default=None)
    parser.add_argument("--bcbf-cf-checkpoint", default=DEFAULT_BCBF)
    parser.add_argument("--bcbf-qp-checkpoint", default=None)
    parser.add_argument("--bcbf-lse-cd-qp-checkpoint", default=None)
    parser.add_argument("--bcbf-nonlse-cd-qp-checkpoint", default=None)
    parser.add_argument("--cbf-lse-cd-qp-cfmatched-checkpoint", default=None)
    parser.add_argument("--bcbf-lse-cd-qp-cfmatched-checkpoint", default=None)
    parser.add_argument("--qp-backend", "--bcbf-qp-backend", dest="qp_backend",
                        choices=("active_set", "qpth", "osqp"), default="osqp",
                        help="QP solver used at evaluation/deployment time (default: exact OSQP).")
    parser.add_argument(
        "--disable-safety-filter", action="store_true",
        help="Run the RAW nominal policy action, bypassing the CBF/bCBF safety "
             "filter entirely (u_s = u_bar). The filter still runs internally "
             "so filter_active/correction metrics keep logging normally -- "
             "only the executed command changes. No effect on method=vanilla, "
             "which never applies a filter to begin with.",
    )
    parser.add_argument(
        "--dynamic-phase-jitter-s", type=float, default=20.0,
        help="For maps with dynamic (moving) obstacles: each case samples a "
             "deterministic (from its own reset_seed) time offset in "
             "[0, value] so cases start at different points in the "
             "obstacles' motion cycle. Ignored for fully-static maps.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("vanilla", "cbf_cf_clip", "cbf_qp", "cbf_lse_qp", "cbf_cd_cf", "cbf_lse_cd_qp", "bcbf_cf", "bcbf_qp", "bcbf_lse_cd_qp", "bcbf_nonlse_cd_qp", "cbf_lse_cd_qp_cfmatched", "bcbf_lse_cd_qp_cfmatched"),
        default=("vanilla", "cbf_cf_clip", "cbf_cd_cf", "bcbf_cf"),
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        default=list(UNSEEN_MAPS),
        choices=list(UNSEEN_MAPS),
    )
    parser.add_argument("--case-ids", nargs="+", type=int, default=None)
    parser.add_argument(
        "--min-endpoint-clearance",
        type=float,
        default=0.10,
        help="Minimum required start/goal clearance to obstacles and arena walls [m].",
    )
    parser.add_argument(
        "--block-margin",
        type=float,
        default=0.10,
        help=(
            "Straight-line segment is considered blocked if an obstacle center is within "
            "radius + obstacle_clearance + block_margin of the segment [m]."
        ),
    )
    parser.add_argument(
        "--blocked-t-min",
        type=float,
        default=0.15,
        help="Minimum segment interpolation value for a blocking obstacle to count.",
    )
    parser.add_argument(
        "--blocked-t-max",
        type=float,
        default=0.85,
        help="Maximum segment interpolation value for a blocking obstacle to count.",
    )
    parser.add_argument(
        "--allow-unblocked-paths",
        action="store_true",
        help="Do not reject start/goal samples whose straight-line path is unblocked.",
    )
    parser.add_argument(
        "--max-sampling-attempts",
        type=int,
        default=10000,
        help="Maximum rejection-sampling attempts per map.",
    )
    parser.add_argument(
        "--yaw-jitter-rad",
        type=float,
        default=0.25,
        help="Uniform yaw jitter around the start-to-goal direction [rad].",
    )
    parser.add_argument(
        "--randomize-start-goal",
        action="store_true",
        help=(
            "ADDITIVE alternative to the fixed STRUCTURED_SCENARIO_REGIONS boxes: "
            "sample both start and goal independently from the full arena interior "
            "(generate_manifest_randomized) instead of a map-specific fixed box "
            "(generate_manifest). Writes its scenario bank under a distinctly-named "
            "'<map_name>_random' subdirectory so it never resumes from or "
            "overwrites an existing structured-region manifest under the same "
            "--run-root/--output-root; generate_manifest() and its saved manifests "
            "are completely unaffected by this flag."
        ),
    )
    parser.add_argument(
        "--min-random-distance-m",
        type=float,
        default=4.0,
        help=(
            "Minimum straight-line start-goal distance when --randomize-start-goal "
            "is set (generate_manifest's own fixed-region sampling keeps its "
            "original 1.0 m minimum, unaffected by this flag)."
        ),
    )
    parser.add_argument("--log-trajectories", action="store_true")
    parser.add_argument("--output-root", default="logs/evaluation/laser_unseen_maps_structured_200k")
    parser.add_argument(
        "--run-root",
        default=None,
        help=(
            "Write directly into an existing/fixed run directory. Existing "
            "scenario manifests and complete per-method episode CSVs are reused."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def optional_mean(values: list[float]) -> float:
    return mean(values) if values else math.nan


def mean_stats(
    values: list[float], prefix: str, *, critical_value: float = 1.96
) -> dict[str, float | int]:
    """Sample mean, SD, SEM, and two-sided 95% CI."""
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {f"n_{prefix}": 0, f"mean_{prefix}": math.nan,
                f"std_{prefix}": math.nan, f"sem_{prefix}": math.nan,
                f"ci95_low_{prefix}": math.nan, f"ci95_high_{prefix}": math.nan}
    avg = mean(clean)
    sd = sample_std(clean)
    sem = sd / math.sqrt(len(clean)) if len(clean) >= 2 else 0.0
    return {f"n_{prefix}": len(clean), f"mean_{prefix}": avg,
            f"std_{prefix}": sd, f"sem_{prefix}": sem,
            f"ci95_low_{prefix}": avg - critical_value * sem,
            f"ci95_high_{prefix}": avg + critical_value * sem}


def t95_critical(sample_count: int) -> float:
    """Two-sided 95% Student-t critical value (small table, normal fallback)."""
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
             7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
             12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
             17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 25: 2.064,
             30: 2.045}
    if sample_count <= 1:
        return 0.0
    if sample_count in table:
        return table[sample_count]
    lower = max((n for n in table if n <= sample_count), default=30)
    return table[lower] if sample_count < 30 else 1.96


def proportion_stats(successes: int, total: int, prefix: str) -> dict[str, float | int]:
    """Rate, binomial standard error, and Wilson 95% confidence interval."""
    if total <= 0:
        return {f"count_{prefix}": 0, f"{prefix}": math.nan,
                f"sem_{prefix}": math.nan, f"ci95_low_{prefix}": math.nan,
                f"ci95_high_{prefix}": math.nan}
    rate = successes / total
    sem = math.sqrt(rate * (1.0 - rate) / total)
    z = 1.96
    denom = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denom
    return {f"count_{prefix}": successes, f"{prefix}": rate, f"sem_{prefix}": sem,
            f"ci95_low_{prefix}": max(0.0, center - half),
            f"ci95_high_{prefix}": min(1.0, center + half)}


def summarize_episodes(episodes: list[dict[str, object]], **labels: object) -> dict[str, object]:
    """Create plot-ready episode-level descriptive and uncertainty statistics."""
    row: dict[str, object] = dict(labels)
    row["cases"] = len(episodes)
    for field, label in (("success", "success_rate"), ("collision", "collision_rate"),
                         ("safety_violation", "laser_safety_violation_rate"),
                         ("timeout", "timeout_rate"), ("fallen", "fall_rate"),
                         ("outside", "outside_rate")):
        row.update(proportion_stats(sum(int(r[field]) for r in episodes), len(episodes), label))
    successes = [r for r in episodes if int(r["success"]) == 1]
    row.update(mean_stats([float(r["return"]) for r in episodes], "return"))
    row.update(mean_stats([float(r["path_length_m"]) for r in successes], "success_path_length_m"))
    row.update(mean_stats([float(r["episode_duration_s"]) for r in successes], "success_goal_time_s"))
    row.update(mean_stats([float(r["minimum_physical_clearance_m"]) for r in episodes], "minimum_clearance_m"))
    row["minimum_clearance_overall_m"] = min(
        (float(r["minimum_physical_clearance_m"]) for r in episodes), default=math.nan
    )
    row.update(mean_stats([float(r["filter_active_fraction"]) for r in episodes], "filter_active_fraction"))
    row.update(mean_stats([float(r["mean_command_correction_norm"]) for r in episodes], "command_correction_norm"))
    for field, label in (
        ("mean_safe_measured_velocity_error_norm", "safe_measured_velocity_error_norm"),
        ("mean_policy_inference_batch_ms", "policy_inference_batch_ms"),
        ("mean_safety_filter_batch_ms", "safety_filter_batch_ms"),
        ("mean_total_actor_inference_batch_ms", "total_actor_inference_batch_ms"),
        ("mean_total_actor_inference_amortized_per_env_ms", "total_actor_inference_amortized_per_env_ms"),
        ("inference_deadline_miss_fraction", "inference_deadline_miss_fraction"),
    ):
        if episodes and field in episodes[0]:
            row.update(mean_stats([float(r[field]) for r in episodes], label))
    return row


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _device_synchronize(device: torch.device | str) -> None:
    """Make accelerator timings represent completed work, not queued kernels."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_deterministic_actor(actor, actor_obs: torch.Tensor):
    """Return deterministic commands plus policy/filter/end-to-end latency.

    Latencies describe the complete batch.  The policy measurement covers the
    MLP and tanh; the filter measurement covers scan extraction and safety
    synthesis.  CUDA synchronization is deliberately outside each interval.
    """
    _device_synchronize(actor_obs.device)
    start_ns = time.perf_counter_ns()
    mean_action = actor.backbone(actor_obs).chunk(2, dim=-1)[0]
    nominal_normalized = torch.tanh(mean_action)
    _device_synchronize(actor_obs.device)
    policy_end_ns = time.perf_counter_ns()
    safe_normalized, safe_command, safety = actor.apply_safety(nominal_normalized, actor_obs)
    _device_synchronize(actor_obs.device)
    filter_end_ns = time.perf_counter_ns()
    return (
        actor.normalized_to_command(nominal_normalized),
        safe_command,
        safety,
        (policy_end_ns - start_ns) / 1.0e6,
        (filter_end_ns - policy_end_ns) / 1.0e6,
        (filter_end_ns - start_ns) / 1.0e6,
    )


def warmup_actor_timing(actor, env, iterations: int = 10) -> None:
    """Populate lazy CUDA kernels/caches before collecting latency samples."""
    with torch.no_grad():
        actor_obs = env.actor_tensor(env.get_observations())
        for _ in range(iterations):
            actor.deterministic(actor_obs)
        _device_synchronize(actor_obs.device)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def make_map_arena(map_name: str) -> ArenaConfig:
    obstacles = UNSEEN_MAPS[map_name]
    # max_obstacles must equal the actual obstacle count exactly: the
    # compiled MuJoCo scene only creates one body per real obstacle, so any
    # padding slot (max_obstacles > len(obstacles)) has no matching body and
    # verify_compiled_physical_map() fails looking for it. The 5 original
    # maps all happen to have exactly 9 obstacles, which is why the old
    # hardcoded default of 9 never surfaced this.
    arena = ArenaConfig(obstacles=obstacles, max_obstacles=len(obstacles))
    arena.validate()
    return arena


def static_obstacle_count(map_name: str) -> int:
    """Number of UNSEEN_MAPS[map_name] obstacles that are NOT dynamic."""
    dynamic_count = DYNAMIC_OBSTACLE_COUNTS.get(map_name, 0)
    return len(UNSEEN_MAPS[map_name]) - dynamic_count


def install_physical_map(map_name: str) -> None:
    # Only the STATIC prefix is compiled into the shared "navigation_arena"
    # entity via physical_arena_cfg. Dynamic obstacles (the trailing
    # DYNAMIC_OBSTACLE_COUNTS[map_name] entries) are compiled separately as
    # individually addressable mocap bodies in build_environment(), matching
    # the naming convention (obstacle_{index}/obstacle_{index}_collision)
    # physical_arena_cfg._make_arena_xml() would have used for them, so
    # verify_compiled_physical_map/verify_parallel_lidar_geometry and the
    # LiDAR raycasting code find them the same way regardless of which path
    # compiled them.
    source = UNSEEN_MAPS[map_name][: static_obstacle_count(map_name)]
    physical_arena_cfg.OBSTACLES = tuple(
        physical_arena_cfg.CircularObstacle(
            x=float(obs.x),
            y=float(obs.y),
            radius=float(obs.radius),
            height=float(obs.height),
            shape=getattr(obs, "shape", "circle"),
        )
        for obs in source
    )


def add_navigation_laser_sensor(env_cfg, cfg: LaserSacConfig) -> None:
    laser = cfg.laser
    contact = cfg.contact_collision
    managed_names = {
        laser.sensor_name,
        contact.terminal_sensor_name,
        contact.penalized_sensor_name,
    }
    existing = tuple(
        sensor for sensor in (env_cfg.scene.sensors or ())
        if sensor.name not in managed_names
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
        # contact-kernel cost and are the only consumers of
        # step_safe()'s contact-force readout. "circle" mode (the default,
        # and the only mode used by every local eval/training run so far)
        # never reads them, so skip creating them entirely in that mode
        # rather than paying for sensors whose output is always discarded.
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
        nonfoot_contact = ContactSensorCfg(
            name=contact.penalized_sensor_name,
            primary=ContactMatch(
                mode="geom",
                pattern=r".*_collision\d*$",
                entity="robot",
                exclude=tuple(
                    f"{leg}_foot_collision" for leg in ("FR", "FL", "RR", "RL")
                ),
            ),
            secondary=None,
            fields=("found", "force", "normal", "tangent"),
            reduce="maxforce",
            num_slots=1,
            global_frame=True,
        )
        extra_sensors = (terminal_contact, nonfoot_contact)
    env_cfg.scene.sensors = existing + (navigation_laser,) + extra_sensors


def make_laser_config(
    method: str,
    episode_duration_s: float,
    arena: ArenaConfig | None,
    *,
    laser_num_rays: int = 61,
    laser_fov_deg: float = 180.0,
    enable_lidar_noise: bool = False,
    hard_clip_output: bool = True,
    collision_geometry: str = "ellipse",
    collision_circle_radius_m: float = 0.30,
    collision_margin_m: float = 0.05,
    collision_termination_mode: str = "circle",
    qp_backend: str = "active_set",
    terminal_backup_mode: str = "legacy_inflated_ellipse",
    terminal_geom_scale: float = 1.0,
    terminal_rec_scale: float = 1.0,
    legacy_projection: bool = False,
    action_max_vx: float = 0.5,
    action_min_vx: float | None = None,
    action_max_vy: float = 0.3,
    action_max_wz: float = 0.5,
    backup_margin: float | None = None,
    high_level_dt: float | None = None,
    integration_dt: float | None = None,
    backup_rollout_substeps: int | None = None,
    laser_debug_vis: bool = False,
) -> LaserSacConfig:
    base = LaserSacConfig()
    # Default min_vx to -max_vx (symmetric) when not explicitly overridden,
    # matching every action_max_vx-only caller that predates this field --
    # NOT LaserSafetyConfig's own hardcoded -0.5 default, which would silently
    # pair an overridden max_vx with a stale, unrelated min_vx otherwise.
    resolved_min_vx = -float(action_max_vx) if action_min_vx is None else float(action_min_vx)
    action = replace(
        base.navigation.action,
        max_vx=float(action_max_vx),
        min_vx=resolved_min_vx,
        max_vy=float(action_max_vy),
        max_wz=float(action_max_wz),
    )
    nav = replace(
        base.navigation,
        episode_duration_s=float(episode_duration_s),
        action=action,
        **({} if high_level_dt is None else {"high_level_dt": float(high_level_dt)}),
    )
    laser = replace(
        base.laser,
        num_rays=int(laser_num_rays),
        fov_deg=float(laser_fov_deg),
        enable_lidar_noise=bool(enable_lidar_noise),
        debug_vis=bool(laser_debug_vis),
    )
    # Match the training entry point's conversion from the requested front-ray
    # clearance to the additional margin around the elliptical body footprint.
    front_body_distance = max(
        0.5 * float(base.safety.robot_length_m) - float(base.safety.lidar_x_m),
        1.0e-8,
    )
    footprint_clearance = max(0.0, float(base.safety.d_safe) - front_body_distance)
    safety = replace(
        base.safety,
        footprint_clearance_m=footprint_clearance,
        collision_geometry=collision_geometry,
        collision_circle_radius_m=float(collision_circle_radius_m),
        collision_margin_m=float(collision_margin_m),
        hard_clip_output=bool(hard_clip_output),
        qp_backend=qp_backend,
        terminal_backup_mode=terminal_backup_mode,
        terminal_geom_scale=float(terminal_geom_scale),
        terminal_rec_scale=float(terminal_rec_scale),
        legacy_projection=bool(legacy_projection),
        **({} if backup_margin is None else {"backup_margin": float(backup_margin)}),
        **({} if integration_dt is None else {"integration_dt": float(integration_dt)}),
        **({} if backup_rollout_substeps is None else {"backup_rollout_substeps": int(backup_rollout_substeps)}),
    )
    reward = replace(
        base.reward,
        cbf_correction_weight=(0.0 if method == "vanilla" else base.reward.cbf_correction_weight),
    )
    if arena is not None:
        nav = replace(
            nav,
            arena=replace(arena, wall_clearance=0.30, obstacle_clearance=0.30),
        )
    contact_collision = replace(
        base.contact_collision,
        termination_mode=collision_termination_mode,
    )
    return replace(
        base,
        method=method,
        navigation=nav,
        laser=laser,
        safety=safety,
        reward=reward,
        contact_collision=contact_collision,
    )


def _find_named_id(model, object_type, target_name: str, count: int) -> int:
    exact_id = mujoco.mj_name2id(model, object_type, target_name)
    if exact_id >= 0:
        return exact_id
    matches = [
        object_id
        for object_id in range(count)
        if (name := mujoco.mj_id2name(model, object_type, object_id)) is not None
        and (name == target_name or name.endswith("/" + target_name))
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return -1
    raise RuntimeError(f"Multiple compiled objects match {target_name!r}: {matches}")


class DynamicObstacleRuntime:
    """Move a map's dynamic (mocap-body) obstacles to their position at a
    given per-env simulation time, keeping the MuJoCo scene, the LiDAR
    sensor, and env.obstacle_centers (read by the CBF/bCBF clearance math)
    all in sync. Ported from evaluate_laser_dynamic_obstacles_three_methods.py's
    DynamicObstacleRuntime, generalized to per-env times (one scalar time
    per parallel environment, not one shared across the whole batch) so a
    per-case random phase offset can be applied.
    """

    def __init__(self, *, raw_env, env, map_name: str):
        self.raw_env = raw_env
        self.env = env
        self.static_count = static_obstacle_count(map_name)
        self.specs = DYNAMIC_OBSTACLE_SPECS.get(map_name, ())
        model = raw_env.sim.mj_model
        self.mocap_ids: list[int] = []
        for j in range(len(self.specs)):
            index = self.static_count + j
            body_name = f"obstacle_{index}"
            body_id = _find_named_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name, model.nbody)
            if body_id < 0:
                raise RuntimeError(f"Missing MuJoCo body for dynamic obstacle {body_name!r}.")
            mocap_id = int(model.body_mocapid[body_id])
            if mocap_id < 0:
                raise RuntimeError(f"Dynamic obstacle {body_name!r} was not compiled as a mocap body.")
            self.mocap_ids.append(mocap_id)

    @property
    def active(self) -> bool:
        return len(self.specs) > 0

    def apply(self, t_per_env: torch.Tensor) -> None:
        """t_per_env: 1D tensor, one simulation time (seconds) per parallel env."""
        if not self.active:
            return
        times = t_per_env.detach().cpu().tolist()
        mocap_pos = self.raw_env.sim.data.mocap_pos
        centers_xy: list[list[tuple[float, float]]] = []  # [obstacle][env] -> (x, y)
        for spec, mocap_id in zip(self.specs, self.mocap_ids):
            positions = [spec.position_at(float(t)) for t in times]
            centers_xy.append(positions)
            xs = torch.tensor([p[0] for p in positions], device=mocap_pos.device, dtype=mocap_pos.dtype)
            ys = torch.tensor([p[1] for p in positions], device=mocap_pos.device, dtype=mocap_pos.dtype)
            if mocap_pos.ndim == 3:
                mocap_pos[:, mocap_id, 0] = xs
                mocap_pos[:, mocap_id, 1] = ys
            elif mocap_pos.ndim == 2:
                mocap_pos[mocap_id, 0] = xs[0]
                mocap_pos[mocap_id, 1] = ys[0]
            else:
                raise RuntimeError(f"Unexpected mocap_pos shape: {tuple(mocap_pos.shape)}")

        if hasattr(self.env, "obstacle_centers"):
            centers_t = torch.tensor(
                centers_xy, device=self.env.device, dtype=self.env.obstacle_centers.dtype
            ).permute(1, 0, 2)  # (num_envs, num_dynamic, 2)
            self.env.obstacle_centers[:, self.static_count : self.static_count + len(self.specs), :] = centers_t

        # Recompute forward kinematics for the new mocap positions, then
        # invalidate MJLab's per-step sensor cache and explicitly re-raycast
        # -- matching the same pattern verify_parallel_lidar_geometry uses
        # after any geometry change.
        self.raw_env.sim.forward()
        self.raw_env.scene.update(dt=0.0)
        sense = getattr(self.raw_env.sim, "sense", None)
        if callable(sense):
            sense()


def build_environment(
    *,
    task_id: str,
    low_checkpoint: str,
    method: str,
    episode_duration_s: float,
    device: str,
    map_name: str | None = None,
    laser_num_rays: int = 61,
    laser_fov_deg: float = 180.0,
    enable_lidar_noise: bool = False,
    hard_clip_output: bool = True,
    collision_geometry: str = "ellipse",
    collision_circle_radius_m: float = 0.30,
    collision_margin_m: float = 0.05,
    collision_termination_mode: str = "circle",
    qp_backend: str = "active_set",
    num_envs: int = 1,
    terminal_backup_mode: str = "legacy_inflated_ellipse",
    terminal_geom_scale: float = 1.0,
    terminal_rec_scale: float = 1.0,
    legacy_projection: bool = False,
    action_max_vx: float = 0.5,
    action_min_vx: float | None = None,
    action_max_vy: float = 0.3,
    action_max_wz: float = 0.5,
    backup_margin: float | None = None,
    high_level_dt: float | None = None,
    integration_dt: float | None = None,
    backup_rollout_substeps: int | None = None,
    laser_debug_vis: bool = False,
):
    configure_torch_backends()

    arena = None
    if map_name is not None:
        install_physical_map(map_name)
        arena = make_map_arena(map_name)

    cfg = make_laser_config(
        method,
        episode_duration_s,
        arena,
        laser_num_rays=laser_num_rays,
        laser_fov_deg=laser_fov_deg,
        enable_lidar_noise=enable_lidar_noise,
        hard_clip_output=hard_clip_output,
        collision_geometry=collision_geometry,
        collision_circle_radius_m=collision_circle_radius_m,
        collision_margin_m=collision_margin_m,
        collision_termination_mode=collision_termination_mode,
        qp_backend=qp_backend,
        terminal_backup_mode=terminal_backup_mode,
        terminal_geom_scale=terminal_geom_scale,
        terminal_rec_scale=terminal_rec_scale,
        legacy_projection=legacy_projection,
        action_max_vx=action_max_vx,
        action_min_vx=action_min_vx,
        action_max_vy=action_max_vy,
        action_max_wz=action_max_wz,
        backup_margin=backup_margin,
        high_level_dt=high_level_dt,
        integration_dt=integration_dt,
        backup_rollout_substeps=backup_rollout_substeps,
        laser_debug_vis=laser_debug_vis,
    )

    env_cfg = load_env_cfg(task_id, play=True)
    low_agent_cfg = load_rl_cfg(task_id)

    if map_name is not None:
        env_cfg.scene.entities["navigation_arena"] = physical_arena_cfg.get_go2_navigation_arena_cfg()
        static_count = static_obstacle_count(map_name)
        for offset, obstacle in enumerate(UNSEEN_MAPS[map_name][static_count:]):
            index = static_count + offset
            half_height = 0.5 * float(obstacle.height)
            xml = f"""
            <mujoco model="dynamic_eval_obstacle_{index}">
              <worldbody>
                <body name="obstacle_{index}" mocap="true" pos="0 0 0">
                  <geom name="obstacle_{index}_collision" type="cylinder"
                        size="{float(obstacle.radius)} {half_height}"
                        rgba="0.85 0.15 0.08 1" contype="1" conaffinity="1"
                        condim="3" friction="0.8 0.05 0.001" group="0"/>
                </body>
              </worldbody>
            </mujoco>
            """
            env_cfg.scene.entities[f"dynamic_eval_obstacle_{index}"] = EntityCfg(
                spec_fn=lambda xml=xml: mujoco.MjSpec.from_string(xml),
                init_state=EntityCfg.InitialStateCfg(
                    pos=(float(obstacle.x), float(obstacle.y), half_height)
                ),
            )

    if int(num_envs) < 1:
        raise ValueError(f"num_envs must be positive, got {num_envs}.")
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.terminations = {}

    # Isolate obstacle-layout generalization. Remove random disturbances.
    for event_name in (
        "push_robot",
        "randomize_terrain",
        "foot_friction",
        "encoder_bias",
        "base_com",
    ):
        if event_name in env_cfg.events:
            del env_cfg.events[event_name]

    add_navigation_laser_sensor(env_cfg, cfg)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    low_env = RslRlVecEnvWrapper(raw_env, clip_actions=low_agent_cfg.clip_actions)
    low_policy = load_frozen_low_level_policy(task_id, low_env, low_checkpoint, device)
    command_if = MjlabVelocityCommandInterface(low_env, command_name="twist")

    env = LaserSacNavigationEnv(
        low_env,
        low_policy,
        command_if,
        cfg.navigation,
        reward_cfg=cfg.reward,
        laser_cfg=cfg.laser,
        safety_cfg=cfg.safety,
        contact_collision_cfg=cfg.contact_collision,
    )

    dynamic_runtime = None
    if map_name is not None and DYNAMIC_OBSTACLE_SPECS.get(map_name):
        dynamic_runtime = DynamicObstacleRuntime(raw_env=raw_env, env=env, map_name=map_name)
        # Force every dynamic obstacle's mocap position to its t=0 value now,
        # rather than trusting EntityCfg.init_state to have applied it to
        # data.mocap_pos (model.body_pos stays (0, 0, 0) for a mocap body at
        # compile time regardless), so verify_compiled_physical_map /
        # verify_parallel_lidar_geometry see correct geometry immediately.
        dynamic_runtime.apply(torch.zeros(env_cfg.scene.num_envs, device=device))

    return raw_env, low_env, env, cfg, dynamic_runtime


def load_actor(*, env: LaserSacNavigationEnv, cfg: LaserSacConfig, checkpoint_path: str, device: str) -> LaserSafeSacActor:
    obs = env.actor_tensor(env.get_observations())
    nav = env.navigation_cfg
    actor = LaserSafeSacActor(
        obs_dim=obs.shape[-1],
        hidden_dims=cfg.sac.hidden_dims,
        action_limits=(nav.action.max_vx, nav.action.max_vy, nav.action.max_wz),
        ray_angles=env.ray_angles,
        range_start=8,
        num_rays=cfg.laser.num_rays,
        max_range=cfg.laser.max_range,
        safety_cfg=cfg.safety,
        method=cfg.method,
        log_std_min=cfg.sac.log_std_min,
        log_std_max=cfg.sac.log_std_max,
        min_vx=nav.action.min_vx,
    ).to(device)

    path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_laser = dict(checkpoint.get("config", {}).get("laser") or {})
    current_laser = dict(asdict(cfg)["laser"])
    # debug_vis is a pure MuJoCo-viewer rendering toggle (draws ray lines);
    # it never affects the ray-casting math, noise, or sensor values the
    # policy actually observes, so it's fine to view with rays on/off
    # regardless of what a checkpoint happened to be trained/evaluated with.
    saved_laser.pop("debug_vis", None)
    current_laser.pop("debug_vis", None)
    if saved_laser != current_laser:
        raise ValueError(
            f"Checkpoint laser configuration does not match evaluation: "
            f"checkpoint={saved_laser}, evaluation={current_laser}"
        )
    saved_safety_raw = checkpoint.get("config", {}).get("safety")
    saved_safety = dict(saved_safety_raw) if isinstance(saved_safety_raw, dict) else saved_safety_raw
    current_safety = asdict(cfg)["safety"]
    # Solver backend and backward estimator are runtime choices and do not
    # alter actor weights or QP rows. Training normally uses active_set+STE;
    # deployment can use exact OSQP without invalidating the checkpoint.
    # cf_straight_through is the same kind of backward-pass-only choice for
    # the CF filter (verified: forward output is bit-identical regardless of
    # its value, only the training-time gradient differs) -- a bcbf_cf
    # ablation checkpoint trained with it True must still be evaluable here.
    for runtime_key in ("qp_backend", "qp_straight_through", "cf_straight_through"):
        if isinstance(saved_safety, dict):
            saved_safety.pop(runtime_key, None)
        current_safety.pop(runtime_key, None)
    # Historical CF checkpoints predate the collision-geometry and QP-solver
    # fields.  Missing new fields are compatible, and every field actually
    # recorded in the checkpoint must still agree -- EXCEPT collision_geometry
    # itself (see the pop below): it's a pure offline analytical-outcome
    # reclassification choice regardless of a checkpoint's vintage, so a
    # legacy checkpoint (implicitly trained/evaluated under the 0.30 m circle
    # definition, before this field existed) is just as eligible for
    # reclassification under --collision-geometry ellipse as a checkpoint
    # that explicitly recorded "circle". No blocking check needed here; the
    # pop below already makes both old and new checkpoints indifferent to it.
    # collision_geometry/collision_margin_m are a pure offline analytical
    # outcome-classification choice (see the "Analytical overlap is retained
    # strictly as an offline diagnostic" note in sac_navigation_env.py): the
    # formal safety violation/reward -- and the actor weights themselves --
    # never depend on it, only the local LiDAR LSE barrier does. Free to
    # swap at eval time without invalidating the checkpoint (2026-09-09,
    # deliberate diagnostic run comparing circle vs. ellipse collision-outcome
    # classification for an already-trained policy). Popped only after the
    # legacy-checkpoint check above, which still needs to see the real value.
    for diagnostic_key in ("collision_geometry", "collision_margin_m"):
        if isinstance(saved_safety, dict):
            saved_safety.pop(diagnostic_key, None)
        current_safety.pop(diagnostic_key, None)
    safety_mismatches = {}
    if isinstance(saved_safety, dict):
        safety_mismatches = {
            key: (value, current_safety.get(key, "<missing>"))
            for key, value in saved_safety.items()
            if key not in current_safety or current_safety[key] != value
        }
    elif saved_safety != current_safety:
        safety_mismatches = {"safety": (saved_safety, current_safety)}
    if safety_mismatches:
        raise ValueError(
            f"Checkpoint safety configuration does not match evaluation defaults: "
            f"mismatches={safety_mismatches}, checkpoint={saved_safety}, "
            f"evaluation={current_safety}"
        )
    # Unseen-map obstacle positions/radii, arena dimensions, and episode duration
    # intentionally differ from training, so the full navigation dictionaries
    # cannot match.
    # Collision and outcome semantics, however, must remain identical.  Check
    # every scalar that controls obstacle/wall collision sampling, command
    # timing, goal priority, and fall classification before evaluating.
    saved_navigation = checkpoint.get("config", {}).get("navigation")
    current_navigation = asdict(cfg)["navigation"]
    collision_paths = (
        ("arena", "robot_radius"),
        ("arena", "wall_clearance"),
        ("arena", "obstacle_clearance"),
        ("action", "max_vx"),
        ("action", "min_vx"),
        ("action", "max_vy"),
        ("action", "max_wz"),
        ("low_level_dt",),
        ("high_level_dt",),
        ("goal_tolerance",),
        ("minimum_base_height",),
        ("maximum_roll_pitch",),
    )

    def nested_value(mapping: dict[str, Any] | None, path: tuple[str, ...]) -> Any:
        value: Any = mapping
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value

    collision_mismatches = []
    for field_path in collision_paths:
        saved_value = nested_value(saved_navigation, field_path)
        current_value = nested_value(current_navigation, field_path)
        if field_path == ("action", "min_vx") and saved_value is None:
            # Checkpoints trained before min_vx existed predate asymmetric
            # vx bounds entirely -- they were always implicitly symmetric
            # (min_vx == -max_vx). Compare against that implied value
            # instead of raising a spurious mismatch on a field that simply
            # didn't exist yet when they were trained.
            saved_max_vx = nested_value(saved_navigation, ("action", "max_vx"))
            saved_value = None if saved_max_vx is None else -saved_max_vx
        if saved_value != current_value:
            collision_mismatches.append(
                (".".join(field_path), saved_value, current_value)
            )
    if collision_mismatches:
        details = "; ".join(
            f"{name}: checkpoint={saved!r}, evaluation={current!r}"
            for name, saved, current in collision_mismatches
        )
        raise ValueError(
            "Checkpoint collision/outcome configuration does not match evaluation: "
            + details
        )
    # The local scan-point v2 model changes the safety-layer state and must not
    # silently load an older frozen-ray checkpoint.
    actor_state_dict = dict(checkpoint["actor"])
    # 2026-09-13: action_scale (a single symmetric per-axis buffer) was
    # replaced by action_low/action_high/action_center/action_half_range (an
    # asymmetric-capable affine map -- see LaserSafeSacActor.min_vx). Both are
    # derived purely from the (already compatibility-checked above) config,
    # never learned, so a legacy checkpoint predating this change is missing
    # the new keys and carries the old one instead -- fill the new keys from
    # the freshly constructed actor's own (already-correct) buffers before
    # loading, and drop the stale old key. strict=True is kept (this is the
    # only special-cased rename) so any genuine missing/unexpected key --
    # e.g. an actual learned parameter -- still raises.
    actor_state_dict.pop("action_scale", None)
    for buffer_name in ("action_low", "action_high", "action_center", "action_half_range"):
        actor_state_dict.setdefault(buffer_name, actor.state_dict()[buffer_name])
    actor.load_state_dict(actor_state_dict, strict=True)
    actor.eval()
    print(
        f"[LASER ACTOR] method={cfg.method} checkpoint={path} "
        f"safety_model={cfg.safety.geometry_model} "
        f"body_clearance={cfg.safety.footprint_clearance_m:.3f}m"
    )
    return actor


def verify_compiled_physical_map(raw_env, env: LaserSacNavigationEnv, map_name: str) -> None:
    model = raw_env.sim.mj_model
    expected_centers = env.obstacle_centers[0].detach().cpu()
    expected_radii = env.obstacle_radii[0].detach().cpu()
    static_count = static_obstacle_count(map_name)
    compiled_centers = []
    compiled_radii = []

    def find_named_id(object_type, target_name: str, count: int) -> int:
        exact_id = mujoco.mj_name2id(model, object_type, target_name)
        if exact_id >= 0:
            return exact_id
        matches = []
        for object_id in range(count):
            compiled_name = mujoco.mj_id2name(model, object_type, object_id)
            if compiled_name is None:
                continue
            if compiled_name == target_name or compiled_name.endswith("/" + target_name):
                matches.append(object_id)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            return -1
        raise RuntimeError(f"Multiple compiled objects match {target_name!r}: {matches}")

    # Dynamic (mocap) obstacles keep model.body_pos == (0, 0, 0) at compile
    # time -- their real position lives in data.mocap_pos/geom_xpos, applied
    # via EntityCfg.init_state at env creation. Force one forward pass so
    # that initial mocap placement has propagated before reading geom_xpos.
    if static_count < expected_centers.shape[0]:
        raw_env.sim.forward()
    geom_xpos = raw_env.sim.data.geom_xpos

    for index in range(expected_centers.shape[0]):
        body_name = f"obstacle_{index}"
        geom_name = f"obstacle_{index}_collision"
        body_id = find_named_id(mujoco.mjtObj.mjOBJ_BODY, body_name, model.nbody)
        geom_id = find_named_id(mujoco.mjtObj.mjOBJ_GEOM, geom_name, model.ngeom)
        if body_id < 0:
            raise RuntimeError(f"Missing MuJoCo body {body_name!r} for map {map_name!r}.")
        if geom_id < 0:
            raise RuntimeError(f"Missing MuJoCo geom {geom_name!r} for map {map_name!r}.")
        if index < static_count:
            compiled_centers.append([float(model.body_pos[body_id, 0]), float(model.body_pos[body_id, 1])])
        else:
            compiled_centers.append(
                [float(geom_xpos[0, geom_id, 0]), float(geom_xpos[0, geom_id, 1])]
            )
        compiled_radii.append(float(model.geom_size[geom_id, 0]))

    compiled_centers_t = torch.tensor(compiled_centers)
    compiled_radii_t = torch.tensor(compiled_radii)
    if not torch.allclose(compiled_centers_t, expected_centers, atol=1.0e-6, rtol=0.0):
        print("Compiled centers:")
        print(compiled_centers_t)
        print("Expected centers:")
        print(expected_centers)
        raise RuntimeError(f"Compiled MuJoCo obstacle centers do not match map {map_name!r}.")
    if not torch.allclose(compiled_radii_t, expected_radii, atol=1.0e-6, rtol=0.0):
        raise RuntimeError(f"Compiled MuJoCo obstacle radii do not match map {map_name!r}.")
    print(f"[PHYSICAL GEOMETRY CHECK] map={map_name} OK")


def verify_parallel_lidar_geometry(
    raw_env,
    env: LaserSacNavigationEnv,
    map_name: str,
    *,
    tolerance_m: float = 0.03,
) -> None:
    """Check every vector world's physical geometry and raw LiDAR distances."""
    sim = raw_env.sim
    sim.forward()
    sense = getattr(sim, "sense", None)
    if callable(sense):
        sense()
    sensor = raw_env.scene[env.laser_obs_cfg.sensor_name]
    # Materialize the sensor data and its exact cached ray origins/directions.
    raw = sensor.data.distances
    max_range = float(env.laser_obs_cfg.max_range)
    measured = torch.where(
        torch.isfinite(raw) & (raw >= 0.0) & (raw < max_range),
        raw,
        torch.full_like(raw, max_range),
    )
    origins = getattr(sensor, "_cached_world_origins", None)
    rays = getattr(sensor, "_cached_world_rays", None)
    if origins is None or rays is None:
        raise RuntimeError("RayCastSensor did not expose its current cast rays.")
    directions = rays / torch.clamp_min(torch.linalg.norm(rays, dim=-1, keepdim=True), 1e-12)

    model = sim.mj_model
    obstacle_geom_ids: list[int] = []
    for index in range(env.obstacle_centers.shape[1]):
        target = f"obstacle_{index}_collision"
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, target)
        if geom_id < 0:
            matches = [
                candidate
                for candidate in range(model.ngeom)
                if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, candidate))
                and (name == target or name.endswith("/" + target))
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Could not uniquely resolve static geom {target!r}: {matches}")
            geom_id = matches[0]
        obstacle_geom_ids.append(geom_id)

    geom_xpos = sim.data.geom_xpos
    geom_size = sim.model.geom_size
    physical_centers = torch.stack(
        [geom_xpos[:, geom_id, :2] for geom_id in obstacle_geom_ids], dim=1
    )
    center_error = torch.abs(physical_centers - env.obstacle_centers)
    if float(center_error.max()) > 1e-4:
        flat = int(center_error.argmax())
        component = flat % 2
        slot = (flat // 2) % len(obstacle_geom_ids)
        env_id = flat // (2 * len(obstacle_geom_ids))
        raise RuntimeError(
            f"Parallel physical-map mismatch: env={env_id}, obstacle={slot}, "
            f"component={'xy'[component]}, error={float(center_error.max()):.6f} m"
        )

    center_z = torch.stack(
        [geom_xpos[:, geom_id, 2] for geom_id in obstacle_geom_ids], dim=1
    )
    # MuJoCo geom_size is (radius, half_height, unused) for a cylinder but
    # (half_x, half_y, half_z) for a box -- index 1 is only really
    # "half_height" for circles. Read the per-slot-shape-correct component.
    slot_shapes = getattr(env, "_slot_shapes", None)
    square_mask = torch.tensor(
        [shape == "square" for shape in (slot_shapes or ())] or [False] * env.obstacle_centers.shape[1],
        device=env.device, dtype=torch.bool,
    )
    half_height_dim1 = torch.stack(
        [geom_size[:, geom_id, 1] for geom_id in obstacle_geom_ids], dim=1
    )
    half_height_dim2 = torch.stack(
        [geom_size[:, geom_id, 2] for geom_id in obstacle_geom_ids], dim=1
    )
    half_height = torch.where(square_mask[None, :], half_height_dim2, half_height_dim1)
    origin_xy = origins[..., :2]
    direction_xy = directions[..., :2]
    oc = origin_xy[:, None, :, :] - env.obstacle_centers[:, :, None, :]
    d = direction_xy[:, None, :, :]
    a = d.square().sum(dim=-1)
    b = (oc * d).sum(dim=-1)
    c = oc.square().sum(dim=-1) - env.obstacle_radii[:, :, None].square()
    discriminant = b.square() - a * c
    root = torch.sqrt(torch.clamp_min(discriminant, 0.0))
    denominator = torch.clamp_min(a, 1e-12)
    near = (-b - root) / denominator
    far = (-b + root) / denominator
    circle_ray_t = torch.where(near >= 0.0, near, far)
    circle_hit_z = origins[:, None, :, 2] + circle_ray_t * directions[:, None, :, 2]
    circle_valid = (
        env.obstacle_mask[:, :, None]
        & (a > 1e-12)
        & (discriminant >= 0.0)
        & (circle_ray_t >= 0.0)
        & (circle_hit_z >= center_z[:, :, None] - half_height[:, :, None])
        & (circle_hit_z <= center_z[:, :, None] + half_height[:, :, None])
    )

    # Axis-aligned-box (square obstacle) ray intersection via the slab method,
    # since the circle-only quadratic above never matches a real square's
    # physically-raycast surface. square_mask (per-slot "circle"/"square")
    # selects which formula each obstacle slot actually uses; the real LiDAR
    # sensor is a true MuJoCo raycast and is already shape-correct regardless
    # -- only this analytical cross-check needed updating.
    if bool(square_mask.any()):
        half = env.obstacle_radii[:, :, None]
        cx = env.obstacle_centers[:, :, None, 0]
        cy = env.obstacle_centers[:, :, None, 1]
        ox2, oy2 = origin_xy[:, None, :, 0], origin_xy[:, None, :, 1]
        dxr, dyr = direction_xy[:, None, :, 0], direction_xy[:, None, :, 1]
        box_eps = 1e-9
        safe_dx = torch.where(dxr.abs() > box_eps, dxr, torch.full_like(dxr, box_eps))
        safe_dy = torch.where(dyr.abs() > box_eps, dyr, torch.full_like(dyr, box_eps))
        tx1, tx2 = ((cx - half) - ox2) / safe_dx, ((cx + half) - ox2) / safe_dx
        ty1, ty2 = ((cy - half) - oy2) / safe_dy, ((cy + half) - oy2) / safe_dy
        box_tmin = torch.maximum(torch.minimum(tx1, tx2), torch.minimum(ty1, ty2))
        box_tmax = torch.minimum(torch.maximum(tx1, tx2), torch.maximum(ty1, ty2))
        box_ray_t = torch.where(box_tmin >= 0.0, box_tmin, box_tmax)
        box_hit_z = origins[:, None, :, 2] + box_ray_t * directions[:, None, :, 2]
        box_valid = (
            env.obstacle_mask[:, :, None]
            & (box_tmax >= box_tmin)
            & (box_ray_t >= 0.0)
            & (box_hit_z >= center_z[:, :, None] - half_height[:, :, None])
            & (box_hit_z <= center_z[:, :, None] + half_height[:, :, None])
        )
        ray_t = torch.where(square_mask[None, :, None], box_ray_t, circle_ray_t)
        valid = torch.where(square_mask[None, :, None], box_valid, circle_valid)
    else:
        ray_t, valid = circle_ray_t, circle_valid

    obstacle_t = torch.where(
        valid, ray_t, torch.full_like(ray_t, float("inf"))
    ).amin(dim=1)

    arena = env.navigation_cfg.arena
    ox, oy, oz = origins.unbind(dim=-1)
    dx, dy, dz = directions.unbind(dim=-1)
    wall_candidates = []
    eps = 1e-9
    for boundary in (arena.x_min, arena.x_max):
        t = (boundary - ox) / torch.where(dx.abs() > eps, dx, torch.ones_like(dx))
        hit_y = oy + t * dy
        hit_z = oz + t * dz
        valid_wall = (
            (dx.abs() > eps) & (t >= 0.0)
            & (hit_y >= arena.y_min) & (hit_y <= arena.y_max)
            & (hit_z >= 0.0) & (hit_z <= float(arena.wall_height))
        )
        wall_candidates.append(torch.where(valid_wall, t, torch.full_like(t, float("inf"))))
    for boundary in (arena.y_min, arena.y_max):
        t = (boundary - oy) / torch.where(dy.abs() > eps, dy, torch.ones_like(dy))
        hit_x = ox + t * dx
        hit_z = oz + t * dz
        valid_wall = (
            (dy.abs() > eps) & (t >= 0.0)
            & (hit_x >= arena.x_min) & (hit_x <= arena.x_max)
            & (hit_z >= 0.0) & (hit_z <= float(arena.wall_height))
        )
        wall_candidates.append(torch.where(valid_wall, t, torch.full_like(t, float("inf"))))
    wall_t = torch.stack(wall_candidates).amin(dim=0)
    expected = torch.minimum(obstacle_t, wall_t).clamp(max=max_range)
    error = torch.abs(measured - expected)
    max_error = float(error.max())
    if max_error > tolerance_m:
        flat = int(error.argmax())
        ray = flat % measured.shape[1]
        env_id = flat // measured.shape[1]
        raise RuntimeError(
            "Parallel LiDAR/analytical geometry mismatch: "
            f"map={map_name}, env={env_id}, ray={ray}, "
            f"measured={float(measured[env_id, ray]):.4f} m, "
            f"expected={float(expected[env_id, ray]):.4f} m, error={max_error:.4f} m"
        )
    print(
        f"[PARALLEL LIDAR CHECK] map={map_name} envs={env.num_envs} "
        f"max_error={max_error:.4f}m OK"
    )


def physical_clearance(
    env: LaserSacNavigationEnv, position: torch.Tensor, env_id: int = 0
) -> float:
    cfg = env.navigation_cfg
    robot_radius = float(getattr(cfg.arena, "robot_radius", 0.30))
    p = position.reshape(1, 2)

    obstacle_distance = torch.linalg.norm(
        p[:, None, :] - env.obstacle_centers[env_id : env_id + 1], dim=-1
    )
    obstacle_clearance = (
        obstacle_distance - env.obstacle_radii[env_id : env_id + 1] - robot_radius
    )
    obstacle_clearance = torch.where(
        env.obstacle_mask[env_id : env_id + 1],
        obstacle_clearance,
        torch.full_like(obstacle_clearance, float("inf")),
    )

    x = p[0, 0]
    y = p[0, 1]
    wall_clearance = torch.stack(
        (
            x - cfg.arena.x_min - robot_radius,
            cfg.arena.x_max - x - robot_radius,
            y - cfg.arena.y_min - robot_radius,
            cfg.arena.y_max - y - robot_radius,
        )
    )
    value = torch.minimum(obstacle_clearance.min(), wall_clearance.min())
    return float(value.detach().cpu())


def clip_box_to_arena(
    box: tuple[float, float, float, float],
    arena: ArenaConfig,
) -> tuple[float, float, float, float]:
    """Clip a sampling box to the usable arena interior."""
    x0, x1, y0, y1 = map(float, box)
    margin = float(getattr(arena, "wall_clearance", 0.0))
    x0 = max(x0, float(arena.x_min) + margin)
    x1 = min(x1, float(arena.x_max) - margin)
    y0 = max(y0, float(arena.y_min) + margin)
    y1 = min(y1, float(arena.y_max) - margin)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"Sampling box {box} is empty after clipping to arena {arena}.")
    return x0, x1, y0, y1


def sample_point_from_box(
    rng: random.Random,
    box: tuple[float, float, float, float],
) -> torch.Tensor:
    x0, x1, y0, y1 = box
    return torch.tensor(
        [rng.uniform(x0, x1), rng.uniform(y0, y1)],
        dtype=torch.float32,
    )


def generate_manifest(
    *,
    env: LaserSacNavigationEnv,
    arena: ArenaConfig,
    map_name: str,
    count: int,
    seed: int,
    output_path: Path,
    min_endpoint_clearance: float = 0.10,
    block_margin: float = 0.10,
    require_blocked_path: bool = True,
    blocked_t_min: float = 0.15,
    blocked_t_max: float = 0.85,
    max_sampling_attempts: int = 10000,
    yaw_jitter_rad: float = 0.25,
) -> list[dict[str, object]]:
    """Generate structured start/goal cases for one map.

    Unlike the original random reset-based manifest, this function samples
    start and goal from map-specific boxes, then rejects cases that do not
    exercise the obstacle layout.
    """
    if map_name not in STRUCTURED_SCENARIO_REGIONS:
        raise KeyError(f"No structured start/goal regions defined for map {map_name!r}.")

    env.clear_fixed_scenario()
    rows: list[dict[str, object]] = []
    rng = random.Random(int(seed))

    regions = STRUCTURED_SCENARIO_REGIONS[map_name]
    start_box_raw = regions["start"]
    goal_box_raw = regions["goal"]
    start_box = clip_box_to_arena(start_box_raw, arena)
    goal_box = clip_box_to_arena(goal_box_raw, arena)

    centers_cpu = env.obstacle_centers[0].detach().cpu()
    radii_cpu = env.obstacle_radii[0].detach().cpu()
    mask_cpu = env.obstacle_mask[0].detach().cpu().bool()
    obstacle_margin = float(getattr(arena, "obstacle_clearance", 0.30))

    def obstacle_clearance(point: torch.Tensor) -> float:
        point_cpu = point.detach().cpu().to(dtype=centers_cpu.dtype)
        distances = torch.linalg.norm(centers_cpu - point_cpu.unsqueeze(0), dim=-1)
        clearances = distances - radii_cpu - obstacle_margin
        clearances = clearances[mask_cpu]
        return float(clearances.min().item()) if clearances.numel() else math.inf

    def wall_clearance(point: torch.Tensor) -> float:
        x = float(point[0])
        y = float(point[1])
        margin = float(arena.wall_clearance)
        return float(min(
            x - (arena.x_min + margin),
            (arena.x_max - margin) - x,
            y - (arena.y_min + margin),
            (arena.y_max - margin) - y,
        ))

    def total_clearance(point: torch.Tensor) -> float:
        return min(obstacle_clearance(point), wall_clearance(point))

    def straight_path_min_clearance(start: torch.Tensor, goal: torch.Tensor, samples: int = 201) -> float:
        minimum = math.inf
        for fraction in torch.linspace(0.0, 1.0, samples, dtype=start.dtype):
            point = start + fraction * (goal - start)
            minimum = min(minimum, total_clearance(point))
        return float(minimum)

    def blocking_obstacle_details(start: torch.Tensor, goal: torch.Tensor) -> dict[str, object]:
        """Return nearest obstacle-to-segment details and whether the segment is blocked.

        A segment is counted as blocked/near-blocked when an obstacle lies between
        the endpoints and the center-to-segment distance is less than
        radius + obstacle_margin + block_margin.
        """
        seg = goal - start
        seg_len_sq = float(torch.dot(seg, seg).item())
        if seg_len_sq <= 1.0e-12:
            return {
                "direct_path_blocked": False,
                "blocking_obstacle_id": -1,
                "blocking_t": math.nan,
                "blocking_distance_m": math.nan,
                "blocking_clearance_m": math.nan,
                "blocking_threshold_m": math.nan,
            }

        best = {
            "direct_path_blocked": False,
            "blocking_obstacle_id": -1,
            "blocking_t": math.nan,
            "blocking_distance_m": math.inf,
            "blocking_clearance_m": math.inf,
            "blocking_threshold_m": math.nan,
        }

        for obstacle_id, (center, radius, is_valid) in enumerate(zip(centers_cpu, radii_cpu, mask_cpu)):
            if not bool(is_valid):
                continue
            center = center.to(dtype=start.dtype)
            t = float(torch.dot(center - start, seg).item() / seg_len_sq)
            if t < blocked_t_min or t > blocked_t_max:
                continue
            closest = start + t * seg
            distance = float(torch.linalg.norm(center - closest).item())
            radius_value = float(radius.item())
            threshold = radius_value + obstacle_margin + float(block_margin)
            clearance = distance - radius_value - obstacle_margin
            blocked = distance <= threshold

            # Prefer a true blocking obstacle. Otherwise keep the closest candidate
            # for diagnostics.
            current_best_blocked = bool(best["direct_path_blocked"])
            if (blocked and not current_best_blocked) or (blocked == current_best_blocked and clearance < float(best["blocking_clearance_m"])):
                best = {
                    "direct_path_blocked": bool(blocked),
                    "blocking_obstacle_id": int(obstacle_id),
                    "blocking_t": t,
                    "blocking_distance_m": distance,
                    "blocking_clearance_m": clearance,
                    "blocking_threshold_m": threshold,
                }

        return best

    attempts = 0
    while len(rows) < count and attempts < max_sampling_attempts:
        attempts += 1
        start_point = sample_point_from_box(rng, start_box)
        goal_point = sample_point_from_box(rng, goal_box)

        start_clearance = total_clearance(start_point)
        if start_clearance < min_endpoint_clearance:
            continue
        goal_clearance = total_clearance(goal_point)
        if goal_clearance < min_endpoint_clearance:
            continue

        straight_line_distance = float(torch.linalg.norm(goal_point - start_point).item())
        if straight_line_distance <= 1.0:
            continue

        block_info = blocking_obstacle_details(start_point, goal_point)
        if require_blocked_path and not bool(block_info["direct_path_blocked"]):
            continue

        direct_path_clearance = straight_path_min_clearance(start_point, goal_point)
        yaw_nominal = math.atan2(float(goal_point[1] - start_point[1]), float(goal_point[0] - start_point[0]))
        yaw = yaw_nominal + rng.uniform(-float(yaw_jitter_rad), float(yaw_jitter_rad))

        case_id = len(rows)
        rows.append({
            "map_name": map_name,
            "case_id": case_id,
            "reset_seed": int(seed + case_id),
            "sampling_attempt": attempts,
            "start_region_x_min": start_box_raw[0],
            "start_region_x_max": start_box_raw[1],
            "start_region_y_min": start_box_raw[2],
            "start_region_y_max": start_box_raw[3],
            "goal_region_x_min": goal_box_raw[0],
            "goal_region_x_max": goal_box_raw[1],
            "goal_region_y_min": goal_box_raw[2],
            "goal_region_y_max": goal_box_raw[3],
            "start_x": float(start_point[0]),
            "start_y": float(start_point[1]),
            "start_yaw": float(yaw),
            "start_yaw_nominal_to_goal": float(yaw_nominal),
            "goal_x": float(goal_point[0]),
            "goal_y": float(goal_point[1]),
            "start_clearance_m": start_clearance,
            "goal_clearance_m": goal_clearance,
            "straight_line_distance_m": straight_line_distance,
            "straight_path_min_clearance_m": direct_path_clearance,
            "direct_path_blocked": int(bool(block_info["direct_path_blocked"])),
            "blocking_obstacle_id": int(block_info["blocking_obstacle_id"]),
            "blocking_t": float(block_info["blocking_t"]),
            "blocking_distance_m": float(block_info["blocking_distance_m"]),
            "blocking_clearance_m": float(block_info["blocking_clearance_m"]),
            "blocking_threshold_m": float(block_info["blocking_threshold_m"]),
            "block_margin_m": float(block_margin),
            "min_endpoint_clearance_m": float(min_endpoint_clearance),
        })

    if len(rows) < count:
        raise RuntimeError(
            f"Could only sample {len(rows)}/{count} valid structured cases for map {map_name!r} "
            f"after {attempts} attempts. Try lowering --min-endpoint-clearance or --block-margin, "
            "or check the start/goal boxes."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    blocked = sum(int(row["direct_path_blocked"]) for row in rows)
    print(f"\n[STRUCTURED MANIFEST] map={map_name}, cases={len(rows)}, file={output_path}")
    print(f"  start_box: {start_box_raw} -> clipped {tuple(round(v, 3) for v in start_box)}")
    print(f"  goal_box:  {goal_box_raw} -> clipped {tuple(round(v, 3) for v in goal_box)}")
    print(f"  attempts: {attempts}")
    print(f"  direct_path_blocked_or_near_blocked: {blocked}/{len(rows)}")
    print(f"  min_start_clearance: {min(float(r['start_clearance_m']) for r in rows):+.3f} m")
    print(f"  min_goal_clearance:  {min(float(r['goal_clearance_m']) for r in rows):+.3f} m")
    print(f"  min_straight_path_clearance: {min(float(r['straight_path_min_clearance_m']) for r in rows):+.3f} m")
    return rows


def generate_manifest_randomized(
    *,
    env: LaserSacNavigationEnv,
    arena: ArenaConfig,
    map_name: str,
    count: int,
    seed: int,
    output_path: Path,
    min_endpoint_clearance: float = 0.10,
    block_margin: float = 0.10,
    require_blocked_path: bool = True,
    blocked_t_min: float = 0.15,
    blocked_t_max: float = 0.85,
    max_sampling_attempts: int = 40000,
    yaw_jitter_rad: float = 0.25,
    min_straight_line_distance: float = 4.0,
) -> list[dict[str, object]]:
    """Generate randomized start/goal cases for one map.

    This is an ADDITIVE alternative to generate_manifest(): instead of
    drawing start/goal from STRUCTURED_SCENARIO_REGIONS's fixed, map-specific
    boxes (a narrow vertical strip near one wall for start, the opposite wall
    for goal), both endpoints are drawn independently from the FULL arena
    interior. Every other validity check (obstacle/wall clearance, a
    configurable minimum straight-line distance, and -- by default -- the
    requirement that the direct start-goal segment actually pass near an
    obstacle) is identical in spirit to generate_manifest(), duplicated here
    rather than shared so that function is never touched by this one.

    generate_manifest() and its STRUCTURED_SCENARIO_REGIONS-based scenario
    banks are left completely unmodified; this function only adds a second,
    independently-seeded and independently-stored scenario bank alongside it
    (see main()'s --randomize-start-goal flag, which writes this bank under
    a distinctly-named "<map_name>_random" directory so it can never resume
    from, or silently overwrite, an existing structured-region manifest).
    """
    env.clear_fixed_scenario()
    rows: list[dict[str, object]] = []
    rng = random.Random(int(seed))

    full_box_raw = (float(arena.x_min), float(arena.x_max), float(arena.y_min), float(arena.y_max))
    start_box = clip_box_to_arena(full_box_raw, arena)
    goal_box = start_box

    centers_cpu = env.obstacle_centers[0].detach().cpu()
    radii_cpu = env.obstacle_radii[0].detach().cpu()
    mask_cpu = env.obstacle_mask[0].detach().cpu().bool()
    obstacle_margin = float(getattr(arena, "obstacle_clearance", 0.30))

    def obstacle_clearance(point: torch.Tensor) -> float:
        point_cpu = point.detach().cpu().to(dtype=centers_cpu.dtype)
        distances = torch.linalg.norm(centers_cpu - point_cpu.unsqueeze(0), dim=-1)
        clearances = distances - radii_cpu - obstacle_margin
        clearances = clearances[mask_cpu]
        return float(clearances.min().item()) if clearances.numel() else math.inf

    def wall_clearance(point: torch.Tensor) -> float:
        x = float(point[0])
        y = float(point[1])
        margin = float(arena.wall_clearance)
        return float(min(
            x - (arena.x_min + margin),
            (arena.x_max - margin) - x,
            y - (arena.y_min + margin),
            (arena.y_max - margin) - y,
        ))

    def total_clearance(point: torch.Tensor) -> float:
        return min(obstacle_clearance(point), wall_clearance(point))

    def straight_path_min_clearance(start: torch.Tensor, goal: torch.Tensor, samples: int = 201) -> float:
        minimum = math.inf
        for fraction in torch.linspace(0.0, 1.0, samples, dtype=start.dtype):
            point = start + fraction * (goal - start)
            minimum = min(minimum, total_clearance(point))
        return float(minimum)

    def blocking_obstacle_details(start: torch.Tensor, goal: torch.Tensor) -> dict[str, object]:
        seg = goal - start
        seg_len_sq = float(torch.dot(seg, seg).item())
        if seg_len_sq <= 1.0e-12:
            return {
                "direct_path_blocked": False,
                "blocking_obstacle_id": -1,
                "blocking_t": math.nan,
                "blocking_distance_m": math.nan,
                "blocking_clearance_m": math.nan,
                "blocking_threshold_m": math.nan,
            }

        best = {
            "direct_path_blocked": False,
            "blocking_obstacle_id": -1,
            "blocking_t": math.nan,
            "blocking_distance_m": math.inf,
            "blocking_clearance_m": math.inf,
            "blocking_threshold_m": math.nan,
        }

        for obstacle_id, (center, radius, is_valid) in enumerate(zip(centers_cpu, radii_cpu, mask_cpu)):
            if not bool(is_valid):
                continue
            center = center.to(dtype=start.dtype)
            t = float(torch.dot(center - start, seg).item() / seg_len_sq)
            if t < blocked_t_min or t > blocked_t_max:
                continue
            closest = start + t * seg
            distance = float(torch.linalg.norm(center - closest).item())
            radius_value = float(radius.item())
            threshold = radius_value + obstacle_margin + float(block_margin)
            clearance = distance - radius_value - obstacle_margin
            blocked = distance <= threshold

            current_best_blocked = bool(best["direct_path_blocked"])
            if (blocked and not current_best_blocked) or (blocked == current_best_blocked and clearance < float(best["blocking_clearance_m"])):
                best = {
                    "direct_path_blocked": bool(blocked),
                    "blocking_obstacle_id": int(obstacle_id),
                    "blocking_t": t,
                    "blocking_distance_m": distance,
                    "blocking_clearance_m": clearance,
                    "blocking_threshold_m": threshold,
                }

        return best

    attempts = 0
    while len(rows) < count and attempts < max_sampling_attempts:
        attempts += 1
        start_point = sample_point_from_box(rng, start_box)
        goal_point = sample_point_from_box(rng, goal_box)

        start_clearance = total_clearance(start_point)
        if start_clearance < min_endpoint_clearance:
            continue
        goal_clearance = total_clearance(goal_point)
        if goal_clearance < min_endpoint_clearance:
            continue

        straight_line_distance = float(torch.linalg.norm(goal_point - start_point).item())
        if straight_line_distance <= float(min_straight_line_distance):
            continue

        block_info = blocking_obstacle_details(start_point, goal_point)
        if require_blocked_path and not bool(block_info["direct_path_blocked"]):
            continue

        direct_path_clearance = straight_path_min_clearance(start_point, goal_point)
        yaw_nominal = math.atan2(float(goal_point[1] - start_point[1]), float(goal_point[0] - start_point[0]))
        yaw_unwrapped = yaw_nominal + rng.uniform(-float(yaw_jitter_rad), float(yaw_jitter_rad))
        # Full-arena start/goal pairs can point in ANY direction, including
        # close to +/-180 degrees -- unlike generate_manifest()'s narrow,
        # consistently-oriented boxes, jitter here can push yaw_nominal past
        # +/-pi. Wrap to the same canonical (-pi, pi] range the environment
        # itself normalizes yaw to, or env.set_fixed_scenario()'s actual yaw
        # will not bitwise-match this saved value (see evaluate_method's/
        # evaluate_method_vectorized's yaw-mismatch sanity check).
        yaw = math.atan2(math.sin(yaw_unwrapped), math.cos(yaw_unwrapped))

        case_id = len(rows)
        rows.append({
            "map_name": map_name,
            "case_id": case_id,
            "reset_seed": int(seed + case_id),
            "sampling_attempt": attempts,
            "start_region_x_min": start_box[0],
            "start_region_x_max": start_box[1],
            "start_region_y_min": start_box[2],
            "start_region_y_max": start_box[3],
            "goal_region_x_min": goal_box[0],
            "goal_region_x_max": goal_box[1],
            "goal_region_y_min": goal_box[2],
            "goal_region_y_max": goal_box[3],
            "start_x": float(start_point[0]),
            "start_y": float(start_point[1]),
            "start_yaw": float(yaw),
            "start_yaw_nominal_to_goal": float(yaw_nominal),
            "goal_x": float(goal_point[0]),
            "goal_y": float(goal_point[1]),
            "start_clearance_m": start_clearance,
            "goal_clearance_m": goal_clearance,
            "straight_line_distance_m": straight_line_distance,
            "straight_path_min_clearance_m": direct_path_clearance,
            "direct_path_blocked": int(bool(block_info["direct_path_blocked"])),
            "blocking_obstacle_id": int(block_info["blocking_obstacle_id"]),
            "blocking_t": float(block_info["blocking_t"]),
            "blocking_distance_m": float(block_info["blocking_distance_m"]),
            "blocking_clearance_m": float(block_info["blocking_clearance_m"]),
            "blocking_threshold_m": float(block_info["blocking_threshold_m"]),
            "block_margin_m": float(block_margin),
            "min_endpoint_clearance_m": float(min_endpoint_clearance),
            "min_straight_line_distance_m": float(min_straight_line_distance),
        })

    if len(rows) < count:
        raise RuntimeError(
            f"Could only sample {len(rows)}/{count} valid randomized cases for map {map_name!r} "
            f"after {attempts} attempts. Try lowering --min-endpoint-clearance, --block-margin, "
            "or --min-random-distance-m, or raising --max-sampling-attempts."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    blocked = sum(int(row["direct_path_blocked"]) for row in rows)
    print(f"\n[RANDOMIZED MANIFEST] map={map_name}, cases={len(rows)}, file={output_path}")
    print(f"  sampling_box (full arena, clipped): {tuple(round(v, 3) for v in start_box)}")
    print(f"  attempts: {attempts}")
    print(f"  direct_path_blocked_or_near_blocked: {blocked}/{len(rows)}")
    print(f"  min_start_clearance: {min(float(r['start_clearance_m']) for r in rows):+.3f} m")
    print(f"  min_goal_clearance:  {min(float(r['goal_clearance_m']) for r in rows):+.3f} m")
    print(f"  min_straight_path_clearance: {min(float(r['straight_path_min_clearance_m']) for r in rows):+.3f} m")
    print(f"  min_straight_line_distance: {min(float(r['straight_line_distance_m']) for r in rows):+.3f} m")
    return rows


def row_to_scenario(row: dict[str, float | int | str]) -> SimpleNamespace:
    return SimpleNamespace(
        name=f"{row.get('map_name', 'map')}_{int(row['case_id']):03d}",
        start_xy=(float(row["start_x"]), float(row["start_y"])),
        start_yaw=float(row["start_yaw"]),
        goal_xy=(float(row["goal_x"]), float(row["goal_y"])),
    )


def evaluate_method(
    *,
    method: str,
    task_id: str,
    low_checkpoint: str,
    high_checkpoint: str,
    manifest: list[dict[str, object]],
    episode_duration_s: float,
    device: str,
    print_every: int,
    output_dir: Path,
    map_name: str,
    log_trajectories: bool,
    laser_num_rays: int,
    laser_fov_deg: float,
    enable_lidar_noise: bool,
    hard_clip_output: bool,
    collision_geometry: str,
    collision_circle_radius_m: float,
    collision_margin_m: float,
    collision_termination_mode: str,
    qp_backend: str,
    terminal_backup_mode: str = "legacy_inflated_ellipse",
    terminal_geom_scale: float = 1.0,
    terminal_rec_scale: float = 1.0,
    legacy_projection: bool = False,
    action_max_vx: float = 0.5,
    action_min_vx: float | None = None,
    action_max_vy: float = 0.3,
    action_max_wz: float = 0.5,
    backup_margin: float | None = None,
    high_level_dt: float | None = None,
    integration_dt: float | None = None,
    backup_rollout_substeps: int | None = None,
    disable_safety_filter: bool = False,
) -> Path:
    base_seed = int(manifest[0]["reset_seed"]) if manifest else 0
    set_seed(base_seed)

    raw_env, low_env, env, cfg, _dynamic_runtime = build_environment(
        task_id=task_id,
        low_checkpoint=low_checkpoint,
        method=method,
        episode_duration_s=episode_duration_s,
        device=device,
        map_name=map_name,
        laser_num_rays=laser_num_rays,
        laser_fov_deg=laser_fov_deg,
        enable_lidar_noise=enable_lidar_noise,
        hard_clip_output=hard_clip_output,
        collision_geometry=collision_geometry,
        collision_circle_radius_m=collision_circle_radius_m,
        collision_margin_m=collision_margin_m,
        collision_termination_mode=collision_termination_mode,
        qp_backend=qp_backend,
        terminal_backup_mode=terminal_backup_mode,
        terminal_geom_scale=terminal_geom_scale,
        terminal_rec_scale=terminal_rec_scale,
        legacy_projection=legacy_projection,
        action_max_vx=action_max_vx,
        action_min_vx=action_min_vx,
        action_max_vy=action_max_vy,
        action_max_wz=action_max_wz,
        backup_margin=backup_margin,
        high_level_dt=high_level_dt,
        integration_dt=integration_dt,
        backup_rollout_substeps=backup_rollout_substeps,
    )
    if _dynamic_runtime is not None and _dynamic_runtime.active:
        raise NotImplementedError(
            "Dynamic-obstacle maps are only supported through evaluate_method_vectorized "
            "(the batched entry point used by every eval CLI path this session)."
        )

    records: list[dict[str, object]] = []
    trajectory_records: list[dict[str, object]] = []

    try:
        verify_compiled_physical_map(raw_env, env, map_name)
        actor = load_actor(env=env, cfg=cfg, checkpoint_path=high_checkpoint, device=device)
        actor.disable_safety_filter = disable_safety_filter
        warmup_actor_timing(actor, env)

        for index, case in enumerate(manifest):
            case_id = int(case["case_id"])
            set_seed(int(case["reset_seed"]))
            env.set_fixed_scenario(row_to_scenario(case))
            obs = env.get_observations()

            actual_position, actual_yaw = env.safety_context()
            actual_goal = env.goal_positions[0]
            expected_start = torch.tensor([float(case["start_x"]), float(case["start_y"])], device=actual_position.device, dtype=actual_position.dtype)
            expected_goal = torch.tensor([float(case["goal_x"]), float(case["goal_y"])], device=actual_goal.device, dtype=actual_goal.dtype)
            expected_yaw = torch.tensor(float(case["start_yaw"]), device=actual_yaw.device, dtype=actual_yaw.dtype)
            if not torch.allclose(actual_position[0], expected_start, atol=1.0e-4, rtol=0.0):
                raise RuntimeError(f"{method}/{map_name}/case {case_id}: start mismatch.")
            if not torch.allclose(actual_goal, expected_goal, atol=1.0e-4, rtol=0.0):
                raise RuntimeError(f"{method}/{map_name}/case {case_id}: goal mismatch.")
            if not torch.allclose(actual_yaw[0], expected_yaw, atol=1.0e-4, rtol=0.0):
                raise RuntimeError(f"{method}/{map_name}/case {case_id}: yaw mismatch.")

            previous_position = actual_position[0].detach().clone()
            path_length = 0.0
            minimum_clearance = physical_clearance(env, previous_position)
            active_steps = 0
            correction_sum = 0.0
            return_value = 0.0
            step_count = 0
            policy_latencies_ms: list[float] = []
            filter_latencies_ms: list[float] = []
            inference_latencies_ms: list[float] = []
            velocity_error_sum = 0.0

            outcome = "other_failure"
            success = collision = safety_violation = timeout = fallen = outside = False
            duration_s = 0.0

            with torch.no_grad():
                while True:
                    actor_obs = env.actor_tensor(obs)
                    nominal, safe, safety, policy_ms, filter_ms, inference_ms = (
                        timed_deterministic_actor(actor, actor_obs)
                    )
                    policy_latencies_ms.append(policy_ms)
                    filter_latencies_ms.append(filter_ms)
                    inference_latencies_ms.append(inference_ms)

                    next_obs, reward, done, extras = env.step_safe(nominal, safe, safety)
                    step_diagnostics = extras["collision_trajectory_step"]
                    measured_command = step_diagnostics["measured_command"]
                    velocity_error = torch.linalg.norm(measured_command[0] - safe[0])
                    velocity_error_value = float(velocity_error.detach().cpu())
                    velocity_error_sum += velocity_error_value
                    step_count += 1
                    step_reward = float(reward[0].detach().cpu())
                    return_value += step_reward

                    if bool(done[0]):
                        terminal_positions = extras.get("completed_terminal_positions")
                        if terminal_positions is not None and terminal_positions.numel() > 0:
                            endpoint = terminal_positions[0]
                        else:
                            endpoint = env.safety_context()[0][0]
                    else:
                        endpoint = env.safety_context()[0][0]

                    step_start_position = previous_position.detach().clone()
                    path_length += float(torch.linalg.norm(endpoint - previous_position).detach().cpu())
                    previous_position = endpoint.detach().clone()
                    clearance = physical_clearance(env, endpoint)
                    minimum_clearance = min(minimum_clearance, clearance)

                    correction = torch.linalg.norm(safe[0] - nominal[0])
                    correction_value = float(correction.detach().cpu())
                    correction_sum += correction_value
                    filter_active = False if method == "vanilla" else correction_value > 1.0e-8
                    if filter_active:
                        active_steps += 1

                    if log_trajectories:
                        pos, yaw = env.safety_context()
                        trajectory_records.append({
                            "method": method,
                            "map_name": map_name,
                            "evaluation_seed": int(case.get("evaluation_seed", case["reset_seed"])),
                            "case_id": case_id,
                            "reset_seed": int(case["reset_seed"]),
                            "step": step_count,
                            "time_s": step_count * env.navigation_cfg.high_level_dt,
                            "pre_x_m": float(step_start_position[0].detach().cpu()),
                            "pre_y_m": float(step_start_position[1].detach().cpu()),
                            "post_x_m": float(endpoint[0].detach().cpu()),
                            "post_y_m": float(endpoint[1].detach().cpu()),
                            "yaw_rad": float(yaw[0].detach().cpu()),
                            "start_x_m": float(case["start_x"]),
                            "start_y_m": float(case["start_y"]),
                            "goal_x_m": float(case["goal_x"]),
                            "goal_y_m": float(case["goal_y"]),
                            "nominal_vx_mps": float(nominal[0, 0].detach().cpu()),
                            "nominal_vy_mps": float(nominal[0, 1].detach().cpu()),
                            "nominal_wz_radps": float(nominal[0, 2].detach().cpu()),
                            "safe_vx_mps": float(safe[0, 0].detach().cpu()),
                            "safe_vy_mps": float(safe[0, 1].detach().cpu()),
                            "safe_wz_radps": float(safe[0, 2].detach().cpu()),
                            "measured_vx_mps": float(measured_command[0, 0].detach().cpu()),
                            "measured_vy_mps": float(measured_command[0, 1].detach().cpu()),
                            "measured_wz_radps": float(measured_command[0, 2].detach().cpu()),
                            "safe_measured_velocity_error_norm": velocity_error_value,
                            "policy_inference_batch_ms": policy_ms,
                            "safety_filter_batch_ms": filter_ms,
                            "total_actor_inference_batch_ms": inference_ms,
                            "minimum_laser_range_m": float(
                                step_diagnostics["laser_min_range"][0].detach().cpu()
                            ),
                            "obstacle_surface_clearance_m": float(
                                step_diagnostics["obstacle_surface_clearance"][0].detach().cpu()
                            ),
                            "h_lse": float(step_diagnostics["h_lse"][0].detach().cpu()),
                            "minimum_safety_term": float(
                                step_diagnostics["minimum_safety_term"][0].detach().cpu()
                            ),
                            "command_correction_norm": correction_value,
                            "filter_active": int(filter_active),
                            "physical_clearance_m": clearance,
                            "cumulative_path_length_m": path_length,
                            "step_reward": step_reward,
                            "cumulative_episode_return": return_value,
                            "reward_component_sum": sum(
                                float(value[0].detach().cpu())
                                for value in step_diagnostics.get("reward_components", {}).values()
                            ),
                            **{
                                f"reward_{name}": float(value[0].detach().cpu())
                                for name, value in step_diagnostics.get("reward_components", {}).items()
                            },
                            "done": int(bool(done[0])),
                            "outcome": "",
                            "success": 0,
                            "collision": 0,
                            "safety_violation": 0,
                            "timeout": 0,
                            "fallen": 0,
                            "outside": 0,
                        })

                    if bool(done[0]):
                        success = bool(extras["completed_success"][0])
                        collision = bool(extras["completed_collision"][0])
                        safety_violation = bool(extras.get(
                            "completed_laser_safety_violation",
                            extras.get("completed_safety_violation", torch.zeros(1, dtype=torch.bool, device=env.device)),
                        )[0])
                        timeout = bool(extras["completed_timeouts"][0])
                        fallen = bool(extras.get("completed_fallen", torch.zeros(1, dtype=torch.bool, device=env.device))[0])
                        outside = bool(extras.get("completed_outside", torch.zeros(1, dtype=torch.bool, device=env.device))[0])
                        completed_steps = int(extras["completed_lengths"][0])
                        duration_s = completed_steps * env.navigation_cfg.high_level_dt
                        completed_returns = extras.get("completed_returns")
                        if completed_returns is not None and completed_returns.numel() > 0:
                            return_value = float(completed_returns[0].detach().cpu())

                        if success and not collision:
                            outcome = "success"
                        elif collision:
                            outcome = "collision"
                        elif timeout:
                            outcome = "timeout"
                        elif fallen:
                            outcome = "fall"
                        elif outside:
                            outcome = "outside"
                        else:
                            outcome = "other_failure"

                        if log_trajectories and trajectory_records:
                            trajectory_records[-1].update({
                                "outcome": outcome,
                                "success": int(success),
                                "collision": int(collision),
                                "safety_violation": int(safety_violation),
                                "timeout": int(timeout),
                                "fallen": int(fallen),
                                "outside": int(outside),
                            })
                        break

                    obs = next_obs

            cumulative_mean_return = (
                sum(float(row["return"]) for row in records) + return_value
            ) / (len(records) + 1)
            record = {
                "method": method,
                "map_name": map_name,
                "evaluation_seed": int(case.get("evaluation_seed", case["reset_seed"])),
                "case_id": case_id,
                "reset_seed": int(case["reset_seed"]),
                "start_x": float(case["start_x"]),
                "start_y": float(case["start_y"]),
                "start_yaw": float(case["start_yaw"]),
                "goal_x": float(case["goal_x"]),
                "goal_y": float(case["goal_y"]),
                "start_clearance_m": float(case.get("start_clearance_m", math.nan)),
                "goal_clearance_m": float(case.get("goal_clearance_m", math.nan)),
                "straight_line_distance_m": float(case.get("straight_line_distance_m", math.nan)),
                "straight_path_min_clearance_m": float(case.get("straight_path_min_clearance_m", math.nan)),
                "direct_path_blocked": int(case.get("direct_path_blocked", 0)),
                "blocking_obstacle_id": int(case.get("blocking_obstacle_id", -1)),
                "blocking_t": float(case.get("blocking_t", math.nan)),
                "blocking_clearance_m": float(case.get("blocking_clearance_m", math.nan)),
                "sampling_attempt": int(case.get("sampling_attempt", -1)),
                "outcome": outcome,
                "success": int(success),
                "collision": int(collision),
                "safety_violation": int(safety_violation),
                "timeout": int(timeout),
                "fallen": int(fallen),
                "outside": int(outside),
                "return": return_value,
                "cumulative_mean_return": cumulative_mean_return,
                "path_length_m": path_length,
                "episode_duration_s": duration_s,
                "episode_steps": step_count,
                "minimum_physical_clearance_m": minimum_clearance,
                "filter_active_fraction": active_steps / max(step_count, 1),
                "mean_command_correction_norm": correction_sum / max(step_count, 1),
                "mean_safe_measured_velocity_error_norm": velocity_error_sum / max(step_count, 1),
                "mean_policy_inference_batch_ms": mean(policy_latencies_ms),
                "mean_safety_filter_batch_ms": mean(filter_latencies_ms),
                "mean_total_actor_inference_batch_ms": mean(inference_latencies_ms),
                "mean_total_actor_inference_amortized_per_env_ms": mean(inference_latencies_ms),
                "p95_total_actor_inference_batch_ms": percentile(inference_latencies_ms, 0.95),
                "p99_total_actor_inference_batch_ms": percentile(inference_latencies_ms, 0.99),
                "max_total_actor_inference_batch_ms": max(inference_latencies_ms),
                "inference_deadline_miss_fraction": sum(
                    value > 1000.0 * env.navigation_cfg.high_level_dt
                    for value in inference_latencies_ms
                ) / max(len(inference_latencies_ms), 1),
            }
            records.append(record)

            if print_every > 0 and (index + 1) % print_every == 0:
                print(
                    f"[{method}] map={map_name:10s} case={index + 1:3d}/{len(manifest)} "
                    f"outcome={outcome:13s} return={return_value:+8.2f} "
                    f"path={path_length:7.3f} m time={duration_s:6.2f} s "
                    f"clearance={minimum_clearance:+.3f} m",
                    flush=True,
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        episode_csv = output_dir / f"{method}_episodes.csv"
        summary_csv = output_dir / f"{method}_summary.csv"
        trajectory_csv = output_dir / f"{method}_trajectory.csv"

        with episode_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        if log_trajectories:
            with trajectory_csv.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(trajectory_records[0].keys()))
                writer.writeheader()
                writer.writerows(trajectory_records)
            print(f"  trajectory_csv: {trajectory_csv}")

        successes = [row for row in records if int(row["success"]) == 1]
        summary = {
            "method": method,
            "map_name": map_name,
            "cases": len(records),
            "success_rate": sum(int(row["success"]) for row in records) / len(records),
            "collision_rate": sum(int(row["collision"]) for row in records) / len(records),
            "laser_safety_violation_rate": sum(int(row["safety_violation"]) for row in records) / len(records),
            "timeout_rate": sum(int(row["timeout"]) for row in records) / len(records),
            "fall_rate": sum(int(row["fallen"]) for row in records) / len(records),
            "outside_rate": sum(int(row["outside"]) for row in records) / len(records),
            "mean_return": mean(float(row["return"]) for row in records),
            "std_return": sample_std([float(row["return"]) for row in records]),
            "mean_success_path_length_m": optional_mean([float(row["path_length_m"]) for row in successes]),
            "std_success_path_length_m": sample_std([float(row["path_length_m"]) for row in successes]),
            "mean_success_goal_time_s": optional_mean([float(row["episode_duration_s"]) for row in successes]),
            "std_success_goal_time_s": sample_std([float(row["episode_duration_s"]) for row in successes]),
            "mean_minimum_clearance_m": mean(float(row["minimum_physical_clearance_m"]) for row in records),
            "minimum_clearance_overall_m": min(float(row["minimum_physical_clearance_m"]) for row in records),
            "mean_filter_active_fraction": mean(float(row["filter_active_fraction"]) for row in records),
            "mean_command_correction_norm": mean(float(row["mean_command_correction_norm"]) for row in records),
            "mean_safe_measured_velocity_error_norm": mean(float(row["mean_safe_measured_velocity_error_norm"]) for row in records),
            "mean_policy_inference_batch_ms": mean(float(row["mean_policy_inference_batch_ms"]) for row in records),
            "mean_safety_filter_batch_ms": mean(float(row["mean_safety_filter_batch_ms"]) for row in records),
            "mean_total_actor_inference_batch_ms": mean(float(row["mean_total_actor_inference_batch_ms"]) for row in records),
            "mean_total_actor_inference_amortized_per_env_ms": mean(float(row["mean_total_actor_inference_amortized_per_env_ms"]) for row in records),
            "mean_inference_deadline_miss_fraction": mean(float(row["inference_deadline_miss_fraction"]) for row in records),
        }
        with summary_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)

        print(f"\n[{method} SUMMARY] map={map_name}")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return summary_csv

    finally:
        try:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        except Exception as exc:
            print(f"[WARNING] Failed to close {method}/{map_name}: {exc}", flush=True)
        for name in ("actor", "env", "low_env", "raw_env"):
            if name in locals():
                del locals()[name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_method_vectorized(
    *,
    method: str,
    task_id: str,
    low_checkpoint: str,
    high_checkpoint: str,
    manifest: list[dict[str, object]],
    episode_duration_s: float,
    device: str,
    print_every: int,
    output_dir: Path,
    map_name: str,
    log_trajectories: bool,
    laser_num_rays: int,
    laser_fov_deg: float,
    enable_lidar_noise: bool,
    hard_clip_output: bool,
    collision_geometry: str,
    collision_circle_radius_m: float,
    collision_margin_m: float,
    collision_termination_mode: str,
    qp_backend: str,
    num_envs: int,
    dynamic_phase_jitter_s: float = 20.0,
    terminal_backup_mode: str = "legacy_inflated_ellipse",
    terminal_geom_scale: float = 1.0,
    terminal_rec_scale: float = 1.0,
    legacy_projection: bool = False,
    action_max_vx: float = 0.5,
    action_min_vx: float | None = None,
    action_max_vy: float = 0.3,
    action_max_wz: float = 0.5,
    backup_margin: float | None = None,
    high_level_dt: float | None = None,
    integration_dt: float | None = None,
    backup_rollout_substeps: int | None = None,
    disable_safety_filter: bool = False,
) -> Path:
    """Evaluate manifest cases in fixed-size vectorized batches."""
    if not manifest:
        raise ValueError("Cannot evaluate an empty manifest.")
    batch_size = min(int(num_envs), len(manifest))
    set_seed(int(manifest[0]["reset_seed"]))
    raw_env, low_env, env, cfg, dynamic_runtime = build_environment(
        task_id=task_id,
        low_checkpoint=low_checkpoint,
        method=method,
        episode_duration_s=episode_duration_s,
        device=device,
        map_name=map_name,
        laser_num_rays=laser_num_rays,
        laser_fov_deg=laser_fov_deg,
        enable_lidar_noise=enable_lidar_noise,
        hard_clip_output=hard_clip_output,
        collision_geometry=collision_geometry,
        collision_circle_radius_m=collision_circle_radius_m,
        collision_margin_m=collision_margin_m,
        collision_termination_mode=collision_termination_mode,
        qp_backend=qp_backend,
        num_envs=batch_size,
        terminal_backup_mode=terminal_backup_mode,
        terminal_geom_scale=terminal_geom_scale,
        terminal_rec_scale=terminal_rec_scale,
        legacy_projection=legacy_projection,
        action_max_vx=action_max_vx,
        action_min_vx=action_min_vx,
        action_max_vy=action_max_vy,
        action_max_wz=action_max_wz,
        backup_margin=backup_margin,
        high_level_dt=high_level_dt,
        integration_dt=integration_dt,
        backup_rollout_substeps=backup_rollout_substeps,
    )
    records: list[dict[str, object]] = []
    trajectory_records: list[dict[str, object]] = []
    try:
        verify_compiled_physical_map(raw_env, env, map_name)
        actor = load_actor(env=env, cfg=cfg, checkpoint_path=high_checkpoint, device=device)
        actor.disable_safety_filter = disable_safety_filter
        warmup_actor_timing(actor, env)
        dt = float(env.navigation_cfg.high_level_dt)

        for batch_start in range(0, len(manifest), batch_size):
            real_cases = manifest[batch_start : batch_start + batch_size]
            padded_cases = list(real_cases)
            while len(padded_cases) < batch_size:
                padded_cases.append(real_cases[-1])
            env.set_fixed_scenarios([row_to_scenario(case) for case in padded_cases])
            verify_parallel_lidar_geometry(raw_env, env, map_name)

            dynamic_active = dynamic_runtime is not None and dynamic_runtime.active
            global_step = 0
            phase_offsets = torch.zeros(batch_size, device=device)
            if dynamic_active:
                # Deterministic per-case phase offset (from that case's own
                # reset_seed) so different cases see the gates at different
                # points in their cycle, rather than every case starting
                # from the identical t=0 configuration.
                offsets = []
                for case in padded_cases:
                    case_rng = random.Random(int(case["reset_seed"]))
                    offsets.append(case_rng.uniform(0.0, dynamic_phase_jitter_s))
                phase_offsets = torch.tensor(offsets, device=device)
                dynamic_runtime.apply(phase_offsets)
            obs = env.get_observations()

            positions, yaws = env.safety_context()
            goals = env.goal_positions
            for env_id, case in enumerate(real_cases):
                expected_start = positions.new_tensor(
                    [float(case["start_x"]), float(case["start_y"])]
                )
                expected_goal = goals.new_tensor(
                    [float(case["goal_x"]), float(case["goal_y"])]
                )
                expected_yaw = yaws.new_tensor(float(case["start_yaw"]))
                if not torch.allclose(positions[env_id], expected_start, atol=1e-4, rtol=0):
                    raise RuntimeError(f"{method}/{map_name}/case {case['case_id']}: start mismatch.")
                if not torch.allclose(goals[env_id], expected_goal, atol=1e-4, rtol=0):
                    raise RuntimeError(f"{method}/{map_name}/case {case['case_id']}: goal mismatch.")
                if not torch.allclose(yaws[env_id], expected_yaw, atol=1e-4, rtol=0):
                    raise RuntimeError(f"{method}/{map_name}/case {case['case_id']}: yaw mismatch.")

            active_count = len(real_cases)
            finished = [False] * active_count
            previous_positions = positions[:active_count].detach().clone()
            path_lengths = [0.0] * active_count
            minimum_clearances = [
                physical_clearance(env, previous_positions[i], i) for i in range(active_count)
            ]
            active_steps = [0] * active_count
            infeasible_steps = [0] * active_count
            correction_sums = [0.0] * active_count
            returns = [0.0] * active_count
            step_counts = [0] * active_count
            velocity_error_sums = [0.0] * active_count
            policy_latency_rows: list[list[float]] = [[] for _ in range(active_count)]
            filter_latency_rows: list[list[float]] = [[] for _ in range(active_count)]
            inference_latency_rows: list[list[float]] = [[] for _ in range(active_count)]

            with torch.no_grad():
                while not all(finished):
                    pre_positions, pre_yaws = env.safety_context()
                    actor_obs = env.actor_tensor(obs)
                    nominal, safe, safety, policy_ms, filter_ms, inference_ms = (
                        timed_deterministic_actor(actor, actor_obs)
                    )
                    next_obs, reward, done, extras = env.step_safe(nominal, safe, safety)
                    if dynamic_active:
                        global_step += 1
                        dynamic_runtime.apply(global_step * dt + phase_offsets)
                    step_diagnostics = extras["collision_trajectory_step"]
                    measured_command = step_diagnostics["measured_command"]

                    completed_ids = extras.get("completed_env_ids", torch.empty(0, device=env.device, dtype=torch.long))
                    completed_lookup = {
                        int(env_id): row for row, env_id in enumerate(completed_ids.tolist())
                    }
                    current_positions, _ = env.safety_context()

                    for env_id, case in enumerate(real_cases):
                        if finished[env_id]:
                            continue
                        step_counts[env_id] += 1
                        policy_latency_rows[env_id].append(policy_ms)
                        filter_latency_rows[env_id].append(filter_ms)
                        inference_latency_rows[env_id].append(inference_ms)
                        step_reward = float(reward[env_id].detach().cpu())
                        returns[env_id] += step_reward
                        completed_row = completed_lookup.get(env_id)
                        is_done = completed_row is not None
                        if is_done:
                            endpoint = extras["completed_terminal_positions"][completed_row]
                        else:
                            endpoint = current_positions[env_id]

                        start_position = previous_positions[env_id].detach().clone()
                        path_lengths[env_id] += float(
                            torch.linalg.norm(endpoint - start_position).detach().cpu()
                        )
                        previous_positions[env_id] = endpoint.detach().clone()
                        clearance = physical_clearance(env, endpoint, env_id)
                        minimum_clearances[env_id] = min(minimum_clearances[env_id], clearance)
                        correction_value = float(
                            torch.linalg.norm(safe[env_id] - nominal[env_id]).detach().cpu()
                        )
                        correction_sums[env_id] += correction_value
                        velocity_error_value = float(
                            torch.linalg.norm(measured_command[env_id] - safe[env_id])
                            .detach().cpu()
                        )
                        velocity_error_sums[env_id] += velocity_error_value
                        filter_active = method != "vanilla" and correction_value > 1e-8
                        active_steps[env_id] += int(filter_active)
                        feasible_flags = getattr(safety, "projection_feasible", None)
                        if feasible_flags is not None:
                            infeasible_steps[env_id] += int(not bool(feasible_flags[env_id]))

                        trajectory_row = None
                        if log_trajectories:
                            trajectory_row = {
                                "method": method, "map_name": map_name,
                                "evaluation_seed": int(case.get("evaluation_seed", case["reset_seed"])),
                                "case_id": int(case["case_id"]),
                                "reset_seed": int(case["reset_seed"]),
                                "env_id": env_id, "step": step_counts[env_id],
                                "time_s": step_counts[env_id] * dt,
                                "pre_x_m": float(start_position[0].detach().cpu()),
                                "pre_y_m": float(start_position[1].detach().cpu()),
                                "post_x_m": float(endpoint[0].detach().cpu()),
                                "post_y_m": float(endpoint[1].detach().cpu()),
                                "yaw_rad": float(pre_yaws[env_id].detach().cpu()),
                                "start_x_m": float(case["start_x"]), "start_y_m": float(case["start_y"]),
                                "goal_x_m": float(case["goal_x"]), "goal_y_m": float(case["goal_y"]),
                                "nominal_vx_mps": float(nominal[env_id, 0].detach().cpu()),
                                "nominal_vy_mps": float(nominal[env_id, 1].detach().cpu()),
                                "nominal_wz_radps": float(nominal[env_id, 2].detach().cpu()),
                                "safe_vx_mps": float(safe[env_id, 0].detach().cpu()),
                                "safe_vy_mps": float(safe[env_id, 1].detach().cpu()),
                                "safe_wz_radps": float(safe[env_id, 2].detach().cpu()),
                                "measured_vx_mps": float(
                                    measured_command[env_id, 0].detach().cpu()
                                ),
                                "measured_vy_mps": float(
                                    measured_command[env_id, 1].detach().cpu()
                                ),
                                "measured_wz_radps": float(
                                    measured_command[env_id, 2].detach().cpu()
                                ),
                                "safe_measured_velocity_error_norm": velocity_error_value,
                                "policy_inference_batch_ms": policy_ms,
                                "safety_filter_batch_ms": filter_ms,
                                "total_actor_inference_batch_ms": inference_ms,
                                "total_actor_inference_amortized_per_env_ms": inference_ms / batch_size,
                                "minimum_laser_range_m": float(
                                    step_diagnostics["laser_min_range"][env_id].detach().cpu()
                                ),
                                "obstacle_surface_clearance_m": float(
                                    step_diagnostics["obstacle_surface_clearance"][env_id]
                                    .detach().cpu()
                                ),
                                "h_lse": float(
                                    step_diagnostics["h_lse"][env_id].detach().cpu()
                                ),
                                "minimum_safety_term": float(
                                    step_diagnostics["minimum_safety_term"][env_id]
                                    .detach().cpu()
                                ),
                                "command_correction_norm": correction_value,
                                "filter_active": int(filter_active),
                                "physical_clearance_m": clearance,
                                "cumulative_path_length_m": path_lengths[env_id],
                                "step_reward": step_reward,
                                "cumulative_episode_return": returns[env_id],
                                "reward_component_sum": sum(
                                    float(value[env_id].detach().cpu())
                                    for value in step_diagnostics.get("reward_components", {}).values()
                                ),
                                **{
                                    f"reward_{name}": float(value[env_id].detach().cpu())
                                    for name, value in step_diagnostics.get("reward_components", {}).items()
                                },
                                "done": int(is_done), "outcome": "", "success": 0,
                                "collision": 0, "safety_violation": 0,
                                "timeout": 0, "fallen": 0, "outside": 0,
                            }

                        if is_done:
                            success = bool(extras["completed_success"][completed_row])
                            collision = bool(extras["completed_collision"][completed_row])
                            safety_violation = bool(extras.get(
                                "completed_laser_safety_violation", extras["completed_safety_violation"]
                            )[completed_row])
                            timeout = bool(extras["completed_timeouts"][completed_row])
                            fallen = bool(extras["completed_fallen"][completed_row])
                            outside = bool(extras["completed_outside"][completed_row])
                            if success and not collision:
                                outcome = "success"
                            elif collision:
                                outcome = "collision"
                            elif timeout:
                                outcome = "timeout"
                            elif fallen:
                                outcome = "fall"
                            elif outside:
                                outcome = "outside"
                            else:
                                outcome = "other_failure"
                            returns[env_id] = float(extras["completed_returns"][completed_row].detach().cpu())
                            duration = int(extras["completed_lengths"][completed_row]) * dt
                            if trajectory_row is not None:
                                trajectory_row.update({
                                    "outcome": outcome, "success": int(success),
                                    "collision": int(collision), "timeout": int(timeout),
                                    "safety_violation": int(safety_violation),
                                    "fallen": int(fallen), "outside": int(outside),
                                })
                            cumulative_mean = (
                                sum(float(row["return"]) for row in records) + returns[env_id]
                            ) / (len(records) + 1)
                            records.append({
                                "method": method, "map_name": map_name,
                                "evaluation_seed": int(case.get("evaluation_seed", case["reset_seed"])),
                                "case_id": int(case["case_id"]), "reset_seed": int(case["reset_seed"]),
                                "start_x": float(case["start_x"]), "start_y": float(case["start_y"]),
                                "start_yaw": float(case["start_yaw"]),
                                "goal_x": float(case["goal_x"]), "goal_y": float(case["goal_y"]),
                                "start_clearance_m": float(case.get("start_clearance_m", math.nan)),
                                "goal_clearance_m": float(case.get("goal_clearance_m", math.nan)),
                                "straight_line_distance_m": float(case.get("straight_line_distance_m", math.nan)),
                                "straight_path_min_clearance_m": float(case.get("straight_path_min_clearance_m", math.nan)),
                                "direct_path_blocked": int(case.get("direct_path_blocked", 0)),
                                "blocking_obstacle_id": int(case.get("blocking_obstacle_id", -1)),
                                "blocking_t": float(case.get("blocking_t", math.nan)),
                                "blocking_clearance_m": float(case.get("blocking_clearance_m", math.nan)),
                                "sampling_attempt": int(case.get("sampling_attempt", -1)),
                                "outcome": outcome, "success": int(success), "collision": int(collision),
                                "safety_violation": int(safety_violation),
                                "timeout": int(timeout), "fallen": int(fallen), "outside": int(outside),
                                "return": returns[env_id], "cumulative_mean_return": cumulative_mean,
                                "path_length_m": path_lengths[env_id], "episode_duration_s": duration,
                                "episode_steps": step_counts[env_id],
                                "minimum_physical_clearance_m": minimum_clearances[env_id],
                                "filter_active_fraction": active_steps[env_id] / max(step_counts[env_id], 1),
                                "infeasible_fraction": infeasible_steps[env_id] / max(step_counts[env_id], 1),
                                "mean_command_correction_norm": correction_sums[env_id] / max(step_counts[env_id], 1),
                                "mean_safe_measured_velocity_error_norm": velocity_error_sums[env_id] / max(step_counts[env_id], 1),
                                "mean_policy_inference_batch_ms": mean(policy_latency_rows[env_id]),
                                "mean_safety_filter_batch_ms": mean(filter_latency_rows[env_id]),
                                "mean_total_actor_inference_batch_ms": mean(inference_latency_rows[env_id]),
                                "mean_total_actor_inference_amortized_per_env_ms": mean(inference_latency_rows[env_id]) / batch_size,
                                "p95_total_actor_inference_batch_ms": percentile(inference_latency_rows[env_id], 0.95),
                                "p99_total_actor_inference_batch_ms": percentile(inference_latency_rows[env_id], 0.99),
                                "max_total_actor_inference_batch_ms": max(inference_latency_rows[env_id]),
                                "inference_deadline_miss_fraction": sum(
                                    value > 1000.0 * dt for value in inference_latency_rows[env_id]
                                ) / max(len(inference_latency_rows[env_id]), 1),
                            })
                            finished[env_id] = True
                            completed_total = batch_start + sum(finished)
                            if print_every > 0 and completed_total % print_every == 0:
                                print(
                                    f"[{method}] map={map_name:10s} completed={completed_total:3d}/{len(manifest)} "
                                    f"case={int(case['case_id']):3d} outcome={outcome:13s} "
                                    f"return={returns[env_id]:+8.2f} path={path_lengths[env_id]:7.3f} m "
                                    f"time={duration:6.2f} s clearance={minimum_clearances[env_id]:+.3f} m",
                                    flush=True,
                                )
                        if trajectory_row is not None:
                            trajectory_records.append(trajectory_row)
                    # For dynamic maps, next_obs was computed inside step_safe
                    # BEFORE this iteration's obstacle motion was applied above,
                    # so it would be one step stale; re-fetch instead.
                    obs = env.get_observations() if dynamic_active else next_obs

        records.sort(key=lambda row: int(row["case_id"]))
        trajectory_records.sort(key=lambda row: (int(row["case_id"]), int(row["step"])))
        output_dir.mkdir(parents=True, exist_ok=True)
        episode_csv = output_dir / f"{method}_episodes.csv"
        summary_csv = output_dir / f"{method}_summary.csv"
        with episode_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
            writer.writeheader(); writer.writerows(records)
        if log_trajectories:
            trajectory_csv = output_dir / f"{method}_trajectory.csv"
            with trajectory_csv.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(trajectory_records[0].keys()))
                writer.writeheader(); writer.writerows(trajectory_records)
            print(f"  trajectory_csv: {trajectory_csv}")
        successes = [row for row in records if int(row["success"]) == 1]
        summary = {
            "method": method, "map_name": map_name, "cases": len(records),
            "success_rate": sum(int(row["success"]) for row in records) / len(records),
            "collision_rate": sum(int(row["collision"]) for row in records) / len(records),
            "laser_safety_violation_rate": sum(int(row["safety_violation"]) for row in records) / len(records),
            "timeout_rate": sum(int(row["timeout"]) for row in records) / len(records),
            "fall_rate": sum(int(row["fallen"]) for row in records) / len(records),
            "outside_rate": sum(int(row["outside"]) for row in records) / len(records),
            "mean_return": mean(float(row["return"]) for row in records),
            "std_return": sample_std([float(row["return"]) for row in records]),
            "mean_success_path_length_m": optional_mean([float(row["path_length_m"]) for row in successes]),
            "std_success_path_length_m": sample_std([float(row["path_length_m"]) for row in successes]),
            "mean_success_goal_time_s": optional_mean([float(row["episode_duration_s"]) for row in successes]),
            "std_success_goal_time_s": sample_std([float(row["episode_duration_s"]) for row in successes]),
            "mean_minimum_clearance_m": mean(float(row["minimum_physical_clearance_m"]) for row in records),
            "minimum_clearance_overall_m": min(float(row["minimum_physical_clearance_m"]) for row in records),
            "mean_filter_active_fraction": mean(float(row["filter_active_fraction"]) for row in records),
            "mean_infeasible_fraction": mean(float(row["infeasible_fraction"]) for row in records),
            "mean_command_correction_norm": mean(float(row["mean_command_correction_norm"]) for row in records),
            "mean_safe_measured_velocity_error_norm": mean(float(row["mean_safe_measured_velocity_error_norm"]) for row in records),
            "mean_policy_inference_batch_ms": mean(float(row["mean_policy_inference_batch_ms"]) for row in records),
            "mean_safety_filter_batch_ms": mean(float(row["mean_safety_filter_batch_ms"]) for row in records),
            "mean_total_actor_inference_batch_ms": mean(float(row["mean_total_actor_inference_batch_ms"]) for row in records),
            "mean_total_actor_inference_amortized_per_env_ms": mean(float(row["mean_total_actor_inference_amortized_per_env_ms"]) for row in records),
            "mean_inference_deadline_miss_fraction": mean(float(row["inference_deadline_miss_fraction"]) for row in records),
        }
        with summary_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
            writer.writeheader(); writer.writerow(summary)
        print(f"\n[{method} SUMMARY] map={map_name}")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return summary_csv
    finally:
        try:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        except Exception as exc:
            print(f"[WARNING] Failed to close {method}/{map_name}: {exc}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_map_metadata(map_dir: Path, map_name: str, arena: ArenaConfig) -> None:
    path = map_dir / "map.csv"
    with path.open("w", newline="") as file:
        fields = ["map_name", "obstacle_id", "x", "y", "radius", "height"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for obstacle_id, obs in enumerate(arena.obstacles):
            writer.writerow({
                "map_name": map_name,
                "obstacle_id": obstacle_id,
                "x": obs.x,
                "y": obs.y,
                "radius": obs.radius,
                "height": obs.height,
            })


def main() -> None:
    args = parse_args()
    if args.cases_per_seed <= 0 or args.num_seeds <= 0 or args.seed_stride <= 0:
        raise ValueError("--cases-per-seed, --num-seeds, and --seed-stride must be positive.")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_seeds = [int(args.seed) + i * int(args.seed_stride) for i in range(args.num_seeds)]
    total_per_map = args.num_seeds * args.cases_per_seed
    root = Path(args.run_root) if args.run_root else Path(args.output_root) / (
        f"{timestamp}_seeds{evaluation_seeds[0]}-{evaluation_seeds[-1]}_"
        f"{args.cases_per_seed}perseed_{total_per_map}permap"
    )
    root.mkdir(parents=True, exist_ok=True)
    experiment_config_path = root / "experiment_config.csv"
    if not experiment_config_path.is_file():
        write_rows(experiment_config_path, [{
        "base_seed": args.seed, "num_seeds": args.num_seeds,
        "seed_stride": args.seed_stride, "evaluation_seeds": ";".join(map(str, evaluation_seeds)),
        "cases_per_seed_per_map": args.cases_per_seed,
        "total_cases_per_map_per_method": total_per_map,
        "maps": ";".join(args.maps), "methods": ";".join(args.methods),
        "num_parallel_envs": args.num_envs, "episode_duration_s": args.episode_duration_s,
        "laser_num_rays": args.laser_num_rays, "laser_fov_deg": args.laser_fov_deg,
        "lidar_noise": int(args.enable_lidar_noise),
        "collision_geometry": args.collision_geometry,
        "collision_circle_radius_m": args.collision_circle_radius_m,
        "collision_margin_m": args.collision_margin_m,
        "collision_termination_mode": args.collision_termination_mode,
        "inference_timing_warmup_calls": 10,
        "inference_timing_cuda_synchronized": 1,
        "inference_timing_policy_scope": "actor_mlp_plus_tanh",
        "inference_timing_filter_scope": "scan_extraction_plus_safety_filter",
        "inference_deadline_ms": 100.0,
        }])
    print(
        f"Evaluation design: {args.num_seeds} seeds x {args.cases_per_seed} cases "
        f"= {total_per_map} episodes per map and method"
    )

    checkpoints = {
        "vanilla": args.vanilla_checkpoint,
        "cbf_cf_clip": args.cbf_cf_clip_checkpoint,
        "cbf_qp": args.cbf_qp_checkpoint,
        "cbf_lse_qp": args.cbf_lse_qp_checkpoint,
        "cbf_cd_cf": args.cbf_cd_cf_checkpoint,
        "cbf_lse_cd_qp": args.cbf_lse_cd_qp_checkpoint,
        "bcbf_cf": args.bcbf_cf_checkpoint,
        "bcbf_qp": args.bcbf_qp_checkpoint,
        "bcbf_lse_cd_qp": args.bcbf_lse_cd_qp_checkpoint,
        "bcbf_nonlse_cd_qp": args.bcbf_nonlse_cd_qp_checkpoint,
        "cbf_lse_cd_qp_cfmatched": args.cbf_lse_cd_qp_cfmatched_checkpoint,
        "bcbf_lse_cd_qp_cfmatched": args.bcbf_lse_cd_qp_cfmatched_checkpoint,
    }
    for method in args.methods:
        if checkpoints[method] is None:
            raise ValueError(f"--{method.replace('_', '-')}-checkpoint is required when evaluating {method}")
        path = Path(checkpoints[method]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {method}: {path}")
    low_path = Path(args.low_level_checkpoint).expanduser()
    if not low_path.is_file():
        raise FileNotFoundError(f"Missing low-level checkpoint: {low_path}")

    all_episodes: list[dict[str, object]] = []
    seed_map_rows: list[dict[str, object]] = []
    seed_aggregate_rows: list[dict[str, object]] = []

    # 2026-09-23: manifests for every seed of a map are now built up front,
    # then every not-yet-resumed seed's cases are pooled into ONE manifest
    # per (map, method) before evaluating -- the same "manifests now pooled,
    # one big batch per map" fix already applied to the CBF-RL eval scripts,
    # mirrored here. batch_size inside evaluate_method_vectorized is
    # min(--num-envs, pooled case count), so a sufficiently large --num-envs
    # (e.g. --num-envs 500 with --num-seeds 25 --cases-per-seed 20) now
    # processes an entire map's cases in a single batch instead of 25
    # separate cases-per-seed-sized (20) batches, each previously paying its
    # own "wait for that batch's slowest straggler" cost. Purely a
    # wall-clock speedup: scenario content, outcome computation, and the
    # per-seed seed_X/map/method/method_episodes.csv file layout are all
    # unchanged, so existing resume behavior and every downstream analysis
    # script are unaffected. The one behavior change is coarser RNG seeding
    # (evaluate_method_vectorized seeds once from the pooled manifest's
    # first case's reset_seed, rather than once per original seed-sized
    # batch) -- this only affects the exact LiDAR-noise draw sequence, not
    # scenario content or any statistical property of the results.
    map_dirs: dict[tuple[str, int], Path] = {}
    pooled_root = root / "_pooled_batches"
    for map_name in args.maps:
        seed_manifests: dict[int, list[dict[str, object]]] = {}
        for evaluation_seed in evaluation_seeds:
            seed_dir = root / f"seed_{evaluation_seed}"
            print("\n" + "=" * 72)
            print(f"LASER STRUCTURED UNSEEN MAP: {map_name} | EVALUATION SEED: {evaluation_seed}")
            print("=" * 72)
            # A distinctly-named directory when --randomize-start-goal is set:
            # this new randomized-endpoint scenario bank can never resume from,
            # or silently overwrite, an existing structured-region manifest
            # even if pointed at the same --run-root/--output-root.
            map_dir = seed_dir / (f"{map_name}_random" if args.randomize_start_goal else map_name)
            map_dir.mkdir(parents=True, exist_ok=True)
            # Match the wall_clearance/obstacle_clearance actually enforced by
            # the simulated environment (see make_laser_config's override) --
            # using the raw ArenaConfig default (0.15) here let scenarios spawn
            # as close as 0.25m to a wall/obstacle, which is inside the real
            # 0.30m margin the env and safety filter assume, and can trigger an
            # immediate collision/fall at episode start.
            arena = replace(make_map_arena(map_name), wall_clearance=0.30, obstacle_clearance=0.30)
            write_map_metadata(map_dir, map_name, arena)
            map_index = list(UNSEEN_MAPS).index(map_name)
            manifest_seed = evaluation_seed + 10000 * map_index
            manifest_path = map_dir / "scenarios.csv"
            if manifest_path.is_file():
                with manifest_path.open(newline="") as file:
                    manifest = [dict(row) for row in csv.DictReader(file)]
                if len(manifest) != args.cases_per_seed:
                    raise RuntimeError(
                        f"Existing manifest {manifest_path} has {len(manifest)} cases; "
                        f"expected {args.cases_per_seed}."
                    )
                print(f"[RESUME] reusing {manifest_path}")
            else:
                set_seed(manifest_seed)
                raw_env, low_env, env, _, _ = build_environment(
                    task_id=args.task_id, low_checkpoint=args.low_level_checkpoint,
                    method="vanilla", episode_duration_s=args.episode_duration_s,
                    device=device, map_name=map_name, laser_num_rays=args.laser_num_rays,
                    laser_fov_deg=args.laser_fov_deg, enable_lidar_noise=args.enable_lidar_noise,
                    hard_clip_output=args.hard_clip_output,
                    collision_geometry=args.collision_geometry,
                    collision_circle_radius_m=args.collision_circle_radius_m,
                    collision_margin_m=args.collision_margin_m,
                    collision_termination_mode=args.collision_termination_mode,
                    action_max_vx=args.action_max_vx,
                    action_min_vx=args.action_min_vx,
                    action_max_vy=args.action_max_vy,
                    action_max_wz=args.action_max_wz,
                )
                try:
                    manifest_fn = (
                        generate_manifest_randomized if args.randomize_start_goal else generate_manifest
                    )
                    manifest_kwargs = dict(
                    env=env, arena=arena, map_name=map_name, count=args.cases_per_seed,
                    seed=manifest_seed, output_path=manifest_path,
                    min_endpoint_clearance=args.min_endpoint_clearance,
                    block_margin=args.block_margin,
                    require_blocked_path=not args.allow_unblocked_paths,
                    blocked_t_min=args.blocked_t_min, blocked_t_max=args.blocked_t_max,
                    max_sampling_attempts=args.max_sampling_attempts,
                    yaw_jitter_rad=args.yaw_jitter_rad,
                )
                    if args.randomize_start_goal:
                        manifest_kwargs["min_straight_line_distance"] = args.min_random_distance_m
                    manifest = manifest_fn(**manifest_kwargs)
                    for case in manifest:
                        case["evaluation_seed"] = evaluation_seed
                    write_rows(manifest_path, manifest)
                finally:
                    close = getattr(env, "close", None)
                    if callable(close): close()
                    del env, low_env, raw_env
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

            evaluation_manifest = manifest
            if args.case_ids is not None:
                requested = set(args.case_ids)
                evaluation_manifest = [r for r in manifest if int(r["case_id"]) in requested]
                missing = requested - {int(r["case_id"]) for r in evaluation_manifest}
                if missing: raise ValueError(f"Requested case IDs not found: {sorted(missing)}")

            map_dirs[(map_name, evaluation_seed)] = map_dir
            seed_manifests[evaluation_seed] = evaluation_manifest

        for method in args.methods:
            pooled_manifest: list[dict[str, object]] = []
            for evaluation_seed in evaluation_seeds:
                manifest = seed_manifests[evaluation_seed]
                method_dir = map_dirs[(map_name, evaluation_seed)] / method
                episode_path = method_dir / f"{method}_episodes.csv"
                if episode_path.is_file():
                    with episode_path.open(newline="") as file:
                        existing_episodes = [dict(row) for row in csv.DictReader(file)]
                    if len(existing_episodes) == len(manifest):
                        print(f"[RESUME] skipping complete {method}/{map_name}/seed_{evaluation_seed}")
                        continue
                pooled_manifest.extend(manifest)

            if not pooled_manifest:
                continue
            pooled_seed_count = len({int(r["evaluation_seed"]) for r in pooled_manifest})
            print("\n" + "=" * 72)
            print(
                f"LASER STRUCTURED UNSEEN MAP: {map_name} | METHOD: {method} | "
                f"pooling {len(pooled_manifest)} case(s) across {pooled_seed_count} not-yet-resumed seed(s)"
            )
            print("=" * 72)
            pooled_output_dir = pooled_root / map_name / method
            evaluator = evaluate_method if args.num_envs == 1 else evaluate_method_vectorized
            kwargs = dict(
                method=method, task_id=args.task_id, low_checkpoint=args.low_level_checkpoint,
                high_checkpoint=checkpoints[method], manifest=pooled_manifest,
                episode_duration_s=args.episode_duration_s, device=device,
                print_every=args.print_every, output_dir=pooled_output_dir, map_name=map_name,
                log_trajectories=args.log_trajectories, laser_num_rays=args.laser_num_rays,
                laser_fov_deg=args.laser_fov_deg, enable_lidar_noise=args.enable_lidar_noise,
                hard_clip_output=args.hard_clip_output,
                collision_geometry=args.collision_geometry,
                collision_circle_radius_m=args.collision_circle_radius_m,
                collision_margin_m=args.collision_margin_m,
                collision_termination_mode=args.collision_termination_mode,
                qp_backend=(
                    args.qp_backend
                    if method in ("cbf_qp", "cbf_lse_qp", "cbf_lse_cd_qp", "bcbf_qp", "bcbf_lse_cd_qp", "bcbf_nonlse_cd_qp", "cbf_lse_cd_qp_cfmatched", "bcbf_lse_cd_qp_cfmatched")
                    else "active_set"
                ),
                terminal_backup_mode=args.terminal_backup_mode,
                terminal_geom_scale=args.terminal_geom_scale,
                terminal_rec_scale=args.terminal_rec_scale,
                legacy_projection=args.legacy_projection,
                action_max_vx=args.action_max_vx,
                action_min_vx=args.action_min_vx,
                action_max_vy=args.action_max_vy,
                action_max_wz=args.action_max_wz,
                backup_margin=args.backup_margin,
                high_level_dt=args.high_level_dt,
                integration_dt=args.integration_dt,
                backup_rollout_substeps=args.backup_rollout_substeps,
                disable_safety_filter=args.disable_safety_filter,
            )
            if args.num_envs != 1:
                kwargs["num_envs"] = args.num_envs
                kwargs["dynamic_phase_jitter_s"] = args.dynamic_phase_jitter_s
            evaluator(**kwargs)
            with (pooled_output_dir / f"{method}_episodes.csv").open(newline="") as file:
                combined_episodes = [dict(row) for row in csv.DictReader(file)]
            by_seed: dict[int, list[dict[str, object]]] = {}
            for row in combined_episodes:
                by_seed.setdefault(int(row["evaluation_seed"]), []).append(row)
            for evaluation_seed, rows in by_seed.items():
                method_dir = map_dirs[(map_name, evaluation_seed)] / method
                method_dir.mkdir(parents=True, exist_ok=True)
                write_rows(method_dir / f"{method}_episodes.csv", rows)

    # Every (seed, map, method) now has its own episodes.csv on disk --
    # either pre-existing/resumed, or just written above -- so rebuild the
    # standard per-seed bookkeeping by reading them back.
    for evaluation_seed in evaluation_seeds:
        seed_episodes: list[dict[str, object]] = []
        for map_name in args.maps:
            map_dir = map_dirs[(map_name, evaluation_seed)]
            for method in args.methods:
                method_dir = map_dir / method
                with (method_dir / f"{method}_episodes.csv").open(newline="") as file:
                    episodes = [dict(row) for row in csv.DictReader(file)]
                all_episodes.extend(episodes)
                seed_episodes.extend(episodes)
                seed_map_rows.append(summarize_episodes(
                    episodes, evaluation_seed=evaluation_seed, method=method, map_name=map_name
                ))
        for method in args.methods:
            episodes = [r for r in seed_episodes if r["method"] == method]
            seed_aggregate_rows.append(summarize_episodes(
                episodes, evaluation_seed=evaluation_seed, method=method, map_name="ALL"
            ))

    pooled_map_rows = [summarize_episodes(
        [r for r in all_episodes if r["method"] == method and r["map_name"] == map_name],
        method=method, map_name=map_name,
    ) for map_name in args.maps for method in args.methods]
    pooled_aggregate_rows = [summarize_episodes(
        [r for r in all_episodes if r["method"] == method], method=method, map_name="ALL"
    ) for method in args.methods]

    seed_metrics = ["success_rate", "collision_rate", "laser_safety_violation_rate",
                    "timeout_rate", "fall_rate", "outside_rate",
                    "mean_return", "mean_success_path_length_m", "mean_success_goal_time_s",
                    "mean_minimum_clearance_m", "minimum_clearance_overall_m",
                    "mean_filter_active_fraction", "mean_command_correction_norm"]
    def across_seed(rows: list[dict[str, object]], **labels: object) -> dict[str, object]:
        result: dict[str, object] = dict(labels)
        result["num_seeds"] = len(rows)
        result["cases_per_seed"] = args.cases_per_seed
        result["total_cases"] = sum(int(r["cases"]) for r in rows)
        for metric in seed_metrics:
            result.update(mean_stats(
                [float(r[metric]) for r in rows], metric,
                critical_value=t95_critical(len(rows)),
            ))
        return result

    def paired_comparisons(rows: list[dict[str, object]], map_name: str) -> list[dict[str, object]]:
        """Across-seed paired method deltas; positive means method_a is larger."""
        output: list[dict[str, object]] = []
        for index, method_a in enumerate(args.methods):
            for method_b in args.methods[index + 1:]:
                a_by_seed = {int(r["evaluation_seed"]): r for r in rows
                             if r["method"] == method_a and r["map_name"] == map_name}
                b_by_seed = {int(r["evaluation_seed"]): r for r in rows
                             if r["method"] == method_b and r["map_name"] == map_name}
                shared = sorted(a_by_seed.keys() & b_by_seed.keys())
                result: dict[str, object] = {
                    "map_name": map_name, "method_a": method_a, "method_b": method_b,
                    "num_paired_seeds": len(shared),
                }
                for metric in seed_metrics:
                    deltas = [float(a_by_seed[s][metric]) - float(b_by_seed[s][metric]) for s in shared]
                    result.update(mean_stats(
                        deltas, f"delta_{metric}", critical_value=t95_critical(len(deltas))
                    ))
                output.append(result)
        return output

    across_map_rows = [across_seed(
        [r for r in seed_map_rows if r["method"] == method and r["map_name"] == map_name],
        method=method, map_name=map_name,
    ) for map_name in args.maps for method in args.methods]
    across_aggregate_rows = [across_seed(
        [r for r in seed_aggregate_rows if r["method"] == method], method=method, map_name="ALL"
    ) for method in args.methods]
    paired_map_rows = [row for map_name in args.maps
                       for row in paired_comparisons(seed_map_rows, map_name)]
    paired_aggregate_rows = paired_comparisons(seed_aggregate_rows, "ALL")

    write_rows(root / "all_episodes.csv", all_episodes)
    write_rows(root / "seed_summary_by_map.csv", seed_map_rows)
    write_rows(root / "seed_summary_aggregate.csv", seed_aggregate_rows)
    write_rows(root / "pooled_summary_by_map.csv", pooled_map_rows)
    write_rows(root / "pooled_summary_aggregate.csv", pooled_aggregate_rows)
    write_rows(root / "across_seed_summary_by_map.csv", across_map_rows)
    write_rows(root / "across_seed_summary_aggregate.csv", across_aggregate_rows)
    write_rows(root / "paired_method_comparisons_by_map.csv", paired_map_rows)
    write_rows(root / "paired_method_comparisons_aggregate.csv", paired_aggregate_rows)
    # Backward-compatible names now point to the statistically richer pooled summaries.
    combined_path = root / "combined_summary_by_map.csv"
    aggregate_path = root / "aggregate_summary.csv"
    write_rows(combined_path, pooled_map_rows)
    write_rows(aggregate_path, pooled_aggregate_rows)

    print("\n[POOLED AGGREGATE SUMMARY]")
    for row in pooled_aggregate_rows:
        print(row)

    print("\n" + "=" * 72)
    print("Laser structured unseen-map evaluation completed")
    print(f"Results:   {root}")
    print(f"By map:    {combined_path}")
    print(f"Aggregate: {aggregate_path}")
    print(f"Across seed: {root / 'across_seed_summary_by_map.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
