"""Resolve the runtime LaserSafetyConfig produced by the training command,
using the same formula as run() in train_SAC_policies.py:

    front_body_distance = max(0.5*robot_length_m - lidar_x_m, 1e-8)
    footprint_clearance  = max(0.0, safety_clearance - front_body_distance)
    training_safety_cfg  = replace(base_safety, d_safe=safety_clearance,
                                    footprint_clearance_m=footprint_clearance, ...)

Shared by the pre-training validation and the configuration dump so they
stay consistent with the training script.
"""
from __future__ import annotations

from dataclasses import replace

from SAFTER_src.high_level_laser_sac.config import LaserSafetyConfig


def resolve_runtime_safety_config(
    safety_clearance: float,
    *,
    terminal_backup_mode: str = "legacy_inflated_ellipse",
    terminal_geom_scale: float = 1.0,
    terminal_rec_scale: float = 1.0,
    base_safety: LaserSafetyConfig | None = None,
) -> LaserSafetyConfig:
    """Replicate the training script's --safety-clearance handling.

    ``base_safety`` defaults to ``LaserSafetyConfig()``, which matches the
    training script's safety config for the bcbf_cf command.
    """
    if base_safety is None:
        base_safety = LaserSafetyConfig()
    front_body_distance = max(
        0.5 * float(base_safety.robot_length_m) - float(base_safety.lidar_x_m),
        1.0e-8,
    )
    footprint_clearance = max(0.0, float(safety_clearance) - front_body_distance)
    return replace(
        base_safety,
        d_safe=float(safety_clearance),
        footprint_clearance_m=footprint_clearance,
        terminal_backup_mode=terminal_backup_mode,
        terminal_geom_scale=float(terminal_geom_scale),
        terminal_rec_scale=float(terminal_rec_scale),
    )
