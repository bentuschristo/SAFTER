"""Fresh smoke training run (Task 12) for the ZOH-consistent joint
box+half-space projection fix + reset-feasibility guard (2026-09 fix).

Same configuration as the earlier smoke run
(2026-09-07_15-33-05_smoke_bcbf_cf_invariant_ellipse_dsafe030_seed0):
method=bcbf_cf, num_envs=64, vector_steps=2000, learning_starts=200,
updates_per_step=1 (default), d_safe=0.30, 120 rays, 360deg FOV, LiDAR noise
enabled, terminal_backup_mode=invariant_ellipse, terminal scales=1.0,
hard_clip_output=True, same 100-map/20,000-scenario bank, same curriculum
-- but now with the new projection + reset-feasibility guard active, and
enable_bcbf_diagnostics=True so the new TensorBoard scalars are populated.

Run:
    python scripts_august/scripts_v6/run_smoke_bcbf_cf_jointproj.py
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "scripts_august/scripts_v6")
train_mod = importlib.import_module("train_high_level_sac_laser_four_methods")


def main() -> None:
    cfg = train_mod.TrainLaserSacConfig(
        low_level_checkpoint="logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt",
        method="bcbf_cf",
        num_envs=64,
        device="cuda:0",
        vector_steps=2000,
        save_interval=1000,
        log_interval=10,
        seed=0,
        run_name="smoke_bcbf_cf_jointproj_certifiedinit_dsafe030_seed0",
        curriculum_mode="quarters",
        laser_num_rays=120,
        laser_fov_deg=360.0,
        enable_lidar_noise=True,
        safety_clearance=0.30,
        collision_geometry="circle",
        collision_circle_radius_m=0.30,
        collision_termination_mode="circle",
        hard_clip_output=True,
        procedural_map_bank="generated/procedural_maps/seanav_100_seed42_200scen",
        expected_num_maps=100,
        expected_scenarios_per_map=200,
        enable_collision_initialization=False,
        enable_bcbf_diagnostics=True,
    )
    cfg = replace(
        cfg,
        config=replace(
            cfg.config,
            safety=replace(
                cfg.config.safety,
                terminal_backup_mode="invariant_ellipse",
                terminal_geom_scale=1.0,
                terminal_rec_scale=1.0,
            ),
            sac=replace(cfg.config.sac, learning_starts=200),
        ),
    )
    train_mod.run("Unitree-Go2-Safe-Navigation", cfg)


if __name__ == "__main__":
    main()
