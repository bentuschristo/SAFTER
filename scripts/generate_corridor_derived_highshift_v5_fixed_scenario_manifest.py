#!/usr/bin/env python3
"""Write a manually-pinned scenarios.csv for a fixed-scenario repeated-trial
run of cbf_cd_cf + bcbf_cf on corridor_derived_highshift_v5 -- the same
mechanism used for circle_and_square's fixed-scenario figure (see
scripts/plot_circle_and_square_fixed_scenario_trajectories.py's docstring):
evaluate_laser_unseen_maps_structured.py resumes from an existing
<run_root>/seed_<seed>/<map>/scenarios.csv instead of generating a fresh
random manifest, so pre-writing one with N rows that all share the SAME
start/goal/yaw (only case_id/reset_seed vary) forces N repeated trials of
one exact scenario.

This is the 4-method trajectory-sample-comparison request (2026-09-24):
CBF-RL / SEA-Nav / bCBF-CF (SAFTER) / CBF-CF (cbf_cd_cf) on the v5 map, 8
samples per method reflecting each method's measured success rate, all
sharing the same initial condition and goal. This script covers the
cbf_cd_cf + bcbf_cf half (both run together off ONE shared manifest since
evaluate_method_vectorized batches all requested --methods against the same
manifest cases). scripts/eval_cbf_rl_v5_fixed_scenario_repeated.py and
scripts/generate_seanav_v5_fixed_scenario_bank.py cover CBF-RL/SEA-Nav with
the identical START_XY/GOAL_XY/START_YAW.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

MAP_NAME = "corridor_derived_highshift_v5"
START_REGION = (-2.0, 2.0, 4.0, 7.5)
GOAL_REGION = (-4.0, 4.0, -7.5, -4.0)

START_XY = (0.0, 6.2)
GOAL_XY = (0.0, -6.2)
START_YAW = math.atan2(GOAL_XY[1] - START_XY[1], GOAL_XY[0] - START_XY[0])

N_REPEATS = 30
BASE_SEED = 5000

RUN_ROOT = Path(
    "logs/evaluation/laser_unseen_maps_structured_refixed/"
    "corridor_derived_highshift_v5_fixed_scenario_repeated"
)


def main() -> None:
    straight_line_distance = math.hypot(GOAL_XY[0] - START_XY[0], GOAL_XY[1] - START_XY[1])
    rows = []
    for i in range(N_REPEATS):
        rows.append({
            "map_name": MAP_NAME,
            "case_id": i,
            "reset_seed": BASE_SEED + i,
            "sampling_attempt": 1,
            "start_region_x_min": START_REGION[0], "start_region_x_max": START_REGION[1],
            "start_region_y_min": START_REGION[2], "start_region_y_max": START_REGION[3],
            "goal_region_x_min": GOAL_REGION[0], "goal_region_x_max": GOAL_REGION[1],
            "goal_region_y_min": GOAL_REGION[2], "goal_region_y_max": GOAL_REGION[3],
            "start_x": START_XY[0], "start_y": START_XY[1],
            "start_yaw": START_YAW, "start_yaw_nominal_to_goal": START_YAW,
            "goal_x": GOAL_XY[0], "goal_y": GOAL_XY[1],
            "start_clearance_m": 0.0, "goal_clearance_m": 0.0,
            "straight_line_distance_m": straight_line_distance,
            "straight_path_min_clearance_m": 0.0,
            "direct_path_blocked": 1,
            "blocking_obstacle_id": -1,
            "blocking_t": 0.5,
            "blocking_distance_m": 0.0,
            "blocking_clearance_m": 0.0,
            "blocking_threshold_m": 0.0,
            "block_margin_m": 0.10,
            "min_endpoint_clearance_m": 0.10,
        })

    seed_dir = RUN_ROOT / f"seed_{BASE_SEED}" / MAP_NAME
    seed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = seed_dir / "scenarios.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} repeated-scenario rows to {manifest_path}")
    print(f"START_XY={START_XY}  GOAL_XY={GOAL_XY}  START_YAW={START_YAW:.6f} rad")


if __name__ == "__main__":
    main()
