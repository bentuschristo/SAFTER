#!/usr/bin/env python3
"""One-map SeaNavMapBank for a corridor_derived_highshift_v5 fixed-scenario
repeated-trial run -- a SINGLE forced scenario (the exact start/goal/yaw
shared with the SAC-laser fixed-scenario repeats
(scripts/generate_corridor_derived_highshift_v5_fixed_scenario_manifest.py)
and the CBF-RL fixed-scenario repeats
(scripts/eval_cbf_rl_v5_fixed_scenario_repeated.py)), so
SeaNavMapBank.sample_scenario() always returns that one row regardless of
RNG state -- every reset replays the identical episode. Part of the 4-method
8-sample trajectory comparison figure requested 2026-09-24.

Obstacle layout copied verbatim from
SAFTER_scripts/laser_unseen_eval/evaluate_laser_unseen_maps_structured.py's
UNSEEN_MAPS["corridor_derived_highshift_v5"] / scripts/
plot_corridor_derived_highshift_v5_preview.py.
"""
from __future__ import annotations

import json
import csv
import math
from pathlib import Path

ARENA_WIDTH = 15.0
OBSTACLE_HEIGHT = 1.0

OBSTACLES = [
    (-4.000, 2.2, 0.65, "circle"),
    (-2.000, 2.2, 0.65, "circle"),
    (2.000, 2.2, 0.65, "circle"),
    (4.000, 2.2, 0.65, "circle"),
    (-4.080, -0.4, 0.50, "circle"),
    (-2.350, -1.3, 0.65, "circle"),
    (0.000, -2.2, 0.90, "square"),
    (2.350, -1.3, 0.65, "circle"),
    (4.080, -0.4, 0.50, "circle"),
]

# Shared with the SAC-laser and CBF-RL fixed-scenario repeats for this same
# request.
START_XY = (0.0, 6.2)
GOAL_XY = (0.0, -6.2)
START_YAW = math.atan2(GOAL_XY[1] - START_XY[1], GOAL_XY[0] - START_XY[0])

FIXED_SCENARIO = {
    "start_x": START_XY[0], "start_y": START_XY[1],
    "start_yaw": START_YAW,
    "goal_x": GOAL_XY[0], "goal_y": GOAL_XY[1],
}


def main() -> None:
    # terrain.py's SeaNavMapBank requires the bank's per-difficulty map
    # counts to exactly fill a num_rows x num_cols grid, and the env config
    # this task uses assigns a fixed 3 rows (one per difficulty tier) x 1
    # col -- a bank with only 1 map fails ("Bank has 1 'hard' maps but the
    # assigned rows [0,3) x 1 cols need exactly 3"). Work around this by
    # writing the SAME v5 map/scenario three times, one per difficulty
    # label (easy/medium/hard) -- the label is cosmetic bookkeeping only,
    # every one of the 3 "maps" is physically identical, so this just
    # yields 3x the repeats for the one scenario we actually care about.
    out_root = Path("generated/procedural_maps/seanav_corridor_derived_highshift_v5_fixed_scenario")
    (out_root / "maps").mkdir(parents=True, exist_ok=True)

    difficulties = ["easy", "medium", "hard"]
    manifest_maps = []
    scenario_rows = []
    for map_id, difficulty in enumerate(difficulties):
        map_json = {
            "map_id": map_id,
            "name": "corridor_derived_highshift_v5",
            "difficulty": difficulty,
            "arena": {"width": ARENA_WIDTH, "length": ARENA_WIDTH},
            "obstacles": [
                {"x": x, "y": y, "radius": r, "height": OBSTACLE_HEIGHT, "shape": shape}
                for (x, y, r, shape) in OBSTACLES
            ],
        }
        (out_root / "maps" / f"map_{map_id:03d}.json").write_text(json.dumps(map_json, indent=2))
        manifest_maps.append({"map_id": map_id, "difficulty": difficulty, "file": f"maps/map_{map_id:03d}.json"})
        scenario_rows.append({
            "scenario_id": 0, "map_id": map_id,
            "start_x": FIXED_SCENARIO["start_x"], "start_y": FIXED_SCENARIO["start_y"],
            "goal_x": FIXED_SCENARIO["goal_x"], "goal_y": FIXED_SCENARIO["goal_y"],
            "initial_yaw_rad": FIXED_SCENARIO["start_yaw"],
        })

    (out_root / "bank.json").write_text(json.dumps({
        "schema_version": 1,
        "generator": "generate_seanav_v5_fixed_scenario_bank.py",
        "num_maps": len(manifest_maps),
        "scenarios_per_map": 1,
        "maps": manifest_maps,
    }, indent=2))

    fieldnames = ["scenario_id", "map_id", "start_x", "start_y", "goal_x", "goal_y", "initial_yaw_rad"]
    with open(out_root / "scenarios.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scenario_rows)

    print(f"Wrote bank to {out_root}/")
    print(f"START_XY={START_XY}  GOAL_XY={GOAL_XY}  START_YAW={START_YAW:.6f} rad")


if __name__ == "__main__":
    main()
