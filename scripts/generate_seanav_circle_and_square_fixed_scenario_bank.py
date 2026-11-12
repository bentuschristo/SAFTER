#!/usr/bin/env python3
"""One-map SeaNavMapBank for the circle_and_square (High-Shift/Hard) fixed-
scenario repeated-trial experiment: a SINGLE forced scenario (the exact
start/goal/yaw shared with the SAC-laser and CBF-RL fixed-scenario repeats
for this same request), so SeaNavMapBank.sample_scenario() always returns
that one row regardless of RNG state -- every reset replays the identical
episode, matching the pattern in generate_seanav_trajectory_demo_bank.py.

Obstacle layout copied verbatim from
scripts/laser_unseen_eval/evaluate_laser_unseen_maps_structured.py's
UNSEEN_MAPS["circle_and_square"] (confirmed identical to
generated/procedural_maps/seanav_native_circle_and_square/maps/map_002.json).

Fixed scenario = circle_and_square case_id=3, reset_seed=31003,
evaluation_seed=1000 (from all_maps_bcbf_cf_1500cases.csv / the raw
logs/evaluation/circle_and_square_seaspeed/ eval), already known to be a
case where bcbf_cf=success, vanilla=collision, cbf_cd_cf=collision.
"""
from __future__ import annotations

import json
import csv
from pathlib import Path

ARENA_WIDTH = 15.0
OBSTACLE_HEIGHT = 1.0

OBSTACLES = [
    (-1.6, 3.4, 0.35, "circle"),
    (1.6, 3.4, 0.35, "circle"),
    (0.0, 1.9, 1.00, "circle"),
    (-1.5, -0.2, 0.35, "circle"),
    (1.5, -0.2, 0.35, "circle"),
    (-4.6, -2.3, 0.65, "circle"),
    (-1.0, -2.6, 0.40, "square"),
    (1.0, -2.6, 0.40, "square"),
    (4.6, -2.3, 0.65, "circle"),
    (-3.6, 0.4, 0.70, "square"),
    (3.6, 0.4, 0.70, "square"),
]

FIXED_SCENARIO = {
    "start_x": -0.948674738407135,
    "start_y": 6.362679481506348,
    "start_yaw": -1.3482832581941522,
    "goal_x": -0.3806767463684082,
    "goal_y": -5.078561305999756,
}


def main() -> None:
    # terrain.py's SeaNavMapBank requires the bank's per-difficulty map
    # counts to exactly fill a num_rows x num_cols grid, and the env config
    # this task uses assigns a fixed 3 rows (one per difficulty tier) x 1
    # col -- so a bank with only 1 map fails ("Bank has 1 'hard' maps but
    # the assigned rows [0,3) x 1 cols need exactly 3"). Work around this by
    # writing the SAME circle_and_square map/scenario three times, one per
    # difficulty label (easy/medium/hard) -- the label is cosmetic bookkeeping
    # only, every one of the 3 "maps" is physically identical, so this just
    # yields 3x the repeats for the one scenario we actually care about.
    out_root = Path("generated/procedural_maps/seanav_circle_and_square_fixed_scenario")
    (out_root / "maps").mkdir(parents=True, exist_ok=True)

    difficulties = ["easy", "medium", "hard"]
    manifest_maps = []
    scenario_rows = []
    for map_id, difficulty in enumerate(difficulties):
        map_json = {
            "map_id": map_id,
            "name": "circle_and_square",
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
        "generator": "generate_seanav_circle_and_square_fixed_scenario_bank.py",
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


if __name__ == "__main__":
    main()
