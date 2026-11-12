#!/usr/bin/env python3
"""One-off 3-map SeaNavMapBank for the fail/fail/fail/succeed trajectory demo
(slalom / corridor [original] / clustered_v2 [latest]), each map holding a
SINGLE forced scenario (the exact start/goal/yaw shared with the laser-SAC
methods' trajectory logging for the same comparison), so
evaluate_native_maps.py's SeaNavMapBank.sample_scenario() always returns
that one row regardless of RNG state -- every reset for that terrain row
replays the identical episode.

Obstacle layouts copied verbatim from
scripts/laser_unseen_eval/evaluate_laser_unseen_maps_structured.py's
UNSEEN_MAPS["slalom"/"corridor"/"clustered_v2"] (NOT "new_medium_v2" --
this demo deliberately uses the original "corridor", matching what the
laser-SAC trajectory comparison also uses for this specific request).
"""
from __future__ import annotations

import json
import csv
import math
from pathlib import Path

ARENA_WIDTH = 15.0
OBSTACLE_HEIGHT = 1.0

MAPS = {
    "slalom": {
        "difficulty": "easy",
        "obstacles": [
            (-5.2, -2.3, 0.50), (-3.9, 2.1, 0.50), (-2.6, -2.2, 0.50),
            (-1.3, 2.2, 0.50), (0.0, -2.2, 0.50), (1.3, 2.2, 0.50),
            (2.6, -2.2, 0.50), (3.9, 2.1, 0.50), (5.2, -2.3, 0.50),
        ],
    },
    "corridor": {
        "difficulty": "medium",
        "obstacles": [
            (-3.8, -2.2, 0.50), (-1.9, -2.2, 0.50), (0.0, -2.2, 0.50),
            (1.9, -2.2, 0.50), (3.8, -2.2, 0.50), (-3.8, 2.2, 0.50),
            (-1.9, 2.2, 0.50), (1.9, 2.2, 0.50), (3.8, 2.2, 0.50),
        ],
    },
    "clustered_v2": {
        "difficulty": "hard",
        "obstacles": [
            (-3.0, 1.4, 0.55), (-1.3, 2.7, 0.45), (-0.08, 0.55, 0.60),
            (-2.4, 4.1, 0.50), (0.5, 3.9, 0.40), (1.98, -2.71, 0.50),
            (3.35, -1.44, 0.45), (-0.42, -1.55, 0.55), (3.47, -3.86, 0.50),
        ],
    },
}

# Filled in by pick_scenarios.py cross-referencing the laser eval's chosen
# fail/fail/fail/succeed cases -- one (start_x, start_y, goal_x, goal_y, yaw)
# per map, in the SAME world-centered (+-7.5) coordinate frame as the laser
# UNSEEN_MAPS definitions.
FIXED_SCENARIOS = {
    "slalom": (3.918546438217163, 5.2431817054748535, 2.661121368408203, -4.5701727867126465, -1.6340805884506502),
    "corridor": (-1.1047950983047485, 4.677678108215332, 2.4622621536254883, -4.331889629364014, -1.0268072382693632),
    "clustered_v2": (-5.67845344543457, 6.151569366455078, 6.6676025390625, -6.458984851837158, -0.9433746979934128),
}


def main() -> None:
    out_root = Path("generated/procedural_maps/seanav_trajectory_demo_slalom_corridor_clustered_v2")
    (out_root / "maps").mkdir(parents=True, exist_ok=True)

    manifest_maps = []
    all_scenarios = []
    for map_id, (name, spec) in enumerate(MAPS.items()):
        if FIXED_SCENARIOS.get(name) is None:
            raise RuntimeError(f"FIXED_SCENARIOS[{name!r}] not set yet -- fill it in before running.")
        manifest_maps.append({"map_id": map_id, "difficulty": spec["difficulty"], "file": f"maps/map_{map_id:03d}.json"})
        map_json = {
            "map_id": map_id,
            "name": name,
            "difficulty": spec["difficulty"],
            "arena": {"width": ARENA_WIDTH, "length": ARENA_WIDTH},
            "obstacles": [
                {"x": x, "y": y, "radius": r, "height": OBSTACLE_HEIGHT}
                for (x, y, r) in spec["obstacles"]
            ],
        }
        (out_root / "maps" / f"map_{map_id:03d}.json").write_text(json.dumps(map_json, indent=2))
        sx, sy, gx, gy, yaw = FIXED_SCENARIOS[name]
        all_scenarios.append({
            "scenario_id": 0, "map_id": map_id,
            "start_x": sx, "start_y": sy, "goal_x": gx, "goal_y": gy,
            "initial_yaw_rad": yaw,
        })
        print(f"{name} (map_id={map_id}, {spec['difficulty']}): 1 forced scenario written")

    (out_root / "bank.json").write_text(json.dumps({
        "schema_version": 1,
        "generator": "generate_seanav_trajectory_demo_bank.py",
        "num_maps": len(manifest_maps),
        "scenarios_per_map": 1,
        "maps": manifest_maps,
    }, indent=2))

    fieldnames = ["scenario_id", "map_id", "start_x", "start_y", "goal_x", "goal_y", "initial_yaw_rad"]
    with open(out_root / "scenarios.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_scenarios)

    print(f"\nWrote bank to {out_root}/")


if __name__ == "__main__":
    main()
