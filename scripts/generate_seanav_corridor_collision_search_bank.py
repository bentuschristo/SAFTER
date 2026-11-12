#!/usr/bin/env python3
"""SeaNavMapBank carrying 40 NEW shared (start, goal, yaw) scenarios on
corridor, drawn from a fresh structured run (evaluation_seed=5000, case_id
0-39) NOT overlapping the original 25 used in the main medium-shift figure
-- requested 2026-09-24 to find genuine SEA-Nav collisions (not stuck/fall)
to replace the figure's 5 "stuck" failure cases, since collisions are rare
enough (~1/25 in the original pool) that a bigger fresh pool is needed.
Mirrors scripts/generate_seanav_corridor_shared_scenarios_bank.py.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ARENA_WIDTH = 15.0
OBSTACLE_HEIGHT = 1.0

OBSTACLES = [
    (-3.8, -2.2, 0.50, "circle"),
    (-1.9, -2.2, 0.50, "circle"),
    (0.0, -2.2, 0.50, "circle"),
    (1.9, -2.2, 0.50, "circle"),
    (3.8, -2.2, 0.50, "circle"),
    (-3.8, 2.2, 0.50, "circle"),
    (-1.9, 2.2, 0.50, "circle"),
    (1.9, 2.2, 0.50, "circle"),
    (3.8, 2.2, 0.50, "circle"),
]

SHARED_MANIFEST = Path(
    "logs/evaluation/laser_unseen_maps_structured_refixed/"
    "corridor_shared_scenarios_medium_shift_collision_search/seed_5000/corridor/scenarios.csv"
)
NUM_SHARED_CASES = 40


def load_shared_scenarios() -> list[dict[str, float]]:
    with SHARED_MANIFEST.open() as f:
        rows = [r for r in csv.DictReader(f)]
    rows = [r for r in rows if int(r["case_id"]) < NUM_SHARED_CASES]
    rows.sort(key=lambda r: int(r["case_id"]))
    assert len(rows) == NUM_SHARED_CASES, f"expected {NUM_SHARED_CASES} rows, got {len(rows)}"
    return [
        {"case_id": int(r["case_id"]), "start_x": float(r["start_x"]), "start_y": float(r["start_y"]),
         "start_yaw": float(r["start_yaw"]), "goal_x": float(r["goal_x"]), "goal_y": float(r["goal_y"])}
        for r in rows
    ]


def main() -> None:
    shared = load_shared_scenarios()
    out_root = Path("generated/procedural_maps/seanav_corridor_collision_search")
    (out_root / "maps").mkdir(parents=True, exist_ok=True)

    difficulties = ["easy", "medium", "hard"]
    manifest_maps = []
    all_scenarios = []
    for map_id, difficulty in enumerate(difficulties):
        map_json = {
            "map_id": map_id, "name": "corridor", "difficulty": difficulty,
            "arena": {"width": ARENA_WIDTH, "length": ARENA_WIDTH},
            "obstacles": [
                {"x": x, "y": y, "radius": r, "height": OBSTACLE_HEIGHT, "shape": shape}
                for (x, y, r, shape) in OBSTACLES
            ],
        }
        (out_root / "maps" / f"map_{map_id:03d}.json").write_text(json.dumps(map_json, indent=2))
        manifest_maps.append({"map_id": map_id, "difficulty": difficulty, "file": f"maps/map_{map_id:03d}.json"})
        for s in shared:
            all_scenarios.append({
                "scenario_id": s["case_id"], "map_id": map_id,
                "start_x": s["start_x"], "start_y": s["start_y"],
                "goal_x": s["goal_x"], "goal_y": s["goal_y"],
                "initial_yaw_rad": s["start_yaw"],
            })

    (out_root / "bank.json").write_text(json.dumps({
        "schema_version": 1,
        "generator": "generate_seanav_corridor_collision_search_bank.py",
        "num_maps": len(manifest_maps),
        "scenarios_per_map": len(shared),
        "maps": manifest_maps,
    }, indent=2))

    fieldnames = ["scenario_id", "map_id", "start_x", "start_y", "goal_x", "goal_y", "initial_yaw_rad"]
    with open(out_root / "scenarios.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_scenarios)

    print(f"Wrote bank to {out_root}/ ({len(shared)} shared scenarios x 3 difficulty labels)")


if __name__ == "__main__":
    main()
