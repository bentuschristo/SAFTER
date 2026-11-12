from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from SAFTER_src.high_level_rl.arena import arena_from_json
from SAFTER_src.high_level_rl.config import ArenaConfig


DIFFICULTIES = ("easy", "medium", "hard")
CURRICULUM_VERSION = "procedural_maps_quarters_v1"
CURRICULUM_WEIGHTS = (
    (0.80, 0.15, 0.05),
    (0.40, 0.45, 0.15),
    (0.20, 0.45, 0.35),
    (0.10, 0.30, 0.60),
)


@dataclass(frozen=True)
class ProceduralScenario:
    scenario_id: int
    map_id: int
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    start_yaw: float
    difficulty: str
    difficulty_score: float
    direct_distance: float
    path_length: float
    detour_ratio: float
    minimum_path_clearance: float
    direct_path_blocked: bool
    path_turn_count: int


@dataclass(frozen=True)
class ProceduralMap:
    map_id: int
    difficulty: str
    difficulty_score: float
    arena: ArenaConfig
    scenarios: tuple[ProceduralScenario, ...]


@dataclass(frozen=True)
class ProceduralMapBank:
    root: Path
    maps: tuple[ProceduralMap, ...]
    generation_seed: int
    num_scenarios: int
    max_obstacles: int
    scenarios_per_map: int
    content_hash: str
    schema_version: int
    difficulty_mode: str = "mixed"

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "path": str(self.root),
            "generation_seed": self.generation_seed,
            "num_maps": len(self.maps),
            "num_scenarios": self.num_scenarios,
            "content_hash": self.content_hash,
            "schema_version": self.schema_version,
            "scenarios_per_map": self.scenarios_per_map,
            "curriculum_version": CURRICULUM_VERSION,
            "curriculum_weights": CURRICULUM_WEIGHTS,
            "max_obstacles": self.max_obstacles,
            "difficulty_mode": self.difficulty_mode,
        }


def _canonical_content_hash(root: Path) -> str:
    required = [root / "bank.json", root / "generation_config.json", root / "scenarios.csv"]
    files = required + sorted((root / "arenas").glob("*.json")) + sorted((root / "maps").glob("*.json"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing authoritative bank files: {missing}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big")); digest.update(relative)
        digest.update(len(content).to_bytes(8, "big")); digest.update(content)
    return digest.hexdigest()


def load_procedural_map_bank(
    bank_dir: str | Path,
    *,
    expected_num_maps: int | None = 100,
    expected_scenarios_per_map: int | None = 64,
    expected_max_obstacles: int | None = 12,
) -> ProceduralMapBank:
    root = Path(bank_dir).expanduser().resolve()
    bank_path = root / "bank.json"
    scenario_path = root / "scenarios.csv"
    if not bank_path.is_file() or not scenario_path.is_file():
        raise FileNotFoundError(f"Incomplete procedural map bank: {root}")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    entries = bank.get("maps")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Procedural map bank has no maps: {bank_path}")

    scenarios_by_map: dict[int, list[ProceduralScenario]] = {}
    required_columns = {
        "scenario_id", "map_id", "difficulty", "difficulty_score", "start_x", "start_y",
        "goal_x", "goal_y", "initial_yaw_rad", "direct_distance", "path_length", "detour_ratio",
        "minimum_path_clearance", "direct_path_blocked", "path_turn_count",
    }
    with scenario_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"scenarios.csv is missing required columns: {sorted(missing_columns)}")
        for row in reader:
            difficulty = row["difficulty"]
            if difficulty not in DIFFICULTIES:
                raise ValueError(f"Scenario {row['scenario_id']} has invalid difficulty {difficulty!r}")
            blocked = row["direct_path_blocked"].strip().lower()
            if blocked not in {"true", "false"}:
                raise ValueError(
                    f"Scenario {row['scenario_id']} has invalid direct_path_blocked={row['direct_path_blocked']!r}"
                )
            scenario = ProceduralScenario(
                scenario_id=int(row["scenario_id"]), map_id=int(row["map_id"]),
                start_xy=(float(row["start_x"]), float(row["start_y"])),
                goal_xy=(float(row["goal_x"]), float(row["goal_y"])),
                start_yaw=float(row["initial_yaw_rad"]),
                difficulty=difficulty, difficulty_score=float(row["difficulty_score"]),
                direct_distance=float(row["direct_distance"]), path_length=float(row["path_length"]),
                detour_ratio=float(row["detour_ratio"]),
                minimum_path_clearance=float(row["minimum_path_clearance"]),
                direct_path_blocked=blocked == "true",
                path_turn_count=int(row["path_turn_count"]),
            )
            scenarios_by_map.setdefault(scenario.map_id, []).append(scenario)

    maps: list[ProceduralMap] = []
    seen_ids: set[int] = set()
    for entry in entries:
        map_id = int(entry["map_id"])
        difficulty = str(entry["difficulty"])
        if map_id in seen_ids or difficulty not in DIFFICULTIES:
            raise ValueError(f"Invalid or duplicate map entry: {entry}")
        scenarios = tuple(scenarios_by_map.get(map_id, ()))
        if not scenarios:
            raise ValueError(f"Map {map_id} has no saved scenarios")
        maps.append(ProceduralMap(
            map_id=map_id, difficulty=difficulty,
            difficulty_score=float(entry["difficulty_score"]),
            arena=arena_from_json(root / str(entry["arena_file"])), scenarios=scenarios,
        ))
        seen_ids.add(map_id)
    if len(maps) != int(bank.get("num_maps", len(maps))):
        raise ValueError("bank.json num_maps does not match its entries")
    num_scenarios = sum(len(item.scenarios) for item in maps)
    if num_scenarios != int(bank.get("num_scenarios", num_scenarios)):
        raise ValueError("bank.json num_scenarios does not match scenarios.csv")
    scenario_ids = sorted(s.scenario_id for item in maps for s in item.scenarios)
    if scenario_ids != list(range(num_scenarios)):
        raise ValueError("Scenario IDs must be unique and contiguous from zero")
    map_ids = {item.map_id for item in maps}
    if any(s.map_id not in map_ids for scenarios in scenarios_by_map.values() for s in scenarios):
        raise ValueError("A scenario references a nonexistent map")
    counts_per_map = {item.map_id: len(item.scenarios) for item in maps}
    unique_counts = set(counts_per_map.values())
    if len(unique_counts) != 1:
        raise ValueError(f"Maps have inconsistent scenario counts: {counts_per_map}")
    scenarios_per_map = next(iter(unique_counts))
    difficulty_mode = str(bank.get("difficulty_mode", "mixed"))
    if difficulty_mode not in ("mixed", *DIFFICULTIES):
        raise ValueError(f"Invalid bank difficulty_mode: {difficulty_mode!r}")
    required_difficulties = DIFFICULTIES if difficulty_mode == "mixed" else (difficulty_mode,)
    for difficulty in required_difficulties:
        if not any(s.difficulty == difficulty for item in maps for s in item.scenarios):
            raise ValueError(f"Bank has no {difficulty!r} scenarios")
    if difficulty_mode != "mixed":
        unexpected = sorted({s.difficulty for item in maps for s in item.scenarios} - {difficulty_mode})
        if unexpected:
            raise ValueError(
                f"Single-difficulty bank {difficulty_mode!r} contains scenario labels {unexpected}"
            )
    actual_max_slots = int(bank["max_obstacles"])
    invalid_capacity = [
        item.map_id for item in maps
        if item.arena.max_obstacles != actual_max_slots or len(item.arena.obstacles) > actual_max_slots
    ]
    if invalid_capacity:
        raise ValueError(
            f"Arena max_obstacles/count disagrees with bank capacity {actual_max_slots}: maps={invalid_capacity}"
        )
    expectations = {
        "maps": (len(maps), expected_num_maps),
        "scenarios_per_map": (scenarios_per_map, expected_scenarios_per_map),
        "max_obstacles": (actual_max_slots, expected_max_obstacles),
    }
    failures = [f"{name}: actual={actual}, expected={expected}" for name, (actual, expected) in expectations.items()
                if expected is not None and actual != expected]
    if failures:
        raise ValueError("Procedural bank expectation failure: " + "; ".join(failures))
    return ProceduralMapBank(
        root=root, maps=tuple(maps), generation_seed=int(bank["seed"]),
        num_scenarios=num_scenarios, max_obstacles=actual_max_slots,
        scenarios_per_map=scenarios_per_map, content_hash=_canonical_content_hash(root),
        schema_version=int(bank["schema_version"]),
        difficulty_mode=difficulty_mode,
    )


def resolve_slot_shapes(maps: tuple[ProceduralMap, ...], max_obstacles: int) -> tuple[str, ...]:
    """Return the physical shape ("circle"/"square") of each of the bank's
    ``max_obstacles`` reusable obstacle slots.

    Every map in a bank shares one pool of ``max_obstacles`` reusable mocap
    bodies; a given map's i-th obstacle always occupies slot i (see
    ``pad_obstacle_slots`` in the generator). A physical MJCF geom's *type*
    (box vs. cylinder) cannot change at runtime -- only its size/pose can --
    so every map that uses slot i must request the same shape there. This is
    trivially satisfied by a single-map bank (e.g. a hand-authored custom
    map); a multi-map bank must keep each shared slot's shape consistent
    across all of its maps, or this raises.
    """
    slot_shapes: list[str | None] = [None] * max_obstacles
    for item in maps:
        for obstacle_index, obstacle in enumerate(item.arena.obstacles):
            shape = getattr(obstacle, "shape", "circle")
            if shape not in ("circle", "square"):
                raise ValueError(
                    f"Map {item.map_id} obstacle {obstacle_index} has unsupported shape={shape!r}"
                )
            existing = slot_shapes[obstacle_index]
            if existing is None:
                slot_shapes[obstacle_index] = shape
            elif existing != shape:
                raise ValueError(
                    f"Obstacle slot {obstacle_index} is {existing!r} in an earlier map but "
                    f"{shape!r} in map_id={item.map_id}. Every map in one bank must place the "
                    "same shape in the same physical slot (slot index follows each map's own "
                    "obstacle list order), since a slot's MJCF geom type is fixed at scene-"
                    "authoring time and cannot change per map at runtime."
                )
    return tuple(shape or "circle" for shape in slot_shapes)
