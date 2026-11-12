#!/usr/bin/env python3
"""Generate and save a reproducible procedural map bank for Go2 navigation.

The generator creates circular-obstacle maps for the 15 m x 15 m arena used by
``src.high_level_rl``.  It is intended to produce the offline map/scenario bank
that will later be loaded by the laser-SAC training environment.

Main guarantees
---------------
* 0--12 active obstacles per map (configurable).
* Separated, clustered, and mixed obstacle-placement families.
* Obstacles never intersect arena walls or one another.
* At least one traversable path exists after inflating geometry by the robot
  footprint and safety clearance.
* Multiple prevalidated start/goal/yaw scenarios are saved for every map.
* Deterministic output for a fixed seed.
* Easy/medium/hard labels are assigned from measured geometric difficulty,
  rather than obstacle count alone.

Default output structure
------------------------
``generated/procedural_maps/train_100_seed42/``
    generation_config.json
    bank.json
    map_summary.csv
    scenarios.csv
    maps/map_000.json
    arenas/map_000_arena.json
    occupancy/map_000.npz
    previews/map_000.png
    previews/all_maps_mosaic.png

The files under ``arenas/`` use the same JSON field structure as
``src.high_level_rl.arena.arena_from_json``.  The richer files under ``maps/``
contain metadata and fixed 12-slot obstacle arrays for future runtime loading.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np


LayoutFamily = Literal["empty", "separated", "clustered", "mixed"]
Difficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class GeneratorConfig:
    # Bank size and reproducibility.
    num_maps: int = 100
    scenarios_per_map: int = 64
    seed: int = 42

    # Arena geometry.  These match the current navigation defaults.
    arena_width: float = 15.0
    arena_length: float = 15.0
    wall_thickness: float = 0.12
    wall_height: float = 0.80

    # Robot/safety geometry used for path feasibility.
    robot_radius: float = 0.30
    wall_clearance: float = 0.15
    obstacle_clearance: float = 0.15
    sample_margin_from_walls: float = 0.80

    # Obstacle bank.
    min_obstacles: int = 0
    max_obstacles: int = 12
    obstacle_radius_min: float = 0.40
    obstacle_radius_max: float = 0.60
    obstacle_height: float = 1.00
    obstacle_wall_gap: float = 0.20

    # Placement-family gaps are edge-to-edge distances.
    separated_gap_min: float = 0.55
    separated_gap_max: float = 1.00
    clustered_gap_min: float = 0.05
    clustered_gap_max: float = 0.25
    cluster_spread_min: float = 0.75
    cluster_spread_max: float = 1.65

    # Feasibility grid and scenario generation.
    grid_resolution: float = 0.10
    minimum_start_goal_distance: float = 4.0
    maximum_path_length: float = 30.0
    target_blocked_fraction: float = 0.70
    blocked_path_margin: float = 0.10
    scenario_sampling_attempts: int = 20_000

    # Reject maps whose largest traversable component is too small/narrow.
    minimum_largest_component_fraction: float = 0.30
    minimum_component_span: float = 5.0

    # Candidate-generation limits.
    obstacle_sampling_attempts: int = 4_000
    map_sampling_attempts: int = 800

    # Difficulty quotas used to bias generation.  Final labels are assigned
    # from measured scores with these same 30/40/30 proportions.
    easy_fraction: float = 0.30
    medium_fraction: float = 0.40
    hard_fraction: float = 0.30


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float
    height: float


@dataclass
class Scenario:
    scenario_id: int
    map_id: int
    start_x: float
    start_y: float
    goal_x: float
    goal_y: float
    initial_yaw_rad: float
    direct_distance: float
    path_length: float
    detour_ratio: float
    minimum_path_clearance: float
    direct_path_blocked: bool
    path_turn_count: int
    raw_difficulty_score: float = 0.0
    difficulty_score: float = 0.0
    difficulty: Difficulty = "medium"


@dataclass
class GeneratedMap:
    map_id: int
    generation_seed: int
    generation_target: Difficulty
    layout_family: LayoutFamily
    obstacles: list[Obstacle]
    occupancy_hash: str
    free_fraction: float
    largest_component_fraction: float
    largest_component_span_x: float
    largest_component_span_y: float
    mean_scenario_detour_ratio: float
    mean_scenario_minimum_clearance: float
    blocked_scenario_fraction: float
    mean_scenario_turn_count: float
    raw_difficulty_score: float = 0.0
    difficulty_score: float = 0.0
    difficulty: Difficulty = "medium"


@dataclass(frozen=True)
class GridData:
    occupancy: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    component_labels: np.ndarray
    largest_component_label: int
    largest_component_cells: np.ndarray
    free_fraction: float
    largest_component_fraction: float
    span_x: float
    span_y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible procedural navigation map/scenario bank."
    )
    parser.add_argument(
        "--output-dir",
        default="generated/procedural_maps/train_100_seed42",
        help="Directory in which the generated map bank is saved.",
    )
    parser.add_argument("--num-maps", type=int, default=100)
    parser.add_argument("--scenarios-per-map", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-obstacles", type=int, default=0)
    parser.add_argument("--max-obstacles", type=int, default=12)
    parser.add_argument("--radius-min", type=float, default=0.40)
    parser.add_argument("--radius-max", type=float, default=0.60)
    parser.add_argument("--grid-resolution", type=float, default=0.10)
    parser.add_argument("--minimum-start-goal-distance", type=float, default=4.0)
    parser.add_argument("--maximum-path-length", type=float, default=30.0)
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Skip individual map PNGs and the all-map mosaic.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def validate_config(cfg: GeneratorConfig) -> None:
    if cfg.num_maps <= 0:
        raise ValueError("num_maps must be positive.")
    if cfg.scenarios_per_map <= 0:
        raise ValueError("scenarios_per_map must be positive.")
    if not (0 <= cfg.min_obstacles <= cfg.max_obstacles):
        raise ValueError("Require 0 <= min_obstacles <= max_obstacles.")
    if cfg.max_obstacles <= 0:
        raise ValueError("max_obstacles must be positive.")
    if not (0 < cfg.obstacle_radius_min <= cfg.obstacle_radius_max):
        raise ValueError("Invalid obstacle radius range.")
    if cfg.grid_resolution <= 0:
        raise ValueError("grid_resolution must be positive.")
    if cfg.minimum_start_goal_distance <= 0:
        raise ValueError("minimum_start_goal_distance must be positive.")
    fractions = cfg.easy_fraction + cfg.medium_fraction + cfg.hard_fraction
    if not math.isclose(fractions, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Difficulty fractions must sum to 1.0.")
    if not (0.0 <= cfg.target_blocked_fraction <= 1.0):
        raise ValueError("target_blocked_fraction must lie in [0, 1].")


def difficulty_counts(num_maps: int, cfg: GeneratorConfig) -> dict[Difficulty, int]:
    easy = int(round(num_maps * cfg.easy_fraction))
    medium = int(round(num_maps * cfg.medium_fraction))
    hard = num_maps - easy - medium
    if hard < 0:
        medium += hard
        hard = 0
    return {"easy": easy, "medium": medium, "hard": hard}


def target_sequence(num_maps: int, cfg: GeneratorConfig) -> list[Difficulty]:
    counts = difficulty_counts(num_maps, cfg)
    sequence: list[Difficulty] = []
    for level in ("easy", "medium", "hard"):
        sequence.extend([level] * counts[level])
    return sequence


def obstacle_count_for_target(
    target: Difficulty,
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    index_within_level: int,
) -> int:
    # Guarantee at least one empty map and one fully populated map when the
    # requested global range includes them.
    if target == "easy" and index_within_level == 0 and cfg.min_obstacles == 0:
        return 0
    if target == "hard" and index_within_level == 0:
        return cfg.max_obstacles

    if target == "easy":
        lo, hi = cfg.min_obstacles, min(cfg.max_obstacles, 4)
    elif target == "medium":
        lo, hi = max(cfg.min_obstacles, 3), min(cfg.max_obstacles, 8)
    else:
        lo, hi = max(cfg.min_obstacles, 7), cfg.max_obstacles

    if lo > hi:
        lo, hi = cfg.min_obstacles, cfg.max_obstacles
    return int(rng.integers(lo, hi + 1))


def family_for_target(
    target: Difficulty,
    obstacle_count: int,
    rng: np.random.Generator,
) -> LayoutFamily:
    if obstacle_count == 0:
        return "empty"

    if target == "easy":
        choices: Sequence[LayoutFamily] = ("separated", "mixed", "clustered")
        probabilities = (0.75, 0.20, 0.05)
    elif target == "medium":
        choices = ("separated", "mixed", "clustered")
        probabilities = (0.35, 0.35, 0.30)
    else:
        choices = ("separated", "mixed", "clustered")
        probabilities = (0.10, 0.35, 0.55)
    return rng.choice(choices, p=probabilities).item()


def obstacle_center_bounds(radius: float, cfg: GeneratorConfig) -> tuple[float, float, float, float]:
    half_w = 0.5 * cfg.arena_width
    half_l = 0.5 * cfg.arena_length
    margin = radius + cfg.obstacle_wall_gap
    return -half_w + margin, half_w - margin, -half_l + margin, half_l - margin


def edge_gap(a: Obstacle, b: Obstacle) -> float:
    return math.hypot(a.x - b.x, a.y - b.y) - a.radius - b.radius


def candidate_is_valid(
    candidate: Obstacle,
    placed: Sequence[Obstacle],
    minimum_gap: float,
    cfg: GeneratorConfig,
) -> bool:
    x_lo, x_hi, y_lo, y_hi = obstacle_center_bounds(candidate.radius, cfg)
    if not (x_lo <= candidate.x <= x_hi and y_lo <= candidate.y <= y_hi):
        return False
    return all(edge_gap(candidate, other) >= minimum_gap for other in placed)


def sample_uniform_obstacle(rng: np.random.Generator, cfg: GeneratorConfig) -> Obstacle:
    radius = float(rng.uniform(cfg.obstacle_radius_min, cfg.obstacle_radius_max))
    x_lo, x_hi, y_lo, y_hi = obstacle_center_bounds(radius, cfg)
    return Obstacle(
        x=float(rng.uniform(x_lo, x_hi)),
        y=float(rng.uniform(y_lo, y_hi)),
        radius=radius,
        height=cfg.obstacle_height,
    )


def place_separated_obstacles(
    count: int,
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    target: Difficulty,
) -> list[Obstacle] | None:
    if count == 0:
        return []

    if target == "easy":
        gap_lo, gap_hi = max(cfg.separated_gap_min, 0.80), cfg.separated_gap_max
    elif target == "medium":
        gap_lo, gap_hi = cfg.separated_gap_min, 0.85
    else:
        gap_lo, gap_hi = 0.30, cfg.separated_gap_min
    minimum_gap = float(rng.uniform(gap_lo, max(gap_lo, gap_hi)))

    placed: list[Obstacle] = []
    for _ in range(cfg.obstacle_sampling_attempts):
        candidate = sample_uniform_obstacle(rng, cfg)
        if candidate_is_valid(candidate, placed, minimum_gap, cfg):
            placed.append(candidate)
            if len(placed) == count:
                return placed
    return None


def sample_cluster_centers(
    num_clusters: int,
    rng: np.random.Generator,
    cfg: GeneratorConfig,
) -> list[tuple[float, float]]:
    centers: list[tuple[float, float]] = []
    # Keep cluster anchors well inside the walls so the sampled rings fit.
    x_lo = -0.5 * cfg.arena_width + 1.75
    x_hi = 0.5 * cfg.arena_width - 1.75
    y_lo = -0.5 * cfg.arena_length + 1.75
    y_hi = 0.5 * cfg.arena_length - 1.75
    for _ in range(2_000):
        candidate = (float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi)))
        if all(math.hypot(candidate[0] - x, candidate[1] - y) >= 3.0 for x, y in centers):
            centers.append(candidate)
            if len(centers) == num_clusters:
                return centers
    return centers


def place_clustered_obstacles(
    count: int,
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    target: Difficulty,
) -> list[Obstacle] | None:
    if count == 0:
        return []

    if count <= 4:
        num_clusters = 1
    elif count <= 8:
        num_clusters = int(rng.integers(1, 3))
    else:
        num_clusters = int(rng.integers(2, 4))

    cluster_centers = sample_cluster_centers(num_clusters, rng, cfg)
    if len(cluster_centers) != num_clusters:
        return None

    if target == "hard":
        spread_lo, spread_hi = cfg.cluster_spread_min, 1.25
        gap_lo, gap_hi = cfg.clustered_gap_min, 0.14
    elif target == "medium":
        spread_lo, spread_hi = 0.90, cfg.cluster_spread_max
        gap_lo, gap_hi = 0.10, cfg.clustered_gap_max
    else:
        spread_lo, spread_hi = 1.20, 1.90
        gap_lo, gap_hi = 0.20, 0.40

    minimum_gap = float(rng.uniform(gap_lo, max(gap_lo, gap_hi)))
    placed: list[Obstacle] = []
    assignments = np.arange(count) % num_clusters
    rng.shuffle(assignments)

    for cluster_id in assignments:
        anchor_x, anchor_y = cluster_centers[int(cluster_id)]
        accepted = False
        for _ in range(cfg.obstacle_sampling_attempts):
            radius = float(rng.uniform(cfg.obstacle_radius_min, cfg.obstacle_radius_max))
            radial_distance = float(rng.uniform(spread_lo, spread_hi))
            angle = float(rng.uniform(-math.pi, math.pi))
            candidate = Obstacle(
                x=anchor_x + radial_distance * math.cos(angle),
                y=anchor_y + radial_distance * math.sin(angle),
                radius=radius,
                height=cfg.obstacle_height,
            )
            if candidate_is_valid(candidate, placed, minimum_gap, cfg):
                placed.append(candidate)
                accepted = True
                break
        if not accepted:
            return None
    return placed


def place_mixed_obstacles(
    count: int,
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    target: Difficulty,
) -> list[Obstacle] | None:
    if count == 0:
        return []
    if count == 1:
        return place_separated_obstacles(count, rng, cfg, target)

    cluster_count = max(2, int(round(count * float(rng.uniform(0.45, 0.70)))))
    cluster_count = min(cluster_count, count)
    clustered = place_clustered_obstacles(cluster_count, rng, cfg, target)
    if clustered is None:
        return None

    remaining = count - cluster_count
    if remaining == 0:
        return clustered

    minimum_gap = 0.35 if target == "hard" else 0.55
    placed = list(clustered)
    for _ in range(cfg.obstacle_sampling_attempts):
        candidate = sample_uniform_obstacle(rng, cfg)
        if candidate_is_valid(candidate, placed, minimum_gap, cfg):
            placed.append(candidate)
            if len(placed) == count:
                return placed
    return None


def generate_obstacles(
    count: int,
    family: LayoutFamily,
    target: Difficulty,
    rng: np.random.Generator,
    cfg: GeneratorConfig,
) -> list[Obstacle] | None:
    if family == "empty":
        return [] if count == 0 else None
    if family == "separated":
        return place_separated_obstacles(count, rng, cfg, target)
    if family == "clustered":
        return place_clustered_obstacles(count, rng, cfg, target)
    if family == "mixed":
        return place_mixed_obstacles(count, rng, cfg, target)
    raise ValueError(f"Unsupported layout family: {family}")


def build_grid(obstacles: Sequence[Obstacle], cfg: GeneratorConfig) -> GridData:
    resolution = cfg.grid_resolution
    nx = int(math.floor(cfg.arena_width / resolution))
    ny = int(math.floor(cfg.arena_length / resolution))
    x_min = -0.5 * cfg.arena_width
    y_min = -0.5 * cfg.arena_length
    xs = x_min + (np.arange(nx, dtype=np.float64) + 0.5) * resolution
    ys = y_min + (np.arange(ny, dtype=np.float64) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys)

    planning_margin = cfg.robot_radius + cfg.wall_clearance
    occupancy = (
        (xx < x_min + planning_margin)
        | (xx > -x_min - planning_margin)
        | (yy < y_min + planning_margin)
        | (yy > -y_min - planning_margin)
    )

    obstacle_inflation = cfg.robot_radius + cfg.obstacle_clearance
    for obstacle in obstacles:
        inflated_radius = obstacle.radius + obstacle_inflation
        occupancy |= (xx - obstacle.x) ** 2 + (yy - obstacle.y) ** 2 <= inflated_radius**2

    labels, largest_label, largest_cells = connected_components(occupancy)
    free_count = int((~occupancy).sum())
    largest_count = int(largest_cells.shape[0])
    free_fraction = free_count / float(occupancy.size)
    largest_fraction = largest_count / float(max(free_count, 1))

    if largest_count:
        rows = largest_cells[:, 0]
        cols = largest_cells[:, 1]
        span_x = float(xs[cols].max() - xs[cols].min())
        span_y = float(ys[rows].max() - ys[rows].min())
    else:
        span_x = span_y = 0.0

    return GridData(
        occupancy=occupancy,
        xs=xs,
        ys=ys,
        component_labels=labels,
        largest_component_label=largest_label,
        largest_component_cells=largest_cells,
        free_fraction=free_fraction,
        largest_component_fraction=largest_fraction,
        span_x=span_x,
        span_y=span_y,
    )


def connected_components(occupancy: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    height, width = occupancy.shape
    labels = np.full((height, width), -1, dtype=np.int32)
    label = 0
    largest_label = -1
    largest_cells: list[tuple[int, int]] = []
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for row in range(height):
        for col in range(width):
            if occupancy[row, col] or labels[row, col] >= 0:
                continue
            queue: deque[tuple[int, int]] = deque([(row, col)])
            labels[row, col] = label
            cells: list[tuple[int, int]] = []
            while queue:
                current_row, current_col = queue.popleft()
                cells.append((current_row, current_col))
                for dr, dc in neighbors:
                    nr, nc = current_row + dr, current_col + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if occupancy[nr, nc] or labels[nr, nc] >= 0:
                        continue
                    labels[nr, nc] = label
                    queue.append((nr, nc))
            if len(cells) > len(largest_cells):
                largest_cells = cells
                largest_label = label
            label += 1

    return labels, largest_label, np.asarray(largest_cells, dtype=np.int32)


def endpoint_candidate_cells(grid: GridData, cfg: GeneratorConfig) -> np.ndarray:
    cells = grid.largest_component_cells
    if not cells.size:
        return cells
    rows = cells[:, 0]
    cols = cells[:, 1]
    x = grid.xs[cols]
    y = grid.ys[rows]
    half_w = 0.5 * cfg.arena_width
    half_l = 0.5 * cfg.arena_length
    valid = (
        (x >= -half_w + cfg.sample_margin_from_walls)
        & (x <= half_w - cfg.sample_margin_from_walls)
        & (y >= -half_l + cfg.sample_margin_from_walls)
        & (y <= half_l - cfg.sample_margin_from_walls)
    )
    return cells[valid]


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def astar_path(
    occupancy: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    resolution: float,
) -> tuple[list[tuple[int, int]], float] | None:
    if start == goal:
        return [start], 0.0

    moves = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    height, width = occupancy.shape
    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, current_g * resolution
        closed.add(current)

        row, col = current
        for dr, dc, move_cost in moves:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if occupancy[nr, nc]:
                continue
            # Prevent diagonal corner cutting between two occupied cardinal cells.
            if dr != 0 and dc != 0:
                if occupancy[row + dr, col] or occupancy[row, col + dc]:
                    continue
            neighbor = (nr, nc)
            tentative = current_g + move_cost
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            priority = tentative + heuristic(neighbor, goal)
            heapq.heappush(open_heap, (priority, tentative, neighbor))
    return None



def dijkstra_from_start(
    occupancy: np.ndarray,
    start: tuple[int, int],
    resolution: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute shortest-path distances and predecessors from one start cell."""
    moves = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    height, width = occupancy.shape
    distances = np.full((height, width), np.inf, dtype=np.float32)
    parent_row = np.full((height, width), -1, dtype=np.int16)
    parent_col = np.full((height, width), -1, dtype=np.int16)
    distances[start] = 0.0
    heap: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]

    while heap:
        current_distance, row, col = heapq.heappop(heap)
        if current_distance > float(distances[row, col]) + 1e-7:
            continue
        for dr, dc, move_cost in moves:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if occupancy[nr, nc]:
                continue
            if dr != 0 and dc != 0:
                if occupancy[row + dr, col] or occupancy[row, col + dc]:
                    continue
            candidate = current_distance + move_cost * resolution
            if candidate + 1e-7 >= float(distances[nr, nc]):
                continue
            distances[nr, nc] = candidate
            parent_row[nr, nc] = row
            parent_col[nr, nc] = col
            heapq.heappush(heap, (candidate, nr, nc))

    return distances, parent_row, parent_col


def reconstruct_dijkstra_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    parent_row: np.ndarray,
    parent_col: np.ndarray,
) -> list[tuple[int, int]] | None:
    if start == goal:
        return [start]
    path = [goal]
    current = goal
    maximum_steps = parent_row.size + 1
    for _ in range(maximum_steps):
        row, col = current
        previous_row = int(parent_row[row, col])
        previous_col = int(parent_col[row, col])
        if previous_row < 0 or previous_col < 0:
            return None
        current = (previous_row, previous_col)
        path.append(current)
        if current == start:
            path.reverse()
            return path
    return None

def segment_point_distance(
    start: tuple[float, float],
    goal: tuple[float, float],
    point: tuple[float, float],
) -> tuple[float, float]:
    sx, sy = start
    gx, gy = goal
    px, py = point
    vx, vy = gx - sx, gy - sy
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return math.hypot(px - sx, py - sy), 0.0
    t = ((px - sx) * vx + (py - sy) * vy) / denom
    t_clamped = min(1.0, max(0.0, t))
    qx, qy = sx + t_clamped * vx, sy + t_clamped * vy
    return math.hypot(px - qx, py - qy), t_clamped


def direct_path_is_blocked(
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: Sequence[Obstacle],
    cfg: GeneratorConfig,
) -> bool:
    inflation = cfg.robot_radius + cfg.obstacle_clearance + cfg.blocked_path_margin
    for obstacle in obstacles:
        distance, t = segment_point_distance(start, goal, (obstacle.x, obstacle.y))
        if 0.05 <= t <= 0.95 and distance <= obstacle.radius + inflation:
            return True
    return False


def path_minimum_clearance(
    path: Sequence[tuple[int, int]],
    grid: GridData,
    obstacles: Sequence[Obstacle],
    cfg: GeneratorConfig,
) -> float:
    if not path:
        return 0.0
    inflation = cfg.robot_radius + cfg.obstacle_clearance
    wall_margin = cfg.robot_radius + cfg.wall_clearance
    half_w = 0.5 * cfg.arena_width
    half_l = 0.5 * cfg.arena_length
    minimum = math.inf
    # Subsample long paths to reduce computation while preserving 0.2 m spacing.
    stride = max(1, int(round(0.20 / cfg.grid_resolution)))
    for row, col in path[::stride]:
        x = float(grid.xs[col])
        y = float(grid.ys[row])
        wall_extra = min(
            x - (-half_w + wall_margin),
            (half_w - wall_margin) - x,
            y - (-half_l + wall_margin),
            (half_l - wall_margin) - y,
        )
        minimum = min(minimum, wall_extra)
        for obstacle in obstacles:
            extra = math.hypot(x - obstacle.x, y - obstacle.y) - (obstacle.radius + inflation)
            minimum = min(minimum, extra)
    return max(0.0, float(minimum))


def count_path_turns(path: Sequence[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    # Compress identical direction segments and count meaningful direction changes.
    directions: list[tuple[int, int]] = []
    for a, b in zip(path[:-1], path[1:]):
        direction = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
        if not directions or direction != directions[-1]:
            directions.append(direction)
    return max(0, len(directions) - 1)


def scenario_raw_score(
    obstacle_count: int,
    detour_ratio: float,
    minimum_clearance: float,
    path_turn_count: int,
    direct_path_blocked: bool,
    path_length: float,
    cfg: GeneratorConfig,
) -> float:
    count_term = obstacle_count / max(cfg.max_obstacles, 1)
    detour_term = np.clip((detour_ratio - 1.0) / 1.0, 0.0, 1.0)
    clearance_term = 1.0 - np.clip(minimum_clearance / 0.80, 0.0, 1.0)
    turns_term = np.clip(path_turn_count / 10.0, 0.0, 1.0)
    blocked_term = 1.0 if direct_path_blocked else 0.0
    length_term = np.clip(path_length / max(cfg.maximum_path_length, 1e-6), 0.0, 1.0)
    return float(
        0.18 * count_term
        + 0.27 * detour_term
        + 0.28 * clearance_term
        + 0.10 * turns_term
        + 0.10 * blocked_term
        + 0.07 * length_term
    )


def generate_scenarios(
    map_id: int,
    obstacles: Sequence[Obstacle],
    grid: GridData,
    rng: np.random.Generator,
    cfg: GeneratorConfig,
) -> list[Scenario] | None:
    candidates = endpoint_candidate_cells(grid, cfg)
    if candidates.shape[0] < 2:
        return None

    scenarios: list[Scenario] = []
    seen_pairs: set[tuple[int, int, int, int]] = set()
    target_blocked = (
        int(round(cfg.scenarios_per_map * cfg.target_blocked_fraction))
        if obstacles
        else 0
    )
    blocked_count = 0

    # Reuse one full-grid shortest-path solve for several goals.  This reduces
    # the default 64-scenario map from roughly 64 A* searches to about 8--16
    # Dijkstra searches, which is substantially faster for a 150 x 150 grid.
    goals_per_start = 8
    max_start_solves = max(12, int(math.ceil(cfg.scenarios_per_map / goals_per_start)) * 3)

    for _ in range(max_start_solves):
        if len(scenarios) >= cfg.scenarios_per_map:
            break
        start_index = int(rng.integers(0, candidates.shape[0]))
        start_cell = tuple(int(v) for v in candidates[start_index])
        start = (float(grid.xs[start_cell[1]]), float(grid.ys[start_cell[0]]))
        distances, parent_row, parent_col = dijkstra_from_start(
            grid.occupancy, start_cell, cfg.grid_resolution
        )

        goal_order = rng.permutation(candidates.shape[0])
        accepted_for_start = 0
        for goal_index in goal_order:
            if len(scenarios) >= cfg.scenarios_per_map or accepted_for_start >= goals_per_start:
                break
            goal_cell = tuple(int(v) for v in candidates[int(goal_index)])
            if goal_cell == start_cell:
                continue
            pair_key = (*start_cell, *goal_cell)
            reverse_key = (*goal_cell, *start_cell)
            if pair_key in seen_pairs or reverse_key in seen_pairs:
                continue

            path_length = float(distances[goal_cell])
            if not math.isfinite(path_length) or path_length > cfg.maximum_path_length:
                continue
            goal = (float(grid.xs[goal_cell[1]]), float(grid.ys[goal_cell[0]]))
            direct_distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
            if direct_distance < cfg.minimum_start_goal_distance:
                continue

            blocked = direct_path_is_blocked(start, goal, obstacles, cfg)
            remaining = cfg.scenarios_per_map - len(scenarios)
            blocked_needed = max(0, target_blocked - blocked_count)
            if blocked_needed >= remaining and not blocked:
                continue
            if blocked_count >= target_blocked and blocked and rng.random() < 0.90:
                continue

            path = reconstruct_dijkstra_path(start_cell, goal_cell, parent_row, parent_col)
            if path is None:
                continue
            detour_ratio = path_length / max(direct_distance, 1e-9)
            minimum_clearance = path_minimum_clearance(path, grid, obstacles, cfg)
            path_turn_count = count_path_turns(path)
            yaw = float(rng.uniform(-math.pi, math.pi))
            score = scenario_raw_score(
                len(obstacles),
                detour_ratio,
                minimum_clearance,
                path_turn_count,
                blocked,
                path_length,
                cfg,
            )

            scenarios.append(
                Scenario(
                    scenario_id=-1,
                    map_id=map_id,
                    start_x=start[0],
                    start_y=start[1],
                    goal_x=goal[0],
                    goal_y=goal[1],
                    initial_yaw_rad=yaw,
                    direct_distance=direct_distance,
                    path_length=path_length,
                    detour_ratio=detour_ratio,
                    minimum_path_clearance=minimum_clearance,
                    direct_path_blocked=blocked,
                    path_turn_count=path_turn_count,
                    raw_difficulty_score=score,
                )
            )
            seen_pairs.add(pair_key)
            blocked_count += int(blocked)
            accepted_for_start += 1

    if len(scenarios) != cfg.scenarios_per_map:
        return None
    return scenarios


def occupancy_hash(occupancy: np.ndarray) -> str:
    packed = np.packbits(occupancy.astype(np.uint8), axis=None)
    return hashlib.sha256(packed.tobytes()).hexdigest()


def map_raw_score(generated_map: GeneratedMap, cfg: GeneratorConfig) -> float:
    count_term = len(generated_map.obstacles) / max(cfg.max_obstacles, 1)
    detour_term = np.clip((generated_map.mean_scenario_detour_ratio - 1.0) / 1.0, 0.0, 1.0)
    clearance_term = 1.0 - np.clip(generated_map.mean_scenario_minimum_clearance / 0.80, 0.0, 1.0)
    blocked_term = generated_map.blocked_scenario_fraction
    turns_term = np.clip(generated_map.mean_scenario_turn_count / 10.0, 0.0, 1.0)
    occupied_term = 1.0 - generated_map.free_fraction
    return float(
        0.22 * count_term
        + 0.24 * detour_term
        + 0.25 * clearance_term
        + 0.12 * blocked_term
        + 0.09 * turns_term
        + 0.08 * occupied_term
    )


def percentile_labels(
    values: Sequence[float],
    easy_fraction: float,
    medium_fraction: float,
) -> tuple[list[Difficulty], list[float]]:
    if not values:
        return [], []
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
    labels: list[Difficulty] = ["medium"] * len(values)
    normalized: list[float] = [0.0] * len(values)
    easy_count = int(round(len(values) * easy_fraction))
    medium_count = int(round(len(values) * medium_fraction))
    hard_start = min(len(values), easy_count + medium_count)

    for rank, index in enumerate(order):
        if rank < easy_count:
            labels[int(index)] = "easy"
        elif rank < hard_start:
            labels[int(index)] = "medium"
        else:
            labels[int(index)] = "hard"
        normalized[int(index)] = rank / max(len(values) - 1, 1)
    return labels, normalized


def pad_obstacle_slots(obstacles: Sequence[Obstacle], max_obstacles: int) -> list[dict[str, object]]:
    slots: list[dict[str, object]] = []
    for index in range(max_obstacles):
        if index < len(obstacles):
            obstacle = obstacles[index]
            slots.append(
                {
                    "slot": index,
                    "active": True,
                    "x": obstacle.x,
                    "y": obstacle.y,
                    "radius": obstacle.radius,
                    "height": obstacle.height,
                }
            )
        else:
            slots.append(
                {
                    "slot": index,
                    "active": False,
                    "x": 50.0 + index,
                    "y": 50.0,
                    "radius": 0.0,
                    "height": 0.0,
                }
            )
    return slots


def arena_json(generated_map: GeneratedMap, cfg: GeneratorConfig) -> dict[str, object]:
    return {
        "width": cfg.arena_width,
        "length": cfg.arena_length,
        "wall_thickness": cfg.wall_thickness,
        "wall_height": cfg.wall_height,
        "robot_radius": cfg.robot_radius,
        "wall_clearance": cfg.wall_clearance,
        "obstacle_clearance": cfg.obstacle_clearance,
        "max_obstacles": cfg.max_obstacles,
        "obstacles": [asdict(obstacle) for obstacle in generated_map.obstacles],
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_map_files(
    output_dir: Path,
    generated_map: GeneratedMap,
    scenarios: Sequence[Scenario],
    grid: GridData,
    cfg: GeneratorConfig,
) -> None:
    map_name = f"map_{generated_map.map_id:03d}"
    map_payload = {
        "schema_version": 1,
        "map_id": generated_map.map_id,
        "generation_seed": generated_map.generation_seed,
        "generation_target": generated_map.generation_target,
        "difficulty": generated_map.difficulty,
        "difficulty_score": generated_map.difficulty_score,
        "raw_difficulty_score": generated_map.raw_difficulty_score,
        "layout_family": generated_map.layout_family,
        "arena": {
            "width": cfg.arena_width,
            "length": cfg.arena_length,
            "wall_thickness": cfg.wall_thickness,
            "wall_height": cfg.wall_height,
        },
        "planning": {
            "grid_resolution": cfg.grid_resolution,
            "robot_radius": cfg.robot_radius,
            "wall_clearance": cfg.wall_clearance,
            "obstacle_clearance": cfg.obstacle_clearance,
            "minimum_start_goal_distance": cfg.minimum_start_goal_distance,
            "maximum_path_length": cfg.maximum_path_length,
        },
        "obstacle_count": len(generated_map.obstacles),
        "obstacles": [asdict(obstacle) for obstacle in generated_map.obstacles],
        "obstacle_slots": pad_obstacle_slots(generated_map.obstacles, cfg.max_obstacles),
        "occupancy_hash": generated_map.occupancy_hash,
        "map_metrics": {
            "free_fraction": generated_map.free_fraction,
            "largest_component_fraction": generated_map.largest_component_fraction,
            "largest_component_span_x": generated_map.largest_component_span_x,
            "largest_component_span_y": generated_map.largest_component_span_y,
            "mean_scenario_detour_ratio": generated_map.mean_scenario_detour_ratio,
            "mean_scenario_minimum_clearance": generated_map.mean_scenario_minimum_clearance,
            "blocked_scenario_fraction": generated_map.blocked_scenario_fraction,
            "mean_scenario_turn_count": generated_map.mean_scenario_turn_count,
        },
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
    }
    write_json(output_dir / "maps" / f"{map_name}.json", map_payload)
    write_json(output_dir / "arenas" / f"{map_name}_arena.json", arena_json(generated_map, cfg))
    np.savez_compressed(
        output_dir / "occupancy" / f"{map_name}.npz",
        occupancy=grid.occupancy,
        xs=grid.xs,
        ys=grid.ys,
        component_labels=grid.component_labels,
        largest_component_label=np.asarray(grid.largest_component_label, dtype=np.int32),
    )


def map_summary_row(generated_map: GeneratedMap) -> dict[str, object]:
    return {
        "map_id": generated_map.map_id,
        "difficulty": generated_map.difficulty,
        "difficulty_score": generated_map.difficulty_score,
        "raw_difficulty_score": generated_map.raw_difficulty_score,
        "generation_target": generated_map.generation_target,
        "layout_family": generated_map.layout_family,
        "obstacle_count": len(generated_map.obstacles),
        "free_fraction": generated_map.free_fraction,
        "largest_component_fraction": generated_map.largest_component_fraction,
        "largest_component_span_x": generated_map.largest_component_span_x,
        "largest_component_span_y": generated_map.largest_component_span_y,
        "mean_scenario_detour_ratio": generated_map.mean_scenario_detour_ratio,
        "mean_scenario_minimum_clearance": generated_map.mean_scenario_minimum_clearance,
        "blocked_scenario_fraction": generated_map.blocked_scenario_fraction,
        "mean_scenario_turn_count": generated_map.mean_scenario_turn_count,
        "occupancy_hash": generated_map.occupancy_hash,
        "generation_seed": generated_map.generation_seed,
    }


def scenario_row(scenario: Scenario) -> dict[str, object]:
    return asdict(scenario)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def preview_map(
    path: Path,
    generated_map: GeneratedMap,
    scenarios: Sequence[Scenario],
    cfg: GeneratorConfig,
    max_scenarios: int = 8,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    half_w = 0.5 * cfg.arena_width
    half_l = 0.5 * cfg.arena_length
    ax.add_patch(
        Rectangle(
            (-half_w, -half_l),
            cfg.arena_width,
            cfg.arena_length,
            fill=False,
            linewidth=2.0,
        )
    )
    for obstacle in generated_map.obstacles:
        ax.add_patch(Circle((obstacle.x, obstacle.y), obstacle.radius, alpha=0.65))
        inflated = obstacle.radius + cfg.robot_radius + cfg.obstacle_clearance
        ax.add_patch(Circle((obstacle.x, obstacle.y), inflated, fill=False, linestyle="--", alpha=0.30))

    for scenario in scenarios[:max_scenarios]:
        ax.plot([scenario.start_x, scenario.goal_x], [scenario.start_y, scenario.goal_y], alpha=0.22)
        ax.scatter([scenario.start_x], [scenario.start_y], marker="o", s=18)
        ax.scatter([scenario.goal_x], [scenario.goal_y], marker="*", s=35)

    ax.set_xlim(-half_w - 0.25, half_w + 0.25)
    ax.set_ylim(-half_l - 0.25, half_l + 0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Map {generated_map.map_id:03d} | {generated_map.difficulty} | "
        f"{generated_map.layout_family} | N={len(generated_map.obstacles)}"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def preview_mosaic(
    path: Path,
    maps: Sequence[GeneratedMap],
    cfg: GeneratorConfig,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    columns = 10
    rows = int(math.ceil(len(maps) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(20, 2.05 * rows), squeeze=False)
    half_w = 0.5 * cfg.arena_width
    half_l = 0.5 * cfg.arena_length
    for axis in axes.flat:
        axis.set_axis_off()
    for axis, generated_map in zip(axes.flat, maps):
        axis.set_axis_on()
        axis.add_patch(
            Rectangle(
                (-half_w, -half_l),
                cfg.arena_width,
                cfg.arena_length,
                fill=False,
                linewidth=0.7,
            )
        )
        for obstacle in generated_map.obstacles:
            axis.add_patch(Circle((obstacle.x, obstacle.y), obstacle.radius, alpha=0.75))
        axis.set_xlim(-half_w, half_w)
        axis.set_ylim(-half_l, half_l)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            f"{generated_map.map_id:03d} {generated_map.difficulty[0].upper()} "
            f"{generated_map.layout_family[0].upper()} N={len(generated_map.obstacles)}",
            fontsize=7,
        )
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_bank(cfg: GeneratorConfig) -> tuple[list[GeneratedMap], list[Scenario], dict[int, GridData]]:
    rng = np.random.default_rng(cfg.seed)
    targets = target_sequence(cfg.num_maps, cfg)
    target_seen: dict[Difficulty, int] = {"easy": 0, "medium": 0, "hard": 0}
    generated_maps: list[GeneratedMap] = []
    all_scenarios: list[Scenario] = []
    grids: dict[int, GridData] = {}
    seen_hashes: set[str] = set()

    for map_id, target in enumerate(targets):
        level_index = target_seen[target]
        target_seen[target] += 1
        accepted = False
        for map_attempt in range(cfg.map_sampling_attempts):
            # First attempt preserves the guaranteed 0/12 edge cases; later
            # attempts may resample the count to avoid an impossible geometry.
            count_index = level_index if map_attempt == 0 else level_index + map_attempt
            obstacle_count = obstacle_count_for_target(target, rng, cfg, count_index)
            family = family_for_target(target, obstacle_count, rng)
            obstacles = generate_obstacles(obstacle_count, family, target, rng, cfg)
            if obstacles is None:
                continue

            grid = build_grid(obstacles, cfg)
            if grid.largest_component_fraction < cfg.minimum_largest_component_fraction:
                continue
            if max(grid.span_x, grid.span_y) < cfg.minimum_component_span:
                continue

            digest = occupancy_hash(grid.occupancy)
            if digest in seen_hashes:
                continue

            map_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            scenario_rng = np.random.default_rng(map_seed)
            scenarios = generate_scenarios(map_id, obstacles, grid, scenario_rng, cfg)
            if scenarios is None:
                continue

            detours = [scenario.detour_ratio for scenario in scenarios]
            clearances = [scenario.minimum_path_clearance for scenario in scenarios]
            blocked_fraction = float(np.mean([scenario.direct_path_blocked for scenario in scenarios]))
            turns = [scenario.path_turn_count for scenario in scenarios]
            generated_map = GeneratedMap(
                map_id=map_id,
                generation_seed=map_seed,
                generation_target=target,
                layout_family=family,
                obstacles=list(obstacles),
                occupancy_hash=digest,
                free_fraction=grid.free_fraction,
                largest_component_fraction=grid.largest_component_fraction,
                largest_component_span_x=grid.span_x,
                largest_component_span_y=grid.span_y,
                mean_scenario_detour_ratio=float(np.mean(detours)),
                mean_scenario_minimum_clearance=float(np.mean(clearances)),
                blocked_scenario_fraction=blocked_fraction,
                mean_scenario_turn_count=float(np.mean(turns)),
            )
            generated_map.raw_difficulty_score = map_raw_score(generated_map, cfg)
            generated_maps.append(generated_map)
            all_scenarios.extend(scenarios)
            grids[map_id] = grid
            seen_hashes.add(digest)
            accepted = True
            print(
                f"[MAP {map_id + 1:03d}/{cfg.num_maps:03d}] "
                f"target={target:<6} family={family:<9} obstacles={obstacle_count:2d} "
                f"blocked={blocked_fraction:5.1%} attempts={map_attempt + 1}"
            )
            break

        if not accepted:
            raise RuntimeError(
                f"Could not generate map {map_id} ({target}) after "
                f"{cfg.map_sampling_attempts} attempts. Consider relaxing placement or path constraints."
            )

    # Assign map-level labels from measured difficulty scores.
    map_labels, map_normalized = percentile_labels(
        [generated_map.raw_difficulty_score for generated_map in generated_maps],
        cfg.easy_fraction,
        cfg.medium_fraction,
    )
    for generated_map, label, normalized in zip(generated_maps, map_labels, map_normalized):
        generated_map.difficulty = label
        generated_map.difficulty_score = normalized

    # Assign scenario-level labels separately; this captures easy and hard
    # start-goal pairs within the same obstacle map.
    scenario_labels, scenario_normalized = percentile_labels(
        [scenario.raw_difficulty_score for scenario in all_scenarios],
        cfg.easy_fraction,
        cfg.medium_fraction,
    )
    for scenario_id, (scenario, label, normalized) in enumerate(
        zip(all_scenarios, scenario_labels, scenario_normalized)
    ):
        scenario.scenario_id = scenario_id
        scenario.difficulty = label
        scenario.difficulty_score = normalized

    return generated_maps, all_scenarios, grids


def save_bank(
    output_dir: Path,
    generated_maps: Sequence[GeneratedMap],
    scenarios: Sequence[Scenario],
    grids: dict[int, GridData],
    cfg: GeneratorConfig,
    save_previews: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("maps", "arenas", "occupancy"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    if save_previews:
        (output_dir / "previews").mkdir(parents=True, exist_ok=True)

    scenarios_by_map: dict[int, list[Scenario]] = {map_id: [] for map_id in range(len(generated_maps))}
    for scenario in scenarios:
        scenarios_by_map[scenario.map_id].append(scenario)

    for generated_map in generated_maps:
        map_scenarios = scenarios_by_map[generated_map.map_id]
        write_map_files(output_dir, generated_map, map_scenarios, grids[generated_map.map_id], cfg)
        if save_previews:
            preview_map(
                output_dir / "previews" / f"map_{generated_map.map_id:03d}.png",
                generated_map,
                map_scenarios,
                cfg,
            )

    write_json(output_dir / "generation_config.json", asdict(cfg))
    write_csv(output_dir / "map_summary.csv", [map_summary_row(item) for item in generated_maps])
    write_csv(output_dir / "scenarios.csv", [scenario_row(item) for item in scenarios])

    bank_payload = {
        "schema_version": 1,
        "generator": Path(__file__).name,
        "num_maps": len(generated_maps),
        "scenarios_per_map": cfg.scenarios_per_map,
        "num_scenarios": len(scenarios),
        "seed": cfg.seed,
        "max_obstacles": cfg.max_obstacles,
        "difficulty_counts": {
            level: sum(item.difficulty == level for item in generated_maps)
            for level in ("easy", "medium", "hard")
        },
        "scenario_difficulty_counts": {
            level: sum(item.difficulty == level for item in scenarios)
            for level in ("easy", "medium", "hard")
        },
        "maps": [
            {
                "map_id": item.map_id,
                "file": f"maps/map_{item.map_id:03d}.json",
                "arena_file": f"arenas/map_{item.map_id:03d}_arena.json",
                "occupancy_file": f"occupancy/map_{item.map_id:03d}.npz",
                "difficulty": item.difficulty,
                "difficulty_score": item.difficulty_score,
                "layout_family": item.layout_family,
                "obstacle_count": len(item.obstacles),
                "scenario_id_start": scenarios_by_map[item.map_id][0].scenario_id,
                "scenario_id_end": scenarios_by_map[item.map_id][-1].scenario_id,
            }
            for item in generated_maps
        ],
    }
    write_json(output_dir / "bank.json", bank_payload)

    if save_previews:
        preview_mosaic(output_dir / "previews" / "all_maps_mosaic.png", generated_maps, cfg)


def print_summary(
    output_dir: Path,
    generated_maps: Sequence[GeneratedMap],
    scenarios: Sequence[Scenario],
) -> None:
    print("\n[COMPLETE]")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Maps:             {len(generated_maps)}")
    print(f"Scenarios:        {len(scenarios)}")
    print(
        "Obstacle range:   "
        f"{min(len(item.obstacles) for item in generated_maps)}--"
        f"{max(len(item.obstacles) for item in generated_maps)}"
    )
    print(
        "Map levels:       "
        + ", ".join(
            f"{level}={sum(item.difficulty == level for item in generated_maps)}"
            for level in ("easy", "medium", "hard")
        )
    )
    print(
        "Layouts:          "
        + ", ".join(
            f"{family}={sum(item.layout_family == family for item in generated_maps)}"
            for family in ("empty", "separated", "clustered", "mixed")
        )
    )
    print("Main files:")
    print(f"  {output_dir / 'bank.json'}")
    print(f"  {output_dir / 'map_summary.csv'}")
    print(f"  {output_dir / 'scenarios.csv'}")


def main() -> None:
    args = parse_args()
    cfg = GeneratorConfig(
        num_maps=args.num_maps,
        scenarios_per_map=args.scenarios_per_map,
        seed=args.seed,
        min_obstacles=args.min_obstacles,
        max_obstacles=args.max_obstacles,
        obstacle_radius_min=args.radius_min,
        obstacle_radius_max=args.radius_max,
        grid_resolution=args.grid_resolution,
        minimum_start_goal_distance=args.minimum_start_goal_distance,
        maximum_path_length=args.maximum_path_length,
    )
    validate_config(cfg)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite or choose another directory."
        )

    generated_maps, scenarios, grids = generate_bank(cfg)
    save_bank(output_dir, generated_maps, scenarios, grids, cfg, save_previews=not args.no_previews)
    print_summary(output_dir, generated_maps, scenarios)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
