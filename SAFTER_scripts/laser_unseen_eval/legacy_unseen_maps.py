#!/usr/bin/env python3
"""Retired unseen-map variants for the laser structured evaluator.

The structured evaluator's default set is the three maps "low_shift",
"medium_shift" and "high_shift", loaded from "Maps/Evaluation Maps/". This
module keeps the other static, out-of-distribution and dynamic-obstacle map
variants. It is self-contained (it duplicates the small DynamicObstacleSpec
helper instead of importing the heavier evaluator) and is not imported by
evaluate_laser_unseen_maps_structured.py. Import LEGACY_UNSEEN_MAPS /
LEGACY_STRUCTURED_SCENARIO_REGIONS (and LEGACY_DYNAMIC_OBSTACLE_SPECS /
LEGACY_DYNAMIC_OBSTACLE_COUNTS) from here to use these maps.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

# Make the repository root importable when run standalone.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SAFTER_src.high_level_rl.config import CircularObstacle


LEGACY_UNSEEN_MAPS: dict[str, tuple[CircularObstacle, ...]] = {
    "shifted": (
        CircularObstacle(-4.2, -1.0, 0.55),
        CircularObstacle(-2.0,  3.1, 0.45),
        CircularObstacle( 0.4, -2.8, 0.60),
        CircularObstacle( 2.8,  3.7, 0.50),
        CircularObstacle( 4.4,  0.2, 0.40),
        CircularObstacle(-5.4,  5.2, 0.50),
        CircularObstacle(-5.2, -4.8, 0.45),
        CircularObstacle( 5.3,  4.7, 0.55),
        CircularObstacle( 4.8, -5.1, 0.50),
    ),
    # Two-row pillar layout (as in medium_shift) with mixed shapes per row:
    # circle(0.50), square(0.65), circle(0.75), square(0.65), circle(0.50);
    # every in-row gap is 0.9 m.
    "corridor_derived_highshift": (
        CircularObstacle(-4.35, -2.2, 0.50, shape="circle"),
        CircularObstacle(-2.30, -2.2, 0.65, shape="square"),
        CircularObstacle( 0.00, -2.2, 0.75, shape="circle"),
        CircularObstacle( 2.30, -2.2, 0.65, shape="square"),
        CircularObstacle( 4.35, -2.2, 0.50, shape="circle"),
        CircularObstacle(-4.35,  2.2, 0.50, shape="circle"),
        CircularObstacle(-2.30,  2.2, 0.65, shape="square"),
        CircularObstacle( 0.00,  2.2, 0.75, shape="circle"),
        CircularObstacle( 2.30,  2.2, 0.65, shape="square"),
        CircularObstacle( 4.35,  2.2, 0.50, shape="circle"),
    ),
    # Variant: four top-row obstacles with the two inner ones square; the
    # bottom row has a single large center square (half-side 0.90).
    "corridor_derived_highshift_v2": (
        CircularObstacle(-4.00,  2.2, 0.65, shape="circle"),
        CircularObstacle(-2.00,  2.2, 0.65, shape="square"),
        CircularObstacle( 2.00,  2.2, 0.65, shape="square"),
        CircularObstacle( 4.00,  2.2, 0.65, shape="circle"),
        CircularObstacle(-4.50, -2.2, 0.50, shape="circle"),
        CircularObstacle(-2.45, -2.2, 0.65, shape="circle"),
        CircularObstacle( 0.00, -2.2, 0.90, shape="square"),
        CircularObstacle( 2.45, -2.2, 0.65, shape="circle"),
        CircularObstacle( 4.50, -2.2, 0.50, shape="circle"),
    ),
    # Variant: all top-row obstacles circular; the bottom-row center obstacle
    # is the only square.
    "corridor_derived_highshift_v4": (
        CircularObstacle(-4.00,  2.2, 0.65, shape="circle"),
        CircularObstacle(-2.00,  2.2, 0.65, shape="circle"),
        CircularObstacle( 2.00,  2.2, 0.65, shape="circle"),
        CircularObstacle( 4.00,  2.2, 0.65, shape="circle"),
        CircularObstacle(-4.50, -2.2, 0.50, shape="circle"),
        CircularObstacle(-2.45, -2.2, 0.65, shape="circle"),
        CircularObstacle( 0.00, -2.2, 0.90, shape="square"),
        CircularObstacle( 2.45, -2.2, 0.65, shape="circle"),
        CircularObstacle( 4.50, -2.2, 0.50, shape="circle"),
    ),
    # Inverted-V layout with varied radii (0.45-1.00 m): a large apex obstacle on
    # the direct start-goal line below the start region, two large obstacles
    # behind it, and two 2-obstacle legs toward the goal (min edge gap 0.968 m).
    "new_medium_v2": (
        CircularObstacle( 0.0,  2.4, 1.00),
        CircularObstacle(-1.6, -0.1, 1.00),
        CircularObstacle( 1.6, -0.1, 1.00),
        CircularObstacle(-3.7, -1.9, 0.45),
        CircularObstacle(-4.9, -3.4, 0.50),
        CircularObstacle( 3.7, -1.9, 0.45),
        CircularObstacle( 4.9, -3.4, 0.50),
    ),
    # Mirror-symmetric rows mixing circles and squares (radius = half-side for
    # squares), radii 0.35-0.70 m; the near-start gate is the main bottleneck.
    "circle_and_square": (
        CircularObstacle(-1.6,  3.4, 0.35, shape="circle"),
        CircularObstacle( 1.6,  3.4, 0.35, shape="circle"),
        CircularObstacle( 0.0,  1.9, 0.70, shape="square"),
        CircularObstacle(-1.5, -0.2, 0.35, shape="circle"),
        CircularObstacle( 1.5, -0.2, 0.35, shape="circle"),
        CircularObstacle(-4.6, -2.3, 0.65, shape="circle"),
        CircularObstacle(-1.0, -2.6, 0.40, shape="square"),
        CircularObstacle( 1.0, -2.6, 0.40, shape="square"),
        CircularObstacle( 4.6, -2.3, 0.65, shape="circle"),
        CircularObstacle(-3.6,  0.4, 0.70, shape="square"),
        CircularObstacle( 3.6,  0.4, 0.70, shape="square"),
    ),
    "clustered": (
        CircularObstacle(-2.6,  1.7, 0.55),
        CircularObstacle(-1.3,  2.7, 0.45),
        CircularObstacle(-0.2,  1.5, 0.60),
        CircularObstacle(-2.1,  3.8, 0.50),
        CircularObstacle( 0.1,  3.6, 0.40),
        CircularObstacle( 3.8, -3.6, 0.50),
        CircularObstacle( 5.0, -2.4, 0.45),
        CircularObstacle( 3.1, -1.9, 0.55),
        CircularObstacle( 5.2, -4.8, 0.50),
    ),
    # clustered_gap_Ncm: identical to "clustered" except obstacle 1 at
    # (-1.3, 2.7) is shrunk by N cm, widening its 0.640 m gap to obstacle 0
    # by exactly N cm.
    "clustered_gap1cm": (
        CircularObstacle(-2.6,  1.7, 0.55),
        CircularObstacle(-1.3,  2.7, 0.44),
        CircularObstacle(-0.2,  1.5, 0.60),
        CircularObstacle(-2.1,  3.8, 0.50),
        CircularObstacle( 0.1,  3.6, 0.40),
        CircularObstacle( 3.8, -3.6, 0.50),
        CircularObstacle( 5.0, -2.4, 0.45),
        CircularObstacle( 3.1, -1.9, 0.55),
        CircularObstacle( 5.2, -4.8, 0.50),
    ),
    "clustered_gap2cm": (
        CircularObstacle(-2.6,  1.7, 0.55),
        CircularObstacle(-1.3,  2.7, 0.43),
        CircularObstacle(-0.2,  1.5, 0.60),
        CircularObstacle(-2.1,  3.8, 0.50),
        CircularObstacle( 0.1,  3.6, 0.40),
        CircularObstacle( 3.8, -3.6, 0.50),
        CircularObstacle( 5.0, -2.4, 0.45),
        CircularObstacle( 3.1, -1.9, 0.55),
        CircularObstacle( 5.2, -4.8, 0.50),
    ),
    "clustered_gap3cm": (
        CircularObstacle(-2.6,  1.7, 0.55),
        CircularObstacle(-1.3,  2.7, 0.42),
        CircularObstacle(-0.2,  1.5, 0.60),
        CircularObstacle(-2.1,  3.8, 0.50),
        CircularObstacle( 0.1,  3.6, 0.40),
        CircularObstacle( 3.8, -3.6, 0.50),
        CircularObstacle( 5.0, -2.4, 0.45),
        CircularObstacle( 3.1, -1.9, 0.55),
        CircularObstacle( 5.2, -4.8, 0.50),
    ),
    # clustered_v2: both clusters of "clustered" spread out, with the
    # bottom-right cluster moved toward the arena center. The tightest gap is
    # the top cluster's 0.830 m (0.640 m in "clustered"); goal-region
    # clearance is at least 2.11 m.
    "clustered_v2": (
        CircularObstacle(-3.0,   1.4,  0.55),
        CircularObstacle(-1.3,   2.7,  0.45),
        CircularObstacle(-0.08,  0.55, 0.60),
        CircularObstacle(-2.4,   4.1,  0.50),
        CircularObstacle( 0.5,   3.9,  0.40),
        CircularObstacle( 1.98, -2.71, 0.50),
        CircularObstacle( 3.35, -1.44, 0.45),
        CircularObstacle(-0.42, -1.55, 0.55),
        CircularObstacle( 3.47, -3.86, 0.50),
    ),
    "bottleneck": (
        CircularObstacle(-4.8,  4.5, 0.50),
        CircularObstacle(-4.8, -4.5, 0.50),
        CircularObstacle( 4.8,  4.5, 0.50),
        CircularObstacle( 4.8, -4.5, 0.50),
        CircularObstacle(-1.8,  1.15, 0.55),
        CircularObstacle( 0.0,  1.15, 0.55),
        CircularObstacle( 1.8,  1.15, 0.55),
        CircularObstacle(-0.9, -1.15, 0.55),
        CircularObstacle( 0.9, -1.15, 0.55),
    ),
    # Out-of-distribution maps: larger obstacle radii (0.40-0.60 m vs. 0.25 m
    # in training) and counts beyond the training bands, placed as discrete
    # points on the vertical start-goal path with open space between them.
    "ood_easy": (
        CircularObstacle( 0.0,  0.3, 0.60),
        CircularObstacle(-4.5,  3.0, 0.50),
        CircularObstacle( 4.5,  3.0, 0.50),
        CircularObstacle(-4.5, -3.0, 0.50),
        CircularObstacle( 4.5, -3.0, 0.50),
        CircularObstacle(-5.8,  0.0, 0.45),
        CircularObstacle( 5.8,  0.0, 0.45),
        CircularObstacle( 3.2,  5.8, 0.40),
    ),
    # ood_medium/ood_hard follow the same discrete-point design.
    "ood_medium": (
        CircularObstacle(-1.8,  2.5, 0.55),
        CircularObstacle( 1.8, -2.0, 0.55),
        CircularObstacle(-5.0,  5.0, 0.50),
        CircularObstacle( 5.0,  5.0, 0.50),
        CircularObstacle(-5.0, -5.0, 0.50),
        CircularObstacle( 5.0, -5.0, 0.50),
        CircularObstacle(-6.2,  0.0, 0.45),
        CircularObstacle( 6.2,  0.0, 0.45),
        CircularObstacle( 3.5,  3.0, 0.40),
        CircularObstacle(-3.5, -3.5, 0.40),
        CircularObstacle( 2.5,  0.5, 0.45),
        CircularObstacle(-2.5, -0.8, 0.45),
    ),
    "ood_hard": (
        CircularObstacle(-1.5,  3.2, 0.60),
        CircularObstacle( 1.6,  1.0, 0.60),
        CircularObstacle(-1.6, -1.5, 0.60),
        CircularObstacle( 1.5, -3.2, 0.55),
        CircularObstacle(-5.2,  5.5, 0.50),
        CircularObstacle( 5.2,  5.5, 0.50),
        CircularObstacle(-5.2, -5.5, 0.50),
        CircularObstacle( 5.2, -5.5, 0.50),
        CircularObstacle(-6.3,  2.0, 0.45),
        CircularObstacle( 6.3,  2.0, 0.45),
        CircularObstacle(-6.3, -2.0, 0.45),
        CircularObstacle( 6.3, -2.0, 0.45),
        CircularObstacle(-4.0,  0.0, 0.40),
        CircularObstacle( 4.0,  0.0, 0.40),
        CircularObstacle(-3.2,  5.8, 0.40),
        CircularObstacle( 3.2, -5.8, 0.40),
    ),
    # Maps for randomized start/goal evaluation (--randomize-start-goal), built
    # from isolated 2-obstacle gates at separated locations around the arena.
    # in_dist_random uses the training obstacle radius (0.25 m).
    "in_dist_random": (
        CircularObstacle(-4.21,  2.79, 0.25),
        CircularObstacle(-2.79,  4.21, 0.25),
        CircularObstacle( 4.21,  2.79, 0.25),
        CircularObstacle( 2.79,  4.21, 0.25),
        CircularObstacle( 2.79, -4.21, 0.25),
        CircularObstacle( 4.21, -2.79, 0.25),
        CircularObstacle(-2.79, -4.21, 0.25),
        CircularObstacle(-4.21, -2.79, 0.25),
        CircularObstacle(-0.00, -1.00, 0.25),
        CircularObstacle( 0.00,  1.00, 0.25),
    ),
    # shift_random: the same gate layout with larger radii (0.45-0.55 m).
    "shift_random": (
        CircularObstacle(-4.38,  2.62, 0.50),
        CircularObstacle(-2.62,  4.38, 0.50),
        CircularObstacle( 4.38,  2.62, 0.50),
        CircularObstacle( 2.62,  4.38, 0.50),
        CircularObstacle( 2.62, -4.38, 0.55),
        CircularObstacle( 4.38, -2.62, 0.55),
        CircularObstacle(-2.62, -4.38, 0.55),
        CircularObstacle(-4.38, -2.62, 0.55),
        CircularObstacle(-1.20,  1.50, 0.45),
        CircularObstacle( 1.20,  1.50, 0.45),
        CircularObstacle(-0.00, -2.70, 0.45),
        CircularObstacle( 0.00, -0.30, 0.45),
    ),
    # In-distribution training maps (maps 030/080/070 of the procedural
    # training bank), evaluated on their own saved scenarios.
    "train_easy": (
        CircularObstacle(5.6058, 5.7400, 0.250),
        CircularObstacle(6.1317, 5.1576, 0.250),
        CircularObstacle(4.7730, 4.1428, 0.250),
        CircularObstacle(5.4157, 4.6400, 0.250),
    ),
    "train_medium": (
        CircularObstacle(-0.3597, 3.6194, 0.250),
        CircularObstacle(0.9941, 4.8083, 0.250),
        CircularObstacle(-0.4779, 5.6526, 0.250),
        CircularObstacle(0.3136, 5.7173, 0.250),
        CircularObstacle(-0.8003, 4.4698, 0.250),
        CircularObstacle(-2.9884, 3.1548, 0.250),
        CircularObstacle(5.1119, 4.7310, 0.250),
        CircularObstacle(-6.7150, -6.2563, 0.250),
        CircularObstacle(-3.7407, 0.6807, 0.250),
        CircularObstacle(4.1669, -0.1565, 0.250),
    ),
    "train_hard": (
        CircularObstacle(6.1729, 0.6300, 0.250),
        CircularObstacle(-4.1342, 0.1770, 0.250),
        CircularObstacle(-3.8262, 1.1108, 0.250),
        CircularObstacle(5.4796, -1.1147, 0.250),
        CircularObstacle(-0.2860, -0.5486, 0.250),
        CircularObstacle(4.8039, 0.5327, 0.250),
        CircularObstacle(-5.7563, 1.2886, 0.250),
        CircularObstacle(-0.1059, 1.4136, 0.250),
        CircularObstacle(0.4974, -0.4283, 0.250),
        CircularObstacle(-4.8607, 0.2542, 0.250),
        CircularObstacle(-5.4284, 2.1003, 0.250),
        CircularObstacle(6.5111, -0.3278, 0.250),
        CircularObstacle(-1.0509, 0.7066, 0.250),
        CircularObstacle(1.0843, -0.0536, 0.250),
        CircularObstacle(-5.8880, 0.3319, 0.250),
        CircularObstacle(6.3735, -1.1679, 0.250),
        CircularObstacle(0.4610, 1.2849, 0.250),
        CircularObstacle(5.3723, 0.6172, 0.250),
        CircularObstacle(0.0858, -1.0023, 0.250),
        CircularObstacle(-4.4488, 1.7068, 0.250),
        CircularObstacle(4.9081, -0.9732, 0.250),
        CircularObstacle(6.1830, 3.2201, 0.250),
        CircularObstacle(4.5191, 3.5159, 0.250),
        CircularObstacle(-5.3884, -3.6894, 0.250),
        CircularObstacle(4.0452, 5.8255, 0.250),
        CircularObstacle(3.3350, 2.5094, 0.250),
        CircularObstacle(-4.3917, 4.1628, 0.250),
        CircularObstacle(3.1279, 5.8918, 0.250),
        CircularObstacle(-2.0861, 1.9490, 0.250),
        CircularObstacle(6.8615, -5.7273, 0.250),
        CircularObstacle(5.2951, -3.0449, 0.250),
        CircularObstacle(-1.6528, 6.7677, 0.250),
        CircularObstacle(-2.8014, 3.8097, 0.250),
        CircularObstacle(1.4997, 2.2346, 0.250),
        CircularObstacle(-6.7052, -3.9458, 0.250),
        CircularObstacle(-2.1393, 5.1479, 0.250),
        CircularObstacle(3.5209, -3.4177, 0.250),
        CircularObstacle(0.4412, 3.9377, 0.250),
        CircularObstacle(-3.7965, -6.6667, 0.250),
        CircularObstacle(1.9802, -4.2084, 0.250),
        CircularObstacle(4.3957, -2.1212, 0.250),
        CircularObstacle(-5.0615, 6.6601, 0.250),
        CircularObstacle(5.9488, -2.3665, 0.250),
    ),
    # Distribution-shift maps: symmetric 5x5-lattice designs (shift_map1,
    # shift_map2) with increasing shape/size complexity toward the center, and a
    # cluttered all-square field (shift_map3). Evaluated with
    # --randomize-start-goal --min-endpoint-clearance 0.30.
    "shift_map1": (
        CircularObstacle( -5.40,  -5.40, 0.30, shape="circle"),
        CircularObstacle( -5.40,  -2.70, 0.30, shape="circle"),
        CircularObstacle( -5.40,   0.00, 0.30, shape="circle"),
        CircularObstacle( -5.40,   2.70, 0.30, shape="circle"),
        CircularObstacle( -5.40,   5.40, 0.30, shape="circle"),
        CircularObstacle( -2.70,  -5.40, 0.30, shape="circle"),
        CircularObstacle( -2.70,  -2.70, 0.30, shape="circle"),
        CircularObstacle( -2.70,   0.00, 0.30, shape="circle"),
        CircularObstacle( -2.70,   2.70, 0.30, shape="circle"),
        CircularObstacle( -2.70,   5.40, 0.30, shape="circle"),
        CircularObstacle(  0.00,  -5.40, 0.30, shape="circle"),
        CircularObstacle(  0.00,  -2.70, 0.30, shape="circle"),
        CircularObstacle(  0.00,   2.70, 0.30, shape="circle"),
        CircularObstacle(  0.00,   5.40, 0.30, shape="circle"),
        CircularObstacle(  2.70,  -5.40, 0.30, shape="circle"),
        CircularObstacle(  2.70,  -2.70, 0.30, shape="circle"),
        CircularObstacle(  2.70,   0.00, 0.30, shape="circle"),
        CircularObstacle(  2.70,   2.70, 0.30, shape="circle"),
        CircularObstacle(  2.70,   5.40, 0.30, shape="circle"),
        CircularObstacle(  5.40,  -5.40, 0.30, shape="circle"),
        CircularObstacle(  5.40,  -2.70, 0.30, shape="circle"),
        CircularObstacle(  5.40,   0.00, 0.30, shape="circle"),
        CircularObstacle(  5.40,   2.70, 0.30, shape="circle"),
        CircularObstacle(  5.40,   5.40, 0.30, shape="circle"),
        CircularObstacle(  0.00,   0.00, 1.40, shape="circle"),
    ),
    "shift_map2": (
        CircularObstacle( -5.40,  -5.40, 0.30, shape="circle"),
        CircularObstacle( -5.40,  -2.70, 0.30, shape="circle"),
        CircularObstacle( -5.40,   0.00, 0.30, shape="circle"),
        CircularObstacle( -5.40,   2.70, 0.30, shape="circle"),
        CircularObstacle( -5.40,   5.40, 0.30, shape="circle"),
        CircularObstacle( -2.70,  -5.40, 0.30, shape="circle"),
        CircularObstacle( -2.70,   5.40, 0.30, shape="circle"),
        CircularObstacle(  0.00,  -5.40, 0.30, shape="circle"),
        CircularObstacle(  0.00,   5.40, 0.30, shape="circle"),
        CircularObstacle(  2.70,  -5.40, 0.30, shape="circle"),
        CircularObstacle(  2.70,   5.40, 0.30, shape="circle"),
        CircularObstacle(  5.40,  -5.40, 0.30, shape="circle"),
        CircularObstacle(  5.40,  -2.70, 0.30, shape="circle"),
        CircularObstacle(  5.40,   0.00, 0.30, shape="circle"),
        CircularObstacle(  5.40,   2.70, 0.30, shape="circle"),
        CircularObstacle(  5.40,   5.40, 0.30, shape="circle"),
        CircularObstacle( -2.70,  -2.70, 0.65, shape="square"),
        CircularObstacle( -2.70,   0.00, 0.70, shape="circle"),
        CircularObstacle( -2.70,   2.70, 0.65, shape="square"),
        CircularObstacle(  0.00,  -2.70, 0.70, shape="circle"),
        CircularObstacle(  0.00,   0.00, 0.70, shape="circle"),
        CircularObstacle(  0.00,   2.70, 0.70, shape="circle"),
        CircularObstacle(  2.70,  -2.70, 0.65, shape="square"),
        CircularObstacle(  2.70,   0.00, 0.70, shape="circle"),
        CircularObstacle(  2.70,   2.70, 0.65, shape="square"),
    ),
    "shift_map3": (
        CircularObstacle(  0.30,  -0.20, 0.80, shape="square"),
        CircularObstacle(  2.64,  -6.27, 0.55, shape="square"),
        CircularObstacle(  6.49,   3.63, 0.55, shape="square"),
        CircularObstacle( -5.48,  -2.24, 0.35, shape="square"),
        CircularObstacle(  3.98,   2.29, 0.40, shape="square"),
        CircularObstacle(  6.17,   0.97, 0.25, shape="square"),
        CircularObstacle( -3.48,  -2.66, 0.55, shape="square"),
        CircularObstacle(  0.14,   2.00, 0.65, shape="square"),
        CircularObstacle( -1.70,   4.09, 0.55, shape="square"),
        CircularObstacle( -1.08,   6.18, 0.45, shape="square"),
        CircularObstacle(  0.25,   5.85, 0.35, shape="square"),
        CircularObstacle(  5.58,  -5.36, 0.55, shape="square"),
        CircularObstacle(  5.27,  -0.89, 0.20, shape="square"),
        CircularObstacle( -2.01,   0.55, 0.25, shape="square"),
        CircularObstacle(  3.17,  -0.90, 0.45, shape="square"),
        CircularObstacle( -5.35,   2.23, 0.40, shape="square"),
        CircularObstacle(  6.48,  -4.03, 0.20, shape="square"),
        CircularObstacle( -3.05,   3.47, 0.35, shape="square"),
        CircularObstacle(  4.30,   0.02, 0.55, shape="square"),
        CircularObstacle(  4.70,   4.00, 0.45, shape="square"),
        CircularObstacle(  3.07,  -1.99, 0.30, shape="square"),
        CircularObstacle(  6.37,  -6.17, 0.20, shape="square"),
        CircularObstacle( -2.52,  -3.98, 0.55, shape="square"),
    ),
    # shift_map2_v2: shift_map2's lattice with an axis-aligned center square
    # (half-side 0.60; rotated boxes are not supported by the geometry
    # pipeline), 0.60 m inner circles, and outer circles alternating 0.45/0.30 m
    # in a checkerboard pattern that preserves the lattice symmetry.
    "shift_map2_v2": (
        CircularObstacle( -5.40,  -5.40, 0.45, shape="circle"),
        CircularObstacle( -5.40,  -2.70, 0.30, shape="circle"),
        CircularObstacle( -5.40,   0.00, 0.45, shape="circle"),
        CircularObstacle( -5.40,   2.70, 0.30, shape="circle"),
        CircularObstacle( -5.40,   5.40, 0.45, shape="circle"),
        CircularObstacle( -2.70,  -5.40, 0.30, shape="circle"),
        CircularObstacle( -2.70,   5.40, 0.30, shape="circle"),
        CircularObstacle(  0.00,  -5.40, 0.45, shape="circle"),
        CircularObstacle(  0.00,   5.40, 0.45, shape="circle"),
        CircularObstacle(  2.70,  -5.40, 0.30, shape="circle"),
        CircularObstacle(  2.70,   5.40, 0.30, shape="circle"),
        CircularObstacle(  5.40,  -5.40, 0.45, shape="circle"),
        CircularObstacle(  5.40,  -2.70, 0.30, shape="circle"),
        CircularObstacle(  5.40,   0.00, 0.45, shape="circle"),
        CircularObstacle(  5.40,   2.70, 0.30, shape="circle"),
        CircularObstacle(  5.40,   5.40, 0.45, shape="circle"),
        CircularObstacle( -2.70,   0.00, 0.60, shape="circle"),
        CircularObstacle(  2.70,   0.00, 0.60, shape="circle"),
        CircularObstacle(  0.00,  -2.70, 0.60, shape="circle"),
        CircularObstacle(  0.00,   2.70, 0.60, shape="circle"),
        CircularObstacle( -2.70,  -2.70, 0.65, shape="square"),
        CircularObstacle( -2.70,   2.70, 0.65, shape="square"),
        CircularObstacle(  2.70,  -2.70, 0.65, shape="square"),
        CircularObstacle(  2.70,   2.70, 0.65, shape="square"),
        CircularObstacle(  0.00,   0.00, 0.60, shape="square"),
    ),
    # shift_map3_v2: 17 axis-aligned squares (half-side 0.25-0.55 m) placed by
    # the procedural generator's cell-growth algorithm at level 6; squares that
    # overlap after resizing are dropped.
    "shift_map3_v2": (
        CircularObstacle( -4.25,  -2.25, 0.55, shape="square"),
        CircularObstacle( -3.75,  -3.75, 0.45, shape="square"),
        CircularObstacle( -3.25,   3.25, 0.40, shape="square"),
        CircularObstacle( -2.75,  -1.75, 0.35, shape="square"),
        CircularObstacle( -2.75,   0.75, 0.35, shape="square"),
        CircularObstacle( -2.25,   2.75, 0.35, shape="square"),
        CircularObstacle( -2.25,   3.75, 0.40, shape="square"),
        CircularObstacle( -1.75,  -1.75, 0.45, shape="square"),
        CircularObstacle( -1.25,   2.75, 0.35, shape="square"),
        CircularObstacle( -0.75,  -4.25, 0.55, shape="square"),
        CircularObstacle( -0.25,   0.25, 0.45, shape="square"),
        CircularObstacle( -0.25,   1.75, 0.55, shape="square"),
        CircularObstacle(  1.25,   1.75, 0.40, shape="square"),
        CircularObstacle(  1.25,   4.25, 0.30, shape="square"),
        CircularObstacle(  4.25,  -2.25, 0.30, shape="square"),
        CircularObstacle(  4.25,  -0.75, 0.45, shape="square"),
        CircularObstacle(  4.25,   2.25, 0.40, shape="square"),
    ),
    # shift_map1_square: shift_map1 with every obstacle as an axis-aligned square.
    "shift_map1_square": (
        CircularObstacle( -5.40,  -5.40, 0.30, shape="square"),
        CircularObstacle( -5.40,  -2.70, 0.30, shape="square"),
        CircularObstacle( -5.40,   0.00, 0.30, shape="square"),
        CircularObstacle( -5.40,   2.70, 0.30, shape="square"),
        CircularObstacle( -5.40,   5.40, 0.30, shape="square"),
        CircularObstacle( -2.70,  -5.40, 0.30, shape="square"),
        CircularObstacle( -2.70,  -2.70, 0.30, shape="square"),
        CircularObstacle( -2.70,   0.00, 0.30, shape="square"),
        CircularObstacle( -2.70,   2.70, 0.30, shape="square"),
        CircularObstacle( -2.70,   5.40, 0.30, shape="square"),
        CircularObstacle(  0.00,  -5.40, 0.30, shape="square"),
        CircularObstacle(  0.00,  -2.70, 0.30, shape="square"),
        CircularObstacle(  0.00,   2.70, 0.30, shape="square"),
        CircularObstacle(  0.00,   5.40, 0.30, shape="square"),
        CircularObstacle(  2.70,  -5.40, 0.30, shape="square"),
        CircularObstacle(  2.70,  -2.70, 0.30, shape="square"),
        CircularObstacle(  2.70,   0.00, 0.30, shape="square"),
        CircularObstacle(  2.70,   2.70, 0.30, shape="square"),
        CircularObstacle(  2.70,   5.40, 0.30, shape="square"),
        CircularObstacle(  5.40,  -5.40, 0.30, shape="square"),
        CircularObstacle(  5.40,  -2.70, 0.30, shape="square"),
        CircularObstacle(  5.40,   0.00, 0.30, shape="square"),
        CircularObstacle(  5.40,   2.70, 0.30, shape="square"),
        CircularObstacle(  5.40,   5.40, 0.30, shape="square"),
        CircularObstacle(  0.00,   0.00, 1.40, shape="square"),
    ),
    # speed_pillar: a single r=1.8 obstacle between opposite-corner start/goal
    # boxes (superseded by speed_gate).
    "speed_pillar": (
        CircularObstacle(0.00, 0.00, 1.80, shape="circle"),
    ),
    # speed_gate: two wall segments (3 touching r=1.0 circles each) leaving a
    # single 0.65 m gap on the horizontal centerline, with a ~13.8 m straight
    # run-up from the west start box to the gate.
    "speed_gate": (
        CircularObstacle(0.00,  1.4, 1.0, shape="circle"),
        CircularObstacle(0.00,  3.4, 1.0, shape="circle"),
        CircularObstacle(0.00,  5.4, 1.0, shape="circle"),
        CircularObstacle(0.00, -1.4, 1.0, shape="circle"),
        CircularObstacle(0.00, -3.4, 1.0, shape="circle"),
        CircularObstacle(0.00, -5.4, 1.0, shape="circle"),
    ),
    # diagonal: bottom-left to top-right traversal with 180-degree point symmetry
    # about the arena center: an r=1.0 center blocker plus 5 symmetric pairs
    # (r=0.50-0.65). The P2-P3 gate and its mirror are 0.899 m; the next
    # tightest gap is 1.258 m.
    "diagonal": (
        CircularObstacle(0.0, 0.0, 1.00, shape="circle"),
        CircularObstacle(-4.5, -0.75, 0.65, shape="circle"),
        CircularObstacle(4.5, 0.75, 0.65, shape="circle"),
        CircularObstacle(-0.7, -3.6, 0.60, shape="circle"),
        CircularObstacle(0.7, 3.6, 0.60, shape="circle"),
        CircularObstacle(-2.569, -2.759, 0.55, shape="circle"),
        CircularObstacle(2.569, 2.759, 0.55, shape="circle"),
        CircularObstacle(-4.7, 1.7, 0.55, shape="circle"),
        CircularObstacle(4.7, -1.7, 0.55, shape="circle"),
        CircularObstacle(-1.7, 4.7, 0.50, shape="circle"),
        CircularObstacle(1.7, -4.7, 0.50, shape="circle"),
    ),
}


# Map-specific structured start/goal regions.
# Format: (x_min, x_max, y_min, y_max).
# The sampler also clips these boxes to the arena wall-clearance margin.
LEGACY_STRUCTURED_SCENARIO_REGIONS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    # Left-to-right traversal through the bottleneck.
    "bottleneck": {
        "start": (-7.5, -4.0, -2.0, 2.0),
        "goal": (4.0, 7.5, -2.0, 2.0),
    },
    # Diagonal traversal through the obstacle cluster.
    "clustered": {
        "start": (-7.5, -4.0, 4.0, 7.5),
        "goal": (6.0, 7.5, -7.5, -6.0),
    },
    # Same start region as "clustered"; 2.0x2.0 m goal box in the corner.
    "clustered_v2": {
        "start": (-7.5, -4.0, 4.0, 7.5),
        "goal": (5.5, 7.5, -7.5, -5.5),
    },
    "clustered_gap1cm": {
        "start": (-7.5, -4.0, 4.0, 7.5),
        "goal": (6.0, 7.5, -7.5, -6.0),
    },
    "clustered_gap2cm": {
        "start": (-7.5, -4.0, 4.0, 7.5),
        "goal": (6.0, 7.5, -7.5, -6.0),
    },
    "clustered_gap3cm": {
        "start": (-7.5, -4.0, 4.0, 7.5),
        "goal": (6.0, 7.5, -7.5, -6.0),
    },
    # Same regions as medium_shift.
    "corridor_derived_highshift": {
        "start": (-2.0, 2.0, 4.0, 7.5),
        "goal": (-4.0, 4.0, -7.5, -4.0),
    },
    "corridor_derived_highshift_v2": {
        "start": (-2.0, 2.0, 4.0, 7.5),
        "goal": (-4.0, 4.0, -7.5, -4.0),
    },
    "corridor_derived_highshift_v4": {
        "start": (-2.0, 2.0, 4.0, 7.5),
        "goal": (-4.0, 4.0, -7.5, -4.0),
    },
    # Top-to-bottom traversal; the large apex obstacle sits just below the start box.
    "new_medium_v2": {
        "start": (-2.0, 2.0, 4.0, 7.5),
        "goal": (-4.0, 4.0, -7.5, -4.0),
    },
    # Smaller start box and larger goal box than new_medium_v2.
    "circle_and_square": {
        "start": (-1.0, 1.0, 5.0, 6.8),
        "goal": (-5.5, 5.5, -7.0, -4.0),
    },
    # Scattered map: start and goal on opposite sides.
    "shifted": {
        "start": (-7.0, -5.0, 1.0, 3.0),
        "goal": (5.0, 7.0, -3.0, -1.0),
    },
    # Out-of-distribution trio: identical centered vertical start/goal boxes.
    "ood_easy": {
        "start": (-1.5, 1.5, 4.0, 7.0),
        "goal": (-1.5, 1.5, -7.0, -4.0),
    },
    "ood_medium": {
        "start": (-1.5, 1.5, 4.0, 7.0),
        "goal": (-1.5, 1.5, -7.0, -4.0),
    },
    "ood_hard": {
        "start": (-1.5, 1.5, 4.0, 7.0),
        "goal": (-1.5, 1.5, -7.0, -4.0),
    },
    # Opposite-corner boxes, ~17.8 m apart center-to-center, forcing a long
    # straight run-up directly at the single central obstacle (speed_pillar).
    "speed_pillar": {
        "start": (-7.2, -5.4, 5.4, 7.2),
        "goal": (5.4, 7.2, -7.2, -5.4),
    },
    # Far west/east edges, centered on the gate's y=0 gap: a clean, nearly
    # straight ~13.8 m horizontal run-up directly at the gate (speed_gate).
    "speed_gate": {
        "start": (-7.2, -6.6, -0.3, 0.3),
        "goal": (6.6, 7.2, -0.3, 0.3),
    },
    # Bottom-left -> top-right diagonal traversal.
    "diagonal": {
        "start": (-7.5, -4.6, -7.5, -4.6),
        "goal": (4.6, 7.5, 4.6, 7.5),
    },
}


# -----------------------------------------------------------------------------
# Dynamic (moving) obstacles.
#
# Copy of the evaluator's DynamicObstacleSpec/_position_on_waypoints, kept
# here so this module does not import the heavier evaluator.
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


# Number of trailing LEGACY_UNSEEN_MAPS[name] obstacles that are dynamic, keyed
# by map name. A map absent from this dict is treated as fully static.
# Populated below from len(LEGACY_DYNAMIC_OBSTACLE_SPECS[name]).
LEGACY_DYNAMIC_OBSTACLE_COUNTS: dict[str, int] = {}

# The DynamicObstacleSpec for each dynamic obstacle, in the same trailing
# order as LEGACY_DYNAMIC_OBSTACLE_COUNTS/LEGACY_UNSEEN_MAPS.
LEGACY_DYNAMIC_OBSTACLE_SPECS: dict[str, tuple[DynamicObstacleSpec, ...]] = {}


def _dynamic_spec_positions_at_t0(specs: tuple[DynamicObstacleSpec, ...]) -> tuple[tuple[float, float], ...]:
    return tuple(spec.position_at(0.0) for spec in specs)


# Crossing-obstacle maps: each dynamic obstacle oscillates across the robot's
# diagonal path with open space around each crossing, so the risk depends on
# reaction timing rather than passage width.
# ood_medium_dynamic: static obstacles plus 2 crossings.
# ood_hard_dynamic: 3 crossings (near start, middle, near goal), no static
# obstacles.
_CROSS_RADIUS = 0.45
_CROSS_AMPLITUDE = 2.2   # full perpendicular sweep half-width (open field beyond)
_CROSS_SPEED = 0.5       # matches the robot's own max forward speed (vx)
_CROSS_HEADING_DEG = -45.0  # perpendicular to the (-6,-6)->(6,6) diagonal

# ood_medium_dynamic static layout: a center obstacle, two 3-obstacle clusters
# in the off-diagonal quadrants, and 4 corridor-edge obstacles at 3.1 m
# perpendicular offset, just outside each crossing obstacle's reach
# (2.2 m amplitude plus radii). Crossing speeds are 0.20-0.30 m/s.
_medium_dynamic_static = (
    CircularObstacle(0.0, 0.0, 0.5),
    CircularObstacle(-5.5, 4.0, 0.50), CircularObstacle(-3.0, 5.5, 0.45), CircularObstacle(-3.8, 1.0, 0.45),
    CircularObstacle(5.5, -4.0, 0.50), CircularObstacle(3.0, -5.5, 0.45), CircularObstacle(3.8, -1.0, 0.45),
    CircularObstacle(-5.021, -0.636, 0.45), CircularObstacle(-0.636, -5.021, 0.45),
    CircularObstacle(0.636, 5.021, 0.45), CircularObstacle(5.021, 0.636, 0.45),
)
# Variant with only the single center static obstacle.
_medium_dynamic_static_orig = (
    CircularObstacle(0.0, 0.0, 0.5),
)
_medium_dynamic_dynamic_specs = (
    DynamicObstacleSpec(
        name="start_crossing", radius=_CROSS_RADIUS, trajectory_type="line_oscillate",
        p0=(-2.828, -2.828), heading_deg=_CROSS_HEADING_DEG, amplitude=_CROSS_AMPLITUDE,
        speed=0.25, phase_deg=0.0,
    ),
    DynamicObstacleSpec(
        name="goal_crossing", radius=_CROSS_RADIUS, trajectory_type="line_oscillate",
        p0=(2.828, 2.828), heading_deg=_CROSS_HEADING_DEG, amplitude=_CROSS_AMPLITUDE,
        speed=0.25, phase_deg=240.0,
    ),
)
_hard_dynamic_dynamic_specs = (
    DynamicObstacleSpec(
        name="start_crossing", radius=_CROSS_RADIUS, trajectory_type="line_oscillate",
        p0=(-2.828, -2.828), heading_deg=_CROSS_HEADING_DEG, amplitude=_CROSS_AMPLITUDE,
        speed=0.20, phase_deg=0.0,
    ),
    DynamicObstacleSpec(
        name="mid_crossing", radius=_CROSS_RADIUS, trajectory_type="line_oscillate",
        p0=(0.0, 0.0), heading_deg=_CROSS_HEADING_DEG, amplitude=_CROSS_AMPLITUDE,
        speed=0.25, phase_deg=120.0,
    ),
    DynamicObstacleSpec(
        name="goal_crossing", radius=_CROSS_RADIUS, trajectory_type="line_oscillate",
        p0=(2.828, 2.828), heading_deg=_CROSS_HEADING_DEG, amplitude=_CROSS_AMPLITUDE,
        speed=0.30, phase_deg=240.0,
    ),
)

# ood_medhard_dynamic: between medium and hard. Start/goal crossings as in
# medium; the middle crossing is split into two short-range (1.2 m amplitude),
# slow (0.2 m/s) obstacles anchored at +-1.0 m along the diagonal, leaving a
# gap around the arena center. No static obstacles.
_medhard_dynamic_dynamic_specs = (
    DynamicObstacleSpec(
        name="start_crossing", radius=_CROSS_RADIUS, trajectory_type="line_oscillate",
        p0=(-2.828, -2.828), heading_deg=_CROSS_HEADING_DEG, amplitude=_CROSS_AMPLITUDE,
        speed=0.25, phase_deg=0.0,
    ),
    DynamicObstacleSpec(
        name="mid_a_crossing", radius=_CROSS_RADIUS, trajectory_type="line_oscillate",
        p0=(-1.0, -1.0), heading_deg=_CROSS_HEADING_DEG, amplitude=1.2,
        speed=0.20, phase_deg=90.0,
    ),
    DynamicObstacleSpec(
        name="mid_b_crossing", radius=_CROSS_RADIUS, trajectory_type="line_oscillate",
        p0=(1.0, 1.0), heading_deg=_CROSS_HEADING_DEG, amplitude=1.2,
        speed=0.20, phase_deg=200.0,
    ),
    DynamicObstacleSpec(
        name="goal_crossing", radius=_CROSS_RADIUS, trajectory_type="line_oscillate",
        p0=(2.828, 2.828), heading_deg=_CROSS_HEADING_DEG, amplitude=_CROSS_AMPLITUDE,
        speed=0.25, phase_deg=240.0,
    ),
)

LEGACY_DYNAMIC_OBSTACLE_SPECS["ood_medium_dynamic"] = _medium_dynamic_dynamic_specs
LEGACY_DYNAMIC_OBSTACLE_SPECS["ood_hard_dynamic"] = _hard_dynamic_dynamic_specs
LEGACY_DYNAMIC_OBSTACLE_SPECS["ood_medhard_dynamic"] = _medhard_dynamic_dynamic_specs
for _map_name, _specs in LEGACY_DYNAMIC_OBSTACLE_SPECS.items():
    LEGACY_DYNAMIC_OBSTACLE_COUNTS[_map_name] = len(_specs)

LEGACY_UNSEEN_MAPS["ood_medium_dynamic"] = _medium_dynamic_static + tuple(
    CircularObstacle(x, y, _CROSS_RADIUS)
    for x, y in _dynamic_spec_positions_at_t0(_medium_dynamic_dynamic_specs)
)
LEGACY_DYNAMIC_OBSTACLE_SPECS["ood_medium_dynamic_orig"] = _medium_dynamic_dynamic_specs
LEGACY_DYNAMIC_OBSTACLE_COUNTS["ood_medium_dynamic_orig"] = len(_medium_dynamic_dynamic_specs)
LEGACY_UNSEEN_MAPS["ood_medium_dynamic_orig"] = _medium_dynamic_static_orig + tuple(
    CircularObstacle(x, y, _CROSS_RADIUS)
    for x, y in _dynamic_spec_positions_at_t0(_medium_dynamic_dynamic_specs)
)
LEGACY_UNSEEN_MAPS["ood_hard_dynamic"] = tuple(
    CircularObstacle(x, y, _CROSS_RADIUS)
    for x, y in _dynamic_spec_positions_at_t0(_hard_dynamic_dynamic_specs)
)
LEGACY_UNSEEN_MAPS["ood_medhard_dynamic"] = tuple(
    CircularObstacle(x, y, _CROSS_RADIUS)
    for x, y in _dynamic_spec_positions_at_t0(_medhard_dynamic_dynamic_specs)
)

LEGACY_STRUCTURED_SCENARIO_REGIONS["ood_medium_dynamic"] = {
    "start": (-7.2, -5.0, -7.2, -5.0),
    "goal": (5.0, 7.2, 5.0, 7.2),
}
LEGACY_STRUCTURED_SCENARIO_REGIONS["ood_medium_dynamic_orig"] = {
    "start": (-7.2, -5.0, -7.2, -5.0),
    "goal": (5.0, 7.2, 5.0, 7.2),
}
LEGACY_STRUCTURED_SCENARIO_REGIONS["ood_hard_dynamic"] = {
    "start": (-7.2, -5.0, -7.2, -5.0),
    "goal": (5.0, 7.2, 5.0, 7.2),
}
LEGACY_STRUCTURED_SCENARIO_REGIONS["ood_medhard_dynamic"] = {
    "start": (-7.2, -5.0, -7.2, -5.0),
    "goal": (5.0, 7.2, 5.0, 7.2),
}

# shifted_v2: "shifted" with its obstacle at (0.4, -2.8) made dynamic,
# oscillating perpendicular to the start->goal direction (-18.43 deg); at
# t=0 it sits at its static position.
_shifted_v2_static = (
    CircularObstacle(-4.2, -1.0, 0.55),
    CircularObstacle(-2.0,  3.1, 0.45),
    CircularObstacle( 2.8,  3.7, 0.50),
    CircularObstacle( 4.4,  0.2, 0.40),
    CircularObstacle(-5.4,  5.2, 0.50),
    CircularObstacle(-5.2, -4.8, 0.45),
    CircularObstacle( 5.3,  4.7, 0.55),
    CircularObstacle( 4.8, -5.1, 0.50),
)
_SHIFTED_V2_HEADING_DEG = 71.57  # perpendicular to shifted's -18.43 deg start->goal direction
_shifted_v2_dynamic_specs = (
    DynamicObstacleSpec(
        name="mid_crossing", radius=0.60, trajectory_type="line_oscillate",
        p0=(0.4, -2.8), heading_deg=_SHIFTED_V2_HEADING_DEG, amplitude=_CROSS_AMPLITUDE,
        speed=0.25, phase_deg=0.0,
    ),
)
LEGACY_DYNAMIC_OBSTACLE_SPECS["shifted_v2"] = _shifted_v2_dynamic_specs
LEGACY_DYNAMIC_OBSTACLE_COUNTS["shifted_v2"] = len(_shifted_v2_dynamic_specs)
LEGACY_UNSEEN_MAPS["shifted_v2"] = _shifted_v2_static + tuple(
    CircularObstacle(x, y, 0.60)
    for x, y in _dynamic_spec_positions_at_t0(_shifted_v2_dynamic_specs)
)
LEGACY_STRUCTURED_SCENARIO_REGIONS["shifted_v2"] = dict(LEGACY_STRUCTURED_SCENARIO_REGIONS["shifted"])
