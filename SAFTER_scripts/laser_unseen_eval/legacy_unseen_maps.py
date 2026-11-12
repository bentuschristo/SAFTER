#!/usr/bin/env python3
"""Legacy/retired unseen-map variants for the laser structured evaluator.

As of 2026-11, SAFTER_scripts/laser_unseen_eval/evaluate_laser_unseen_maps_structured.py
reduced its default evaluation set to exactly three maps -- "low_shift",
"medium_shift", "high_shift" (formerly named "slalom", "corridor", and
"corridor_derived_highshift_v5" respectively) -- loaded from JSON files under
"Maps/Evaluation Maps/". Every OTHER map variant that evaluator used to define
inline (all the exploratory/superseded/OOD/distribution-shift/dynamic-obstacle
designs, with their full design-rationale comments) was relocated here VERBATIM
rather than deleted, since the comments document real, non-obvious experimental
history (what was tried, what broke, why a revision was made) that is still
useful if any of these variants are revisited later.

This module is self-contained and importable on its own (it duplicates the
small DynamicObstacleSpec/_position_on_waypoints dynamic-obstacle-trajectory
helpers that the main evaluator also defines, rather than importing that heavy
module -- which pulls in torch/mujoco/mjlab -- just for two small helpers).
It is intentionally NOT imported by evaluate_laser_unseen_maps_structured.py:
the whole point of this split is that these maps are no longer part of the
default pipeline. Import LEGACY_UNSEEN_MAPS / LEGACY_STRUCTURED_SCENARIO_REGIONS
(and LEGACY_DYNAMIC_OBSTACLE_SPECS / LEGACY_DYNAMIC_OBSTACLE_COUNTS for the
dynamic-obstacle maps) directly from this module to use any of these maps again.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

# Allow running/importing standalone, matching the convention used by
# evaluate_laser_unseen_maps_structured.py.
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
    # 2026-09-24: candidate new High-Shift map, derived directly from
    # corridor's own proven two-row pillar layout rather than continuing to
    # tune circle_and_square (whose center-obstacle difficulty scales very
    # differently for bcbf_cf vs. SEA-Nav -- see scripts/
    # plot_corridor_derived_highshift_preview.py for the full design
    # writeup and the radius-sweep investigation that motivated this).
    # Same two rows / same start-goal regions as corridor, but each row's
    # pattern (outer->center, mirrored) is circle(0.50) -> square(0.65) ->
    # circle(0.75, center) -> square(0.65) -> circle(0.50) instead of five
    # uniform 0.50m circles -- every adjacent-obstacle gap within a row is
    # solved to stay exactly 0.9m, corridor's own gap width, so local
    # per-gate difficulty should match corridor's while the added obstacle
    # variety (bigger center circle, squares) raises overall difficulty.
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
    # 2026-09-24 v2 (revision 2): v1's full 5-obstacle top row (with squares
    # in the bottom row too) proved too hard for bcbf_cf (27.0% success)
    # while nailing SEA-Nav's target difficulty (48.8%, matching corridor's
    # own ~50%). First v2 attempt reverted the top row to 4 uniform circles
    # with the CENTER one removed entirely (wide-open passage) -- collision-
    # sample trajectories showed SEA-Nav never used that opening at all
    # (collapsed to 0% success, always detouring into the tight side gates
    # instead even when the direct start-goal line passed straight through
    # the open middle), so the removed-obstacle approach was abandoned.
    # This revision instead keeps all 4 top-row obstacles (no removal) but
    # makes the two INNERMOST ones (flanking where the open gate used to be)
    # squares, and simplifies the bottom row to only ONE square (the big
    # 0.90 center obstacle) with the two former flanking squares reverted to
    # circles at the same radius -- isolating the square-corner effect to
    # exactly the obstacles adjacent to each row's key passage. See
    # scripts/plot_corridor_derived_highshift_v2_preview.py.
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
    # 2026-09-24 v4: v2's two top-row squares (flanking the gate) proved
    # catastrophic for SEA-Nav -- 0.0% success, confirmed reproducibly on
    # TWO separate configurations (center removed, and center kept), with
    # collision samples showing it fails at the circle-square junction
    # every single time, never even reaching the bottom row. This reverts
    # the two inner top-row obstacles back to circles (identical position/
    # radius/gate to v2, shape only), keeping the bottom row's single
    # center square unchanged -- isolating the square-corner lever to only
    # the one obstacle (bottom-row center) that hasn't broken SEA-Nav.
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
    # 2026-09-12: replaces "corridor" in the medium slot of the
    # easy/medium/hard trio. Introduces varied obstacle radius (0.45-1.0m,
    # vs. the fixed 0.5m used by every other map so far) and an inverted-V
    # (apex-up) layout: THREE large (1.0m radius) obstacles -- the apex,
    # sitting dead on the direct start-goal line just below the start
    # region (the "V slot"), plus two more directly behind it (i.e. inside
    # the V, between the two legs) -- followed by two 2-obstacle legs
    # descending outward toward the goal. All pairwise obstacle-edge gaps
    # are >=0.968 m, comfortably above the reworked clustered_v2's own
    # tightest gap (0.830 m). Same start/goal regions as "corridor" for a
    # like-for-like top-to-bottom traversal.
    "new_medium_v2": (
        CircularObstacle( 0.0,  2.4, 1.00),
        CircularObstacle(-1.6, -0.1, 1.00),
        CircularObstacle( 1.6, -0.1, 1.00),
        CircularObstacle(-3.7, -1.9, 0.45),
        CircularObstacle(-4.9, -3.4, 0.50),
        CircularObstacle( 3.7, -1.9, 0.45),
        CircularObstacle( 4.9, -3.4, 0.50),
    ),
    # 2026-09-13: new evaluation-only map (added alongside, not replacing,
    # any existing map). Approved sketch-based variant from
    # plot_new_medium_v2_shape_variant_preview.py's "PROPOSED" design --
    # mirror-symmetric row layout (small pair -> big center circle -> small
    # pair -> big circle/square/square/big circle row -> two large squares
    # flanking the cluster), mixing CircularObstacle's "circle" and "square"
    # shapes (radius is interpreted as half-side for squares) and varied
    # obstacle radii (0.35-1.00 m), unlike every other map so far which is
    # all-circle. Same start/goal box convention as the rest of this dict.
    # 2026-09-24: bcbf_cf full-scale evaluation (500 cases) on the original
    # layout scored only 46.6% success / 53.4% collision -- far below its
    # 83-99% range on slalom/corridor/diagonal. First eased by shrinking the
    # center circle radius 1.00 -> 0.70 AND removing both small squares at
    # (-1.0/1.0, -2.6) -- got bcbf_cf to 70.6% but, per a per-obstacle
    # collision-clustering analysis of both bcbf_cf and SEA-Nav's original-
    # layout failures, over-corrected: 58.8% of bcbf_cf's original collisions
    # and ~99% of SEA-Nav's clustered at the near-start gate (this center
    # circle + the (-1.6,3.4) flanking circle) -- the squares near the goal
    # only ever accounted for ~7.5% of bcbf_cf's failures, and SEA-Nav's
    # failures never even reached them (blocked earlier at the start gate).
    # Middle-ground revision: keep the center-circle shrink (targets the
    # actual shared bottleneck) but restore both squares (removing them
    # wasn't addressing the real problem and was likely why SEA-Nav's eased-
    # map success rate jumped all the way to 92.6%, above even its own
    # Medium-Shift number -- breaking the intended Low<Medium<High ordering).
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
    # clustered_gap_Ncm (2026-09-09): identical to "clustered" except the
    # obstacle at index 1 ((-1.3, 2.7), the near side of the top cluster's
    # tightest passable gap to obstacle 0 at (-2.6, 1.7)) has its radius
    # shrunk by N cm, widening that one specific gap (raw surface-to-surface
    # clearance 0.640 m at N=0) by exactly N cm without touching any other
    # obstacle in the map. This is the gap the new (invariant_ellipse)
    # bCBF-CF policy's clustered collisions are concentrated at (confirmed
    # via the seed_1000 collision-trajectory log: every logged collision's
    # final position sits within ~5 cm of this gap's midpoint). Sweep
    # candidates for finding the smallest widening that flips bCBF-CF's
    # ~1-2 cm shortfall to a clearance, without also rescuing vanilla/CBF.
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
    # clustered_v2 (2026-09-11): eases the two clusters that made "clustered"
    # the hardest of the five maps by a wide margin. Each cluster's obstacles
    # are scaled 1.30x outward from that cluster's own centroid (uniformly
    # widening every internal pairwise clearance -- every in-cluster gap is
    # now >=0.83 m, vs. the original's tightest 0.640 m), the bottom-right
    # cluster's centroid is additionally shifted 1.8 m toward the arena
    # center with its leftmost obstacle (index 7) nudged a further 0.3 m
    # left, and then the whole bottom-right group of four is shifted another
    # 0.4 m left (to remove a near-direct unobstructed diagonal from the
    # goal region to the start region past the right side of the cluster).
    # That last shift retightened the cross-cluster gap between obstacle
    # index 2 and index 7 to 0.526 m. Obstacle index 7 was then moved a
    # further 1.0 m left (to sit directly in the cross-cluster corridor
    # rather than off to its side) and 1.0 m down (which reopened that same
    # gap to 1.498 m).
    #
    # 2026-09-11 revision (a): evaluation showed several methods near-ceiling
    # (83-99.8% success) on clustered_v2, well above the original
    # clustered's numbers on the SAME checkpoints -- the diagonal path from
    # start to goal was too directly walkable, since the widening/shifting
    # above opened every gap well past the original map's tightest (0.640
    # m). Two more obstacles were retightened to restore comparable
    # difficulty: index 2 moved to (-0.1, 0.4) (tightening the 2-7 gap to
    # 0.675 m), and indices 5/6/8 scaled 0.70x inward (tightening 5-8 to
    # 0.640 m). This produced a good overall difficulty match to the
    # original clustered map (Vanilla 15.4%, close to the original's own
    # 14.4%), but also revealed a real, quantified difference between
    # bCBF-CF and bCBF-QP (91-92% vs. 39-42% success, ~0% infeasibility for
    # both but ~2.4x larger mean command-correction norm for bCBF-QP) --
    # the retightened map's genuine <=0.70 m pinch points amplify bCBF-QP's
    # inherently more conservative true-multi-row constraint enforcement
    # (vs. bCBF-CF's single blended softmin row) far more than the CBF
    # (non-backup) pair, which was unaffected (CBF-CF/CBF-QP stayed
    # near-identical to each other throughout).
    #
    # 2026-09-11 revision (b): rather than accept that amplification, the
    # specific pinch points identified above are loosened to a middle
    # ground -- tight enough to keep Vanilla/CBF-CF/CBF-QP meaningfully
    # below the bCBF pair, loose enough that bCBF-QP's extra conservatism
    # doesn't force it into as many timeouts/collisions as bCBF-CF avoids.
    # Index 2 moved to (-0.08, 0.55) and index 7 to (-0.42, -1.55)
    # (2-7 gap: 0.675 -> 0.977 m); indices 5/6/8 scaled 1.15x outward from
    # their own centroid (5-6: 0.918 m, 5-8: 0.882 m, 6-8: 1.473 m). The
    # overall tightest gap in the whole map is now the top cluster's own
    # fixed 1-3 internal gap (0.830 m, never touched by any of these
    # revisions), looser than the original map's 0.640 m but tighter than
    # the fully-loose first clustered_v2 attempt. Goal-region clearance
    # unaffected (>=2.11 m).
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
    # Out-of-distribution vs. the seanav_100_seed42_200scen training bank:
    # larger fixed obstacle radii (0.40-0.60m vs. training's fixed 0.25m) and
    # obstacle counts (8/12/16) that exceed training's typical easy/medium/hard
    # bands (~0-7/1-10/6-12). Each map places its obstacles as discrete
    # "anticipation points" directly on the vertical start-goal path, with open
    # recovery space between them rather than sustained bilateral corridors --
    # this favors bcbf_cf's backup-rollout foresight (it can start correcting
    # before an instantaneous filter like cbf_cd_cf would react) without
    # triggering the sustained-pressure over-conservatism that hurts bcbf_cf
    # on the dense, continuous "corridor" map.
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
    # ood_medium/ood_hard (below): reverted 2026-09-06 to the ORIGINAL
    # "anticipation points" design (matching
    # logs/evaluation/figures/three_ood_maps_easy_medium_hard.png, the
    # reference image the user asked to recreate), after the "hard-turn"
    # redesign (double S-curve concave corner) was set aside -- that design
    # is preserved at
    # logs/evaluation/ood_medium_hard_HARDTURN_DESIGN_BACKUP_20260906.py in
    # case it's worth revisiting. This original design gives 0% failures for
    # every method incl. vanilla at 500 cases/method/map (see
    # ood_maps_zero_failure_finding.md) -- it is not a differentiating map by
    # itself; it's being restored as the reference OOD static layout while
    # dynamic-obstacle work (see dynamic_obstacle_eval_design.md) is the
    # active differentiation avenue.
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
    # New maps for the randomized-endpoint milestone experiments (2026-09-08):
    # additive, do not reuse/modify ood_easy/ood_medium/ood_hard above. Both
    # are only ever meant to be run with --randomize-start-goal (full-arena
    # start/goal, not a fixed corridor).
    #
    # Design history (kept here since it took 3 real iterations to learn):
    #  v1 evenly-spaced single obstacles -> 100% success for every method
    #     (zero real failures, trivial).
    #  v2/v3 dense multi-obstacle CLUSTERS (5-10 obstacles within ~1m of each
    #     other) -> real failures finally appeared, but bcbf_cf collided MORE
    #     than cbf_cd_cf/vanilla (87.0% vs 94.5%/90.5% success on the harder
    #     map, 5 seeds x 40 cases), and the per-episode data showed bcbf_cf
    #     uniquely failing on specific starts inside/near the dense cluster
    #     that vanilla and cbf_cd_cf both passed cleanly every seed -- dense
    #     multi-obstacle clutter is apparently a genuine weakness for the
    #     invariant-ellipse backup mechanism, not a strength.
    #  v4 (current): isolated 2-obstacle "gates" (a narrow, single-file gap
    #     between exactly two obstacles, computed so the passable width after
    #     robot+clearance inflation is ~0.2-0.3m) at several separated
    #     locations/orientations around the arena, with each gate isolated
    #     from the others -- narrow enough to require genuine, several-step-
    #     ahead lateral commitment (unlike a single point obstacle, which a
    #     one-step-reactive filter dodges trivially), without the >2-nearby-
    #     obstacles clutter that v2/v3 showed hurts bcbf_cf specifically.
    #
    # "in_dist_random": matches the seanav_100_seed42_200scen training bank's
    # own "hard" band (12 obstacles) and fixed obstacle radius (0.25m).
    # Intended for milestone experiment (1): in-distribution generalization.
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
    # "shift_random": distribution shift vs. the training bank on BOTH obstacle
    # count (20, beyond training's 0-12 range) and size (0.45-0.55m fixed
    # radii, vs. training's fixed 0.25m) -- matching the same two axes the
    # ood_easy/medium/hard trio above already varies, using the same clustered
    # (not corridor/anticipation-point) layout principle. Intended for
    # milestone experiment (2): robustness under distribution shift.
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
    # Actual IN-DISTRIBUTION training maps (2026-09-08): copied verbatim from
    # generated/procedural_maps/seanav_100_seed42_200scen/maps/map_{030,080,070}.json
    # -- the exact map_id selection plotted in
    # logs/evaluation/figures/training_vs_evaluation_maps_combined.png's
    # "Training Map Samples" row (TRAINING_MAP_SELECTION in
    # plot_training_vs_evaluation_maps_combined.py). Evaluated against a
    # 100-of-200 subsample of that SAME map_id's own saved scenarios.csv rows
    # (see build_training_map_manifests in the scratch tooling), not a fresh
    # random/structured sample -- these are scenarios the training curriculum
    # itself could actually draw, the truest in-distribution generalization
    # check available.
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
    # Distribution-shift maps (2026-09-08), hand-sketched by the user:
    # symmetric 5x5-lattice designs for shift_map1/shift_map2, escalating
    # shape/size complexity toward the center; shift_map3 is a cluttered,
    # non-symmetric all-square field. All evaluated with --randomize-start-goal
    # --min-endpoint-clearance 0.30 (goal/start >=0.30m from any obstacle,
    # matching the 0.30m termination/collision footprint) and the default
    # require_blocked_path=True (direct path must pass near an obstacle).
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
    # shift_map2_v2 / shift_map3_v2 (2026-09-08): additive revisions -- do NOT
    # modify shift_map2/shift_map3 above, which are kept as the original,
    # already-evaluated designs. shift_map1 is deliberately left completely
    # untouched: across the full 5-seed run it is the first map this whole
    # project has found where bCBF-CF outright beats CBF-CF on raw success
    # rate (91.8% vs 87.6%, both beating Vanilla's 81.2%) *and* has zero
    # collisions across all 500 episodes -- see table_shift_maps_v2.tex.
    #
    # shift_map2_v2: same symmetric 5x5-lattice skeleton as shift_map2, with
    # 3 changes: (1) the center element is now an axis-aligned SQUARE
    # (half-side 0.60) instead of a circle -- true 45-degree rotation is not
    # supported anywhere in this codebase's geometry pipeline (MJCF geom
    # orientation, the ray-slab LiDAR-verification math, and the analytical
    # collision-termination math all assume axis-aligned boxes only; adding
    # real rotation support would mean building rotated intersection math in
    # all three places), so this keeps the "square instead of circle" part of
    # the request while skipping the unsupported rotation. (2) the 4 inner
    # "plus" edge circles shrink slightly, 0.70m -> 0.60m. (3) the 16 outer
    # lattice circles now alternate radius by (row_index + col_index) parity
    # -- a checkerboard "big-small-big-small" pattern (0.45m / 0.30m) that
    # still respects every reflection symmetry of the lattice, since any two
    # lattice-adjacent cells always differ in parity.
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
    # shift_map3_v2: less cluttered than shift_map3 (17 vs 23 obstacles),
    # generated by reusing generate_seanav_navigation_maps.py's actual
    # SEA-Nav-style cell-growth algorithm (grow_shape/generate_cells, the
    # same random-walk grid-cluster growth the real training-bank generator
    # uses) at level=6 (SEA-Nav's own "medium" tier) instead of ad-hoc
    # rejection sampling, then rendered as axis-aligned squares of varied
    # size (half-side 0.25-0.55m, vs. the cell-grid's native fixed 0.25m) --
    # any two resulting squares that overlap once resized are dropped rather
    # than shrunk, which further thins the layout.
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
    # shift_map1_square (2026-09-08): shift_map1's exact skeleton (same 25
    # positions/sizes: 24x r=0.30 in the 5x5 lattice + 1x r=1.40 at the empty
    # center) with every obstacle changed from circle to axis-aligned square,
    # to isolate whether shift_map1's "landslide" result (bCBF-CF beats
    # CBF-CF/Vanilla on success rate AND has zero collisions across all 5
    # seeds) depends on circular shape specifically, or just on "one large,
    # simple, isolated, symmetric obstacle." shift_map1 itself is untouched.
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
    # speed_pillar (2026-09-09, superseded by speed_gate below): first
    # landslide-hunting attempt -- a single r=1.8 obstacle dead-center
    # between opposite-corner boxes, following shift_map1's "one large
    # isolated obstacle" mechanism (bCBF-CF 91.8% vs vanilla 81.2%/CBF-CD-CF
    # 87.6%, 0 collisions, under --randomize-start-goal). Smoke-tested at
    # 10 cases: bcbf_cf=100%, vanilla=90%, cbf_cd_cf=100% success -- no
    # differentiation at all. Diagnosis: a single obstacle in an otherwise-
    # empty 15x15 m arena leaves enormous clearance on both sides (arena
    # half-width 7.5 m vs obstacle radius 1.8 m), so going around it is
    # trivial regardless of approach speed; shift_map1's harder cases came
    # from its *random* start/goal sampling occasionally forcing a tight
    # angle of attack, which a single fixed corner-to-corner path can't
    # reproduce. Kept only for the record; speed_gate replaces it.
    "speed_pillar": (
        CircularObstacle(0.00, 0.00, 1.80, shape="circle"),
    ),
    # speed_gate (2026-09-09): a genuine forced bottleneck instead of an
    # obstacle to steer wide around. Two solid wall segments (each a
    # touching 3-circle chain, r=1.0, reaching to within ~0.875 m of the
    # top/bottom arena walls -- comparable to the gate's own width, so
    # skirting around either end isn't an obviously easier alternative)
    # leave a single 0.65 m gap on the horizontal centerline: the same
    # order of magnitude as "clustered"'s empirically-confirmed difficult
    # gap (~0.64 m raw surface clearance, the one every method's clustered
    # collisions cluster at within 1-2 cm). Start/goal boxes sit on the far
    # west/east edges, centered on the gap's y-level, giving a clean ~13.8 m
    # straight run-up with nothing else in the way -- long enough to reach
    # and hold cruising speed well before the mandatory, precisely-timed
    # threading of the gate. This is the regime where a controller
    # reasoning over a full backup/braking rollout (bCBF-CF) should
    # separate from one reasoning only about the current instant (vanilla,
    # cbf_cd_cf): both now have to decide, while still at speed, whether
    # they can actually stop/turn in time to thread a gap barely wider than
    # the robot's own footprint, rather than just steering around open air.
    "speed_gate": (
        CircularObstacle(0.00,  1.4, 1.0, shape="circle"),
        CircularObstacle(0.00,  3.4, 1.0, shape="circle"),
        CircularObstacle(0.00,  5.4, 1.0, shape="circle"),
        CircularObstacle(0.00, -1.4, 1.0, shape="circle"),
        CircularObstacle(0.00, -3.4, 1.0, shape="circle"),
        CircularObstacle(0.00, -5.4, 1.0, shape="circle"),
    ),
    # diagonal (2026-09-23): new structured eval map, intended difficulty
    # between corridor (Medium-Shift) and circle_and_square (High-Shift).
    # Bottom-left -> top-right diagonal traversal (the opposite diagonal
    # from clustered_v2, which also sits between corridor and
    # circle_and_square in difficulty, to keep the two visually/structurally
    # distinct). Explicit 180-degree point symmetry about the arena center
    # ((x,y) and (-x,-y) both present for every obstacle except the
    # dead-center one) -- see scripts/plot_diagonal_shift_map_preview.py for
    # the full design writeup and the validated preview figure. All circles,
    # all large (r=0.50-1.00, vs. corridor's uniform r=0.50): one r=1.0
    # blocker dead-center (matches circle_and_square's own center circle),
    # plus 5 symmetric pairs. Deliberately loose -- minimum obstacle-
    # obstacle surface gap is 1.04 m (looser than corridor's ~0.9 m
    # tightest gap), so SAFTER/bCBF's backup filter isn't forced into an
    # over-conservative regime by a tight pinch point; difficulty above
    # corridor instead comes from more (11 vs. 9) and larger obstacles
    # blocking substantially more of the arena, plus the dead-center
    # blocker removing any straight-through shortcut entirely.
    # v4 (2026-09-23): tightened the P2-P3 gate (and its symmetric mirror
    # P2'-P3') from 1.04 m to 0.88 m -- matching corridor's own ~0.9 m
    # tightest gap -- by moving P3/P3' toward P2/P2' along their connecting
    # line. Root-caused via corridor's own SEA-Nav trajectories: its
    # CBF-shield policy physically stalls (not collides) right at a
    # genuinely tight gate, a failure mode v3's loosest 1.04 m gap never
    # triggered (SEA-Nav got 89% on v3 vs. only 48.6% on corridor).
    #
    # v4.1 (2026-09-23): 0.88 m overshot -- it dragged SAFTER down to 42%
    # (below CBF-CF's 49%, the opposite of what we want). Backed off to
    # 0.95 m: fixed SAFTER (67%, clearly ahead again) but SEA-Nav's stall
    # failure mode didn't trigger at all (90%, same as the ungated v3), and
    # bCBF-QP-cfmatched barely improved (8% vs. 0.88 m's 2%) despite the
    # gate loosening -- meaning something ELSE in the map was also hurting
    # bCBF-QP disproportionately, not just this one gate.
    #
    # v5 (2026-09-23): per-episode trajectory data (v4.1 run) showed exactly
    # that -- bCBF-QP-cfmatched's and CBF-QP-cfmatched's collisions cluster
    # heavily around (-3.2,-1.5), right at the P1-P3 gap, which was NEVER a
    # deliberate gate (1.154 m, the map's second-tightest gap by accident)
    # but was apparently tight enough to trip the QP-trained policies'
    # brittleness anyway; SAFTER's (bcbf_cf) collisions were spread out with
    # no such hotspot there. Fix: P1/P1' moved radially outward from the
    # center (t=1.25 scale: (-3.6,-0.6)->(-4.5,-0.75)) to open P1-P3 from
    # 1.154 m to 1.587 m -- now comfortably looser than every other
    # non-gate gap, no longer an accidental pinch. Separately, the P2-P3
    # gate itself retargeted to 0.899 m -- corridor's own literal tightest
    # gap (obstacles 1.9 m apart, r=0.5 each: 1.9-1.0=0.9 m) -- since the
    # observed SEA-Nav-stall threshold sits somewhere in the narrow 0.88-
    # 0.95 m band and 0.90 m is the one value we know triggers it on
    # corridor itself. New second-tightest gap is P1-P4 at 1.258 m. See
    # scripts/plot_diagonal_shift_map_preview.py for the full writeup.
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
    # Same start region as "clustered"; goal region enlarged from a 1.5x1.5 m
    # box to 2.0x2.0 m (still tucked in the corner, doesn't reach the moved
    # bottom-right obstacle cluster -- nearest obstacle is ~1.6 m clear).
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
    # Same regions as "corridor" -- see UNSEEN_MAPS["corridor_derived_highshift"].
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
    # Same top-to-bottom traversal as "corridor" (which this replaces in the
    # medium slot); the large apex obstacle sits just below the start box.
    "new_medium_v2": {
        "start": (-2.0, 2.0, 4.0, 7.5),
        "goal": (-4.0, 4.0, -7.5, -4.0),
    },
    # Matches plot_new_medium_v2_shape_variant_preview.py's PROPOSED start/goal
    # boxes exactly (smaller start box, larger goal box than new_medium_v2).
    "circle_and_square": {
        "start": (-1.0, 1.0, 5.0, 6.8),
        "goal": (-5.5, 5.5, -7.0, -4.0),
    },
    # Scattered map: use opposite sides and require at least one obstacle to lie
    # between start and goal under the blocking test below.
    "shifted": {
        "start": (-7.0, -5.0, 1.0, 3.0),
        "goal": (5.0, 7.0, -3.0, -1.0),
    },
    # Out-of-distribution Easy/Medium/Hard trio (see UNSEEN_MAPS above).
    # Identical, square, centered vertical start/goal boxes across all three,
    # matching logs/evaluation/figures/three_ood_maps_easy_medium_hard.png.
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
    # Bottom-left -> top-right diagonal traversal (see UNSEEN_MAPS["diagonal"]
    # above). Matches scripts/plot_diagonal_shift_map_preview.py's
    # DIAGONAL_REGIONS exactly.
    "diagonal": {
        "start": (-7.5, -4.6, -7.5, -4.6),
        "goal": (4.6, 7.5, 4.6, 7.5),
    },
}


# -----------------------------------------------------------------------------
# Dynamic (moving) obstacles.
#
# Duplicated from evaluate_laser_unseen_maps_structured.py's own
# DynamicObstacleSpec/_position_on_waypoints (kept here too so this module has
# no dependency on that heavier, torch/mujoco-importing script) -- see that
# file for the original ported-from-prototype history of this class.
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


# Redesigned from scratch (v2) after the v1 "narrowing gate" design (see git
# history / session notes) backfired: it made bcbf_cf collide MORE than
# cbf_cd_cf (15-40% vs 10-25%, vanilla only 5%), because a gate that narrows
# a shared passage down to near the minimum passable width turns caution
# into a liability -- bcbf_cf's larger required margin makes it hesitate
# longer deciding the gap is "wide enough", so it's more often still
# mid-transit when the gate closes again, INCREASING exposure to the closing
# threat rather than protecting against it. That is a margin-vs-space
# competition, which structurally favors whoever needs less space (same
# lesson as the static tight-gap maps earlier this session).
#
# v2 instead tests genuine crossing-obstacle timing, not space competition:
# a single obstacle per encounter sweeps back and forth ACROSS the robot's
# open diagonal path (not narrowing a shared passage), at a speed capped to
# the robot's own max forward speed (~0.5 m/s), with an OPEN, uncluttered
# arena around each crossing point so there is always ample room to detour
# once you react -- the risk is purely about WHEN you start reacting, not
# whether physical room exists. Since the analytic safety filter's clearance
# check is `distance - radius - obstacle_clearance_margin`, a larger margin
# (bcbf_cf's backup_margin stacks on top of the base clearance) makes the
# constraint activate at a larger stand-off distance for the SAME crossing
# obstacle -- i.e. bcbf_cf reacts earlier/farther out automatically, with no
# explicit velocity prediction needed. Reacting earlier against a moving
# threat gives more time/room to resolve smoothly; reacting late (cbf_cd_cf,
# smaller margin) risks a genuine race-condition collision, since the
# obstacle moves at a comparable speed to the robot's own.
#
# ood_medium_dynamic: 2 static anchor obstacles (off to the side, not
# blocking) + 1 dynamic crossing obstacle at mid-path ("partial dynamic").
# ood_hard_dynamic: 3 dynamic crossing obstacles spread along the path
# (near-start/mid/near-goal), staggered phases, 0 static obstacles ("all
# dynamic").
_CROSS_RADIUS = 0.45
_CROSS_AMPLITUDE = 2.2   # full perpendicular sweep half-width (open field beyond)
_CROSS_SPEED = 0.5       # matches the robot's own max forward speed (vx)
_CROSS_HEADING_DEG = -45.0  # perpendicular to the (-6,-6)->(6,6) diagonal

# v2 (2026-09-06): both maps now share the SAME 2-crossing recipe
# (start_crossing + goal_crossing, dropping mid_crossing) -- v1's 3-crossing
# ood_hard_dynamic was the first config all session where bcbf_cf beat
# cbf_cd_cf on success rate (15% vs 5%, see logs/evaluation/dynamic_pilot/
# ood_hard_dynamic_v1_severe_3crossing_CONFIG_BACKUP.py for the preserved
# original), but at 85-95% collision for everyone -- too severe to be a
# useful statistical test. Softened to 2 crossings here. ood_medium_dynamic
# was single-crossing before (cbf_cd_cf beat bcbf_cf there, 75% vs 65%);
# giving it the SAME 2-crossing compounding as hard, but keeping it "partial"
# dynamic via 2 static anchor obstacles (hard stays "all dynamic", 0 static),
# tests whether compounding two encounters is what's needed for bcbf_cf's
# edge to show up, matching the static "hard-turn" maps' finding that
# bcbf_cf needed TWO sequential corners, not one.
# v3 (2026-09-06): per user request, slow the dynamic obstacles down (base
# speed 0.5 = robot's max vx was apparently too fast) and diversify hard's
# three crossings to different speeds/periods rather than identical physics
# with only a phase offset. ood_medium_dynamic: replaced the 2 off-to-the-
# side static anchors with a SINGLE static obstacle dead center at the
# origin (on the direct diagonal path), keeping both dynamic crossings at
# their original positions/amplitude but slowed to 0.25 m/s (half of
# _CROSS_SPEED). ood_hard_dynamic: same 3 crossing positions as before
# (start/mid/goal), but each now has its own distinct (slower) speed instead
# of sharing _CROSS_SPEED.
# v4/v5 (2026-09-07, per user request/approval): added 6 static obstacles
# filling the two previously-empty off-diagonal quadrants (upper-left /
# lower-right), symmetric about the main diagonal. The innermost obstacle in
# each quadrant cluster is pulled toward center along the same
# anti-diagonal-ish line, so the two 3-obstacle clusters read as a second
# diagonal arm crossing the main start->goal diagonal near the middle (an "X"
# viewed along the diagonal) rather than two disconnected corner clusters.
# All 6 stay clear of the crossing obstacles' +-2.2m oscillation corridor
# (perpendicular distance from the main diagonal is ~3.39m for the innermost
# points vs. the 2.2+0.45=2.65m oscillation reach) -- this is a general
# clutter/framing increase, not a corridor-narrowing change (kept separate
# deliberately: narrowing was previously found to structurally favor
# whichever method needs less clearance margin, not a fair difficulty
# increase). Preview: logs/evaluation/figures/
# easy_ood_medium_ood_medium_dynamic_v5_proposed_xshape.png (user-approved).
# v6 (2026-09-07, per user request/approval, option 1 "corridor edges"): 4
# new "corridor edge" walls added, one flanking each crossing obstacle's own
# longitudinal position (same s as start_crossing/goal_crossing), at
# perpendicular offset 3.1m -- just outside the crossing obstacle's own max
# reach (2.2 amplitude + 0.45 radius + this obstacle's own 0.45 radius =
# 3.05m), geometrically guaranteed to never overlap the dynamic obstacle at
# any point in its oscillation. Removes the previously-unlimited open buffer
# beyond the oscillation zone (arena walls were ~7.5m away), bounding the
# outer edge of the safe zone right past where the dynamic obstacle can
# reach. Preview: logs/evaluation/figures/
# easy_ood_medium_ood_medium_dynamic_v6_proposed_corridorwalls.png
# (user-approved).
_medium_dynamic_static = (
    CircularObstacle(0.0, 0.0, 0.5),
    CircularObstacle(-5.5, 4.0, 0.50), CircularObstacle(-3.0, 5.5, 0.45), CircularObstacle(-3.8, 1.0, 0.45),
    CircularObstacle(5.5, -4.0, 0.50), CircularObstacle(3.0, -5.5, 0.45), CircularObstacle(3.8, -1.0, 0.45),
    CircularObstacle(-5.021, -0.636, 0.45), CircularObstacle(-0.636, -5.021, 0.45),
    CircularObstacle(0.636, 5.021, 0.45), CircularObstacle(5.021, 0.636, 0.45),
)
# v3 original design (2026-09-06, single static obstacle dead center + the
# same 2 crossings) -- preserved under a separate map name so it stays
# evaluable on demand without disturbing the current (v6) "ood_medium_dynamic"
# definition above. Reuses the exact same dynamic specs/regions as v6, only
# the static-obstacle list differs.
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

# ood_medhard_dynamic (2026-09-06, per user request): a rung between medium
# and hard. "1-2-1" layout -- start_crossing and goal_crossing reuse medium's
# exact position/amplitude/speed (0.25 m/s), and the single hard mid_crossing
# is split into TWO obstacles (mid_a/mid_b) anchored off-center along the
# diagonal at +-1.0m instead of one obstacle anchored at the true center
# (0,0). Splitting the anchor (not the amplitude) is what actually opens a
# ~2.83m gap-free stretch of the direct path around the arena's center -- the
# short 1.2 amplitude (vs 2.2 for the outer two) only narrows each mid
# obstacle's own lateral danger zone, keeping them individually less
# threatening than start/goal. Slower 0.2 m/s matches hard's slowest
# (start_crossing). 0 static obstacles (fully dynamic, like hard), but 4
# dynamic obstacles total (one more than hard's 3) -- the extra count is
# offset by the middle pair being slow, short-range, and gap-exploitable.
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

# shifted_v2 (2026-09-09, per user request): identical to "shifted" except
# one of the five central obstacles -- (0.4, -2.8, 0.60), the one closest to
# the arena's own center among them -- is made dynamic instead of static,
# reusing the same validated "crossing" recipe that gave bcbf_cf its one
# confirmed win over cbf_cd_cf via moving obstacles (see the ood_*_dynamic
# design notes above): line_oscillate perpendicular to the direct path,
# capped well under the robot's own max speed, with an open field either
# side of the sweep so the risk is purely about reaction timing, not a
# narrowed passage (narrowing was already found to backfire against
# bcbf_cf's larger required margin). heading_deg is perpendicular to
# shifted's own start(-6,2)->goal(6,-2) direction (atan2(-4,12)=-18.43 deg),
# not the generic -45 used by the ood_*_dynamic diagonal maps, since
# shifted's start/goal boxes sit on a different line. phase_deg=0 so the
# obstacle's t=0 position exactly matches its original static coordinate --
# the map is byte-identical to "shifted" at the first instant, only moving
# from there.
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
