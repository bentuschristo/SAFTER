# NOTE: this is a DUPLICATE of src/high_level_rl/evaluation_scenarios.py, kept in sync manually.
# It exists here because src/high_level_rl/evaluation_scenarios.py is also load-bearing for
# map-based code (high_level_sac/high_level_bcbf_sac/high_level_cbf_rl) that
# intentionally stays outside SAFTER_src. See SAFTER_src/README.md.

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EvaluationScenario:
    name: str
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    start_yaw: float


def yaw_toward_goal(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> float:
    return math.atan2(
        goal_xy[1] - start_xy[1],
        goal_xy[0] - start_xy[0],
    )


_EASY_START = (-6.0, 0.0)
_EASY_GOAL = (0.0, 0.0)

_MEDIUM_START = (-6.0, 0.0)
_MEDIUM_GOAL = (4.0, -3.4)

_HARD_START = (-6.0, 2.734)
_HARD_GOAL = (4.0, -1.0)


EASY = EvaluationScenario(
    name="easy",
    start_xy=_EASY_START,
    goal_xy=_EASY_GOAL,
    start_yaw=yaw_toward_goal(_EASY_START, _EASY_GOAL),
)

MEDIUM = EvaluationScenario(
    name="medium",
    start_xy=_MEDIUM_START,
    goal_xy=_MEDIUM_GOAL,
    start_yaw=yaw_toward_goal(_MEDIUM_START, _MEDIUM_GOAL),
)

HARD = EvaluationScenario(
    name="hard",
    start_xy=_HARD_START,
    goal_xy=_HARD_GOAL,
    start_yaw=yaw_toward_goal(_HARD_START, _HARD_GOAL),
)


SCENARIOS = {
    EASY.name: EASY,
    MEDIUM.name: MEDIUM,
    HARD.name: HARD,
}
