from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .config import ObstacleConfig, SafetyConfig
from .model import HighLevelKinematicModel


@dataclass(frozen=True)
class SafetyResult:
    command: np.ndarray
    nominal_command: np.ndarray
    h: float
    cbf_value_nominal: float
    active: bool
    correction_norm: float


class IdentitySafetyFilter:
    def filter(self, pose: np.ndarray, nominal_command: np.ndarray) -> SafetyResult:
        del pose
        u = np.asarray(nominal_command, dtype=np.float64).copy()
        return SafetyResult(u, u.copy(), math.inf, math.inf, False, 0.0)


class ClosedFormCircleCbf:
    """Closed-form zeroing-CBF filter for one circular obstacle.

    h = (x-x_o)^2 + (y-y_o)^2 - R_safe^2.
    Constraint: L_g h U + alpha h >= 0.

    Because the controlled point is directly velocity controlled through Xdot=B U,
    the constraint is affine in U and the Euclidean projection has a closed form.
    """

    def __init__(
        self,
        model: HighLevelKinematicModel,
        obstacle: ObstacleConfig,
        cfg: SafetyConfig,
    ) -> None:
        self.model = model
        self.obstacle = obstacle
        self.cfg = cfg

    def filter(self, pose: np.ndarray, nominal_command: np.ndarray) -> SafetyResult:
        u_nom = np.asarray(nominal_command, dtype=np.float64)
        dx = float(pose[0] - self.obstacle.x)
        dy = float(pose[1] - self.obstacle.y)
        h = dx * dx + dy * dy - self.obstacle.safe_radius**2

        grad_h_x = np.array([2.0 * dx, 2.0 * dy, 0.0], dtype=np.float64)
        a = grad_h_x @ self.model.input_matrix(float(pose[2]))  # row vector in U
        b_nom = float(a @ u_nom + self.cfg.alpha * h)

        denom = float(a @ a)
        if b_nom >= 0.0 or denom <= self.cfg.epsilon:
            return SafetyResult(u_nom.copy(), u_nom.copy(), h, b_nom, False, 0.0)

        # Projection of u_nom onto a u + alpha h >= 0.
        correction = (-b_nom / denom) * a
        u_safe = u_nom + correction
        return SafetyResult(
            command=u_safe,
            nominal_command=u_nom.copy(),
            h=h,
            cbf_value_nominal=b_nom,
            active=True,
            correction_norm=float(np.linalg.norm(correction)),
        )
