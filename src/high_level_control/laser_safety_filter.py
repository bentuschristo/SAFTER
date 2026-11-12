from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class LaserSafetyResult:
    command: np.ndarray
    nominal_command: np.ndarray
    h: float
    cbf_value_nominal: float
    active: bool
    correction_norm: float
    lambda_value: float = 0.0


def make_ray_angles(num_rays: int, fov_deg: float) -> np.ndarray:
    if num_rays <= 1:
        return np.zeros(1, dtype=np.float64)
    fov = math.radians(float(fov_deg))
    return np.linspace(-0.5 * fov, 0.5 * fov, int(num_rays), dtype=np.float64)


def _softmin_value_grad(terms: np.ndarray, grads: np.ndarray, kappa: float) -> tuple[float, np.ndarray]:
    scaled = -float(kappa) * terms
    zmax = float(np.max(scaled))
    weights = np.exp(scaled - zmax)
    denom = float(np.sum(weights))
    if denom <= 0.0 or not np.isfinite(denom):
        idx = int(np.argmin(terms))
        return float(terms[idx]), np.asarray(grads[idx], dtype=np.float64).copy()
    w = weights / denom
    value = -(math.log(denom) + zmax) / float(kappa)
    grad = np.sum(w[:, None] * grads, axis=0)
    return float(value), grad


class LaserScanCbfClosedForm:
    """Deployment closed-form direct laser/ray CBF.

    h_i = r_i - d_safe
    [-cos(beta_i), -sin(beta_i), 0] u + alpha h_i >= 0
    """

    def __init__(
        self,
        *,
        angles: np.ndarray,
        d_safe: float = 0.30,
        alpha: float = 1.5,
        kappa: float = 30.0,
        limits: tuple[float, float, float] = (0.5, 0.3, 0.5),
        epsilon: float = 1.0e-8,
        hard_clip_output: bool = True,
    ) -> None:
        self.angles = np.asarray(angles, dtype=np.float64)
        self.d_safe = float(d_safe)
        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self.limits = np.asarray(limits, dtype=np.float64)
        self.epsilon = float(epsilon)
        self.hard_clip_output = bool(hard_clip_output)
        self.ray_rows = np.column_stack(
            (-np.cos(self.angles), -np.sin(self.angles), np.zeros_like(self.angles))
        )

    def filter(self, ranges: np.ndarray, nominal_command: np.ndarray) -> LaserSafetyResult:
        r = np.asarray(ranges, dtype=np.float64).reshape(-1)
        u_nom = np.asarray(nominal_command, dtype=np.float64).reshape(3)
        if r.shape[0] != self.ray_rows.shape[0]:
            raise ValueError(f"ranges has {r.shape[0]} rays but angles has {self.ray_rows.shape[0]}")
        h_i = r - self.d_safe
        scaled = -self.kappa * h_i
        zmax = float(np.max(scaled))
        w = np.exp(scaled - zmax)
        w = w / float(np.sum(w))
        h = -(math.log(float(np.sum(np.exp(scaled - zmax)))) + zmax) / self.kappa
        a = w @ self.ray_rows
        omega = float(a @ u_nom + self.alpha * h)
        denom = float(a @ a)
        if omega >= 0.0 or denom <= self.epsilon:
            u = u_nom.copy()
            lam = 0.0
            active = False
            corr = np.zeros(3, dtype=np.float64)
        else:
            lam = -omega / max(denom, self.epsilon)
            corr = lam * a
            u = u_nom + corr
            active = True
        if self.hard_clip_output:
            u = np.clip(u, -self.limits, self.limits)
        return LaserSafetyResult(u, u_nom.copy(), float(h), omega, active, float(np.linalg.norm(corr)), float(lam))


class LaserZeroBackupLseCbfClosedForm:
    """Deployment laser version of the v4 closed-form zero-backup bCBF-CD.

    This is the NumPy counterpart of ``src.high_level_rl.laser_cbf_filters``.
    It uses the current LiDAR scan as the barrier reference and keeps an
    internal actual-command state u.  The terminal term is directional:

        h_B,i = r_i(T)-d_safe-d_B
                -(vx_T cos beta_i + vy_T sin beta_i)/a_min.
    """

    def __init__(
        self,
        *,
        angles: np.ndarray,
        d_safe: float = 0.30,
        backup_margin: float = 0.30,
        backup_horizon_s: float = 4.0,
        backup_samples: int = 21,
        hocbf_k0: float = 2.0,
        kappa_backup: float = 50.0,
        kappa_input: float = 20.0,
        kappa_outer: float = 20.0,
        alpha_outer: float = 0.8,
        limits: tuple[float, float, float] = (0.5, 0.3, 0.5),
        control_dynamics_a: tuple[float, float, float] = (8.0, 8.0, 8.0),
        tracking_sigma: tuple[float, float, float] = (8.0, 8.0, 8.0),
        dt: float = 0.10,
        epsilon: float = 1.0e-8,
        hard_clip_output: bool = True,
    ) -> None:
        self.angles = np.asarray(angles, dtype=np.float64)
        self.cos_b = np.cos(self.angles)
        self.sin_b = np.sin(self.angles)
        self.d_safe = float(d_safe)
        self.backup_margin = float(backup_margin)
        self.backup_horizon_s = float(backup_horizon_s)
        self.backup_samples = int(backup_samples)
        self.k0 = float(hocbf_k0)
        self.kappa_backup = float(kappa_backup)
        self.kappa_input = float(kappa_input)
        self.kappa_outer = float(kappa_outer)
        self.alpha_outer = float(alpha_outer)
        self.limits = np.asarray(limits, dtype=np.float64)
        self.a = np.asarray(control_dynamics_a, dtype=np.float64)
        self.sigma = np.asarray(tracking_sigma, dtype=np.float64)
        self.dt = float(dt)
        self.epsilon = float(epsilon)
        self.hard_clip_output = bool(hard_clip_output)
        self.initialized = False
        self.nominal_history_valid = False
        self.u_state = np.zeros(3, dtype=np.float64)
        self.ud_prev = np.zeros(3, dtype=np.float64)

    def reset(self, command: np.ndarray | None = None) -> None:
        self.initialized = False
        self.nominal_history_valid = False
        if command is not None:
            u = np.asarray(command, dtype=np.float64).reshape(3)
            self.u_state = np.clip(u, -self.limits, self.limits)
            self.ud_prev = u.copy()
            self.initialized = True

    def _backup_terms_and_grads(self, ranges: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(ranges, dtype=np.float64).reshape(-1)
        if r.shape[0] != self.angles.shape[0]:
            raise ValueError(f"ranges has {r.shape[0]} rays but angles has {self.angles.shape[0]}")
        vx, vy = float(u[0]), float(u[1])
        ax, ay = float(self.a[0]), float(self.a[1])
        a_min = max(min(ax, ay), self.epsilon)
        terms = []
        grads = []

        # tau=0 HOCBF lift.
        psi0 = -(self.cos_b * vx + self.sin_b * vy) + self.k0 * (r - self.d_safe)
        grad0 = np.zeros((r.shape[0], 6), dtype=np.float64)
        grad0[:, 3] = -self.cos_b
        grad0[:, 4] = -self.sin_b
        terms.append(psi0)
        grads.append(grad0)

        samples = max(self.backup_samples, 2)
        horizon = self.backup_horizon_s
        for sample in range(1, samples):
            t = horizon * float(sample) / float(samples - 1)
            ix = (1.0 - math.exp(-ax * t)) / ax
            iy = (1.0 - math.exp(-ay * t)) / ay
            r_t = r - self.cos_b * vx * ix - self.sin_b * vy * iy
            h_t = r_t - self.d_safe
            grad_t = np.zeros((r.shape[0], 6), dtype=np.float64)
            grad_t[:, 3] = -self.cos_b * ix
            grad_t[:, 4] = -self.sin_b * iy
            terms.append(h_t)
            grads.append(grad_t)

        T = horizon
        ixT = (1.0 - math.exp(-ax * T)) / ax
        iyT = (1.0 - math.exp(-ay * T)) / ay
        exT = math.exp(-ax * T)
        eyT = math.exp(-ay * T)
        r_T = r - self.cos_b * vx * ixT - self.sin_b * vy * iyT
        vx_T = vx * exT
        vy_T = vy * eyT
        hB = r_T - self.d_safe - self.backup_margin - (self.cos_b * vx_T + self.sin_b * vy_T) / a_min
        gradB = np.zeros((r.shape[0], 6), dtype=np.float64)
        gradB[:, 3] = -self.cos_b * ixT - self.cos_b * exT / a_min
        gradB[:, 4] = -self.sin_b * iyT - self.sin_b * eyT / a_min
        terms.append(hB)
        grads.append(gradB)
        return np.concatenate(terms, axis=0), np.concatenate(grads, axis=0)

    def _input_lse(self, u: np.ndarray) -> tuple[float, np.ndarray]:
        terms = np.array(
            [
                self.limits[0] - u[0], u[0] + self.limits[0],
                self.limits[1] - u[1], u[1] + self.limits[1],
                self.limits[2] - u[2], u[2] + self.limits[2],
            ],
            dtype=np.float64,
        )
        grads = np.zeros((6, 6), dtype=np.float64)
        for j in range(3):
            grads[2 * j, 3 + j] = -1.0
            grads[2 * j + 1, 3 + j] = 1.0
        return _softmin_value_grad(terms, grads, self.kappa_input)

    def filter(self, ranges: np.ndarray, nominal_command: np.ndarray) -> LaserSafetyResult:
        ud = np.asarray(nominal_command, dtype=np.float64).reshape(3)
        if not self.initialized:
            self.u_state = np.clip(ud, -self.limits, self.limits)
            self.ud_prev = ud.copy()
            self.initialized = True
            self.nominal_history_valid = False
        u = self.u_state.copy()

        terms, grads = self._backup_terms_and_grads(ranges, u)
        h_backup, grad_backup = _softmin_value_grad(terms, grads, self.kappa_backup)
        h_input, grad_input = self._input_lse(u)
        h_aug, grad = _softmin_value_grad(
            np.array([h_backup, h_input], dtype=np.float64),
            np.stack([grad_backup, grad_input], axis=0),
            self.kappa_outer,
        )

        drift = np.zeros(6, dtype=np.float64)
        drift[3:] = -self.a * u
        lf = float(grad @ drift)
        lg = grad[3:] * self.a

        if self.nominal_history_valid:
            ud_dot = (ud - self.ud_prev) / max(self.dt, self.epsilon)
        else:
            ud_dot = np.zeros(3, dtype=np.float64)
        uhat_d = u + ud_dot / self.a + (self.sigma / self.a) * (ud - u)
        omega = float(lf + lg @ uhat_d + self.alpha_outer * h_aug)
        denom = float(lg @ lg)
        if omega < 0.0 and denom > self.epsilon:
            lam = -omega / denom
            uhat = uhat_d + lam * lg
            active = True
        elif omega < 0.0:
            lam = 0.0
            uhat = np.zeros(3, dtype=np.float64)
            active = True
        else:
            lam = 0.0
            uhat = uhat_d
            active = False

        decay = np.exp(-self.a * self.dt)
        u_next = decay * u + (1.0 - decay) * uhat
        if self.hard_clip_output:
            u_next = np.clip(u_next, -self.limits, self.limits)

        self.u_state = u_next.copy()
        self.ud_prev = ud.copy()
        self.nominal_history_valid = True
        correction = uhat - uhat_d
        return LaserSafetyResult(
            command=u_next,
            nominal_command=ud.copy(),
            h=float(h_aug),
            cbf_value_nominal=omega,
            active=active,
            correction_norm=float(np.linalg.norm(correction)),
            lambda_value=float(lam),
        )
