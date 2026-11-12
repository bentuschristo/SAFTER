"""Finite-difference verification of the analytic gradients in
_LocalScanPointGeometry (src/high_level_rl/laser_cbf_filters.py):

  * current_psi_and_grad          -- lifted current-time ellipse barrier
  * flow_h_and_grad_future        -- positive-time flow barrier
  * terminal_invariant_h_and_grad_future -- terminal invariant-ellipse pair
                                     (h_geom, h_rec) -- the mode actually
                                     used by every bcbf_cf/bcbf_lse_cd_qp
                                     checkpoint trained/evaluated this
                                     session (terminal_backup_mode=
                                     "invariant_ellipse").

For each function, perturbs every component of the relevant state
(position/yaw/velocity/yaw-rate as applicable) with central finite
differences and compares against the analytic gradient the function itself
returns. Reports the actual observed maximum error per function/component.

Run: python3 scripts/test_ellipse_terminal_gradients_fd.py
"""
from __future__ import annotations

import torch

from src.high_level_rl.laser_cbf_filters import _LocalScanPointGeometry

DOUBLE = torch.float64
FD_EPS = 1.0e-6
TOL = 1.0e-4


def _geom(dtype=DOUBLE):
    return _LocalScanPointGeometry(
        d_safe=0.30,
        robot_length_m=0.70,
        robot_width_m=0.31,
        lidar_x_m=0.28945,
        lidar_y_m=0.0,
        footprint_clearance_m=0.15,
        use_elliptic_footprint=True,
        max_range=5.0,
        no_return_tolerance_m=1.0e-4,
        inactive_barrier_value=20.0,
        eps=1.0e-8,
    )


def _random_points(batch: int, rays: int, dtype=DOUBLE) -> torch.Tensor:
    torch.manual_seed(1)
    # Points within ~1-4m of the origin in random directions -- comfortably
    # away from the ellipse boundary (axis ~0.5m) to avoid the log(E)
    # singularity in the invariant-ellipse test.
    angle = torch.rand(batch, rays, dtype=dtype) * 2 * torch.pi
    radius = 1.0 + 3.0 * torch.rand(batch, rays, dtype=dtype)
    return torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)


def _fd_grad(f, z0: torch.Tensor, component: int, eps: float = FD_EPS) -> torch.Tensor:
    z_plus = z0.clone()
    z_plus[:, component] += eps
    z_minus = z0.clone()
    z_minus[:, component] -= eps
    return (f(z_plus) - f(z_minus)) / (2 * eps)


def check_current_psi_and_grad(geom: _LocalScanPointGeometry, batch: int, rays: int, k0: float) -> float:
    """FD-check current_psi_and_grad's full 6-component gradient.

    current_psi_and_grad hardwires p=0, theta=0 (the scan-fixed frame is
    defined at the robot's current pose). To FD-check the position/yaw part
    of its gradient too (not just the velocity part it directly exposes),
    this uses the fact that shifting/rotating the *points* by (-p, -theta)
    is exactly equivalent to evaluating the barrier at state (p, theta):
    y(p, theta) = R(-theta)(q - p) is precisely the same transform
    flow_h_and_grad_future applies, and current_psi_and_grad's formula is
    that same y evaluated at p=0, theta=0. So a generalized wrapper that
    pre-transforms points by a probe (p, theta) and then calls
    current_psi_and_grad on the transformed points reproduces psi(p, theta,
    u) for arbitrary p, theta -- letting FD probe those two components too,
    not just velocity.
    """
    points = _random_points(batch, rays)
    u0 = (torch.rand(batch, 3, dtype=DOUBLE) - 0.5) * torch.tensor([1.0, 0.6, 1.0], dtype=DOUBLE)

    def psi_of_z(z: torch.Tensor) -> torch.Tensor:
        p = z[:, 0:2]
        theta = z[:, 2]
        u = z[:, 3:6]
        c, s = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
        rel = points - p[:, None, :]
        yx = c * rel[..., 0] + s * rel[..., 1]
        yy = -s * rel[..., 0] + c * rel[..., 1]
        y_shifted = torch.stack((yx, yy), dim=-1)
        psi, _ = geom.current_psi_and_grad(y_shifted, u, k0)
        return psi

    z0 = torch.cat((torch.zeros(batch, 3, dtype=DOUBLE), u0), dim=1)
    _, grad_analytic = geom.current_psi_and_grad(points, u0, k0)

    max_err = 0.0
    for comp in range(6):
        fd = _fd_grad(psi_of_z, z0, comp)
        err = (fd - grad_analytic[..., comp]).abs().max().item()
        max_err = max(max_err, err)
        print(f"    current_psi_and_grad component {comp}: max|FD-analytic| = {err:.3e}")
    return max_err


def check_flow_h_and_grad_future(geom: _LocalScanPointGeometry, batch: int, rays: int) -> float:
    points = _random_points(batch, rays)
    torch.manual_seed(2)
    p = (torch.rand(batch, 2, dtype=DOUBLE) - 0.5) * 2.0
    theta = (torch.rand(batch, dtype=DOUBLE) - 0.5) * torch.pi
    v = (torch.rand(batch, 3, dtype=DOUBLE) - 0.5) * torch.tensor([1.0, 0.6, 1.0], dtype=DOUBLE)
    z0 = torch.cat((p, theta[:, None], v), dim=1)

    def h_of_z(z: torch.Tensor) -> torch.Tensor:
        h, _ = geom.flow_h_and_grad_future(points, z)
        return h

    _, grad_analytic = geom.flow_h_and_grad_future(points, z0)

    max_err = 0.0
    for comp in range(6):
        fd = _fd_grad(h_of_z, z0, comp)
        err = (fd - grad_analytic[..., comp]).abs().max().item()
        max_err = max(max_err, err)
        print(f"    flow_h_and_grad_future component {comp}: max|FD-analytic| = {err:.3e}")
    return max_err


def check_terminal_invariant(geom: _LocalScanPointGeometry, batch: int, rays: int) -> float:
    points = _random_points(batch, rays)
    torch.manual_seed(3)
    p = (torch.rand(batch, 2, dtype=DOUBLE) - 0.5) * 2.0
    theta = (torch.rand(batch, dtype=DOUBLE) - 0.5) * torch.pi
    v = (torch.rand(batch, 3, dtype=DOUBLE) - 0.5) * torch.tensor([1.0, 0.6, 1.0], dtype=DOUBLE)
    z0 = torch.cat((p, theta[:, None], v), dim=1)

    big_r_x = geom.axis_x + 0.6
    big_r_y = geom.axis_y + 0.6
    c_theta = 0.5 * abs(big_r_x / big_r_y - big_r_y / big_r_x)
    a_decay_min = torch.tensor(8.0, dtype=DOUBLE)

    def h_geom_of_z(z: torch.Tensor) -> torch.Tensor:
        h_geom, _, _, _ = geom.terminal_invariant_h_and_grad_future(
            points, z, big_r_x=big_r_x, big_r_y=big_r_y, c_theta=c_theta, a_decay_min=a_decay_min
        )
        return h_geom

    def h_rec_of_z(z: torch.Tensor) -> torch.Tensor:
        _, _, h_rec, _ = geom.terminal_invariant_h_and_grad_future(
            points, z, big_r_x=big_r_x, big_r_y=big_r_y, c_theta=c_theta, a_decay_min=a_decay_min
        )
        return h_rec

    h_geom0, grad_geom, h_rec0, grad_rec = geom.terminal_invariant_h_and_grad_future(
        points, z0, big_r_x=big_r_x, big_r_y=big_r_y, c_theta=c_theta, a_decay_min=a_decay_min
    )

    max_err = 0.0
    for comp in range(6):
        fd_geom = _fd_grad(h_geom_of_z, z0, comp)
        err_geom = (fd_geom - grad_geom[..., comp]).abs().max().item()
        fd_rec = _fd_grad(h_rec_of_z, z0, comp)
        err_rec = (fd_rec - grad_rec[..., comp]).abs().max().item()
        max_err = max(max_err, err_geom, err_rec)
        print(
            f"    terminal_invariant component {comp}: "
            f"h_geom max|FD-analytic| = {err_geom:.3e}, h_rec max|FD-analytic| = {err_rec:.3e}"
        )
    return max_err


def main() -> None:
    geom = _geom()
    batch, rays = 32, 16

    print("=== current_psi_and_grad (k0=2.0) ===")
    err1 = check_current_psi_and_grad(geom, batch, rays, k0=2.0)

    print("\n=== flow_h_and_grad_future ===")
    err2 = check_flow_h_and_grad_future(geom, batch, rays)

    print("\n=== terminal_invariant_h_and_grad_future ===")
    err3 = check_terminal_invariant(geom, batch, rays)

    print(f"\nmax error: current_psi_and_grad = {err1:.3e}")
    print(f"max error: flow_h_and_grad_future = {err2:.3e}")
    print(f"max error: terminal_invariant_h_and_grad_future = {err3:.3e}")

    overall = max(err1, err2, err3)
    print(f"\noverall max FD-vs-analytic error: {overall:.3e} (tol {TOL:.1e})")
    assert overall < TOL, "analytic gradient disagrees with finite differences beyond tolerance"
    print("All gradients verified correct.")


if __name__ == "__main__":
    main()
