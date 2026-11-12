"""Numerically validate the local scan-point LiDAR CBF implementation.

Checks performed:
  1. analytic gradient of psi=hdot+k0*h against PyTorch autograd;
  2. analytic gradient of the velocity-embedded terminal h_B;
  3. exact affine residual after the direct CBF, CBF-CD, and bCBF projections;
  4. finite outputs and actor-gradient connectivity.

This is a formulation/unit test.  It does not replace closed-loop MuJoCo tests.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from src.high_level_rl.laser_cbf_filters import (
    LaserScanCbfClosedForm,
    LaserScanCbfControlDynamicsClosedForm,
    LaserZeroBackupLseCbfClosedForm,
)


@dataclass(frozen=True)
class Tolerances:
    gradient: float = 5.0e-5
    residual: float = 5.0e-5


def _rotation(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack((torch.stack((c, -s)), torch.stack((s, c))))


def _psi_autograd(
    z: torch.Tensor,
    q: torch.Tensor,
    *,
    axis_x: float,
    axis_y: float,
    k0: float,
) -> torch.Tensor:
    p = z[:2]
    theta = z[2]
    u = z[3:]
    R = _rotation(theta)
    y = R.T @ (q - p)
    P = torch.diag(z.new_tensor((axis_x**-2, axis_y**-2)))
    J = z.new_tensor(((0.0, -1.0), (1.0, 0.0)))
    h = y @ P @ y - 1.0
    ydot = -u[:2] - u[2] * (J @ y)
    hdot = 2.0 * y @ P @ ydot
    return hdot + k0 * h


def _terminal_autograd(
    z: torch.Tensor,
    q: torch.Tensor,
    *,
    axis_x: float,
    axis_y: float,
    backup_margin: float,
    a_min: float,
    eps: float,
) -> torch.Tensor:
    p = z[:2]
    theta = z[2]
    R = _rotation(theta)
    y = R.T @ (q - p)
    speed = torch.sqrt(z[3].square() + z[4].square() + eps * eps) - eps
    margin = backup_margin + speed / a_min
    ax = axis_x + margin
    ay = axis_y + margin
    return y[0].square() / ax.square() + y[1].square() / ay.square() - 1.0


def validate_gradients(device: torch.device, dtype: torch.dtype, tol: Tolerances) -> None:
    filt = LaserZeroBackupLseCbfClosedForm(
        backup_samples=5,
        backup_rollout_substeps=4,
        hard_clip_output=False,
    )
    angles = torch.tensor((-1.0, -0.25, 0.45, 1.1), device=device, dtype=dtype)
    ranges = torch.tensor(((0.75, 0.62, 0.80, 0.70),), device=device, dtype=dtype)
    command = torch.tensor(((0.18, -0.07, 0.21),), device=device, dtype=dtype)
    points, _ = filt.scan_points(ranges, angles, 1, device=device, dtype=dtype)

    _, grad_psi = filt.current_psi_and_grad(points, command, filt.k0)
    z0 = torch.cat((torch.zeros(3, device=device, dtype=dtype), command[0])).requires_grad_(True)
    psi_errors = []
    for ray in range(points.shape[1]):
        jac = torch.autograd.functional.jacobian(
            lambda z: _psi_autograd(
                z,
                points[0, ray],
                axis_x=filt.axis_x,
                axis_y=filt.axis_y,
                k0=filt.k0,
            ),
            z0,
        )
        psi_errors.append(torch.max(torch.abs(jac - grad_psi[0, ray])).item())

    zt = torch.tensor((0.08, -0.04, 0.20, 0.16, -0.09, 0.12), device=device, dtype=dtype)
    a_min = torch.tensor(min(filt.a_tuple[:2]), device=device, dtype=dtype)
    _, grad_terminal = filt.terminal_h_and_grad_future(
        points,
        zt[None, :],
        backup_margin=filt.backup_margin,
        a_min=a_min,
    )
    terminal_errors = []
    zt_req = zt.clone().requires_grad_(True)
    for ray in range(points.shape[1]):
        jac = torch.autograd.functional.jacobian(
            lambda z: _terminal_autograd(
                z,
                points[0, ray],
                axis_x=filt.axis_x,
                axis_y=filt.axis_y,
                backup_margin=filt.backup_margin,
                a_min=float(a_min),
                eps=filt.eps,
            ),
            zt_req,
        )
        terminal_errors.append(torch.max(torch.abs(jac - grad_terminal[0, ray])).item())

    max_psi = max(psi_errors)
    max_terminal = max(terminal_errors)
    print(f"psi gradient max error:      {max_psi:.3e}")
    print(f"terminal h_B max error:      {max_terminal:.3e}")
    if max_psi > tol.gradient or max_terminal > tol.gradient:
        raise AssertionError("Analytic local-point gradient validation failed")
    if torch.linalg.norm(grad_terminal[0, :, 3:5]).item() <= 0.0:
        raise AssertionError("Terminal h_B has no translational-command gradient")

    z_rest = torch.zeros(1, 6, device=device, dtype=dtype)
    _, grad_rest = filt.terminal_h_and_grad_future(
        points,
        z_rest,
        backup_margin=filt.backup_margin,
        a_min=a_min,
    )
    smooth_speed_at_rest = (
        torch.sqrt(z_rest[:, 3].square() + z_rest[:, 4].square() + filt.eps**2)
        - filt.eps
    )
    if not torch.equal(smooth_speed_at_rest, torch.zeros_like(smooth_speed_at_rest)):
        raise AssertionError("Smoothed terminal speed is not exactly zero at rest")
    if not torch.equal(grad_rest[..., 3:5], torch.zeros_like(grad_rest[..., 3:5])):
        raise AssertionError("Terminal velocity gradient is not zero at rest")
    print("terminal stopping speed at rest: exactly zero")


def validate_projection(device: torch.device, dtype: torch.dtype, tol: Tolerances) -> None:
    batch = 6
    rays = 13
    angles = torch.linspace(-torch.pi, torch.pi, rays, device=device, dtype=dtype)
    ranges = torch.full((batch, rays), 5.0, device=device, dtype=dtype)
    # Close visible returns around the front and sides.
    ranges[:, rays // 2] = 0.24
    ranges[:, rays // 2 - 2] = 0.34
    ranges[:, rays // 2 + 2] = 0.34
    nominal = torch.tensor((0.30, 0.04, 0.25), device=device, dtype=dtype).repeat(batch, 1)
    nominal.requires_grad_(True)
    current = torch.tensor((0.22, 0.02, 0.12), device=device, dtype=dtype).repeat(batch, 1)
    previous = torch.zeros_like(current)
    history = torch.ones(batch, 1, device=device, dtype=dtype)

    filters = (
        LaserScanCbfClosedForm(hard_clip_output=False),
        LaserScanCbfControlDynamicsClosedForm(hard_clip_output=False),
        LaserZeroBackupLseCbfClosedForm(
            backup_samples=7,
            backup_rollout_substeps=4,
            hard_clip_output=False,
        ),
    )

    for filt in filters:
        if isinstance(filt, LaserScanCbfClosedForm):
            result = filt.filter(ranges, angles, nominal)
        else:
            result = filt.filter(ranges, angles, current, nominal, previous, history)
        if result.projection_feasible is None or result.projected_cbf_residual is None:
            raise AssertionError(f"{type(filt).__name__} did not return projection diagnostics")
        feasible = result.projection_feasible
        minimum = result.projected_cbf_residual[feasible].min().item() if torch.any(feasible) else float("nan")
        print(
            f"{type(filt).__name__:43s} active={int(result.active.sum())}/{batch} "
            f"feasible={int(feasible.sum())}/{batch} min_residual={minimum:.3e}"
        )
        if torch.any(result.projected_cbf_residual[feasible] < -tol.residual):
            raise AssertionError(f"{type(filt).__name__} projection violated its affine CBF constraint")
        if not torch.isfinite(result.command).all():
            raise AssertionError(f"{type(filt).__name__} produced non-finite commands")
        result.command.sum().backward(retain_graph=True)
        if nominal.grad is None or not torch.isfinite(nominal.grad).all():
            raise AssertionError(f"{type(filt).__name__} broke actor-gradient connectivity")
        nominal.grad.zero_()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    tol = Tolerances(
        gradient=5.0e-5 if dtype == torch.float64 else 2.0e-3,
        residual=5.0e-5 if dtype == torch.float64 else 2.0e-4,
    )
    validate_gradients(device, dtype, tol)
    validate_projection(device, dtype, tol)
    print("Local scan-point LiDAR CBF validation: PASS")


if __name__ == "__main__":
    main()
