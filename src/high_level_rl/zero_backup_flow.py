from __future__ import annotations

import torch

"""Exact closed-form zero-surrogate-backup flow and sensitivity.

Shared by the map-based (`bcbf_lse_cf_filter.py`) and LiDAR-based
(`laser_cbf_filters.py`) bCBF-LSE-CF safety filters. Both roll out the same
augmented command/pose dynamics under a zero backup surrogate input

    u_dot = -A u,   theta_dot = wz,   p_dot = R(theta) u_xy,

per ``bcbf_lse_cf_safe_rl_technical_details_v2.pdf`` Eq. (13)/(41). Only the
barrier evaluated along the rollout differs between the two filters (global
obstacle map vs. local LiDAR points); the flow and its sensitivity matrix
Phi_B = d z(t) / d z0 are identical.

Command and yaw evolve in closed form. Planar position is the integral of a
rotated, exponentially-decaying velocity, which has no elementary closed
form, so it is evaluated with fixed 8-point Gauss-Legendre quadrature. Because
that quadrature formula is a known, smooth function of z0, its exact
derivative is written out below directly instead of obtained through
`torch.func.vmap`/`jacrev` or explicit-Euler integration -- both of which
this module replaces.
"""

_GL8_NODES = (
    -0.9602898564975363,
    -0.7966664774136267,
    -0.5255324099163290,
    -0.1834346424956498,
    0.1834346424956498,
    0.5255324099163290,
    0.7966664774136267,
    0.9602898564975363,
)
_GL8_WEIGHTS = (
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763,
)


def exact_zero_backup_state_and_phi(
    z0: torch.Tensor,
    times: torch.Tensor,
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact zero-surrogate backup flow and its Jacobian, at several times.

    Args:
        z0: ``(B, 6)`` initial augmented state ``[x, y, yaw, vx, vy, wz]``.
        times: ``(K,)`` absolute rollout times, each measured from tau=0.
        a: ``(3,)`` command decay rates.

    Returns:
        z_t: ``(B, K, 6)`` state at each requested time.
        phi_t: ``(B, K, 6, 6)`` Jacobian ``d z_t / d z0``.
    """
    device, dtype = z0.device, z0.dtype
    batch = z0.shape[0]
    num_times = times.shape[0]

    p0 = z0[:, :2]
    yaw0 = z0[:, 2]
    u0 = z0[:, 3:]
    vx0, vy0, wz0 = u0[:, 0], u0[:, 1], u0[:, 2]

    ax, ay, awz = a[0], a[1], a[2]

    decay_t = torch.exp(-a[None, :] * times[:, None])  # (K, 3)
    u_t = decay_t[None, :, :] * u0[:, None, :]  # (B, K, 3)

    c2_t = (1.0 - torch.exp(-awz * times)) / awz  # (K,)
    yaw_t = yaw0[:, None] + u0[:, 2:3] * c2_t[None, :]  # (B, K)

    nodes = torch.tensor(_GL8_NODES, device=device, dtype=dtype)
    weights = torch.tensor(_GL8_WEIGHTS, device=device, dtype=dtype)

    tau = 0.5 * times[:, None] * (nodes[None, :] + 1.0)  # (K, 8)
    quad_w = (0.5 * times[:, None] * weights[None, :])[None, :, :]  # (1, K, 8)

    exp_ax_tau = torch.exp(-ax * tau)[None, :, :]  # (1, K, 8)
    exp_ay_tau = torch.exp(-ay * tau)[None, :, :]
    c2_tau = ((1.0 - torch.exp(-awz * tau)) / awz)[None, :, :]  # (1, K, 8)

    vx_tau = vx0[:, None, None] * exp_ax_tau  # (B, K, 8)
    vy_tau = vy0[:, None, None] * exp_ay_tau
    yaw_tau = yaw0[:, None, None] + wz0[:, None, None] * c2_tau  # (B, K, 8)

    c = torch.cos(yaw_tau)
    s = torch.sin(yaw_tau)
    x_dot = c * vx_tau - s * vy_tau
    y_dot = s * vx_tau + c * vy_tau

    delta_px = torch.sum(quad_w * x_dot, dim=-1)  # (B, K)
    delta_py = torch.sum(quad_w * y_dot, dim=-1)
    p_t = p0[:, None, :] + torch.stack((delta_px, delta_py), dim=-1)  # (B, K, 2)

    z_t = torch.cat((p_t, yaw_t[..., None], u_t), dim=-1)  # (B, K, 6)

    # --- Exact Jacobian of the quadrature-based flow map, w.r.t. z0 ---
    dxdot_dyaw0 = -s * vx_tau - c * vy_tau
    dydot_dyaw0 = c * vx_tau - s * vy_tau
    dxdot_dwz0 = dxdot_dyaw0 * c2_tau
    dydot_dwz0 = dydot_dyaw0 * c2_tau
    dxdot_dvx0 = c * exp_ax_tau
    dydot_dvx0 = s * exp_ax_tau
    dxdot_dvy0 = -s * exp_ay_tau
    dydot_dvy0 = c * exp_ay_tau

    dpx_dyaw0 = torch.sum(quad_w * dxdot_dyaw0, dim=-1)  # (B, K)
    dpy_dyaw0 = torch.sum(quad_w * dydot_dyaw0, dim=-1)
    dpx_dvx0 = torch.sum(quad_w * dxdot_dvx0, dim=-1)
    dpy_dvx0 = torch.sum(quad_w * dydot_dvx0, dim=-1)
    dpx_dvy0 = torch.sum(quad_w * dxdot_dvy0, dim=-1)
    dpy_dvy0 = torch.sum(quad_w * dydot_dvy0, dim=-1)
    dpx_dwz0 = torch.sum(quad_w * dxdot_dwz0, dim=-1)
    dpy_dwz0 = torch.sum(quad_w * dydot_dwz0, dim=-1)

    phi_t = torch.zeros(batch, num_times, 6, 6, device=device, dtype=dtype)
    phi_t[:, :, 0, 0] = 1.0
    phi_t[:, :, 1, 1] = 1.0
    phi_t[:, :, 0, 2] = dpx_dyaw0
    phi_t[:, :, 1, 2] = dpy_dyaw0
    phi_t[:, :, 0, 3] = dpx_dvx0
    phi_t[:, :, 1, 3] = dpy_dvx0
    phi_t[:, :, 0, 4] = dpx_dvy0
    phi_t[:, :, 1, 4] = dpy_dvy0
    phi_t[:, :, 0, 5] = dpx_dwz0
    phi_t[:, :, 1, 5] = dpy_dwz0
    phi_t[:, :, 2, 2] = 1.0
    phi_t[:, :, 2, 5] = c2_t[None, :]
    phi_t[:, :, 3, 3] = decay_t[None, :, 0]
    phi_t[:, :, 4, 4] = decay_t[None, :, 1]
    phi_t[:, :, 5, 5] = decay_t[None, :, 2]

    return z_t, phi_t
