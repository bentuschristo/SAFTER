from __future__ import annotations

from dataclasses import dataclass

import torch

from .zero_backup_flow import exact_zero_backup_state_and_phi


def _asymmetric_limits(
    max_vx: float, max_vy: float, max_wz: float, min_vx: float | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Per-axis (lower, upper) admissible-command bounds for a safety filter.

    vy/wz stay symmetric (+-max_vy/+-max_wz); vx alone can have an
    independent lower bound (min_vx), matching
    SAFTER_src/high_level_laser_sac/networks.py's actor-side asymmetric
    action range. The filter's own admissible input set must match what the
    frozen low-level locomotion policy was actually trained to track
    (e.g. vx in [-1.0, 2.0], not a wider or narrower box of the filter's
    own invention) -- commanding a "safe" velocity the low-level policy
    never learned to track is not actually safe in practice, regardless of
    what the high-level CBF proof says. min_vx=None reproduces the
    original symmetric [-max_vx, max_vx] box exactly, so every existing
    caller that never passes min_vx is unaffected.
    """
    effective_min_vx = -float(max_vx) if min_vx is None else float(min_vx)
    lower = (effective_min_vx, -float(max_vy), -float(max_wz))
    upper = (float(max_vx), float(max_vy), float(max_wz))
    return lower, upper


@dataclass
class LaserSafetyFilterResult:
    command: torch.Tensor
    h_lse: torch.Tensor
    individual_h: torch.Tensor
    active: torch.Tensor
    correction_norm: torch.Tensor
    cbf_value_nominal: torch.Tensor
    lambda_value: torch.Tensor
    surrogate_command: torch.Tensor | None = None
    desired_surrogate_command: torch.Tensor | None = None
    h_backup_lse: torch.Tensor | None = None
    h_input_lse: torch.Tensor | None = None
    mu_value: torch.Tensor | None = None
    input_bound_violation: torch.Tensor | None = None
    input_bound_active: torch.Tensor | None = None
    projection_feasible: torch.Tensor | None = None
    projected_cbf_residual: torch.Tensor | None = None
    # D_aug = ||Lg_hat h_aug||^2 (the closed-form projection's denominator).
    # Populated only by LaserZeroBackupLseCbfClosedForm.filter() -- purely
    # additive/diagnostic, does not affect `command` or any other field.
    lg_aug_sq_norm: torch.Tensor | None = None
    # Pre-hard-clip command (decay*current + (1-decay)*surrogate, before the
    # optional torch.clamp to the action limits). Populated only by
    # LaserZeroBackupLseCbfClosedForm.filter() -- diagnostic only, so
    # ||command - command_preclip|| can be checked without changing what
    # `command` itself is.
    command_preclip: torch.Tensor | None = None
    # --- ZOH-consistent joint box+half-space projection diagnostics (2026-09
    # hard-clip-consistency fix). All populated only by
    # LaserZeroBackupLseCbfClosedForm.filter(); purely additive/diagnostic. ---
    # True where the ZOH box and the half-space had an EMPTY intersection, so
    # the residual-maximizing box corner (u_maxsafe) was returned instead of a
    # true joint-feasible point. Expected to be all-False in a healthy run.
    joint_projection_infeasible: torch.Tensor | None = None
    # ||command - command_preclip||: how much the final hard clip (Task 4's
    # failsafe, kept per spec) still had to move the command. Should be ~0
    # once the box constraint is already enforced inside the projection.
    clip_delta: torch.Tensor | None = None
    clip_changed: torch.Tensor | None = None
    # Diagnostic-only recovery of the surrogate command that would exactly
    # reproduce the ACTUALLY EXECUTED (post-clip) command through the ZOH
    # update, and the CBF residual evaluated at that surrogate. Verification
    # only -- never used to alter the filter's own decision.
    residual_exec: torch.Tensor | None = None


def _as_batched_ranges(ranges: torch.Tensor) -> torch.Tensor:
    if ranges.ndim == 1:
        return ranges.unsqueeze(0)
    if ranges.ndim != 2:
        raise ValueError(f"ranges must have shape [B,R] or [R], got {tuple(ranges.shape)}")
    return ranges


def _as_command_batched_ranges(ranges: torch.Tensor, batch_size: int) -> torch.Tensor:
    r = _as_batched_ranges(ranges)
    if r.shape[0] == batch_size:
        return r
    if r.shape[0] == 1:
        return r.expand(batch_size, -1)
    raise ValueError(f"ranges batch has {r.shape[0]} rows but command batch has {batch_size}")


def _softmin_value_grad(
    terms: torch.Tensor,
    grads: torch.Tensor,
    kappa: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable LSE soft-min and weighted gradient.

    terms: [B,M]
    grads: [B,M,D]
    """
    scaled = -float(kappa) * terms
    weights = torch.softmax(scaled, dim=1)
    value = -torch.logsumexp(scaled, dim=1) / float(kappa)
    grad = torch.sum(weights[..., None] * grads, dim=1)
    return value, grad


def _softmin_value_lie(
    terms: torch.Tensor,
    lf_terms: torch.Tensor,
    lg_terms: torch.Tensor,
    kappa: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable LSE soft-min with Lie derivatives of the composite.

    terms:    [B,M]
    lf_terms: [B,M]
    lg_terms: [B,M,U]
    """
    scaled = -float(kappa) * terms
    weights = torch.softmax(scaled, dim=1)
    value = -torch.logsumexp(scaled, dim=1) / float(kappa)
    lf = torch.sum(weights * lf_terms, dim=1)
    lg = torch.sum(weights[..., None] * lg_terms, dim=1)
    return value, lf, lg


def _exact_projection(
    desired: torch.Tensor,
    lg: torch.Tensor,
    drift_plus_alpha_h: torch.Tensor,
    *,
    eps: float,
    fallback: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact Euclidean projection onto one affine half-space.

    Constraint:
        drift_plus_alpha_h + lg @ u >= 0.

    A violated constraint with ||lg||^2 <= eps is marked infeasible and the
    supplied fallback is returned.  No denominator damping is used in the
    feasible case, so the projected affine residual is zero up to roundoff.
    """
    omega = drift_plus_alpha_h + torch.sum(lg * desired, dim=1)
    denom = torch.sum(lg.square(), dim=1)
    needs = omega < 0.0
    nondegenerate = denom > float(eps)
    feasible = (~needs) | nondegenerate

    lam = torch.where(
        needs & nondegenerate,
        -omega / torch.where(nondegenerate, denom, torch.ones_like(denom)),
        torch.zeros_like(omega),
    )
    projected = desired + lam[:, None] * lg
    projected = torch.where((needs & (~nondegenerate))[:, None], fallback, projected)
    residual = drift_plus_alpha_h + torch.sum(lg * projected, dim=1)
    return projected, omega, lam, feasible, residual


def zoh_surrogate_bounds(
    current_command: torch.Tensor,
    limits_lower: torch.Tensor,
    limits_upper: torch.Tensor,
    a: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact ZOH-consistent surrogate bounds (2026-09-07 hard-clip-consistency
    fix task 1).

    Physical first-order command dynamics u_dot = -A u + A u_hat, held
    constant over one high-level interval dt, give in closed form
        u_next = Lambda u + (1 - Lambda) u_hat,   Lambda = exp(-A dt).
    Requiring u_min <= u_next <= u_max and inverting (A, hence Lambda, is
    diagonal, so this is elementwise) gives
        u_hat_lower = (u_min - Lambda u) / (1 - Lambda)
        u_hat_upper = (u_max - Lambda u) / (1 - Lambda).
    These are NOT the physical [u_min, u_max] bounds themselves -- do not
    conflate the two (that was the original bug: hard-clipping u_next to
    [u_min, u_max] after the fact is not equivalent to bounding u_hat this
    way, which is why the composite CBF residual could go negative).

    ``1 - Lambda`` is computed as ``-expm1(-a*dt)`` rather than ``1 -
    exp(-a*dt)`` for numerical stability (avoids catastrophic cancellation
    when a*dt is small); for the current parameters (a=8, dt=0.1) Lambda ~=
    0.449329, 1-Lambda ~= 0.550671, matching the closed-form values quoted
    in the task notes.

    Args:
        current_command: ``(B,3)`` u_t (the augmented state's velocity part).
        limits_lower: ``(3,)`` u_min. May differ from -limits_upper (e.g. vx
            with an independent, asymmetric min_vx); vy/wz are still
            symmetric by convention at the caller.
        limits_upper: ``(3,)`` u_max.
        a: ``(3,)`` control_dynamics_a.
        dt: integration_dt (the same value used for `decay` elsewhere).

    Returns:
        ``(lower, upper, decay)``, each ``(B,3)``/``(3,)``/``(3,)``.
    """
    decay = torch.exp(-a * dt)
    one_minus_decay = -torch.expm1(-a * dt)
    lower = (limits_lower[None, :] - decay[None, :] * current_command) / one_minus_decay[None, :]
    upper = (limits_upper[None, :] - decay[None, :] * current_command) / one_minus_decay[None, :]
    return lower, upper, decay


_BOX_HALFSPACE_ACTIVE_SET_STATES = tuple(
    (i, j, k) for i in range(3) for j in range(3) for k in range(3)
)  # 3^3=27: each of the 3 dims is 0=free, 1=lower-active, 2=upper-active.

# Precomputed once (a fixed constant, independent of any call's B/device/
# dtype) and cached per (device, dtype) the first time each is seen, so the
# hot filter() path never repeats 27 small host->device tensor constructions
# per control step -- keeps this on the same "no unnecessary per-step
# overhead" footing as the rest of this filter.
_BOX_HALFSPACE_STATE_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _box_halfspace_state_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    cached = _BOX_HALFSPACE_STATE_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(_BOX_HALFSPACE_ACTIVE_SET_STATES, device=device, dtype=dtype)
        _BOX_HALFSPACE_STATE_CACHE[key] = cached
    return cached


def _exact_box_halfspace_projection(
    desired: torch.Tensor,
    lg: torch.Tensor,
    drift_plus_alpha_h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    eps: float,
    tol: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact Euclidean projection of `desired` onto

        {u : drift_plus_alpha_h + lg @ u >= 0}  INTERSECT  {u : lower<=u<=upper}

    (one affine half-space and a 3-D axis-aligned box), by finite active-set
    enumeration -- NOT an iterative/generic QP solver (2026-09-07 joint
    box+half-space projection fix, task 2).

    Algorithm (m=3, so at most 3^3=27 candidates plus the box-only point):
      1. Box-project `desired`; if it already satisfies the half-space,
         it IS the exact solution (convexity: the box projection is the
         closest point in the box, and it's also in the half-space).
      2. Otherwise the optimum lies on the half-space boundary with some
         subset of the 3 dims pinned to a bound. Enumerate all 27
         (free/lower/upper) assignments. For each: dims pinned to a bound
         take that value; free dims solve the LINEAR equality
         `drift_plus_alpha_h + lg@u = 0` in closed form (least-squares over
         only the free dims, since fixing the bound dims reduces this to a
         single scalar equation in the remaining free coordinates). A
         candidate is accepted only if (a) every free dim's solved value
         stays within its own [lower, upper], and (b) the resulting point
         satisfies the half-space to within `tol` (this second check also
         correctly captures the "0 free dims" / box-corner candidates,
         and the degenerate-gradient sub-case where the free dims can't
         move the residual at all -- see the two branches below).
      3. If a valid candidate exists, return the one with minimum
         ||u-desired||^2 (guaranteed to include the true optimum, since its
         exact active set is among the 27 enumerated combinations, and any
         OTHER accepted candidate is by construction primal-feasible, so
         the true global optimum -- unique by strict convexity -- has the
         smallest objective among all accepted candidates).
      4. If NO candidate is valid, the box and half-space do not intersect
         for that sample (JOINT_PROJECTION_INFEASIBLE): return the box
         point that maximizes the affine residual instead (still inside
         the box; the closest thing to "CBF feasible" achievable under the
         physical bound), and flag it.

    Returns:
        ``(projected, omega, lam, feasible, residual, joint_infeasible)``:
          - projected ``(B,3)``: the exact projection (or the box point
            maximizing the affine residual, if infeasible).
          - omega ``(B,)``: drift_plus_alpha_h + lg@desired (unprojected
            residual -- same meaning as in `_exact_projection`).
          - lam ``(B,)``: the half-space multiplier at the CHOSEN candidate
            (0 where the half-space constraint is inactive there).
          - feasible ``(B,)`` bool: True unless the box/half-space
            intersection is empty.
          - residual ``(B,)``: drift_plus_alpha_h + lg@projected.
          - joint_infeasible ``(B,)`` bool: True exactly where feasible is
            False (kept as a separate return for a clearer diagnostic name
            at the call site).
    """
    B = desired.shape[0]
    device, dtype = desired.device, desired.dtype
    omega = drift_plus_alpha_h + torch.sum(lg * desired, dim=1)

    u_box = torch.clamp(desired, lower, upper)
    box_residual = drift_plus_alpha_h + torch.sum(lg * u_box, dim=1)
    box_only_ok = box_residual >= -tol

    best = u_box.clone()
    best_lam = torch.zeros_like(omega)
    best_obj = torch.where(
        box_only_ok,
        torch.sum((u_box - desired).square(), dim=1),
        torch.full((B,), float("inf"), device=device, dtype=dtype),
    )

    state_tensor = _box_halfspace_state_tensor(device, torch.long)  # (27, 3), cached
    for row in range(state_tensor.shape[0]):
        state_t = state_tensor[row]
        free_col = (state_t == 0).to(dtype)[None, :].expand(B, 3)
        lower_col = (state_t == 1).to(dtype)[None, :].expand(B, 3)
        upper_col = (state_t == 2).to(dtype)[None, :].expand(B, 3)
        bound_col = lower_col + upper_col
        fixed_val = lower_col * lower + upper_col * upper

        sum_a2_free = torch.sum((lg * free_col).square(), dim=1)
        nondegenerate = sum_a2_free > eps

        # Non-degenerate branch: solve the scalar equality for the free dims.
        rhs = -(
            drift_plus_alpha_h
            + torch.sum(lg * bound_col * fixed_val, dim=1)
            + torch.sum(lg * free_col * desired, dim=1)
        )
        lam_eq = rhs / torch.where(nondegenerate, sum_a2_free, torch.ones_like(sum_a2_free))
        u_free_eq = desired + lam_eq[:, None] * lg

        # Degenerate branch (free dims' gradient ~0, can't move the
        # residual): closest-in-box value for the free dims; feasibility
        # is then checked directly below rather than assumed.
        u_free_degenerate = torch.clamp(desired, lower, upper)

        u_free = torch.where(nondegenerate[:, None], u_free_eq, u_free_degenerate)
        u_candidate = free_col * u_free + bound_col * fixed_val
        lam_candidate = torch.where(nondegenerate, lam_eq, torch.zeros_like(lam_eq))

        within_bounds = torch.all(
            (u_candidate >= lower - tol) & (u_candidate <= upper + tol), dim=1
        )
        candidate_residual = drift_plus_alpha_h + torch.sum(lg * u_candidate, dim=1)
        valid = within_bounds & (candidate_residual >= -tol)

        obj = torch.sum((u_candidate - desired).square(), dim=1)
        improves = valid & (obj < best_obj)
        best = torch.where(improves[:, None], u_candidate, best)
        best_lam = torch.where(improves, lam_candidate, best_lam)
        best_obj = torch.where(improves, obj, best_obj)

    # Task 3: box+half-space intersection is empty for these samples. Fall
    # back to the box point maximizing the affine residual -- do NOT hide
    # this with denominator/epsilon regularization. Computed unconditionally
    # (no `if torch.any(...)` host-device sync) so this stays a pure batched
    # GPU op with no per-control-step CPU<->GPU synchronization, matching the
    # no-per-step-sync discipline used everywhere else in this filter/the
    # training loop's diagnostic accumulators.
    joint_infeasible = ~torch.isfinite(best_obj)
    u_maxsafe = torch.where(lg >= 0.0, upper, lower)
    best = torch.where(joint_infeasible[:, None], u_maxsafe, best)
    best_lam = torch.where(joint_infeasible, torch.zeros_like(best_lam), best_lam)

    feasible = ~joint_infeasible
    residual = drift_plus_alpha_h + torch.sum(lg * best, dim=1)
    return best, omega, best_lam, feasible, residual, joint_infeasible


class _LocalScanPointGeometry:
    """Fixed analytic geometry shared by all local LiDAR CBF variants.

    At each filter call the current scan defines a temporary scan-fixed frame.
    Every valid ray endpoint is converted to a point q_i in that frame.  The
    points stay fixed during the current CBF/bCBF calculation, while the robot
    pose evolves relative to them.  No global obstacle map is used.
    """

    def __init__(
        self,
        *,
        d_safe: float,
        robot_length_m: float,
        robot_width_m: float,
        lidar_x_m: float,
        lidar_y_m: float,
        footprint_clearance_m: float | None,
        use_elliptic_footprint: bool,
        max_range: float,
        no_return_tolerance_m: float,
        inactive_barrier_value: float,
        eps: float,
    ) -> None:
        self.d_safe = float(d_safe)
        self.robot_length_m = float(robot_length_m)
        self.robot_width_m = float(robot_width_m)
        self.lidar_x_m = float(lidar_x_m)
        self.lidar_y_m = float(lidar_y_m)
        self.use_elliptic_footprint = bool(use_elliptic_footprint)
        self.max_range = float(max_range)
        self.no_return_tolerance_m = float(no_return_tolerance_m)
        self.inactive_barrier_value = float(inactive_barrier_value)
        self.eps = float(eps)

        half_length = 0.5 * self.robot_length_m
        half_width = 0.5 * self.robot_width_m
        if footprint_clearance_m is None:
            # Backward-compatible interpretation of d_safe: desired threshold
            # along the front ray measured from the LiDAR origin.
            front_body_distance = max(half_length - self.lidar_x_m, self.eps)
            clearance = max(0.0, self.d_safe - front_body_distance)
        else:
            clearance = float(footprint_clearance_m)
        if clearance < 0.0:
            raise ValueError("footprint_clearance_m must be non-negative")
        self.footprint_clearance_m = clearance

        if self.use_elliptic_footprint:
            self.axis_x = half_length + clearance
            self.axis_y = half_width + clearance
        else:
            radius = max(half_length, half_width) + clearance
            self.axis_x = radius
            self.axis_y = radius
        if self.axis_x <= 0.0 or self.axis_y <= 0.0:
            raise ValueError("inflated footprint axes must be positive")

    def _constants(self, reference: torch.Tensor) -> tuple[torch.Tensor, ...]:
        device, dtype = reference.device, reference.dtype
        return (
            torch.as_tensor(self.axis_x, device=device, dtype=dtype),
            torch.as_tensor(self.axis_y, device=device, dtype=dtype),
            torch.as_tensor(self.lidar_x_m, device=device, dtype=dtype),
            torch.as_tensor(self.lidar_y_m, device=device, dtype=dtype),
        )

    def scan_points(
        self,
        ranges: torch.Tensor,
        angles: torch.Tensor,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r = _as_command_batched_ranges(ranges, batch_size).to(device=device, dtype=dtype)
        beta = angles.to(device=device, dtype=dtype)
        if r.shape[1] != beta.numel():
            raise ValueError(f"ranges has {r.shape[1]} rays but angles has {beta.numel()}")
        _, _, lidar_x, lidar_y = self._constants(r)
        directions = torch.stack((torch.cos(beta), torch.sin(beta)), dim=-1)
        points = r[..., None] * directions[None, :, :]
        points[..., 0] += lidar_x
        points[..., 1] += lidar_y
        valid = r < (self.max_range - self.no_return_tolerance_m)
        return points, valid

    def current_h_and_direct_row(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Primitive ellipse barrier and hdot row at p=0, theta=0.

        y_i=q_i in the current scan frame and
            h_i = y_i^T P y_i - 1.
        For body command u=[vx,vy,w],
            ydot = -v_xy - w J y,
        which yields hdot_i = a_i u.
        """
        ax, ay, _, _ = self._constants(points)
        px = 1.0 / (ax * ax)
        py = 1.0 / (ay * ay)
        x = points[..., 0]
        y = points[..., 1]
        h = px * x.square() + py * y.square() - 1.0
        cross = (py - px) * x * y  # y^T P J y
        row = torch.stack((-2.0 * px * x, -2.0 * py * y, -2.0 * cross), dim=-1)
        return h, row

    def current_psi_and_grad(
        self,
        points: torch.Tensor,
        current_command: torch.Tensor,
        k0: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lifted current barrier psi=hdot+k0*h and grad wrt z=[p,th,u]."""
        B, R, _ = points.shape
        ax, ay, _, _ = self._constants(points)
        px = 1.0 / (ax * ax)
        py = 1.0 / (ay * ay)
        x = points[..., 0]
        y = points[..., 1]
        vx = current_command[:, 0:1]
        vy = current_command[:, 1:2]
        wz = current_command[:, 2:3]

        h = px * x.square() + py * y.square() - 1.0
        cross = (py - px) * x * y
        hdot = -2.0 * (px * x * vx + py * y * vy) - 2.0 * wz * cross
        psi = hdot + float(k0) * h

        # Gradients wrt local point coordinate y=q-p in the robot frame.
        d = py - px
        grad_y_h_x = 2.0 * px * x
        grad_y_h_y = 2.0 * py * y
        grad_y_hdot_x = -2.0 * px * vx - 2.0 * wz * d * y
        grad_y_hdot_y = -2.0 * py * vy - 2.0 * wz * d * x
        gy_x = grad_y_hdot_x + float(k0) * grad_y_h_x
        gy_y = grad_y_hdot_y + float(k0) * grad_y_h_y

        grad = torch.zeros(B, R, 6, device=points.device, dtype=points.dtype)
        # At current scan frame R(theta)=I and dy/dp=-I.
        grad[..., 0] = -gy_x
        grad[..., 1] = -gy_y
        # dy/dtheta=-J y, so dpsi/dtheta=-grad_y^T J y.
        grad[..., 2] = gy_x * y - gy_y * x
        grad[..., 3] = -2.0 * px * x
        grad[..., 4] = -2.0 * py * y
        grad[..., 5] = -2.0 * cross
        return psi, grad

    def flow_h_and_grad_future(
        self,
        points: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Primitive barrier and gradient wrt the future augmented state."""
        B, R, _ = points.shape
        ax, ay, _, _ = self._constants(points)
        inv_x2 = 1.0 / (ax * ax)
        inv_y2 = 1.0 / (ay * ay)
        p = z[:, 0:2]
        theta = z[:, 2]
        c = torch.cos(theta)[:, None]
        s = torch.sin(theta)[:, None]
        rel_x = points[..., 0] - p[:, None, 0]
        rel_y = points[..., 1] - p[:, None, 1]
        yx = c * rel_x + s * rel_y
        yy = -s * rel_x + c * rel_y

        h = inv_x2 * yx.square() + inv_y2 * yy.square() - 1.0
        gyx = 2.0 * inv_x2 * yx
        gyy = 2.0 * inv_y2 * yy

        grad = torch.zeros(B, R, 6, device=points.device, dtype=points.dtype)
        # grad_p = -R(theta) grad_y.
        grad[..., 0] = -(c * gyx - s * gyy)
        grad[..., 1] = -(s * gyx + c * gyy)
        grad[..., 2] = gyx * yy - gyy * yx
        return h, grad

    def terminal_h_and_grad_future(
        self,
        points: torch.Tensor,
        z: torch.Tensor,
        *,
        backup_margin: float,
        a_min: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Velocity-embedded terminal backup barrier.

        The inflated semi-axes are enlarged by
            d_B + ||u_xy||/a_min.
        Therefore h_B depends explicitly on the translational command state and
        is relative degree one with respect to the augmented surrogate input.
        """
        B, R, _ = points.shape
        ax0, ay0, _, _ = self._constants(points)
        p = z[:, 0:2]
        theta = z[:, 2]
        vx = z[:, 3]
        vy = z[:, 4]
        # Smooth speed with exactly zero residual stopping distance at rest.
        speed = torch.sqrt(vx.square() + vy.square() + self.eps * self.eps) - self.eps
        stop_margin = float(backup_margin) + speed / a_min
        ax = ax0 + stop_margin
        ay = ay0 + stop_margin

        c = torch.cos(theta)[:, None]
        s = torch.sin(theta)[:, None]
        rel_x = points[..., 0] - p[:, None, 0]
        rel_y = points[..., 1] - p[:, None, 1]
        yx = c * rel_x + s * rel_y
        yy = -s * rel_x + c * rel_y

        inv_x2 = 1.0 / ax[:, None].square()
        inv_y2 = 1.0 / ay[:, None].square()
        h = inv_x2 * yx.square() + inv_y2 * yy.square() - 1.0
        gyx = 2.0 * inv_x2 * yx
        gyy = 2.0 * inv_y2 * yy

        grad = torch.zeros(B, R, 6, device=points.device, dtype=points.dtype)
        grad[..., 0] = -(c * gyx - s * gyy)
        grad[..., 1] = -(s * gyx + c * gyy)
        grad[..., 2] = gyx * yy - gyy * yx

        dh_dmargin = -2.0 * yx.square() / ax[:, None].pow(3) - 2.0 * yy.square() / ay[:, None].pow(3)
        smooth_speed = speed + self.eps
        grad[..., 3] = dh_dmargin * (vx / (a_min * smooth_speed))[:, None]
        grad[..., 4] = dh_dmargin * (vy / (a_min * smooth_speed))[:, None]
        return h, grad

    def terminal_invariant_h_and_grad_future(
        self,
        points: torch.Tensor,
        z: torch.Tensor,
        *,
        big_r_x: float,
        big_r_y: float,
        c_theta: float,
        a_decay_min: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward-invariant terminal backup-set pair (v6).

        Replaces the single velocity-inflated terminal ellipse
        ``terminal_h_and_grad_future`` with a FIXED enlarged ellipse
        (semi-axes ``big_r_x = r_x + backup_margin``,
        ``big_r_y = r_y + backup_margin``, no residual-velocity inflation)
        intersected with a separate residual-motion recoverability
        condition. See LaserSafetyConfig ``terminal_backup_mode=
        "invariant_ellipse"`` and the derivation this implements
        (2026-09-07 terminal-invariance fix).

        Geometric branch (dimensionless elliptical log-clearance):
            E_i   = y_x^2/big_r_x^2 + y_y^2/big_r_y^2
            g_i   = 0.5 log(E_i)
            h_B_geom,i = g_i

        Residual-motion recoverability branch:
            Q     = v_x^2/big_r_x^2 + v_y^2/big_r_y^2 + c_theta^2 omega^2
            h_B_rec,i  = g_i^2 - 2 Q / a_decay_min^2

        Both are dimensionless; no epsilon-based norm smoothing is used
        (in particular no ``sqrt(Q+eps^2)-eps`` substitute for ``sqrt(Q)``
        -- the two-function representation above encodes
        ``g_i >= sqrt(2Q)/a_decay_min`` without ever evaluating a
        nondifferentiable square root at Q=0). ``log(E_i)`` is evaluated
        with no numerical floor: E_i can only vanish if a LiDAR point
        coincides exactly with the terminal robot center, which the
        diagnostic suite checks for and reports rather than silently
        guarding against.

        Args:
            points: ``(B,R,2)`` LiDAR points in the scan-fixed frame.
            z: ``(B,6)`` terminal augmented state ``[p_x,p_y,theta,v_x,v_y,omega]``.
            big_r_x, big_r_y: fixed terminal semi-axes (config-level
                constants, precomputed once by the caller -- not
                recomputed per call).
            c_theta: ``0.5*|big_r_x/big_r_y - big_r_y/big_r_x|`` (also a
                config-level constant, precomputed once by the caller).
            a_decay_min: ``min(a_x,a_y,a_omega)`` scalar tensor.

        Returns:
            ``(h_geom, grad_geom, h_rec, grad_rec)`` with
            ``h_geom, h_rec: (B,R)`` and ``grad_geom, grad_rec: (B,R,6)``,
            gradients taken with respect to the TERMINAL state ``z``
            (chained through Phi_B by the caller).
        """
        p = z[:, 0:2]
        theta = z[:, 2]
        vx = z[:, 3]
        vy = z[:, 4]
        omega = z[:, 5]

        c = torch.cos(theta)[:, None]
        s = torch.sin(theta)[:, None]
        rel_x = points[..., 0] - p[:, None, 0]
        rel_y = points[..., 1] - p[:, None, 1]
        yx = c * rel_x + s * rel_y
        yy = -s * rel_x + c * rel_y

        inv_x2 = 1.0 / (big_r_x * big_r_x)
        inv_y2 = 1.0 / (big_r_y * big_r_y)
        big_e = inv_x2 * yx.square() + inv_y2 * yy.square()
        g = 0.5 * torch.log(big_e)

        # eta = P_B y / E; dg/dy = eta exactly (no factor of 2: the 1/E from
        # d(0.5 log E)/dE cancels the factor of 2 from dE/dy = 2 P_B y).
        eta_x = inv_x2 * yx / big_e
        eta_y = inv_y2 * yy / big_e

        grad_g = torch.zeros(*points.shape[:2], 6, device=points.device, dtype=points.dtype)
        grad_g[..., 0] = -(c * eta_x - s * eta_y)
        grad_g[..., 1] = -(s * eta_x + c * eta_y)
        grad_g[..., 2] = eta_x * yy - eta_y * yx
        # grad_g[..., 3:6] stay 0: g_i has no explicit velocity/yaw-rate
        # dependence (unlike the legacy velocity-inflated h_terminal).

        h_geom = g

        big_q = (
            vx.square() * inv_x2
            + vy.square() * inv_y2
            + (c_theta * c_theta) * omega.square()
        )
        a_decay_min_sq = a_decay_min * a_decay_min
        h_rec = g.square() - (2.0 / a_decay_min_sq) * big_q[:, None]

        grad_big_q = torch.zeros(z.shape[0], 6, device=z.device, dtype=z.dtype)
        grad_big_q[:, 3] = 2.0 * vx * inv_x2
        grad_big_q[:, 4] = 2.0 * vy * inv_y2
        grad_big_q[:, 5] = 2.0 * (c_theta * c_theta) * omega

        grad_rec = 2.0 * g[..., None] * grad_g - (2.0 / a_decay_min_sq) * grad_big_q[:, None, :]

        return h_geom, grad_g, h_rec, grad_rec

    def mask_terms(
        self,
        terms: torch.Tensor,
        valid: torch.Tensor,
        *derivatives: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        masked_terms = torch.where(
            valid,
            terms,
            torch.full_like(terms, self.inactive_barrier_value),
        )
        outputs: list[torch.Tensor] = [masked_terms]
        for derivative in derivatives:
            mask = valid
            while mask.ndim < derivative.ndim:
                mask = mask.unsqueeze(-1)
            outputs.append(torch.where(mask, derivative, torch.zeros_like(derivative)))
        return tuple(outputs)


class LaserScanCbfClosedForm(_LocalScanPointGeometry):
    """Direct local-point LiDAR CBF with exact one-constraint projection.

    Each current LiDAR endpoint is a fixed obstacle-boundary point in a
    temporary local scan frame.  The primitive barrier is the implicit
    inflated-footprint ellipse h_i=q_i^T P q_i-1.  The direct velocity command
    appears in hdot_i, including yaw sensitivity for a non-circular footprint.
    """

    def __init__(
        self,
        *,
        d_safe: float = 0.30,
        alpha: float = 1.5,
        kappa: float = 30.0,
        max_vx: float = 0.5,
        max_vy: float = 0.3,
        max_wz: float = 0.5,
        min_vx: float | None = None,
        robot_length_m: float = 0.70,
        robot_width_m: float = 0.31,
        lidar_x_m: float = 0.28945,
        lidar_y_m: float = 0.0,
        footprint_clearance_m: float | None = 0.15,
        use_elliptic_footprint: bool = True,
        max_range: float = 5.0,
        no_return_tolerance_m: float = 1.0e-4,
        inactive_barrier_value: float = 20.0,
        denominator_epsilon: float = 1.0e-8,
        hard_clip_output: bool = True,
    ) -> None:
        super().__init__(
            d_safe=d_safe,
            robot_length_m=robot_length_m,
            robot_width_m=robot_width_m,
            lidar_x_m=lidar_x_m,
            lidar_y_m=lidar_y_m,
            footprint_clearance_m=footprint_clearance_m,
            use_elliptic_footprint=use_elliptic_footprint,
            max_range=max_range,
            no_return_tolerance_m=no_return_tolerance_m,
            inactive_barrier_value=inactive_barrier_value,
            eps=denominator_epsilon,
        )
        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self.limits_lower, self.limits_upper = _asymmetric_limits(max_vx, max_vy, max_wz, min_vx)
        self.hard_clip_output = bool(hard_clip_output)

    def filter(
        self,
        ranges: torch.Tensor,
        angles: torch.Tensor,
        nominal_command: torch.Tensor,
    ) -> LaserSafetyFilterResult:
        B = nominal_command.shape[0]
        points, valid = self.scan_points(
            ranges,
            angles,
            B,
            device=nominal_command.device,
            dtype=nominal_command.dtype,
        )
        h_i, rows = self.current_h_and_direct_row(points)
        h_i, rows = self.mask_terms(h_i, valid, rows)
        h_lse, a = _softmin_value_grad(h_i, rows, self.kappa)

        zero = torch.zeros_like(nominal_command)
        command_unclipped, omega, lam, feasible, residual = _exact_projection(
            nominal_command,
            a,
            self.alpha * h_lse,
            eps=self.eps,
            fallback=zero,
        )
        limits_lower = torch.as_tensor(self.limits_lower, device=nominal_command.device, dtype=nominal_command.dtype)
        limits_upper = torch.as_tensor(self.limits_upper, device=nominal_command.device, dtype=nominal_command.dtype)
        command = torch.clamp(command_unclipped, limits_lower, limits_upper) if self.hard_clip_output else command_unclipped
        correction = command_unclipped - nominal_command
        violation = torch.relu(torch.maximum(limits_lower - command_unclipped, command_unclipped - limits_upper)).max(dim=1).values

        return LaserSafetyFilterResult(
            command=command,
            h_lse=h_lse,
            individual_h=h_i,
            active=omega < 0.0,
            correction_norm=torch.linalg.norm(correction, dim=1),
            cbf_value_nominal=omega,
            lambda_value=lam,
            input_bound_violation=violation,
            input_bound_active=violation > 1.0e-6,
            projection_feasible=feasible,
            projected_cbf_residual=residual,
        )


class LaserScanCbfQp(_LocalScanPointGeometry):
    """Direct, unaggregated LiDAR CBF-QP without control dynamics.

    The QP decision is the applied body command u=[vx,vy,w]. Each valid laser
    endpoint contributes its own relative-degree-one condition
        hdot_i + alpha h_i >= 0.
    The only additional rows are the direct hard command bounds. There is no
    HOCBF lift, command-state model, backup rollout, terminal set, or LSE.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.5,
        max_vx: float = 0.5,
        max_vy: float = 0.3,
        max_wz: float = 0.5,
        min_vx: float | None = None,
        max_active_candidates: int = 18,
        feasibility_tolerance: float = 1.0e-5,
        qp_regularization: float = 1.0e-5,
        straight_through: bool = True,
        qp_backend: str = "active_set",
        hard_clip_output: bool = True,
        denominator_epsilon: float = 1.0e-8,
        **geometry_kwargs,
    ) -> None:
        super().__init__(eps=denominator_epsilon, **geometry_kwargs)
        from .bcbf_qp_filter import OsqpProjector, QpthProjector, SmallBoxQpProjector

        self.alpha = float(alpha)
        self.limits_lower, self.limits_upper = _asymmetric_limits(max_vx, max_vy, max_wz, min_vx)
        self.hard_clip_output = bool(hard_clip_output)
        if qp_backend == "active_set":
            self.projector = SmallBoxQpProjector(
                max_candidates=max_active_candidates,
                feasibility_tol=feasibility_tolerance,
                regularization=qp_regularization,
                straight_through=straight_through,
            )
        elif qp_backend == "osqp":
            self.projector = OsqpProjector(
                feasibility_tol=feasibility_tolerance,
                regularization=qp_regularization,
                straight_through=straight_through,
            )
        elif qp_backend == "qpth":
            self.projector = QpthProjector(
                feasibility_tol=feasibility_tolerance,
                regularization=qp_regularization,
            )
        else:
            raise ValueError(f"Unknown QP backend {qp_backend!r}")
        self.qp_backend = qp_backend

    def filter(self, ranges, angles, nominal_command) -> LaserSafetyFilterResult:
        B = nominal_command.shape[0]
        points, valid = self.scan_points(
            ranges, angles, B,
            device=nominal_command.device, dtype=nominal_command.dtype,
        )
        h, lg = self.current_h_and_direct_row(points)
        h, lg = self.mask_terms(h, valid, lg)
        A_laser = -lg
        b_laser = self.alpha * h

        limits_lower = torch.as_tensor(
            self.limits_lower, device=nominal_command.device, dtype=nominal_command.dtype
        )
        limits_upper = torch.as_tensor(
            self.limits_upper, device=nominal_command.device, dtype=nominal_command.dtype
        )
        eye = torch.eye(3, device=nominal_command.device, dtype=nominal_command.dtype)
        A_box = torch.cat((eye, -eye), dim=0).unsqueeze(0).expand(B, -1, -1)
        b_box = torch.cat((limits_upper, -limits_lower)).unsqueeze(0).expand(B, -1)
        A = torch.cat((A_laser, A_box), dim=1)
        b = torch.cat((b_laser, b_box), dim=1)
        weights = torch.ones(3, device=nominal_command.device, dtype=nominal_command.dtype)
        command_unclipped, diag = self.projector.project(nominal_command, A, b, weights)
        violation = torch.relu(torch.maximum(limits_lower - command_unclipped, command_unclipped - limits_upper)).max(dim=1).values
        command = (
            torch.clamp(command_unclipped, limits_lower, limits_upper)
            if self.hard_clip_output else command_unclipped
        )
        correction = command_unclipped - nominal_command
        feasible = diag.max_violation <= self.projector.feasibility_tol
        return LaserSafetyFilterResult(
            command=command,
            surrogate_command=command_unclipped,
            desired_surrogate_command=nominal_command,
            h_lse=h.min(dim=1).values,  # compatibility field: hard min, not LSE
            individual_h=h,
            active=torch.linalg.norm(correction.detach(), dim=1) > 1.0e-6,
            correction_norm=torch.linalg.norm(correction.detach(), dim=1),
            cbf_value_nominal=-diag.max_violation.detach(),
            lambda_value=diag.active_rows.detach(),
            input_bound_violation=violation,
            input_bound_active=violation > 1.0e-6,
            projection_feasible=feasible,
            projected_cbf_residual=(-diag.max_violation).detach(),
        )


class LaserScanCbfLseQp(LaserScanCbfQp):
    """Direct LiDAR CBF-QP with one LSE ray row and six box rows.

    This is the seven-constraint comparison requested for the direct-command
    model.  Only the 120 geometric ray barriers are soft-min aggregated; the
    physical command bounds remain six explicit inequalities.
    """

    def __init__(self, *, kappa: float = 30.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.kappa = float(kappa)

    def filter(self, ranges, angles, nominal_command) -> LaserSafetyFilterResult:
        B = nominal_command.shape[0]
        points, valid = self.scan_points(
            ranges, angles, B,
            device=nominal_command.device, dtype=nominal_command.dtype,
        )
        h, lg = self.current_h_and_direct_row(points)
        h, lg = self.mask_terms(h, valid, lg)
        zeros = torch.zeros_like(h)
        h_lse, _, lg_lse = _softmin_value_lie(h, zeros, lg, self.kappa)

        limits_lower = torch.as_tensor(
            self.limits_lower, device=nominal_command.device, dtype=nominal_command.dtype
        )
        limits_upper = torch.as_tensor(
            self.limits_upper, device=nominal_command.device, dtype=nominal_command.dtype
        )
        eye = torch.eye(3, device=nominal_command.device, dtype=nominal_command.dtype)
        A_box = torch.cat((eye, -eye), dim=0).unsqueeze(0).expand(B, -1, -1)
        b_box = torch.cat((limits_upper, -limits_lower)).unsqueeze(0).expand(B, -1)
        A = torch.cat((-lg_lse[:, None, :], A_box), dim=1)
        b = torch.cat(((self.alpha * h_lse)[:, None], b_box), dim=1)
        weights = torch.ones(3, device=nominal_command.device, dtype=nominal_command.dtype)
        command_unclipped, diag = self.projector.project(nominal_command, A, b, weights)
        violation = torch.relu(torch.maximum(limits_lower - command_unclipped, command_unclipped - limits_upper)).max(dim=1).values
        command = (
            torch.clamp(command_unclipped, limits_lower, limits_upper)
            if self.hard_clip_output else command_unclipped
        )
        correction = command_unclipped - nominal_command
        feasible = diag.max_violation <= self.projector.feasibility_tol
        return LaserSafetyFilterResult(
            command=command,
            surrogate_command=command_unclipped,
            desired_surrogate_command=nominal_command,
            h_lse=h_lse,
            individual_h=h,
            active=torch.linalg.norm(correction.detach(), dim=1) > 1.0e-6,
            correction_norm=torch.linalg.norm(correction.detach(), dim=1),
            cbf_value_nominal=-diag.max_violation.detach(),
            lambda_value=diag.active_rows.detach(),
            input_bound_violation=violation,
            input_bound_active=violation > 1.0e-6,
            projection_feasible=feasible,
            projected_cbf_residual=(-diag.max_violation).detach(),
        )


class LaserScanCbfControlDynamicsClosedForm(_LocalScanPointGeometry):
    """Local-point CBF-CD-CF with full lifted-barrier derivatives.

    State z=[p_x,p_y,theta,v_x,v_y,w] and decision input uhat.  The current
    scan points are fixed in a temporary local frame.  The primitive geometric
    barrier has relative degree two wrt uhat; psi_i=hdot_i+k0 h_i is used and
    differentiated with respect to the complete augmented state.  This includes
    the previously omitted k0*hdot_i contribution.
    """

    def __init__(
        self,
        *,
        d_safe: float = 0.30,
        hocbf_k0: float = 2.0,
        alpha_outer: float = 0.8,
        kappa: float = 30.0,
        kappa_input: float = 20.0,
        kappa_outer: float = 20.0,
        barrier_rate_scale: float = 1.0,
        max_vx: float = 0.5,
        max_vy: float = 0.3,
        max_wz: float = 0.5,
        min_vx: float | None = None,
        control_dynamics_a: tuple[float, float, float] = (8.0, 8.0, 8.0),
        tracking_sigma: tuple[float, float, float] = (8.0, 8.0, 8.0),
        integration_dt: float = 0.10,
        robot_length_m: float = 0.70,
        robot_width_m: float = 0.31,
        lidar_x_m: float = 0.28945,
        lidar_y_m: float = 0.0,
        footprint_clearance_m: float | None = 0.15,
        use_elliptic_footprint: bool = True,
        max_range: float = 5.0,
        no_return_tolerance_m: float = 1.0e-4,
        inactive_barrier_value: float = 20.0,
        denominator_epsilon: float = 1.0e-8,
        hard_clip_output: bool = True,
    ) -> None:
        super().__init__(
            d_safe=d_safe,
            robot_length_m=robot_length_m,
            robot_width_m=robot_width_m,
            lidar_x_m=lidar_x_m,
            lidar_y_m=lidar_y_m,
            footprint_clearance_m=footprint_clearance_m,
            use_elliptic_footprint=use_elliptic_footprint,
            max_range=max_range,
            no_return_tolerance_m=no_return_tolerance_m,
            inactive_barrier_value=inactive_barrier_value,
            eps=denominator_epsilon,
        )
        self.k0 = float(hocbf_k0)
        self.alpha_outer = float(alpha_outer)
        self.kappa_ray = float(kappa)
        self.kappa_input = float(kappa_input)
        self.kappa_outer = float(kappa_outer)
        self.rate_scale = float(barrier_rate_scale)
        if self.rate_scale <= 0.0:
            raise ValueError("barrier_rate_scale must be positive")
        self.limits_lower, self.limits_upper = _asymmetric_limits(max_vx, max_vy, max_wz, min_vx)
        self.a_tuple = tuple(float(x) for x in control_dynamics_a)
        self.sigma_tuple = tuple(float(x) for x in tracking_sigma)
        self.dt = float(integration_dt)
        self.hard_clip_output = bool(hard_clip_output)

    def _input_terms_and_lie(
        self,
        current_command: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = current_command.shape[0]
        a = torch.as_tensor(self.a_tuple, device=current_command.device, dtype=current_command.dtype)
        limits_lower = torch.as_tensor(self.limits_lower, device=current_command.device, dtype=current_command.dtype)
        limits_upper = torch.as_tensor(self.limits_upper, device=current_command.device, dtype=current_command.dtype)
        terms = torch.stack(
            tuple(
                value
                for j in range(3)
                for value in (
                    (limits_upper[j] - current_command[:, j]) / limits_upper[j].abs(),
                    (current_command[:, j] - limits_lower[j]) / limits_lower[j].abs(),
                )
            ),
            dim=1,
        )
        grad_u = torch.zeros(B, 6, 3, device=current_command.device, dtype=current_command.dtype)
        for j in range(3):
            grad_u[:, 2 * j, j] = -1.0 / limits_upper[j].abs()
            grad_u[:, 2 * j + 1, j] = 1.0 / limits_lower[j].abs()
        drift_u = -a * current_command
        lf = torch.sum(grad_u * drift_u[:, None, :], dim=-1)
        lg = grad_u * a[None, None, :]
        return terms, lf, lg

    def filter(
        self,
        ranges: torch.Tensor,
        angles: torch.Tensor,
        current_command: torch.Tensor,
        nominal_command: torch.Tensor,
        previous_nominal_command: torch.Tensor | None = None,
        nominal_history_valid: torch.Tensor | None = None,
    ) -> LaserSafetyFilterResult:
        B = current_command.shape[0]
        points, valid = self.scan_points(
            ranges,
            angles,
            B,
            device=current_command.device,
            dtype=current_command.dtype,
        )
        psi, grad_psi = self.current_psi_and_grad(points, current_command, self.k0)
        psi, grad_psi = self.mask_terms(psi, valid, grad_psi)
        psi_n = psi / self.rate_scale
        grad_psi_n = grad_psi / self.rate_scale

        a_cd = torch.as_tensor(self.a_tuple, device=current_command.device, dtype=current_command.dtype)
        fhat0 = torch.cat((current_command, -a_cd * current_command), dim=1)
        lf_ray_terms = torch.sum(grad_psi_n * fhat0[:, None, :], dim=-1)
        lg_ray_terms = grad_psi_n[..., 3:] * a_cd[None, None, :]
        h_ray, lf_ray, lg_ray = _softmin_value_lie(
            psi_n, lf_ray_terms, lg_ray_terms, self.kappa_ray
        )

        input_terms, lf_input_terms, lg_input_terms = self._input_terms_and_lie(current_command)
        h_input, lf_input, lg_input = _softmin_value_lie(
            input_terms, lf_input_terms, lg_input_terms, self.kappa_input
        )
        h_aug, lf, lg = _softmin_value_lie(
            torch.stack((h_ray, h_input), dim=1),
            torch.stack((lf_ray, lf_input), dim=1),
            torch.stack((lg_ray, lg_input), dim=1),
            self.kappa_outer,
        )

        sigma = torch.as_tensor(self.sigma_tuple, device=current_command.device, dtype=current_command.dtype)
        if previous_nominal_command is None:
            previous_nominal_command = nominal_command.detach()
        if nominal_history_valid is None:
            valid_history = torch.ones(B, 1, device=current_command.device, dtype=current_command.dtype)
        else:
            valid_history = nominal_history_valid.to(device=current_command.device, dtype=current_command.dtype)
            if valid_history.ndim == 1:
                valid_history = valid_history[:, None]
        nominal_derivative = valid_history * (nominal_command - previous_nominal_command) / max(self.dt, self.eps)
        desired_surrogate = (
            current_command
            + nominal_derivative / a_cd
            + (sigma / a_cd) * (nominal_command - current_command)
        )

        fallback = torch.zeros_like(desired_surrogate)
        surrogate, omega, lam, feasible, residual = _exact_projection(
            desired_surrogate,
            lg,
            lf + self.alpha_outer * h_aug,
            eps=self.eps,
            fallback=fallback,
        )
        decay = torch.exp(-a_cd * self.dt)
        command_unclipped = decay * current_command + (1.0 - decay) * surrogate
        limits_lower = torch.as_tensor(self.limits_lower, device=current_command.device, dtype=current_command.dtype)
        limits_upper = torch.as_tensor(self.limits_upper, device=current_command.device, dtype=current_command.dtype)
        command = torch.clamp(command_unclipped, limits_lower, limits_upper) if self.hard_clip_output else command_unclipped
        correction = surrogate - desired_surrogate
        violation = torch.relu(torch.maximum(limits_lower - command_unclipped, command_unclipped - limits_upper)).max(dim=1).values

        return LaserSafetyFilterResult(
            command=command,
            h_lse=h_aug,
            individual_h=torch.cat((psi_n, input_terms), dim=1),
            active=omega < 0.0,
            correction_norm=torch.linalg.norm(correction, dim=1),
            cbf_value_nominal=omega,
            lambda_value=lam,
            surrogate_command=surrogate,
            desired_surrogate_command=desired_surrogate,
            h_backup_lse=h_ray,
            h_input_lse=h_input,
            mu_value=torch.zeros_like(lam),
            input_bound_violation=violation,
            input_bound_active=violation > 1.0e-6,
            projection_feasible=feasible,
            projected_cbf_residual=residual,
        )


class LaserScanCbfLseControlDynamicsQp(LaserScanCbfControlDynamicsClosedForm):
    """One-row non-backup LSE-CBF-QP with command control dynamics.

    This is the synthesis-only QP counterpart of ``cbf_cd_cf``.  It uses the same
    lifted ray HOCBF, first-order command dynamics, tracking surrogate,
    geometry, gains, nested LSE aggregation, and exact zero-order-hold command
    update.  All ray and command-state input-limit terms are composed into the
    same single affine CBF inequality used by the analytical filter; only the
    projection solver changes.  The three QP decision variables are the
    surrogate command ``uhat=[vx, vy, wz]``.
    """

    def __init__(
        self,
        *,
        max_active_candidates: int = 18,
        feasibility_tolerance: float = 1.0e-5,
        qp_regularization: float = 1.0e-5,
        straight_through: bool = True,
        qp_backend: str = "active_set",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        from .bcbf_qp_filter import OsqpProjector, QpthProjector, SmallBoxQpProjector

        if qp_backend == "active_set":
            self.projector = SmallBoxQpProjector(
                max_candidates=max_active_candidates,
                feasibility_tol=feasibility_tolerance,
                regularization=qp_regularization,
                straight_through=straight_through,
            )
        elif qp_backend == "osqp":
            self.projector = OsqpProjector(
                feasibility_tol=feasibility_tolerance,
                regularization=qp_regularization,
                straight_through=straight_through,
            )
        elif qp_backend == "qpth":
            self.projector = QpthProjector(
                feasibility_tol=feasibility_tolerance,
                regularization=qp_regularization,
            )
        else:
            raise ValueError(f"Unknown QP backend {qp_backend!r}")
        self.qp_backend = qp_backend

    def filter(
        self,
        ranges: torch.Tensor,
        angles: torch.Tensor,
        current_command: torch.Tensor,
        nominal_command: torch.Tensor,
        previous_nominal_command: torch.Tensor | None = None,
        nominal_history_valid: torch.Tensor | None = None,
    ) -> LaserSafetyFilterResult:
        batch_size = current_command.shape[0]
        points, valid = self.scan_points(
            ranges,
            angles,
            batch_size,
            device=current_command.device,
            dtype=current_command.dtype,
        )
        psi, grad_psi = self.current_psi_and_grad(points, current_command, self.k0)
        psi, grad_psi = self.mask_terms(psi, valid, grad_psi)
        psi_n = psi / self.rate_scale
        grad_psi_n = grad_psi / self.rate_scale

        a_cd = torch.as_tensor(
            self.a_tuple, device=current_command.device, dtype=current_command.dtype
        )
        fhat0 = torch.cat((current_command, -a_cd * current_command), dim=1)
        lf_ray_terms = torch.sum(grad_psi_n * fhat0[:, None, :], dim=-1)
        lg_ray_terms = grad_psi_n[..., 3:] * a_cd[None, None, :]
        h_ray, lf_ray, lg_ray = _softmin_value_lie(
            psi_n, lf_ray_terms, lg_ray_terms, self.kappa_ray
        )

        input_terms, lf_input_terms, lg_input_terms = self._input_terms_and_lie(current_command)
        h_input, lf_input, lg_input = _softmin_value_lie(
            input_terms, lf_input_terms, lg_input_terms, self.kappa_input
        )
        h_aug, lf, lg = _softmin_value_lie(
            torch.stack((h_ray, h_input), dim=1),
            torch.stack((lf_ray, lf_input), dim=1),
            torch.stack((lg_ray, lg_input), dim=1),
            self.kappa_outer,
        )

        sigma = torch.as_tensor(
            self.sigma_tuple, device=current_command.device, dtype=current_command.dtype
        )
        if previous_nominal_command is None:
            previous_nominal_command = nominal_command.detach()
        if nominal_history_valid is None:
            valid_history = torch.ones(
                batch_size, 1, device=current_command.device, dtype=current_command.dtype
            )
        else:
            valid_history = nominal_history_valid.to(
                device=current_command.device, dtype=current_command.dtype
            )
            if valid_history.ndim == 1:
                valid_history = valid_history[:, None]
        nominal_derivative = (
            valid_history * (nominal_command - previous_nominal_command) / max(self.dt, self.eps)
        )
        desired_surrogate = (
            current_command
            + nominal_derivative / a_cd
            + (sigma / a_cd) * (nominal_command - current_command)
        )

        # lf + lg*uhat + alpha*h >= 0  <=>  -lg*uhat <= lf + alpha*h.
        A = -lg[:, None, :]
        b = (lf + self.alpha_outer * h_aug)[:, None]
        weights = torch.ones(3, device=current_command.device, dtype=current_command.dtype)
        surrogate, diag = self.projector.project(desired_surrogate, A, b, weights)

        decay = torch.exp(-a_cd * self.dt)
        command_unclipped = decay * current_command + (1.0 - decay) * surrogate
        limits_lower = torch.as_tensor(
            self.limits_lower, device=current_command.device, dtype=current_command.dtype
        )
        limits_upper = torch.as_tensor(
            self.limits_upper, device=current_command.device, dtype=current_command.dtype
        )
        violation = torch.relu(torch.maximum(limits_lower - command_unclipped, command_unclipped - limits_upper)).max(dim=1).values
        command = (
            torch.clamp(command_unclipped, limits_lower, limits_upper)
            if self.hard_clip_output
            else command_unclipped
        )
        correction = surrogate - desired_surrogate
        feasible = diag.max_violation <= self.projector.feasibility_tol
        return LaserSafetyFilterResult(
            command=command,
            surrogate_command=surrogate,
            desired_surrogate_command=desired_surrogate,
            h_lse=h_aug,
            h_backup_lse=h_ray,
            h_input_lse=h_input,
            individual_h=torch.cat((psi_n, input_terms), dim=1),
            active=torch.linalg.norm(correction.detach(), dim=1) > 1.0e-6,
            correction_norm=torch.linalg.norm(correction.detach(), dim=1),
            cbf_value_nominal=-diag.max_violation.detach(),
            lambda_value=diag.active_rows.detach(),
            mu_value=torch.zeros_like(h_aug),
            input_bound_violation=violation,
            input_bound_active=violation > 1.0e-6,
            projection_feasible=feasible,
            projected_cbf_residual=(-diag.max_violation).detach(),
        )


class LaserScanCbfLseControlDynamicsQpCfMatched(LaserScanCbfControlDynamicsClosedForm):
    """CF-matched differentiable-QP baseline for the laser (non-backup)
    CBF-CD filter -- the cbf_cd_cf / cbf_lse_cd_qp counterpart of
    LaserZeroBackupLseControlDynamicsQpCfMatched (see that class's
    docstring for the full rationale). Builds the IDENTICAL single-row
    ray+input LSE aggregate LaserScanCbfControlDynamicsClosedForm.filter()
    solves in closed form (same kappa_ray, kappa_input, kappa_outer,
    alpha_outer), always solves it with QpthProjector (exact differentiable
    QP/KKT, no straight-through/.detach()), and explicitly reproduces the
    closed form's active-but-degenerate zero-surrogate fallback, which
    LaserScanCbfLseControlDynamicsQp (the existing QP variant) does not.
    """

    def __init__(
        self,
        *,
        max_active_candidates: int = 18,
        feasibility_tolerance: float = 1.0e-6,
        qp_regularization: float = 1.0e-6,
        straight_through: bool = True,
        qp_backend: str = "active_set",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        from .bcbf_qp_filter import QpthProjector

        self.projector = QpthProjector(
            feasibility_tol=feasibility_tolerance,
            regularization=qp_regularization,
        )

    def filter(
        self,
        ranges: torch.Tensor,
        angles: torch.Tensor,
        current_command: torch.Tensor,
        nominal_command: torch.Tensor,
        previous_nominal_command: torch.Tensor | None = None,
        nominal_history_valid: torch.Tensor | None = None,
    ) -> LaserSafetyFilterResult:
        batch_size = current_command.shape[0]
        points, valid = self.scan_points(
            ranges, angles, batch_size, device=current_command.device, dtype=current_command.dtype
        )
        psi, grad_psi = self.current_psi_and_grad(points, current_command, self.k0)
        psi, grad_psi = self.mask_terms(psi, valid, grad_psi)
        psi_n = psi / self.rate_scale
        grad_psi_n = grad_psi / self.rate_scale

        a_cd = torch.as_tensor(self.a_tuple, device=current_command.device, dtype=current_command.dtype)
        fhat0 = torch.cat((current_command, -a_cd * current_command), dim=1)
        lf_ray_terms = torch.sum(grad_psi_n * fhat0[:, None, :], dim=-1)
        lg_ray_terms = grad_psi_n[..., 3:] * a_cd[None, None, :]
        h_ray, lf_ray, lg_ray = _softmin_value_lie(psi_n, lf_ray_terms, lg_ray_terms, self.kappa_ray)

        input_terms, lf_input_terms, lg_input_terms = self._input_terms_and_lie(current_command)
        h_input, lf_input, lg_input = _softmin_value_lie(
            input_terms, lf_input_terms, lg_input_terms, self.kappa_input
        )
        h_aug, lf, lg = _softmin_value_lie(
            torch.stack((h_ray, h_input), dim=1),
            torch.stack((lf_ray, lf_input), dim=1),
            torch.stack((lg_ray, lg_input), dim=1),
            self.kappa_outer,
        )

        sigma = torch.as_tensor(self.sigma_tuple, device=current_command.device, dtype=current_command.dtype)
        if previous_nominal_command is None:
            previous_nominal_command = nominal_command.detach()
        if nominal_history_valid is None:
            valid_history = torch.ones(
                batch_size, 1, device=current_command.device, dtype=current_command.dtype
            )
        else:
            valid_history = nominal_history_valid.to(
                device=current_command.device, dtype=current_command.dtype
            )
            if valid_history.ndim == 1:
                valid_history = valid_history[:, None]
        nominal_derivative = (
            valid_history * (nominal_command - previous_nominal_command) / max(self.dt, self.eps)
        )
        desired_surrogate = (
            current_command
            + nominal_derivative / a_cd
            + (sigma / a_cd) * (nominal_command - current_command)
        )

        omega = lf + torch.sum(lg * desired_surrogate, dim=1) + self.alpha_outer * h_aug
        denom = torch.sum(lg.square(), dim=1)
        needs_correction = omega < 0.0
        nondegenerate = denom > self.eps
        degenerate_active = needs_correction & (~nondegenerate)
        lambda_value = torch.where(
            needs_correction & nondegenerate,
            -omega / torch.clamp(denom, min=self.eps),
            torch.zeros_like(omega),
        )

        A = -lg[:, None, :]
        b = (lf + self.alpha_outer * h_aug)[:, None]
        weights = torch.ones(3, device=current_command.device, dtype=current_command.dtype)
        qp_surrogate, diag = self.projector.project(desired_surrogate, A, b, weights)
        surrogate = torch.where(
            degenerate_active[:, None], torch.zeros_like(qp_surrogate), qp_surrogate
        )

        decay = torch.exp(-a_cd * self.dt)
        command_unclipped = decay * current_command + (1.0 - decay) * surrogate
        limits_lower = torch.as_tensor(
            self.limits_lower, device=current_command.device, dtype=current_command.dtype
        )
        limits_upper = torch.as_tensor(
            self.limits_upper, device=current_command.device, dtype=current_command.dtype
        )
        violation = torch.relu(torch.maximum(limits_lower - command_unclipped, command_unclipped - limits_upper)).max(dim=1).values
        command = (
            torch.clamp(command_unclipped, limits_lower, limits_upper)
            if self.hard_clip_output
            else command_unclipped
        )
        correction = surrogate - desired_surrogate
        feasible = diag.max_violation <= self.projector.feasibility_tol
        return LaserSafetyFilterResult(
            command=command,
            surrogate_command=surrogate,
            desired_surrogate_command=desired_surrogate,
            h_lse=h_aug,
            h_backup_lse=h_ray,
            h_input_lse=h_input,
            individual_h=torch.cat((psi_n, input_terms), dim=1),
            active=needs_correction,
            correction_norm=torch.linalg.norm(correction, dim=1),
            cbf_value_nominal=omega,
            lambda_value=lambda_value,
            mu_value=torch.zeros_like(h_aug),
            input_bound_violation=violation,
            input_bound_active=violation > 1.0e-6,
            projection_feasible=feasible,
            projected_cbf_residual=(-diag.max_violation).detach(),
        )


class LaserZeroBackupLseCbfClosedForm(_LocalScanPointGeometry):
    """Local-point zero-backup bCBF-LSE-CF with command dynamics.

    This is the LiDAR analogue of the map-based ``bcbf_cf_zero_v4`` model:
      * q_i are current LiDAR endpoints in a temporary scan-fixed frame;
      * z=[p_x,p_y,theta,v_x,v_y,w];
      * uhat_B=0 and u_B decays under first-order command dynamics;
      * the full pose/command backup flow and sensitivity Phi_B are propagated;
      * tau=0 uses psi_i=hdot_i+k0 h_i;
      * nonzero-time terms use h_i(z_B(tau));
      * terminal h_B enlarges the footprint by d_B+||u_xy||/a_min,
        giving explicit relative degree one wrt the augmented input.
    """

    def __init__(
        self,
        *,
        d_safe: float = 0.30,
        backup_margin: float = 0.30,
        backup_horizon_s: float = 4.0,
        backup_samples: int = 21,
        backup_rollout_substeps: int = 8,
        hocbf_k0: float = 2.0,
        barrier_rate_scale: float = 1.0,
        kappa_backup: float = 50.0,
        kappa_input: float = 20.0,
        kappa_outer: float = 20.0,
        alpha_outer: float = 0.8,
        max_vx: float = 0.5,
        max_vy: float = 0.3,
        max_wz: float = 0.5,
        min_vx: float | None = None,
        control_dynamics_a: tuple[float, float, float] = (8.0, 8.0, 8.0),
        tracking_sigma: tuple[float, float, float] = (8.0, 8.0, 8.0),
        integration_dt: float = 0.10,
        robot_length_m: float = 0.70,
        robot_width_m: float = 0.31,
        lidar_x_m: float = 0.28945,
        lidar_y_m: float = 0.0,
        footprint_clearance_m: float | None = 0.15,
        use_elliptic_footprint: bool = True,
        max_range: float = 5.0,
        no_return_tolerance_m: float = 1.0e-4,
        inactive_barrier_value: float = 20.0,
        denominator_epsilon: float = 1.0e-8,
        hard_clip_output: bool = True,
        terminal_backup_mode: str = "invariant_ellipse",
        terminal_geom_scale: float = 1.0,
        terminal_rec_scale: float = 1.0,
        legacy_projection: bool = True,
        straight_through: bool = False,
    ) -> None:
        super().__init__(
            d_safe=d_safe,
            robot_length_m=robot_length_m,
            robot_width_m=robot_width_m,
            lidar_x_m=lidar_x_m,
            lidar_y_m=lidar_y_m,
            footprint_clearance_m=footprint_clearance_m,
            use_elliptic_footprint=use_elliptic_footprint,
            max_range=max_range,
            no_return_tolerance_m=no_return_tolerance_m,
            inactive_barrier_value=inactive_barrier_value,
            eps=denominator_epsilon,
        )
        self.straight_through = bool(straight_through)
        self.backup_margin = float(backup_margin)
        self.backup_horizon_s = float(backup_horizon_s)
        self.backup_samples = int(backup_samples)
        self.backup_rollout_substeps = int(backup_rollout_substeps)
        if self.backup_samples < 2:
            raise ValueError("backup_samples must be at least 2")
        if self.backup_rollout_substeps < 1:
            raise ValueError("backup_rollout_substeps must be at least 1")
        self.k0 = float(hocbf_k0)
        self.rate_scale = float(barrier_rate_scale)
        if self.rate_scale <= 0.0:
            raise ValueError("barrier_rate_scale must be positive")
        self.kappa_backup = float(kappa_backup)
        self.kappa_input = float(kappa_input)
        self.kappa_outer = float(kappa_outer)
        self.alpha_outer = float(alpha_outer)
        self.limits_lower, self.limits_upper = _asymmetric_limits(max_vx, max_vy, max_wz, min_vx)
        self.a_tuple = tuple(float(x) for x in control_dynamics_a)
        self.sigma_tuple = tuple(float(x) for x in tracking_sigma)
        self.dt = float(integration_dt)
        self.hard_clip_output = bool(hard_clip_output)
        # legacy_projection=True reverts ONLY the projection/hard-clip-
        # consistency step to the pre-2026-09-07 behavior (the same
        # single-half-space _exact_projection + naive post-hoc clamp that
        # cbf_cd_cf's filter still uses today, and that v4 -- which trained
        # every checkpoint before that date -- used for bcbf_cf too): no
        # joint box+half-space bound on the achievable surrogate given the
        # ZOH decay. This is independent of terminal_backup_mode (one vs
        # two terminal sets), which governs a different piece of the same
        # filter. Kept separate so the two can be isolated one at a time.
        self.legacy_projection = bool(legacy_projection)

        # --- v6 invariant-ellipse terminal set (config-level constants,
        # computed once here rather than per filter() call). ---
        if terminal_backup_mode not in ("legacy_inflated_ellipse", "invariant_ellipse"):
            raise ValueError(f"Unknown terminal_backup_mode {terminal_backup_mode!r}")
        self.terminal_backup_mode = terminal_backup_mode
        self.terminal_geom_scale = float(terminal_geom_scale)
        self.terminal_rec_scale = float(terminal_rec_scale)
        if self.terminal_geom_scale <= 0.0 or self.terminal_rec_scale <= 0.0:
            raise ValueError("terminal_geom_scale/terminal_rec_scale must be positive")
        # R_x = r_x + backup_margin, R_y = r_y + backup_margin -- FIXED, no
        # residual-velocity inflation (that is exactly what "legacy" does and
        # what makes the legacy terminal set not generally forward-invariant).
        self.terminal_big_r_x = self.axis_x + self.backup_margin
        self.terminal_big_r_y = self.axis_y + self.backup_margin
        self.terminal_c_theta = 0.5 * abs(
            self.terminal_big_r_x / self.terminal_big_r_y
            - self.terminal_big_r_y / self.terminal_big_r_x
        )

    def _fhat_and_jacobian(self, z: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = z.shape[0]
        theta = z[:, 2]
        vx = z[:, 3]
        vy = z[:, 4]
        wz = z[:, 5]
        c = torch.cos(theta)
        s = torch.sin(theta)

        f = torch.zeros(B, 6, device=z.device, dtype=z.dtype)
        f[:, 0] = c * vx - s * vy
        f[:, 1] = s * vx + c * vy
        f[:, 2] = wz
        f[:, 3:] = -a * z[:, 3:]

        F = torch.zeros(B, 6, 6, device=z.device, dtype=z.dtype)
        F[:, 0, 2] = -s * vx - c * vy
        F[:, 0, 3] = c
        F[:, 0, 4] = -s
        F[:, 1, 2] = c * vx - s * vy
        F[:, 1, 3] = s
        F[:, 1, 4] = c
        F[:, 2, 5] = 1.0
        F[:, 3, 3] = -a[0]
        F[:, 4, 4] = -a[1]
        F[:, 5, 5] = -a[2]
        return f, F

    def _input_lse_lie(
        self,
        current_command: torch.Tensor,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = current_command.shape[0]
        limits_lower = torch.as_tensor(self.limits_lower, device=current_command.device, dtype=current_command.dtype)
        limits_upper = torch.as_tensor(self.limits_upper, device=current_command.device, dtype=current_command.dtype)
        terms = torch.stack(
            tuple(
                value
                for j in range(3)
                for value in (
                    (limits_upper[j] - current_command[:, j]) / limits_upper[j].abs(),
                    (current_command[:, j] - limits_lower[j]) / limits_lower[j].abs(),
                )
            ),
            dim=1,
        )
        grads = torch.zeros(B, 6, 6, device=current_command.device, dtype=current_command.dtype)
        for j in range(3):
            grads[:, 2 * j, 3 + j] = -1.0 / limits_upper[j].abs()
            grads[:, 2 * j + 1, 3 + j] = 1.0 / limits_lower[j].abs()
        z0 = torch.cat((torch.zeros(B, 3, device=current_command.device, dtype=current_command.dtype), current_command), dim=1)
        fhat0, _ = self._fhat_and_jacobian(z0, a)
        lf_terms = torch.sum(grads * fhat0[:, None, :], dim=-1)
        lg_terms = grads[..., 3:] * a[None, None, :]
        h, lf, lg = _softmin_value_lie(terms, lf_terms, lg_terms, self.kappa_input)
        return h, lf, lg, terms

    def _backup_terms_and_lie(
        self,
        ranges: torch.Tensor,
        angles: torch.Tensor,
        current_command: torch.Tensor,
        *,
        aggregate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = current_command.shape[0]
        points, valid = self.scan_points(
            ranges,
            angles,
            B,
            device=current_command.device,
            dtype=current_command.dtype,
        )
        a = torch.as_tensor(self.a_tuple, device=current_command.device, dtype=current_command.dtype)
        a_min = torch.clamp(torch.min(a[:2]), min=self.eps)
        z0 = torch.cat((torch.zeros(B, 3, device=current_command.device, dtype=current_command.dtype), current_command), dim=1)
        fhat0, _ = self._fhat_and_jacobian(z0, a)

        all_terms: list[torch.Tensor] = []
        all_lf: list[torch.Tensor] = []
        all_lg: list[torch.Tensor] = []

        psi, grad_psi = self.current_psi_and_grad(points, current_command, self.k0)
        psi, grad_psi = self.mask_terms(psi, valid, grad_psi)
        psi_n = psi / self.rate_scale
        grad_psi_n = grad_psi / self.rate_scale
        all_terms.append(psi_n)
        all_lf.append(torch.sum(grad_psi_n * fhat0[:, None, :], dim=-1))
        all_lg.append(grad_psi_n[..., 3:] * a[None, None, :])

        # Positive-time rollout terms. z0=[0,0,0,u] in the scan-fixed frame,
        # so the exact zero-surrogate flow and its sensitivity Phi_B are the
        # same closed-form solution used by the map-based bCBF filter (only
        # the barrier evaluated along the flow differs: LiDAR points here
        # instead of a global obstacle map). All samples are independent
        # closed-form evaluations, computed in one vectorized call instead of
        # a sequential fixed-step Euler rollout.
        R = points.shape[1]
        num_samples = self.backup_samples - 1
        sample_times = self.backup_horizon_s * torch.arange(
            1, self.backup_samples, device=current_command.device, dtype=current_command.dtype
        ) / float(self.backup_samples - 1)
        z_t, phi_t = exact_zero_backup_state_and_phi(z0, sample_times, a)  # (B,K,6), (B,K,6,6)

        z_flat = z_t.reshape(B * num_samples, 6)
        points_flat = (
            points.unsqueeze(1)
            .expand(B, num_samples, R, 2)
            .reshape(B * num_samples, R, 2)
        )
        h_future_flat, grad_future_flat = self.flow_h_and_grad_future(points_flat, z_flat)
        h_future = h_future_flat.reshape(B, num_samples, R)
        grad_future = grad_future_flat.reshape(B, num_samples, R, 6)
        grad_current = torch.einsum("bkrm,bkmj->bkrj", grad_future, phi_t)

        valid_k = valid.unsqueeze(1).expand(B, num_samples, R)
        h_future, grad_current = self.mask_terms(h_future, valid_k, grad_current)

        all_terms.append(h_future.reshape(B, num_samples * R))
        all_lf.append(
            torch.sum(grad_current * fhat0[:, None, None, :], dim=-1).reshape(B, num_samples * R)
        )
        all_lg.append(
            (grad_current[..., 3:] * a[None, None, None, :]).reshape(B, num_samples * R, 3)
        )

        # Terminal augmented backup-set term(s), reusing the already-computed
        # state/Jacobian at the last sample time, which equals the horizon.
        z_terminal = z_t[:, -1, :]
        phi_terminal = phi_t[:, -1, :, :]

        if self.terminal_backup_mode == "legacy_inflated_ellipse":
            h_terminal, grad_terminal_future = self.terminal_h_and_grad_future(
                points,
                z_terminal,
                backup_margin=self.backup_margin,
                a_min=a_min,
            )
            grad_terminal = torch.einsum("brm,bmj->brj", grad_terminal_future, phi_terminal)
            h_terminal, grad_terminal = self.mask_terms(h_terminal, valid, grad_terminal)
            all_terms.append(h_terminal)
            all_lf.append(torch.sum(grad_terminal * fhat0[:, None, :], dim=-1))
            all_lg.append(grad_terminal[..., 3:] * a[None, None, :])
        elif self.terminal_backup_mode == "invariant_ellipse":
            a_decay_min = torch.clamp(torch.min(a), min=self.eps)
            h_geom_future, grad_geom_future, h_rec_future, grad_rec_future = (
                self.terminal_invariant_h_and_grad_future(
                    points,
                    z_terminal,
                    big_r_x=self.terminal_big_r_x,
                    big_r_y=self.terminal_big_r_y,
                    c_theta=self.terminal_c_theta,
                    a_decay_min=a_decay_min,
                )
            )
            grad_geom = torch.einsum("brm,bmj->brj", grad_geom_future, phi_terminal)
            grad_rec = torch.einsum("brm,bmj->brj", grad_rec_future, phi_terminal)

            # Scale BEFORE masking so inactive rays stay pinned at exactly
            # self.inactive_barrier_value regardless of the chosen scale
            # (masking after scaling would shrink/grow the sentinel itself).
            h_geom_n, grad_geom_n = self.mask_terms(
                self.terminal_geom_scale * h_geom_future, valid, self.terminal_geom_scale * grad_geom
            )
            h_rec_n, grad_rec_n = self.mask_terms(
                self.terminal_rec_scale * h_rec_future, valid, self.terminal_rec_scale * grad_rec
            )

            all_terms.append(h_geom_n)
            all_lf.append(torch.sum(grad_geom_n * fhat0[:, None, :], dim=-1))
            all_lg.append(grad_geom_n[..., 3:] * a[None, None, :])
            all_terms.append(h_rec_n)
            all_lf.append(torch.sum(grad_rec_n * fhat0[:, None, :], dim=-1))
            all_lg.append(grad_rec_n[..., 3:] * a[None, None, :])
        else:
            raise ValueError(f"Unknown terminal_backup_mode {self.terminal_backup_mode!r}")

        terms = torch.cat(all_terms, dim=1)
        lf_terms = torch.cat(all_lf, dim=1)
        lg_terms = torch.cat(all_lg, dim=1)
        if aggregate:
            h_backup, lf_backup, lg_backup = _softmin_value_lie(
                terms, lf_terms, lg_terms, self.kappa_backup
            )
        else:
            # The non-LSE QP consumes the individual arrays below.  Return the
            # hard-min row only for compatibility/diagnostics without computing
            # an otherwise unused soft-min aggregation.
            hard_index = terms.argmin(dim=1)
            batch_index = torch.arange(B, device=terms.device)
            h_backup = terms[batch_index, hard_index]
            lf_backup = lf_terms[batch_index, hard_index]
            lg_backup = lg_terms[batch_index, hard_index]
        return h_backup, lf_backup, lg_backup, terms, lf_terms, lg_terms

    def filter(
        self,
        ranges: torch.Tensor,
        angles: torch.Tensor,
        current_command: torch.Tensor,
        nominal_command: torch.Tensor,
        previous_nominal_command: torch.Tensor | None = None,
        nominal_history_valid: torch.Tensor | None = None,
    ):
        B = current_command.shape[0]
        a = torch.as_tensor(self.a_tuple, device=current_command.device, dtype=current_command.dtype)
        sigma = torch.as_tensor(self.sigma_tuple, device=current_command.device, dtype=current_command.dtype)

        h_backup, lf_backup, lg_backup, backup_terms, _, _ = self._backup_terms_and_lie(
            ranges, angles, current_command
        )
        h_input, lf_input, lg_input, input_terms = self._input_lse_lie(current_command, a)
        h_aug, lf, lg = _softmin_value_lie(
            torch.stack((h_backup, h_input), dim=1),
            torch.stack((lf_backup, lf_input), dim=1),
            torch.stack((lg_backup, lg_input), dim=1),
            self.kappa_outer,
        )

        if previous_nominal_command is None:
            previous_nominal_command = nominal_command.detach()
        if nominal_history_valid is None:
            valid_history = torch.ones(B, 1, device=current_command.device, dtype=current_command.dtype)
        else:
            valid_history = nominal_history_valid.to(device=current_command.device, dtype=current_command.dtype)
            if valid_history.ndim == 1:
                valid_history = valid_history[:, None]
        nominal_derivative = valid_history * (nominal_command - previous_nominal_command) / max(self.dt, self.eps)
        desired_surrogate = (
            current_command
            + nominal_derivative / a
            + (sigma / a) * (nominal_command - current_command)
        )

        limits_lower = torch.as_tensor(self.limits_lower, device=current_command.device, dtype=current_command.dtype)
        limits_upper = torch.as_tensor(self.limits_upper, device=current_command.device, dtype=current_command.dtype)
        if self.legacy_projection:
            # Pre-2026-09-07 behavior: project onto the CBF half-space alone
            # (no box constraint baked into the projection itself), matching
            # cbf_cd_cf's filter and every bcbf_cf checkpoint trained before
            # that date. A naive post-hoc clamp (below) is the only place
            # the command magnitude is bounded, which is exactly the
            # inconsistency the joint projection below was built to fix.
            decay = torch.exp(-a * self.dt)
            fallback = torch.zeros_like(desired_surrogate)
            surrogate, omega, lam, feasible, residual = _exact_projection(
                desired_surrogate,
                lg,
                lf + self.alpha_outer * h_aug,
                eps=self.eps,
                fallback=fallback,
            )
            joint_infeasible = ~feasible
        else:
            # Exact ZOH inversion: u_next = decay*u + (1-decay)*u_hat must
            # satisfy the PHYSICAL box [limits_lower,limits_upper] -- this is
            # a bound on u_hat, NOT the same as the physical bound itself
            # (see zoh_surrogate_bounds).
            u_hat_lower, u_hat_upper, decay = zoh_surrogate_bounds(
                current_command, limits_lower, limits_upper, a, self.dt
            )
            surrogate, omega, lam, feasible, residual, joint_infeasible = _exact_box_halfspace_projection(
                desired_surrogate,
                lg,
                lf + self.alpha_outer * h_aug,
                u_hat_lower,
                u_hat_upper,
                eps=self.eps,
            )
        if self.straight_through:
            # Same trick QpSolveDiagnostics-producing projectors use
            # (bcbf_qp_filter.py's qp_solution = desired + (qp_solution -
            # desired).detach()): the forward pass still executes the exact,
            # correctly-projected `surrogate` below, but backprop sees
            # d(surrogate)/d(desired_surrogate) = I instead of the true CBF-
            # projection Jacobian -- isolates whether the closed-form
            # filter's real differentiability (vs. an identity surrogate
            # gradient) is what drives bcbf_cf's training benefit.
            surrogate = desired_surrogate + (surrogate - desired_surrogate).detach()
        command_unclipped = decay * current_command + (1.0 - decay) * surrogate
        violation = torch.relu(torch.maximum(limits_lower - command_unclipped, command_unclipped - limits_upper)).max(dim=1).values
        # Hard clip is kept as a final numerical/physical failsafe (Task 4);
        # with the box already enforced inside the projection above, this
        # should now be a no-op up to floating-point error.
        command = torch.clamp(command_unclipped, limits_lower, limits_upper) if self.hard_clip_output else command_unclipped
        clip_delta = torch.linalg.norm(command - command_unclipped, dim=1)
        correction = surrogate - desired_surrogate

        # Task 5 (diagnostic only, does not feed back into the filter): the
        # surrogate that would have exactly produced the ACTUALLY EXECUTED
        # command through the same ZOH update, and its CBF residual.
        one_minus_decay = torch.clamp(1.0 - decay, min=self.eps)
        u_hat_exec = (command - decay * current_command) / one_minus_decay
        residual_exec = (lf + self.alpha_outer * h_aug) + torch.sum(lg * u_hat_exec, dim=1)

        kwargs = dict(
            command=command,
            surrogate_command=surrogate,
            desired_surrogate_command=desired_surrogate,
            h_lse=h_aug,
            h_backup_lse=h_backup,
            h_input_lse=h_input,
            individual_h=torch.cat((backup_terms, input_terms), dim=1),
            active=omega < 0.0,
            correction_norm=torch.linalg.norm(correction, dim=1),
            cbf_value_nominal=omega,
            lambda_value=lam,
            mu_value=torch.zeros_like(lam),
            input_bound_violation=violation,
            input_bound_active=violation > 1.0e-6,
        )
        return LaserSafetyFilterResult(
            **kwargs,
            projection_feasible=feasible,
            projected_cbf_residual=residual,
            # D_aug = ||Lg_hat h_aug||^2, the exact-projection denominator.
            # Diagnostic only -- `lg` here is the same aggregated gradient
            # already used above for `omega`/`surrogate`; this does not
            # change anything upstream, just exposes it for logging.
            lg_aug_sq_norm=torch.sum(lg.square(), dim=1),
            command_preclip=command_unclipped,
            joint_projection_infeasible=joint_infeasible,
            clip_delta=clip_delta,
            clip_changed=clip_delta > 1.0e-7,
            residual_exec=residual_exec,
        )


def build_invariant_ellipse_bcbf_filter(
    safety_cfg,
    action_limits: tuple[float, float, float],
    laser_max_range: float,
    min_vx: float | None = None,
) -> LaserZeroBackupLseCbfClosedForm:
    """Construct the bCBF-CF invariant-ellipse filter from a LaserSafetyConfig.

    Mirrors LaserSafeSacActor.__init__'s ``method == "bcbf_cf"`` branch
    exactly (same kwargs, same source config fields) so that scenario
    feasibility certification (reset-time reject/resample, and any offline
    certified-subset generation) always evaluates h_aug(z0) with the SAME
    geometry/terminal-set the actor's own filter would use -- never a
    reimplementation that could silently drift from it. ``terminal_backup_mode``
    is always forced to "invariant_ellipse" here (irrespective of what a given
    training run's own method/config uses), since the invariant-ellipse
    theorem is the one certification criterion Task 9 requires ALL methods to
    share for a fair initial-state distribution.
    """
    return LaserZeroBackupLseCbfClosedForm(
        d_safe=safety_cfg.d_safe,
        backup_margin=safety_cfg.backup_margin,
        backup_horizon_s=safety_cfg.backup_horizon_s,
        backup_samples=safety_cfg.backup_samples,
        backup_rollout_substeps=safety_cfg.backup_rollout_substeps,
        hocbf_k0=safety_cfg.hocbf_k0,
        barrier_rate_scale=safety_cfg.barrier_rate_scale,
        kappa_backup=safety_cfg.kappa_backup,
        kappa_input=safety_cfg.kappa_input,
        kappa_outer=safety_cfg.kappa_outer,
        alpha_outer=safety_cfg.alpha_outer,
        max_vx=action_limits[0],
        max_vy=action_limits[1],
        max_wz=action_limits[2],
        min_vx=min_vx,
        control_dynamics_a=safety_cfg.control_dynamics_a,
        tracking_sigma=safety_cfg.tracking_sigma,
        integration_dt=safety_cfg.integration_dt,
        robot_length_m=safety_cfg.robot_length_m,
        robot_width_m=safety_cfg.robot_width_m,
        lidar_x_m=safety_cfg.lidar_x_m,
        lidar_y_m=safety_cfg.lidar_y_m,
        footprint_clearance_m=safety_cfg.footprint_clearance_m,
        use_elliptic_footprint=safety_cfg.use_elliptic_footprint,
        max_range=laser_max_range,
        no_return_tolerance_m=safety_cfg.no_return_tolerance_m,
        inactive_barrier_value=safety_cfg.inactive_barrier_value,
        denominator_epsilon=safety_cfg.denominator_epsilon,
        hard_clip_output=safety_cfg.hard_clip_output,
        terminal_backup_mode="invariant_ellipse",
        terminal_geom_scale=safety_cfg.terminal_geom_scale,
        terminal_rec_scale=safety_cfg.terminal_rec_scale,
    )


class LaserZeroBackupLseControlDynamicsQp(LaserZeroBackupLseCbfClosedForm):
    """QP synthesis of the same augmented LSE barrier used by bCBF-LSE-CD-CF.

    The lifted current-time HOCBF terms, positive-time backup-flow terms,
    terminal terms, and six command-state input-limit barriers are aggregated
    exactly as in :class:`LaserZeroBackupLseCbfClosedForm`.  The only intended
    experimental difference is synthesis: this class sends the resulting one
    affine CBF inequality to the selected QP projector instead of applying the
    analytical half-space projection.
    """

    def __init__(
        self,
        *,
        max_active_candidates: int = 18,
        feasibility_tolerance: float = 1.0e-5,
        qp_regularization: float = 1.0e-5,
        straight_through: bool = True,
        qp_backend: str = "active_set",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        from .bcbf_qp_filter import OsqpProjector, QpthProjector, SmallBoxQpProjector

        if qp_backend == "active_set":
            self.projector = SmallBoxQpProjector(
                max_candidates=max_active_candidates,
                feasibility_tol=feasibility_tolerance,
                regularization=qp_regularization,
                straight_through=straight_through,
            )
        elif qp_backend == "osqp":
            self.projector = OsqpProjector(
                feasibility_tol=feasibility_tolerance,
                regularization=qp_regularization,
                straight_through=straight_through,
            )
        elif qp_backend == "qpth":
            self.projector = QpthProjector(
                feasibility_tol=feasibility_tolerance,
                regularization=qp_regularization,
            )
        else:
            raise ValueError(f"Unknown QP backend {qp_backend!r}")
        self.qp_backend = qp_backend

    def filter(
        self,
        ranges: torch.Tensor,
        angles: torch.Tensor,
        current_command: torch.Tensor,
        nominal_command: torch.Tensor,
        previous_nominal_command: torch.Tensor | None = None,
        nominal_history_valid: torch.Tensor | None = None,
    ) -> LaserSafetyFilterResult:
        B = current_command.shape[0]
        a = torch.as_tensor(self.a_tuple, device=current_command.device, dtype=current_command.dtype)
        sigma = torch.as_tensor(
            self.sigma_tuple, device=current_command.device, dtype=current_command.dtype
        )
        h_backup, lf_backup, lg_backup, backup_terms, _, _ = self._backup_terms_and_lie(
            ranges, angles, current_command
        )
        h_input, lf_input, lg_input, input_terms = self._input_lse_lie(current_command, a)
        h_aug, lf, lg = _softmin_value_lie(
            torch.stack((h_backup, h_input), dim=1),
            torch.stack((lf_backup, lf_input), dim=1),
            torch.stack((lg_backup, lg_input), dim=1),
            self.kappa_outer,
        )

        if previous_nominal_command is None:
            previous_nominal_command = nominal_command.detach()
        if nominal_history_valid is None:
            valid_history = torch.ones(
                B, 1, device=current_command.device, dtype=current_command.dtype
            )
        else:
            valid_history = nominal_history_valid.to(
                device=current_command.device, dtype=current_command.dtype
            )
            if valid_history.ndim == 1:
                valid_history = valid_history[:, None]
        nominal_derivative = (
            valid_history * (nominal_command - previous_nominal_command) / max(self.dt, self.eps)
        )
        desired_surrogate = (
            current_command
            + nominal_derivative / a
            + (sigma / a) * (nominal_command - current_command)
        )

        # lf + lg uhat + alpha(h) >= 0  <=>  -lg uhat <= lf + alpha(h).
        A = -lg[:, None, :]
        b = (lf + self.alpha_outer * h_aug)[:, None]
        weights = torch.ones(3, device=current_command.device, dtype=current_command.dtype)
        surrogate, diag = self.projector.project(desired_surrogate, A, b, weights)

        decay = torch.exp(-a * self.dt)
        command_unclipped = decay * current_command + (1.0 - decay) * surrogate
        limits_lower = torch.as_tensor(
            self.limits_lower, device=current_command.device, dtype=current_command.dtype
        )
        limits_upper = torch.as_tensor(
            self.limits_upper, device=current_command.device, dtype=current_command.dtype
        )
        violation = torch.relu(torch.maximum(limits_lower - command_unclipped, command_unclipped - limits_upper)).max(dim=1).values
        command = (
            torch.clamp(command_unclipped, limits_lower, limits_upper)
            if self.hard_clip_output else command_unclipped
        )
        correction = surrogate - desired_surrogate
        feasible = diag.max_violation <= self.projector.feasibility_tol
        return LaserSafetyFilterResult(
            command=command,
            surrogate_command=surrogate,
            desired_surrogate_command=desired_surrogate,
            h_lse=h_aug,
            h_backup_lse=h_backup,
            h_input_lse=h_input,
            individual_h=torch.cat((backup_terms, input_terms), dim=1),
            active=torch.linalg.norm(correction.detach(), dim=1) > 1.0e-6,
            correction_norm=torch.linalg.norm(correction.detach(), dim=1),
            cbf_value_nominal=-diag.max_violation.detach(),
            lambda_value=diag.active_rows.detach(),
            mu_value=torch.zeros_like(h_aug),
            input_bound_violation=violation,
            input_bound_active=violation > 1.0e-6,
            projection_feasible=feasible,
            projected_cbf_residual=(-diag.max_violation).detach(),
        )


class LaserZeroBackupLseControlDynamicsQpCfMatched(LaserZeroBackupLseCbfClosedForm):
    """CF-matched differentiable-QP baseline for the laser bCBF-LSE-CD filter.

    Builds the IDENTICAL single affine constraint that
    LaserZeroBackupLseCbfClosedForm.filter() solves in closed form -- same
    _backup_terms_and_lie (kappa_backup), same _input_lse_lie (kappa_input),
    same outer LSE aggregation (kappa_outer) into one scalar h_aug/Lf/Lg row,
    same alpha_outer, same desired_surrogate. This is the laser analogue of
    ZeroBackupBcbfQpFilter's "cd_lse_qp_cfmatched" variant in
    bcbf_qp_filter.py (see that module's docstring for the full derivation
    and scripts/test_cf_matched_qp_regression.py for the map-based
    forward/gradient regression tests against the closed form).

    Unlike LaserZeroBackupLseControlDynamicsQp (the existing QP variant,
    whose default training backend is SmallBoxQpProjector with a
    straight-through estimator), this class ALWAYS solves with
    QpthProjector -- an exact batched differentiable QP/KKT solve, with no
    .detach() on the actor path -- regardless of the qp_backend/
    straight_through/max_active_candidates constructor arguments, which are
    accepted only so LaserSafeSacActor.__init__ can pass this class the same
    kwargs shape as every other QP filter and are otherwise unused here.

    Also unlike LaserZeroBackupLseControlDynamicsQp (which does not
    reproduce it), this class explicitly reproduces the closed form's
    active-but-degenerate fallback (||Lg h_aug||^2 ~ 0 while still needing
    correction -> zero surrogate) as a post-solve override, since a generic
    QP solver has no notion of that project-specific convention and this
    variant is meant to be a true apples-to-apples ablation on the
    projection method alone.
    """

    def __init__(
        self,
        *,
        max_active_candidates: int = 18,
        feasibility_tolerance: float = 1.0e-6,
        qp_regularization: float = 1.0e-6,
        straight_through: bool = True,
        qp_backend: str = "active_set",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        from .bcbf_qp_filter import QpthProjector

        self.projector = QpthProjector(
            feasibility_tol=feasibility_tolerance,
            regularization=qp_regularization,
        )

    def filter(
        self,
        ranges: torch.Tensor,
        angles: torch.Tensor,
        current_command: torch.Tensor,
        nominal_command: torch.Tensor,
        previous_nominal_command: torch.Tensor | None = None,
        nominal_history_valid: torch.Tensor | None = None,
    ) -> LaserSafetyFilterResult:
        B = current_command.shape[0]
        a = torch.as_tensor(self.a_tuple, device=current_command.device, dtype=current_command.dtype)
        sigma = torch.as_tensor(
            self.sigma_tuple, device=current_command.device, dtype=current_command.dtype
        )
        h_backup, lf_backup, lg_backup, backup_terms, _, _ = self._backup_terms_and_lie(
            ranges, angles, current_command
        )
        h_input, lf_input, lg_input, input_terms = self._input_lse_lie(current_command, a)
        h_aug, lf, lg = _softmin_value_lie(
            torch.stack((h_backup, h_input), dim=1),
            torch.stack((lf_backup, lf_input), dim=1),
            torch.stack((lg_backup, lg_input), dim=1),
            self.kappa_outer,
        )

        if previous_nominal_command is None:
            previous_nominal_command = nominal_command.detach()
        if nominal_history_valid is None:
            valid_history = torch.ones(
                B, 1, device=current_command.device, dtype=current_command.dtype
            )
        else:
            valid_history = nominal_history_valid.to(
                device=current_command.device, dtype=current_command.dtype
            )
            if valid_history.ndim == 1:
                valid_history = valid_history[:, None]
        nominal_derivative = (
            valid_history * (nominal_command - previous_nominal_command) / max(self.dt, self.eps)
        )
        desired_surrogate = (
            current_command
            + nominal_derivative / a
            + (sigma / a) * (nominal_command - current_command)
        )

        omega = lf + torch.sum(lg * desired_surrogate, dim=1) + self.alpha_outer * h_aug
        denom = torch.sum(lg.square(), dim=1)
        needs_correction = omega < 0.0
        nondegenerate = denom > self.eps
        degenerate_active = needs_correction & (~nondegenerate)
        lambda_value = torch.where(
            needs_correction & nondegenerate,
            -omega / torch.clamp(denom, min=self.eps),
            torch.zeros_like(omega),
        )

        # lf + lg uhat + alpha(h) >= 0  <=>  -lg uhat <= lf + alpha(h).
        A = -lg[:, None, :]
        b = (lf + self.alpha_outer * h_aug)[:, None]
        weights = torch.ones(3, device=current_command.device, dtype=current_command.dtype)
        qp_surrogate, diag = self.projector.project(desired_surrogate, A, b, weights)
        surrogate = torch.where(
            degenerate_active[:, None], torch.zeros_like(qp_surrogate), qp_surrogate
        )

        decay = torch.exp(-a * self.dt)
        command_unclipped = decay * current_command + (1.0 - decay) * surrogate
        limits_lower = torch.as_tensor(
            self.limits_lower, device=current_command.device, dtype=current_command.dtype
        )
        limits_upper = torch.as_tensor(
            self.limits_upper, device=current_command.device, dtype=current_command.dtype
        )
        violation = torch.relu(torch.maximum(limits_lower - command_unclipped, command_unclipped - limits_upper)).max(dim=1).values
        command = (
            torch.clamp(command_unclipped, limits_lower, limits_upper)
            if self.hard_clip_output else command_unclipped
        )
        correction = surrogate - desired_surrogate
        feasible = diag.max_violation <= self.projector.feasibility_tol
        return LaserSafetyFilterResult(
            command=command,
            surrogate_command=surrogate,
            desired_surrogate_command=desired_surrogate,
            h_lse=h_aug,
            h_backup_lse=h_backup,
            h_input_lse=h_input,
            individual_h=torch.cat((backup_terms, input_terms), dim=1),
            active=needs_correction,
            correction_norm=torch.linalg.norm(correction, dim=1),
            cbf_value_nominal=omega,
            lambda_value=lambda_value,
            mu_value=torch.zeros_like(h_aug),
            input_bound_violation=violation,
            input_bound_active=violation > 1.0e-6,
            projection_feasible=feasible,
            projected_cbf_residual=(-diag.max_violation).detach(),
        )


class LaserZeroBackupCbfQp(LaserZeroBackupLseCbfClosedForm):
    """Seven-row LSE zero-backup CBF-QP with control dynamics.

    All current-time HOCBF, positive-time backup-flow, and terminal-time terms
    are aggregated into one soft-min barrier row.  The six command-state input
    limit CBFs remain individual rows, matching
    ``Model-Based/bcbf_lse_circle_obstacles_v4_cd.c``.  Thus the QP has exactly
    seven inequalities and three surrogate-command decision variables.
    """

    def __init__(
        self,
        *,
        max_active_candidates: int = 18,
        feasibility_tolerance: float = 1.0e-5,
        qp_regularization: float = 1.0e-5,
        straight_through: bool = True,
        qp_backend: str = "active_set",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Local import avoids coupling the closed-form module to the legacy
        # map-based QP code unless this method is selected.
        from .bcbf_qp_filter import OsqpProjector, QpthProjector, SmallBoxQpProjector

        if qp_backend == "active_set":
            self.projector = SmallBoxQpProjector(
                max_candidates=max_active_candidates,
                feasibility_tol=feasibility_tolerance,
                regularization=qp_regularization,
                straight_through=straight_through,
            )
        elif qp_backend == "osqp":
            self.projector = OsqpProjector(
                feasibility_tol=feasibility_tolerance,
                regularization=qp_regularization,
                straight_through=straight_through,
            )
        elif qp_backend == "qpth":
            self.projector = QpthProjector(
                feasibility_tol=feasibility_tolerance,
                regularization=qp_regularization,
            )
        else:
            raise ValueError(f"Unknown QP backend {qp_backend!r}")
        self.qp_backend = qp_backend

    def _individual_input_rows(
        self,
        current_command: torch.Tensor,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw input barriers and their individual QP rows."""
        B = current_command.shape[0]
        limits_lower = torch.as_tensor(
            self.limits_lower, device=current_command.device, dtype=current_command.dtype
        )
        limits_upper = torch.as_tensor(
            self.limits_upper, device=current_command.device, dtype=current_command.dtype
        )
        terms = torch.stack(
            tuple(
                value
                for j in range(3)
                for value in (
                    (limits_upper[j] - current_command[:, j]) / limits_upper[j].abs(),
                    (current_command[:, j] - limits_lower[j]) / limits_lower[j].abs(),
                )
            ),
            dim=1,
        )
        grads_u = torch.zeros(B, 6, 3, device=current_command.device, dtype=current_command.dtype)
        for j in range(3):
            grads_u[:, 2 * j, j] = -1.0 / limits_upper[j].abs()
            grads_u[:, 2 * j + 1, j] = 1.0 / limits_lower[j].abs()
        drift_u = -a * current_command
        lf = torch.sum(grads_u * drift_u[:, None, :], dim=-1)
        lg = grads_u * a[None, None, :]
        return terms, lf, lg, limits_lower, limits_upper

    def filter(
        self,
        ranges: torch.Tensor,
        angles: torch.Tensor,
        current_command: torch.Tensor,
        nominal_command: torch.Tensor,
        previous_nominal_command: torch.Tensor | None = None,
        nominal_history_valid: torch.Tensor | None = None,
    ) -> LaserSafetyFilterResult:
        B = current_command.shape[0]
        a = torch.as_tensor(self.a_tuple, device=current_command.device, dtype=current_command.dtype)
        sigma = torch.as_tensor(
            self.sigma_tuple, device=current_command.device, dtype=current_command.dtype
        )
        _, _, _, backup_terms, backup_lf, backup_lg = self._backup_terms_and_lie(
            ranges, angles, current_command
        )
        input_terms, input_lf, input_lg, limits_lower, limits_upper = self._individual_input_rows(
            current_command, a
        )

        h_backup, lf_backup, lg_backup = _softmin_value_lie(
            backup_terms, backup_lf, backup_lg, self.kappa_backup
        )
        # One backup LSE row plus six individual command-state input rows.
        # The six input rows use alpha=a (the per-axis control-dynamics gain,
        # matching the historical Model-Based/bcbf_lse_circle_obstacles_v4_cd.c
        # reference's dedicated input-row alpha, distinct from alpha_outer),
        # NOT alpha_outer -- alpha_outer is calibrated for the aggregated
        # backup+input LSE row used by the closed-form/LSE-QP filters, where
        # the six raw input rows are softened through kappa_input before
        # alpha_outer ever applies to them, and that softmin combination
        # cancels almost exactly at the box center (u=0), leaving that row
        # nearly inert there. This class enforces the six rows individually
        # (that is the point of the QP), so reusing alpha_outer directly on
        # the raw, un-softened rows made them ~10x too restrictive even at
        # the box center, far from any real obstacle or input-limit
        # constraint (verified: with alpha_outer=0.8, a=8.0, limit=0.5, the
        # row bound at u=0 is 0.05 instead of the physical 0.5). Setting each
        # row's alpha to that axis's own `a` makes the bound reduce to
        # exactly the physical limit independent of current_command, i.e.
        # "surrogate must lie in the box" -- the intended behavior.
        alpha_input_rows = a.repeat_interleave(2)
        all_terms = torch.cat((h_backup[:, None], input_terms), dim=1)
        all_lf = torch.cat((lf_backup[:, None], input_lf), dim=1)
        all_lg = torch.cat((lg_backup[:, None, :], input_lg), dim=1)
        alpha_rows = torch.cat((
            torch.full((1,), self.alpha_outer, device=a.device, dtype=a.dtype),
            alpha_input_rows,
        ))
        # lf_i + lg_i uhat + alpha_i h_i >= 0 <=> -lg_i uhat <= lf_i+alpha_i h_i.
        A = -all_lg
        b = all_lf + alpha_rows[None, :] * all_terms

        if previous_nominal_command is None:
            previous_nominal_command = nominal_command.detach()
        if nominal_history_valid is None:
            valid_history = torch.ones(B, 1, device=current_command.device, dtype=current_command.dtype)
        else:
            valid_history = nominal_history_valid.to(
                device=current_command.device, dtype=current_command.dtype
            )
            if valid_history.ndim == 1:
                valid_history = valid_history[:, None]
        nominal_derivative = (
            valid_history * (nominal_command - previous_nominal_command) / max(self.dt, self.eps)
        )
        desired_surrogate = (
            current_command
            + nominal_derivative / a
            + (sigma / a) * (nominal_command - current_command)
        )
        weights = torch.ones(3, device=current_command.device, dtype=current_command.dtype)
        surrogate, diag = self.projector.project(desired_surrogate, A, b, weights)

        decay = torch.exp(-a * self.dt)
        command_unclipped = decay * current_command + (1.0 - decay) * surrogate
        violation = torch.relu(torch.maximum(limits_lower - command_unclipped, command_unclipped - limits_upper)).max(dim=1).values
        command = (
            torch.clamp(command_unclipped, limits_lower, limits_upper)
            if self.hard_clip_output else command_unclipped
        )
        correction = surrogate - desired_surrogate
        hard_input = input_terms.min(dim=1).values
        hard_all = torch.minimum(h_backup, hard_input)
        feasible = diag.max_violation <= self.projector.feasibility_tol
        return LaserSafetyFilterResult(
            command=command,
            surrogate_command=surrogate,
            desired_surrogate_command=desired_surrogate,
            h_lse=hard_all,
            h_backup_lse=h_backup,
            h_input_lse=hard_input,
            individual_h=all_terms,
            active=torch.linalg.norm(correction.detach(), dim=1) > 1.0e-6,
            correction_norm=torch.linalg.norm(correction.detach(), dim=1),
            cbf_value_nominal=-diag.max_violation.detach(),
            lambda_value=diag.active_rows.detach(),
            mu_value=torch.zeros_like(hard_all),
            input_bound_violation=violation,
            input_bound_active=violation > 1.0e-6,
            projection_feasible=feasible,
            projected_cbf_residual=(-diag.max_violation).detach(),
        )


class LaserZeroBackupNonLseControlDynamicsQp(LaserZeroBackupCbfQp):
    """Fully unaggregated local-scan bCBF-QP with control dynamics.

    Every current HOCBF, positive-time backup-flow, and terminal-time ray term
    is retained as an individual affine QP row.  The six command-state input
    barriers are also individual rows.  With the standard 120 rays and 21
    backup samples and ``terminal_backup_mode="legacy_inflated_ellipse"``
    this produces ``(21 + 1) * 120 + 6 = 2646`` inequalities: 120 current
    HOCBF rows, 20*120 future rows, 120 terminal rows, and six input rows.
    With ``terminal_backup_mode="invariant_ellipse"`` the single terminal
    term per ray is replaced by two (h_B_geom, h_B_rec), adding another 120
    rows for a total of 2766. No LSE value participates in the QP
    construction.
    """

    def filter(
        self,
        ranges: torch.Tensor,
        angles: torch.Tensor,
        current_command: torch.Tensor,
        nominal_command: torch.Tensor,
        previous_nominal_command: torch.Tensor | None = None,
        nominal_history_valid: torch.Tensor | None = None,
    ) -> LaserSafetyFilterResult:
        batch_size = current_command.shape[0]
        a = torch.as_tensor(
            self.a_tuple, device=current_command.device, dtype=current_command.dtype
        )
        sigma = torch.as_tensor(
            self.sigma_tuple, device=current_command.device, dtype=current_command.dtype
        )
        _, _, _, backup_terms, backup_lf, backup_lg = self._backup_terms_and_lie(
            ranges, angles, current_command, aggregate=False
        )
        input_terms, input_lf, input_lg, limits_lower, limits_upper = self._individual_input_rows(
            current_command, a
        )

        all_terms = torch.cat((backup_terms, input_terms), dim=1)
        all_lf = torch.cat((backup_lf, input_lf), dim=1)
        all_lg = torch.cat((backup_lg, input_lg), dim=1)
        # Same fix as LaserZeroBackupCbfQp.filter(): the six individual
        # input-limit rows use alpha=a (per-axis control-dynamics gain), not
        # alpha_outer -- see that method's comment for the full derivation.
        num_backup_rows = backup_terms.shape[1]
        alpha_input_rows = a.repeat_interleave(2)
        alpha_rows = torch.cat((
            torch.full((num_backup_rows,), self.alpha_outer, device=a.device, dtype=a.dtype),
            alpha_input_rows,
        ))
        # lf_i + lg_i uhat + alpha_i*h_i >= 0  <=>  -lg_i uhat <= lf_i + alpha_i*h_i.
        A = -all_lg
        b = all_lf + alpha_rows[None, :] * all_terms

        if previous_nominal_command is None:
            previous_nominal_command = nominal_command.detach()
        if nominal_history_valid is None:
            valid_history = torch.ones(
                batch_size, 1, device=current_command.device, dtype=current_command.dtype
            )
        else:
            valid_history = nominal_history_valid.to(
                device=current_command.device, dtype=current_command.dtype
            )
            if valid_history.ndim == 1:
                valid_history = valid_history[:, None]
        nominal_derivative = (
            valid_history
            * (nominal_command - previous_nominal_command)
            / max(self.dt, self.eps)
        )
        desired_surrogate = (
            current_command
            + nominal_derivative / a
            + (sigma / a) * (nominal_command - current_command)
        )
        weights = torch.ones(
            3, device=current_command.device, dtype=current_command.dtype
        )
        surrogate, diag = self.projector.project(desired_surrogate, A, b, weights)

        decay = torch.exp(-a * self.dt)
        command_unclipped = decay * current_command + (1.0 - decay) * surrogate
        violation = torch.relu(torch.maximum(limits_lower - command_unclipped, command_unclipped - limits_upper)).max(dim=1).values
        command = (
            torch.clamp(command_unclipped, limits_lower, limits_upper)
            if self.hard_clip_output
            else command_unclipped
        )
        correction = surrogate - desired_surrogate
        hard_backup = backup_terms.min(dim=1).values
        hard_input = input_terms.min(dim=1).values
        hard_all = all_terms.min(dim=1).values
        feasible = diag.max_violation <= self.projector.feasibility_tol
        return LaserSafetyFilterResult(
            command=command,
            surrogate_command=surrogate,
            desired_surrogate_command=desired_surrogate,
            # These compatibility fields are hard minima for this non-LSE method.
            h_lse=hard_all,
            h_backup_lse=hard_backup,
            h_input_lse=hard_input,
            individual_h=all_terms,
            active=torch.linalg.norm(correction.detach(), dim=1) > 1.0e-6,
            correction_norm=torch.linalg.norm(correction.detach(), dim=1),
            cbf_value_nominal=-diag.max_violation.detach(),
            lambda_value=diag.active_rows.detach(),
            mu_value=torch.zeros_like(hard_all),
            input_bound_violation=violation,
            input_bound_active=violation > 1.0e-6,
            projection_feasible=feasible,
            projected_cbf_residual=(-diag.max_violation).detach(),
        )


class LaserZeroBackupLseQp(LaserScanCbfLseQp):
    """Zero-backup LiDAR LSE-CBF-QP WITHOUT control dynamics.

    Laser port of src/high_level_rl/bcbf_qp_filter.py's ``noncd_lse_qp``
    variant (direct-command bCBF-QP, one LSE backup row): the QP decision is
    the directly-applied body command u=[vx,vy,w] -- there is no first-order
    command-state surrogate/control-dynamics state, unlike
    LaserZeroBackupLseControlDynamicsQpCfMatched ("bCBF-QP" in
    scripts/benchmark_safety_filter_latency.py).

    Under this kinematic (no-lag) model the backup law is exactly zero
    velocity, so the robot pose -- and every scan-fixed ray point, since the
    LiDAR is body-mounted -- stays fixed for the whole backup horizon: there
    is no flow to integrate and no sensitivity matrix to propagate (both the
    exact closed-form flow of LaserZeroBackupLseCbfClosedForm and this
    class's backup-rollout cost collapse to repeating the SAME current-time
    barrier). Concretely:
      * `backup_samples` identical copies of the current-time ray barrier
        (`current_h_and_direct_row`) -- kept only for row-count/compute
        parity with the control-dynamics variant's distinct backup-flow
        samples, since softmin-aggregating identical terms is a no-op;
      * one terminal ray barrier, same points, footprint inflated by
        `backup_margin` (no residual-velocity stopping-distance term --
        there is no coasting velocity to account for when the backup
        command is exactly zero);
    all soft-min aggregated (kappa_backup) into a single row, then combined
    with the six hard command-box rows into a 7-row QP -- the direct-command
    analogue of LaserZeroBackupCbfQp (which keeps the control-dynamics
    surrogate state).

    Accepts the same 6-positional-argument ``filter(...)`` signature as
    every control-dynamics laser filter so it is a drop-in addition to
    benchmark/ablation harnesses that call every filter uniformly;
    `current_command`, `previous_nominal_command`, and
    `nominal_history_valid` are accepted but unused (there is no command
    state here for them to describe).
    """

    def __init__(
        self,
        *,
        backup_margin: float = 0.3,
        backup_samples: int = 21,
        kappa_backup: float = 50.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.backup_margin = float(backup_margin)
        self.backup_samples = max(int(backup_samples), 1)
        self.kappa_backup = float(kappa_backup)

    def _terminal_h_and_direct_row(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Same ellipse barrier as `current_h_and_direct_row`, with both
        semi-axes inflated by `backup_margin` (no velocity/stopping-distance
        term: the zero-velocity backup command never coasts)."""
        ax, ay, _, _ = self._constants(points)
        ax = ax + self.backup_margin
        ay = ay + self.backup_margin
        px = 1.0 / (ax * ax)
        py = 1.0 / (ay * ay)
        x = points[..., 0]
        y = points[..., 1]
        h = px * x.square() + py * y.square() - 1.0
        cross = (py - px) * x * y
        row = torch.stack((-2.0 * px * x, -2.0 * py * y, -2.0 * cross), dim=-1)
        return h, row

    def filter(
        self,
        ranges: torch.Tensor,
        angles: torch.Tensor,
        current_command: torch.Tensor,
        nominal_command: torch.Tensor,
        previous_nominal_command: torch.Tensor | None = None,
        nominal_history_valid: torch.Tensor | None = None,
    ) -> LaserSafetyFilterResult:
        del current_command, previous_nominal_command, nominal_history_valid
        B = nominal_command.shape[0]
        points, valid = self.scan_points(
            ranges, angles, B,
            device=nominal_command.device, dtype=nominal_command.dtype,
        )
        h, lg = self.current_h_and_direct_row(points)
        h_term, lg_term = self._terminal_h_and_direct_row(points)
        h, lg = self.mask_terms(h, valid, lg)
        h_term, lg_term = self.mask_terms(h_term, valid, lg_term)

        terms = torch.cat([h] * self.backup_samples + [h_term], dim=1)
        lg_terms = torch.cat([lg] * self.backup_samples + [lg_term], dim=1)
        zeros = torch.zeros_like(terms)
        h_backup, _, lg_backup = _softmin_value_lie(terms, zeros, lg_terms, self.kappa_backup)

        limits_lower = torch.as_tensor(
            self.limits_lower, device=nominal_command.device, dtype=nominal_command.dtype
        )
        limits_upper = torch.as_tensor(
            self.limits_upper, device=nominal_command.device, dtype=nominal_command.dtype
        )
        eye = torch.eye(3, device=nominal_command.device, dtype=nominal_command.dtype)
        A_box = torch.cat((eye, -eye), dim=0).unsqueeze(0).expand(B, -1, -1)
        b_box = torch.cat((limits_upper, -limits_lower)).unsqueeze(0).expand(B, -1)
        A = torch.cat((-lg_backup[:, None, :], A_box), dim=1)
        b = torch.cat(((self.alpha * h_backup)[:, None], b_box), dim=1)
        weights = torch.ones(3, device=nominal_command.device, dtype=nominal_command.dtype)
        command_unclipped, diag = self.projector.project(nominal_command, A, b, weights)
        violation = torch.relu(
            torch.maximum(limits_lower - command_unclipped, command_unclipped - limits_upper)
        ).max(dim=1).values
        command = (
            torch.clamp(command_unclipped, limits_lower, limits_upper)
            if self.hard_clip_output else command_unclipped
        )
        correction = command_unclipped - nominal_command
        feasible = diag.max_violation <= self.projector.feasibility_tol
        return LaserSafetyFilterResult(
            command=command,
            surrogate_command=command_unclipped,
            desired_surrogate_command=nominal_command,
            h_lse=h_backup,
            h_backup_lse=h_backup,
            individual_h=h,
            active=torch.linalg.norm(correction.detach(), dim=1) > 1.0e-6,
            correction_norm=torch.linalg.norm(correction.detach(), dim=1),
            cbf_value_nominal=-diag.max_violation.detach(),
            lambda_value=diag.active_rows.detach(),
            input_bound_violation=violation,
            input_bound_active=violation > 1.0e-6,
            projection_feasible=feasible,
            projected_cbf_residual=(-diag.max_violation).detach(),
        )
