from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ArenaConfig, BackupSafetyConfig, HighLevelActionConfig
from .zero_backup_flow import exact_zero_backup_state_and_phi


# Fallback values are used only when an older BackupSafetyConfig does not yet
# expose the corresponding fields. Values in config.py take precedence.
_DEFAULT_KAPPA_BACKUP = 20.0
_DEFAULT_KAPPA_INPUT = 20.0
_DEFAULT_KAPPA_OUTER = 50.0
_DEFAULT_ALPHA_OUTER = 0.8
_DEFAULT_HOCBF_K0 = 2.0


@dataclass
class BackupSafetyFilterResult:
    command: torch.Tensor
    surrogate_command: torch.Tensor
    desired_surrogate_command: torch.Tensor
    h_lse: torch.Tensor
    h_backup_lse: torch.Tensor
    h_input_lse: torch.Tensor
    individual_h: torch.Tensor
    active: torch.Tensor
    correction_norm: torch.Tensor
    cbf_value_nominal: torch.Tensor
    lambda_value: torch.Tensor
    mu_value: torch.Tensor
    input_bound_violation: torch.Tensor
    input_bound_active: torch.Tensor


class ZeroBackupLseCbfClosedForm:
    """Differentiable augmented-state zero-backup bCBF-LSE-CF layer.

    State ``z=[x,y,yaw,vx,vy,wz]``. The backup surrogate input is zero, so
    during the backup rollout ``u_dot=-A u``.

    Proposed-framework details implemented here:

    * The current-time (tau=0) physical clearances are relative degree two
      with respect to the surrogate input. They are replaced in the backup
      LSE by the lifted terms ``psi_1 = h_dot + k0 h``.
    * Positive-time rollout clearances remain ordinary flow-composed terms;
      they are generally relative degree one through the backup-flow
      sensitivity.
    * The terminal backup functions depend on both clearance and the
      residual translational command through the conservative stopping
      distance ``||u_xy||/a_min``. No separate near-stop barrier is used.
    * The final controller is the ordinary one-constraint CBF KKT solution,
      without the relaxed-CBF ``mu h`` slack term.
    """

    def __init__(
        self,
        arena: ArenaConfig,
        action_cfg: HighLevelActionConfig,
        cfg: BackupSafetyConfig,
    ) -> None:
        self.arena = arena
        self.action_cfg = action_cfg
        self.cfg = cfg

    def _cfg_float(self, name: str, default: float) -> float:
        return float(getattr(self.cfg, name, default))

    def _action_bounds(
        self, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-axis (lower, upper) command bounds. vx is asymmetric
        (min_vx, max_vx); vy/wz stay symmetric (+-max_vy, +-max_wz)."""
        lower = torch.tensor(
            [self.action_cfg.min_vx, -self.action_cfg.max_vy, -self.action_cfg.max_wz],
            device=reference.device,
            dtype=reference.dtype,
        )
        upper = torch.tensor(
            [self.action_cfg.max_vx, self.action_cfg.max_vy, self.action_cfg.max_wz],
            device=reference.device,
            dtype=reference.dtype,
        )
        return lower, upper

    @staticmethod
    def _softmin_value_grad(
        terms: torch.Tensor,
        grads: torch.Tensor,
        kappa: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scaled = -float(kappa) * terms
        weights = torch.softmax(scaled, dim=1)
        value = -torch.logsumexp(scaled, dim=1) / float(kappa)
        grad = torch.sum(weights[..., None] * grads, dim=1)
        return value, grad

    def _primitive_clearances_and_future_grads(
        self,
        z: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
        obstacle_mask: torch.Tensor,
        terminal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return primitive flow/terminal terms and future-state gradients.

        Circular obstacles use the same squared-distance barrier as the v4 C
        implementation. For a nonterminal rollout sample,

            h_i = ||p-p_i||^2 - r_safe,i^2.

        At the terminal time, the effective obstacle radius additionally
        contains the backup margin and a conservative residual stopping
        distance ||u_xy||/a_min. Walls remain linear clearance barriers and
        use the analogous stopping-distance subtraction at the terminal time.
        """
        eps = self._cfg_float("denominator_epsilon", 1.0e-9)
        p = z[:, :2]
        rel = p[:, None, :] - obstacle_centers
        dist_sq = torch.sum(rel * rel, dim=-1)

        base_radii = obstacle_radii + float(self.arena.obstacle_clearance)
        if terminal:
            a = torch.as_tensor(
                self.cfg.control_dynamics_a, device=z.device, dtype=z.dtype
            )
            a_min = torch.clamp(torch.min(a[:2]), min=1.0e-9)
            u_xy = z[:, 3:5]
            speed = torch.sqrt(torch.sum(u_xy * u_xy, dim=1) + eps)
            stopping_distance = speed / a_min
            effective_radii = (
                base_radii
                + self._cfg_float("backup_margin", 0.15)
                + stopping_distance[:, None]
            )
        else:
            speed = None
            stopping_distance = None
            a_min = None
            effective_radii = base_radii

        h_obs = dist_sq - effective_radii.square()
        grad_obs_xy = 2.0 * rel

        h_obs = torch.where(obstacle_mask, h_obs, torch.full_like(h_obs, 1.0e3))
        grad_obs_xy = torch.where(
            obstacle_mask[..., None], grad_obs_xy, torch.zeros_like(grad_obs_xy)
        )

        x, y = p.unbind(dim=-1)
        wall_margin = float(self.arena.wall_clearance)
        if terminal:
            wall_margin += self._cfg_float("backup_margin", 0.15)

        h_walls = torch.stack(
            (
                x - float(self.arena.x_min) - wall_margin,
                float(self.arena.x_max) - x - wall_margin,
                y - float(self.arena.y_min) - wall_margin,
                float(self.arena.y_max) - y - wall_margin,
            ),
            dim=1,
        )
        if terminal:
            h_walls = h_walls - stopping_distance[:, None]

        grad_walls_xy = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            device=z.device,
            dtype=z.dtype,
        ).expand(z.shape[0], -1, -1)

        h = torch.cat((h_obs, h_walls), dim=1)
        grad_future = torch.zeros(
            z.shape[0], h.shape[1], 6, device=z.device, dtype=z.dtype
        )
        n_obs = h_obs.shape[1]
        grad_future[:, :n_obs, :2] = grad_obs_xy
        grad_future[:, n_obs:, :2] = grad_walls_xy

        if terminal:
            # Obstacle terminal derivative:
            # d[-r_eff^2]/du = -2 r_eff * u/(a_min ||u||).
            grad_obs_u = (
                -2.0
                * effective_radii[..., None]
                * z[:, None, 3:5]
                / (a_min * speed[:, None, None])
            )
            grad_obs_u = torch.where(
                obstacle_mask[..., None], grad_obs_u, torch.zeros_like(grad_obs_u)
            )
            grad_future[:, :n_obs, 3:5] = grad_obs_u

            # Wall terminal derivative of -||u_xy||/a_min.
            grad_wall_u = -z[:, 3:5] / (a_min * speed[:, None])
            grad_future[:, n_obs:, 3:5] = grad_wall_u[:, None, :]

        return h, grad_future

    def _tau0_lifted_terms_and_grads(
        self,
        z0: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
        obstacle_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return relative-degree-one HOCBF terms at tau=0.

        Circular obstacles use h=||p-p_i||^2-r_safe^2, so
        h_dot=2(p-p_i)^T v_world. Walls use linear clearances.

        The gradient is the exact hand-derived Jacobian of this closed-form
        expression w.r.t. z0 (batched directly, no autodiff transform).
        """
        k0 = self._cfg_float("hocbf_k0", _DEFAULT_HOCBF_K0)
        batch = z0.shape[0]

        p = z0[:, :2]
        yaw = z0[:, 2]
        vx, vy = z0[:, 3], z0[:, 4]
        c, s = torch.cos(yaw), torch.sin(yaw)
        velocity_w = torch.stack((c * vx - s * vy, s * vx + c * vy), dim=-1)

        rel = p[:, None, :] - obstacle_centers
        rel_x, rel_y = rel[..., 0], rel[..., 1]
        safe_radii = obstacle_radii + float(self.arena.obstacle_clearance)
        h_obs = torch.sum(rel * rel, dim=-1) - safe_radii.square()
        hdot_obs = 2.0 * torch.sum(rel * velocity_w[:, None, :], dim=-1)
        psi_obs = hdot_obs + k0 * h_obs
        psi_obs = torch.where(obstacle_mask, psi_obs, torch.full_like(psi_obs, 1.0e3))

        dpsi_obs_dyaw = 2.0 * (
            rel_x * (-s[:, None] * vx[:, None] - c[:, None] * vy[:, None])
            + rel_y * (c[:, None] * vx[:, None] - s[:, None] * vy[:, None])
        )
        dpsi_obs_dvx = 2.0 * (rel_x * c[:, None] + rel_y * s[:, None])
        dpsi_obs_dvy = 2.0 * (-rel_x * s[:, None] + rel_y * c[:, None])

        grad_obs = torch.zeros(
            batch, obstacle_centers.shape[1], 6, device=z0.device, dtype=z0.dtype
        )
        grad_obs[..., 0] = 2.0 * velocity_w[:, None, 0] + 2.0 * k0 * rel_x
        grad_obs[..., 1] = 2.0 * velocity_w[:, None, 1] + 2.0 * k0 * rel_y
        grad_obs[..., 2] = dpsi_obs_dyaw
        grad_obs[..., 3] = dpsi_obs_dvx
        grad_obs[..., 4] = dpsi_obs_dvy
        grad_obs = torch.where(
            obstacle_mask[..., None], grad_obs, torch.zeros_like(grad_obs)
        )

        x, y = p[:, 0], p[:, 1]
        h_walls = torch.stack(
            (
                x - float(self.arena.x_min) - float(self.arena.wall_clearance),
                float(self.arena.x_max) - x - float(self.arena.wall_clearance),
                y - float(self.arena.y_min) - float(self.arena.wall_clearance),
                float(self.arena.y_max) - y - float(self.arena.wall_clearance),
            ),
            dim=1,
        )
        wall_normals = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            device=z0.device,
            dtype=z0.dtype,
        )
        hdot_walls = velocity_w @ wall_normals.T
        psi_walls = hdot_walls + k0 * h_walls

        dvw_dyaw = torch.stack((-s * vx - c * vy, c * vx - s * vy), dim=-1)
        dvw_dvx = torch.stack((c, s), dim=-1)
        dvw_dvy = torch.stack((-s, c), dim=-1)

        grad_walls = torch.zeros(batch, 4, 6, device=z0.device, dtype=z0.dtype)
        grad_walls[:, :, 0] = k0 * wall_normals[:, 0][None, :]
        grad_walls[:, :, 1] = k0 * wall_normals[:, 1][None, :]
        grad_walls[:, :, 2] = dvw_dyaw @ wall_normals.T
        grad_walls[:, :, 3] = dvw_dvx @ wall_normals.T
        grad_walls[:, :, 4] = dvw_dvy @ wall_normals.T

        terms = torch.cat((psi_obs, psi_walls), dim=1)
        grads = torch.cat((grad_obs, grad_walls), dim=1)
        return terms, grads

    def _backup_rollout_terms_and_grads(
        self,
        position_w: torch.Tensor,
        yaw: torch.Tensor,
        current_command: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
        obstacle_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z0 = torch.cat((position_w, yaw[:, None], current_command), dim=1)
        batch = z0.shape[0]

        samples = max(int(self.cfg.backup_samples), 2)
        horizon = float(self.cfg.backup_horizon_s)
        a = torch.as_tensor(self.cfg.control_dynamics_a, device=z0.device, dtype=z0.dtype)

        # tau=0: replace raw RD2 physical constraints by the lifted RD1 terms.
        tau0_terms, tau0_grads = self._tau0_lifted_terms_and_grads(
            z0, obstacle_centers, obstacle_radii, obstacle_mask
        )

        # Positive-time rollout constraints on the original safe set. All
        # samples are independent closed-form evaluations of the same flow,
        # so they are computed in one vectorized call instead of a Python
        # loop over samples plus per-sample autodiff.
        num_samples = samples - 1
        times = horizon * torch.arange(
            1, samples, device=z0.device, dtype=z0.dtype
        ) / float(samples - 1)
        z_t, phi_t = exact_zero_backup_state_and_phi(z0, times, a)  # (B,K,6), (B,K,6,6)

        n_obs = obstacle_centers.shape[1]
        z_flat = z_t.reshape(batch * num_samples, 6)
        centers_flat = (
            obstacle_centers.unsqueeze(1)
            .expand(batch, num_samples, n_obs, 2)
            .reshape(batch * num_samples, n_obs, 2)
        )
        radii_flat = (
            obstacle_radii.unsqueeze(1)
            .expand(batch, num_samples, n_obs)
            .reshape(batch * num_samples, n_obs)
        )
        mask_flat = (
            obstacle_mask.unsqueeze(1)
            .expand(batch, num_samples, n_obs)
            .reshape(batch * num_samples, n_obs)
        )
        h_flat, grad_future_flat = self._primitive_clearances_and_future_grads(
            z_flat, centers_flat, radii_flat, mask_flat, terminal=False
        )
        n_terms = h_flat.shape[1]
        h_nonterm = h_flat.reshape(batch, num_samples * n_terms)
        grad_future_nonterm = grad_future_flat.reshape(batch, num_samples, n_terms, 6)
        grad_current_nonterm = torch.einsum(
            "bkni,bkij->bknj", grad_future_nonterm, phi_t
        ).reshape(batch, num_samples * n_terms, 6)

        # Terminal augmented backup-set terms with stopping-distance
        # dependence, reusing the already-computed state/Jacobian at the
        # last sample time, which equals the horizon.
        z_terminal = z_t[:, -1, :]
        phi_terminal = phi_t[:, -1, :, :]
        h_terminal, grad_terminal_future = self._primitive_clearances_and_future_grads(
            z_terminal,
            obstacle_centers,
            obstacle_radii,
            obstacle_mask,
            terminal=True,
        )
        grad_terminal = torch.einsum(
            "bni,bij->bnj", grad_terminal_future, phi_terminal
        )

        terms = torch.cat((tau0_terms, h_nonterm, h_terminal), dim=1)
        grads = torch.cat((tau0_grads, grad_current_nonterm, grad_terminal), dim=1)
        return terms, grads

    def _input_lse(
        self,
        current_command: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lower, upper = self._action_bounds(current_command)
        terms = torch.stack(
            (
                upper[:, None] - current_command.T,
                current_command.T - lower[:, None],
            ),
            dim=2,
        ).permute(1, 0, 2).reshape(current_command.shape[0], 6)

        grads = torch.zeros(
            current_command.shape[0],
            6,
            6,
            device=current_command.device,
            dtype=current_command.dtype,
        )
        for j in range(3):
            grads[:, 2 * j, 3 + j] = -1.0
            grads[:, 2 * j + 1, 3 + j] = 1.0
        return self._softmin_value_grad(
            terms, grads, self._cfg_float("kappa_input", _DEFAULT_KAPPA_INPUT)
        )

    def _current_original_barriers(
        self,
        position_w: torch.Tensor,
        yaw: torch.Tensor,
        current_command: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
        obstacle_mask: torch.Tensor,
    ) -> torch.Tensor:
        z = torch.cat((position_w, yaw[:, None], current_command), dim=1)
        h, _ = self._primitive_clearances_and_future_grads(
            z, obstacle_centers, obstacle_radii, obstacle_mask, terminal=False
        )
        return h

    def filter(
        self,
        position_w: torch.Tensor,
        yaw: torch.Tensor,
        current_command: torch.Tensor,
        nominal_command: torch.Tensor,
        previous_nominal_command: torch.Tensor,
        nominal_history_valid: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
        obstacle_mask: torch.Tensor,
    ) -> BackupSafetyFilterResult:
        eps = float(self.cfg.denominator_epsilon)
        terms, grads = self._backup_rollout_terms_and_grads(
            position_w,
            yaw,
            current_command,
            obstacle_centers,
            obstacle_radii,
            obstacle_mask,
        )
        h_backup, grad_backup = self._softmin_value_grad(
            terms, grads, self._cfg_float("kappa_backup", _DEFAULT_KAPPA_BACKUP)
        )
        h_input, grad_input = self._input_lse(current_command)

        outer_terms = torch.stack((h_backup, h_input), dim=1)
        outer_grads = torch.stack((grad_backup, grad_input), dim=1)
        h_aug, grad = self._softmin_value_grad(
            outer_terms, outer_grads, self._cfg_float("kappa_outer", _DEFAULT_KAPPA_OUTER)
        )

        a = torch.tensor(
            self.cfg.control_dynamics_a,
            device=current_command.device,
            dtype=current_command.dtype,
        )
        sigma = torch.tensor(
            self.cfg.tracking_sigma,
            device=current_command.device,
            dtype=current_command.dtype,
        )

        theta = yaw
        vx, vy, wz = current_command.unbind(dim=1)
        c, s = torch.cos(theta), torch.sin(theta)
        drift = torch.stack(
            (
                c * vx - s * vy,
                s * vx + c * vy,
                wz,
                -a[0] * vx,
                -a[1] * vy,
                -a[2] * wz,
            ),
            dim=1,
        )
        lf = torch.sum(grad * drift, dim=1)
        lg = grad[:, 3:] * a

        dt = float(self.cfg.integration_dt)
        valid = nominal_history_valid.to(
            device=nominal_command.device, dtype=nominal_command.dtype
        )
        if valid.ndim == 1:
            valid = valid[:, None]
        nominal_derivative = (
            valid * (nominal_command - previous_nominal_command) / dt
        )
        desired_surrogate = (
            current_command
            + nominal_derivative / a
            + (sigma / a) * (nominal_command - current_command)
        )

        omega = (
            lf
            + torch.sum(lg * desired_surrogate, dim=1)
            + self._cfg_float("alpha_outer", _DEFAULT_ALPHA_OUTER) * h_aug
        )

        # Ordinary single-constraint CBF KKT solution (no relaxed-CBF slack).
        # Optional diagonal cost weights preserve the general W^{-1} formula;
        # older configs default to W=I.
        cost_weights = torch.as_tensor(
            getattr(self.cfg, "surrogate_cost_weights", (1.0, 1.0, 1.0)),
            device=lg.device,
            dtype=lg.dtype,
        )
        cost_weights = torch.clamp(cost_weights, min=eps)
        denominator_raw = torch.sum(lg.square() / cost_weights, dim=1)
        needs_correction = omega < 0.0
        nondegenerate = denominator_raw > eps
        lambda_value = torch.where(
            needs_correction & nondegenerate,
            -omega / torch.clamp(denominator_raw, min=eps),
            torch.zeros_like(omega),
        )
        corrected_surrogate = (
            desired_surrogate
            + lambda_value[:, None] * lg / cost_weights
        )

        # Match the C fallback in the rare active-but-degenerate case: apply
        # the zero surrogate backup rather than an undefined KKT correction.
        degenerate_active = needs_correction & (~nondegenerate)
        surrogate = torch.where(
            degenerate_active[:, None],
            torch.zeros_like(corrected_surrogate),
            corrected_surrogate,
        )
        mu = torch.zeros_like(lambda_value)

        # Exact zero-order-hold integration over one high-level interval.
        decay = torch.exp(-a * dt)
        command_unclipped = decay * current_command + (1.0 - decay) * surrogate

        lower, upper = self._action_bounds(command_unclipped)
        input_bound_violation = torch.maximum(
            torch.relu(command_unclipped - upper),
            torch.relu(lower - command_unclipped),
        ).max(dim=1).values
        # Numerical guard matching the C implementation. Frequent clipping is
        # still a diagnostic that the sampled input-barrier dynamics need review.
        command = torch.clamp(command_unclipped, min=lower, max=upper)

        correction = surrogate - desired_surrogate
        individual_h = self._current_original_barriers(
            position_w,
            yaw,
            current_command,
            obstacle_centers,
            obstacle_radii,
            obstacle_mask,
        )
        return BackupSafetyFilterResult(
            command=command,
            surrogate_command=surrogate,
            desired_surrogate_command=desired_surrogate,
            h_lse=h_aug,
            h_backup_lse=h_backup,
            h_input_lse=h_input,
            individual_h=individual_h,
            active=needs_correction,
            correction_norm=torch.linalg.norm(correction, dim=1),
            cbf_value_nominal=omega,
            lambda_value=lambda_value,
            mu_value=mu,
            input_bound_violation=input_bound_violation,
            input_bound_active=input_bound_violation > 1.0e-6,
        )
