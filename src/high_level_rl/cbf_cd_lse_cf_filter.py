from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.func import jacrev, vmap

from .config import ArenaConfig, BackupSafetyConfig, HighLevelActionConfig


@dataclass
class CbfCdSafetyFilterResult:
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


class LseCbfControlDynamicsClosedForm:
    """Normal CBF-CF with command/input dynamics.

    Augmented state is z=[x, y, yaw, vx, vy, wz].  The command state u=[vx,vy,wz]
    evolves as

        u_dot = -A u + A uhat.

    Raw obstacle/wall clearances h(x) are relative degree two with respect to
    uhat, so this layer does *not* apply a first-order CBF directly to h.  It
    first lifts each physical barrier to

        psi_i(z) = hdot_i(z) + k0 h_i(x),

    which is relative degree one with respect to uhat.  The input box limits are
    already state barriers on u and are also relative degree one.  A single LSE
    soft-min is then formed over all lifted physical barriers and input barriers,
    and a one-constraint closed-form CBF correction is applied to uhat.

    This is the Python/Torch analogue of the uploaded C control-dynamics CBF-CF
    idea, with gradients computed by torch.func rather than finite differences.
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

    def _lifted_physical_terms_single(
        self,
        state: torch.Tensor,
        centers: torch.Tensor,
        radii: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        p = state[:2]
        yaw = state[2]
        vx, vy = state[3], state[4]
        c, s = torch.cos(yaw), torch.sin(yaw)
        velocity_w = torch.stack((c * vx - s * vy, s * vx + c * vy))

        k0 = self._cfg_float("hocbf_k0", 2.0)

        rel = p[None, :] - centers
        safe_radii = radii + float(self.arena.obstacle_clearance)
        h_obs = torch.sum(rel * rel, dim=-1) - safe_radii.square()
        hdot_obs = 2.0 * torch.sum(rel * velocity_w[None, :], dim=-1)
        psi_obs = hdot_obs + k0 * h_obs
        psi_obs = torch.where(mask, psi_obs, torch.full_like(psi_obs, 1.0e3))

        x, y = p.unbind(dim=0)
        h_walls = torch.stack(
            (
                x - float(self.arena.x_min) - float(self.arena.wall_clearance),
                float(self.arena.x_max) - x - float(self.arena.wall_clearance),
                y - float(self.arena.y_min) - float(self.arena.wall_clearance),
                float(self.arena.y_max) - y - float(self.arena.wall_clearance),
            )
        )
        wall_normals = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            device=state.device,
            dtype=state.dtype,
        )
        hdot_walls = wall_normals @ velocity_w
        psi_walls = hdot_walls + k0 * h_walls
        return torch.cat((psi_obs, psi_walls), dim=0)

    def _input_terms_single(self, state: torch.Tensor) -> torch.Tensor:
        lower, upper = self._action_bounds(state)
        u = state[3:6]
        return torch.cat((upper - u, u - lower), dim=0)

    def _all_terms_single(
        self,
        state: torch.Tensor,
        centers: torch.Tensor,
        radii: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                self._lifted_physical_terms_single(state, centers, radii, mask),
                self._input_terms_single(state),
            ),
            dim=0,
        )

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
    ) -> CbfCdSafetyFilterResult:
        eps = float(self.cfg.denominator_epsilon)
        z = torch.cat((position_w, yaw[:, None], current_command), dim=1)

        physical_terms = vmap(self._lifted_physical_terms_single)(
            z, obstacle_centers, obstacle_radii, obstacle_mask
        )
        physical_grads = vmap(jacrev(self._lifted_physical_terms_single, argnums=0))(
            z, obstacle_centers, obstacle_radii, obstacle_mask
        )
        h_physical, grad_physical = self._softmin_value_grad(
            physical_terms,
            physical_grads,
            self._cfg_float("kappa_backup", 50.0),
        )

        input_terms = vmap(self._input_terms_single)(z)
        input_grads = vmap(jacrev(self._input_terms_single))(z)
        h_input, grad_input = self._softmin_value_grad(
            input_terms,
            input_grads,
            self._cfg_float("kappa_input", 20.0),
        )

        all_terms = torch.cat((physical_terms, input_terms), dim=1)
        all_grads = torch.cat((physical_grads, input_grads), dim=1)
        h_aug, grad = self._softmin_value_grad(
            all_terms,
            all_grads,
            self._cfg_float("kappa_outer", 20.0),
        )

        a = torch.as_tensor(
            self.cfg.control_dynamics_a,
            device=current_command.device,
            dtype=current_command.dtype,
        )
        sigma = torch.as_tensor(
            self.cfg.tracking_sigma,
            device=current_command.device,
            dtype=current_command.dtype,
        )

        vx, vy, wz = current_command.unbind(dim=1)
        c, s = torch.cos(yaw), torch.sin(yaw)
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
        valid = nominal_history_valid.to(device=nominal_command.device, dtype=nominal_command.dtype)
        if valid.ndim == 1:
            valid = valid[:, None]
        nominal_derivative = valid * (nominal_command - previous_nominal_command) / dt
        desired_surrogate = (
            current_command
            + nominal_derivative / a
            + (sigma / a) * (nominal_command - current_command)
        )

        omega = (
            lf
            + torch.sum(lg * desired_surrogate, dim=1)
            + self._cfg_float("alpha_outer", 0.8) * h_aug
        )

        cost_weights = torch.as_tensor(
            getattr(self.cfg, "surrogate_cost_weights", (1.0, 1.0, 1.0)),
            device=lg.device,
            dtype=lg.dtype,
        )
        cost_weights = torch.clamp(cost_weights, min=eps)
        gamma_mu = max(self._cfg_float("gamma_mu", 1.0e4), eps)
        denominator_raw = torch.sum(lg.square() / cost_weights, dim=1) + h_aug.square() / gamma_mu

        needs_correction = omega < 0.0
        nondegenerate = denominator_raw > eps
        lambda_value = torch.where(
            needs_correction & nondegenerate,
            -omega / torch.clamp(denominator_raw, min=eps),
            torch.zeros_like(omega),
        )
        surrogate = desired_surrogate + lambda_value[:, None] * lg / cost_weights
        mu = h_aug * lambda_value / gamma_mu

        decay = torch.exp(-a * dt)
        command = decay * current_command + (1.0 - decay) * surrogate

        lower, upper = self._action_bounds(command)
        input_bound_violation = torch.maximum(
            torch.relu(command - upper),
            torch.relu(lower - command),
        ).max(dim=1).values
        input_bound_active = input_terms.min(dim=1).values < 0.05

        correction = surrogate - desired_surrogate
        cbf_value_nominal = omega

        return CbfCdSafetyFilterResult(
            command=command,
            surrogate_command=surrogate,
            desired_surrogate_command=desired_surrogate,
            h_lse=h_aug,
            h_backup_lse=h_physical,
            h_input_lse=h_input,
            individual_h=all_terms,
            active=lambda_value > 0.0,
            correction_norm=torch.linalg.norm(correction, dim=1),
            cbf_value_nominal=cbf_value_nominal,
            lambda_value=lambda_value,
            mu_value=mu,
            input_bound_violation=input_bound_violation,
            input_bound_active=input_bound_active,
        )
