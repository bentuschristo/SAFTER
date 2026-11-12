"""Regression tests: CF-matched differentiable QP vs. the closed-form filter.

Compares ZeroBackupBcbfQpFilter(variant="cd_lse_qp_cfmatched") -- which
solves the same single affine constraint as ZeroBackupLseCbfClosedForm via
QpthProjector's exact differentiable QP/KKT solve instead of the closed
form's hand-derived analytic projection -- against
ZeroBackupLseCbfClosedForm itself, on:

  * an inactive-constraint case (no correction needed)
  * an active-constraint case (correction needed, single active row)
  * several random obstacle/state batches

For each case this checks:
  1. surrogate_command agrees between CF and the CF-matched QP
  2. final command (post zero-order-hold decay + clip) agrees
  3. the shared CF affine-constraint residual, evaluated at each method's
     own surrogate, agrees between the two
  4. (away from the active/inactive and degenerate switching boundaries)
     the autograd gradient of a fixed random scalar probe of `command`
     w.r.t. `nominal_command` agrees between CF and the CF-matched QP

Targets ~1e-4 forward/gradient agreement; prints the actual observed
maxima. Requires the `qpth` package (`pip install qpth`).

Run: python -m scripts.test_cf_matched_qp_regression
"""
from __future__ import annotations

import torch

from src.high_level_rl.bcbf_lse_cf_filter import ZeroBackupLseCbfClosedForm
from src.high_level_rl.bcbf_qp_filter import ZeroBackupBcbfQpFilter
from src.high_level_rl.config import ArenaConfig, BackupSafetyConfig, HighLevelActionConfig

FORWARD_TOL = 1.0e-4
GRAD_TOL = 1.0e-4
# Rows within this margin of the active/inactive (omega=0) or degenerate
# (denominator_raw=eps) switching boundary are excluded from the gradient
# comparison only: CF and the QP are built to share the exact same
# needs_correction/degenerate_active booleans (computed once, identically,
# before either solve), so there is no risk of them disagreeing on which
# branch is active. But *at* a boundary a one-sided derivative is
# ambiguous by construction -- both sides can be "correct" there -- so
# comparing gradients that close is not a meaningful test of solver
# equivalence.
OMEGA_MARGIN = 0.05
DENOM_MARGIN = 1.0e-4


def _make_filters(device: torch.device, dtype: torch.dtype):
    arena = ArenaConfig()
    action = HighLevelActionConfig()
    cfg = BackupSafetyConfig()
    cf = ZeroBackupLseCbfClosedForm(arena, action, cfg)
    qp = ZeroBackupBcbfQpFilter(arena, action, cfg, variant="cd_lse_qp_cfmatched")
    return arena, action, cfg, cf, qp


def _obstacle_tensors(arena: ArenaConfig, batch: int, device: torch.device, dtype: torch.dtype):
    centers = torch.tensor([[o.x, o.y] for o in arena.obstacles], device=device, dtype=dtype)
    radii = torch.tensor([o.radius for o in arena.obstacles], device=device, dtype=dtype)
    centers = centers.unsqueeze(0).expand(batch, -1, -1).clone()
    radii = radii.unsqueeze(0).expand(batch, -1).clone()
    mask = torch.ones(batch, len(arena.obstacles), dtype=torch.bool, device=device)
    return centers, radii, mask


def _cf_terms(
    cf: ZeroBackupLseCbfClosedForm,
    position_w: torch.Tensor,
    yaw: torch.Tensor,
    current_command: torch.Tensor,
    obstacle_centers: torch.Tensor,
    obstacle_radii: torch.Tensor,
    obstacle_mask: torch.Tensor,
):
    """Recompute h_aug/Lf/Lg/alpha via the SAME primitives both classes
    share, independent of either filter's own forward pass -- used only to
    evaluate the shared affine-constraint residual at an arbitrary
    surrogate for the cross-check below."""
    terms, grads = cf._backup_rollout_terms_and_grads(
        position_w, yaw, current_command, obstacle_centers, obstacle_radii, obstacle_mask
    )
    h_backup, grad_backup = cf._softmin_value_grad(terms, grads, cf._cfg_float("kappa_backup", 20.0))
    h_input, grad_input = cf._input_lse(current_command)
    outer_terms = torch.stack((h_backup, h_input), dim=1)
    outer_grads = torch.stack((grad_backup, grad_input), dim=1)
    h_aug, grad = cf._softmin_value_grad(outer_terms, outer_grads, cf._cfg_float("kappa_outer", 50.0))

    a = torch.tensor(cf.cfg.control_dynamics_a, device=current_command.device, dtype=current_command.dtype)
    theta = yaw
    vx, vy, wz = current_command.unbind(dim=1)
    c, s = torch.cos(theta), torch.sin(theta)
    drift = torch.stack(
        (c * vx - s * vy, s * vx + c * vy, wz, -a[0] * vx, -a[1] * vy, -a[2] * wz), dim=1
    )
    lf = torch.sum(grad * drift, dim=1)
    lg = grad[:, 3:] * a
    alpha_outer = cf._cfg_float("alpha_outer", 0.8)
    return h_aug, lf, lg, alpha_outer


def _residual(lf, lg, alpha_outer, h_aug, surrogate):
    return lf + alpha_outer * h_aug + torch.sum(lg * surrogate, dim=1)


class CaseResult:
    def __init__(self, name: str):
        self.name = name
        self.max_surrogate_err = 0.0
        self.max_command_err = 0.0
        self.max_residual_err = 0.0
        self.max_grad_err = float("nan")
        self.n_grad_checked = 0
        self.n_total = 0


def run_case(
    name: str,
    cf: ZeroBackupLseCbfClosedForm,
    qp: ZeroBackupBcbfQpFilter,
    position: torch.Tensor,
    yaw: torch.Tensor,
    current: torch.Tensor,
    nominal: torch.Tensor,
    prev_nominal: torch.Tensor,
    history_valid: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    mask: torch.Tensor,
    probe: torch.Tensor,
) -> CaseResult:
    result = CaseResult(name)
    result.n_total = position.shape[0]

    nominal_cf = nominal.clone().detach().requires_grad_(True)
    nominal_qp = nominal.clone().detach().requires_grad_(True)

    out_cf = cf.filter(position, yaw, current, nominal_cf, prev_nominal, history_valid, centers, radii, mask)
    out_qp = qp.filter(position, yaw, current, nominal_qp, prev_nominal, history_valid, centers, radii, mask)

    result.max_surrogate_err = (out_cf.surrogate_command - out_qp.surrogate_command).abs().max().item()
    result.max_command_err = (out_cf.command - out_qp.command).abs().max().item()

    h_aug, lf, lg, alpha_outer = _cf_terms(cf, position, yaw, current, centers, radii, mask)
    residual_cf = _residual(lf, lg, alpha_outer, h_aug, out_cf.surrogate_command)
    residual_qp = _residual(lf, lg, alpha_outer, h_aug, out_qp.surrogate_command)
    result.max_residual_err = (residual_cf - residual_qp).abs().max().item()

    # omega == residual evaluated at desired_surrogate == cbf_value_nominal
    # for both filters by construction (shared formula, computed pre-solve).
    omega = out_cf.cbf_value_nominal
    needs_correction = omega < 0.0
    denom = torch.sum(lg.square(), dim=1)  # unweighted here is fine for a margin test
    degenerate_active = needs_correction & (denom <= DENOM_MARGIN)
    away_from_boundary = (omega.abs() > OMEGA_MARGIN) & (~degenerate_active)

    (out_cf.command * probe[None, :]).sum().backward()
    (out_qp.command * probe[None, :]).sum().backward()

    grad_cf = nominal_cf.grad
    grad_qp = nominal_qp.grad
    if away_from_boundary.any():
        idx = away_from_boundary
        result.max_grad_err = (grad_cf[idx] - grad_qp[idx]).abs().max().item()
        result.n_grad_checked = int(idx.sum().item())
    return result


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torch.manual_seed(0)

    arena, action, cfg, cf, qp = _make_filters(device, dtype)
    results: list[CaseResult] = []

    # --- Case 1: inactive constraint (open space, mild motion). ---
    batch = 4
    position = torch.zeros(batch, 2, device=device, dtype=dtype)
    yaw = torch.zeros(batch, device=device, dtype=dtype)
    current = torch.tensor([[0.1, 0.0, 0.0]] * batch, device=device, dtype=dtype)
    nominal = torch.tensor([[0.1, 0.0, 0.0]] * batch, device=device, dtype=dtype)
    prev_nominal = nominal.clone()
    history_valid = torch.ones(batch, 1, device=device, dtype=dtype)
    centers, radii, mask = _obstacle_tensors(arena, batch, device, dtype)
    probe = torch.tensor([1.3, -0.7, 0.5], device=device, dtype=dtype)
    results.append(
        run_case("inactive", cf, qp, position, yaw, current, nominal, prev_nominal, history_valid, centers, radii, mask, probe)
    )

    # --- Case 2: active constraint (close to an obstacle, driving at it). ---
    batch = 4
    # Obstacle at (1.2, -1.0), radius 0.6; approach from the left along +x.
    position = torch.tensor([[0.3, -1.0]] * batch, device=device, dtype=dtype)
    yaw = torch.zeros(batch, device=device, dtype=dtype)
    current = torch.tensor([[action.max_vx, 0.0, 0.0]] * batch, device=device, dtype=dtype)
    nominal = torch.tensor([[action.max_vx, 0.0, 0.0]] * batch, device=device, dtype=dtype)
    prev_nominal = nominal.clone()
    history_valid = torch.ones(batch, 1, device=device, dtype=dtype)
    centers, radii, mask = _obstacle_tensors(arena, batch, device, dtype)
    results.append(
        run_case("active", cf, qp, position, yaw, current, nominal, prev_nominal, history_valid, centers, radii, mask, probe)
    )

    # --- Case 3: several random obstacle/state batches. ---
    for trial in range(5):
        batch = 64
        position = (torch.rand(batch, 2, device=device, dtype=dtype) * 2 - 1) * 6.0
        yaw = (torch.rand(batch, device=device, dtype=dtype) * 2 - 1) * torch.pi
        current = torch.stack(
            (
                torch.rand(batch, device=device, dtype=dtype) * (action.max_vx - action.min_vx) + action.min_vx,
                (torch.rand(batch, device=device, dtype=dtype) * 2 - 1) * action.max_vy,
                (torch.rand(batch, device=device, dtype=dtype) * 2 - 1) * action.max_wz,
            ),
            dim=1,
        )
        nominal = torch.stack(
            (
                torch.rand(batch, device=device, dtype=dtype) * (action.max_vx - action.min_vx) + action.min_vx,
                (torch.rand(batch, device=device, dtype=dtype) * 2 - 1) * action.max_vy,
                (torch.rand(batch, device=device, dtype=dtype) * 2 - 1) * action.max_wz,
            ),
            dim=1,
        )
        prev_nominal = torch.stack(
            (
                torch.rand(batch, device=device, dtype=dtype) * (action.max_vx - action.min_vx) + action.min_vx,
                (torch.rand(batch, device=device, dtype=dtype) * 2 - 1) * action.max_vy,
                (torch.rand(batch, device=device, dtype=dtype) * 2 - 1) * action.max_wz,
            ),
            dim=1,
        )
        history_valid = (torch.rand(batch, 1, device=device, dtype=dtype) > 0.5).to(dtype)
        centers, radii, mask = _obstacle_tensors(arena, batch, device, dtype)
        results.append(
            run_case(
                f"random_{trial}", cf, qp, position, yaw, current, nominal, prev_nominal, history_valid, centers, radii, mask, probe
            )
        )

    print(f"{'case':<12} {'n':>4} {'surrogate':>12} {'command':>12} {'residual':>12} {'grad(n=?)':>18}")
    max_fwd = 0.0
    max_grad = 0.0
    for r in results:
        grad_str = f"{r.max_grad_err:.3e} (n={r.n_grad_checked})" if r.n_grad_checked else "skipped (all near boundary)"
        print(
            f"{r.name:<12} {r.n_total:>4} {r.max_surrogate_err:>12.3e} {r.max_command_err:>12.3e} "
            f"{r.max_residual_err:>12.3e} {grad_str:>18}"
        )
        max_fwd = max(max_fwd, r.max_surrogate_err, r.max_command_err, r.max_residual_err)
        if r.n_grad_checked:
            max_grad = max(max_grad, r.max_grad_err)

    print(f"\nmax forward error (surrogate/command/residual) across all cases: {max_fwd:.3e}")
    print(f"max gradient error (away from switching boundary) across all cases: {max_grad:.3e}")

    assert max_fwd < FORWARD_TOL, f"forward disagreement {max_fwd:.3e} exceeds tolerance {FORWARD_TOL:.3e}"
    assert max_grad < GRAD_TOL, f"gradient disagreement {max_grad:.3e} exceeds tolerance {GRAD_TOL:.3e}"
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
