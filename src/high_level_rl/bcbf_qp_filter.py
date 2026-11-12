from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import torch
from torch.func import jacrev, vmap

from .bcbf_lse_cf_filter import (
    _DEFAULT_ALPHA_OUTER,
    _DEFAULT_KAPPA_BACKUP,
    _DEFAULT_KAPPA_OUTER,
    BackupSafetyFilterResult,
    ZeroBackupLseCbfClosedForm,
)
from .config import ArenaConfig, BackupSafetyConfig, HighLevelActionConfig

QpVariant = Literal[
    "cbf_qp_lim",
    "cbf_lse_qp_lim",
    "noncd_nonlse_qp",
    "noncd_lse_qp",
    "cd_nonlse_qp",
    "cd_lse_qp",
    "cd_lse_qp_cfmatched",
]


@dataclass(frozen=True)
class QpSolveDiagnostics:
    active_rows: torch.Tensor
    max_violation: torch.Tensor
    objective: torch.Tensor


class SmallBoxQpProjector:
    """GPU-batched active-set QP projector for 3-D command filters.

    It solves/approximates
        min_x 0.5 * sum_j w_j (x_j - x0_j)^2
        s.t. A x <= b

    For a three-variable strictly convex QP an optimum has at most three
    linearly independent active rows.  For small LSE QPs all such active sets
    are enumerated exactly.  For large unaggregated laser QPs, active sets are
    enumerated over the most violated rows and the candidate solutions are
    checked against *all* constraints.

    Unlike the archived implementation, this class contains no loop over the
    environment/replay batch and no scalar GPU-to-CPU synchronization.  Active
    sets are evaluated in tensor chunks across the complete batch.  The default
    straight-through backward pass is deliberate: use a dedicated
    differentiable QP package if exact KKT sensitivities are required.
    """

    def __init__(
        self,
        max_candidates: int = 18,
        feasibility_tol: float = 1.0e-5,
        regularization: float = 1.0e-5,
        straight_through: bool = True,
        subset_chunk_size: int = 64,
    ) -> None:
        self.max_candidates = int(max_candidates)
        self.feasibility_tol = float(feasibility_tol)
        self.regularization = float(regularization)
        self.straight_through = bool(straight_through)
        self.subset_chunk_size = int(subset_chunk_size)
        if self.subset_chunk_size < 1:
            raise ValueError("subset_chunk_size must be positive")

    @staticmethod
    def _candidate_subsets(n: int) -> list[tuple[int, ...]]:
        idx = range(n)
        out: list[tuple[int, ...]] = [tuple()]
        out.extend((i,) for i in idx)
        out.extend(combinations(idx, 2))
        out.extend(combinations(idx, 3))
        return out

    def project(
        self,
        desired: torch.Tensor,
        A: torch.Tensor,
        b: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, QpSolveDiagnostics]:
        if desired.ndim != 2 or desired.shape[-1] != 3:
            raise ValueError("desired must have shape [B, 3]")
        if A.ndim != 3 or A.shape[0] != desired.shape[0] or A.shape[2] != 3:
            raise ValueError("A must have shape [B, M, 3]")
        if b.ndim != 2 or b.shape[:2] != A.shape[:2]:
            raise ValueError("b must have shape [B, M]")

        # Defensive cleanup: QP rows may contain numerically degenerate terms
        # when obstacles are masked or when repeated horizon rows become identical.
        # Do not let one NaN/Inf kill the whole vectorized training run.
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        b = torch.nan_to_num(b, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)

        B, M, _ = A.shape
        device = desired.device
        dtype = desired.dtype
        if weights is None:
            weights = torch.ones(3, device=device, dtype=dtype)
        weights = weights.to(device=device, dtype=dtype).clamp_min(self.regularization)
        winv = 1.0 / weights

        # First check if the desired command is already feasible.
        violation = A @ desired[:, :, None]
        violation = violation.squeeze(-1) - b
        feasible_desired = violation.max(dim=1).values <= self.feasibility_tol

        # Projection selection does not need a graph for the straight-through
        # estimator. Detaching here prevents a very large candidate graph from
        # being retained during SAC actor updates.
        solve_desired = desired.detach() if self.straight_through else desired
        solve_A = A.detach() if self.straight_through else A
        solve_b = b.detach() if self.straight_through else b

        k = min(self.max_candidates, M)
        candidate_rows = torch.topk(violation.detach(), k=k, dim=1, largest=True).indices
        gather_A = candidate_rows[:, :, None].expand(-1, -1, 3)
        candidate_A = torch.gather(solve_A, 1, gather_A)
        candidate_b = torch.gather(solve_b, 1, candidate_rows)

        best_x = solve_desired.clone()
        best_obj = torch.zeros(B, device=device, dtype=dtype)
        best_maxv = violation.detach().max(dim=1).values
        best_active = torch.zeros(B, device=device, dtype=dtype)
        large = torch.as_tensor(1.0e12, device=device, dtype=dtype)
        best_score = torch.where(feasible_desired, best_obj, large + torch.relu(best_maxv))

        for active_size in (1, 2, 3):
            if k < active_size:
                continue
            subsets = torch.as_tensor(
                list(combinations(range(k), active_size)), device=device, dtype=torch.long
            )
            eye = torch.eye(active_size, device=device, dtype=dtype)
            for start in range(0, subsets.shape[0], self.subset_chunk_size):
                subset = subsets[start : start + self.subset_chunk_size]
                C = candidate_A[:, subset, :]  # [B,K,r,3]
                d = candidate_b[:, subset]     # [B,K,r]
                CW = C * winv[None, None, None, :]
                KKT = CW @ C.transpose(-1, -2)
                KKT = 0.5 * (KKT + KKT.transpose(-1, -2))
                diag_scale = torch.diagonal(KKT, dim1=-2, dim2=-1).abs().amax(-1).clamp_min(1.0)
                KKT = KKT + self.regularization * diag_scale[..., None, None] * eye
                rhs = (C @ solve_desired[:, None, :, None]).squeeze(-1) - d
                lam = torch.linalg.solve(KKT, rhs[..., None]).squeeze(-1)
                x = solve_desired[:, None, :] - torch.einsum(
                    "bkrj,bkr->bkj", CW, lam
                )

                # Every candidate is checked against the complete constraint set.
                all_v = torch.einsum("bmj,bkj->bkm", solve_A, x) - solve_b[:, None, :]
                maxv = all_v.amax(dim=2)
                obj = 0.5 * torch.sum(
                    weights[None, None, :] * (x - solve_desired[:, None, :]).square(), dim=2
                )
                feasible = maxv <= self.feasibility_tol
                score = torch.where(feasible, obj, large + torch.relu(maxv))
                chunk_score, chunk_index = score.min(dim=1)
                batch_index = torch.arange(B, device=device)
                improve = chunk_score < best_score
                chosen_x = x[batch_index, chunk_index]
                chosen_obj = obj[batch_index, chunk_index]
                chosen_maxv = maxv[batch_index, chunk_index]
                best_x = torch.where(improve[:, None], chosen_x, best_x)
                best_obj = torch.where(improve, chosen_obj, best_obj)
                best_maxv = torch.where(improve, chosen_maxv, best_maxv)
                best_active = torch.where(
                    improve, torch.full_like(best_active, float(active_size)), best_active
                )
                best_score = torch.where(improve, chunk_score, best_score)

        # Vectorized feasibility fallback for large candidate-screened problems.
        x = best_x
        for _ in range(64):
            all_v = torch.einsum("bmj,bj->bm", solve_A, x) - solve_b
            row = all_v.argmax(dim=1)
            batch_index = torch.arange(B, device=device)
            row_v = all_v[batch_index, row]
            arow = solve_A[batch_index, row]
            denom = torch.sum(arow.square() * winv[None, :], dim=1).clamp_min(self.regularization)
            correction = (torch.relu(row_v) / denom)[:, None] * winv[None, :] * arow
            x = x - correction
        fallback_v = torch.einsum("bmj,bj->bm", solve_A, x) - solve_b
        fallback_maxv = fallback_v.amax(dim=1)
        fallback_obj = 0.5 * torch.sum(weights[None, :] * (x - solve_desired).square(), dim=1)
        use_fallback = (best_maxv > self.feasibility_tol) & (fallback_maxv < best_maxv)
        best_x = torch.where(use_fallback[:, None], x, best_x)
        best_obj = torch.where(use_fallback, fallback_obj, best_obj)
        best_maxv = torch.where(use_fallback, fallback_maxv, best_maxv)
        best_active = torch.where(use_fallback, torch.full_like(best_active, -1.0), best_active)

        qp_solution = best_x
        if self.straight_through:
            qp_solution = desired + (qp_solution - desired).detach()

        diag = QpSolveDiagnostics(
            active_rows=best_active,
            max_violation=best_maxv,
            objective=best_obj,
        )
        return qp_solution, diag


class OsqpProjector:
    """Exact CPU OSQP backend for deployment/evaluation of the 3-D QP.

    OSQP is deliberately imported lazily: GPU training does not depend on it.
    The returned tensor uses a straight-through gradient only when explicitly
    requested, which is useful for small diagnostic training runs but not
    efficient for vectorized SAC.
    """

    def __init__(
        self,
        feasibility_tol: float = 1.0e-5,
        regularization: float = 1.0e-6,
        straight_through: bool = False,
    ) -> None:
        self.feasibility_tol = float(feasibility_tol)
        self.regularization = float(regularization)
        self.straight_through = bool(straight_through)

    def project(self, desired, A, b, weights=None):
        try:
            import osqp
            import numpy as np
            from scipy import sparse
        except ImportError as exc:
            raise RuntimeError(
                "The bCBF-QP OSQP backend requires the 'osqp' and 'scipy' packages. "
                "Install osqp in the deployment environment or select the active_set backend."
            ) from exc
        if desired.ndim != 2 or desired.shape[-1] != 3:
            raise ValueError("desired must have shape [B,3]")
        if weights is None:
            weights = torch.ones(3, device=desired.device, dtype=desired.dtype)
        w = weights.detach().cpu().double().numpy()
        P = sparse.diags(np.maximum(w, self.regularization), format="csc")
        xs, active, maxvs, objectives = [], [], [], []
        for i in range(desired.shape[0]):
            x0 = desired[i].detach().cpu().double().numpy()
            Ai = A[i].detach().cpu().double().numpy()
            bi = b[i].detach().cpu().double().numpy()
            Ai = np.nan_to_num(Ai, nan=0.0, posinf=0.0, neginf=0.0)
            bi = np.nan_to_num(bi, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)
            solver = osqp.OSQP()
            solver.setup(
                P=P, q=-w * x0, A=sparse.csc_matrix(Ai),
                l=np.full(bi.shape, -np.inf), u=bi,
                verbose=False, polishing=True,
                eps_abs=self.feasibility_tol, eps_rel=self.feasibility_tol,
                max_iter=10_000,
            )
            result = solver.solve()
            solved = result.info.status_val in (1, 2) and result.x is not None
            x = result.x if solved else np.zeros(3, dtype=np.float64)
            violation = Ai @ x - bi
            maxv = float(np.max(violation))
            xs.append(torch.as_tensor(x, device=desired.device, dtype=desired.dtype))
            active.append(int(np.count_nonzero(np.abs(violation) <= 5.0 * self.feasibility_tol)))
            maxvs.append(maxv)
            objectives.append(0.5 * float(np.sum(w * (x - x0) ** 2)))
        solution = torch.stack(xs)
        if self.straight_through:
            solution = desired + (solution - desired).detach()
        diag = QpSolveDiagnostics(
            active_rows=torch.as_tensor(active, device=desired.device, dtype=desired.dtype),
            max_violation=torch.as_tensor(maxvs, device=desired.device, dtype=desired.dtype),
            objective=torch.as_tensor(objectives, device=desired.device, dtype=desired.dtype),
        )
        return solution, diag


class QpthProjector:
    """Exact batched differentiable QP backend used by Mod-RL-RCBF.

    ``qpth`` differentiates the primal-dual KKT solution and therefore does not
    use the straight-through estimator.  It is optional so evaluation machines
    can continue to use OSQP and installations without qpth can use the
    vectorized three-variable active-set backend.
    """

    def __init__(
        self,
        feasibility_tol: float = 1.0e-5,
        regularization: float = 1.0e-5,
    ) -> None:
        self.feasibility_tol = float(feasibility_tol)
        self.regularization = float(regularization)

    def project(self, desired, A, b, weights=None):
        try:
            from qpth.qp import QPFunction
        except ImportError as exc:
            raise RuntimeError(
                "The differentiable batched QP backend requires qpth. Install "
                "it with `python -m pip install qpth`, or select active_set/osqp."
            ) from exc
        if weights is None:
            weights = torch.ones(3, device=desired.device, dtype=desired.dtype)
        weights = weights.to(device=desired.device, dtype=desired.dtype).clamp_min(
            self.regularization
        )
        batch = desired.shape[0]
        Q = torch.diag(weights).unsqueeze(0).expand(batch, -1, -1).contiguous()
        p = -weights[None, :] * desired

        # Constraint normalization substantially improves primal-dual scaling
        # while leaving each halfspace unchanged.
        scale = torch.cat((A, b[..., None]), dim=2).abs().amax(dim=2).clamp_min(1.0e-8)
        G = A / scale[..., None]
        h = b / scale
        empty_A = desired.new_empty((batch, 0, 3))
        empty_b = desired.new_empty((batch, 0))
        solution = QPFunction(
            verbose=0,
            check_Q_spd=False,
            maxIter=100,
            notImprovedLim=10,
            eps=self.feasibility_tol,
        )(Q, p, G, h, empty_A, empty_b)
        residuals = torch.einsum("bmj,bj->bm", A, solution) - b
        maxv = residuals.amax(dim=1)
        objective = 0.5 * torch.sum(weights[None, :] * (solution - desired).square(), dim=1)
        active = (residuals.abs() <= 5.0 * self.feasibility_tol).sum(dim=1)
        return solution, QpSolveDiagnostics(
            active_rows=active.to(desired.dtype),
            max_violation=maxv,
            objective=objective,
        )


class ZeroBackupBcbfQpFilter(ZeroBackupLseCbfClosedForm):
    """Python QP baselines using the same bCBF-CF parameters.

    Variants:
      * cbf_qp_lim:      direct-command normal CBF-QP + hard input box rows; no backup, no CD, no LSE
      * cbf_lse_qp_lim:  direct-command normal CBF-QP + one LSE CBF row + hard input box rows; no backup, no CD
      * noncd_nonlse_qp: direct-command bCBF-QP, individual flow and terminal rows; no LSE
      * noncd_lse_qp:    direct-command bCBF-QP, one LSE backup row
      * cd_nonlse_qp:    surrogate-command bCBF-QP, individual flow and terminal rows + input CBF rows; no LSE
      * cd_lse_qp:       surrogate-command bCBF-QP, one LSE backup row + input CBF rows
      * cd_lse_qp_cfmatched: CF-matched differentiable-QP baseline. Solves
        EXACTLY the same single affine constraint that
        ZeroBackupLseCbfClosedForm.filter() solves in closed form -- the
        backup LSE row (kappa_backup) and the input LSE row (kappa_input)
        are aggregated through the same outer LSE (kappa_outer) into one
        scalar h_aug/Lf/Lg, with no separate box/input-CBF rows appended
        (unlike every other QP variant above). It always uses QpthProjector
        (exact batched differentiable QP/KKT solve, no straight-through
        estimator, no .detach() on the actor path) regardless of the
        straight_through/max_active_candidates constructor arguments, which
        only affect the other variants. Intended as an apples-to-apples
        ablation against the closed form: same terms, same gradients, same
        ZOH propagation and degenerate fallback -- only the projection
        solver differs (QP vs. hand-derived analytic KKT formula). See
        scripts/test_cf_matched_qp_regression.py.

    All variants except cd_lse_qp_cfmatched are intended for SAC ablation
    training via the batched active-set QP (SmallBoxQpProjector), which uses
    a straight-through estimator by default because that solver is not a
    smooth map everywhere for the multi-row QPs those variants build.
    """

    def __init__(
        self,
        arena: ArenaConfig,
        action_cfg: HighLevelActionConfig,
        cfg: BackupSafetyConfig,
        variant: QpVariant = "cd_lse_qp",
        max_active_candidates: int = 18,
        straight_through: bool = True,
    ) -> None:
        super().__init__(arena, action_cfg, cfg)
        self.variant = variant
        self.projector = SmallBoxQpProjector(
            max_candidates=max_active_candidates,
            straight_through=straight_through,
        )
        # cd_lse_qp_cfmatched always solves its single affine constraint with
        # the exact differentiable QP backend, independent of the
        # straight_through argument above (which only governs the other,
        # multi-row variants' SmallBoxQpProjector). Constructing this is
        # cheap and side-effect-free (qpth is imported lazily inside
        # QpthProjector.project, not here), so it is always available.
        self.cfmatched_projector = QpthProjector(
            feasibility_tol=1.0e-6,
            regularization=1.0e-6,
        )

    def _uses_lse(self) -> bool:
        # Important: do not use substring checks such as "lse" in variant,
        # because "nonlse" also contains the letters "lse".
        return self.variant in ("cbf_lse_qp_lim", "noncd_lse_qp", "cd_lse_qp")

    def _limits(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            [self.action_cfg.max_vx, self.action_cfg.max_vy, self.action_cfg.max_wz],
            device=device,
            dtype=dtype,
        )

    def _direct_primitive_terms_and_grads(
        self,
        position_w: torch.Tensor,
        yaw: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
        obstacle_mask: torch.Tensor,
        terminal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clearance terms and gradients wrt [x,y,yaw] for the non-CD model."""
        p = position_w
        rel = p[:, None, :] - obstacle_centers
        if terminal:
            safe_radii = (
                obstacle_radii
                + float(self.arena.obstacle_clearance)
                + float(self.cfg.backup_margin)
            )
            wall_margin = float(self.arena.wall_clearance) + float(self.cfg.backup_margin)
        else:
            safe_radii = obstacle_radii + float(self.arena.obstacle_clearance)
            wall_margin = float(self.arena.wall_clearance)

        h_obs = torch.sum(rel * rel, dim=-1) - safe_radii.square()
        h_obs = torch.where(obstacle_mask, h_obs, torch.full_like(h_obs, 1.0e3))
        grad_obs = torch.zeros(
            p.shape[0],
            h_obs.shape[1],
            3,
            device=p.device,
            dtype=p.dtype,
        )
        grad_obs[:, :, :2] = 2.0 * rel
        grad_obs = torch.where(obstacle_mask[:, :, None], grad_obs, torch.zeros_like(grad_obs))

        x, y = p.unbind(dim=1)
        h_walls = torch.stack(
            (
                x - float(self.arena.x_min) - wall_margin,
                float(self.arena.x_max) - x - wall_margin,
                y - float(self.arena.y_min) - wall_margin,
                float(self.arena.y_max) - y - wall_margin,
            ),
            dim=1,
        )
        grad_walls = torch.zeros(
            p.shape[0], 4, 3, device=p.device, dtype=p.dtype
        )
        grad_walls[:, 0, 0] = 1.0
        grad_walls[:, 1, 0] = -1.0
        grad_walls[:, 2, 1] = 1.0
        grad_walls[:, 3, 1] = -1.0
        return torch.cat((h_obs, h_walls), dim=1), torch.cat((grad_obs, grad_walls), dim=1)

    def _cbf_qp_lim_rows(
        self,
        position_w: torch.Tensor,
        yaw: torch.Tensor,
        nominal_command: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
        obstacle_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normal direct-command CBF-QP with hard input limits.

        This is the Python/RL analogue of cbf_circle_obstacles_qp_lim.c:
            min_u 0.5 ||u-u_nom||_W^2
            s.t.  -L_g h_i(x) u <= alpha h_i(x)  for obstacles/walls
                  u_min <= u <= u_max

        It intentionally has no backup rollout, no terminal backup set, no LSE,
        and no control-dynamics surrogate state.  The RL version uses the
        environment's full map: circular obstacles plus four arena-wall CBFs.
        """
        h, grads_xyz = self._direct_primitive_terms_and_grads(
            position_w, yaw, obstacle_centers, obstacle_radii, obstacle_mask, terminal=False
        )
        c, s = torch.cos(yaw), torch.sin(yaw)
        G = torch.zeros(position_w.shape[0], 3, 3, device=position_w.device, dtype=position_w.dtype)
        G[:, 0, 0] = c
        G[:, 0, 1] = -s
        G[:, 1, 0] = s
        G[:, 1, 1] = c
        G[:, 2, 2] = 1.0

        # Lg_i = grad h_i(x,y,yaw) * G_body_to_world.
        # CBF condition Lg_i u + alpha h_i >= 0 is written as:
        #     -Lg_i u <= alpha h_i
        Lg = torch.einsum("bmi,bij->bmj", grads_xyz, G)
        A_cbf = -Lg
        alpha = self._cfg_float("alpha_outer", 0.8)
        b_cbf = alpha * h

        limits = self._limits(position_w.device, position_w.dtype)
        A_box = torch.zeros(position_w.shape[0], 6, 3, device=position_w.device, dtype=position_w.dtype)
        b_box = torch.zeros(position_w.shape[0], 6, device=position_w.device, dtype=position_w.dtype)
        for j in range(3):
            # u_j <= limit_j
            A_box[:, 2 * j, j] = 1.0
            b_box[:, 2 * j] = limits[j]
            # -u_j <= limit_j  <=>  u_j >= -limit_j
            A_box[:, 2 * j + 1, j] = -1.0
            b_box[:, 2 * j + 1] = limits[j]

        return nominal_command, torch.cat((A_cbf, A_box), dim=1), torch.cat((b_cbf, b_box), dim=1), h.min(dim=1).values

    def _cbf_lse_qp_lim_rows(
        self,
        position_w: torch.Tensor,
        yaw: torch.Tensor,
        nominal_command: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
        obstacle_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normal direct-command CBF-QP with one LSE obstacle/wall row.

        This is the LSE counterpart of ``cbf_qp_lim``.  It combines only the
        physical safety terms, i.e. circular obstacles and arena walls, into one
        soft-min/LSE CBF row.  The input limits are *not* included in the LSE;
        they remain hard box rows in the QP:

            min_u 0.5 ||u-u_nom||_W^2
            s.t.  -L_g h_LSE(x) u <= alpha h_LSE(x)
                  u_min <= u <= u_max

        No backup rollout, no terminal backup set, and no control dynamics.
        """
        h_terms, grads_xyz = self._direct_primitive_terms_and_grads(
            position_w, yaw, obstacle_centers, obstacle_radii, obstacle_mask, terminal=False
        )
        h_lse, grad_xyz = self._softmin_value_grad(
            h_terms, grads_xyz, self._cfg_float("kappa_outer", 20.0)
        )

        c, s = torch.cos(yaw), torch.sin(yaw)
        G = torch.zeros(position_w.shape[0], 3, 3, device=position_w.device, dtype=position_w.dtype)
        G[:, 0, 0] = c
        G[:, 0, 1] = -s
        G[:, 1, 0] = s
        G[:, 1, 1] = c
        G[:, 2, 2] = 1.0

        Lg = torch.einsum("bi,bij->bj", grad_xyz, G)
        A_cbf = -Lg[:, None, :]
        b_cbf = (self._cfg_float("alpha_outer", 0.8) * h_lse)[:, None]

        limits = self._limits(position_w.device, position_w.dtype)
        A_box = torch.zeros(position_w.shape[0], 6, 3, device=position_w.device, dtype=position_w.dtype)
        b_box = torch.zeros(position_w.shape[0], 6, device=position_w.device, dtype=position_w.dtype)
        for j in range(3):
            A_box[:, 2 * j, j] = 1.0
            b_box[:, 2 * j] = limits[j]
            A_box[:, 2 * j + 1, j] = -1.0
            b_box[:, 2 * j + 1] = limits[j]

        return nominal_command, torch.cat((A_cbf, A_box), dim=1), torch.cat((b_cbf, b_box), dim=1), h_lse

    def _direct_backup_terms_and_grads(
        self,
        position_w: torch.Tensor,
        yaw: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
        obstacle_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Zero backup for the non-CD kinematic model leaves p,yaw fixed, so
        # positive-time samples duplicate the current safe-set terms.  Keeping
        # the samples preserves the uploaded C-script structure, while the
        # terminal terms use the enlarged backup set. For non-LSE variants, these terminal terms remain individual QP rows; no soft-min/LSE aggregation is used.
        # Since every positive-time sample is identical (p,yaw do not evolve),
        # it is computed once and repeated instead of recomputed `samples`
        # times.
        samples = max(int(self.cfg.backup_samples), 2)
        h, g = self._direct_primitive_terms_and_grads(
            position_w, yaw, obstacle_centers, obstacle_radii, obstacle_mask, terminal=False
        )
        hT, gT = self._direct_primitive_terms_and_grads(
            position_w, yaw, obstacle_centers, obstacle_radii, obstacle_mask, terminal=True
        )
        terms = torch.cat((h.repeat(1, samples), hT), dim=1)
        grads = torch.cat((g.repeat(1, samples, 1), gT), dim=1)
        return terms, grads

    def _direct_rows(
        self,
        position_w: torch.Tensor,
        yaw: torch.Tensor,
        nominal_command: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
        obstacle_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        terms, grads_xyz = self._direct_backup_terms_and_grads(
            position_w, yaw, obstacle_centers, obstacle_radii, obstacle_mask
        )
        c, s = torch.cos(yaw), torch.sin(yaw)
        G = torch.zeros(position_w.shape[0], 3, 3, device=position_w.device, dtype=position_w.dtype)
        G[:, 0, 0] = c
        G[:, 0, 1] = -s
        G[:, 1, 0] = s
        G[:, 1, 1] = c
        G[:, 2, 2] = 1.0

        if self._uses_lse():
            h, grad_xyz = self._softmin_value_grad(
                terms, grads_xyz, self._cfg_float("kappa_backup", 50.0)
            )
            Lg = torch.einsum("bi,bij->bj", grad_xyz, G)
            A_backup = -Lg[:, None, :]
            b_backup = (self._cfg_float("alpha_outer", 0.8) * h)[:, None]
        else:
            Lg = torch.einsum("bmi,bij->bmj", grads_xyz, G)
            A_backup = -Lg
            b_backup = self._cfg_float("alpha_outer", 0.8) * terms

        limits = self._limits(position_w.device, position_w.dtype)
        A_box = torch.zeros(position_w.shape[0], 6, 3, device=position_w.device, dtype=position_w.dtype)
        b_box = torch.zeros(position_w.shape[0], 6, device=position_w.device, dtype=position_w.dtype)
        for j in range(3):
            A_box[:, 2 * j, j] = 1.0
            b_box[:, 2 * j] = limits[j]
            A_box[:, 2 * j + 1, j] = -1.0
            b_box[:, 2 * j + 1] = limits[j]

        return nominal_command, torch.cat((A_backup, A_box), dim=1), torch.cat((b_backup, b_box), dim=1)

    def _cd_rows(
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        terms, grads = self._backup_rollout_terms_and_grads(
            position_w, yaw, current_command, obstacle_centers, obstacle_radii, obstacle_mask
        )
        a = torch.tensor(self.cfg.control_dynamics_a, device=current_command.device, dtype=current_command.dtype)
        sigma = torch.tensor(self.cfg.tracking_sigma, device=current_command.device, dtype=current_command.dtype)
        theta = yaw
        vx, vy, wz = current_command.unbind(dim=1)
        c, s = torch.cos(theta), torch.sin(theta)
        drift = torch.stack(
            (c * vx - s * vy, s * vx + c * vy, wz, -a[0] * vx, -a[1] * vy, -a[2] * wz),
            dim=1,
        )
        dt = float(self.cfg.integration_dt)
        valid = nominal_history_valid.to(device=nominal_command.device, dtype=nominal_command.dtype)
        if valid.ndim == 1:
            valid = valid[:, None]
        nominal_derivative = valid * (nominal_command - previous_nominal_command) / dt
        desired_surrogate = current_command + nominal_derivative / a + (sigma / a) * (nominal_command - current_command)

        if self._uses_lse():
            h_backup, grad_backup = self._softmin_value_grad(
                terms, grads, self._cfg_float("kappa_backup", 50.0)
            )
            lf = torch.sum(grad_backup * drift, dim=1)
            lg = grad_backup[:, 3:] * a
            A_backup = -lg[:, None, :]
            b_backup = (lf + self._cfg_float("alpha_outer", 0.8) * h_backup)[:, None]
            h_log = h_backup
        else:
            lf = torch.einsum("bmi,bi->bm", grads, drift)
            lg = grads[:, :, 3:] * a[None, None, :]
            A_backup = -lg
            b_backup = lf + self._cfg_float("alpha_outer", 0.8) * terms
            h_log = terms.min(dim=1).values  # hard minimum only for logging; non-LSE QP rows are individual

        # Command-state input-limit CBF rows, same logic as the uploaded CD C script.
        limits = self._limits(current_command.device, current_command.dtype)
        alpha_u = float(getattr(self.cfg, "alpha_input", 2.0))
        A_in = torch.zeros(current_command.shape[0], 6, 3, device=current_command.device, dtype=current_command.dtype)
        b_in = torch.zeros(current_command.shape[0], 6, device=current_command.device, dtype=current_command.dtype)
        for j in range(3):
            A_in[:, 2 * j, j] = -a[j]
            b_in[:, 2 * j] = -a[j] * current_command[:, j] + alpha_u * (current_command[:, j] + limits[j])
            A_in[:, 2 * j + 1, j] = a[j]
            b_in[:, 2 * j + 1] = a[j] * current_command[:, j] + alpha_u * (limits[j] - current_command[:, j])

        return desired_surrogate, torch.cat((A_backup, A_in), dim=1), torch.cat((b_backup, b_in), dim=1), h_log, terms, grads

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
        # cd_lse_qp_cfmatched is intercepted before the "cd_" dispatch below
        # (it would otherwise match is_cd and route into _cd_rows, which
        # appends the separate input-CBF rows this variant must NOT have).
        if self.variant == "cd_lse_qp_cfmatched":
            return self._filter_cf_matched(
                position_w,
                yaw,
                current_command,
                nominal_command,
                previous_nominal_command,
                nominal_history_valid,
                obstacle_centers,
                obstacle_radii,
                obstacle_mask,
            )
        cost_weights = torch.as_tensor(
            getattr(self.cfg, "surrogate_cost_weights", (1.0, 1.0, 1.0)),
            device=nominal_command.device,
            dtype=nominal_command.dtype,
        )
        is_cbf_qp_lim = self.variant == "cbf_qp_lim"
        is_cbf_lse_qp_lim = self.variant == "cbf_lse_qp_lim"
        is_cd = self.variant.startswith("cd_")
        if is_cd:
            desired, A, b, h_backup_log, terms, grads = self._cd_rows(
                position_w,
                yaw,
                current_command,
                nominal_command,
                previous_nominal_command,
                nominal_history_valid,
                obstacle_centers,
                obstacle_radii,
                obstacle_mask,
            )
            surrogate, diag = self.projector.project(desired, A, b, cost_weights)
            a = torch.tensor(self.cfg.control_dynamics_a, device=current_command.device, dtype=current_command.dtype)
            decay = torch.exp(-a * float(self.cfg.integration_dt))
            command_unclipped = decay * current_command + (1.0 - decay) * surrogate
            limits = self._limits(command_unclipped.device, command_unclipped.dtype)
            command = torch.clamp(command_unclipped, min=-limits, max=limits)
            desired_surrogate = desired
            correction = surrogate - desired
            h_input, _ = self._input_lse(current_command)
            h_lse = h_backup_log
            h_backup_lse = h_backup_log
            input_bound_violation = torch.relu(torch.abs(command_unclipped) - limits).max(dim=1).values
        else:
            if is_cbf_qp_lim:
                desired, A, b, h_backup_log = self._cbf_qp_lim_rows(
                    position_w,
                    yaw,
                    nominal_command,
                    obstacle_centers,
                    obstacle_radii,
                    obstacle_mask,
                )
            elif is_cbf_lse_qp_lim:
                desired, A, b, h_backup_log = self._cbf_lse_qp_lim_rows(
                    position_w,
                    yaw,
                    nominal_command,
                    obstacle_centers,
                    obstacle_radii,
                    obstacle_mask,
                )
            else:
                desired, A, b = self._direct_rows(
                    position_w,
                    yaw,
                    nominal_command,
                    obstacle_centers,
                    obstacle_radii,
                    obstacle_mask,
                )
                terms, grads_xyz = self._direct_backup_terms_and_grads(
                    position_w, yaw, obstacle_centers, obstacle_radii, obstacle_mask
                )
                h_backup_log = terms.min(dim=1).values  # hard minimum only for logging; non-LSE QP rows are individual

            command, diag = self.projector.project(desired, A, b, cost_weights)
            surrogate = command
            desired_surrogate = nominal_command
            correction = command - nominal_command
            h_lse = h_backup_log
            h_backup_lse = h_backup_log
            limits = self._limits(command.device, command.dtype)
            h_input = (limits - torch.abs(command)).min(dim=1).values
            input_bound_violation = torch.relu(torch.abs(command) - limits).max(dim=1).values

        individual_h = self._current_original_barriers(
            position_w,
            yaw,
            current_command,
            obstacle_centers,
            obstacle_radii,
            obstacle_mask,
        )
        active = torch.linalg.norm(correction.detach(), dim=1) > 1.0e-6
        return BackupSafetyFilterResult(
            command=command,
            surrogate_command=surrogate,
            desired_surrogate_command=desired_surrogate,
            h_lse=h_lse,
            h_backup_lse=h_backup_lse,
            h_input_lse=h_input,
            individual_h=individual_h,
            active=active,
            correction_norm=torch.linalg.norm(correction.detach(), dim=1),
            cbf_value_nominal=-diag.max_violation.detach(),
            lambda_value=diag.active_rows.detach(),
            mu_value=torch.zeros_like(h_lse),
            input_bound_violation=input_bound_violation.detach(),
            input_bound_active=input_bound_violation.detach() > 1.0e-6,
        )

    def _filter_cf_matched(
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
        """CF-matched differentiable-QP baseline (variant "cd_lse_qp_cfmatched").

        Reuses ZeroBackupLseCbfClosedForm's own primitives -- unchanged, not
        reimplemented -- to build the identical single scalar affine
        constraint that ZeroBackupLseCbfClosedForm.filter() solves in closed
        form:

            _backup_rollout_terms_and_grads -> softmin (kappa_backup) -> h_backup
            _input_lse                                              -> h_input
            softmin((h_backup, h_input), kappa_outer)                -> h_aug, grad
            drift(current_command, yaw) -> Lf = grad.drift, Lg = grad[:,3:]*a
            desired_surrogate (same nominal-derivative + tracking-sigma formula)
            alpha_outer, surrogate_cost_weights (same config fields)

        The only thing that differs from the closed form is *how* the
        projection "min 0.5||uhat-uhat_des||_W^2 s.t. Lf+Lg.uhat+alpha*h>=0"
        is solved: QpthProjector performs an exact batched differentiable
        QP/KKT solve (no straight-through estimator, no .detach() on the
        actor path), instead of the closed form's hand-derived analytic KKT
        formula. For a single affine constraint the two solve the same
        convex problem, so surrogate/command/residual and the actor-facing
        gradient should agree to solver tolerance -- that equivalence is
        exactly what scripts/test_cf_matched_qp_regression.py checks.

        The CF's active-but-degenerate fallback (||Lg h_aug||^2/W ~ 0 while
        the constraint is still violated at the desired surrogate) is
        reproduced explicitly here: a generic QP solver has no notion of
        this project-specific convention (apply the zero-surrogate backup
        rather than an undefined KKT correction), so it is applied as a
        post-solve override using the identical degeneracy test CF uses.
        """
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
            self.cfg.control_dynamics_a, device=current_command.device, dtype=current_command.dtype
        )
        sigma = torch.tensor(
            self.cfg.tracking_sigma, device=current_command.device, dtype=current_command.dtype
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
        valid = nominal_history_valid.to(device=nominal_command.device, dtype=nominal_command.dtype)
        if valid.ndim == 1:
            valid = valid[:, None]
        nominal_derivative = valid * (nominal_command - previous_nominal_command) / dt
        desired_surrogate = (
            current_command
            + nominal_derivative / a
            + (sigma / a) * (nominal_command - current_command)
        )

        alpha_outer = self._cfg_float("alpha_outer", _DEFAULT_ALPHA_OUTER)
        omega = lf + torch.sum(lg * desired_surrogate, dim=1) + alpha_outer * h_aug

        cost_weights = torch.as_tensor(
            getattr(self.cfg, "surrogate_cost_weights", (1.0, 1.0, 1.0)),
            device=lg.device,
            dtype=lg.dtype,
        )
        cost_weights = torch.clamp(cost_weights, min=eps)
        denominator_raw = torch.sum(lg.square() / cost_weights, dim=1)
        needs_correction = omega < 0.0
        nondegenerate = denominator_raw > eps
        degenerate_active = needs_correction & (~nondegenerate)

        # Same diagnostic Lagrange multiplier CF would report -- solver
        # independent (a single-halfspace QP's dual variable has this exact
        # closed form regardless of which primal solver produced uhat*), so
        # this is not "cheating": it does not feed into surrogate/command.
        lambda_value = torch.where(
            needs_correction & nondegenerate,
            -omega / torch.clamp(denominator_raw, min=eps),
            torch.zeros_like(omega),
        )

        # Single CF affine constraint, exactly as ZeroBackupLseCbfClosedForm
        # solves it:  Lf + Lg.uhat + alpha_outer*h_aug >= 0
        #         <=> -Lg.uhat <= Lf + alpha_outer*h_aug.
        # No separate box/input-CBF rows: the input limits already entered
        # through h_input inside the same outer-LSE h_aug above.
        A = -lg[:, None, :]
        b = (lf + alpha_outer * h_aug)[:, None]
        qp_surrogate, diag = self.cfmatched_projector.project(desired_surrogate, A, b, cost_weights)

        surrogate = torch.where(
            degenerate_active[:, None],
            torch.zeros_like(qp_surrogate),
            qp_surrogate,
        )

        decay = torch.exp(-a * dt)
        command_unclipped = decay * current_command + (1.0 - decay) * surrogate

        limits = torch.tensor(
            [self.action_cfg.max_vx, self.action_cfg.max_vy, self.action_cfg.max_wz],
            device=command_unclipped.device,
            dtype=command_unclipped.dtype,
        )
        input_bound_violation = torch.relu(torch.abs(command_unclipped) - limits).max(dim=1).values
        command = torch.clamp(command_unclipped, min=-limits, max=limits)

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
            mu_value=torch.zeros_like(omega),
            input_bound_violation=input_bound_violation,
            input_bound_active=input_bound_violation > 1.0e-6,
        )
