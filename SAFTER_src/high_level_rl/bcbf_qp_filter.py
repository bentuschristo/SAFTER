# NOTE: this is a TRIMMED duplicate of src/high_level_rl/bcbf_qp_filter.py --
# only QpSolveDiagnostics/SmallBoxQpProjector/OsqpProjector/QpthProjector
# (the classes laser_cbf_filters.py actually lazy-imports) are kept. The
# original's ZeroBackupBcbfQpFilter class is map-based-only (subclasses the
# map-based ZeroBackupLseCbfClosedForm) and is intentionally dropped here.
# See SAFTER_src/README.md.

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import torch
from torch.func import jacrev, vmap

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

