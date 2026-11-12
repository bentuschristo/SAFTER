"""Unit tests P1-P7 for the ZOH-consistent box+half-space joint projection
(2026-09-07 hard-clip-consistency fix).

Tests the NEW `_exact_box_halfspace_projection` / `zoh_surrogate_bounds`
functions in isolation (synthetic random (desired, lg, b, lower, upper)
tuples) -- does NOT touch or re-run the terminal-set math suite, per task
instructions.

Run:
    python scripts_august/scripts_v6/test_joint_box_halfspace_projection.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.high_level_rl.laser_cbf_filters import (
    _exact_box_halfspace_projection,
    zoh_surrogate_bounds,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
TOL = 1.0e-6


def _report(name: str, ok: bool) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def make_random_problem(n: int, seed: int, *, force_box_active: bool = False):
    torch.manual_seed(seed)
    a = torch.tensor([8.0, 8.0, 8.0], dtype=DTYPE, device=DEVICE)
    dt = 0.1
    limits = torch.tensor([0.5, 0.3, 0.5], dtype=DTYPE, device=DEVICE)
    current_command = (torch.rand(n, 3, dtype=DTYPE, device=DEVICE) - 0.5) * 1.6 * limits
    lower, upper, decay = zoh_surrogate_bounds(current_command, limits, a, dt)

    desired = (torch.rand(n, 3, dtype=DTYPE, device=DEVICE) - 0.5) * 4.0
    if force_box_active:
        # push desired well outside the box so clipping/box-activity is common
        desired = desired * 3.0

    lg = (torch.rand(n, 3, dtype=DTYPE, device=DEVICE) - 0.5) * 4.0
    b = (torch.rand(n, dtype=DTYPE, device=DEVICE) - 0.3) * 3.0  # mix of active/inactive halfspace
    return desired, lg, b, lower, upper, limits, current_command, decay, a, dt


def test_p1_box_feasibility(n: int = 20_000) -> bool:
    print("\n=== P1: box feasibility ===")
    desired, lg, b, lower, upper, *_ = make_random_problem(n, seed=1)
    proj, *_ = _exact_box_halfspace_projection(desired, lg, b, lower, upper, eps=1e-8)
    viol = torch.clamp(lower - proj, min=0.0) + torch.clamp(proj - upper, min=0.0)
    max_viol = float(viol.max())
    print(f"  max box violation: {max_viol:.3e}")
    return _report("u_hat_lower <= u_hat_star <= u_hat_upper (zero violations)", max_viol < 1e-7)


def test_p2_halfspace_feasibility(n: int = 20_000) -> bool:
    print("\n=== P2: half-space feasibility (jointly feasible samples only) ===")
    desired, lg, b, lower, upper, *_ = make_random_problem(n, seed=2)
    proj, omega, lam, feasible, residual, joint_infeasible = _exact_box_halfspace_projection(
        desired, lg, b, lower, upper, eps=1e-8
    )
    res_feasible = residual[feasible]
    min_res = float(res_feasible.min()) if res_feasible.numel() else float("nan")
    print(f"  feasible samples: {int(feasible.sum())}/{n}, min residual: {min_res:.3e}")
    return _report("a@u_hat_star+b >= -tol for all feasible samples", min_res >= -1e-6)


def test_p3_zoh_physical_bounds(n: int = 20_000) -> bool:
    print("\n=== P3: ZOH physical bounds after mapping back through the ZOH update ===")
    desired, lg, b, lower, upper, limits, current_command, decay, a, dt = make_random_problem(n, seed=3)
    proj, *_ = _exact_box_halfspace_projection(desired, lg, b, lower, upper, eps=1e-8)
    u_next = decay[None, :] * current_command + (1.0 - decay[None, :]) * proj
    viol = torch.clamp(-limits[None, :] - u_next, min=0.0) + torch.clamp(u_next - limits[None, :], min=0.0)
    max_viol = float(viol.max())
    print(f"  max physical-bound violation after ZOH mapping: {max_viol:.3e}")
    return _report("u_min <= u_next <= u_max (zero violations)", max_viol < 1e-6)


def _scipy_reference(desired, lg, b, lower, upper):
    """Trusted (but slow, Python-loop, test-only) reference QP via scipy.
    NEVER used in runtime bCBF-CF -- ground truth for P4 only."""
    n = desired.shape[0]
    sols = np.zeros((n, 3))
    objs = np.zeros(n)
    feas = np.zeros(n, dtype=bool)
    for i in range(n):
        u0 = desired[i].cpu().numpy()
        ai = lg[i].cpu().numpy()
        bi = float(b[i])
        lo = lower[i].cpu().numpy()
        hi = upper[i].cpu().numpy()
        cons = [{"type": "ineq", "fun": lambda u, ai=ai, bi=bi: ai @ u + bi}]
        bounds = list(zip(lo.tolist(), hi.tolist()))
        res = minimize(
            lambda u: 0.5 * np.sum((u - u0) ** 2), x0=np.clip(u0, lo, hi),
            jac=lambda u: (u - u0), bounds=bounds, constraints=cons,
            method="SLSQP", options={"ftol": 1e-12, "maxiter": 200},
        )
        sols[i] = res.x
        objs[i] = 0.5 * np.sum((res.x - u0) ** 2)
        feas[i] = bool(res.success) and (ai @ res.x + bi >= -1e-5)
    return sols, objs, feas


def test_p4_exactness_vs_reference(n: int = 300) -> bool:
    print(f"\n=== P4: exactness against scipy SLSQP reference (n={n}) ===")
    desired, lg, b, lower, upper, *_ = make_random_problem(n, seed=4, force_box_active=True)
    proj, omega, lam, feasible, residual, joint_infeasible = _exact_box_halfspace_projection(
        desired, lg, b, lower, upper, eps=1e-8
    )
    # Only compare on jointly-feasible samples (empty-intersection handling
    # is tested separately in P7).
    mask = feasible.cpu().numpy()
    ref_sol, ref_obj, ref_feas = _scipy_reference(desired[mask], lg[mask], b[mask], lower[mask], upper[mask])
    ours_sol = proj[mask].cpu().numpy()
    ours_obj = 0.5 * np.sum((ours_sol - desired[mask].cpu().numpy()) ** 2, axis=1)

    sol_err = np.abs(ours_sol - ref_sol).max()
    obj_err = np.abs(ours_obj - ref_obj).max()
    print(f"  compared {mask.sum()} feasible samples")
    print(f"  max |solution error|: {sol_err:.3e}")
    print(f"  max |objective error|: {obj_err:.3e}")
    print(f"  reference solver reported infeasible on: {int((~ref_feas).sum())} (should be ~0)")
    ok = sol_err < 1e-4 and obj_err < 1e-6
    return _report("solution/objective match scipy reference to near float tolerance", ok)


def test_p5_original_closed_form_recovery(n: int = 20_000) -> bool:
    print("\n=== P5: recovers the original half-space-only solution when it's inside the box ===")
    from src.high_level_rl.laser_cbf_filters import _exact_projection

    torch.manual_seed(5)
    a = torch.tensor([8.0, 8.0, 8.0], dtype=DTYPE, device=DEVICE)
    dt = 0.1
    limits = torch.tensor([0.5, 0.3, 0.5], dtype=DTYPE, device=DEVICE)
    current_command = torch.zeros(n, 3, dtype=DTYPE, device=DEVICE)  # u_t=0 widens the ZOH box maximally
    lower, upper, decay = zoh_surrogate_bounds(current_command, limits, a, dt)

    # Keep `desired` and the halfspace gentle enough that the ORIGINAL
    # (unbounded) projection lands inside this wide box.
    desired = (torch.rand(n, 3, dtype=DTYPE, device=DEVICE) - 0.5) * 0.2
    lg = (torch.rand(n, 3, dtype=DTYPE, device=DEVICE) - 0.5) * 0.6
    b = torch.rand(n, dtype=DTYPE, device=DEVICE) * 0.05  # small positive offset -> mild/no violation

    fallback = torch.zeros_like(desired)
    orig_proj, orig_omega, orig_lam, orig_feasible, orig_residual = _exact_projection(
        desired, lg, b, eps=1e-8, fallback=fallback
    )
    new_proj, new_omega, new_lam, new_feasible, new_residual, new_infeasible = _exact_box_halfspace_projection(
        desired, lg, b, lower, upper, eps=1e-8
    )
    inside_box = torch.all((orig_proj >= lower) & (orig_proj <= upper), dim=1)
    diff = torch.linalg.norm(orig_proj - new_proj, dim=1)
    max_diff_inside = float(diff[inside_box].max()) if torch.any(inside_box) else float("nan")
    print(f"  samples where original solution lies inside the ZOH box: {int(inside_box.sum())}/{n}")
    print(f"  max ||orig_proj - new_proj|| on those samples: {max_diff_inside:.3e}")
    return _report(
        "new projection == original Eq.(20) solution when it's already box-feasible",
        torch.any(inside_box).item() and max_diff_inside < 1e-6,
    )


def test_p6_gradient_check() -> bool:
    print("\n=== P6: autograd vs finite differences across active-set regimes ===")
    torch.manual_seed(6)
    a = torch.tensor([8.0, 8.0, 8.0], dtype=DTYPE, device=DEVICE)
    dt = 0.1
    limits = torch.tensor([0.5, 0.3, 0.5], dtype=DTYPE, device=DEVICE)

    regimes = {
        "inactive (nominal already feasible+in-box)": dict(
            current_command=torch.zeros(1, 3, dtype=DTYPE, device=DEVICE),
            desired=torch.tensor([[0.05, 0.02, -0.03]], dtype=DTYPE, device=DEVICE),
            lg=torch.tensor([[0.1, 0.1, 0.1]], dtype=DTYPE, device=DEVICE),
            b=torch.tensor([2.0], dtype=DTYPE, device=DEVICE),
        ),
        "halfspace-active (box not binding)": dict(
            current_command=torch.zeros(1, 3, dtype=DTYPE, device=DEVICE),
            desired=torch.tensor([[0.05, 0.02, -0.03]], dtype=DTYPE, device=DEVICE),
            lg=torch.tensor([[1.0, 1.0, 1.0]], dtype=DTYPE, device=DEVICE),
            b=torch.tensor([-1.0], dtype=DTYPE, device=DEVICE),
        ),
        "box-active (halfspace not binding)": dict(
            current_command=torch.zeros(1, 3, dtype=DTYPE, device=DEVICE),
            desired=torch.tensor([[5.0, 5.0, -5.0]], dtype=DTYPE, device=DEVICE),
            lg=torch.tensor([[0.01, 0.01, 0.01]], dtype=DTYPE, device=DEVICE),
            b=torch.tensor([5.0], dtype=DTYPE, device=DEVICE),
        ),
        "halfspace+box-active jointly": dict(
            current_command=torch.zeros(1, 3, dtype=DTYPE, device=DEVICE),
            desired=torch.tensor([[5.0, 5.0, -5.0]], dtype=DTYPE, device=DEVICE),
            lg=torch.tensor([[1.0, 1.0, 1.0]], dtype=DTYPE, device=DEVICE),
            b=torch.tensor([-3.0], dtype=DTYPE, device=DEVICE),
        ),
    }

    all_ok = True
    for name, prob in regimes.items():
        lower, upper, decay = zoh_surrogate_bounds(prob["current_command"], limits, a, dt)
        desired = prob["desired"].clone().requires_grad_(True)
        lg = prob["lg"].clone().requires_grad_(True)
        b = prob["b"].clone().requires_grad_(True)

        proj, *_ = _exact_box_halfspace_projection(desired, lg, b, lower, upper, eps=1e-8)
        loss = proj.sum()
        grads = torch.autograd.grad(loss, (desired, lg, b), retain_graph=False)

        def f(desired_v, lg_v, b_v):
            p, *_ = _exact_box_halfspace_projection(desired_v, lg_v, b_v, lower, upper, eps=1e-8)
            return p.sum()

        h = 1e-6
        fd_ok = True
        max_err = 0.0
        for tensor_name, tensor, analytic_grad in (
            ("desired", desired, grads[0]), ("lg", lg, grads[1]), ("b", b, grads[2]),
        ):
            flat = tensor.detach().flatten()
            analytic_flat = analytic_grad.flatten()
            for k in range(flat.numel()):
                plus = flat.clone(); plus[k] += h
                minus = flat.clone(); minus[k] -= h
                args = {"desired": desired.detach(), "lg": lg.detach(), "b": b.detach()}
                args[tensor_name] = plus.reshape(tensor.shape)
                f_plus = f(args["desired"], args["lg"], args["b"])
                args[tensor_name] = minus.reshape(tensor.shape)
                f_minus = f(args["desired"], args["lg"], args["b"])
                fd_grad = (f_plus - f_minus) / (2 * h)
                err = abs(float(fd_grad) - float(analytic_flat[k]))
                max_err = max(max_err, err)
        print(f"  [{name}]: max |autograd - finite-diff| = {max_err:.3e}")
        fd_ok = max_err < 1e-3  # loose tolerance: piecewise/nonsmooth boundaries are expected
        all_ok &= fd_ok
    return _report("gradients match finite differences in all 4 regimes", all_ok)


def test_p7_empty_intersection(n: int = 5000) -> bool:
    print("\n=== P7: empty box/half-space intersection handling ===")
    torch.manual_seed(7)
    a = torch.tensor([8.0, 8.0, 8.0], dtype=DTYPE, device=DEVICE)
    dt = 0.1
    limits = torch.tensor([0.5, 0.3, 0.5], dtype=DTYPE, device=DEVICE)
    current_command = torch.zeros(n, 3, dtype=DTYPE, device=DEVICE)
    lower, upper, decay = zoh_surrogate_bounds(current_command, limits, a, dt)

    # Construct a genuinely infeasible halfspace: require lg@u+b>=0 with lg
    # pointing away from the box entirely (residual_max<0 by construction).
    desired = (torch.rand(n, 3, dtype=DTYPE, device=DEVICE) - 0.5) * 0.2
    lg = torch.ones(n, 3, dtype=DTYPE, device=DEVICE)
    # residual_max = lg@u_maxsafe + b where u_maxsafe=upper (since lg>=0);
    # pick b very negative so even the best box point can't satisfy it.
    u_maxsafe_val = float(torch.sum(torch.ones(3, dtype=DTYPE, device=DEVICE) * upper[0]))
    b = torch.full((n,), -(u_maxsafe_val + 10.0), dtype=DTYPE, device=DEVICE)

    proj, omega, lam, feasible, residual, joint_infeasible = _exact_box_halfspace_projection(
        desired, lg, b, lower, upper, eps=1e-8
    )
    detected = bool(torch.all(joint_infeasible))
    in_box = torch.all((proj >= lower - 1e-7) & (proj <= upper + 1e-7), dim=1)
    all_in_box = bool(torch.all(in_box))
    u_maxsafe = torch.where(lg >= 0.0, upper, lower)
    matches_maxsafe = bool(torch.allclose(proj, u_maxsafe, atol=1e-6))
    print(f"  joint_infeasible detected for all {n} constructed-infeasible samples: {detected}")
    print(f"  returned point stays inside the box for all samples: {all_in_box}")
    print(f"  returned point equals the residual-maximizing box point: {matches_maxsafe}")
    ok = detected and all_in_box and matches_maxsafe
    return _report("empty intersection detected, returned point in-box and residual-maximizing", ok)


def main() -> None:
    print(f"device={DEVICE} dtype={DTYPE}")
    results = []
    results.append(("P1", test_p1_box_feasibility()))
    results.append(("P2", test_p2_halfspace_feasibility()))
    results.append(("P3", test_p3_zoh_physical_bounds()))
    results.append(("P4", test_p4_exactness_vs_reference()))
    results.append(("P5", test_p5_original_closed_form_recovery()))
    results.append(("P6", test_p6_gradient_check()))
    results.append(("P7", test_p7_empty_intersection()))

    print("\n=== Summary ===")
    all_ok = True
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        all_ok &= ok
    print("\nOVERALL:", "PASS" if all_ok else "FAIL")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
