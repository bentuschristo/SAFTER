"""Pre-training validation of the v6 invariant-ellipse bCBF-CF using the
EXACT runtime safety geometry the planned smoke/sweep training command
would actually construct (--safety-clearance 0.30), not raw LaserSafetyConfig
defaults.

Reuses every test function from test_bcbf_terminal_invariance.py UNCHANGED
(no terminal formulation, terminal scale, controller, or SAC logic is
touched here) -- this script only changes WHICH filter instances those
functions are called with, by resolving the runtime geometry first via
_runtime_safety_config.resolve_runtime_safety_config (the same formula the
training script itself uses).

Run:
    python scripts_august/scripts_v6/validate_runtime_geometry_d_safe_030.py
"""
from __future__ import annotations

import math

import torch

from _runtime_safety_config import resolve_runtime_safety_config
from src.high_level_rl.laser_cbf_filters import LaserZeroBackupLseCbfClosedForm
from src.high_level_rl.zero_backup_flow import exact_zero_backup_state_and_phi
from test_bcbf_terminal_invariance import (
    DEVICE,
    DTYPE,
    build_state_bank,
    microbenchmark,
    percentiles_report,
    test_a_geometry_at_rest,
    test_b_residual_motion,
    test_c_analytic_gradient,
    test_d_chain_rule_gradient,
    test_e_forward_invariance,
    test_f_expanded,
    test_g_legacy_vs_new,
    test_h_containment,
    test_nagumo_geom_boundary,
    test_nagumo_rec_boundary,
    test_terminal_influence,
)

SAFETY_CLEARANCE = 0.30
EXPECTED = {
    "d_safe": 0.30,
    "footprint_clearance_m": 0.23945,
    "half_length": 0.35,
    "half_width": 0.155,
    "r_x": 0.58945,
    "r_y": 0.39445,
    "d_b": 0.30,
    "R_x": 0.88945,
    "R_y": 0.69445,
    "c_theta": 0.25002,
    "a_decay_min": 8.0,
}
TOL = 1.0e-4  # matches the 5-decimal precision of the quoted expected values


def build_filter(mode: str) -> LaserZeroBackupLseCbfClosedForm:
    safety_cfg = resolve_runtime_safety_config(
        SAFETY_CLEARANCE,
        terminal_backup_mode=mode,
        terminal_geom_scale=1.0,
        terminal_rec_scale=1.0,
    )
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
        control_dynamics_a=safety_cfg.control_dynamics_a,
        tracking_sigma=safety_cfg.tracking_sigma,
        integration_dt=safety_cfg.integration_dt,
        robot_length_m=safety_cfg.robot_length_m,
        robot_width_m=safety_cfg.robot_width_m,
        lidar_x_m=safety_cfg.lidar_x_m,
        lidar_y_m=safety_cfg.lidar_y_m,
        footprint_clearance_m=safety_cfg.footprint_clearance_m,
        use_elliptic_footprint=safety_cfg.use_elliptic_footprint,
        max_range=5.0,
        no_return_tolerance_m=safety_cfg.no_return_tolerance_m,
        inactive_barrier_value=safety_cfg.inactive_barrier_value,
        denominator_epsilon=safety_cfg.denominator_epsilon,
        hard_clip_output=safety_cfg.hard_clip_output,
        terminal_backup_mode=safety_cfg.terminal_backup_mode,
        terminal_geom_scale=safety_cfg.terminal_geom_scale,
        terminal_rec_scale=safety_cfg.terminal_rec_scale,
    )


def section1_resolve_and_assert() -> LaserZeroBackupLseCbfClosedForm:
    print("=" * 70)
    print("SECTION 1: resolve and print the exact runtime geometry")
    print("=" * 70)

    filt = build_filter("invariant_ellipse")
    a = torch.as_tensor(filt.a_tuple)
    a_decay_min = float(torch.clamp(torch.min(a), min=filt.eps))

    actual = {
        "d_safe": filt.d_safe,
        "footprint_clearance_m": filt.footprint_clearance_m,
        "half_length": 0.5 * filt.robot_length_m,
        "half_width": 0.5 * filt.robot_width_m,
        "r_x": filt.axis_x,
        "r_y": filt.axis_y,
        "d_b": filt.backup_margin,
        "R_x": filt.terminal_big_r_x,
        "R_y": filt.terminal_big_r_y,
        "c_theta": filt.terminal_c_theta,
        "a_decay_min": a_decay_min,
    }

    print(f"{'quantity':24s} {'actual':>14s} {'expected':>14s} {'match':>8s}")
    all_ok = True
    for key, expected_value in EXPECTED.items():
        actual_value = actual[key]
        ok = abs(actual_value - expected_value) < TOL
        all_ok &= ok
        print(f"{key:24s} {actual_value:14.6f} {expected_value:14.6f} {'OK' if ok else 'MISMATCH':>8s}")

    if not all_ok:
        raise AssertionError(
            "Runtime geometry does not match the expected checkpoint-derived "
            "values -- stopping before running any tests, per instructions."
        )
    print("\nAll runtime geometry values match expected checkpoint-derived values.")

    print("\nCode path / formula (train_SAC_policies.py"
          " lines ~407-419, replicated verbatim in _runtime_safety_config.py):")
    print("  front_body_distance = max(0.5*robot_length_m - lidar_x_m, 1e-8)")
    print(f"                      = max(0.5*{filt.robot_length_m} - {filt.lidar_x_m}, 1e-8)")
    print(f"                      = max({0.5*filt.robot_length_m - filt.lidar_x_m:.5f}, 1e-8) "
          f"= {max(0.5*filt.robot_length_m - filt.lidar_x_m, 1e-8):.5f}")
    print("  footprint_clearance_m = max(0.0, safety_clearance - front_body_distance)")
    print(f"                        = max(0.0, {SAFETY_CLEARANCE} - "
          f"{max(0.5*filt.robot_length_m - filt.lidar_x_m, 1e-8):.5f}) = {filt.footprint_clearance_m:.5f}")
    return filt


def section6_numerical_robustness(filt: LaserZeroBackupLseCbfClosedForm, n: int = 50_000, num_rays: int = 60) -> bool:
    print("\n" + "=" * 70)
    print("SECTION 6: numerical robustness (min valid E_i, NaN/Inf, E_i thresholds)")
    print("=" * 70)
    ranges, angles, current_command, nominal_command = build_state_bank(filt, n, num_rays, seed=2026)
    a = torch.as_tensor(filt.a_tuple, dtype=DTYPE, device=DEVICE)
    a_decay_min = torch.clamp(torch.min(a), min=filt.eps)
    horizon = torch.tensor([filt.backup_horizon_s], dtype=DTYPE, device=DEVICE)
    z0 = torch.cat((torch.zeros(n, 3, dtype=DTYPE, device=DEVICE), current_command), dim=1)
    z_t, _ = exact_zero_backup_state_and_phi(z0, horizon, a)
    z_terminal = z_t[:, -1, :]
    points, valid = filt.scan_points(ranges, angles, n, device=DEVICE, dtype=DTYPE)
    h_geom, grad_geom, h_rec, grad_rec = filt.terminal_invariant_h_and_grad_future(
        points, z_terminal, big_r_x=filt.terminal_big_r_x, big_r_y=filt.terminal_big_r_y,
        c_theta=filt.terminal_c_theta, a_decay_min=a_decay_min,
    )
    g_valid = h_geom[valid]
    e_valid = torch.exp(2.0 * g_valid)

    finite_g = torch.all(torch.isfinite(h_geom))
    finite_rec = torch.all(torch.isfinite(h_rec))
    finite_grad_geom = torch.all(torch.isfinite(grad_geom))
    finite_grad_rec = torch.all(torch.isfinite(grad_rec))
    finite_e = torch.all(torch.isfinite(e_valid))
    no_nan_inf = bool(finite_g and finite_rec and finite_grad_geom and finite_grad_rec and finite_e)

    out = filt.filter(ranges, angles, current_command, nominal_command)
    fallback_events = int((~out.projection_feasible).sum()) if out.projection_feasible is not None else -1

    min_valid_e = float(e_valid.min())
    count_1e8 = int((e_valid < 1.0e-8).sum())
    count_1e10 = int((e_valid < 1.0e-10).sum())
    count_1e12 = int((e_valid < 1.0e-12).sum())

    print(f"  tested valid (state,ray) pairs: {int(valid.sum())}")
    print(f"  no NaN/Inf anywhere (h_geom, h_rec, grad_geom, grad_rec, E_i): {no_nan_inf}")
    print(f"  minimum valid E_i: {min_valid_e:.6e}")
    print(f"  count(E_i < 1e-8):  {count_1e8}")
    print(f"  count(E_i < 1e-10): {count_1e10}")
    print(f"  count(E_i < 1e-12): {count_1e12}")
    print(f"  loss-of-authority fallback events (~projection_feasible): {fallback_events} / {n}")
    d_aug = out.lg_aug_sq_norm
    near_singular_1e6 = int((d_aug < 1.0e-6).sum())
    near_singular_1e4 = int((d_aug < 1.0e-4).sum())
    print(f"  near-degenerate D_aug<1e-6 events: {near_singular_1e6} / {n}")
    print(f"  near-degenerate D_aug<1e-4 events: {near_singular_1e4} / {n}")

    ok = no_nan_inf and fallback_events == 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] no NaN/Inf and no unexpected fallback events")
    return ok


def main() -> None:
    print(f"device={DEVICE}  dtype={DTYPE}  safety_clearance={SAFETY_CLEARANCE}\n")
    filt_invariant = section1_resolve_and_assert()
    filt_legacy = build_filter("legacy_inflated_ellipse")

    print("\n" + "=" * 70)
    print("SECTION 2: full terminal/invariance test suite (runtime geometry)")
    print("=" * 70)
    results: list[tuple[str, bool]] = []
    results.append(("A", test_a_geometry_at_rest(filt_invariant)))
    results.append(("B", test_b_residual_motion(filt_invariant)))
    results.append(("C", test_c_analytic_gradient(filt_invariant)))
    results.append(("D", test_d_chain_rule_gradient(filt_invariant)))
    results.append(("E", test_e_forward_invariance(filt_invariant)))
    results.append(("F", test_f_expanded(filt_legacy, filt_invariant)))  # sections 4 also covered here
    results.append(("G", test_g_legacy_vs_new(filt_legacy, filt_invariant)))
    results.append(("Nagumo-A", test_nagumo_rec_boundary(filt_invariant)))
    results.append(("Nagumo-B", test_nagumo_geom_boundary(filt_invariant)))
    results.append(("H-containment", test_h_containment(filt_invariant)))  # section 3

    print("\n" + "=" * 70)
    print("SECTION 5: terminal-constituent influence diagnostic (interpretation only)")
    print("=" * 70)
    test_terminal_influence(filt_invariant)

    results.append(("Numerical-robustness", section6_numerical_robustness(filt_invariant)))

    microbenchmark(filt_legacy, filt_invariant)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_ok = True
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        all_ok &= ok
    print("\nOVERALL:", "PASS" if all_ok else "FAIL")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
