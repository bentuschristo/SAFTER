#!/usr/bin/env python3
"""Latency microbenchmark: SAFTER (bcbf_cf, closed-form) vs bCBF-QP
(bcbf_lse_cd_qp_cfmatched, LSE-aggregated single-row QP) vs Composite-QP
(bcbf_nonlse_cd_qp, fully unaggregated ~2700-row QP -- every current HOCBF,
backup-horizon, and terminal ray term kept as its own individual affine
constraint, per LaserZeroBackupNonLseControlDynamicsQp's docstring).

All three filters are constructed with IDENTICAL geometry/backup/LSE
hyperparameters -- the exact LaserSafetyConfig defaults this project trains
and evaluates with (SAFTER_src/high_level_laser_sac/config.py), with
backup_margin=0.6 and qp_backend="osqp" matching every eval command run
this session -- so the only thing that differs between the three timed
objects is the projection/solve algorithm itself.

Reports two numbers per filter per batch size:
  - "total": the full filter.filter(...) call (constraint construction +
    projection/solve).
  - "projection-only": just the solve step -- self.projector.project(...)
    for both QP variants, the module-level _exact_projection(...) call for
    the closed form (the legacy_projection=True path, matching every
    eval command's --legacy-projection flag).
Both are measured with a warmup phase, explicit torch.cuda.synchronize()
around each timed region (GPU ops are async -- unsynchronized wall-clock
timing is meaningless), and reported as mean/median/p95/p99 in milliseconds
over many repeated calls.

Pure tensor-op benchmark -- no MuJoCo/environment/checkpoint needed, so this
runs entirely locally regardless of server GPU availability.

Usage:
    python -u scripts/benchmark_safety_filter_latency.py --batch-sizes 1 500
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import torch

from SAFTER_src.high_level_rl import laser_cbf_filters as filters_module
from SAFTER_src.high_level_rl.laser_cbf_filters import (
    LaserScanCbfLseQp,
    LaserScanCbfQp,
    LaserZeroBackupLseCbfClosedForm,
    LaserZeroBackupLseControlDynamicsQpCfMatched,
    LaserZeroBackupLseQp,
    LaserZeroBackupNonLseControlDynamicsQp,
)

NUM_RAYS = 120
FOV_DEG = 360.0
MAX_RANGE = 5.0

# LaserSafetyConfig defaults (SAFTER_src/high_level_laser_sac/config.py),
# with backup_margin/qp_backend overridden to match every eval command run
# this session (--backup-margin 0.6 --qp-backend osqp).
SHARED_KWARGS = dict(
    d_safe=0.30,
    max_vx=2.0, max_vy=1.0, max_wz=1.0,
    robot_length_m=0.70,
    robot_width_m=0.31,
    lidar_x_m=0.28945,
    lidar_y_m=0.0,
    footprint_clearance_m=0.15,
    use_elliptic_footprint=True,
    max_range=MAX_RANGE,
    no_return_tolerance_m=1.0e-4,
    inactive_barrier_value=20.0,
    denominator_epsilon=1.0e-8,
    hard_clip_output=True,
)
BACKUP_KWARGS = dict(
    backup_margin=0.6,
    backup_horizon_s=4.0,
    backup_samples=21,
    backup_rollout_substeps=8,
    hocbf_k0=2.0,
    barrier_rate_scale=1.0,
    kappa_backup=50.0,
    kappa_input=20.0,
    kappa_outer=20.0,
    alpha_outer=0.8,
    control_dynamics_a=(8.0, 8.0, 8.0),
    tracking_sigma=(8.0, 8.0, 8.0),
    integration_dt=0.10,
    terminal_backup_mode="invariant_ellipse",
    terminal_geom_scale=1.0,
    terminal_rec_scale=1.0,
)
QP_KWARGS = dict(
    max_active_candidates=18,
    feasibility_tolerance=1.0e-5,
    qp_regularization=1.0e-5,
    straight_through=True,
    qp_backend="osqp",
)
# LaserZeroBackupLseControlDynamicsQpCfMatched ("bCBF-QP" above) ignores its
# own qp_backend/straight_through/max_active_candidates constructor args and
# always solves via QpthProjector internally (see class docstring/__init__,
# laser_cbf_filters.py:2089-2105) -- so for an apples-to-apples solver
# comparison, Composite-QP needs its own qpth-backed build too.
QP_KWARGS_QPTH = dict(QP_KWARGS, qp_backend="qpth")

# LaserZeroBackupLseQp ("bCBF-LSE-QP, no control dynamics") has no first-order
# command-state surrogate, so none of BACKUP_KWARGS' control_dynamics_a/
# tracking_sigma/integration_dt/terminal_backup_mode/hocbf_k0/kappa_input/
# kappa_outer/barrier_rate_scale apply -- it only takes the geometry +
# box-limit kwargs from SHARED_KWARGS, its own alpha (reusing BACKUP_KWARGS'
# alpha_outer value for an apples-to-apples strengthening gain), and its own
# backup_margin/backup_samples/kappa_backup (same values as BACKUP_KWARGS).
NOCD_KWARGS = dict(
    d_safe=SHARED_KWARGS["d_safe"],
    robot_length_m=SHARED_KWARGS["robot_length_m"],
    robot_width_m=SHARED_KWARGS["robot_width_m"],
    lidar_x_m=SHARED_KWARGS["lidar_x_m"],
    lidar_y_m=SHARED_KWARGS["lidar_y_m"],
    footprint_clearance_m=SHARED_KWARGS["footprint_clearance_m"],
    use_elliptic_footprint=SHARED_KWARGS["use_elliptic_footprint"],
    max_range=SHARED_KWARGS["max_range"],
    no_return_tolerance_m=SHARED_KWARGS["no_return_tolerance_m"],
    inactive_barrier_value=SHARED_KWARGS["inactive_barrier_value"],
    denominator_epsilon=SHARED_KWARGS["denominator_epsilon"],
    hard_clip_output=SHARED_KWARGS["hard_clip_output"],
    max_vx=SHARED_KWARGS["max_vx"],
    max_vy=SHARED_KWARGS["max_vy"],
    max_wz=SHARED_KWARGS["max_wz"],
    alpha=BACKUP_KWARGS["alpha_outer"],
    backup_margin=BACKUP_KWARGS["backup_margin"],
    backup_samples=BACKUP_KWARGS["backup_samples"],
    kappa_backup=BACKUP_KWARGS["kappa_backup"],
)


# Pure per-step control-input synthesis: no backup rollout/terminal set at
# all (not even the no-CD filter's trivial repeated-row backup), no
# control-dynamics surrogate. Only the direct relative-degree-one ray
# barrier(s) at the CURRENT instant feed the QP. Same geometry/box-limit/
# alpha/qp-backend values as NOCD_KWARGS, minus the backup_* fields, which
# these classes don't accept.
DIRECT_KWARGS = {
    k: v for k, v in NOCD_KWARGS.items()
    if k not in ("backup_margin", "backup_samples", "kappa_backup")
}


class _DirectFilterAdapter:
    """Adapts LaserScanCbfQp/LaserScanCbfLseQp's filter(ranges, angles,
    nominal_command) -- they take no command-dynamics/backup arguments at
    all -- to the benchmark harness's uniform 6-positional-argument call, so
    these backup-rollout-free, control-dynamics-free filters can be timed in
    the same loop as every other filter."""

    def __init__(self, inner):
        self._inner = inner
        self.projector = inner.projector

    def filter(self, ranges, angles, current_command, nominal_command,
               previous_nominal_command, nominal_history_valid):
        del current_command, previous_nominal_command, nominal_history_valid
        return self._inner.filter(ranges, angles, nominal_command)


def build_filters(device: torch.device):
    safter = LaserZeroBackupLseCbfClosedForm(
        **SHARED_KWARGS, **BACKUP_KWARGS, legacy_projection=True, straight_through=False,
    )
    bcbf_qp = LaserZeroBackupLseControlDynamicsQpCfMatched(
        **SHARED_KWARGS, **BACKUP_KWARGS, **QP_KWARGS,
    )
    bcbf_lse_qp_nocd = LaserZeroBackupLseQp(**NOCD_KWARGS, **QP_KWARGS)
    direct_qp = _DirectFilterAdapter(LaserScanCbfQp(**DIRECT_KWARGS, **QP_KWARGS))
    direct_lse_qp = _DirectFilterAdapter(LaserScanCbfLseQp(kappa=30.0, **DIRECT_KWARGS, **QP_KWARGS))
    composite_qp = LaserZeroBackupNonLseControlDynamicsQp(
        **SHARED_KWARGS, **BACKUP_KWARGS, **QP_KWARGS,
    )
    composite_qp_qpth = LaserZeroBackupNonLseControlDynamicsQp(
        **SHARED_KWARGS, **BACKUP_KWARGS, **QP_KWARGS_QPTH,
    )
    return {
        "SAFTER (bcbf_cf, closed-form)": safter,
        "bCBF-QP (bcbf_lse_cd_qp_cfmatched)": bcbf_qp,
        "bCBF-LSE-QP (no control dynamics)": bcbf_lse_qp_nocd,
        "Direct CBF-QP (no backup, no CD, unaggregated)": direct_qp,
        "Direct CBF-LSE-QP (no backup, no CD)": direct_lse_qp,
        "Composite-QP (bcbf_nonlse_cd_qp, osqp)": composite_qp,
        "Composite-QP (bcbf_nonlse_cd_qp, qpth)": composite_qp_qpth,
    }


def make_inputs(batch_size: int, device: torch.device, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ranges = (0.15 + (MAX_RANGE - 0.15) * torch.rand(batch_size, NUM_RAYS, generator=g)).to(device)
    half_fov = torch.deg2rad(torch.tensor(FOV_DEG / 2.0))
    angles = torch.linspace(-half_fov, half_fov, NUM_RAYS, device=device)
    limits = torch.tensor([2.0, 1.0, 1.0], device=device)
    current_command = (torch.rand(batch_size, 3, generator=g).to(device) * 2 - 1) * limits
    nominal_command = (torch.rand(batch_size, 3, generator=g).to(device) * 2 - 1) * limits
    previous_nominal_command = (torch.rand(batch_size, 3, generator=g).to(device) * 2 - 1) * limits
    nominal_history_valid = torch.ones(batch_size, 1, device=device)
    return ranges, angles, current_command, nominal_command, previous_nominal_command, nominal_history_valid


@dataclass
class TimingAccumulator:
    samples_ms: list[float] = field(default_factory=list)

    def record(self, elapsed_s: float) -> None:
        self.samples_ms.append(elapsed_s * 1000.0)

    def summary(self) -> dict[str, float]:
        s = sorted(self.samples_ms)
        n = len(s)
        return {
            "n": n,
            "mean": statistics.mean(s),
            "std": statistics.stdev(s) if n > 1 else 0.0,
            "min": s[0],
            "median": s[n // 2],
            "p95": s[int(0.95 * (n - 1))],
            "p99": s[int(0.99 * (n - 1))],
            "max": s[-1],
        }


def time_calls(fn, n_calls: int, device: torch.device) -> TimingAccumulator:
    acc = TimingAccumulator()
    is_cuda = device.type == "cuda"
    for _ in range(n_calls):
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize()
        acc.record(time.perf_counter() - start)
    return acc


def make_projection_only_timer(filter_obj, is_qp: bool):
    """Monkeypatches the solve step to accumulate wall-clock time spent
    inside it, returns (timer, restore_fn). QP variants: wrap the instance's
    own self.projector.project (each filter has its own projector instance,
    so this doesn't affect the other filters). Closed form: legacy_projection
    calls the module-level _exact_projection(...) by bare name, so it must be
    patched at module level (restored afterward)."""
    acc = TimingAccumulator()
    device = next(iter(filter_obj.__dict__.values())).device if False else None  # unused placeholder

    if is_qp:
        original_project = filter_obj.projector.project

        def wrapped_project(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = original_project(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            acc.record(time.perf_counter() - start)
            return result

        filter_obj.projector.project = wrapped_project

        def restore():
            filter_obj.projector.project = original_project

        return acc, restore
    else:
        original_fn = filters_module._exact_projection

        def wrapped_fn(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = original_fn(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            acc.record(time.perf_counter() - start)
            return result

        filters_module._exact_projection = wrapped_fn

        def restore():
            filters_module._exact_projection = original_fn

        return acc, restore


def fmt(summary: dict[str, float]) -> str:
    return (
        f"n={int(summary['n']):4d}  mean={summary['mean']:9.3f}ms  std={summary['std']:8.3f}ms  "
        f"min={summary['min']:9.3f}ms  median={summary['median']:9.3f}ms  "
        f"p95={summary['p95']:9.3f}ms  p99={summary['p99']:9.3f}ms  max={summary['max']:9.3f}ms"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 500])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--n-calls", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", type=str, nargs="+", default=None,
                         help="Substring filter(s) on method name; only matching methods are run.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"num_rays={NUM_RAYS} fov_deg={FOV_DEG} max_range={MAX_RANGE} "
          f"backup_margin={BACKUP_KWARGS['backup_margin']} qp_backend={QP_KWARGS['qp_backend']}")

    filters = build_filters(device)
    if args.only:
        filters = {
            name: filt for name, filt in filters.items()
            if any(sub in name for sub in args.only)
        }
        if not filters:
            raise SystemExit(f"--only matched no methods (available: {list(build_filters(device))})")
    for f in filters.values():
        if hasattr(f, "to"):
            pass  # these are plain python objects with internal tensors built on first call's device

    with torch.no_grad():
        for batch_size in args.batch_sizes:
            print(f"\n{'='*90}\nBATCH SIZE = {batch_size}\n{'='*90}")
            inputs = make_inputs(batch_size, device, seed=args.seed)

            for name, filt in filters.items():
                is_qp = hasattr(filt, "projector")

                # Warmup (also lets lazily-built internal buffers settle on device).
                for _ in range(args.warmup):
                    filt.filter(*inputs)

                total_acc = time_calls(lambda: filt.filter(*inputs), args.n_calls, device)

                proj_acc, restore = make_projection_only_timer(filt, is_qp)
                for _ in range(args.n_calls):
                    filt.filter(*inputs)
                restore()

                print(f"{name:38s} TOTAL:            {fmt(total_acc.summary())}")
                print(f"{'':38s} PROJECTION-ONLY:  {fmt(proj_acc.summary())}")


if __name__ == "__main__":
    main()
