"""Timing/diagnostic profiler for LaserScanCbfLseControlDynamicsQpCfMatched
(cbf_lse_cd_qp_cfmatched) vs. LaserZeroBackupLseControlDynamicsQpCfMatched
(bcbf_lse_cd_qp_cfmatched), on identical frozen synthetic batches.

Times four phases per filter() call, with torch.cuda.synchronize() around
each GPU-timed section:
  1. certificate construction  -- everything up to building (A, b)
  2. qpth forward solve        -- self.projector.project(...), no backward
  3. qpth backward             -- .backward() on a fixed scalar probe of
                                   the returned command
  4. total filter() time       -- the full call, forward only

Also logs, per call:
  Omega = Lf*h + Lg*h . u_nom + alpha*h      (cbf_value_nominal)
  active fraction = mean(Omega < 0)
  D = ||Lg h||^2                             min / median / p1

Two identical-shape batches (B, num_rays) are constructed once (same seed)
and reused for both methods, so any timing difference reflects the filter
math itself, not batch composition.

This script does NOT run automatically as part of any training/eval job.
Run manually once the GPU is free:

    python3 scripts/profile_cfmatched_qp_filters.py --batch 256 --iters 30

(use `python3 scripts/profile_cfmatched_qp_filters.py --smoke` for a fast
2-iteration, batch=8 correctness-only check that does not require the GPU
to be free).
"""
from __future__ import annotations

import argparse
import time

import torch

from src.high_level_rl.laser_cbf_filters import (
    LaserScanCbfLseControlDynamicsQpCfMatched,
    LaserZeroBackupLseControlDynamicsQpCfMatched,
)

# Matches the seaspeed config used by every cfmatched training run this
# session: asymmetric vx, backup_margin=0.6, invariant-ellipse terminal set.
COMMON_KWARGS = dict(
    d_safe=0.30,
    max_vx=2.0,
    max_vy=1.0,
    max_wz=1.0,
    robot_length_m=0.70,
    robot_width_m=0.31,
    lidar_x_m=0.28945,
    lidar_y_m=0.0,
    footprint_clearance_m=0.15,
    use_elliptic_footprint=True,
    max_range=5.0,
    no_return_tolerance_m=1.0e-4,
    inactive_barrier_value=20.0,
    denominator_epsilon=1.0e-8,
    hard_clip_output=True,
    control_dynamics_a=(8.0, 8.0, 8.0),
    tracking_sigma=(8.0, 8.0, 8.0),
    integration_dt=0.10,
)
BACKUP_KWARGS = dict(
    backup_margin=0.6,
    backup_horizon_s=4.0,
    backup_samples=21,
    backup_rollout_substeps=8,
    terminal_backup_mode="invariant_ellipse",
    terminal_geom_scale=1.0,
    terminal_rec_scale=1.0,
    legacy_projection=True,
)
NUM_RAYS = 120
FOV_DEG = 360.0


def make_frozen_batch(batch: int, device: torch.device, dtype: torch.dtype, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ranges = (0.5 + 4.0 * torch.rand(batch, NUM_RAYS, generator=g)).to(device=device, dtype=dtype)
    angles = torch.linspace(-3.14159265, 3.14159265, NUM_RAYS, dtype=dtype, device=device)
    vx = (torch.rand(batch, generator=g) * 3.0 - 1.0)
    vy = (torch.rand(batch, generator=g) * 2.0 - 1.0)
    wz = (torch.rand(batch, generator=g) * 2.0 - 1.0)
    current_command = torch.stack((vx, vy, wz), dim=1).to(device=device, dtype=dtype)
    nominal_command = current_command + 0.05 * torch.randn(batch, 3, generator=g).to(device=device, dtype=dtype)
    previous_nominal_command = current_command.clone()
    nominal_history_valid = torch.ones(batch, 1, device=device, dtype=dtype)
    return ranges, angles, current_command, nominal_command, previous_nominal_command, nominal_history_valid


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(t: torch.Tensor, q: float) -> float:
    return torch.quantile(t.detach().float().cpu(), q).item()


def _compute_certificate(filt, ranges, angles, current_command):
    """Replicate each *CfMatched class's filter() up to (h_aug, lf, lg) --
    i.e. everything BEFORE the QP solve -- by calling the same internal
    methods the real filter() uses, so certificate-construction time can be
    isolated from the qpth solve without editing the training-path classes.
    Dispatches on class name since CBF (ray-based) and bCBF (backup-based)
    build their certificate differently.
    """
    from src.high_level_rl.laser_cbf_filters import LaserZeroBackupLseControlDynamicsQpCfMatched, _softmin_value_lie

    if isinstance(filt, LaserZeroBackupLseControlDynamicsQpCfMatched):
        a = torch.as_tensor(filt.a_tuple, device=current_command.device, dtype=current_command.dtype)
        h_backup, lf_backup, lg_backup, _, _, _ = filt._backup_terms_and_lie(ranges, angles, current_command)
        h_input, lf_input, lg_input, _ = filt._input_lse_lie(current_command, a)
        h_aug, lf, lg = _softmin_value_lie(
            torch.stack((h_backup, h_input), dim=1),
            torch.stack((lf_backup, lf_input), dim=1),
            torch.stack((lg_backup, lg_input), dim=1),
            filt.kappa_outer,
        )
    else:
        batch_size = current_command.shape[0]
        points, valid = filt.scan_points(
            ranges, angles, batch_size, device=current_command.device, dtype=current_command.dtype
        )
        psi, grad_psi = filt.current_psi_and_grad(points, current_command, filt.k0)
        psi, grad_psi = filt.mask_terms(psi, valid, grad_psi)
        psi_n = psi / filt.rate_scale
        grad_psi_n = grad_psi / filt.rate_scale
        a_cd = torch.as_tensor(filt.a_tuple, device=current_command.device, dtype=current_command.dtype)
        fhat0 = torch.cat((current_command, -a_cd * current_command), dim=1)
        lf_ray_terms = torch.sum(grad_psi_n * fhat0[:, None, :], dim=-1)
        lg_ray_terms = grad_psi_n[..., 3:] * a_cd[None, None, :]
        h_ray, lf_ray, lg_ray = _softmin_value_lie(psi_n, lf_ray_terms, lg_ray_terms, filt.kappa_ray)
        input_terms, lf_input_terms, lg_input_terms = filt._input_terms_and_lie(current_command)
        h_input, lf_input, lg_input = _softmin_value_lie(
            input_terms, lf_input_terms, lg_input_terms, filt.kappa_input
        )
        h_aug, lf, lg = _softmin_value_lie(
            torch.stack((h_ray, h_input), dim=1),
            torch.stack((lf_ray, lf_input), dim=1),
            torch.stack((lg_ray, lg_input), dim=1),
            filt.kappa_outer,
        )
    return h_aug, lf, lg


def profile_one(name: str, filt, batch_args, device: torch.device, iters: int) -> None:
    ranges, angles, current_command, nominal_command, previous_nominal_command, nominal_history_valid = batch_args
    probe = torch.tensor([1.3, -0.7, 0.5], device=device, dtype=current_command.dtype)

    cert_times, qp_fwd_times, total_times, bwd_times = [], [], [], []
    last_omega = last_active = last_d = None

    for i in range(iters):
        # --- (1) certificate construction, isolated ---
        sync(device)
        t0 = time.perf_counter()
        h_aug, lf, lg = _compute_certificate(filt, ranges, angles, current_command)
        sync(device)
        t1 = time.perf_counter()
        cert_times.append(t1 - t0)

        # --- (2) qpth forward solve, isolated (reusing the certificate
        # just computed, so this measures ONLY projector.project) ---
        A = -lg[:, None, :]
        b = (lf + filt.alpha_outer * h_aug)[:, None]
        weights = torch.ones(3, device=device, dtype=current_command.dtype)
        sync(device)
        t2 = time.perf_counter()
        with torch.no_grad():
            _ = filt.projector.project(current_command, A, b, weights)
        sync(device)
        t3 = time.perf_counter()
        qp_fwd_times.append(t3 - t2)

        # --- (3) total filter() time, ground truth (full real call) ---
        nominal = nominal_command.clone().requires_grad_(True)
        sync(device)
        t4 = time.perf_counter()
        result = filt.filter(
            ranges, angles, current_command, nominal, previous_nominal_command, nominal_history_valid
        )
        sync(device)
        t5 = time.perf_counter()
        total_times.append(t5 - t4)

        # --- (4) backward (actor path), on a fresh forward pass ---
        nominal_b = nominal_command.clone().requires_grad_(True)
        result_b = filt.filter(
            ranges, angles, current_command, nominal_b, previous_nominal_command, nominal_history_valid
        )
        loss = (result_b.command * probe[None, :]).sum()
        sync(device)
        t6 = time.perf_counter()
        loss.backward()
        sync(device)
        t7 = time.perf_counter()
        bwd_times.append(t7 - t6)

        last_omega = result.cbf_value_nominal.detach()
        last_active = (last_omega < 0).float().mean().item()
        # D = ||Lg h||^2 is not directly returned by LaserSafetyFilterResult;
        # recovered here from the same public fields every *CfMatched class
        # returns: lambda_value = -omega/denom when active & nondegenerate,
        # so denom = -omega/lambda_value wherever lambda_value != 0. Rows
        # with lambda_value == 0 (inactive or degenerate) are excluded from
        # this reconstruction -- that's fine for the purposes of this
        # diagnostic (near-degenerate frequency is what we care about).
        lam = result.lambda_value.detach()
        nonzero = lam.abs() > 1.0e-12
        last_d = torch.where(nonzero, (-last_omega / lam.clamp_min(1e-12)).abs(), torch.full_like(lam, float("nan")))

    def stats(xs):
        xs = torch.tensor(xs[max(1, len(xs) // 5):])  # drop first ~20% as warmup
        return xs.mean().item() * 1000, xs.std().item() * 1000

    cert_m, cert_s = stats(cert_times)
    qp_fwd_m, qp_fwd_s = stats(qp_fwd_times)
    total_m, total_s = stats(total_times)
    bwd_m, bwd_s = stats(bwd_times)

    d_valid = last_d[~torch.isnan(last_d)]
    print(f"\n=== {name} ===")
    print(f"  certificate construction:  {cert_m:7.3f} ± {cert_s:6.3f} ms")
    print(f"  qpth forward solve:        {qp_fwd_m:7.3f} ± {qp_fwd_s:6.3f} ms")
    print(f"  total filter() (ground truth, cert+solve+overhead): {total_m:7.3f} ± {total_s:6.3f} ms")
    print(f"  backward (actor path):     {bwd_m:7.3f} ± {bwd_s:6.3f} ms")
    print(f"  [sanity: cert+qp_fwd = {cert_m + qp_fwd_m:7.3f} ms, should be close to total above]")
    print(f"  Omega: mean={last_omega.mean().item():+.4f}  min={last_omega.min().item():+.4f}  max={last_omega.max().item():+.4f}")
    print(f"  active fraction (Omega<0): {last_active:.4f}")
    if d_valid.numel() > 0:
        print(
            f"  D=||Lg h||^2 (n={d_valid.numel()}): "
            f"min={d_valid.min().item():.4e}  p1={percentile(d_valid, 0.01):.4e}  "
            f"median={percentile(d_valid, 0.5):.4e}"
        )
    else:
        print("  D=||Lg h||^2: no active+nondegenerate rows this batch (all inactive)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Fast correctness-only check: batch=8, iters=2, does not require the GPU to be free.",
    )
    args = parser.parse_args()

    batch = 8 if args.smoke else args.batch
    iters = 2 if args.smoke else args.iters
    device = torch.device("cpu" if args.smoke else args.device)
    dtype = torch.float32

    batch_args = make_frozen_batch(batch, device, dtype, args.seed)

    # Plain Python classes, not nn.Module -- no persisted device-resident
    # state, so device is entirely determined by the input tensors passed
    # to filter() (constructed on `device` above). No .to(device) needed.
    cbf = LaserScanCbfLseControlDynamicsQpCfMatched(**COMMON_KWARGS)
    bcbf = LaserZeroBackupLseControlDynamicsQpCfMatched(**COMMON_KWARGS, **BACKUP_KWARGS)

    print(f"device={device}  batch={batch}  iters={iters}  smoke={args.smoke}")
    profile_one("cbf_lse_cd_qp_cfmatched", cbf, batch_args, device, iters)
    profile_one("bcbf_lse_cd_qp_cfmatched", bcbf, batch_args, device, iters)


if __name__ == "__main__":
    main()
