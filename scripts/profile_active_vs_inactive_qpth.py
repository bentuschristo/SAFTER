"""Isolates "constraint active vs. inactive" as the sole variable, using the
SAME certificate class (cbf_lse_cd_qp_cfmatched only) on two hand-constructed
batches:

  * "easy"   -- robot far from all obstacles, mild nominal command -> h_aug
                comfortably positive, constraint essentially always inactive.
  * "hard"   -- robot placed just inside the safety margin of an obstacle,
                nominal command driving straight at it -> h_aug negative,
                constraint essentially always active (needs correction).

Does NOT touch the CBF-vs-bCBF comparison at all -- both batches use the
exact same LaserScanCbfLseControlDynamicsQpCfMatched instance, so any timing
difference here is purely about active vs. inactive, isolating the specific
claim: does qpth's QPFunction genuinely cost more when the true optimum
lands on the constraint boundary (active) vs. safely inside it (inactive)?

Runs on CPU by default so it never touches a GPU that may be busy with a
live training job.

Run: python3 scripts/profile_active_vs_inactive_qpth.py
"""
from __future__ import annotations

import time

import torch

from src.high_level_rl.laser_cbf_filters import LaserScanCbfLseControlDynamicsQpCfMatched

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
NUM_RAYS = 120
BATCH = 256
ITERS = 30


def make_easy_batch(device, dtype):
    """All rays return near max_range (open space); mild nominal motion."""
    ranges = torch.full((BATCH, NUM_RAYS), 4.5, device=device, dtype=dtype)
    angles = torch.linspace(-3.14159265, 3.14159265, NUM_RAYS, dtype=dtype, device=device)
    current_command = torch.tensor([[0.2, 0.0, 0.0]] * BATCH, device=device, dtype=dtype)
    nominal_command = current_command.clone()
    previous_nominal_command = current_command.clone()
    nominal_history_valid = torch.ones(BATCH, 1, device=device, dtype=dtype)
    return ranges, angles, current_command, nominal_command, previous_nominal_command, nominal_history_valid


def make_hard_batch(device, dtype):
    """A ray dead ahead returns 0.02m -- deep inside the inflated ellipse, so
    h_ray is strongly negative. Uses a MODERATE forward command (vx=0.3, not
    max_vx=2.0): at vx=2.0 the command-decay drift term (a=8, ~125ms decay
    constant) alone was recovering the barrier even from a deep violation,
    making omega spuriously positive -- confirmed by direct calculation
    before fixing this. vx=0.3 leaves the drift recovery term too small to
    overcome h_aug's penalty, giving a genuinely active (omega<0) case."""
    ranges = torch.full((BATCH, NUM_RAYS), 4.5, device=device, dtype=dtype)
    ranges[:, NUM_RAYS // 2] = 0.02  # ray pointing straight ahead (angle=0)
    angles = torch.linspace(-3.14159265, 3.14159265, NUM_RAYS, dtype=dtype, device=device)
    current_command = torch.tensor([[0.3, 0.0, 0.0]] * BATCH, device=device, dtype=dtype)
    nominal_command = current_command.clone()
    previous_nominal_command = current_command.clone()
    nominal_history_valid = torch.ones(BATCH, 1, device=device, dtype=dtype)
    return ranges, angles, current_command, nominal_command, previous_nominal_command, nominal_history_valid


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(name, filt, batch_args, device, iters):
    ranges, angles, current_command, nominal_command, previous_nominal_command, nominal_history_valid = batch_args
    probe = torch.tensor([1.3, -0.7, 0.5], device=device, dtype=current_command.dtype)
    fwd_times = []
    omega = active = None
    for i in range(iters):
        nominal = nominal_command.clone().requires_grad_(True)
        sync(device)
        t0 = time.perf_counter()
        result = filt.filter(
            ranges, angles, current_command, nominal, previous_nominal_command, nominal_history_valid
        )
        sync(device)
        t1 = time.perf_counter()
        fwd_times.append(t1 - t0)
        omega = result.cbf_value_nominal.detach()
        active = (omega < 0).float().mean().item()
    xs = torch.tensor(fwd_times[max(1, len(fwd_times) // 5):])
    print(
        f"{name:8s}  forward filter(): {xs.mean().item()*1000:7.3f} +/- {xs.std().item()*1000:6.3f} ms   "
        f"active_fraction={active:.3f}  Omega mean={omega.mean().item():+.3f} min={omega.min().item():+.3f}"
    )


def main():
    device = torch.device("cpu")
    dtype = torch.float32
    filt = LaserScanCbfLseControlDynamicsQpCfMatched(**COMMON_KWARGS)

    print(f"device={device}  batch={BATCH}  iters={ITERS}  (isolating active vs inactive, same certificate class)")
    run("easy", filt, make_easy_batch(device, dtype), device, ITERS)
    run("hard", filt, make_hard_batch(device, dtype), device, ITERS)


if __name__ == "__main__":
    main()
