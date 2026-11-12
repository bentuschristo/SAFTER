"""CPU/GPU smoke test for the differentiable zero-backup bCBF layer."""
from __future__ import annotations

import torch

from src.high_level_rl.config import ArenaConfig, BackupSafetyConfig, HighLevelActionConfig
from src.high_level_rl.bcbf_lse_cf_filter import ZeroBackupLseCbfClosedForm


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    arena = ArenaConfig()
    action = HighLevelActionConfig()
    cfg = BackupSafetyConfig()
    filt = ZeroBackupLseCbfClosedForm(arena, action, cfg)

    position = torch.tensor([[-6.0, 0.0], [6.9, 0.0]], device=device)
    yaw = torch.zeros(2, device=device)
    current = torch.zeros(2, 3, device=device)
    nominal = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], device=device, requires_grad=True)
    centers = torch.tensor(
        [[[o.x, o.y] for o in arena.obstacles]], device=device
    ).expand(2, -1, -1)
    radii = torch.tensor(
        [[o.radius for o in arena.obstacles]], device=device
    ).expand(2, -1)
    mask = torch.ones(2, arena.max_obstacles, dtype=torch.bool, device=device)

    previous_nominal = torch.zeros_like(nominal)
    history_valid = torch.zeros(
        nominal.shape[0],
        1,
        device=nominal.device,
        dtype=nominal.dtype,
    )

    out = filt.filter(
        position,
        yaw,
        current,
        nominal,
        previous_nominal,
        history_valid,
        centers,
        radii,
        mask,
    )
    loss = out.command.square().sum() + out.h_lse.sum()
    loss.backward()

    print("device:", device)
    print("safe command:\n", out.command.detach().cpu())
    print("h_aug:", out.h_lse.detach().cpu())
    print("h_backup:", out.h_backup_lse.detach().cpu())
    print("active:", out.active.detach().cpu())
    print(
        "true bCBF correction norm:",
        out.correction_norm.detach().cpu(),
    )
    print("nominal gradient finite:", bool(torch.isfinite(nominal.grad).all()))

    # Test the restored nominal-command derivative term.
    nominal_2 = nominal.detach().clone().requires_grad_(True)

    command_increment = torch.tensor(
        [
            [0.10, -0.05, 0.02],
            [0.10, -0.05, 0.02],
        ],
        device=device,
        dtype=nominal_2.dtype,
    )

    previous_nominal_2 = nominal_2.detach() - command_increment
    history_valid_2 = torch.ones(
        nominal_2.shape[0],
        1,
        device=device,
        dtype=nominal_2.dtype,
    )

    out_2 = filt.filter(
        position,
        yaw,
        current,
        nominal_2,
        previous_nominal_2,
        history_valid_2,
        centers,
        radii,
        mask,
    )

    expected_derivative = (
        nominal_2.detach() - previous_nominal_2
    ) / cfg.integration_dt

    a = nominal_2.new_tensor(cfg.control_dynamics_a)
    sigma = nominal_2.new_tensor(cfg.tracking_sigma)

    expected_desired_surrogate = (
        current
        + expected_derivative / a
        + (sigma / a) * (nominal_2.detach() - current)
    )

    desired_error = (
        out_2.desired_surrogate_command.detach()
        - expected_desired_surrogate
    ).abs().max()

    print("\nvalid-history test")
    print("nominal derivative:\n", expected_derivative.cpu())
    print(
        "desired surrogate:\n",
        out_2.desired_surrogate_command.detach().cpu(),
    )
    print(
        "expected desired surrogate:\n",
        expected_desired_surrogate.cpu(),
    )
    print(
        "desired-surrogate max error:",
        float(desired_error.cpu()),
    )

    assert desired_error < 1.0e-6


if __name__ == "__main__":
    main()
