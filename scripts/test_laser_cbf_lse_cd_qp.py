#!/usr/bin/env python3
"""CPU equivalence smoke test for the one-row laser CBF-LSE-CD-QP filter."""

from __future__ import annotations

import torch

from src.high_level_rl.laser_cbf_filters import (
    LaserScanCbfControlDynamicsClosedForm,
    LaserScanCbfLseControlDynamicsQp,
)


class CapturingProjector:
    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped
        self.feasibility_tol = wrapped.feasibility_tol
        self.shape: tuple[int, ...] | None = None

    def project(self, desired, A, b, weights):
        self.shape = tuple(A.shape)
        return self.wrapped.project(desired, A, b, weights)


def main() -> None:
    torch.manual_seed(7)
    batch_size, num_rays = 8, 120
    angles = torch.linspace(-torch.pi, torch.pi, num_rays)
    ranges = 0.38 + 2.5 * torch.rand(batch_size, num_rays)
    current = (torch.rand(batch_size, 3) - 0.5) * torch.tensor((0.5, 0.3, 0.5))
    nominal = ((torch.rand(batch_size, 3) - 0.5) * torch.tensor((1.0, 0.6, 1.0))).requires_grad_()

    safety_filter = LaserScanCbfLseControlDynamicsQp(qp_backend="active_set")
    capture = CapturingProjector(safety_filter.projector)
    safety_filter.projector = capture
    result = safety_filter.filter(
        ranges, angles, current, nominal, nominal.detach().clone(), torch.ones(batch_size, 1)
    )

    assert capture.shape == (batch_size, 1, 3)
    assert result.command.shape == (batch_size, 3)
    assert result.individual_h.shape == (batch_size, num_rays + 6)
    assert torch.isfinite(result.command).all()
    result.command.square().mean().backward()
    assert nominal.grad is not None and torch.isfinite(nominal.grad).all()
    cf = LaserScanCbfControlDynamicsClosedForm()
    cf_result = cf.filter(
        ranges, angles, current, nominal.detach(), nominal.detach().clone(), torch.ones(batch_size, 1)
    )
    max_command_delta = torch.max(torch.abs(result.command.detach() - cf_result.command))
    assert max_command_delta < 1.0e-5, max_command_delta
    print(
        "CBF-LSE-CD-QP smoke passed: one QP row; finite SAC gradient; "
        f"max CF/QP command delta={float(max_command_delta):.3e}"
    )


if __name__ == "__main__":
    main()
