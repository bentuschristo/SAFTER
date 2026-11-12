"""Closed-form kinematic CBF safety filter for planar obstacle avoidance."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CbfDiagnostics:
  """Diagnostic quantities returned by the CBF filter."""

  h_individual: torch.Tensor
  h_lse: torch.Tensor
  weights: torch.Tensor
  gradient_world: torch.Tensor
  constraint_vector_body: torch.Tensor
  nominal_constraint_value: torch.Tensor
  correction_norm: torch.Tensor
  active: torch.Tensor


def closed_form_lse_cbf(
  robot_xy: torch.Tensor,
  robot_yaw: torch.Tensor,
  nominal_command_b: torch.Tensor,
  obstacle_xy: torch.Tensor,
  obstacle_radii: torch.Tensor,
  *,
  robot_radius: float,
  safety_margin: float,
  alpha: float,
  kappa: float,
  eps: float = 1.0e-8,
) -> tuple[torch.Tensor, CbfDiagnostics]:
  """Apply a closed-form LSE-CBF safety filter.

  Args:
    robot_xy:
      Robot planar position in environment-local/world-aligned coordinates.
      Shape: [num_envs, 2].

    robot_yaw:
      Robot yaw angle in radians.
      Shape: [num_envs] or [num_envs, 1].

    nominal_command_b:
      Nominal body-frame command [vx, vy, wz].
      Shape: [num_envs, 3].

    obstacle_xy:
      Obstacle centers in the same planar coordinate frame as robot_xy.
      Shape: [num_envs, num_obstacles, 2].

    obstacle_radii:
      Physical obstacle radii.
      Shape: [num_envs, num_obstacles].

    robot_radius:
      Circular approximation of the robot footprint.

    safety_margin:
      Additional obstacle-clearance margin.

    alpha:
      Linear class-K CBF gain.

    kappa:
      LSE soft-min sharpness. Must be positive.

    eps:
      Numerical regularization for the closed-form denominator.

  Returns:
    safe_command_b:
      Filtered body-frame command [vx_safe, vy_safe, wz_nominal].
      Shape: [num_envs, 3].

    diagnostics:
      Barrier values, gradients, intervention flags, and related quantities.
  """
  if kappa <= 0.0:
    raise ValueError(f"kappa must be positive, received {kappa}.")

  if alpha <= 0.0:
    raise ValueError(f"alpha must be positive, received {alpha}.")

  if robot_xy.ndim != 2 or robot_xy.shape[-1] != 2:
    raise ValueError(
      f"robot_xy must have shape [num_envs, 2], got {robot_xy.shape}."
    )

  if nominal_command_b.ndim != 2 or nominal_command_b.shape[-1] != 3:
    raise ValueError(
      "nominal_command_b must have shape [num_envs, 3], "
      f"got {nominal_command_b.shape}."
    )

  if obstacle_xy.ndim != 3 or obstacle_xy.shape[-1] != 2:
    raise ValueError(
      "obstacle_xy must have shape [num_envs, num_obstacles, 2], "
      f"got {obstacle_xy.shape}."
    )

  if obstacle_radii.shape != obstacle_xy.shape[:2]:
    raise ValueError(
      "obstacle_radii must match obstacle_xy's first two dimensions. "
      f"Got {obstacle_radii.shape} and {obstacle_xy.shape}."
    )

  robot_yaw = robot_yaw.reshape(-1)

  num_envs = robot_xy.shape[0]

  if robot_yaw.shape[0] != num_envs:
    raise ValueError("robot_yaw and robot_xy must have the same batch size.")

  if nominal_command_b.shape[0] != num_envs:
    raise ValueError(
      "nominal_command_b and robot_xy must have the same batch size."
    )

  if obstacle_xy.shape[0] != num_envs:
    raise ValueError(
      "obstacle_xy and robot_xy must have the same batch size."
    )

  # Vector from each obstacle center to the robot.
  #
  # delta[e, i] = p_robot[e] - p_obstacle[e, i]
  delta = robot_xy[:, None, :] - obstacle_xy

  effective_radii = (
    obstacle_radii
    + float(robot_radius)
    + float(safety_margin)
  )

  # Individual circular obstacle barriers:
  #
  # h_i = ||p - p_i||^2 - R_i^2
  h_individual = (
    torch.sum(delta.square(), dim=-1)
    - effective_radii.square()
  )

  # Conservative unnormalized soft minimum:
  #
  # h_LSE = -(1/kappa) log sum_i exp(-kappa h_i)
  #
  # This satisfies h_LSE <= min_i h_i.
  h_lse = (
    -torch.logsumexp(-kappa * h_individual, dim=1)
    / kappa
  )

  # Soft-min gradient weights.
  weights = torch.softmax(
    -kappa * h_individual,
    dim=1,
  )

  # grad_p h_i = 2 (p - p_i)
  #
  # grad_p h_LSE = sum_i w_i grad_p h_i
  gradient_world = torch.sum(
    weights[..., None] * 2.0 * delta,
    dim=1,
  )

  # The policy command is body-frame:
  #
  # p_dot_world = R(yaw) u_body
  #
  # Therefore:
  #
  # h_dot = grad_h_world^T R(yaw) u_body
  #       = (R(yaw)^T grad_h_world)^T u_body
  #
  # Thus the affine CBF constraint vector in body coordinates is:
  #
  # a = R(yaw)^T grad_h_world
  cos_yaw = torch.cos(robot_yaw)
  sin_yaw = torch.sin(robot_yaw)

  grad_x = gradient_world[:, 0]
  grad_y = gradient_world[:, 1]

  a_body = torch.stack(
    (
      cos_yaw * grad_x + sin_yaw * grad_y,
      -sin_yaw * grad_x + cos_yaw * grad_y,
    ),
    dim=1,
  )

  nominal_xy = nominal_command_b[:, :2]

  # CBF condition:
  #
  # a^T u + alpha h_LSE >= 0
  #
  # or:
  #
  # a^T u >= -alpha h_LSE
  lower_bound = -alpha * h_lse

  nominal_constraint_value = (
    torch.sum(a_body * nominal_xy, dim=1)
    + alpha * h_lse
  )

  # Amount by which the nominal command violates the constraint.
  #
  # violation = max(0, b - a^T u_nom)
  violation = torch.relu(
    lower_bound
    - torch.sum(a_body * nominal_xy, dim=1)
  )

  denominator = torch.sum(
    a_body.square(),
    dim=1,
  ).clamp_min(eps)

  # Closed-form Euclidean projection onto a single half-space:
  #
  # u_safe = u_nom
  #        + max(0, b - a^T u_nom)/(a^T a) * a
  multiplier = violation / denominator

  correction = multiplier[:, None] * a_body

  safe_xy = nominal_xy + correction

  # This first-order position CBF does not directly constrain yaw rate.
  safe_command_b = nominal_command_b.clone()
  safe_command_b[:, :2] = safe_xy

  diagnostics = CbfDiagnostics(
    h_individual=h_individual,
    h_lse=h_lse,
    weights=weights,
    gradient_world=gradient_world,
    constraint_vector_body=a_body,
    nominal_constraint_value=nominal_constraint_value,
    correction_norm=torch.linalg.vector_norm(correction, dim=1),
    active=violation > 0.0,
  )

  return safe_command_b, diagnostics
