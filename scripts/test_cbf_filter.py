"""Basic numerical test for the closed-form LSE-CBF."""

import torch

from src.tasks.velocity.mdp.cbf_filter import closed_form_lse_cbf


def main() -> None:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # One environment.
  robot_xy = torch.tensor(
    [[0.0, 0.0]],
    dtype=torch.float32,
    device=device,
  )

  robot_yaw = torch.tensor(
    [0.0],
    dtype=torch.float32,
    device=device,
  )

  # Nominally drive directly toward the obstacle.
  nominal_command_b = torch.tensor(
    [[0.8, 0.0, 0.0]],
    dtype=torch.float32,
    device=device,
  )

  obstacle_xy = torch.tensor(
    [[[1.0, 0.0]]],
    dtype=torch.float32,
    device=device,
  )

  obstacle_radii = torch.tensor(
    [[0.30]],
    dtype=torch.float32,
    device=device,
  )

  safe_command, diagnostics = closed_form_lse_cbf(
    robot_xy=robot_xy,
    robot_yaw=robot_yaw,
    nominal_command_b=nominal_command_b,
    obstacle_xy=obstacle_xy,
    obstacle_radii=obstacle_radii,
    robot_radius=0.30,
    safety_margin=0.10,
    alpha=1.0,
    kappa=20.0,
  )

  print("Robot position:", robot_xy)
  print("Nominal command:", nominal_command_b)
  print("Safe command:", safe_command)
  print("Individual h:", diagnostics.h_individual)
  print("LSE h:", diagnostics.h_lse)
  print("CBF vector a:", diagnostics.constraint_vector_body)
  print(
    "Nominal CBF value:",
    diagnostics.nominal_constraint_value,
  )
  print("Correction norm:", diagnostics.correction_norm)
  print("Filter active:", diagnostics.active)

  # Verify the filtered command satisfies the CBF inequality.
  safe_xy = safe_command[:, :2]

  safe_constraint_value = (
    torch.sum(
      diagnostics.constraint_vector_body * safe_xy,
      dim=1,
    )
    + diagnostics.h_lse
  )

  print("Safe CBF value:", safe_constraint_value)

  assert torch.all(safe_constraint_value >= -1.0e-5), (
    "Filtered command violates the CBF constraint."
  )


if __name__ == "__main__":
  main()
