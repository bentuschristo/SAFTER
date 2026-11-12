"""CBF-RL: PPO high-level navigation with dual training-time CBF safety filtering + reward shaping.

Reimplements Yang et al., "CBF-RL: Safety Filtering Reinforcement Learning in
Training with Control Barrier Functions" (Papers_Ref/), using this repo's own
elliptical LiDAR CBF (src.high_level_rl.laser_cbf_filters) instead of the
paper's single-integrator barrier, and rsl_rl PPO instead of a differentiable
SAC actor.
"""
