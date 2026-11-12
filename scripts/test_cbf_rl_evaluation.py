"""CPU regression checks for CBF-RL termination and evaluation bookkeeping.

Run: python scripts/test_cbf_rl_evaluation.py
Physics integration is mocked; geometry bounds are checked against MuJoCo.
"""
from __future__ import annotations

import sys
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import torch

import evaluate_high_level_cbf_rl_training_maps as training_eval
from evaluate_high_level_cbf_rl_unseen_maps import evaluate_map
from evaluate_high_level_cbf_rl_unseen_maps_path_length import evaluate_map_with_path_length
from src.high_level_cbf_rl.cbf_rl_navigation_env import CbfRlNavigationEnv
from src.high_level_cbf_rl.config import CbfRlConfig, CbfRlModeConfig
from src.high_level_rl.config import ArenaConfig, CircularObstacle


def terminal_step():
    cfg = CbfRlConfig()
    n = 6
    zeros = torch.zeros(n)
    pos = torch.zeros(n, 3)
    pos[:, 2] = 0.3
    pos[1:3, 2] = 0.0  # Fallen at the goal; slot 2 also collides.
    goals = torch.zeros(n, 2)
    goals[3:5, 0] = 5.0
    centers = torch.full((n, 1, 2), 100.0)
    centers[2] = 0.0
    safety = NS(command=torch.zeros(n, 3), active=zeros.bool(),
                correction_norm=zeros, cbf_value_nominal=zeros, h_lse=zeros)
    env = NS(
        navigation_cfg=cfg.navigation, laser_safety_cfg=cfg.safety,
        cbf_reward_cfg=cfg.reward, mode_cfg=CbfRlModeConfig(False, False),
        _maybe_refresh_laser_cache=lambda: None, _scale_action=lambda x: x,
        laser_ranges=lambda: torch.ones(n, 1), ray_angles=zeros[:1],
        cbf_filter=NS(filter=lambda *args: safety),
        _action_low=torch.full((3,), -1.0), _action_high=torch.ones(3),
        previous_safe_command=torch.zeros(n, 3),
        previous_goal_distance=torch.zeros(n),
        nominal_history_valid=torch.zeros(n, 1),
        low_env=NS(get_observations=lambda: None, step=lambda action: None),
        low_policy=lambda *args: None,
        _root_state=lambda: (pos, None, None, None, zeros, zeros, zeros),
        goal_positions=goals, obstacle_centers=centers,
        obstacle_radii=torch.full((n, 1), 0.5),
        obstacle_mask=torch.ones(n, 1, dtype=torch.bool), _any_square_slot=False,
        episode_length_buf=torch.tensor([0, 0, 0, cfg.navigation.max_episode_steps - 1,
                                         0, cfg.navigation.max_episode_steps - 1]),
        _episode_outcome_rolling={k: deque(maxlen=100) for k in
                                  ("success", "collision", "timeout", "fall")},
        _compute_observation=lambda: None,
    )

    def reset(ids):
        # Emulate simulator buffers being overwritten during autoreset.
        pos[ids, :2] = 50.0

    env._reset_selected = reset
    return CbfRlNavigationEnv.step(env, torch.zeros(n, 3))


class FakeEvalEnv:
    num_envs = 3
    device = "cpu"
    navigation_cfg = NS(normalize_positions_by=1.0)

    def set_fixed_scenarios(self, scenarios):
        assert len(scenarios) == self.num_envs
        self.t = 0

    def get_observations(self):
        return torch.tensor([[10.0, 0.0]] * self.num_envs)

    actor_tensor = staticmethod(lambda obs: obs)

    def step(self, action):
        self.t += 1
        done = torch.arange(1, self.num_envs + 1) <= self.t
        distance = torch.where(done, 0.1, 10.0 - self.t)
        pos = torch.stack((10.0 - distance, torch.zeros(self.num_envs)), dim=-1)
        extras = {
            "distance_to_goal": distance, "position": pos,
            "completed_env_ids": done.nonzero().flatten(),
            "completed_success": done,
            **{k: torch.zeros_like(done) for k in
               ("completed_collision", "completed_timeouts", "completed_fallen")},
        }
        # Reset observations deliberately differ from terminal observations.
        return self.get_observations(), torch.zeros(self.num_envs), done, extras


def manifest(n):
    return [dict(map_name="test", case_id=i, reset_seed=1000 + i,
                 start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=10.0, goal_y=0.0)
            for i in range(n)]


class CbfRlEvaluationTests(unittest.TestCase):
    def test_outcomes_are_exclusive_and_failure_beats_goal(self):
        _, _, done, extras = terminal_step()
        expected = {
            "completed_success": [1, 0, 0, 0, 0, 1],
            "completed_collision": [0, 0, 1, 0, 0, 0],
            "completed_fallen": [0, 1, 0, 0, 0, 0],
            "completed_timeouts": [0, 0, 0, 1, 0, 0],
        }
        for name, values in expected.items():
            self.assertEqual(extras[name].int().tolist(), values)
        self.assertEqual(sum(extras[k].int() for k in expected).tolist(), done.int().tolist())

    def test_terminal_position_survives_autoreset(self):
        _, _, _, extras = terminal_step()
        torch.testing.assert_close(extras["position"], torch.zeros(6, 2))
        torch.testing.assert_close(extras["distance_to_goal"], torch.tensor([0., 0., 0., 5., 5., 0.]))

    def test_evaluators_score_once_and_use_terminal_distance(self):
        for evaluator in (evaluate_map, evaluate_map_with_path_length):
            for count in (1, 4):  # Fewer cases than envs and a partial last batch.
                for stochastic in (False, True):
                    with self.subTest(evaluator=evaluator.__name__, count=count, stochastic=stochastic):
                        calls = []

                        def policy(obs, stochastic_output=False):
                            calls.append(stochastic_output)
                            return torch.zeros(3, 3)

                        result = evaluator(env=FakeEvalEnv(), policy=policy, manifest=manifest(count),
                                           num_envs=50, print_every=0, stochastic_policy=stochastic)
                        records = result[0] if isinstance(result, tuple) else result
                        self.assertEqual(sorted(r["case_id"] for r in records), list(range(count)))
                        self.assertEqual(sum(r["success"] for r in records), count)
                        self.assertTrue(all(c == stochastic for c in calls))
                        for row in records:
                            self.assertAlmostEqual(row["final_distance_to_goal_m"], 0.1, places=5)
                            self.assertAlmostEqual(row["min_distance_to_goal_m"], 0.1, places=5)
                            if "path_length_m" in row:
                                self.assertAlmostEqual(row["path_length_m"], 9.9, places=5)

    def test_empty_manifest(self):
        kwargs = dict(env=FakeEvalEnv(), policy=None, manifest=[], num_envs=50, print_every=0)
        self.assertEqual(evaluate_map(**kwargs), [])
        self.assertEqual(evaluate_map_with_path_length(**kwargs), ([], []))

    def test_fixed_map_bounds_match_compiled_mujoco_geometry(self):
        obstacles = (CircularObstacle(1., 2., 0.6, 1.4),
                     CircularObstacle(-1., -2., 0.4, 0.8, shape="square"))
        arena = ArenaConfig(obstacles=obstacles, max_obstacles=2)
        model = NS(geom_size=torch.zeros(2, 2, 3), geom_rbound=torch.zeros(2, 2),
                   geom_aabb=torch.zeros(2, 2, 2, 3))
        scene = {f"procedural_obstacle_slot_{i:02d}": NS(
            indexing=NS(geom_ids=[i]), write_mocap_pose_to_sim=Mock()) for i in range(2)}
        raw = NS(sim=NS(model=model, expand_model_fields=Mock(), sense=Mock()), scene=scene)
        env_cfg = NS(scene=NS(num_envs=2, entities={}), events={}, terminations={})
        cfg = training_eval.EvalTrainingMapsConfig(checkpoint="unused", low_level_checkpoint="unused", num_envs=2)
        with patch.multiple(training_eval,
                            load_env_cfg=Mock(return_value=env_cfg),
                            load_rl_cfg=Mock(return_value=NS(clip_actions=None)),
                            add_navigation_laser_sensor=Mock(),
                            ManagerBasedRlEnv=Mock(return_value=raw),
                            RslRlVecEnvWrapper=Mock(return_value=raw),
                            load_frozen_low_level_policy=Mock(),
                            MjlabVelocityCommandInterface=Mock(),
                            CbfRlNavigationEnv=Mock()):
            training_eval.build_training_map_env("unused", NS(arena=arena), cfg, "cpu")
            training_eval.load_env_cfg.assert_called_once_with("unused", play=False)
            method_cfg = training_eval.CbfRlNavigationEnv.call_args.args[3]
            self.assertEqual(method_cfg.navigation.episode_duration_s, 90.0)
        oracle = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom type="cylinder" size="0.6 0.7"/>
          <geom type="box" size="0.4 0.4 0.4"/>
        </worldbody></mujoco>''')
        for field in ("geom_rbound", "geom_aabb"):
            expected = torch.tensor(getattr(oracle, field), dtype=torch.float32)
            if field == "geom_aabb":
                expected = expected.reshape(2, 2, 3)
            torch.testing.assert_close(getattr(model, field), expected.unsqueeze(0).expand_as(getattr(model, field)))


if __name__ == "__main__":
    unittest.main()
