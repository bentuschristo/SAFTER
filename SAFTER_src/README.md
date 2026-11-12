# SAFTER_src / SAFTER_scripts

Source and scripts for the **laser-based** safe-navigation pipeline only:
**Vanilla SAC, CBF-CF, CBF-QP, bCBF-CF ("SAFTER"), bCBF-QP** — trained,
filtered, evaluated, and ablated via LiDAR observations. Split out of
`src/`/`scripts/` on 2026-09-22 so this active pipeline stops sitting next
to (and getting confused with):

- the **original unitree_rl_mjlab** low-level locomotion policy training code
  (`src/tasks/`, `logs/rsl_rl/go2_velocity/...`) — untouched, still under `src/`.
- the **obsolete/unused map-based** high-level pipelines (`src/high_level_sac/`,
  `src/high_level_bcbf_sac/`, `src/high_level_cbf_rl/`,
  `src/high_level_rl/bcbf_lse_cf_filter.py`,
  `src/high_level_rl/cbf_cd_lse_cf_filter.py`) — also untouched, still under
  `src/`. These use a known obstacle map, not LiDAR, and are not part of the
  current SAFTER results.

## Layout

```
SAFTER_src/
  high_level_laser_sac/   # config, networks (actor/critic), env, trainer,
                           # replay buffer, procedural map bank -- MOVED
                           # wholesale from src/high_level_laser_sac/.
  high_level_rl/           # filter math + shared building blocks
  high_level_control/      # command_interface.py (mjlab velocity command glue)

SAFTER_scripts/
  train_SAC_policies.py
  play_high_level_sac_laser_four_methods.py
  laser_unseen_eval/evaluate_laser_unseen_maps_structured.py   # main eval harness
  evaluate_training_maps_no_filter.py
  evaluate_ablation_filter_swap.py
```

The internal `run_*.sh` launcher wrappers (one per method/seed-range/ablation
combo, for our own reproducibility, not meant for public consumption as-is)
are kept out of this folder. A worked example training command for
`train_SAC_policies.py` will be documented here directly.

## Why some `SAFTER_src/high_level_rl/*.py` files are DUPLICATES, not moves

A handful of files turned out to be genuinely load-bearing for BOTH the
laser pipeline and the excluded map-based pipelines (map-based scripts import
them directly, or — in `laser_cbf_filters.py`'s case — the map-based CBF-RL
package's own env class imports a laser filter class for one component).
Moving them would have broken the map-based side, so instead they are
duplicated: the original stays under `src/high_level_rl/` (and
`src/high_level_control/`) for map-based code to keep using, and a copy
lives here for SAFTER_src. Each duplicated file has a `NOTE:` comment at the
top saying so. **If you change one, check whether the other copy needs the
same change**:

- `config.py`, `arena.py`, `lse_safety_filter.py`, `navigation_env.py` —
  `LaserSacNavigationEnv` subclasses the map-based `HighLevelNavigationEnv`.
- `math_utils.py`, `low_level.py`, `raycast_patterns.py` — plain utility
  modules used everywhere.
- `laser_cbf_filters.py` — also imported directly by
  `src/high_level_cbf_rl/cbf_rl_navigation_env.py` (`LaserScanCbfClosedForm`,
  used as one component of CBF-RL's own env, not inherited).
- `zero_backup_flow.py` — also imported by the map-based
  `src/high_level_rl/bcbf_lse_cf_filter.py`.
- `bcbf_qp_filter.py` — the SAFTER_src copy is **trimmed**: only
  `QpSolveDiagnostics`/`SmallBoxQpProjector`/`OsqpProjector`/`QpthProjector`
  (the classes `laser_cbf_filters.py` actually lazy-imports) are kept. The
  original's `ZeroBackupBcbfQpFilter` class is map-based-only (subclasses
  the map-based `ZeroBackupLseCbfClosedForm`) and was dropped here.
- `evaluation_scenarios.py` — copied from (was previously imported
  cross-package from) `src/high_level_sac/evaluation_scenarios.py`.

## The other direction: map-based code that depends on SAFTER_src

`src/high_level_cbf_rl/` (`CbfRlNavigationEnv`) genuinely **subclasses**
`SAFTER_src.high_level_laser_sac.sac_navigation_env.LaserSacNavigationEnv`
and reuses a few of its config dataclasses (`LaserRewardConfig`,
`ContactCollisionConfig`, `LaserObservationConfig`, `LaserSafetyConfig`) and
`procedural_maps.py` directly, rather than duplicating an 85KB,
actively-changing environment class. So `src/high_level_cbf_rl/` (and its
`scripts/*cbf_rl*.py` entrypoints) import straight from `SAFTER_src` for
those — this is intentional, not leftover coupling to clean up.

## Not moved (kept in `scripts/`, path-patched to reach into SAFTER_src)

- `scripts/play_high_level_laser_keyboard.py` — a manual keyboard-teleop demo
  built on the separate, generic `src/high_level_control/` feedback-
  linearization framework, not the SAC-laser training pipeline. Still uses
  `SAFTER_src.high_level_rl.raycast_patterns` for one thing, so it wasn't a
  clean exclude, but it isn't "training/filter/evaluation/ablation" either.
- `scripts/play_bcbf_cf_training_map_viewer.py`,
  `scripts/play_bcbf_cf_slalom_viewer.py` — viewer tools that dynamically
  load `SAFTER_scripts/*.py` by file path (`importlib.util.spec_from_file_location`)
  to reuse `build_environment`/`load_actor`/etc. Path strings updated to
  point at the new `SAFTER_scripts/` locations.

## Explicitly excluded (obsolete/map-based, untouched)

`src/high_level_sac/`, `src/high_level_bcbf_sac/`, `src/high_level_cbf_rl/`
(except the two cross-imports noted above), `src/high_level_rl/bcbf_lse_cf_filter.py`,
`src/high_level_rl/cbf_cd_lse_cf_filter.py`, and every `scripts/*bcbf_sac*`,
`scripts/*cbf_rl*`, `scripts/train_high_level_rl.py`,
`scripts/play_high_level_rl.py` entrypoint. Also `scripts/test_bcbf_filter.py`
and `scripts/test_cf_matched_qp_regression.py` despite similar-sounding
names — both test the map-based filter, not the laser one.

The many other `scripts/*.sh`/`*.py` files not mentioned above (dozens of
one-off exploratory eval variants, e.g.
`laser_unseen_maps_structured_handcrafted2_*` sweeps) were left in place —
they're superseded dev iterations, not part of the current pipeline.
