# Procedural Go2 Navigation Map Generator

Place the script at:

```text
scripts/procedural_maps/generate_procedural_navigation_maps.py
```

Run from the repository root:

```bash
cd ~/unitree_rl_mjlab
conda activate unitree_rl_mjlab

python scripts/procedural_maps/generate_procedural_navigation_maps.py \
  --output-dir generated/procedural_maps/train_100_seed42 \
  --num-maps 100 \
  --scenarios-per-map 64 \
  --seed 42
```

The default bank contains:

- 100 maps;
- 0–12 active circular obstacles;
- obstacle radii sampled from 0.40–0.60 m;
- separated, clustered, and mixed layouts;
- footprint-inflated path-connectivity checks;
- 64 valid start/goal/yaw scenarios per map;
- 30/40/30 easy/medium/hard map and scenario labels;
- a 70/30 mixture of directly blocked and clear start-goal pairs where possible.

Important output files:

```text
generated/procedural_maps/train_100_seed42/
├── generation_config.json
├── bank.json
├── map_summary.csv
├── scenarios.csv
├── maps/
├── arenas/
├── occupancy/
└── previews/
```

The `arenas/map_XXX_arena.json` files are compatible with
`src.high_level_rl.arena.arena_from_json`. The richer `maps/map_XXX.json`
files include fixed 12-slot obstacle arrays and difficulty metadata for the
future training loader.

For a quick smoke test:

```bash
python scripts/procedural_maps/generate_procedural_navigation_maps.py \
  --output-dir generated/procedural_maps/smoke_seed7 \
  --num-maps 6 \
  --scenarios-per-map 8 \
  --seed 7 \
  --no-previews
```

The script only generates and saves the bank. Loading maps dynamically into
MuJoCo and applying the easy/medium/hard curriculum will be implemented in the
training environment next.
