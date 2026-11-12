"""Single-panel trajectory figure for the circle_and_square (High-Shift/Hard)
fixed-scenario repeated-trial experiment: ALL 5 methods run on the EXACT
SAME start/goal/yaw (unlike the earlier combined 3-map figure, where
SEA-Nav/CBF-RL used their own independently-drawn scenarios), filter
ACTIVE, repeated ~10 times per method until a repeat was found where
SAFTER succeeds and every other method fails. See the 2026-09-22 session
for the full methodology:

  - vanilla / CBF-CF / SAFTER: scripts/laser_unseen_eval/
    evaluate_laser_unseen_maps_structured.py, 10 repeats (case_id 0-9,
    reset_seed 90000-90009) on a manually-pinned single-row manifest
    (logs/evaluation/circle_and_square_fixed_scenario_repeated/
    {vanilla_cbf_cd_cf,bcbf_cf}/). Winning trial: case_id=0 (reset_seed
    90000) -- vanilla=collision, CBF-CF=collision, SAFTER=success.
  - CBF-RL: scripts/eval_cbf_rl_fixed_scenario_repeated.py, 10 repeats via
    env.set_fixed_scenarios() (policy samples stochastically at inference,
    so no reseed needed for diversity). No genuine obstacle collision
    appeared in those 10 repeats (6 success, 2 timeout, 1 fall) -- a
    timeout case was used at first as a stand-in failure mode.

    2026-09-23: replaced with a genuine collision. Re-ran 20 repeats with
    --perturb-radius-m 0.4 (each repeat's start jittered within a 0.4 m
    disk of the shared start, clipped to the start region) since the exact
    shared start alone never produced a collision even across many tries.
    case_id=7 (collision, 733 steps, start (-0.62, 6.58)) -- ends 5.50 m
    from goal, well into the obstacle field, unlike every timeout case in
    that batch (all of which died within 0.3-1.3 m of goal). Output:
    logs/evaluation/circle_and_square_fixed_scenario_repeated/cbf_rl_perturbed/.
    Its start is ~0.39 m from the other 4 methods' exact shared start, so
    the "Start" marker below is drawn as a filled disk wide enough to
    visibly contain both, rather than a single point.
  - SEA-Nav: scripts/generate_seanav_circle_and_square_fixed_scenario_bank.py
    + evaluate_native_maps_with_trajectories.py, 30 repeats (same map/
    scenario written 3x under different difficulty labels, a workaround for
    SeaNavMapBank's fixed 3-row grid). Winning trial: difficulty=medium,
    env_id=4, episode_idx=0 -- genuine collision, steps=217 (start
    coordinates present, not the nan-start "other_failure" glitch rows).

2026-09-23: removed the SAFTER backup-rollout fan and its zoomed inset --
the real coast displacement is only ~10-15cm, too small to read even
magnified, so it wasn't earning its space. Just the 5 trajectories now.
"""
import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

FS_TITLE = 30
FS_LABEL = 32
FS_TICK = 26
FS_LEGEND = 26

OBSTACLES = [
    {"x": -1.6, "y": 3.4, "r": 0.35, "shape": "circle"},
    {"x": 1.6, "y": 3.4, "r": 0.35, "shape": "circle"},
    {"x": 0.0, "y": 1.9, "r": 1.0, "shape": "circle"},
    {"x": -1.5, "y": -0.2, "r": 0.35, "shape": "circle"},
    {"x": 1.5, "y": -0.2, "r": 0.35, "shape": "circle"},
    {"x": -4.6, "y": -2.3, "r": 0.65, "shape": "circle"},
    {"x": -1.0, "y": -2.6, "r": 0.4, "shape": "square"},
    {"x": 1.0, "y": -2.6, "r": 0.4, "shape": "square"},
    {"x": 4.6, "y": -2.3, "r": 0.65, "shape": "circle"},
    {"x": -3.6, "y": 0.4, "r": 0.7, "shape": "square"},
    {"x": 3.6, "y": 0.4, "r": 0.7, "shape": "square"},
]
START_REGION = (-1.0, 1.0, 5.0, 6.8)
GOAL_REGION = (-5.5, 5.5, -7.0, -4.0)

# Shared by all 5 methods -- same fixed scenario.
START_XY = (-0.948674738407135, 6.362679481506348)
GOAL_XY = (-0.3806767463684082, -5.078561305999756)
GOAL_TOLERANCE_M = 0.35  # HighLevelActionConfig.goal_tolerance
ROBOT_COLLISION_RADIUS_M = 0.30  # collision_circle_radius_m used across all eval configs

COLORS = {
    "Vanilla": "#1f77b4", "CBF-CF": "#ff7f0e", "SAFTER": "#2ca02c",
    "SEA-Nav": "#9467bd", "CBF-RL": "#d62728",
}

REPEATED_DIR = "logs/evaluation/circle_and_square_fixed_scenario_repeated"


def load_sac(method_key, subdir, case_id):
    path = f"{REPEATED_DIR}/{subdir}/seed_5000/circle_and_square/{method_key}/{method_key}_trajectory.csv"
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if int(r["case_id"]) == case_id]
    xs = [float(rows[0]["pre_x_m"])] + [float(r["post_x_m"]) for r in rows]
    ys = [float(rows[0]["pre_y_m"])] + [float(r["post_y_m"]) for r in rows]
    outcome = rows[-1]["outcome"]
    return xs, ys, outcome, rows


def load_cbf_rl(case_id):
    path = f"{REPEATED_DIR}/cbf_rl_perturbed/cbf_rl_trajectory.csv"
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if int(r["case_id"]) == case_id]
    xs = [float(r["post_x_m"]) for r in rows]
    ys = [float(r["post_y_m"]) for r in rows]
    outcome = rows[-1]["outcome"]
    return xs, ys, outcome


def load_seanav(difficulty, env_id, episode_idx):
    path = f"{REPEATED_DIR}/seanav/trajectory.csv"
    with open(path) as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["difficulty"] == difficulty and int(r["env_id"]) == env_id and int(r["episode_idx"]) == episode_idx
        ]
    xs = [float(r["x"]) for r in rows]
    ys = [float(r["y"]) for r in rows]
    outcome = rows[-1]["outcome"]
    return xs, ys, outcome


def outcome_marker(outcome):
    return {"collision": ("X", 220), "timeout": ("^", 200), "other_failure": ("D", 180), "success": ("*", 480)}.get(outcome, ("o", 140))


def draw_obstacles(target_ax):
    for o in OBSTACLES:
        if o["shape"] == "square":
            s = o["r"]
            target_ax.add_patch(Rectangle((o["x"] - s, o["y"] - s), 2 * s, 2 * s, facecolor="dimgray", edgecolor="black", zorder=2))
        else:
            target_ax.add_patch(Circle((o["x"], o["y"]), o["r"], facecolor="dimgray", edgecolor="black", zorder=2))


fig, ax = plt.subplots(figsize=(15.5, 10.5))

sx0, sx1, sy0, sy1 = START_REGION
gx0, gx1, gy0, gy1 = GOAL_REGION
ax.add_patch(Rectangle((sx0, sy0), sx1 - sx0, sy1 - sy0, facecolor="mediumpurple", alpha=0.35, zorder=1))
ax.add_patch(Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, facecolor="khaki", alpha=0.45, zorder=1))

draw_obstacles(ax)

GOAL_RADIUS_COLOR = "deeppink"

# Exact goal radius (the tolerance a method must reach to count as "success"),
# as a hollow dashed perimeter circle in a contrasting color -- distinct from
# both the coarser sampling GOAL_REGION shading behind it and the black
# start/outcome markers.
ax.add_patch(Circle(GOAL_XY, GOAL_TOLERANCE_M, facecolor="none", edgecolor=GOAL_RADIUS_COLOR,
                     linestyle="-", linewidth=2.4, zorder=4))

TRAJ_LW = 4.5
FOOTPRINT_FILL_ALPHA = 0.55
FOOTPRINT_EDGE_WIDTH = 2.4


def draw_footprint(x, y, color):
    ax.add_patch(Circle((x, y), ROBOT_COLLISION_RADIUS_M, facecolor=to_rgba(color, FOOTPRINT_FILL_ALPHA),
                         edgecolor=color, linewidth=FOOTPRINT_EDGE_WIDTH, zorder=4))


for method_key, subdir, label in [
    ("vanilla", "vanilla_cbf_cd_cf", "Vanilla"),
    ("cbf_cd_cf", "vanilla_cbf_cd_cf", "CBF-CF"),
    ("bcbf_cf", "bcbf_cf", "SAFTER"),
]:
    xs, ys, outcome, rows = load_sac(method_key, subdir, case_id=0)
    ax.plot(xs, ys, color=COLORS[label], linewidth=TRAJ_LW, zorder=3)
    draw_footprint(xs[-1], ys[-1], COLORS[label])
    marker, size = outcome_marker(outcome)
    ax.scatter([xs[-1]], [ys[-1]], marker=marker, s=size, color=COLORS[label], zorder=5, edgecolor="black", linewidth=0.9)

xs, ys, outcome = load_seanav("medium", env_id=4, episode_idx=0)
ax.plot(xs, ys, color=COLORS["SEA-Nav"], linewidth=TRAJ_LW, zorder=3)
draw_footprint(xs[-1], ys[-1], COLORS["SEA-Nav"])
marker, size = outcome_marker(outcome)
ax.scatter([xs[-1]], [ys[-1]], marker=marker, s=size, color=COLORS["SEA-Nav"], zorder=5, edgecolor="black", linewidth=0.9)

xs, ys, outcome = load_cbf_rl(case_id=7)
ax.plot(xs, ys, color=COLORS["CBF-RL"], linewidth=TRAJ_LW, zorder=3)
draw_footprint(xs[-1], ys[-1], COLORS["CBF-RL"])
marker, size = outcome_marker(outcome)
ax.scatter([xs[-1]], [ys[-1]], marker=marker, s=size, color=COLORS["CBF-RL"], zorder=5, edgecolor="black", linewidth=0.9)

# Shared starting point -- drawn as a filled disk (data-unit radius, not a
# fixed-size marker) since CBF-RL's own start is perturbed ~0.39 m away
# from the other 4 methods' exact shared point (see load_cbf_rl). Center it
# on the midpoint between the two actual start points and size the radius
# to be the minimal circle covering both (half their separation) -- both
# points then sit exactly on the boundary, rather than picking one point
# and eyeballing a "comfortably covers it" radius.
cbf_rl_start_xy = (xs[0], ys[0])  # xs/ys still hold CBF-RL's own trajectory here
start_center = (
    0.5 * (START_XY[0] + cbf_rl_start_xy[0]),
    0.5 * (START_XY[1] + cbf_rl_start_xy[1]),
)
start_radius = 0.5 * math.hypot(START_XY[0] - cbf_rl_start_xy[0], START_XY[1] - cbf_rl_start_xy[1])
ax.add_patch(Circle(start_center, start_radius, facecolor="black", edgecolor="none", zorder=6))

ax.set_title("Trajectory Comparison", fontsize=FS_TITLE, fontweight="bold")
ax.set_xlim(-7.5, 7.5)
ax.set_ylim(-8, 8)
ax.set_aspect("equal")
ax.set_xlabel("x [m]", fontsize=FS_LABEL)
ax.set_ylabel("y [m]", fontsize=FS_LABEL)
ax.xaxis.set_major_locator(MultipleLocator(4))
ax.yaxis.set_major_locator(MultipleLocator(4))
ax.tick_params(axis="both", labelsize=FS_TICK)

legend_elems = [
    Rectangle((0, 0), 1, 1, facecolor="mediumpurple", alpha=0.35, label="Initial region"),
    Line2D([0], [0], color=COLORS["Vanilla"], lw=TRAJ_LW, label="Vanilla"),
    Line2D([0], [0], color=COLORS["CBF-CF"], lw=TRAJ_LW, label="CBF-CF"),
    Line2D([0], [0], color=COLORS["SAFTER"], lw=TRAJ_LW, label="SAFTER"),
    Line2D([0], [0], color=COLORS["SEA-Nav"], lw=TRAJ_LW, label="SEA-Nav"),
    Line2D([0], [0], color=COLORS["CBF-RL"], lw=TRAJ_LW, label="CBF-RL"),
    Rectangle((0, 0), 1, 1, facecolor="khaki", alpha=0.45, label="Goal region"),
    Rectangle((0, 0), 1, 1, facecolor="dimgray", label="Obstacle"),
    Line2D([0], [0], marker="o", color="black", lw=0, label="Start", markersize=12),
    Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor=GOAL_RADIUS_COLOR,
           markeredgewidth=2.4, color="none", lw=0, label="Goal radius", markersize=16),
    Line2D([0], [0], marker="o", markerfacecolor=to_rgba("gray", FOOTPRINT_FILL_ALPHA), markeredgecolor="gray",
           markeredgewidth=FOOTPRINT_EDGE_WIDTH, color="none", lw=0, label="Robot footprint", markersize=14),
    Line2D([0], [0], marker="X", color="black", lw=0, label="Collision", markersize=13),
    Line2D([0], [0], marker="^", color="black", lw=0, label="Timeout", markersize=13),
    Line2D([0], [0], marker="*", color="black", lw=0, label="Goal reached", markersize=18),
]

fig.legend(
    handles=legend_elems, loc="center left", ncol=1, fontsize=FS_LEGEND, frameon=False, bbox_to_anchor=(0.01, 0.5),
)
fig.subplots_adjust(left=0.42, right=0.97, top=0.93, bottom=0.08)

OUT_PATH = "logs/evaluation/figures/circle_and_square_fixed_scenario_trajectories.png"
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
