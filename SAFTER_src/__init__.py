"""SAFTER (laser-based safe navigation: Vanilla / CBF-CF / CBF-QP / bCBF-CF / bCBF-QP) source package.

Kept separate from src/ (which trains the frozen low-level locomotion
policy, plus the older map-based high-level pipelines) so the two are never
confused. See SAFTER_src/README.md for the package layout and why a few
files here are deliberate DUPLICATES of files still under src/high_level_rl/
rather than moves (those files are also load-bearing for map-based code that
intentionally stays under src/).
"""

from pathlib import Path

SAFTER_SRC_PATH: Path = Path(__file__).parent
