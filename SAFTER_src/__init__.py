"""SAFTER (laser-based safe navigation: Vanilla / CBF-CF / CBF-QP / bCBF-CF / bCBF-QP) source package.

Kept separate from src/, which trains the frozen low-level locomotion policy.
"""

from pathlib import Path

SAFTER_SRC_PATH: Path = Path(__file__).parent
