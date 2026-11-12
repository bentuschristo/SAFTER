"""Shared math/env/filter building blocks for the laser-based SAFTER pipeline.

Everything here is imported directly by submodule (e.g.
``from SAFTER_src.high_level_rl.laser_cbf_filters import ...``), so this
file intentionally does not eagerly re-export anything -- that keeps one
file's import error from taking down every other file in the package.
"""
