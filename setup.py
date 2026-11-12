"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
]

# packages=["src"] alone does not expose src's own subpackages (src.tasks,
# src.assets, etc.) or SAFTER_src at all -- a plain setuptools packages=[]
# list is not recursive, it only registers exactly the names given. This
# went unnoticed because older/looser editable-install behavior (pre
# PEP 660 "strict" editable wheels) tended to add the whole project
# directory to sys.path regardless of what packages= declared; a modern
# setuptools makes editable installs expose exactly what's declared here,
# surfacing the gap as ModuleNotFoundError for SAFTER_src/src.tasks/etc.
# find_packages(include=...) discovers both package trees recursively
# without sweeping in unrelated top-level clutter some training servers
# have accumulated at the repo root (old script/src archives, stray
# experiment directories, etc).
PACKAGES = find_packages(include=["src", "src.*", "SAFTER_src", "SAFTER_src.*"])

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=PACKAGES,
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
