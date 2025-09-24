#!/usr/bin/env python3
"""
Compatibility shim for legacy imports.

This file keeps old import paths working while the canonical implementation lives
in `src/orbital_avoidance/envs/rl_training_env.py`. It re-exports the public
symbols from the modern package and emits a deprecation warning.
"""

from __future__ import annotations

import warnings
import sys
from pathlib import Path

# If the package isn't installed, ensure `src/` is on sys.path so imports work
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

warnings.warn(
    "Importing rl_training_env from the legacy top-level package is deprecated. "
    "Please use 'from orbital_avoidance.envs import rl_training_env' or install the package in editable mode.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the modern implementation
from orbital_avoidance.envs.rl_training_env import *  # noqa: F401,F403

# Optionally expose the factory function at top-level for legacy scripts
try:
    create_rl_environment = create_rl_environment  # type: ignore[name-defined]
except NameError:
    pass