"""Legacy compatibility package for config_system that re-exports the modern implementation."""
from __future__ import annotations
import sys
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

warnings.warn(
    "Importing config_system from legacy location is deprecated. Use 'orbital_avoidance.config.config_system' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from orbital_avoidance.config.config_system import *  # noqa: F401,F403
