"""
orbital_avoidance package

This package provides:
- RLOrbitalEnvironment: Gymnasium-based environment for orbital debris avoidance
- create_rl_environment: factory to build the environment from scenario configs
- RLAlgorithmFactory: factory for Stable-Baselines3 algorithms
- TrainingManager: handles logging, checkpoints, and evaluation loops
- RewardFunctionManager: manages reward function selection
"""

from __future__ import annotations

# ---------------------------------------------------------
# Core environment imports (required)
# ---------------------------------------------------------
from .envs.configurable_env import RLOrbitalEnvironment, create_rl_environment

# ---------------------------------------------------------
# Algorithm and training imports (optional, fail loudly if missing)
# ---------------------------------------------------------
try:
    from .algo.rl_algorithms import (
        RLAlgorithmFactory,
        TrainingManager,
        RewardFunctionManager,
        create_algorithm,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import RL algorithm components from algo.rl_algorithms: {e}"
    )

# ---------------------------------------------------------
# Configuration system (optional)
# ---------------------------------------------------------
try:
    from .config.config_system import ConfigurationLoader
except ImportError:
    ConfigurationLoader = None

# ---------------------------------------------------------
# Reward system (optional)
# ---------------------------------------------------------
try:
    from .rewards.improved_rewards import ImprovedRewardFunction
except ImportError:
    ImprovedRewardFunction = None

# ---------------------------------------------------------
# Convenience function
# ---------------------------------------------------------
def get_reward_config(name: str):
    """Convenience function to get reward configuration by name."""
    if RewardFunctionManager is None:
        raise ImportError("RewardFunctionManager is not available.")
    manager = RewardFunctionManager()
    return manager.get_reward_config(name)

# ---------------------------------------------------------
# Package version
# ---------------------------------------------------------
__version__ = "1.0.0"

# ---------------------------------------------------------
# Exposed symbols
# ---------------------------------------------------------
__all__ = [
    # Environment
    "RLOrbitalEnvironment",
    "create_rl_environment",

    # Algorithms & training
    "RLAlgorithmFactory",
    "TrainingManager",
    "RewardFunctionManager",
    "create_algorithm",

    # Config & reward helpers
    "ConfigurationLoader",
    "ImprovedRewardFunction",
    "get_reward_config",
]
