# src/orbital_avoidance/__init__.py

"""Unified orbital avoidance package."""

from __future__ import annotations

# Core environment imports
try:
    from .envs.configurable_env import ConfigurableOrbitalEnvironment
    from .envs.rl_training_env import RLOrbitalEnvironment, create_rl_environment
except ImportError as e:
    # Graceful fallback if environment modules aren't available
    ConfigurableOrbitalEnvironment = None
    RLOrbitalEnvironment = None
    create_rl_environment = None

# Algorithm imports - remove the non-existent get_reward_config
try:
    from .algo.rl_algorithms import (
        RLAlgorithmFactory, TrainingManager, RewardFunctionManager, create_algorithm
    )
except ImportError as e:
    # Graceful fallback if RL modules aren't available
    RLAlgorithmFactory = None
    TrainingManager = None
    RewardFunctionManager = None
    create_algorithm = None

# Configuration system imports
try:
    from .config.config_system import ConfigurationLoader
except ImportError as e:
    ConfigurationLoader = None

# Reward system imports
try:
    from .rewards.improved_rewards import ImprovedRewardFunction
except ImportError as e:
    ImprovedRewardFunction = None

# Create a convenience function to replace the missing get_reward_config
def get_reward_config(name: str):
    """Convenience function to get reward configuration by name."""
    if RewardFunctionManager is None:
        raise ImportError("RewardFunctionManager not available")
    manager = RewardFunctionManager()
    return manager.get_reward_config(name)

__version__ = "1.0.0"

__all__ = [
    # Environment classes and functions
    'ConfigurableOrbitalEnvironment',
    'RLOrbitalEnvironment', 
    'create_rl_environment',
    
    # Algorithm classes and functions
    'RLAlgorithmFactory',
    'TrainingManager', 
    'RewardFunctionManager',
    'create_algorithm',
    'get_reward_config',  # Our convenience function
    
    # Configuration
    'ConfigurationLoader',
    
    # Rewards
    'ImprovedRewardFunction',
]