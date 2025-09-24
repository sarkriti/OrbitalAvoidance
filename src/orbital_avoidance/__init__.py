"""Orbital Avoidance unified package.

Subpackages:
- physics: Orekit-backed propagation and orbital mechanics utilities
- envs: Gymnasium RL environments and configurable simulation envs
- algo: Reinforcement learning algorithm wrappers & training manager
- config: Dataclasses and loaders for scenarios & simulation parameters

This package provides a stable public API for external scripts.
"""

from .physics.orekit_physics import (
    OrekitOrbitPropagator, OrbitElements, SpacecraftProperties, create_propagator
)
from .envs.rl_training_env import (
    RLOrbitalEnvironment, create_rl_environment
)
from .envs.configurable_env import (
    ConfigurableOrbitalEnvironment
)
from .algo.rl_algorithms import (
    RLAlgorithmFactory, TrainingManager, RewardFunctionManager, create_algorithm, get_reward_config
)
from .config.config_system import (
    SimulationConfig, SpacecraftConfig, OrbitConfig, DebrisFieldConfig,
    PhysicsConfig, RewardConfig, ObservationConfig, ConfigurationLoader, DebrisGenerator
)

__all__ = [
    # Physics
    'OrekitOrbitPropagator', 'OrbitElements', 'SpacecraftProperties', 'create_propagator',
    # Environments
    'RLOrbitalEnvironment', 'create_rl_environment', 'ConfigurableOrbitalEnvironment',
    # Algorithms
    'RLAlgorithmFactory', 'TrainingManager', 'RewardFunctionManager', 'create_algorithm', 'get_reward_config',
    # Config
    'SimulationConfig', 'SpacecraftConfig', 'OrbitConfig', 'DebrisFieldConfig',
    'PhysicsConfig', 'RewardConfig', 'ObservationConfig', 'ConfigurationLoader', 'DebrisGenerator'
]
