#!/usr/bin/env python3
"""
configurable_env.py

Provides a unified interface for creating and configuring RL environments
for orbital debris avoidance tasks.

Features:
- Reads scenario configuration files
- Allows toggling physics fidelity (gravity, drag, collisions, perturbations)
- Supports reward function injection via RewardFunctionManager
- Compatible with Stable-Baselines3 wrappers
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Core Environment Class
# ---------------------------------------------------------
class RLOrbitalEnvironment(gym.Env):
    """Base RL environment for orbital debris avoidance."""

    metadata = {"render_modes": ["human", "ansi", None]}

    def __init__(
        self,
        config: Dict[str, Any],
        reward_function=None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        self.config = config
        self.reward_function = reward_function
        self.render_mode = render_mode

        # Extract scenario parameters
        self.max_debris = config.get("max_debris_tracked", 10)
        self.physics_cfg = config.get("physics", {})

        # Example state & action spaces (you can adapt based on real model)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.max_debris * 4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Internal state
        self.state = None
        self.t = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.t = 0
        return self.state, {}

    def step(self, action):
        # Example physics integration
        self.t += 1

        # Placeholder state transition
        self.state = self.state + 0.01 * np.random.randn(*self.state.shape)

        # Reward computation (delegated if custom function is provided)
        reward = self.compute_reward(action)

        # Example termination condition
        terminated = self.t >= self.config.get("episode_horizon", 200)
        truncated = False  # could be used for timeout

        info = {"time_step": self.t}
        return self.state, reward, terminated, truncated, info

    def compute_reward(self, action):
        if self.reward_function:
            return self.reward_function(self.state, action)
        return -np.linalg.norm(action)  # penalize large maneuvers by default

    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.t}, State: {self.state[:4]}")

    def close(self):
        pass


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def load_scenario_config(config_file: str) -> Dict[str, Any]:
    """Load scenario JSON config."""
    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"Scenario config file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def create_rl_environment(
    config_file: str,
    max_debris_tracked: int = 10,
    render_mode: Optional[str] = None,
    reward_config: Optional[Dict[str, Any]] = None,
    physics_overrides: Optional[Dict[str, bool]] = None,
) -> RLOrbitalEnvironment:
    """Factory to create a configured RLOrbitalEnvironment."""
    scenario_cfg = load_scenario_config(config_file)

    # Merge runtime overrides
    scenario_cfg["max_debris_tracked"] = max_debris_tracked
    if physics_overrides:
        scenario_cfg["physics"] = {**scenario_cfg.get("physics", {}), **physics_overrides}

    reward_fn = None
    if reward_config:
        try:
            from orbital_avoidance.reward_functions import RewardFunctionManager
            reward_fn = RewardFunctionManager.get_reward_function(reward_config)
        except ImportError:
            logger.warning("RewardFunctionManager not available, using default reward.")

    env = RLOrbitalEnvironment(
        config=scenario_cfg,
        reward_function=reward_fn,
        render_mode=render_mode
    )
    logger.info(f"Created RLOrbitalEnvironment with max_debris={max_debris_tracked}")
    return env
