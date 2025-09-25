#!/usr/bin/env python3
"""
Configurable Orbital Avoidance Environment

This module provides a highly configurable orbital simulator that can be
easily customized through JSON/YAML configuration files. It supports:

- Multiple satellite configurations
- Procedural debris generation
- Modular physics models
- Pluggable reward functions
- Configurable observation spaces
- Scenario templates
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config_system import (
    SimulationConfig, ConfigurationLoader, DebrisGenerator,
    SpacecraftConfig, OrbitConfig, PhysicsConfig, RewardConfig
)
from orekit_physics import create_propagator, OrekitOrbitPropagator

logger = logging.getLogger(__name__)


class ConfigurableOrbitalEnvironment(gym.Env):
    """
    Highly configurable orbital avoidance environment.
    
    Can be configured via JSON/YAML files to support different scenarios:
    - LEO debris avoidance
    - GEO station keeping  
    - Multi-satellite constellations
    - Custom orbital scenarios
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 config: Optional[SimulationConfig] = None,
                 render_mode: Optional[str] = None):
        """
        Initialize configurable environment."""
        
            import warnings
            warnings.warn(
                "Importing 'configurable_env' from the old location is deprecated. Use 'from orbital_avoidance.envs import configurable_env' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            from orbital_avoidance.envs.configurable_env import *  # noqa: F401,F403
                    low=-1.0, high=1.0, 
                    shape=(3,), 
                    dtype=np.float32
                )
            self.action_space = spaces.Dict(action_spaces)
    
    def _setup_rewards(self):
        """Setup reward function."""
        self.reward_config = self.config.rewards
        # Reward computation will be done in _compute_reward()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        elif self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        # Create actual satellite and debris objects
        self.satellites = {}
        self.debris_objects = {}
        
        # Create single shared propagator for all objects
        self.shared_propagator = create_propagator()
        
    # Create satellites
        for name, sat_config in self.satellite_configs.items():
            try:
                # Initialize state using shared propagator
                orbit_elements = sat_config.orbit.to_orbit_elements()
        # Persistent object registration
        self.shared_propagator.create_persistent_object(f"sat::{name}", orbit_elements)
        pos, vel = self.shared_propagator.propagate_persistent_delta(f"sat::{name}", 0.0)
                
                self.satellites[name] = {
                    'config': sat_config,
                    'position': pos,
                    'velocity': vel,
                    'fuel_remaining': sat_config.fuel_capacity_kg,
                    'orbit_elements': orbit_elements
                }
                
                logger.debug(f"Created satellite {name} at altitude {np.linalg.norm(pos) - 6371:.1f} km")
                
            except Exception as e:
                logger.error(f"Failed to create satellite {name}: {e}")
                raise
        
    # Create debris objects
        for name, debris_config in self.debris_configs.items():
            try:
                orbit_elements = debris_config.orbit.to_orbit_elements()
        self.shared_propagator.create_persistent_object(f"deb::{name}", orbit_elements)
        pos, vel = self.shared_propagator.propagate_persistent_delta(f"deb::{name}", 0.0)
                
                self.debris_objects[name] = {
                    'config': debris_config,
                    'position': pos,
                    'velocity': vel,
                    'orbit_elements': orbit_elements
                }
                
            except Exception as e:
                logger.error(f"Failed to create debris {name}: {e}")
                raise
        
        # Reset simulation state
        self.current_time = 0.0
        self.episode_step = 0
        
        # Get initial observation
        observation = self._get_observation()
        
        logger.info(f"Environment reset with {len(self.satellites)} satellites and {len(self.debris_objects)} debris")
        
        return observation, {}
    
    def step(self, action):
        """Execute one simulation step."""
        # Apply actions to controllable satellites
        self._apply_actions(action)
        
        # Propagate all objects
        self._propagate_objects()
        
        # Check for collisions
        collisions = self._check_collisions()
        
        # Compute reward
        reward = self._compute_reward(action, collisions)
        
        # Check if episode is done
        terminated = len(collisions) > 0  # Collision ends episode
        truncated = self.episode_step >= self.max_episode_steps
        
        # Get observation
        observation = self._get_observation()
        
        # Create info dict
        info = {
            'collisions': collisions,
            'time': self.current_time,
            'step': self.episode_step,
            'fuel_remaining': {name: sat['fuel_remaining'] for name, sat in self.satellites.items()}
        }
        
        self.episode_step += 1
        
        return observation, reward, terminated, truncated, info
    
    def _apply_actions(self, action):
        """Apply thrust actions to controllable satellites."""
        controllable_sats = [s for s in self.config.satellites if s.is_controllable]
        
        if len(controllable_sats) == 1:
            # Single satellite action
            sat_name = controllable_sats[0].name
            thrust_vector = np.array(action, dtype=np.float32)
            self._apply_thrust(sat_name, thrust_vector)
        else:
            # Multiple satellite actions
            for sat_name, thrust_vector in action.items():
                if sat_name in self.satellites:
                    self._apply_thrust(sat_name, np.array(thrust_vector, dtype=np.float32))
    
    def _apply_thrust(self, sat_name: str, thrust_vector: np.ndarray):
        """Apply thrust to a specific satellite."""
        if sat_name not in self.satellites:
            return
        
        satellite = self.satellites[sat_name]
        sat_config = satellite['config']
        
        # Limit thrust magnitude
        thrust_magnitude = np.linalg.norm(thrust_vector)
        if thrust_magnitude > sat_config.max_thrust_n:
            thrust_vector = thrust_vector / thrust_magnitude * sat_config.max_thrust_n
        
        # Check fuel availability
        fuel_cost = thrust_magnitude * self.config.physics.time_step_seconds * 0.001  # Simple fuel model
        if satellite['fuel_remaining'] >= fuel_cost:
            # Apply thrust (simplified - just change velocity)
            mass = sat_config.mass_kg
            delta_v = thrust_vector / mass * self.config.physics.time_step_seconds
            satellite['velocity'] += delta_v
            satellite['fuel_remaining'] -= fuel_cost

            # Recompute orbital elements so future propagation reflects thrust change
            # Apply instantaneous delta-v to persistent propagator state
            try:
                self.shared_propagator.apply_instantaneous_delta_v(f"sat::{sat_name}", delta_v/1000.0)
            except Exception as e:
                logger.debug(f"Delta-v application to persistent state failed for {sat_name}: {e}")
    
    def _propagate_objects(self):
        """Propagate all objects forward in time."""
        dt = self.config.physics.time_step_seconds
        self.current_time += dt
        
        # Propagate satellites using persistent delta propagation
        for name, satellite in self.satellites.items():
            try:
                pos, vel = self.shared_propagator.propagate_persistent_delta(f"sat::{name}", dt)
                satellite['position'] = pos
                satellite['velocity'] = vel
            except Exception as e:
                logger.warning(f"Failed to propagate satellite {name}: {e}")
        
        # Propagate debris using persistent delta propagation
        for name, debris in self.debris_objects.items():
            try:
                pos, vel = self.shared_propagator.propagate_persistent_delta(f"deb::{name}", dt)
                debris['position'] = pos
                debris['velocity'] = vel
            except Exception as e:
                logger.warning(f"Failed to propagate debris {name}: {e}")
    
    def _check_collisions(self) -> List[Tuple[str, str, float]]:
        """Check for collisions between objects."""
        collisions = []
        threshold_km = self.config.physics.collision_threshold_km
        
        # Check satellite-debris collisions
        for sat_name, satellite in self.satellites.items():
            for debris_name, debris in self.debris_objects.items():
                distance = np.linalg.norm(satellite['position'] - debris['position'])
                if distance < threshold_km:
                    collisions.append((sat_name, debris_name, distance))
        
        # Check satellite-satellite collisions
        sat_names = list(self.satellites.keys())
        for i, sat1_name in enumerate(sat_names):
            for sat2_name in sat_names[i+1:]:
                sat1 = self.satellites[sat1_name]
                sat2 = self.satellites[sat2_name]
                distance = np.linalg.norm(sat1['position'] - sat2['position'])
                if distance < threshold_km:
                    collisions.append((sat1_name, sat2_name, distance))
        
        return collisions
    
    def _compute_reward(self, action, collisions: List[Tuple[str, str, float]]) -> float:
        """Compute reward based on configuration."""
        reward = 0.0
        
        # Collision penalty
        if collisions:
            reward += self.reward_config.collision_penalty * len(collisions)
        
        # Fuel cost
        if self.reward_config.fuel_cost_weight > 0:
            if isinstance(action, dict):
                fuel_cost = sum(np.linalg.norm(thrust) for thrust in action.values())
            else:
                fuel_cost = np.linalg.norm(action)
            reward -= self.reward_config.fuel_cost_weight * fuel_cost
        
        # Close approach penalties
        if self.reward_config.close_approach_penalty_weight > 0:
            min_distances = []
            for sat_name, satellite in self.satellites.items():
                sat_min_dist = float('inf')
                for debris_name, debris in self.debris_objects.items():
                    dist = np.linalg.norm(satellite['position'] - debris['position'])
                    sat_min_dist = min(sat_min_dist, dist)
                min_distances.append(sat_min_dist)
            
            for dist in min_distances:
                if dist < self.reward_config.safety_distance_km:
                    penalty = (self.reward_config.safety_distance_km - dist) / self.reward_config.safety_distance_km
                    reward -= self.reward_config.close_approach_penalty_weight * penalty
        
        # Mission success bonus (no collisions for full episode)
        if self.episode_step == self.max_episode_steps - 1 and not collisions:
            reward += self.reward_config.mission_success_reward
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs_config = self.config.observations
        observations = []
        
        # Get primary satellite (first one)
        primary_sat_name = list(self.satellites.keys())[0]
        primary_sat = self.satellites[primary_sat_name]
        primary_pos = primary_sat['position']
        primary_vel = primary_sat['velocity']
        
        # Add satellite state
        for sat_name, satellite in self.satellites.items():
            observations.extend(satellite['position'])
            observations.extend(satellite['velocity'])
        
        # Find closest debris objects
        debris_distances = []
        for debris_name, debris in self.debris_objects.items():
            dist = np.linalg.norm(primary_pos - debris['position'])
            debris_distances.append((dist, debris_name, debris))
        
        # Sort by distance and take closest ones
        debris_distances.sort(key=lambda x: x[0])
        max_tracked = min(obs_config.max_tracked_objects, len(debris_distances))
        
        for i in range(max_tracked):
            _, debris_name, debris = debris_distances[i]
            
            if obs_config.include_relative_positions:
                rel_pos = debris['position'] - primary_pos
                observations.extend(rel_pos)
            
            if obs_config.include_relative_velocities:
                rel_vel = debris['velocity'] - primary_vel
                observations.extend(rel_vel)
            
            if obs_config.include_time_to_collision:
                # Simple time to collision estimate
                rel_pos = debris['position'] - primary_pos
                rel_vel = debris['velocity'] - primary_vel
                if np.dot(rel_pos, rel_vel) < 0:  # Approaching
                    ttc = np.linalg.norm(rel_pos) / max(np.linalg.norm(rel_vel), 0.001)
                else:
                    ttc = 1e6  # Very large time if not approaching
                observations.append(ttc)
        
        # Pad with zeros if we have fewer debris than max_tracked
        expected_debris_obs = 0
        if obs_config.include_relative_positions:
            expected_debris_obs += 3
        if obs_config.include_relative_velocities:
            expected_debris_obs += 3
        if obs_config.include_time_to_collision:
            expected_debris_obs += 1
        
        current_debris_obs = len(debris_distances) * expected_debris_obs
        expected_total_debris_obs = obs_config.max_tracked_objects * expected_debris_obs
        padding_needed = expected_total_debris_obs - current_debris_obs
        observations.extend([0.0] * padding_needed)
        
        # Add fuel levels
        if obs_config.include_fuel_levels:
            for sat_name, satellite in self.satellites.items():
                fuel_fraction = satellite['fuel_remaining'] / satellite['config'].fuel_capacity_kg
                observations.append(fuel_fraction)
        
        return np.array(observations, dtype=np.float32)
    
    def render(self):
        """Render the environment (placeholder)."""
        if self.render_mode == "human":
            print(f"Step {self.episode_step}, Time: {self.current_time:.1f}s")
            for sat_name, satellite in self.satellites.items():
                pos = satellite['position']
                altitude = np.linalg.norm(pos) - 6371
                fuel = satellite['fuel_remaining']
                print(f"  {sat_name}: Alt={altitude:.1f}km, Fuel={fuel:.1f}kg")
    
    def close(self):
        """Clean up resources."""
        # Clear references to help with garbage collection
        if hasattr(self, 'shared_propagator'):
            self.shared_propagator = None
        if hasattr(self, 'satellites'):
            self.satellites.clear()
        if hasattr(self, 'debris_objects'):
            self.debris_objects.clear()
        
        # Force garbage collection
        import gc
        gc.collect()


# Convenience functions for loading scenarios
def load_scenario(scenario_name: str) -> ConfigurableOrbitalEnvironment:
    """Load a predefined scenario by name."""
    scenario_path = f"scenarios/{scenario_name}.json"
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
    
    return ConfigurableOrbitalEnvironment(config_file=scenario_path)


def list_scenarios() -> List[str]:
    """List available scenario files."""
    scenarios_dir = "scenarios"
    if not os.path.exists(scenarios_dir):
        return []
    
    scenarios = []
    for filename in os.listdir(scenarios_dir):
        if filename.endswith('.json') or filename.endswith('.yaml'):
            scenarios.append(os.path.splitext(filename)[0])
    
    return scenarios


if __name__ == "__main__":
    # Test the configurable environment
    logging.basicConfig(level=logging.INFO)
    
    print("Available scenarios:")
    for scenario in list_scenarios():
        print(f"  - {scenario}")
    
    # Test with LEO debris avoidance scenario
    try:
        env = load_scenario("leo_debris_avoidance")
        
        print(f"\nTesting {env.config.scenario_name}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        # Take a few random steps
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {step}: reward={reward:.2f}, collisions={len(info['collisions'])}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("✅ Configurable environment test successful!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()