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

from .config_system import (
    SimulationConfig, ConfigurationLoader, DebrisGenerator,
    SpacecraftConfig, OrbitConfig, PhysicsConfig, RewardConfig
)
from .orekit_physics import create_propagator, OrekitOrbitPropagator
from .multiagent_env import MultiAgentOrbitalAvoidance

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
        Initialize configurable environment.
        
        Args:
            config_file: Path to JSON/YAML configuration file
            config: SimulationConfig object (alternative to config_file)
            render_mode: Rendering mode ('human', 'rgb_array', None)
        """
        super().__init__()
        
        # Load configuration
        if config_file:
            self.config = ConfigurationLoader.load_from_file(config_file)
        elif config:
            self.config = config
        else:
            raise ValueError("Must provide either config_file or config object")
        
        # Validate configuration
        errors = ConfigurationLoader.validate_config(self.config)
        if errors:
            raise ValueError(f"Configuration errors: {errors}")
        
        self.render_mode = render_mode
        
        # Initialize components
        self._setup_objects()
        self._setup_physics()
        self._setup_spaces()
        self._setup_rewards()
        
        # State tracking
        self.current_time = 0.0
        self.episode_step = 0
        self.max_episode_steps = int(
            self.config.physics.simulation_duration_seconds / 
            self.config.physics.time_step_seconds
        )
        
        logger.info(f"Initialized {self.config.scenario_name}")
        logger.info(f"Satellites: {len(self.satellites)}, Debris: {len(self.debris_objects)}")
    
    def _setup_objects(self):
        """Setup satellites and debris objects from configuration."""
        # Initialize satellites
        self.satellites = {}
        self.satellite_configs = {}
        
        for sat_config in self.config.satellites:
            self.satellite_configs[sat_config.name] = sat_config
            # Will create actual objects in reset()
        
        # Generate debris from fields
        self.debris_objects = {}
        self.debris_configs = {}
        
        # Add static debris
        for debris_config in self.config.static_debris:
            self.debris_configs[debris_config.name] = debris_config
        
        # Generate debris fields
        for i, field_config in enumerate(self.config.debris_fields):
            generated_debris = DebrisGenerator.generate_debris_field(
                field_config, seed=self.config.random_seed
            )
            for j, debris in enumerate(generated_debris):
                name = f"Generated_Debris_{i}_{j}"
                debris.name = name
                self.debris_configs[name] = debris
        
        logger.info(f"Setup {len(self.satellite_configs)} satellites and {len(self.debris_configs)} debris objects")
    
    def _setup_physics(self):
        """Setup physics propagators."""
        self.propagators = {}
        # Will create propagators for each object in reset()
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        obs_config = self.config.observations
        
        # Calculate observation space size
        num_satellites = len(self.config.satellites)
        max_tracked = min(obs_config.max_tracked_objects, len(self.debris_configs))
        
        obs_size = 0
        if obs_config.include_relative_positions:
            obs_size += max_tracked * 3  # x, y, z for each debris
        if obs_config.include_relative_velocities:
            obs_size += max_tracked * 3  # vx, vy, vz for each debris
        if obs_config.include_fuel_levels:
            obs_size += num_satellites  # fuel level for each satellite
        if obs_config.include_time_to_collision:
            obs_size += max_tracked  # time to collision for each debris
        
        # Add satellite's own state (position, velocity)
        obs_size += 6 * num_satellites  # pos + vel for each satellite
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Action space - thrust commands for each controllable satellite
        controllable_sats = [s for s in self.config.satellites if s.is_controllable]
        if len(controllable_sats) == 1:
            # Single satellite - 3D thrust vector
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, 
                shape=(3,), 
                dtype=np.float32
            )
        else:
            # Multiple satellites - dict of actions
            action_spaces = {}
            for sat in controllable_sats:
                action_spaces[sat.name] = spaces.Box(
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
        self.propagators = {}
        
        # Create satellites
        for name, sat_config in self.satellite_configs.items():
            try:
                propagator = create_propagator()
                self.propagators[name] = propagator
                
                # Initialize state
                orbit_elements = sat_config.orbit.to_orbit_elements()
                pos, vel = propagator.propagate_orbit(orbit_elements, 0.0)
                
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
                propagator = create_propagator()
                self.propagators[name] = propagator
                
                orbit_elements = debris_config.orbit.to_orbit_elements()
                pos, vel = propagator.propagate_orbit(orbit_elements, 0.0)
                
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
    
    def _propagate_objects(self):
        """Propagate all objects forward in time."""
        dt = self.config.physics.time_step_seconds
        self.current_time += dt
        
        # Propagate satellites
        for name, satellite in self.satellites.items():
            try:
                propagator = self.propagators[name]
                pos, vel = propagator.propagate_orbit(satellite['orbit_elements'], self.current_time)
                satellite['position'] = pos
                satellite['velocity'] = vel
            except Exception as e:
                logger.warning(f"Failed to propagate satellite {name}: {e}")
        
        # Propagate debris
        for name, debris in self.debris_objects.items():
            try:
                propagator = self.propagators[name]
                pos, vel = propagator.propagate_orbit(debris['orbit_elements'], self.current_time)
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
        pass


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