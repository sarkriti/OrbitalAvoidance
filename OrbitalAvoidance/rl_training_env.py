#!/usr/bin/env python3
"""
RL Training Environment for Orbital Debris Avoidance

This module provides a reinforcement learning environment specifically designed
for training agents to control satellites in orbital debris fields. Features:

- Exact state representation as specified (position, velocity, fuel, time, debris data)
- Continuous action space (-1 to 1) for X, Y, Z thrust control
- Linear propagation for collision time prediction
- Modular reward system with collision, fuel, orbit, and mission components
- Easy algorithm switching (PPO, SAC, etc.)
"""

from __future__ import annotations

import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Callable
import gymnasium as gym
from gymnasium import spaces
import logging

from .configurable_env import ConfigurableOrbitalEnvironment
from .config_system import SimulationConfig, SpacecraftConfig, OrbitConfig
from .orekit_physics import OrekitOrbitPropagator, OrbitElements

logger = logging.getLogger(__name__)


class RLOrbitalEnvironment(gym.Env):
    """
    Reinforcement Learning environment for orbital debris avoidance.
    
    State Space:
    - Spacecraft state: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
    - Mission info: [fuel_remaining_normalized, time_remaining_normalized]
    - Top N debris: [rel_pos_x, rel_pos_y, rel_pos_z, rel_vel_x, rel_vel_y, rel_vel_z, 
                     miss_distance, time_to_collision] for each debris
    
    Action Space:
    - Continuous thrust control: [thrust_x, thrust_y, thrust_z] in range [-1, 1]
    """
    
    def __init__(self,
                 config_file: Optional[str] = None,
                 config: Optional[SimulationConfig] = None,
                 max_debris_tracked: int = 10,
                 reward_config: Optional[Dict] = None,
                 render_mode: Optional[str] = None):
        """
        Initialize RL environment.
        
        Args:
            config_file: Path to scenario configuration file
            config: Direct configuration object
            max_debris_tracked: Maximum number of debris objects to include in state
            reward_config: Reward function configuration
            render_mode: Rendering mode for visualization
        """
        super().__init__()
        
        # Load configuration
        if config is not None:
            self.config = config
        elif config_file is not None:
            from .config_system import ConfigurationLoader
            loader = ConfigurationLoader()
            self.config = loader.load_config(config_file)
        else:
            raise ValueError("Must provide either config_file or config")
        
        # RL-specific parameters
        self.max_debris_tracked = max_debris_tracked
        self.render_mode = render_mode
        
        # Initialize physics engine
        self.physics = OrekitOrbitPropagator()
        
        # Find controllable satellite
        self.controllable_satellites = [
            sat for sat in self.config.satellites if sat.is_controllable
        ]
        if not self.controllable_satellites:
            raise ValueError("No controllable satellites found in configuration")
        if len(self.controllable_satellites) > 1:
            logger.warning("Multiple controllable satellites found, using first one")
        
        self.main_satellite = self.controllable_satellites[0]
        
        # Reward function setup
        self.reward_config = reward_config or self._default_reward_config()
        self.reward_function = self._create_reward_function()
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Calculate observation space size
        obs_size = self._calculate_observation_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize state
        self._reset_simulation()
        
        # Mission tracking
        self.step_count = 0
        self.max_steps = self.config.simulation.max_steps_per_episode
        self.initial_fuel = self.main_satellite.fuel_capacity_kg
        self.collision_free_steps = 0
        
        # Reference orbital elements for orbit maintenance reward
        self.reference_orbit = self._get_reference_orbit()
        
    def _default_reward_config(self) -> Dict:
        """Default reward configuration matching user specifications."""
        return {
            'weights': {
                'collision': 1.0,    # W_collision
                'fuel': 1.0,         # W_fuel  
                'orbit': 1.0,        # W_orbit
                'mission': 1.0       # W_mission
            },
            'collision': {
                'penalty': -1000.0,
                'threshold': 0.1,    # Collision probability threshold
                'small_reward': 1.0
            },
            'fuel': {
                'efficiency_weight': 0.1
            },
            'orbit': {
                'tolerances': {
                    'semi_major_axis_m': 100.0,     # 100m
                    'eccentricity': 0.01,           # 0.01
                    'inclination_rad': 0.01,        # 0.01 rad
                    'raan_rad': 0.01,               # 0.01 rad (Ω)
                    'arg_periapsis_rad': 0.01       # 0.01 rad (ω)
                }
            },
            'mission': {
                'survival_reward_interval': 500,   # Every 500 steps
                'survival_multiplier': 3.0,
                'mission_weight': 1.0
            }
        }
    
    def _calculate_observation_size(self) -> int:
        """Calculate total observation space size."""
        # Spacecraft state: 6 (position + velocity)
        # Mission info: 2 (fuel + time)
        # Per debris: 8 (rel_pos + rel_vel + miss_distance + time_to_collision)
        return 6 + 2 + (self.max_debris_tracked * 8)
    
    def _get_reference_orbit(self) -> OrbitElements:
        """Get reference orbital elements for orbit maintenance."""
        sat_orbit = self.main_satellite.orbit
        return OrbitElements(
            semi_major_axis_km=sat_orbit.altitude_km + 6371.0,  # Convert to semi-major axis
            eccentricity=sat_orbit.eccentricity,
            inclination_deg=sat_orbit.inclination_deg,
            raan_deg=sat_orbit.raan_deg,
            arg_periapsis_deg=sat_orbit.arg_periapsis_deg,
            mean_anomaly_deg=sat_orbit.mean_anomaly_deg,
            epoch_mjd=sat_orbit.epoch_mjd
        )
    
    def _reset_simulation(self):
        """Reset simulation state."""
        # Initialize base environment
        self.base_env = ConfigurableOrbitalEnvironment(config=self.config)
        self.base_env.reset()
        
        # Get initial state
        self.current_satellite_state = self.base_env.satellites[0]  # Main satellite
        self.current_debris_states = self.base_env.debris_objects
        
        # Reset counters
        self.step_count = 0
        self.collision_free_steps = 0
        
        # Reset fuel
        self.current_satellite_state.fuel_remaining_kg = self.initial_fuel
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return initial observation."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        self._reset_simulation()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Thrust command [thrust_x, thrust_y, thrust_z] in range [-1, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Validate action
        action = np.clip(action, -1.0, 1.0)
        
        # Convert action to actual thrust (scale by max thrust)
        thrust_newtons = action * self.main_satellite.max_thrust_n
        
        # Apply action to satellite
        self._apply_thrust(thrust_newtons)
        
        # Propagate physics
        self._propagate_physics()
        
        # Calculate reward
        reward = self.reward_function(action)
        
        # Check termination conditions
        terminated = self._check_collision()
        truncated = self.step_count >= self.max_steps
        
        # Update counters
        self.step_count += 1
        if not terminated:
            self.collision_free_steps += 1
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_thrust(self, thrust_newtons: np.ndarray):
        """Apply thrust to satellite and update fuel consumption."""
        # Calculate delta-v
        mass_kg = self.current_satellite_state.mass_kg
        dt = self.config.physics.time_step_s
        
        # F = ma, so a = F/m
        acceleration = thrust_newtons / mass_kg
        delta_v = acceleration * dt
        
        # Apply delta-v to velocity
        self.current_satellite_state.velocity_km_s += delta_v / 1000.0  # Convert m/s to km/s
        
        # Calculate fuel consumption (simplified rocket equation)
        thrust_magnitude = np.linalg.norm(thrust_newtons)
        if thrust_magnitude > 0:
            # Simplified fuel consumption: proportional to thrust
            specific_impulse = 300  # seconds (typical for chemical propulsion)
            fuel_flow_rate = thrust_magnitude / (specific_impulse * 9.81)  # kg/s
            fuel_consumed = fuel_flow_rate * dt
            
            self.current_satellite_state.fuel_remaining_kg = max(
                0, self.current_satellite_state.fuel_remaining_kg - fuel_consumed
            )
    
    def _propagate_physics(self):
        """Propagate orbital mechanics for all objects."""
        dt = self.config.physics.time_step_s
        
        # Propagate satellite
        sat_elements = self._state_to_orbit_elements(self.current_satellite_state)
        new_pos, new_vel = self.physics.propagate_orbit(sat_elements, dt)
        self.current_satellite_state.position_km = new_pos
        self.current_satellite_state.velocity_km_s = new_vel
        
        # Propagate debris (Keplerian motion - no thrust)
        for debris in self.current_debris_states:
            debris_elements = self._state_to_orbit_elements(debris)
            new_pos, new_vel = self.physics.propagate_orbit(debris_elements, dt)
            debris.position_km = new_pos
            debris.velocity_km_s = new_vel
    
    def _state_to_orbit_elements(self, spacecraft_state) -> OrbitElements:
        """Convert spacecraft state to orbital elements for propagation."""
        # This would typically use orbital mechanics to convert Cartesian to Keplerian
        # For now, use the stored orbital configuration as a placeholder
        return OrbitElements(
            semi_major_axis_km=6371.0 + 400.0,  # Approximate LEO
            eccentricity=0.001,
            inclination_deg=51.6,
            raan_deg=0.0,
            arg_periapsis_deg=0.0,
            mean_anomaly_deg=0.0,
            epoch_mjd=59945.0
        )
    
    def _get_observation(self) -> np.ndarray:
        """
        Create observation vector with exact format specified:
        [spacecraft_pos_xyz, spacecraft_vel_xyz, fuel_norm, time_norm, 
         debris_rel_pos_xyz, debris_rel_vel_xyz, debris_miss_dist, debris_time_to_collision, ...]
        """
        obs = []
        
        # Spacecraft state (6 elements)
        sat_pos = self.current_satellite_state.position_km
        sat_vel = self.current_satellite_state.velocity_km_s
        obs.extend([sat_pos[0], sat_pos[1], sat_pos[2]])
        obs.extend([sat_vel[0], sat_vel[1], sat_vel[2]])
        
        # Mission info (2 elements)
        fuel_remaining_normalized = (
            self.current_satellite_state.fuel_remaining_kg / self.initial_fuel
        )
        time_remaining_normalized = 1.0 - (self.step_count / self.max_steps)
        obs.extend([fuel_remaining_normalized, time_remaining_normalized])
        
        # Get closest debris objects
        closest_debris = self._get_closest_debris(self.max_debris_tracked)
        
        # Per debris object (8 elements each)
        for i, debris in enumerate(closest_debris):
            # Relative position
            rel_pos = debris.position_km - sat_pos
            obs.extend([rel_pos[0], rel_pos[1], rel_pos[2]])
            
            # Relative velocity  
            rel_vel = debris.velocity_km_s - sat_vel
            obs.extend([rel_vel[0], rel_vel[1], rel_vel[2]])
            
            # Miss distance and time to collision using linear propagation
            miss_distance, time_to_collision = self._calculate_collision_metrics(
                sat_pos, sat_vel, debris.position_km, debris.velocity_km_s
            )
            obs.extend([miss_distance, time_to_collision])
        
        # Pad with zeros if fewer debris than max_debris_tracked
        while len(closest_debris) < self.max_debris_tracked:
            obs.extend([0.0] * 8)  # 8 zeros for missing debris
        
        return np.array(obs, dtype=np.float32)
    
    def _get_closest_debris(self, max_count: int) -> List:
        """Get the closest debris objects by distance."""
        sat_pos = self.current_satellite_state.position_km
        
        # Calculate distances and sort
        debris_with_distance = []
        for debris in self.current_debris_states:
            distance = np.linalg.norm(debris.position_km - sat_pos)
            debris_with_distance.append((distance, debris))
        
        # Sort by distance and return top N
        debris_with_distance.sort(key=lambda x: x[0])
        return [debris for _, debris in debris_with_distance[:max_count]]
    
    def _calculate_collision_metrics(self, sat_pos: np.ndarray, sat_vel: np.ndarray,
                                   debris_pos: np.ndarray, debris_vel: np.ndarray) -> Tuple[float, float]:
        """
        Calculate miss distance and time to collision using linear propagation.
        
        Returns:
            miss_distance: Minimum distance between trajectories (km)
            time_to_collision: Time until closest approach (seconds), -1 if diverging
        """
        # Relative position and velocity
        rel_pos = debris_pos - sat_pos
        rel_vel = debris_vel - sat_vel
        
        # If objects are diverging, return large values
        if np.dot(rel_pos, rel_vel) > 0:
            return np.linalg.norm(rel_pos), -1.0
        
        # Time to closest approach using linear propagation
        # d(t) = |rel_pos + rel_vel * t|
        # Minimize: d²(t) = (rel_pos + rel_vel * t) · (rel_pos + rel_vel * t)
        # Taking derivative: 2 * (rel_pos + rel_vel * t) · rel_vel = 0
        # Solving: t = -(rel_pos · rel_vel) / |rel_vel|²
        
        rel_vel_squared = np.dot(rel_vel, rel_vel)
        if rel_vel_squared < 1e-10:  # Nearly stationary relative motion
            return np.linalg.norm(rel_pos), float('inf')
        
        time_to_closest = -np.dot(rel_pos, rel_vel) / rel_vel_squared
        
        # Calculate position at closest approach
        closest_rel_pos = rel_pos + rel_vel * time_to_closest
        miss_distance = np.linalg.norm(closest_rel_pos)
        
        return miss_distance, time_to_closest
    
    def _check_collision(self) -> bool:
        """Check if satellite has collided with any debris."""
        sat_pos = self.current_satellite_state.position_km
        collision_radius_km = self.main_satellite.collision_radius_m / 1000.0
        
        for debris in self.current_debris_states:
            distance = np.linalg.norm(debris.position_km - sat_pos)
            debris_radius_km = getattr(debris, 'collision_radius_m', 1.0) / 1000.0
            
            if distance < (collision_radius_km + debris_radius_km):
                return True
        
        return False
    
    def _create_reward_function(self) -> Callable:
        """Create reward function based on configuration."""
        def reward_function(action: np.ndarray) -> float:
            """
            Calculate total reward: R_total = W_collision * R_collision + W_fuel * R_fuel + W_orbit * R_orbit + W_mission * R_mission
            """
            weights = self.reward_config['weights']
            
            # R_collision: Negative reward for collision risk
            r_collision = self._calculate_collision_reward()
            
            # R_fuel: Penalize delta-v usage
            r_fuel = self._calculate_fuel_reward(action)
            
            # R_orbit: Penalize orbital element deviations
            r_orbit = self._calculate_orbit_reward()
            
            # R_mission: Reward mission progress
            r_mission = self._calculate_mission_reward()
            
            # Total weighted reward
            total_reward = (
                weights['collision'] * r_collision +
                weights['fuel'] * r_fuel +
                weights['orbit'] * r_orbit +
                weights['mission'] * r_mission
            )
            
            return total_reward
        
        return reward_function
    
    def _calculate_collision_reward(self) -> float:
        """Calculate collision-based reward component."""
        config = self.reward_config['collision']
        
        # Calculate collision probability based on closest debris
        closest_debris = self._get_closest_debris(5)  # Check top 5
        max_collision_prob = 0.0
        
        sat_pos = self.current_satellite_state.position_km
        sat_vel = self.current_satellite_state.velocity_km_s
        
        for debris in closest_debris:
            miss_distance, time_to_collision = self._calculate_collision_metrics(
                sat_pos, sat_vel, debris.position_km, debris.velocity_km_s
            )
            
            # Simple collision probability model
            if time_to_collision > 0 and time_to_collision < 3600:  # Within 1 hour
                collision_radius_km = self.main_satellite.collision_radius_m / 1000.0
                collision_prob = max(0, 1.0 - miss_distance / collision_radius_km)
                max_collision_prob = max(max_collision_prob, collision_prob)
        
        # Apply threshold-based reward
        if max_collision_prob > config['threshold']:
            return config['penalty']
        else:
            return config['small_reward']
    
    def _calculate_fuel_reward(self, action: np.ndarray) -> float:
        """Calculate fuel efficiency reward component."""
        # Delta-v magnitude calculation
        thrust_magnitude = np.linalg.norm(action) * self.main_satellite.max_thrust_n
        mass_kg = self.current_satellite_state.mass_kg
        dt = self.config.physics.time_step_s
        
        delta_v_magnitude = (thrust_magnitude / mass_kg) * dt  # m/s
        
        # Convert to km/s and apply penalty
        efficiency_weight = self.reward_config['fuel']['efficiency_weight']
        return -efficiency_weight * (delta_v_magnitude / 1000.0)
    
    def _calculate_orbit_reward(self) -> float:
        """Calculate orbital element deviation penalty."""
        # Get current orbital elements (simplified - would need proper conversion)
        current_elements = self._state_to_orbit_elements(self.current_satellite_state)
        reference = self.reference_orbit
        tolerances = self.reward_config['orbit']['tolerances']
        
        deviations = []
        
        # Semi-major axis deviation
        a_dev = abs(current_elements.semi_major_axis_km * 1000 - reference.semi_major_axis_km * 1000)
        deviations.append(max(0, a_dev - tolerances['semi_major_axis_m']) / tolerances['semi_major_axis_m'])
        
        # Eccentricity deviation
        e_dev = abs(current_elements.eccentricity - reference.eccentricity)
        deviations.append(max(0, e_dev - tolerances['eccentricity']) / tolerances['eccentricity'])
        
        # Inclination deviation (convert to radians)
        i_dev = abs(math.radians(current_elements.inclination_deg - reference.inclination_deg))
        deviations.append(max(0, i_dev - tolerances['inclination_rad']) / tolerances['inclination_rad'])
        
        # RAAN deviation
        raan_dev = abs(math.radians(current_elements.raan_deg - reference.raan_deg))
        deviations.append(max(0, raan_dev - tolerances['raan_rad']) / tolerances['raan_rad'])
        
        # Argument of periapsis deviation
        argp_dev = abs(math.radians(current_elements.arg_periapsis_deg - reference.arg_periapsis_deg))
        deviations.append(max(0, argp_dev - tolerances['arg_periapsis_rad']) / tolerances['arg_periapsis_rad'])
        
        return -sum(deviations)
    
    def _calculate_mission_reward(self) -> float:
        """Calculate mission progress reward component."""
        config = self.reward_config['mission']
        
        # Survival reward every N steps
        survival_reward = 0.0
        if self.collision_free_steps % config['survival_reward_interval'] == 0 and self.collision_free_steps > 0:
            fuel_ratio = self.current_satellite_state.fuel_remaining_kg / self.initial_fuel
            survival_reward = config['survival_multiplier'] * fuel_ratio * config['mission_weight']
        
        return survival_reward
    
    def _get_info(self) -> Dict:
        """Get additional information dictionary."""
        return {
            'step_count': self.step_count,
            'fuel_remaining_kg': self.current_satellite_state.fuel_remaining_kg,
            'fuel_remaining_percent': (self.current_satellite_state.fuel_remaining_kg / self.initial_fuel) * 100,
            'collision_free_steps': self.collision_free_steps,
            'closest_debris_distance_km': self._get_closest_debris_distance(),
            'satellite_position_km': self.current_satellite_state.position_km.tolist(),
            'satellite_velocity_km_s': self.current_satellite_state.velocity_km_s.tolist()
        }
    
    def _get_closest_debris_distance(self) -> float:
        """Get distance to closest debris object."""
        if not self.current_debris_states:
            return float('inf')
        
        sat_pos = self.current_satellite_state.position_km
        min_distance = float('inf')
        
        for debris in self.current_debris_states:
            distance = np.linalg.norm(debris.position_km - sat_pos)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def render(self, mode: str = 'human'):
        """Render the environment (placeholder for future visualization)."""
        if mode == 'human':
            print(f"Step: {self.step_count}, Fuel: {self.current_satellite_state.fuel_remaining_kg:.1f}kg, "
                  f"Closest debris: {self._get_closest_debris_distance():.3f}km")
        
    def close(self):
        """Clean up environment resources."""
        if hasattr(self, 'base_env'):
            self.base_env.close()


# Convenience function for easy environment creation
def create_rl_environment(config_file: str = "scenarios/leo_debris_avoidance.json", **kwargs) -> RLOrbitalEnvironment:
    """
    Create an RL environment with default settings.
    
    Args:
        config_file: Path to scenario configuration
        **kwargs: Additional environment parameters
        
    Returns:
        Configured RLOrbitalEnvironment
    """
    return RLOrbitalEnvironment(config_file=config_file, **kwargs)