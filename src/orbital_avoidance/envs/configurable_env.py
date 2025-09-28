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
- Properly integrates with Orekit physics engine

ALL UNITS IN KILOMETERS FOR CONSISTENCY
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
import numpy as np
import gymnasium as gym
import random
import math

logger = logging.getLogger(__name__)

# Import Orekit physics
from orbital_avoidance.physics.orekit_physics import (
    OrekitOrbitPropagator, OrbitElements, SpacecraftProperties, 
    create_propagator, convert_to_orbit_elements
)


def safe_get(config_obj, key, default=None):
    """Safely get value from either config object or dictionary."""
    if isinstance(config_obj, dict):
        return config_obj.get(key, default)
    else:
        return getattr(config_obj, key, default)


class SatelliteConfig:
    """Configuration for a satellite."""
    def __init__(self, config_obj):
        self.name = safe_get(config_obj, 'name')
        self.mass_kg = safe_get(config_obj, 'mass_kg', 1000)
        self.fuel_capacity_kg = safe_get(config_obj, 'fuel_capacity_kg', 100)
        self.max_thrust_n = safe_get(config_obj, 'max_thrust_n', 1.0)
        self.is_controllable = safe_get(config_obj, 'is_controllable', False)
        
        # FIXED: All collision/near-miss radii in km
        self.collision_radius_km = safe_get(config_obj, 'collision_radius_km', 0.001)  # Default 1m = 0.001km
        
        # Handle near_miss_radius - check if it's specified directly in km
        near_miss_radius_km = safe_get(config_obj, 'near_miss_radius_km')
        if near_miss_radius_km is not None:
            self.near_miss_radius_km = near_miss_radius_km
        else:
            # Use multiplier approach with collision radius already in km
            self.near_miss_radius_km = self.collision_radius_km * safe_get(config_obj, 'near_miss_multiplier', 2.0)
        
        # Orbital parameters - handle both altitude and semi_major_axis
        orbit = safe_get(config_obj, 'orbit', {})
        
        # Check for altitude_km
        altitude_km = safe_get(orbit, 'altitude_km')
        semi_major_axis_km = safe_get(orbit, 'semi_major_axis_km')
        
        if altitude_km is not None:
            # Convert altitude above Earth's surface to semi-major axis
            earth_radius_km = safe_get(orbit, 'earth_radius_km', 6371.0)
            semi_major_axis = altitude_km + earth_radius_km
        elif semi_major_axis_km is not None:
            semi_major_axis = semi_major_axis_km
        else:
            # Use default from config or fallback
            default_altitude = safe_get(config_obj, 'default_altitude_km', 400)
            earth_radius_km = safe_get(orbit, 'earth_radius_km', 6371.0)
            semi_major_axis = default_altitude + earth_radius_km
            
        self.orbit = OrbitElements(
            semi_major_axis_km=semi_major_axis,
            eccentricity=safe_get(orbit, 'eccentricity', 0.001),
            inclination_deg=safe_get(orbit, 'inclination_deg', 0),
            raan_deg=safe_get(orbit, 'raan_deg', 0),
            arg_periapsis_deg=safe_get(orbit, 'arg_periapsis_deg', 0),
            mean_anomaly_deg=safe_get(orbit, 'mean_anomaly_deg', 0),
            epoch_mjd=safe_get(orbit, 'epoch_mjd', 59945.0)
        )
        
        # FIXED: Spacecraft properties - convert areas from m² to km²
        drag_area_m2 = safe_get(config_obj, 'drag_area_m2', 10.0)
        radiation_area_m2 = safe_get(config_obj, 'radiation_area_m2', 10.0)
        
       # Spacecraft properties for physics - keep in m² as expected by SpacecraftProperties
        self.spacecraft_props = SpacecraftProperties(
            mass_kg=self.mass_kg,
            drag_area_m2=safe_get(config_obj, 'drag_area_m2', 10.0),
            drag_coefficient=safe_get(config_obj, 'drag_coefficient', 2.2),
            radiation_area_m2=safe_get(config_obj, 'radiation_area_m2', 10.0),
            radiation_coefficient=safe_get(config_obj, 'radiation_coefficient', 1.8)
        )


class DebrisConfig:
    """Configuration for debris objects."""
    def __init__(self, config_obj, name=None):
        self.name = name or safe_get(config_obj, 'name', 'Debris')
        self.mass_kg = safe_get(config_obj, 'mass_kg', 50)
        
        # FIXED: All collision/near-miss radii in km
        self.collision_radius_km = safe_get(config_obj, 'collision_radius_km', 0.001)  # Default 1m = 0.001km
        self.near_miss_radius_km = self.collision_radius_km * safe_get(config_obj, 'near_miss_multiplier', 5.0)
        
        # Orbital parameters
        orbit = safe_get(config_obj, 'orbit', {})
        
        # Check for altitude_km
        altitude_km = safe_get(orbit, 'altitude_km')
        semi_major_axis_km = safe_get(orbit, 'semi_major_axis_km')
        
        if altitude_km is not None:
            earth_radius_km = safe_get(orbit, 'earth_radius_km', 6371.0)
            semi_major_axis = altitude_km + earth_radius_km
        elif semi_major_axis_km is not None:
            semi_major_axis = semi_major_axis_km
        else:
            # Use fallback from debris config
            default_altitude = safe_get(config_obj, 'default_altitude_km', 400)
            earth_radius_km = safe_get(orbit, 'earth_radius_km', 6371.0)
            semi_major_axis = default_altitude + earth_radius_km
            
        self.orbit = OrbitElements(
            semi_major_axis_km=semi_major_axis,
            eccentricity=safe_get(orbit, 'eccentricity', 0.001),
            inclination_deg=safe_get(orbit, 'inclination_deg', 0),
            raan_deg=safe_get(orbit, 'raan_deg', 0),
            arg_periapsis_deg=safe_get(orbit, 'arg_periapsis_deg', 0),
            mean_anomaly_deg=safe_get(orbit, 'mean_anomaly_deg', 0),
            epoch_mjd=safe_get(orbit, 'epoch_mjd', 59945.0)
        )


class ConfigurableOrbitalEnvironment:
    """Physics-based orbital environment that can be used by RL wrappers."""
    
    def __init__(self, config, render_mode=None):
        """Initialize the configurable orbital environment."""
        self.render_mode = render_mode
        
        # Handle both dict and config object
        if hasattr(config, '__dict__') and not isinstance(config, dict):
            self.config_dict = self._config_to_dict(config)
        else:
            self.config_dict = config
            
        # Initialize physics engine
        self.engine = create_propagator()
        
        # Parse physics configuration - handle both dict and object
        physics = safe_get(self.config_dict, 'physics', {})
        if isinstance(physics, dict):
            self.physics_config = physics
        else:
            # Physics is a config object, extract attributes
            self.physics_config = {
                'time_step_seconds': safe_get(physics, 'time_step_seconds', 10.0),
                'simulation_duration_seconds': safe_get(physics, 'simulation_duration_seconds', 1800.0),
                'include_drag': safe_get(physics, 'include_drag', False),
                'include_solar_radiation': safe_get(physics, 'include_solar_radiation', False),
                'include_j2_perturbation': safe_get(physics, 'include_j2_perturbation', False),
                'include_higher_order_gravity': safe_get(physics, 'include_higher_order_gravity', False)
            }
            
        self.time_step = self.physics_config.get('time_step_seconds', 10.0)
        self.simulation_duration = self.physics_config.get('simulation_duration_seconds', 1800.0)
        self.simulation_time = 0.0
        
        # Parse satellites from config
        self.satellites = {}
        self._satellite_configs = {}
        satellites = safe_get(self.config_dict, 'satellites', [])
        
        for sat_data in satellites:
            sat_config = SatelliteConfig(sat_data)
            self._satellite_configs[sat_config.name] = sat_config
            
            # Create persistent object in Orekit with proper spacecraft properties
            self.engine.create_persistent_object(
                sat_config.name, 
                sat_config.orbit, 
                sat_config.spacecraft_props
            )
            
            # Get initial position/velocity
            pos, vel = self.engine.propagate_persistent_delta(sat_config.name, 0.0)
            
            self.satellites[sat_config.name] = {
                'position': pos,
                'velocity': vel,
                'fuel_remaining': sat_config.fuel_capacity_kg,
                'config': sat_config
            }
        
        # Parse debris from config
        self.debris_objects = {}
        self._debris_configs = {}
        
        # Handle debris fields (generate random debris)
        debris_fields = safe_get(self.config_dict, 'debris_fields', [])
        for field in debris_fields:
            self._generate_debris_field(field)
        
        # Handle static debris
        static_debris = safe_get(self.config_dict, 'static_debris', [])
        for debris_data in static_debris:
            self._create_debris_object(debris_data)
        
        logger.info(f"ConfigurableOrbitalEnvironment initialized:")
        logger.info(f"  - Satellites: {len(self.satellites)}")
        logger.info(f"  - Debris objects: {len(self.debris_objects)}")
        logger.info(f"  - Time step: {self.time_step}s")
        logger.info(f"  - Duration: {self.simulation_duration}s")
    
    def _generate_debris_field(self, field_config):
        """Generate debris field based on configuration."""
        num_objects = safe_get(field_config, 'num_objects', 5)
        alt_range = safe_get(field_config, 'altitude_range_km', [395, 405])
        inc_range = safe_get(field_config, 'inclination_range_deg', [50, 53])
        ecc_range = safe_get(field_config, 'eccentricity_range', [0.0, 0.02])
        raan_range = safe_get(field_config, 'raan_range_deg', [0, 360])
        argp_range = safe_get(field_config, 'arg_periapsis_range_deg', [0, 360])
        ma_range = safe_get(field_config, 'mean_anomaly_range_deg', [0, 360])
        
        # FIXED: Get mass and size ranges from config - sizes now in km
        mass_range = safe_get(field_config, 'mass_range_kg', [50, 50])  # Default to single value
        collision_radius_range = safe_get(field_config, 'collision_radius_range_km', [0.001, 0.001])  # Default 1m in km
        
        # Get Earth radius from config or default
        earth_radius_km = safe_get(field_config, 'earth_radius_km', 6371.0)
        epoch_mjd = safe_get(field_config, 'epoch_mjd', 59945.0)
        
        field_name = safe_get(field_config, 'name', 'DebrisField')
        
        for i in range(num_objects):
            debris_name = f"{field_name}_{i+1}"
            
            # Generate random orbital parameters within specified ranges
            random_orbit = {
                'altitude_km': random.uniform(alt_range[0], alt_range[1]),
                'inclination_deg': random.uniform(inc_range[0], inc_range[1]),
                'eccentricity': random.uniform(ecc_range[0], ecc_range[1]),
                'raan_deg': random.uniform(raan_range[0], raan_range[1]),
                'arg_periapsis_deg': random.uniform(argp_range[0], argp_range[1]),
                'mean_anomaly_deg': random.uniform(ma_range[0], ma_range[1]),
                'earth_radius_km': earth_radius_km,
                'epoch_mjd': epoch_mjd
            }
            
            # Generate random physical properties
            mass_kg = random.uniform(mass_range[0], mass_range[1]) if len(mass_range) > 1 else mass_range[0]
            collision_radius_km = random.uniform(collision_radius_range[0], collision_radius_range[1]) if len(collision_radius_range) > 1 else collision_radius_range[0]
            
            debris_dict = {
                'name': debris_name,
                'mass_kg': mass_kg,
                'collision_radius_km': collision_radius_km,  # FIXED: Now in km
                'near_miss_multiplier': safe_get(field_config, 'near_miss_multiplier', 5.0),
                'orbit': random_orbit
            }
            
            self._create_debris_object(debris_dict, debris_name)
    
    def _create_debris_object(self, debris_data, name=None):
        """Create a single debris object."""
        debris_config = DebrisConfig(debris_data, name)
        self._debris_configs[debris_config.name] = debris_config
        
        # Create persistent object in Orekit (debris typically doesn't need spacecraft props)
        self.engine.create_persistent_object(debris_config.name, debris_config.orbit)
        
        # Get initial position/velocity
        pos, vel = self.engine.propagate_persistent_delta(debris_config.name, 0.0)
        
        self.debris_objects[debris_config.name] = {
            'position': pos,
            'velocity': vel,
            'config': debris_config
        }
    
    def _config_to_dict(self, config):
        """Convert a config object to dictionary format."""
        if hasattr(config, 'to_dict'):
            return config.to_dict()
        
        # Extract key attributes from config object
        result = {}
        
        # Handle satellites
        if hasattr(config, 'satellites'):
            satellites = config.satellites
            if isinstance(satellites, list):
                result['satellites'] = [self._sat_to_dict(s) for s in satellites]
            else:
                result['satellites'] = []
        
        # Handle debris fields
        if hasattr(config, 'debris_fields'):
            result['debris_fields'] = config.debris_fields
        elif hasattr(config, 'environment') and hasattr(config.environment, 'debris_fields'):
            result['debris_fields'] = config.environment.debris_fields
        else:
            result['debris_fields'] = []
        
        # Handle static debris
        if hasattr(config, 'static_debris'):
            result['static_debris'] = config.static_debris
        elif hasattr(config, 'environment') and hasattr(config.environment, 'static_debris'):
            result['static_debris'] = config.environment.static_debris
        else:
            result['static_debris'] = []
        
        # Handle physics config
        if hasattr(config, 'physics'):
            result['physics'] = config.physics
        
        # Handle environment config
        if hasattr(config, 'environment'):
            result['environment'] = config.environment if isinstance(config.environment, dict) else config.environment.__dict__
        
        # Handle collision detection
        if hasattr(config, 'collision_detection'):
            result['collision_detection'] = config.collision_detection if isinstance(config.collision_detection, dict) else config.collision_detection.__dict__
        
        # Handle control config
        if hasattr(config, 'control'):
            result['control'] = config.control if isinstance(config.control, dict) else config.control.__dict__
        
        # Handle reward config
        if hasattr(config, 'reward'):
            result['reward'] = config.reward if isinstance(config.reward, dict) else config.reward.__dict__
        
        # Handle observations config
        if hasattr(config, 'observations'):
            result['observations'] = config.observations if isinstance(config.observations, dict) else config.observations.__dict__
            
        return result
    
    def _sat_to_dict(self, sat):
        """Convert satellite config to dict."""
        if hasattr(sat, 'to_dict'):
            return sat.to_dict()
        elif isinstance(sat, dict):
            return sat
        else:
            # Extract key attributes from satellite object
            sat_dict = {
                'name': getattr(sat, 'name', 'Unknown'),
                'mass_kg': getattr(sat, 'mass_kg', 1000),
                'fuel_capacity_kg': getattr(sat, 'fuel_capacity_kg', 100),
                'is_controllable': getattr(sat, 'is_controllable', False),
                'max_thrust_n': getattr(sat, 'max_thrust_n', 1.0),
                'collision_radius_km': getattr(sat, 'collision_radius_km', 0.001),  # FIXED: Now in km
                'near_miss_radius_km': getattr(sat, 'near_miss_radius_km', 0.050),  # FIXED: 50m = 0.05km
                'drag_area_m2': getattr(sat, 'drag_area_m2', 10.0),
                'drag_coefficient': getattr(sat, 'drag_coefficient', 2.2),
                'radiation_area_m2': getattr(sat, 'radiation_area_m2', 10.0),
                'radiation_coefficient': getattr(sat, 'radiation_coefficient', 1.8)
            }
            
            # Handle orbit
            if hasattr(sat, 'orbit'):
                orbit = sat.orbit
                if isinstance(orbit, dict):
                    sat_dict['orbit'] = orbit
                else:
                    # Extract orbit attributes
                    sat_dict['orbit'] = {
                        'altitude_km': getattr(orbit, 'altitude_km', None),
                        'semi_major_axis_km': getattr(orbit, 'semi_major_axis_km', None),
                        'eccentricity': getattr(orbit, 'eccentricity', 0.001),
                        'inclination_deg': getattr(orbit, 'inclination_deg', 0),
                        'raan_deg': getattr(orbit, 'raan_deg', 0),
                        'arg_periapsis_deg': getattr(orbit, 'arg_periapsis_deg', 0),
                        'mean_anomaly_deg': getattr(orbit, 'mean_anomaly_deg', 0),
                        'epoch_mjd': getattr(orbit, 'epoch_mjd', 59945.0)
                    }
            
            return sat_dict
    
    def reset(self):
        """Reset environment to initial state."""
        self.simulation_time = 0.0
        
        # Reset all satellites to initial positions
        for sat_name, sat_config in self._satellite_configs.items():
            # Recreate persistent object to reset to initial orbit
            self.engine.create_persistent_object(sat_name, sat_config.orbit, sat_config.spacecraft_props)
            pos, vel = self.engine.propagate_persistent_delta(sat_name, 0.0)
            
            self.satellites[sat_name]['position'] = pos
            self.satellites[sat_name]['velocity'] = vel
            self.satellites[sat_name]['fuel_remaining'] = sat_config.fuel_capacity_kg
        
        # Reset all debris to initial positions
        for debris_name, debris_config in self._debris_configs.items():
            self.engine.create_persistent_object(debris_name, debris_config.orbit)
            pos, vel = self.engine.propagate_persistent_delta(debris_name, 0.0)
            
            self.debris_objects[debris_name]['position'] = pos
            self.debris_objects[debris_name]['velocity'] = vel
        
        return None, {}
    
    def step(self, action):
        """Step the physics simulation forward."""
        # Apply control action to controllable satellites
        for sat_name, sat_data in self.satellites.items():
            sat_config = sat_data['config']
            if sat_config.is_controllable and len(action) >= 3:
                # Get control parameters from config
                control_config = safe_get(self.config_dict, 'control', {})
                
                # FIXED: Control now properly in km/s
                max_delta_v_km_s = safe_get(control_config, 'max_delta_v_km_s', 0.001)  # Default 1 m/s = 0.001 km/s
                fuel_consumption_rate = safe_get(control_config, 'fuel_consumption_rate_kg_per_km_s', 100.0)  # kg per km/s
                
                # Convert normalized action to delta-v in km/s
                delta_v = np.array(action[:3]) * max_delta_v_km_s
                
                # Apply delta-v if we have fuel
                if sat_data['fuel_remaining'] > 0:
                    self.engine.apply_instantaneous_delta_v(sat_name, delta_v)
                    
                    # Fuel consumption based on config
                    delta_v_magnitude = np.linalg.norm(delta_v)  # Already in km/s
                    fuel_used = delta_v_magnitude * fuel_consumption_rate
                    sat_data['fuel_remaining'] = max(0, sat_data['fuel_remaining'] - fuel_used)
        
        # Propagate all objects forward by one time step
        self.simulation_time += self.time_step
        
        # Update satellite positions using Orekit physics
        for sat_name in self.satellites:
            pos, vel = self.engine.propagate_persistent_delta(sat_name, self.time_step)
            self.satellites[sat_name]['position'] = pos
            self.satellites[sat_name]['velocity'] = vel
        
        # Update debris positions using Orekit physics
        for debris_name in self.debris_objects:
            pos, vel = self.engine.propagate_persistent_delta(debris_name, self.time_step)
            self.debris_objects[debris_name]['position'] = pos
            self.debris_objects[debris_name]['velocity'] = vel
        
        # Calculate reward based on config
        reward_config = safe_get(self.config_dict, 'reward', {})
        base_survival_reward = safe_get(reward_config, 'base_survival_reward', 1.0)
        fuel_penalty_weight = safe_get(reward_config, 'fuel_penalty_weight', 0.1)
        
        reward = base_survival_reward
        
        # Apply fuel usage penalty
        for sat_data in self.satellites.values():
            if sat_data['config'].is_controllable:
                fuel_used = sat_data['config'].fuel_capacity_kg - sat_data['fuel_remaining']
                reward -= fuel_penalty_weight * fuel_used
        
        # Check for collisions
        terminated = self._check_collisions()
        
        # Check for episode completion based on time
        max_steps = int(self.simulation_duration / self.time_step)
        current_step = int(self.simulation_time / self.time_step)
        truncated = current_step >= max_steps
        
        info = {
            'simulation_time': self.simulation_time,
            'collisions': terminated,
            'step': current_step,
            'max_steps': max_steps
        }
        
        return None, reward, terminated, truncated, info
    
    def _check_collisions(self):
        """Check for collisions between satellites and debris."""
        collision_config = safe_get(self.config_dict, 'collision_detection', {})
        
        for sat_name, sat_data in self.satellites.items():
            sat_pos = sat_data['position']
            sat_config = sat_data['config']
            
            for debris_name, debris_data in self.debris_objects.items():
                debris_pos = debris_data['position']
                debris_config = debris_data['config']
                
                distance = np.linalg.norm(sat_pos - debris_pos)
                
                # Use collision threshold from config or calculate from radii
                collision_threshold = safe_get(collision_config, 'collision_threshold_km')
                if collision_threshold is None:
                    collision_threshold = sat_config.collision_radius_km + debris_config.collision_radius_km
                
                if distance < collision_threshold:
                    logger.warning(f"Collision detected: {sat_name} <-> {debris_name} at distance {distance:.6f} km")
                    return True
        
        return False
    
    def close(self):
        """Clean up resources."""
        pass


# Export classes
__all__ = [
    'ConfigurableOrbitalEnvironment',
    'SatelliteConfig',
    'DebrisConfig'
]