#!/usr/bin/env python3
"""
Configuration System for Orbital Avoidance Simulator

This module provides a data-driven approach to configuring:
- Satellite specifications and orbits
- Debris fields and distributions  
- Physics models and propagators
- Reward functions and observation spaces
- Simulation parameters and scenarios

Supports JSON, YAML, and Python dataclass configurations.
"""

from __future__ import annotations

import json
import yaml
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import numpy as np

from orekit_physics import OrbitElements, SpacecraftProperties


class PropagatorType(Enum):
    """Available orbital propagators."""
    OREKIT_KEPLERIAN = "orekit_keplerian"
    OREKIT_NUMERICAL = "orekit_numerical"
    SGP4 = "sgp4"  # Future implementation
    

class RewardType(Enum):
    """Available reward function types."""
    DISTANCE_BASED = "distance_based"
    COLLISION_PROBABILITY = "collision_probability"
    FUEL_EFFICIENCY = "fuel_efficiency"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class OrbitConfig:
    """Configuration for a single orbit."""
    altitude_km: float
    inclination_deg: float = 0.0
    """Backward compatibility wrapper for configuration system.

    The implementation moved to `orbital_avoidance.config.config_system`.
    """
    import warnings
    warnings.warn(
        "Importing 'config_system' from the old location is deprecated. Use 'from orbital_avoidance.config import config_system' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from orbital_avoidance.config.config_system import *  # noqa: F401,F403
                "velocities_normalized": True,
                "safety_radius_normalization": True,
                "time_features": True
            }
    
    def has_advanced_features(self) -> bool:
        """Check if this observation config uses advanced features."""
        return (self.include_collision_probability or
                self.include_uncertainty_ellipsoids or
                self.include_miss_distance or
                self.include_delta_v_remaining or
                self.include_orbital_elements or
                self.enhanced_features is not None or
                self.state_history_length > 1)
    

@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    scenario_name: str
    description: str = ""
    satellites: List[SpacecraftConfig] = field(default_factory=list)
    debris_fields: List[DebrisFieldConfig] = field(default_factory=list)
    static_debris: List[SpacecraftConfig] = field(default_factory=list)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    observations: ObservationConfig = field(default_factory=ObservationConfig)
    random_seed: Optional[int] = None
    

class ConfigurationLoader:
    """Loads and validates simulation configurations."""
    
    @staticmethod
    def load_from_file(filepath: str) -> SimulationConfig:
        """Load configuration from JSON or YAML file."""
        with open(filepath, 'r') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return ConfigurationLoader._dict_to_config(data)
    
    @staticmethod
    def _dict_to_config(data: Dict[str, Any]) -> SimulationConfig:
        """Convert dictionary to SimulationConfig."""
        # Convert satellites
        satellites = []
        for sat_data in data.get('satellites', []):
            orbit_data = sat_data['orbit'].copy()
            # Handle parameter name conversion
            if 'semi_major_axis_km' in orbit_data:
                orbit_data['altitude_km'] = orbit_data.pop('semi_major_axis_km') - 6371.0
            if 'argument_of_perigee_deg' in orbit_data:
                orbit_data['arg_periapsis_deg'] = orbit_data.pop('argument_of_perigee_deg')
            if 'true_anomaly_deg' in orbit_data:
                orbit_data['mean_anomaly_deg'] = orbit_data.pop('true_anomaly_deg')
            
            orbit = OrbitConfig(**orbit_data)
            sat_data_copy = sat_data.copy()
            sat_data_copy['orbit'] = orbit
            # Handle spacecraft parameter conversion
            if 'thrust_capacity_n' in sat_data_copy:
                sat_data_copy['max_thrust_n'] = sat_data_copy.pop('thrust_capacity_n')
            
            # Filter out parameters that SpacecraftConfig doesn't support
            sat_data_filtered = ConfigurationLoader._filter_supported_params(
                sat_data_copy, SpacecraftConfig
            )
            satellites.append(SpacecraftConfig(**sat_data_filtered))
        
        # Convert debris fields
        debris_fields = []
        for field_data in data.get('debris_fields', []):
            # Filter out nested complex objects that aren't supported yet
            field_data_simplified = field_data.copy()
            
            # Remove complex nested structures for now (can be added later)
            unsupported_keys = ['covariance_matrix', 'dynamics']
            for key in unsupported_keys:
                if key in field_data_simplified:
                    field_data_simplified.pop(key)
            
            # Filter supported parameters
            field_data_filtered = ConfigurationLoader._filter_supported_params(
                field_data_simplified, DebrisFieldConfig
            )
            debris_fields.append(DebrisFieldConfig(**field_data_filtered))
        
        # Convert static debris
        static_debris = []
        for debris_data in data.get('static_debris', []):
            orbit_data = debris_data['orbit'].copy()
            # Handle parameter name conversion
            if 'semi_major_axis_km' in orbit_data:
                orbit_data['altitude_km'] = orbit_data.pop('semi_major_axis_km') - 6371.0
            if 'argument_of_perigee_deg' in orbit_data:
                orbit_data['arg_periapsis_deg'] = orbit_data.pop('argument_of_perigee_deg')
            if 'true_anomaly_deg' in orbit_data:
                orbit_data['mean_anomaly_deg'] = orbit_data.pop('true_anomaly_deg')
                
            orbit = OrbitConfig(**orbit_data)
            debris_data_copy = debris_data.copy()
            debris_data_copy['orbit'] = orbit
            # Handle spacecraft parameter conversion
            if 'thrust_capacity_n' in debris_data_copy:
                debris_data_copy['max_thrust_n'] = debris_data_copy.pop('thrust_capacity_n')
            
            # Remove legacy parameters and filter supported ones
            debris_data_copy.pop('size_m', None)  # Legacy parameter
            debris_data_copy.pop('covariance_matrix', None)  # Complex structure
            
            # Filter supported parameters
            debris_data_filtered = ConfigurationLoader._filter_supported_params(
                debris_data_copy, SpacecraftConfig
            )
            static_debris.append(SpacecraftConfig(**debris_data_filtered))
            
        # Convert debris (legacy field name handling)
        for debris_data in data.get('debris', []):
            orbit_data = debris_data['orbit'].copy()
            # Handle parameter name conversion
            if 'semi_major_axis_km' in orbit_data:
                orbit_data['altitude_km'] = orbit_data.pop('semi_major_axis_km') - 6371.0
            if 'argument_of_perigee_deg' in orbit_data:
                orbit_data['arg_periapsis_deg'] = orbit_data.pop('argument_of_perigee_deg')
            if 'true_anomaly_deg' in orbit_data:
                orbit_data['mean_anomaly_deg'] = orbit_data.pop('true_anomaly_deg')
                
            orbit = OrbitConfig(**orbit_data)
            debris_data_copy = debris_data.copy()
            debris_data_copy['orbit'] = orbit
            # Handle spacecraft parameter conversion
            if 'thrust_capacity_n' in debris_data_copy:
                debris_data_copy['max_thrust_n'] = debris_data_copy.pop('thrust_capacity_n')
            # Remove parameters not supported by SpacecraftConfig
            debris_data_copy.pop('size_m', None)
            static_debris.append(SpacecraftConfig(**debris_data_copy))
        
        # Convert sub-configurations with filtering
        physics_data = data.get('physics', {})
        physics_filtered = ConfigurationLoader._filter_supported_params(physics_data, PhysicsConfig)
        physics = PhysicsConfig(**physics_filtered)
        
        # Handle rewards configuration conversion
        rewards_data = data.get('rewards', {})
        if 'collision' in rewards_data and 'penalty' in rewards_data['collision']:
            # Extract from nested structure
            reward_params = {
                'collision_penalty': rewards_data['collision'].get('penalty', -1000.0),
                'fuel_cost_weight': rewards_data.get('fuel', {}).get('efficiency_weight', 0.1)
            }
            rewards_filtered = ConfigurationLoader._filter_supported_params(reward_params, RewardConfig)
            rewards = RewardConfig(**rewards_filtered)
        else:
            # Use flat structure or defaults
            rewards_filtered = ConfigurationLoader._filter_supported_params(rewards_data, RewardConfig)
            rewards = RewardConfig(**rewards_filtered)
        
        observations_data = data.get('observations', {})
        observations_filtered = ConfigurationLoader._filter_supported_params(observations_data, ObservationConfig)
        observations = ObservationConfig(**observations_filtered)
        
        return SimulationConfig(
            scenario_name=data.get('scenario_name', data.get('name', 'Unknown Scenario')),
            description=data.get('description', ''),
            satellites=satellites,
            debris_fields=debris_fields,
            static_debris=static_debris,
            physics=physics,
            rewards=rewards,
            observations=observations,
            random_seed=data.get('random_seed')
        )
    
    @staticmethod
    def _filter_supported_params(data: Dict[str, Any], config_class) -> Dict[str, Any]:
        """Filter dictionary to only include parameters supported by the config class."""
        import inspect
        
        # Get the parameter names for the dataclass constructor
        signature = inspect.signature(config_class)
        supported_params = set(signature.parameters.keys())
        
        # Filter the data to only include supported parameters
        filtered_data = {k: v for k, v in data.items() if k in supported_params}
        
        # Log any ignored parameters (optional, for debugging)
        ignored_params = set(data.keys()) - supported_params
        if ignored_params:
            print(f"Warning: Ignoring unsupported parameters for {config_class.__name__}: {ignored_params}")
        
        return filtered_data
    
    @staticmethod
    def save_to_file(config: SimulationConfig, filepath: str):
        """Save configuration to JSON or YAML file."""
        data = asdict(config)
        
        with open(filepath, 'w') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                json.dump(data, f, indent=2)
    
    @staticmethod
    def validate_config(config: SimulationConfig) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Check that we have at least one satellite
        if not config.satellites:
            errors.append("Configuration must include at least one satellite")
        
        # Validate orbits
        for i, sat in enumerate(config.satellites):
            if sat.orbit.altitude_km < 100:
                errors.append(f"Satellite {i} altitude too low: {sat.orbit.altitude_km} km")
            if sat.orbit.altitude_km > 50000:
                errors.append(f"Satellite {i} altitude too high: {sat.orbit.altitude_km} km")
        
        # Validate physics parameters
        if config.physics.time_step_seconds <= 0:
            errors.append("Physics time step must be positive")
        
        if config.physics.collision_threshold_km <= 0:
            errors.append("Collision threshold must be positive")
            
        return errors


class DebrisGenerator:
    """Generates debris objects from DebrisFieldConfig."""
    
    @staticmethod
    def generate_debris_field(field_config: DebrisFieldConfig, 
                            seed: Optional[int] = None) -> List[SpacecraftConfig]:
        """Generate debris objects according to configuration."""
        if seed is not None:
            np.random.seed(seed)
        
        debris_objects = []
        
        for i in range(field_config.num_objects):
            # Generate orbital parameters
            altitude = np.random.uniform(*field_config.altitude_range_km)
            inclination = np.random.uniform(*field_config.inclination_range_deg)
            eccentricity = np.random.uniform(*field_config.eccentricity_range)
            raan = np.random.uniform(0, 360)
            arg_periapsis = np.random.uniform(0, 360)
            mean_anomaly = np.random.uniform(0, 360)
            
            # Generate physical parameters
            mass = np.random.uniform(*field_config.mass_range_kg)
            size = np.random.uniform(*field_config.size_range_m)
            
            orbit = OrbitConfig(
                altitude_km=altitude,
                inclination_deg=inclination,
                eccentricity=eccentricity,
                raan_deg=raan,
                arg_periapsis_deg=arg_periapsis,
                mean_anomaly_deg=mean_anomaly
            )
            
            debris = SpacecraftConfig(
                name=f"Debris_{i}",
                orbit=orbit,
                mass_kg=mass,
                drag_area_m2=size,
                collision_radius_m=size/2,
                is_controllable=False,  # Debris cannot be controlled
                max_thrust_n=0.0,
                fuel_capacity_kg=0.0
            )
            
            debris_objects.append(debris)
        
        return debris_objects


# Example usage and default configurations
def create_default_leo_scenario() -> SimulationConfig:
    """Create a default LEO collision avoidance scenario."""
    
    # Main satellite
    satellite = SpacecraftConfig(
        name="MainSatellite",
        orbit=OrbitConfig(altitude_km=400, inclination_deg=51.6),
        mass_kg=1000,
        is_controllable=True,
        max_thrust_n=1.0,
        fuel_capacity_kg=100
    )
    
    # Debris field
    debris_field = DebrisFieldConfig(
        num_objects=50,
        altitude_range_km=[380, 420],
        inclination_range_deg=[45, 60],
        mass_range_kg=[1, 100],
        size_range_m=[0.1, 2.0]
    )
    
    return SimulationConfig(
        scenario_name="LEO_Debris_Avoidance",
        description="Standard LEO debris avoidance scenario with 50 debris objects",
        satellites=[satellite],
        debris_fields=[debris_field],
        physics=PhysicsConfig(
            time_step_seconds=60,
            simulation_duration_seconds=3600,  # 1 hour
            collision_threshold_km=0.001  # 1 meter
        ),
        rewards=RewardConfig(
            collision_penalty=-1000,
            fuel_cost_weight=0.1,
            safety_distance_km=1.0
        )
    )


if __name__ == "__main__":
    # Create and save a default scenario
    default_scenario = create_default_leo_scenario()
    
    # Save to JSON
    ConfigurationLoader.save_to_file(default_scenario, "scenarios/default_leo.json")
    
    # Validate
    errors = ConfigurationLoader.validate_config(default_scenario)
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration is valid!")
    
    print(f"Scenario: {default_scenario.scenario_name}")
    print(f"Satellites: {len(default_scenario.satellites)}")
    print(f"Debris fields: {len(default_scenario.debris_fields)}")