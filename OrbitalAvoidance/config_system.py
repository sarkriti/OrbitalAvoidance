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
    eccentricity: float = 0.0
    raan_deg: float = 0.0  # Right Ascension of Ascending Node
    arg_periapsis_deg: float = 0.0  # Argument of Periapsis
    mean_anomaly_deg: float = 0.0
    epoch_mjd: float = 59945.0  # Default epoch (2023-01-01)
    
    def to_orbit_elements(self) -> OrbitElements:
        """Convert to OrbitElements for physics engine."""
        return OrbitElements(
            semi_major_axis_km=6371.0 + self.altitude_km,
            eccentricity=self.eccentricity,
            inclination_deg=self.inclination_deg,
            raan_deg=self.raan_deg,
            arg_periapsis_deg=self.arg_periapsis_deg,
            mean_anomaly_deg=self.mean_anomaly_deg,
            epoch_mjd=self.epoch_mjd
        )


@dataclass 
class SpacecraftConfig:
    """Configuration for a spacecraft (satellite or debris)."""
    name: str
    orbit: OrbitConfig
    mass_kg: float = 1000.0
    drag_area_m2: float = 10.0
    drag_coefficient: float = 2.2
    radiation_area_m2: float = 10.0
    radiation_coefficient: float = 1.8
    is_controllable: bool = True  # Can apply thrust
    max_thrust_n: float = 1.0  # Maximum thrust in Newtons
    fuel_capacity_kg: float = 100.0
    collision_radius_m: float = 10.0  # For collision detection
    
    def to_spacecraft_properties(self) -> SpacecraftProperties:
        """Convert to SpacecraftProperties for physics engine."""
        return SpacecraftProperties(
            mass_kg=self.mass_kg,
            drag_area_m2=self.drag_area_m2,
            drag_coefficient=self.drag_coefficient,
            radiation_area_m2=self.radiation_area_m2,
            radiation_coefficient=self.radiation_coefficient
        )


@dataclass
class DebrisFieldConfig:
    """Configuration for generating debris fields."""
    num_objects: int
    altitude_range_km: List[float]  # [min, max]
    inclination_range_deg: List[float] = field(default_factory=lambda: [0, 180])
    eccentricity_range: List[float] = field(default_factory=lambda: [0, 0.1])
    mass_range_kg: List[float] = field(default_factory=lambda: [1, 1000])
    size_range_m: List[float] = field(default_factory=lambda: [0.1, 10])
    distribution_type: str = "uniform"  # "uniform", "gaussian", "realistic"
    
    
@dataclass
class PhysicsConfig:
    """Configuration for physics models and propagation."""
    propagator_type: str = "orekit_keplerian"
    time_step_seconds: float = 60.0
    simulation_duration_seconds: float = 86400.0  # 24 hours
    include_drag: bool = True
    include_solar_radiation: bool = True
    include_j2_perturbation: bool = True
    collision_detection_method: str = "spherical"  # "spherical", "ellipsoidal", "mesh"
    collision_threshold_km: float = 0.01  # 10 meters
    

@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    reward_type: str = "distance_based"
    collision_penalty: float = -1000.0
    fuel_cost_weight: float = 0.1
    safety_distance_km: float = 1.0  # Minimum safe distance
    close_approach_penalty_weight: float = 10.0
    mission_success_reward: float = 100.0
    

@dataclass
class ObservationConfig:
    """Configuration for observation space."""
    include_relative_positions: bool = True
    include_relative_velocities: bool = True
    include_fuel_levels: bool = True
    include_time_to_collision: bool = True
    max_tracked_objects: int = 10  # Limit observation space size
    observation_radius_km: float = 100.0  # Only observe objects within this radius
    

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
            orbit = OrbitConfig(**sat_data['orbit'])
            sat_data_copy = sat_data.copy()
            sat_data_copy['orbit'] = orbit
            satellites.append(SpacecraftConfig(**sat_data_copy))
        
        # Convert debris fields
        debris_fields = []
        for field_data in data.get('debris_fields', []):
            debris_fields.append(DebrisFieldConfig(**field_data))
        
        # Convert static debris
        static_debris = []
        for debris_data in data.get('static_debris', []):
            orbit = OrbitConfig(**debris_data['orbit'])
            debris_data_copy = debris_data.copy()
            debris_data_copy['orbit'] = orbit
            static_debris.append(SpacecraftConfig(**debris_data_copy))
        
        # Convert sub-configurations
        physics = PhysicsConfig(**data.get('physics', {}))
        rewards = RewardConfig(**data.get('rewards', {}))
        observations = ObservationConfig(**data.get('observations', {}))
        
        return SimulationConfig(
            scenario_name=data['scenario_name'],
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