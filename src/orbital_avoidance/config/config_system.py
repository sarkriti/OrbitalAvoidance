"""Configuration system (migrated).

Original: OrbitalAvoidance/config_system.py
Now under orbital_avoidance.config.config_system

ALL UNITS IN KILOMETERS FOR CONSISTENCY
"""

from __future__ import annotations

import json
import yaml
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import numpy as np

from orbital_avoidance.physics.orekit_physics import OrbitElements, SpacecraftProperties


class PropagatorType(Enum):
	OREKIT_KEPLERIAN = "orekit_keplerian"
	OREKIT_NUMERICAL = "orekit_numerical"
	SGP4 = "sgp4"


class RewardType(Enum):
	DISTANCE_BASED = "distance_based"
	COLLISION_PROBABILITY = "collision_probability"
	FUEL_EFFICIENCY = "fuel_efficiency"
	MULTI_OBJECTIVE = "multi_objective"


@dataclass
class OrbitConfig:
	altitude_km: float
	inclination_deg: float = 0.0
	eccentricity: float = 0.0
	raan_deg: float = 0.0
	arg_periapsis_deg: float = 0.0
	mean_anomaly_deg: float = 0.0
	epoch_mjd: float = 59945.0

	def to_orbit_elements(self) -> OrbitElements:
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
class CollisionDetectionConfig:
    method: str = "spherical_with_prediction"
    collision_threshold_km: float = 0.005
    conjunction_screening_km: float = 0.075
    prediction_horizon_seconds: float = 300.0

@dataclass
class ControlConfig:
    max_delta_v_km_s: float = 0.001
    fuel_consumption_rate_kg_per_km_s: float = 100.0
    thrust_vectoring: bool = True
    reaction_wheel_torque_nm: float = 0.1

@dataclass
class SpacecraftConfig:
	name: str
	orbit: OrbitConfig
	mass_kg: float = 1000.0
	drag_area_m2: float = 10.0
	drag_coefficient: float = 2.2
	radiation_area_m2: float = 10.0
	radiation_coefficient: float = 1.8
	is_controllable: bool = True
	max_thrust_n: float = 1.0
	fuel_capacity_kg: float = 100.0
	
	# FIXED: All collision/near-miss radii now in km with explicit units
	collision_radius_km: float = 0.001  # Default 1m = 0.001km
	near_miss_radius_km: float = 0.050  # Default 50m = 0.050km
	
	drag_coefficient_uncertainty: Optional[float] = None
	attitude_dependent_drag: bool = False
	maneuver_efficiency: float = 1.0
	delta_v_budget_ms: Optional[float] = None
	position_uncertainty_km: float = 0.0
	velocity_uncertainty_kms: float = 0.0
	moment_of_inertia_kgm2: Optional[float] = None
	solar_panel_area_m2: Optional[float] = None

	def __post_init__(self):
		if self.drag_coefficient_uncertainty is None and self.attitude_dependent_drag:
			self.drag_coefficient_uncertainty = self.drag_coefficient * 0.1
		if self.delta_v_budget_ms is None and self.is_controllable:
			self.delta_v_budget_ms = self.fuel_capacity_kg * 10.0
		if self.moment_of_inertia_kgm2 is None:
			self.moment_of_inertia_kgm2 = self.mass_kg * 0.5

	def to_spacecraft_properties(self) -> SpacecraftProperties:
		# FIXED: Convert areas from m² to km²
		return SpacecraftProperties(
			mass_kg=self.mass_kg,
			drag_area_km2=self.drag_area_m2 / 1e6,  # Convert m² to km²
			drag_coefficient=self.drag_coefficient,
			radiation_area_km2=self.radiation_area_m2 / 1e6,  # Convert m² to km²
			radiation_coefficient=self.radiation_coefficient
		)


@dataclass
class DebrisFieldConfig:
	num_objects: int
	altitude_range_km: List[float]
	inclination_range_deg: List[float] = field(default_factory=lambda: [0, 180])
	eccentricity_range: List[float] = field(default_factory=lambda: [0, 0.1])
	mass_range_kg: List[float] = field(default_factory=lambda: [1, 1000])
	
	# FIXED: Size range now in km
	size_range_km: List[float] = field(default_factory=lambda: [0.0001, 0.01])  # 0.1m to 10m in km
	
	distribution_type: str = "uniform"
	name: Optional[str] = None
	cluster_centers: int = 1
	cluster_radius_km: float = 50.0
	swarm_center_altitude_km: Optional[float] = None
	swarm_radius_km: float = 25.0
	position_uncertainty_km: float = 0.0
	velocity_uncertainty_km_s: float = 0.0
	enable_dynamics: bool = False
	random_perturbations: bool = False
	perturbation_magnitude_ms: float = 0.01
	update_frequency_seconds: float = 300.0
	orbital_decay_rate: float = 1.0

	def __post_init__(self):
		if self.name is None:
			self.name = f"debris_field_{self.num_objects}obj"
		if self.swarm_center_altitude_km is None:
			self.swarm_center_altitude_km = sum(self.altitude_range_km)/2.0


@dataclass
class PhysicsConfig:
	propagator_type: str = "orekit_numerical"
	time_step_seconds: float = 60.0
	simulation_duration_seconds: float = 86400.0
	include_drag: bool = True
	include_solar_radiation: bool = True
	include_j2_perturbation: bool = True
	collision_detection_method: str = "spherical"
	collision_threshold_km: float = 0.01
	integrator: Optional[Dict[str, Any]] = None
	gravity_model: Optional[Dict[str, Any]] = None
	atmosphere_model: Optional[Dict[str, Any]] = None
	solar_radiation_model: Optional[Dict[str, Any]] = None
	third_body_perturbations: Optional[Dict[str, Any]] = None
	include_higher_order_gravity: bool = False
	relativistic_effects: bool = False
	parallel_propagation: bool = False
	num_propagation_threads: int = 1

	def __post_init__(self):
		if self.integrator is None and self.propagator_type == "orekit_numerical":
			self.integrator = {"type": "dormand_prince_853", "atol": 1e-9, "rtol": 1e-9, "adaptive_step": True, "min_step_seconds": 0.1, "max_step_seconds": 30}
		if self.gravity_model is None and self.include_j2_perturbation:
			self.gravity_model = {"degree": 20, "order": 20, "model": "egm96"}


@dataclass
class RewardConfig:
	reward_type: str = "distance_based"
	collision_penalty: float = -1000.0
	fuel_cost_weight: float = 0.1
	safety_distance_km: float = 1.0
	close_approach_penalty_weight: float = 10.0
	mission_success_reward: float = 100.0
	conjunction_penalty_weight: float = 0.0
	proximity_penalties: Optional[Dict[str, Any]] = None
	fuel_efficiency_bonus: Optional[Dict[str, Any]] = None
	mission_progress_reward: Optional[Dict[str, Any]] = None
	multi_objective_weights: Optional[Dict[str, float]] = None


@dataclass
class ObservationConfig:
	include_relative_positions: bool = True
	include_relative_velocities: bool = True
	include_fuel_levels: bool = True
	include_time_to_collision: bool = True
	max_tracked_objects: int = 10
	observation_radius_km: float = 100.0
	include_collision_probability: bool = False
	include_uncertainty_ellipsoids: bool = False
	include_miss_distance: bool = False
	include_delta_v_remaining: bool = False
	include_orbital_elements: bool = False
	time_horizon_hours: float = 1.0
	observation_frequency_hz: float = 1.0
	state_history_length: int = 1
	enhanced_features: Optional[Dict[str, Any]] = None


@dataclass
class SimulationConfig:
	scenario_name: str
	description: str = ""
	satellites: List[SpacecraftConfig] = field(default_factory=list)
	debris_fields: List[DebrisFieldConfig] = field(default_factory=list)
	static_debris: List[SpacecraftConfig] = field(default_factory=list)
	physics: PhysicsConfig = field(default_factory=PhysicsConfig)
	rewards: RewardConfig = field(default_factory=RewardConfig)
	observations: ObservationConfig = field(default_factory=ObservationConfig)
	collision_detection: CollisionDetectionConfig = field(default_factory=CollisionDetectionConfig)
	control: ControlConfig = field(default_factory=ControlConfig)
	random_seed: Optional[int] = None
	episode_horizon: Optional[int] = None

class ConfigurationLoader:
	@staticmethod
	def load_from_file(filepath: str) -> SimulationConfig:
		with open(filepath, 'r') as f:
			if filepath.endswith(('.yaml', '.yml')):
				data = yaml.safe_load(f)
			else:
				data = json.load(f)
		return ConfigurationLoader._dict_to_config(data)

	@staticmethod
	def _dict_to_config(data: Dict[str, Any]) -> SimulationConfig:
		satellites = []
		for sat in data.get('satellites', []):
			orbit_data = sat['orbit'].copy()
			if 'semi_major_axis_km' in orbit_data:
				orbit_data['altitude_km'] = orbit_data.pop('semi_major_axis_km') - 6371.0
			if 'argument_of_perigee_deg' in orbit_data:
				orbit_data['arg_periapsis_deg'] = orbit_data.pop('argument_of_perigee_deg')
			if 'true_anomaly_deg' in orbit_data:
				orbit_data['mean_anomaly_deg'] = orbit_data.pop('true_anomaly_deg')
			orbit = OrbitConfig(**orbit_data)
			sat_copy = sat.copy(); sat_copy['orbit'] = orbit
			if 'thrust_capacity_n' in sat_copy:
				sat_copy['max_thrust_n'] = sat_copy.pop('thrust_capacity_n')
			
			# FIXED: Handle collision radius units - convert from meters to km if needed
			if 'collision_radius' in sat_copy:
				collision_radius_val = sat_copy.pop('collision_radius')
				# If value > 1, assume it's in meters and convert to km
				if collision_radius_val > 1:
					sat_copy['collision_radius_km'] = collision_radius_val / 1000.0
				else:
					sat_copy['collision_radius_km'] = collision_radius_val
			
			# FIXED: Handle near-miss radius units - convert from meters to km if needed
			if 'near_miss_radius' in sat_copy:
				near_miss_radius_val = sat_copy.pop('near_miss_radius')
				# If value > 1, assume it's in meters and convert to km
				if near_miss_radius_val > 1:
					sat_copy['near_miss_radius_km'] = near_miss_radius_val / 1000.0
				else:
					sat_copy['near_miss_radius_km'] = near_miss_radius_val
			
			satellites.append(SpacecraftConfig(**ConfigurationLoader._filter_supported_params(sat_copy, SpacecraftConfig)))

		debris_fields = []
		for field in data.get('debris_fields', []):
			# FIXED: Handle size_range_m -> size_range_km conversion
			if 'size_range_m' in field:
				size_range_m = field.pop('size_range_m')
				field['size_range_km'] = [s / 1000.0 for s in size_range_m]
			
			debris_fields.append(DebrisFieldConfig(**ConfigurationLoader._filter_supported_params(field, DebrisFieldConfig)))

		static_debris = []
		for debris in data.get('static_debris', []):
			orbit_data = debris['orbit'].copy()
			if 'semi_major_axis_km' in orbit_data:
				orbit_data['altitude_km'] = orbit_data.pop('semi_major_axis_km') - 6371.0
			if 'argument_of_perigee_deg' in orbit_data:
				orbit_data['arg_periapsis_deg'] = orbit_data.pop('argument_of_perigee_deg')
			if 'true_anomaly_deg' in orbit_data:
				orbit_data['mean_anomaly_deg'] = orbit_data.pop('true_anomaly_deg')
			orbit = OrbitConfig(**orbit_data)
			debris_copy = debris.copy(); debris_copy['orbit'] = orbit
			if 'thrust_capacity_n' in debris_copy:
				debris_copy['max_thrust_n'] = debris_copy.pop('thrust_capacity_n')
			
			# FIXED: Handle collision radius for static debris
			if 'collision_radius' in debris_copy:
				collision_radius_val = debris_copy.pop('collision_radius')
				if collision_radius_val > 1:
					debris_copy['collision_radius_km'] = collision_radius_val / 1000.0
				else:
					debris_copy['collision_radius_km'] = collision_radius_val
			
			debris_copy.pop('size_m', None)
			static_debris.append(SpacecraftConfig(**ConfigurationLoader._filter_supported_params(debris_copy, SpacecraftConfig)))

		physics = PhysicsConfig(**ConfigurationLoader._filter_supported_params(data.get('physics', {}), PhysicsConfig))
		rewards = RewardConfig(**ConfigurationLoader._filter_supported_params(data.get('rewards', {}), RewardConfig))
		observations = ObservationConfig(**ConfigurationLoader._filter_supported_params(data.get('observations', {}), ObservationConfig))
		collision_detection = CollisionDetectionConfig(**ConfigurationLoader._filter_supported_params(data.get('collision_detection', {}), CollisionDetectionConfig))
		control = ControlConfig(**ConfigurationLoader._filter_supported_params(data.get('control', {}), ControlConfig))
		return SimulationConfig(
			scenario_name=data.get('scenario_name', data.get('name', 'Unknown Scenario')),
			description=data.get('description', ''),
			satellites=satellites,
			debris_fields=debris_fields,
			static_debris=static_debris,
			collision_detection=collision_detection,
    		control=control,  
			physics=physics,
			rewards=rewards,
			observations=observations,
			random_seed=data.get('random_seed')
		)

	@staticmethod
	def _filter_supported_params(data: Dict[str, Any], config_class) -> Dict[str, Any]:
		import inspect
		signature = inspect.signature(config_class)
		supported = set(signature.parameters.keys())
		return {k: v for k, v in data.items() if k in supported}

	@staticmethod
	def save_to_file(config: SimulationConfig, filepath: str):
		data = asdict(config)
		with open(filepath, 'w') as f:
			if filepath.endswith(('.yaml', '.yml')):
				yaml.dump(data, f, default_flow_style=False, indent=2)
			else:
				json.dump(data, f, indent=2)

	@staticmethod
	def validate_config(config: SimulationConfig) -> List[str]:
		errors = []
		if not config.satellites:
			errors.append("Configuration must include at least one satellite")
		for i, sat in enumerate(config.satellites):
			if sat.orbit.altitude_km < 100:
				errors.append(f"Satellite {i} altitude too low: {sat.orbit.altitude_km} km")
			if sat.orbit.altitude_km > 50000:
				errors.append(f"Satellite {i} altitude too high: {sat.orbit.altitude_km} km")
		if config.physics.time_step_seconds <= 0:
			errors.append("Physics time step must be positive")
		if config.physics.collision_threshold_km <= 0:
			errors.append("Collision threshold must be positive")
		return errors


class DebrisGenerator:
	@staticmethod
	def generate_debris_field(field_config: DebrisFieldConfig, seed: Optional[int] = None) -> List[SpacecraftConfig]:
		if seed is not None:
			np.random.seed(seed)
		debris_objects = []
		for i in range(field_config.num_objects):
			altitude = np.random.uniform(*field_config.altitude_range_km)
			inclination = np.random.uniform(*field_config.inclination_range_deg)
			eccentricity = np.random.uniform(*field_config.eccentricity_range)
			raan = np.random.uniform(0, 360)
			argp = np.random.uniform(0, 360)
			ma = np.random.uniform(0, 360)
			orbit = OrbitConfig(altitude_km=altitude, inclination_deg=inclination, eccentricity=eccentricity, raan_deg=raan, arg_periapsis_deg=argp, mean_anomaly_deg=ma)
			mass = np.random.uniform(*field_config.mass_range_kg)
			size_km = np.random.uniform(*field_config.size_range_km)
			
			# FIXED: Use km units consistently
			debris = SpacecraftConfig(
				name=f"Debris_{i}",
				orbit=orbit,
				mass_kg=mass,
				drag_area_m2=size_km * 1e6,  # Convert km² back to m² for drag area
				collision_radius_km=size_km/2,  # Half of size as collision radius
				is_controllable=False,
				max_thrust_n=0.0,
				fuel_capacity_kg=0.0
			)
			debris_objects.append(debris)
		return debris_objects


def create_default_leo_scenario() -> SimulationConfig:
	satellite = SpacecraftConfig(
		name="MainSatellite",
		orbit=OrbitConfig(altitude_km=400, inclination_deg=51.6),
		mass_kg=1000,
		is_controllable=True,
		max_thrust_n=1.0,
		fuel_capacity_kg=100,
		collision_radius_km=0.001,  # 1m = 0.001km
		near_miss_radius_km=0.050   # 50m = 0.050km
	)
	debris_field = DebrisFieldConfig(
		num_objects=50,
		altitude_range_km=[380, 420],
		inclination_range_deg=[45, 60],
		mass_range_kg=[1, 100],
		size_range_km=[0.0001, 0.002]  # 0.1m to 2m in km
	)
	return SimulationConfig(
		scenario_name="LEO_Debris_Avoidance",
		description="Standard LEO debris avoidance scenario with 50 debris objects",
		satellites=[satellite],
		debris_fields=[debris_field],
	)


__all__ = [
	'PropagatorType','RewardType','OrbitConfig','SpacecraftConfig','DebrisFieldConfig','PhysicsConfig','RewardConfig',
	'ObservationConfig','SimulationConfig','ConfigurationLoader','DebrisGenerator','create_default_leo_scenario'
]