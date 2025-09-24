"""RL orbital training environment with improved reward functions."""

from __future__ import annotations
import numpy as np
import math
import logging
from typing import Dict, List, Optional, Callable
import gymnasium as gym
from gymnasium import spaces

# Import your real environment
from orbital_avoidance.envs.configurable_env import ConfigurableOrbitalEnvironment
from orbital_avoidance.config.config_system import SimulationConfig
from orbital_avoidance.physics.orekit_physics import OrbitElements

logger = logging.getLogger(__name__)


class StateWrapper:
    """Wrap satellite or debris state for convenience."""
    def __init__(self, state_dict):
        self.position_km = state_dict['position']
        self.velocity_km_s = state_dict['velocity']
        self.fuel_remaining_kg = state_dict.get('fuel_remaining', 0)
        self.mass_kg = state_dict.get('config').mass_kg if 'config' in state_dict else 1000
        self._dict = state_dict

    def update_from_dict(self, d):
        self.position_km = d['position']
        self.velocity_km_s = d['velocity']
        self.fuel_remaining_kg = d.get('fuel_remaining', 0)
        self.mass_kg = d.get('config').mass_kg if 'config' in d else 1000
        self._dict = d

    def __getitem__(self, k):
        return {'position': self.position_km, 'velocity': self.velocity_km_s,
                'fuel_remaining': self.fuel_remaining_kg}.get(k, self._dict[k])

    def __setitem__(self, k, v):
        if k == 'position':
            self.position_km = v
        elif k == 'velocity':
            self.velocity_km_s = v
        elif k == 'fuel_remaining':
            self.fuel_remaining_kg = v
        self._dict[k] = v


class RLOrbitalEnvironment(gym.Env):
    """Gym wrapper around ConfigurableOrbitalEnvironment with improved rewards."""

    def __init__(self, config_file: Optional[str] = None, config: Optional[SimulationConfig] = None,
                 max_debris_tracked: int = 10, reward_config: Optional[Dict] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        if config is not None:
            self.config = config
        elif config_file is not None:
            from orbital_avoidance.config.config_system import ConfigurationLoader
            self.config = ConfigurationLoader.load_from_file(config_file)
        else:
            raise ValueError("Must provide config_file or config")

        self.max_debris_tracked = max_debris_tracked
        self.render_mode = render_mode

        controllable = [s for s in self.config.satellites if s.is_controllable]
        if not controllable:
            raise ValueError("No controllable satellites in config")

        self.main_satellite = controllable[0]
        self.main_satellite_name = self.main_satellite.name
        self.initial_fuel = self.main_satellite.fuel_capacity_kg

        self.reward_config = reward_config or self._default_reward_config()
        self.reward_function = self._create_reward_function()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        obs_size = 6 + 2 + (self.max_debris_tracked * 8)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self._base_env = ConfigurableOrbitalEnvironment(config=self.config)
        self.reference_orbit = self._get_reference_orbit()

        self.step_count = 0
        self.collision_free_steps = 0

        self.current_satellite_state = None
        self.current_debris_states: List[StateWrapper] = []

    def _default_reward_config(self) -> Dict:
        return {
            'weights': {'collision': 4.0, 'fuel': 1.0, 'orbit': 2.0, 'mission': 1.0},
            'collision': {'safety_margin_multiplier': 5.0, 'lookahead_seconds': 7200, 'danger_threshold': 0.1},
            'fuel': {'critical_ratio': 0.3, 'low_ratio': 0.6, 'precision_threshold': 0.1},
            'orbit': {'tolerances': {'semi_major_axis_m': 100.0, 'eccentricity': 0.01,
                                     'inclination_rad': 0.01, 'raan_rad': 0.01}},
            'mission': {'milestone_interval': 100, 'completion_threshold': 0.8, 'midpoint_threshold': 0.5}
        }

    def _get_reference_orbit(self) -> OrbitElements:
        sat_orbit = self.main_satellite.orbit
        return OrbitElements(
            semi_major_axis_km=sat_orbit.altitude_km + 6371.0,
            eccentricity=sat_orbit.eccentricity,
            inclination_deg=sat_orbit.inclination_deg,
            raan_deg=sat_orbit.raan_deg,
            arg_periapsis_deg=sat_orbit.arg_periapsis_deg,
            mean_anomaly_deg=sat_orbit.mean_anomaly_deg,
            epoch_mjd=sat_orbit.epoch_mjd
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        # Base env may return (obs, info) or (obs, reward, terminated, truncated, info)
        base_reset_result = self._base_env.reset()
        if isinstance(base_reset_result, tuple):
            if len(base_reset_result) >= 1:
                base_obs = base_reset_result[0]
            else:
                base_obs = None
        else:
            base_obs = base_reset_result
        sat_dict = self._base_env.satellites[self.main_satellite_name]
        self.current_satellite_state = StateWrapper(sat_dict)
        self.current_debris_states = [StateWrapper(d) for d in self._base_env.debris_objects.values()]

        self.step_count = 0
        self.collision_free_steps = 0
        self.reference_orbit = self._calculate_reference_orbit_from_state()
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        thrust_newtons = action * self.main_satellite.max_thrust_n
        # Accept multiple return signatures from base env (ignore values; we read state dicts)
        base_step_result = self._base_env.step(thrust_newtons)
        if isinstance(base_step_result, tuple) and len(base_step_result) >= 1:
            base_obs = base_step_result[0]
        else:
            base_obs = base_step_result

        sat_dict = self._base_env.satellites[self.main_satellite_name]
        self.current_satellite_state.update_from_dict(sat_dict)
        for i, debris_dict in enumerate(self._base_env.debris_objects.values()):
            if i < len(self.current_debris_states):
                self.current_debris_states[i].update_from_dict(debris_dict)

        # Debug logging: simulation time, main satellite position, and first debris position
        try:
            sim_time = getattr(self._base_env, 'simulation_time_seconds', None)
            # Fallback: infer from step_count and physics timestep
            if sim_time is None:
                sim_time = self.step_count * getattr(self.config.physics, 'time_step_seconds', 0)
            sat_pos = self.current_satellite_state.position_km
            first_debris_pos = None
            if self.current_debris_states:
                first_debris_pos = self.current_debris_states[0].position_km
            logger.info(f"[ENV DEBUG] sim_time={sim_time} s, sat_pos_km={sat_pos}, debris0_pos_km={first_debris_pos}")
        except Exception as e:
            logger.warning(f"Failed to log env debug info: {e}")

        reward = self.reward_function(action)
        terminated = self._check_collision()
        truncated = self.step_count >= int(self.config.physics.simulation_duration_seconds / self.config.physics.time_step_seconds)

        self.step_count += 1
        if not terminated:
            self.collision_free_steps += 1

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _calculate_reference_orbit_from_state(self) -> OrbitElements:
        pos = self.current_satellite_state.position_km
        vel = self.current_satellite_state.velocity_km_s
        return self._base_env.engine.cartesian_to_orbit_elements(pos, vel, 59945.0)

    def _get_observation(self) -> np.ndarray:
        obs = []
        sat_pos = self.current_satellite_state.position_km
        sat_vel = self.current_satellite_state.velocity_km_s
        obs.extend([*sat_pos, *sat_vel])

        fuel_norm = self.current_satellite_state.fuel_remaining_kg / self.initial_fuel
        time_norm = 1.0 - (self.step_count / int(self.config.physics.simulation_duration_seconds / self.config.physics.time_step_seconds))
        obs.extend([fuel_norm, time_norm])

        closest = self._get_closest_debris(self.max_debris_tracked)
        for debris in closest:
            rel_pos = debris.position_km - sat_pos
            rel_vel = debris.velocity_km_s - sat_vel
            miss_distance, ttc = self._calculate_collision_metrics(
                sat_pos, sat_vel, debris.position_km, debris.velocity_km_s
            )
            obs.extend([*rel_pos, *rel_vel, miss_distance, ttc])
        for _ in range(len(closest), self.max_debris_tracked):
            obs.extend([0.0] * 8)

        return np.array(obs, dtype=np.float32)

    def _get_closest_debris(self, max_count: int) -> List[StateWrapper]:
        sat_pos = self.current_satellite_state.position_km
        debris_with_dist = [(np.linalg.norm(d.position_km - sat_pos), d) for d in self.current_debris_states]
        debris_with_dist.sort(key=lambda x: x[0])
        return [d for _, d in debris_with_dist[:max_count]]

    def _calculate_collision_metrics(self, sat_pos, sat_vel, debris_pos, debris_vel):
        rel_pos = debris_pos - sat_pos
        rel_vel = debris_vel - sat_vel
        if np.dot(rel_pos, rel_vel) > 0:
            return np.linalg.norm(rel_pos), -1.0
        rel_vel_sq = np.dot(rel_vel, rel_vel)
        if rel_vel_sq < 1e-10:
            return np.linalg.norm(rel_pos), float('inf')
        t = -np.dot(rel_pos, rel_vel) / rel_vel_sq
        miss_distance = np.linalg.norm(rel_pos + rel_vel * t)
        return miss_distance, t

    def _check_collision(self) -> bool:
        sat_pos = self.current_satellite_state.position_km
        collision_radius_km = self.main_satellite.collision_radius_m / 1000.0
        for debris in self.current_debris_states:
            if np.linalg.norm(debris.position_km - sat_pos) < collision_radius_km:
                return True
        return False

    def _create_reward_function(self) -> Callable:
        """Hook to improved reward function."""
        from orbital_avoidance.rewards.improved_rewards import create_improved_reward_function
        return create_improved_reward_function(self)

    def _get_info(self) -> Dict:
        return {
            'step_count': self.step_count,
            'fuel_remaining_kg': self.current_satellite_state.fuel_remaining_kg,
            'fuel_remaining_percent': self.current_satellite_state.fuel_remaining_kg / self.initial_fuel * 100,
            'collision_free_steps': self.collision_free_steps
        }

    def render(self, mode: str = 'human'):
        if mode == 'human':
            print(f"Step {self.step_count}, Fuel {self.current_satellite_state.fuel_remaining_kg:.1f} kg")

    def close(self):
        if self._base_env:
            self._base_env.close()


def create_rl_environment(config_file: str = "scenarios/leo_debris_avoidance.json", **kwargs) -> RLOrbitalEnvironment:
    return RLOrbitalEnvironment(config_file=config_file, **kwargs)


__all__ = ['RLOrbitalEnvironment', 'create_rl_environment']
