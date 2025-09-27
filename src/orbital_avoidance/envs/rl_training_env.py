"""RL orbital training environment compatible with updated ConfigurableOrbitalEnvironment."""

from __future__ import annotations
import numpy as np
import logging
from typing import Dict, List, Optional, Callable
import gymnasium as gym
from gymnasium import spaces

from orbital_avoidance.envs.configurable_env import ConfigurableOrbitalEnvironment
from orbital_avoidance.config.config_system import SimulationConfig
from orbital_avoidance.physics.orekit_physics import OrbitElements
from orbital_avoidance.stochastic.stochastic_noise import StochasticNoiseManager
logger = logging.getLogger(__name__)


class StateWrapper:
    """Wrap satellite or debris state for convenience."""
    def __init__(self, state_dict):
        self.position_km = np.array(state_dict['position'], dtype=np.float64)
        self.velocity_km_s = np.array(state_dict['velocity'], dtype=np.float64)
        self.fuel_remaining_kg = state_dict.get('fuel_remaining', 0.0)
        self.mass_kg = state_dict.get('config').mass_kg if 'config' in state_dict else 1000.0
        self.collision_radius = getattr(state_dict.get('config', None), 'collision_radius_km', 0.005)
        self.near_miss_radius = getattr(state_dict.get('config', None), 'near_miss_radius_km', 0.025)
        self._dict = state_dict

    def update_from_dict(self, d):
        self.position_km = np.array(d['position'], dtype=np.float64)
        self.velocity_km_s = np.array(d['velocity'], dtype=np.float64)
        self.fuel_remaining_kg = d.get('fuel_remaining', 0.0)
        self.mass_kg = d.get('config').mass_kg if 'config' in d else 1000.0
        self.collision_radius = getattr(d.get('config', None), 'collision_radius_km', 0.005)
        self.near_miss_radius = getattr(d.get('config', None), 'near_miss_radius_km', 0.025)
        self._dict = d

    def __getitem__(self, k):
        return getattr(self, k, self._dict.get(k))

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

    def __init__(self,
                 config_file: Optional[str] = None,
                 config: Optional[SimulationConfig] = None,
                 max_debris_tracked: int = 10,
                 reward_config: Optional[Dict] = None,
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

        self._base_env = ConfigurableOrbitalEnvironment(config=self.config, render_mode=render_mode)
        self.reward_config = reward_config or self._default_reward_config()
        self.reward_function = self._create_reward_function()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # obs: pos(3)+vel(3)+fuel+time + debris*10 (includes normalized near-miss)
        obs_size = 6 + 2 + (self.max_debris_tracked * 10)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.reference_orbit = self._get_reference_orbit()

        self.step_count = 0
        self.collision_free_steps = 0

        self.current_satellite_state: Optional[StateWrapper] = None
        self.current_debris_states: List[StateWrapper] = []

        # Determine maximum episode steps based on config
        self.max_episode_steps = self.config.environment.get('episode_horizon', 0)
        if self.max_episode_steps <= 0:
            self.max_episode_steps = int(self.config.physics.simulation_duration_seconds / self.config.physics.time_step_seconds)
        
        stochasticity_config = config.get('stochasticity', {'enabled': False})
        self.noise_manager = StochasticNoiseManager(stochasticity_config)

    def _default_reward_config(self) -> Dict:
        return {
            'weights': {'collision': 4.0, 'fuel': 1.0, 'orbit': 2.0, 'mission': 1.0},
            'collision': {'safety_margin_multiplier': 1.0, 'lookahead_seconds': 7200, 'danger_threshold': 0.1},
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

        base_obs, _ = self._base_env.reset()
        sat_dict = self._base_env.satellites[self.main_satellite_name]
        self.current_satellite_state = StateWrapper(sat_dict)
        
        # Apply initial state noise
        if self.noise_manager.enabled:
            noisy_pos, noisy_vel = self.noise_manager.apply_initial_state_noise(
                self.current_satellite_state.position_km.copy(),
                self.current_satellite_state.velocity_km_s.copy()
            )
            self.current_satellite_state.position_km = noisy_pos
            self.current_satellite_state.velocity_km_s = noisy_vel
            # Update the base environment state if possible
            # (This depends on your ConfigurableOrbitalEnvironment implementation)

        self.current_debris_states = [StateWrapper(d) for d in self._base_env.debris_objects.values()]
        
        # Apply noise to debris initial states too
        if self.noise_manager.enabled:
            for debris_state in self.current_debris_states:
                noisy_pos, noisy_vel = self.noise_manager.apply_initial_state_noise(
                    debris_state.position_km.copy(),
                    debris_state.velocity_km_s.copy()
                )
                debris_state.position_km = noisy_pos
                debris_state.velocity_km_s = noisy_vel

        self.step_count = 0
        self.collision_free_steps = 0
        self.reference_orbit = self._calculate_reference_orbit_from_state()

        return self._get_observation(), self._get_info()

    # Modify the step method to add control and dynamics noise:
    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        
        # Apply control noise to action
        noisy_action = self.noise_manager.apply_control_noise(action.copy())
        noisy_action = np.clip(noisy_action, -1.0, 1.0)  # Re-clip after noise
        
        # Pass noisy action to base env
        base_obs, reward_dummy, terminated_dummy, truncated_dummy, info_dummy = self._base_env.step(noisy_action)

        # Update state wrappers and apply dynamics noise
        sat_dict = self._base_env.satellites[self.main_satellite_name]
        self.current_satellite_state.update_from_dict(sat_dict)
        
        if self.noise_manager.enabled:
            noisy_pos, noisy_vel = self.noise_manager.apply_dynamics_noise(
                self.current_satellite_state.position_km.copy(),
                self.current_satellite_state.velocity_km_s.copy()
            )
            self.current_satellite_state.position_km = noisy_pos
            self.current_satellite_state.velocity_km_s = noisy_vel

        for i, debris_dict in enumerate(self._base_env.debris_objects.values()):
            if i < len(self.current_debris_states):
                self.current_debris_states[i].update_from_dict(debris_dict)
                if self.noise_manager.enabled:
                    noisy_pos, noisy_vel = self.noise_manager.apply_dynamics_noise(
                        self.current_debris_states[i].position_km.copy(),
                        self.current_debris_states[i].velocity_km_s.copy()
                    )
                    self.current_debris_states[i].position_km = noisy_pos
                    self.current_debris_states[i].velocity_km_s = noisy_vel

        reward = self.reward_function(action)  # Use original action for reward

        terminated = self._check_collision()
        truncated = self.step_count >= self.max_episode_steps

        self.step_count += 1
        if not terminated:
            self.collision_free_steps += 1

        # Apply observation noise to the returned observation
        observation = self._get_observation()
        noisy_observation = self.noise_manager.apply_observation_noise(observation)

        return noisy_observation, reward, terminated, truncated, self._get_info()

    def _calculate_reference_orbit_from_state(self) -> OrbitElements:
        pos = self.current_satellite_state.position_km
        vel = self.current_satellite_state.velocity_km_s
        return self._base_env.engine.cartesian_to_orbit_elements(pos, vel, self.main_satellite.mass_kg)

    def _get_observation(self) -> np.ndarray:
        obs = []
        sat_pos = self.current_satellite_state.position_km
        sat_vel = self.current_satellite_state.velocity_km_s
        obs.extend(sat_pos.tolist())
        obs.extend(sat_vel.tolist())

        fuel_norm = self.current_satellite_state.fuel_remaining_kg / self.initial_fuel
        time_norm = 1.0 - (self.step_count / self.max_episode_steps)
        obs.extend([fuel_norm, time_norm])

        closest = self._get_closest_debris(self.max_debris_tracked)
        for debris in closest:
            rel_pos = debris.position_km - sat_pos
            rel_vel = debris.velocity_km_s - sat_vel
            miss_distance, ttc = self._calculate_collision_metrics(sat_pos, sat_vel, debris.position_km, debris.velocity_km_s)
            normalized_miss = miss_distance / (self.main_satellite.near_miss_radius + debris.near_miss_radius)
            obs.extend([*rel_pos, *rel_vel, miss_distance, normalized_miss, ttc])
        for _ in range(len(closest), self.max_debris_tracked):
            obs.extend([0.0] * 10)

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
        for debris in self.current_debris_states:
            distance = np.linalg.norm(debris.position_km - sat_pos)
            if distance < (self.main_satellite.collision_radius + debris.collision_radius):
                return True
        return False

    def _create_reward_function(self) -> Callable:
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
