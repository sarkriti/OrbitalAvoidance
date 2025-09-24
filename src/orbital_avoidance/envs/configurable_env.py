"""Configurable orbital environment with dynamic debris support and improved rewards."""

from __future__ import annotations
import os, logging, math
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from orbital_avoidance.config.config_system import (
    SimulationConfig, ConfigurationLoader, SpacecraftConfig
)
from orbital_avoidance.physics.orekit_physics import create_propagator, OrbitElements

logger = logging.getLogger(__name__)


class ConfigurableOrbitalEnvironment(gym.Env):
    def __init__(self, config_file: Optional[str] = None,
                 config: Optional[SimulationConfig] = None,
                 render_mode: Optional[str] = None,
                 reward_config: Optional[Dict] = None):
        super().__init__()
        if config_file:
            self.config = ConfigurationLoader.load_from_file(config_file)
        elif config:
            self.config = config
        else:
            raise ValueError("Must provide config_file or config")

        errors = ConfigurationLoader.validate_config(self.config)
        if errors:
            raise ValueError(errors)

        self.render_mode = render_mode
        self.engine = create_propagator()
        self.start_date = 0.0
        self._setup_objects()
        self._setup_spaces()

        self.current_time = 0.0
        self.episode_step = 0
        self.max_episode_steps = int(
            self.config.physics.simulation_duration_seconds /
            self.config.physics.time_step_seconds
        )

        # Improved reward config + function
        from orbital_avoidance.rewards.improved_rewards import create_improved_reward_function, _improved_reward_config
        self.reward_config = reward_config or _improved_reward_config(self)
        self.reward_function: Callable = create_improved_reward_function(self)

        # For reward helpers
        controllable = [s for s in self.config.satellites if s.is_controllable]
        if not controllable:
            raise ValueError("No controllable satellites in config")
        self.main_satellite = controllable[0]
        self.initial_fuel = self.main_satellite.fuel_capacity_kg
        self.collision_free_steps = 0
        self.reference_orbit = self._get_reference_orbit()

        # State caches (used in improved reward funcs)
        self.current_satellite_state = None
        self.current_debris_states = []

    # ---------------- setup ---------------- #

    def _setup_objects(self):
        self.satellite_configs = {sat.name: sat for sat in self.config.satellites}
        self.satellites = {}
        for name, sat_cfg in self.satellite_configs.items():
            orbit = sat_cfg.orbit.to_orbit_elements()
            self.engine.create_persistent_object(f"sat::{name}", orbit)
            pos, vel = self.engine.propagate_persistent_delta(f"sat::{name}", 0.0)
            self.satellites[name] = {
                'config': sat_cfg,
                'position': pos,
                'velocity': vel,
                'fuel_remaining': sat_cfg.fuel_capacity_kg,
            }

        # --- debris handling unchanged ---
        self.debris_configs = {}
        if getattr(self.config, 'static_debris', None):
            # Treat legacy 'static_debris' entries as dynamic by default so all debris are propagated
            for d in self.config.static_debris:
                if not hasattr(d, 'dynamic'):
                    setattr(d, 'dynamic', True)
            self.debris_configs.update({d.name: d for d in self.config.static_debris})
        if getattr(self.config, 'debris_fields', None):
            from orbital_avoidance.config.config_system import DebrisGenerator
            for field in self.config.debris_fields:
                # Default to dynamic debris unless explicitly disabled in the field config
                field_dynamic = True
                if isinstance(field, dict):
                    field_dynamic = field.get('dynamics', {}).get('enable_dynamics', True)
                generated = DebrisGenerator.generate_debris_field(field, seed=self.config.random_seed)
                for obj in generated:
                    # Ensure generated debris are marked dynamic by default
                    if not hasattr(obj, 'dynamic'):
                        setattr(obj, 'dynamic', field_dynamic)
                    self.debris_configs[obj.name] = obj
        if getattr(self.config, 'debris', None):
            self.debris_configs.update({d.name: d for d in self.config.debris})

        self.debris_objects = {}
        for name, deb_cfg in self.debris_configs.items():
            orbit = deb_cfg.orbit.to_orbit_elements()
            dynamic = getattr(deb_cfg, 'dynamic', getattr(deb_cfg, 'enable_dynamics', False))
            if dynamic:
                self.engine.create_persistent_object(f"debris::{name}", orbit)
                pos, vel = self.engine.propagate_persistent_delta(f"debris::{name}", 0.0)
            else:
                pos, vel = self.engine.propagate_orbit(orbit, self.start_date)
            self.debris_objects[name] = {
                'config': deb_cfg,
                'dynamic': dynamic,
                'position': pos,
                'velocity': vel,
                'orbit_elements': orbit,
            }

    def _setup_spaces(self):
        obs_cfg = self.config.observations
        num_sat = len(self.config.satellites)
        max_tracked = min(obs_cfg.max_tracked_objects, len(self.debris_configs))

        obs_size = 6 * num_sat
        if obs_cfg.include_relative_positions:
            obs_size += max_tracked * 3
        if obs_cfg.include_relative_velocities:
            obs_size += max_tracked * 3
        if obs_cfg.include_time_to_collision:
            obs_size += max_tracked
        if obs_cfg.include_fuel_levels:
            obs_size += num_sat

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        controllable = [s for s in self.config.satellites if s.is_controllable]
        if len(controllable) == 1:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        else:
            self.action_space = spaces.Dict({
                s.name: spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
                for s in controllable
            })

    # ---------------- gym API ---------------- #

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self._setup_objects()
        self.current_time = 0.0
        self.episode_step = 0
        self.collision_free_steps = 0

        # cache states for reward fn
        sat_dict = self.satellites[self.main_satellite.name]
        self.current_satellite_state = sat_dict
        self.current_debris_states = list(self.debris_objects.values())

        return self._get_observation(), {}

    def step(self, action):
        self._apply_actions(action)
        self._propagate_objects()
        collisions = self._check_collisions()

        # update state wrappers for reward
        sat_dict = self.satellites[self.main_satellite.name]
        self.current_satellite_state = sat_dict
        self.current_debris_states = list(self.debris_objects.values())

        reward = self.reward_function(action)
        terminated = len(collisions) > 0
        truncated = self.episode_step >= self.max_episode_steps

        if not terminated:
            self.collision_free_steps += 1

        obs = self._get_observation()
        info = {
            'collisions': collisions,
            'time': self.current_time,
            'step': self.episode_step,
            'fuel_remaining': {n: s['fuel_remaining'] for n, s in self.satellites.items()},
        }

        self.episode_step += 1
        return obs, reward, terminated, truncated, info

    # ---------------- reward helpers ---------------- #

    def _get_reference_orbit(self) -> OrbitElements:
        sat_orbit = self.main_satellite.orbit
        return OrbitElements(
            semi_major_axis_km=sat_orbit.altitude_km + 6371.0,
            eccentricity=sat_orbit.eccentricity,
            inclination_deg=sat_orbit.inclination_deg,
            raan_deg=sat_orbit.raan_deg,
            arg_periapsis_deg=sat_orbit.arg_periapsis_deg,
            mean_anomaly_deg=sat_orbit.mean_anomaly_deg,
            epoch_mjd=sat_orbit.epoch_mjd,
        )

        # ---------------- physics methods ---------------- #

    def _apply_actions(self, action: Any):
        """Apply agent actions (thrust vectors) to controllable satellites."""
        controllable = [s for s in self.config.satellites if s.is_controllable]
        if len(controllable) == 1:
            actions = {controllable[0].name: action}
        else:
            actions = action

        for sat_name, act in actions.items():
            if not isinstance(act, np.ndarray):
                act = np.array(act, dtype=np.float32)
            self._apply_thrust(sat_name, act)

    def _apply_thrust(self, sat_name: str, action: np.ndarray):
        """Convert normalized thrust [-1,1] into acceleration and apply to satellite."""
        sat = self.satellites[sat_name]
        cfg: SpacecraftConfig = sat['config']

        # `SpacecraftConfig` uses `max_thrust_n` (legacy code sometimes used `max_thrust_newton`)
        max_thrust = getattr(cfg, 'max_thrust_n', getattr(cfg, 'max_thrust_newton', 1.0))
        thrust = action * max_thrust
        mass = cfg.mass_kg
        accel = thrust / mass

        dt = self.config.physics.time_step_seconds

        # Compute delta-v (m/s) then convert to km/s to send to Orekit
        delta_v_m_s = accel * dt
        delta_v_km_s = delta_v_m_s / 1000.0

        # Apply instantaneous delta-v to the persistent Orekit state so the propagator
        # reflects the maneuver (Orekit expects km/s)
        try:
            self.engine.apply_instantaneous_delta_v(f"sat::{sat_name}", np.array(delta_v_km_s, dtype=np.float64))
        except Exception as e:
            logger.debug(f"Engine apply_instantaneous_delta_v failed for {sat_name}: {e}")

        # Update local cached velocity (in km/s)
        vel = np.array(sat['velocity'], dtype=np.float64)
        vel = vel + delta_v_km_s
        sat['velocity'] = vel.tolist()

        # Fuel usage (assume linear to thrust magnitude, ISP=300s)
        fuel_use = np.linalg.norm(thrust) * dt / (300.0 * 9.81)
        sat['fuel_remaining'] = max(0.0, sat['fuel_remaining'] - fuel_use)

    def _propagate_objects(self):
        """Advance all satellites and debris by dt."""
        dt = self.config.physics.time_step_seconds
        self.current_time += dt

        # propagate satellites using the persistent Orekit propagator so physics are realistic
        for sat_name, sat in self.satellites.items():
            try:
                pos, vel = self.engine.propagate_persistent_delta(f"sat::{sat_name}", dt)
                sat['position'] = pos
                sat['velocity'] = vel
            except Exception as e:
                # Fallback to simple Euler step if propagator fails (should not happen normally)
                logger.warning(f"Failed to propagate satellite {sat_name} with engine: {e}. Falling back to simple integrator.")
                pos = np.array(sat['position'], dtype=np.float64)
                vel = np.array(sat['velocity'], dtype=np.float64)
                pos = pos + vel * dt
                sat['position'] = pos.tolist()

        # propagate debris (dynamic ones via engine, static are fixed)
        for name, deb in self.debris_objects.items():
            if deb['dynamic']:
                pos, vel = self.engine.propagate_persistent_delta(f"debris::{name}", dt)
                deb['position'] = pos
                deb['velocity'] = vel

    def _check_collisions(self) -> List[str]:
        """Check for collisions between main satellite and debris."""
        collisions = []
        sat = self.satellites[self.main_satellite.name]
        sat_pos = np.array(sat['position'])

        for name, deb in self.debris_objects.items():
            deb_pos = np.array(deb['position'])
            dist = np.linalg.norm(sat_pos - deb_pos)
            if dist < 0.05:  # km (50 m safety threshold)
                collisions.append(name)

        return collisions

    def _get_observation(self) -> np.ndarray:
        """Build observation vector from satellite and debris states."""
        obs_cfg = self.config.observations
        obs: List[float] = []

        # satellites: pos + vel
        for sat in self.satellites.values():
            obs.extend(sat['position'])
            obs.extend(sat['velocity'])

        # debris tracking (relative info to main satellite)
        if self.debris_objects:
            main_sat = self.satellites[self.main_satellite.name]
            sat_pos = np.array(main_sat['position'])
            sat_vel = np.array(main_sat['velocity'])
            tracked = list(self.debris_objects.values())[:obs_cfg.max_tracked_objects]

            for deb in tracked:
                deb_pos = np.array(deb['position'])
                deb_vel = np.array(deb['velocity'])
                rel_pos = deb_pos - sat_pos
                rel_vel = deb_vel - sat_vel

                if obs_cfg.include_relative_positions:
                    obs.extend(rel_pos.tolist())
                if obs_cfg.include_relative_velocities:
                    obs.extend(rel_vel.tolist())
                if obs_cfg.include_time_to_collision:
                    rel_speed = np.linalg.norm(rel_vel)
                    if rel_speed > 1e-6:
                        ttc = np.linalg.norm(rel_pos) / rel_speed
                    else:
                        ttc = 1e6
                    obs.append(ttc)

        if obs_cfg.include_fuel_levels:
            for sat in self.satellites.values():
                obs.append(sat['fuel_remaining'])

        return np.array(obs, dtype=np.float32)

    # ---------------- rendering / cleanup ---------------- #

    def render(self, mode="human"):
        sat = self.satellites[self.main_satellite.name]
        pos = sat['position']
        print(f"t={self.current_time:.1f}s, pos={pos}, fuel={sat['fuel_remaining']:.2f}kg")

    def close(self):
        pass
