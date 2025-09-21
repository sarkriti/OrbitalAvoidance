"""
Research-grade Gymnasium environment for orbital debris avoidance.

Features:
- Accurate Cowell propagation with configurable perturbations (J2, drag, SRP)
- RTN delta-v actions (impulsive) with fuel penalty
- Observations include relative state to K-nearest debris + scalar context
- Reward blends Pc-style risk, distance shaping, and propellant use
- Deterministic seeding and scenario generation that guarantees close approaches

This environment targets algorithm benchmarking with SB3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from astropy import units as u
from astropy.time import Time
from poliastro.twobody import Orbit
from poliastro.bodies import Earth

from .physics_models import (
    ForceConfig,
    EarthParams,
    apply_delta_v_rtn,
    circular_orbit,
    propagate_orbit_cowell,
    seed_close_encounters,
)
from .pc_conjunction import ConjunctionConfig, batch_min_distance_km


@dataclass
class EnvConfig:
    dt_s: float = 20.0
    steps_per_episode: int = 600  # 200 minutes
    a_km: float = 6771.0  # Fixed: 400 km altitude (6371 + 400)
    inc_deg: float = 51.6
    n_debris: int = 12  # Reduced for better performance 
    k_nearest: int = 4  # Reduced to match GUI expectations

    # Action/obs scaling
    max_dv_rtn_m_s: float = 0.5  # small nudges per step
    max_range_km: float = 50.0

    # Rewards
    w_collision: float = 5.0
    w_warning: float = 1.0
    w_distance: float = 0.2
    w_fuel: float = 0.05

    # Termination
    collision_radius_km: float = 0.2  # More realistic collision radius


class OrbitalDebrisAvoidanceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, cfg: Optional[EnvConfig] = None, forces: Optional[ForceConfig] = None, seed: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.forces = forces or ForceConfig()
        self.rng = np.random.default_rng(seed)

        # Action: RTN impulsive dv in m/s, clipped
        dv_high = np.array([self.cfg.max_dv_rtn_m_s] * 3, dtype=np.float32)
        self.action_space = spaces.Box(low=-dv_high, high=dv_high, dtype=np.float32)

        # Observation: own state + K nearest debris relative states
        # own: [alt_km, speed_km_s] 
        # per debris: [dx, dy, dz, dvx, dvy, dvz] relative position and velocity
        k = self.cfg.k_nearest
        obs_dim = 2 + k * 6  # 2 + 4 * 6 = 26 for k=4
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32)

        self.earth = EarthParams()
        self.conj = ConjunctionConfig(collision_radius_km=self.cfg.collision_radius_km)

        self._reset_scenario()

    # --- Scenario ---------------------------------------------------------------
    def _reset_scenario(self):
        epoch = Time.now()
        self.sat_orbit = circular_orbit(self.cfg.a_km, self.cfg.inc_deg, 0.0, epoch)
        self.debris_orbits = seed_close_encounters(self.sat_orbit, n_debris=self.cfg.n_debris,
                                                   max_offset_km=8.0, inc_spread_deg=0.8)
        # Pre-propagate phasing to ensure approaches inside 200 min
        phase_steps = 200
        r_sat, v_sat = propagate_orbit_cowell(self.sat_orbit, self.cfg.dt_s, phase_steps, self.forces)
        debris_tracks = []
        new_debris = []
        for orb in self.debris_orbits:
            r_d, v_d = propagate_orbit_cowell(orb, self.cfg.dt_s, phase_steps, self.forces)
            debris_tracks.append(r_d)
            new_debris.append(orb)
        self.sat_track_cache = r_sat
        self.debris_track_cache = debris_tracks
        self.debris_orbits = new_debris

        self.step_idx = 0
        self.total_dv_m_s = 0.0
        self.done = False

    # --- Gym API ----------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_scenario()
        obs = self._get_obs()
        info = {"total_dv_m_s": self.total_dv_m_s}
        return obs, info

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset before stepping a finished episode")
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply impulsive dv in RTN frame
        r = self.sat_orbit.r.to_value(u.km)
        v = self.sat_orbit.v.to_value(u.km / u.s)
        dv_rtn_km_s = action / 1000.0
        v_new = apply_delta_v_rtn(r, v, dv_rtn_km_s)
        # Create a new orbit from updated state
        self.sat_orbit = Orbit.from_vectors(Earth, r * u.km, v_new * u.km / u.s, epoch=self.sat_orbit.epoch)

        # Propagate all orbits by dt
        self._propagate_dt()

        # Compute reward and termination
        min_d_km, _, _ = batch_min_distance_km(
            self.sat_track_cache[max(0, self.step_idx-1):self.step_idx+1],
            [d[max(0, self.step_idx-1):self.step_idx+1] for d in self.debris_track_cache]
        )
        collision = min_d_km <= self.cfg.collision_radius_km
        warning = min_d_km <= 5.0

        reward = 0.0
        if collision:
            reward -= self.cfg.w_collision
        elif warning:
            reward -= self.cfg.w_warning * (5.0 - min_d_km) / 5.0
        # distance shaping (encourage > 50 km)
        if min_d_km < 50.0:
            reward -= self.cfg.w_distance * (50.0 - min_d_km) / 50.0

        # fuel penalty
        step_dv = float(np.linalg.norm(action))
        self.total_dv_m_s += step_dv
        reward -= self.cfg.w_fuel * (step_dv / self.cfg.max_dv_rtn_m_s)

        self.done = collision or (self.step_idx >= self.cfg.steps_per_episode - 2)

        obs = self._get_obs()
        info = {
            "min_distance_km": float(min_d_km),
            "collision": collision,
            "warning": warning,
            "step_idx": self.step_idx,
            "total_dv_m_s": self.total_dv_m_s,
        }
        return obs, reward, self.done, False, info

    # --- Helpers -----------------------------------------------------------------
    def _propagate_dt(self):
        # advance one step with Cowell for sat and debris
        r_sat, v_sat = propagate_orbit_cowell(self.sat_orbit, self.cfg.dt_s, 1, self.forces)
        self.sat_orbit = Orbit.from_vectors(Earth, r_sat[-1] * u.km, v_sat[-1] * u.km / u.s, epoch=self.sat_orbit.epoch)

        new_tracks = []
        new_debris_orbits = []
        for orb in self.debris_orbits:
            r_d, v_d = propagate_orbit_cowell(orb, self.cfg.dt_s, 1, self.forces)
            new_tracks.append(r_d)
            # Update debris orbit objects like we do for satellite
            updated_orbit = Orbit.from_vectors(Earth, r_d[-1] * u.km, v_d[-1] * u.km / u.s, epoch=orb.epoch)
            new_debris_orbits.append(updated_orbit)
        
        # Update debris orbits to new states
        self.debris_orbits = new_debris_orbits
        
        # append to cache
        self.sat_track_cache = np.vstack((self.sat_track_cache, r_sat[-1][None, :]))
        for i, r_d in enumerate(new_tracks):
            self.debris_track_cache[i] = np.vstack((self.debris_track_cache[i], r_d[-1][None, :]))
        self.step_idx += 1

    def _get_obs(self) -> np.ndarray:
        # own state scalars
        r = self.sat_orbit.r.to_value(u.km)
        v = self.sat_orbit.v.to_value(u.km / u.s)
        alt = np.linalg.norm(r) - self.earth.r
        speed = np.linalg.norm(v)
        own = np.array([alt, speed], dtype=np.float32)

        # distances to debris at current step
        rs = self.sat_track_cache[self.step_idx]
        dists = []
        rel_states = []
        for track in self.debris_track_cache:
            dr = track[self.step_idx] - rs
            dists.append(np.linalg.norm(dr))
            # velocity diffs approximately from latest two samples
            if self.step_idx >= 1:
                dv = (track[self.step_idx] - track[self.step_idx-1]) / (self.cfg.dt_s)
                sv = (self.sat_track_cache[self.step_idx] - self.sat_track_cache[self.step_idx-1]) / (self.cfg.dt_s)
                dv_rel = dv - sv
            else:
                dv_rel = np.zeros(3)
            rel_states.append(np.hstack((dr, dv_rel)))
        dists = np.array(dists)
        order = np.argsort(dists)[: self.cfg.k_nearest]
        stacked = np.hstack([rel_states[i] for i in order]).astype(np.float32)

        return np.hstack((own, stacked)).astype(np.float32)

    # Optional render hook (text only for now)
    def render(self):
        print(f"step={self.step_idx} alt_km={self.sat_orbit.a.to_value(u.km) - self.earth.r:.1f}")


# Convenience factory

def make_env(seed: Optional[int] = None) -> OrbitalDebrisAvoidanceEnv:
    return OrbitalDebrisAvoidanceEnv(seed=seed)
