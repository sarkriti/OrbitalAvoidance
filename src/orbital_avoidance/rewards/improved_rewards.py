"""
Improved reward functions for orbital avoidance environments.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Callable, Any


def _improved_reward_config(env) -> Dict[str, Any]:
    """
    Default config values for improved reward calculation.
    Bound to the environment when called.
    """
    return {
        "collision_penalty": -100.0,
        "step_penalty": -0.01,
        "fuel_penalty": -0.05,
        "safe_flight_bonus": 0.05,
        "proximity_bonus": 1.0,
        "min_safe_distance_km": 0.05,  # 50m
        "station_keeping_bonus": 0.1,
        "target_orbit_tolerance_km": 1.0,
    }


def create_improved_reward_function(env) -> Callable[[Any], float]:
    """
    Factory: return a reward function bound to the given environment.
    """

    cfg = env.reward_config or {}
    # Provide safe defaults for expected keys so this reward works with both
    # flat and nested reward_config formats used across wrappers.
    cfg = {
        'collision_penalty': cfg.get('collision_penalty', -100.0),
        'step_penalty': cfg.get('step_penalty', -0.01),
        'fuel_penalty': cfg.get('fuel_penalty', -0.05),
        'safe_flight_bonus': cfg.get('safe_flight_bonus', 0.05),
        'proximity_bonus': cfg.get('proximity_bonus', 1.0),
        'min_safe_distance_km': cfg.get('min_safe_distance_km', 0.05),
        'station_keeping_bonus': cfg.get('station_keeping_bonus', 0.1),
        'target_orbit_tolerance_km': cfg.get('target_orbit_tolerance_km', 1.0),
    }

    def reward_fn(action) -> float:
        total_reward = 0.0
        sat = env.current_satellite_state
        debris = env.current_debris_states

        # 1. Collision penalty - support both env APIs (_check_collisions vs _check_collision)
        if hasattr(env, '_check_collisions'):
            collisions = env._check_collisions()
        elif hasattr(env, '_check_collision'):
            collisions = []
            # _check_collision returns bool for RL wrapper; convert to list for compatibility
            collided = env._check_collision()
            if collided:
                collisions = [True]
        else:
            collisions = []
        if collisions:
            total_reward += cfg["collision_penalty"]

        # 2. Step penalty (small negative per step)
        total_reward += cfg["step_penalty"]

        # 3. Fuel usage penalty
        if sat is not None:
            fuel_used = env.initial_fuel - sat["fuel_remaining"]
            total_reward += cfg["fuel_penalty"] * fuel_used

        # 4. Safe flight bonus (for surviving steps without collision)
        if not collisions:
            total_reward += cfg["safe_flight_bonus"]

        # 5. Proximity bonus (discourage being too close to debris)
        if sat is not None and debris:
            sat_pos = np.array(sat["position"])
            min_dist = min(np.linalg.norm(np.array(d["position"]) - sat_pos)
                           for d in debris)
            if min_dist < cfg["min_safe_distance_km"]:
                # inside unsafe zone → negative
                total_reward -= cfg["proximity_bonus"]
            else:
                # safe distance → small positive
                total_reward += cfg["proximity_bonus"]

        # 6. Station keeping bonus (stay near reference orbit)
        if sat is not None and env.reference_orbit is not None:
            ref_sma = env.reference_orbit.semi_major_axis_km
            cur_sma = np.linalg.norm(sat["position"])
            if abs(cur_sma - ref_sma) < cfg["target_orbit_tolerance_km"]:
                total_reward += cfg["station_keeping_bonus"]

        return float(total_reward)

    return reward_fn
