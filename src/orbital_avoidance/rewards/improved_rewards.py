"""
Improved reward functions for orbital avoidance environments.
Now includes:
- Rescaled collision penalty
- Near-miss penalty (<50 km zone)
- Per-step thrust cost instead of cumulative fuel usage
- Balanced bonuses for station-keeping and safe flight
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Callable, Any


def _improved_reward_config(env) -> Dict[str, Any]:
    """Default config values for improved reward calculation."""
    return {
        "collision_penalty": -20.0,        # Scaled down from -100
        "near_miss_distance_km": 50.0,     # Within 50 km = risky
        "min_safe_distance_km": 0.05,      # 50 m collision zone

        "step_penalty": -0.01,             # Small negative each step
        "fuel_penalty": -0.1,              # Scales with thrust per step
        "safe_flight_bonus": 0.05,         # Survive = small reward
        "proximity_bonus": 0.2,            # Safely far from debris
        "station_keeping_bonus": 0.1,      # Stay near reference orbit
        "target_orbit_tolerance_km": 1.0,  # Station-keeping band
    }


def create_improved_reward_function(env) -> Callable[[Any], float]:
    """
    Factory: return a reward function bound to the given environment.
    Reads config from env.reward_config if available, else defaults.
    """

    # Merge defaults with overrides
    cfg = _improved_reward_config(env)
    if env.reward_config:
        cfg.update(env.reward_config)

    def reward_fn(action) -> float:
        total_reward = 0.0
        sat = env.current_satellite_state
        debris = env.current_debris_states

        # 1. Collision penalty
        if hasattr(env, "_check_collisions"):
            collisions = env._check_collisions()
        elif hasattr(env, "_check_collision"):
            collisions = []
            if env._check_collision():
                collisions = [True]
        else:
            collisions = []

        if collisions:
            total_reward += cfg["collision_penalty"]

        # 2. Step penalty
        total_reward += cfg["step_penalty"]

        # 3. Fuel usage penalty (per-step thrust cost)
        if action is not None and sat is not None:
            thrust_mag = float(np.linalg.norm(action))
            total_reward += cfg["fuel_penalty"] * thrust_mag

        # 4. Safe flight bonus (if not collided)
        if not collisions:
            total_reward += cfg["safe_flight_bonus"]

        # 5. Proximity penalties/bonuses
        if sat is not None and debris:
            sat_pos = np.array(sat["position"])
            min_dist = min(np.linalg.norm(np.array(d["position"]) - sat_pos)
                           for d in debris)

            if min_dist < cfg["min_safe_distance_km"]:
                # Direct collision zone (already penalized above)
                total_reward += cfg["collision_penalty"]

            elif min_dist < cfg["near_miss_distance_km"]:
                # Near-miss penalty scaled by distance
                penalty_scale = 1.0 - (min_dist - cfg["min_safe_distance_km"]) / (
                    cfg["near_miss_distance_km"] - cfg["min_safe_distance_km"]
                )
                total_reward -= penalty_scale * cfg["proximity_bonus"]

            else:
                # Safely far → reward
                total_reward += cfg["proximity_bonus"]

        # 6. Station keeping bonus (stay near reference orbit SMA)
        if sat is not None and env.reference_orbit is not None:
            ref_sma = env.reference_orbit.semi_major_axis_km
            cur_sma = np.linalg.norm(sat["position"])
            if abs(cur_sma - ref_sma) < cfg["target_orbit_tolerance_km"]:
                total_reward += cfg["station_keeping_bonus"]

        return float(total_reward)

    return reward_fn
