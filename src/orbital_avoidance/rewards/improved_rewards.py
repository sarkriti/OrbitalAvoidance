"""
PPO-optimized reward functions for orbital avoidance environments.

Features:
- Proper reward scaling for PPO (collision: -50, per-step: ~0.01)
- Proximity-based penalties with distance zones
- Fuel efficiency incentives
- Mission success bonuses
- All units in kilometers for consistency

Reward Philosophy:
- Terminal penalties: -50 (collision)
- Per-step penalties: -10 (critical proximity), -1 (warning proximity)
- Per-step costs: ~0.01 (fuel, time)
- Success bonus: +10
- Expected range: -50 (immediate collision) to +15 (perfect episode)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_min_debris_distance(satellites: Dict, debris_objects: Dict) -> float:
    """
    Calculate minimum distance from any satellite to any debris.
    
    Args:
        satellites: Dictionary of satellite states with 'position' keys
        debris_objects: Dictionary of debris states with 'position' keys
        
    Returns:
        Minimum distance in kilometers, or inf if no debris/satellites
    """
    if not satellites or not debris_objects:
        return float('inf')
    
    min_dist = float('inf')
    
    for sat_name, sat_data in satellites.items():
        sat_pos = sat_data['position']
        
        for debris_name, debris_data in debris_objects.items():
            debris_pos = debris_data['position']
            distance = np.linalg.norm(sat_pos - debris_pos)
            min_dist = min(min_dist, distance)
    
    return min_dist


def safe_reward_get(cfg, key, default=None):
    """
    Safely get value from either config object or dictionary.
    
    Args:
        cfg: Either a dictionary or a config object
        key: Key/attribute name to retrieve
        default: Default value if not found
        
    Returns:
        Retrieved value or default
    """
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    else:
        return getattr(cfg, key, default)


def create_ppo_optimized_reward_function(env) -> Callable[[Any, Dict], float]:
    """
    Factory: return a PPO-optimized reward function bound to the environment.
    
    Properly scaled for advantage estimation and value function learning.
    Reads configuration from env.config_dict['rewards'] section.
    
    Args:
        env: Environment instance with config_dict attribute
        
    Returns:
        Reward function that takes (action, info) and returns float reward
    """
    
    # Get rewards config from JSON - check multiple possible locations
    rewards_cfg = {}
    if hasattr(env, 'config_dict'):
        rewards_cfg = safe_reward_get(env.config_dict, 'rewards', {})
    elif hasattr(env, 'reward_config'):
        rewards_cfg = env.reward_config
    
    # PPO-OPTIMIZED REWARD SCALES
    # Terminal events (episode-ending)
    collision_penalty = float(safe_reward_get(rewards_cfg, 'collision_penalty', -50.0))
    mission_success_reward = float(safe_reward_get(rewards_cfg, 'mission_success_reward', 10.0))
    
    # Per-step costs (should sum to reasonable values over episode)
    fuel_cost_weight = float(safe_reward_get(rewards_cfg, 'fuel_cost_weight', 0.01))
    survival_bonus = float(safe_reward_get(rewards_cfg, 'survival_bonus', 0.0))
    time_penalty = float(safe_reward_get(rewards_cfg, 'time_penalty', 0.001))  # Configurable now
    
    # Proximity penalties - get from config or scale from collision penalty
    proximity_cfg = safe_reward_get(rewards_cfg, 'proximity_penalties', {})
    
    # Handle proximity_cfg being either dict or object
    critical_distance_km = float(safe_reward_get(proximity_cfg, 'critical_distance_km', 0.008))  # 8m
    warning_distance_km = float(safe_reward_get(proximity_cfg, 'warning_distance_km', 0.025))   # 25m
    safe_distance_km = float(safe_reward_get(proximity_cfg, 'safe_distance_km', 0.075))         # 75m
    
    # Get penalties from config or calculate as percentage of collision penalty
    critical_penalty = float(safe_reward_get(proximity_cfg, 'critical_penalty', collision_penalty * 0.2))  # Default: -10
    warning_penalty = float(safe_reward_get(proximity_cfg, 'warning_penalty', collision_penalty * 0.02))   # Default: -1
    safe_bonus = float(safe_reward_get(proximity_cfg, 'safe_bonus', 0.01))  # Small bonus for safety
    
    logger.info("=" * 50)
    logger.info("PPO Reward Function Configuration")
    logger.info("=" * 50)
    logger.info(f"Collision penalty: {collision_penalty:.2f}")
    logger.info(f"Critical proximity penalty (<{critical_distance_km*1000:.1f}m): {critical_penalty:.2f}")
    logger.info(f"Warning proximity penalty (<{warning_distance_km*1000:.1f}m): {warning_penalty:.2f}")
    logger.info(f"Safe distance bonus (>{safe_distance_km*1000:.1f}m): {safe_bonus:.3f}")
    logger.info(f"Mission success reward: {mission_success_reward:.2f}")
    logger.info(f"Survival bonus per step: {survival_bonus:.4f}")
    logger.info(f"Fuel cost weight: {fuel_cost_weight:.4f}")
    logger.info(f"Time penalty per step: {time_penalty:.4f}")
    
    # Calculate expected reward range
    max_episode_reward = mission_success_reward + (survival_bonus + safe_bonus - time_penalty) * 1500
    logger.info(f"Expected episode reward range: [{collision_penalty:.1f} to {max_episode_reward:.1f}]")
    logger.info("=" * 50)
    
    def reward_fn(action: Optional[np.ndarray], info: Dict[str, Any]) -> float:
        """
        Calculate PPO-optimized reward for current step.
        
        Args:
            action: Control action (can be None)
            info: Dictionary containing:
                - collisions: bool, whether collision occurred
                - min_debris_distance_km: float, minimum distance to debris
                - truncated: bool, whether episode hit time limit
                - success: bool, whether episode completed successfully
                
        Returns:
            Scalar reward value
        """
        # if info.get('step', 0) % 100 == 0:  # Log every 100 steps
        #     logger.info(f"=== REWARD DEBUG at step {info.get('step', 0)} ===")
        #     logger.info(f"survival_bonus={survival_bonus}")
        #     logger.info(f"collision_penalty={collision_penalty}")
        #     logger.info(f"min_distance={info.get('min_debris_distance_km', float('inf'))}")
        #     logger.info(f"collisions={info.get('collisions', False)}")

        # Start with survival bonus (helps with early learning)
        total_reward = survival_bonus
        
        # 1. TERMINAL COLLISION PENALTY - Episode ending event
        if info.get('collisions', False):
            return float(collision_penalty)
        
        # 2. PROXIMITY PENALTIES - Based on closest approach to debris
        min_distance = info.get('min_debris_distance_km', float('inf'))
        
        if min_distance < critical_distance_km:
            # Critical zone - very dangerous, large per-step penalty
            total_reward += critical_penalty
            
        elif min_distance < warning_distance_km:
            # Warning zone - scaled penalty based on distance
            # Closer to critical = more penalty
            scale = (warning_distance_km - min_distance) / (warning_distance_km - critical_distance_km)
            total_reward += warning_penalty * (0.1 + 0.9 * scale)  # Min 10% of warning penalty
            
        elif min_distance < safe_distance_km:
            # Transitional zone - very small penalty
            scale = (safe_distance_km - min_distance) / (safe_distance_km - warning_distance_km)
            total_reward -= warning_penalty * 0.1 * scale
            
        else:
            # Safe zone - small bonus for maintaining safe distance
            total_reward += safe_bonus
        
        # 3. FUEL USAGE PENALTY - Proportional to thrust magnitude
        if action is not None:
            thrust_mag = float(np.linalg.norm(action))
            total_reward -= fuel_cost_weight * thrust_mag
        
        # 4. TIME PENALTY - Encourage efficiency (very small)
        total_reward -= time_penalty
        
        # 5. MISSION SUCCESS BONUS - Only if episode completes successfully
        if info.get('success', False):
            total_reward += mission_success_reward
        elif info.get('truncated', False) and not info.get('collisions', False):
            # Also give success bonus if episode completes without collision
            total_reward += mission_success_reward
        
        return float(total_reward)
    
    return reward_fn


# Legacy function name for backward compatibility
def create_improved_reward_function(env) -> Callable[[Any, Dict], float]:
    """
    Legacy function name - redirects to PPO-optimized version.
    """
    return create_ppo_optimized_reward_function(env)


# Export public API
__all__ = [
    'create_ppo_optimized_reward_function',
    'create_improved_reward_function',
    'calculate_min_debris_distance'
]