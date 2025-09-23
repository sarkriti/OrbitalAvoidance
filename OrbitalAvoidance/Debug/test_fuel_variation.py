#!/usr/bin/env python3
"""
Test the fixed reward system to see proper fuel reward variation.
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_fuel_variation():
    """Test that fuel rewards now vary properly without being overwhelmed by orbit penalties."""
    print("=== TESTING FUEL REWARD VARIATION ===")
    
    from rl_training_env import create_rl_environment
    
    env = create_rl_environment("scenarios/leo_debris_avoidance_fixed.json")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    rewards = []
    fuel_rewards = []
    orbit_rewards = []
    
    # Test different action magnitudes
    action_magnitudes = [0.0, 0.1, 0.5, 1.0]
    
    for magnitude in action_magnitudes:
        # Reset environment for each test
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # Take action with different thrust magnitude
        action = np.array([magnitude, 0.0, 0.0])  # X-direction thrust
        obs, reward, done, truncated, info = env.step(action)
        
        # Get component rewards
        fuel_reward = env._calculate_fuel_reward(action)
        orbit_reward = env._calculate_orbit_reward()
        
        rewards.append(reward)
        fuel_rewards.append(fuel_reward)
        orbit_rewards.append(orbit_reward)
        
        print(f"Action magnitude: {magnitude:.1f}")
        print(f"  Total reward: {reward:.6f}")
        print(f"  Fuel reward: {fuel_reward:.6f}")
        print(f"  Orbit reward: {orbit_reward:.6f}")
        print(f"  Fuel remaining: {env.current_satellite_state.fuel_remaining_kg:.3f} kg")
        print()
    
    print("=== REWARD VARIATION ANALYSIS ===")
    print(f"Total reward range: {min(rewards):.6f} to {max(rewards):.6f}")
    print(f"Fuel reward range: {min(fuel_rewards):.6f} to {max(fuel_rewards):.6f}")
    print(f"Orbit reward range: {min(orbit_rewards):.6f} to {max(orbit_rewards):.6f}")
    
    # Check if fuel rewards are now significant
    fuel_range = max(fuel_rewards) - min(fuel_rewards)
    total_range = max(rewards) - min(rewards)
    
    print(f"\nFuel reward variation: {fuel_range:.6f}")
    print(f"Total reward variation: {total_range:.6f}")
    print(f"Fuel contribution to total variation: {(fuel_range/total_range)*100:.1f}%")
    
    env.close()

if __name__ == "__main__":
    test_fuel_variation()