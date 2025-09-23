#!/usr/bin/env python3
"""
Test the fixed orbit reward system that compares to reference orbital elements.
"""

import numpy as np
import sys
import os
import math

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_fixed_orbit_reward():
    """Test the fixed orbit reward system."""
    print("=== TESTING FIXED ORBIT REWARD SYSTEM ===")
    
    from rl_training_env import create_rl_environment
    
    # Create environment
    env = create_rl_environment("scenarios/leo_debris_avoidance_fixed.json")
    
    # Reset to initialize reference orbit
    print("Resetting environment to calculate reference orbital elements...")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    print(f"✅ Environment reset complete")
    
    # Check the reference orbital elements
    print(f"\n=== REFERENCE ORBITAL ELEMENTS (from initial state) ===")
    ref = env.reference_orbit
    print(f"Semi-major axis: {ref.semi_major_axis_km:.3f} km ({ref.semi_major_axis_km - 6371:.1f} km altitude)")
    print(f"Eccentricity: {ref.eccentricity:.6f}")
    print(f"Inclination: {ref.inclination_deg:.3f}°")
    print(f"RAAN: {ref.raan_deg:.3f}°")
    print(f"Arg Periapsis: {ref.arg_periapsis_deg:.3f}°")
    print(f"Mean Anomaly: {ref.mean_anomaly_deg:.3f}°")
    
    # Test orbit reward calculation at start (should be ~0)
    print(f"\n=== INITIAL ORBIT REWARD ===")
    initial_orbit_reward = env._calculate_orbit_reward()
    print(f"Initial orbit reward: {initial_orbit_reward:.6f}")
    
    if abs(initial_orbit_reward) < 0.001:
        print("✅ SUCCESS: Initial orbit reward is near zero (no deviation from reference)")
    else:
        print(f"⚠️  WARNING: Initial orbit reward is {initial_orbit_reward:.6f}, expected ~0")
    
    # Test different actions and see how orbit reward changes
    test_actions = [
        ("No thrust", np.array([0.0, 0.0, 0.0])),
        ("Small thrust", np.array([0.1, 0.0, 0.0])),
        ("Medium thrust", np.array([0.5, 0.0, 0.0])),
        ("Large thrust", np.array([1.0, 0.0, 0.0])),
    ]
    
    print(f"\n=== TESTING ORBIT REWARD WITH DIFFERENT ACTIONS ===")
    
    rewards = []
    orbit_rewards = []
    
    for description, action in test_actions:
        # Reset for clean test
        env.reset()
        
        # Take one step with the action
        obs, total_reward, done, truncated, info = env.step(action)
        
        # Get individual reward components
        orbit_reward = env._calculate_orbit_reward()
        fuel_reward = env._calculate_fuel_reward(action)
        collision_reward = env._calculate_collision_reward()
        mission_reward = env._calculate_mission_reward()
        
        # Calculate expected total
        weights = env.reward_config['weights']
        expected_total = (weights['orbit'] * orbit_reward + 
                         weights['fuel'] * fuel_reward +
                         weights['collision'] * collision_reward +
                         weights['mission'] * mission_reward)
        
        print(f"\n{description}:")
        print(f"  Action: {action}")
        print(f"  Orbit reward: {orbit_reward:.6f}")
        print(f"  Fuel reward: {fuel_reward:.6f}")
        print(f"  Collision reward: {collision_reward:.6f}")
        print(f"  Mission reward: {mission_reward:.6f}")
        print(f"  Expected total: {expected_total:.6f}")
        print(f"  Actual total: {total_reward:.6f}")
        
        rewards.append(total_reward)
        orbit_rewards.append(orbit_reward)
        
        # Check if total matches expected
        if abs(total_reward - expected_total) < 0.001:
            print(f"  ✅ Total reward calculation correct")
        else:
            print(f"  ❌ Total reward mismatch!")
    
    # Analysis
    print(f"\n=== ANALYSIS ===")
    reward_range = max(rewards) - min(rewards)
    orbit_reward_range = max(orbit_rewards) - min(orbit_rewards)
    
    print(f"Total reward range: {reward_range:.6f}")
    print(f"Orbit reward range: {orbit_reward_range:.6f}")
    
    if reward_range > 0.001:
        print("✅ SUCCESS: Total rewards vary significantly with actions")
    else:
        print("❌ PROBLEM: Total rewards show insufficient variation")
    
    if orbit_reward_range > 0.001:
        print("✅ SUCCESS: Orbit rewards vary with actions (thrust affects orbit)")
    else:
        print("⚠️  INFO: Orbit rewards consistent (may be expected for small thrusts)")
    
    # Check that we're not seeing the old -2.9 penalty
    large_penalty_count = sum(1 for r in rewards if r < -2.0)
    if large_penalty_count == 0:
        print("✅ SUCCESS: No large negative penalties detected")
    else:
        print(f"❌ PROBLEM: {large_penalty_count} cases with large negative penalties")
    
    print(f"\nReward values: {[f'{r:.6f}' for r in rewards]}")
    
    env.close()
    
    return reward_range > 0.001 and large_penalty_count == 0

if __name__ == "__main__":
    success = test_fixed_orbit_reward()
    if success:
        print("\n🎉 ORBIT REWARD SYSTEM FIXED!")
        print("🎉 Ready for proper PPO training!")
    else:
        print("\n🔧 May need additional investigation")