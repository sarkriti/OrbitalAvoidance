#!/usr/bin/env python3
"""
Test the updated reward system with positive incentives for fuel conservation and orbit maintenance.
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_positive_rewards():
    """Test that new positive rewards work correctly."""
    print("=== TESTING UPDATED REWARD SYSTEM WITH POSITIVE INCENTIVES ===")
    
    from rl_training_env import create_rl_environment
    
    env = create_rl_environment("scenarios/leo_debris_avoidance_fixed.json")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    test_cases = [
        {"action": np.array([0.0, 0.0, 0.0]), "name": "No Action (Conservation)"},
        {"action": np.array([0.1, 0.0, 0.0]), "name": "Small Thrust"},
        {"action": np.array([0.5, 0.0, 0.0]), "name": "Medium Thrust"},
        {"action": np.array([1.0, 0.0, 0.0]), "name": "Maximum Thrust"}
    ]
    
    results = []
    
    for test_case in test_cases:
        # Reset environment for each test
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        action = test_case["action"]
        name = test_case["name"]
        
        # Take the action
        obs, total_reward, done, truncated, info = env.step(action)
        
        # Get individual reward components
        collision_reward = env._calculate_collision_reward()
        fuel_reward = env._calculate_fuel_reward(action)
        orbit_reward = env._calculate_orbit_reward()
        mission_reward = env._calculate_mission_reward()
        
        # Calculate theoretical total (for verification)
        weights = env.reward_config['weights']
        theoretical_total = (
            weights['collision'] * collision_reward +
            weights['fuel'] * fuel_reward +
            weights['orbit'] * orbit_reward +
            weights['mission'] * mission_reward
        )
        
        result = {
            'name': name,
            'action_magnitude': np.linalg.norm(action),
            'total_reward': total_reward,
            'collision_reward': collision_reward,
            'fuel_reward': fuel_reward,
            'orbit_reward': orbit_reward,
            'mission_reward': mission_reward,
            'theoretical_total': theoretical_total,
            'fuel_remaining': env.current_satellite_state.fuel_remaining_kg
        }
        results.append(result)
        
        print(f"\n{name}:")
        print(f"  Action magnitude: {result['action_magnitude']:.3f}")
        print(f"  Total reward: {result['total_reward']:.6f}")
        print(f"  Collision reward: {result['collision_reward']:.6f}")
        print(f"  Fuel reward: {result['fuel_reward']:.6f} {'(CONSERVATION BONUS!)' if result['fuel_reward'] > 0 else ''}")
        print(f"  Orbit reward: {result['orbit_reward']:.6f} {'(MAINTENANCE BONUS!)' if result['orbit_reward'] > 0 else ''}")
        print(f"  Mission reward: {result['mission_reward']:.6f}")
        print(f"  Fuel remaining: {result['fuel_remaining']:.3f} kg")
        print(f"  Theoretical total: {result['theoretical_total']:.6f}")
    
    print("\n=== REWARD ANALYSIS ===")
    
    # Check fuel conservation incentive
    no_action_result = results[0]
    small_thrust_result = results[1]
    
    print(f"\nFuel Conservation Analysis:")
    print(f"No action fuel reward: {no_action_result['fuel_reward']:.6f}")
    print(f"Small thrust fuel reward: {small_thrust_result['fuel_reward']:.6f}")
    
    if no_action_result['fuel_reward'] > small_thrust_result['fuel_reward']:
        print("✅ Fuel conservation incentive working: No action gets higher fuel reward!")
    else:
        print("❌ Fuel conservation incentive not working properly")
    
    # Check orbit maintenance bonus
    orbit_rewards = [r['orbit_reward'] for r in results]
    if any(r > 0 for r in orbit_rewards):
        print(f"\n✅ Orbit maintenance bonus working: Found positive orbit rewards!")
        print(f"Orbit reward range: {min(orbit_rewards):.6f} to {max(orbit_rewards):.6f}")
    else:
        print(f"\n❌ Orbit maintenance bonus not working: All orbit rewards negative/zero")
        print(f"Orbit reward range: {min(orbit_rewards):.6f} to {max(orbit_rewards):.6f}")
    
    # Check total reward trend
    total_rewards = [r['total_reward'] for r in results]
    action_magnitudes = [r['action_magnitude'] for r in results]
    
    print(f"\nTotal Reward vs Action Magnitude:")
    for result in results:
        print(f"  {result['action_magnitude']:.1f}: {result['total_reward']:.6f}")
    
    print(f"\nTotal reward range: {min(total_rewards):.6f} to {max(total_rewards):.6f}")
    print(f"Reward variation: {max(total_rewards) - min(total_rewards):.6f}")
    
    # Verify conservation is rewarded
    if total_rewards[0] > total_rewards[-1]:  # No action > max thrust
        print("✅ Conservation rewarded: No action gets higher total reward than maximum thrust!")
    else:
        print("❌ Conservation not properly rewarded")
    
    env.close()

if __name__ == "__main__":
    test_positive_rewards()