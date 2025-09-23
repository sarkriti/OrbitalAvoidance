#!/usr/bin/env python3
"""
Check all reward components to see what's dominating the total.
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_reward_components():
    """Analyze all reward components to see what's dominating."""
    print("=== ANALYZING ALL REWARD COMPONENTS ===")
    
    from rl_training_env import create_rl_environment
    
    env = create_rl_environment("scenarios/leo_debris_avoidance_fixed.json")
    env.reset()
    
    # Test with no action to see baseline rewards
    action = np.array([0.0, 0.0, 0.0])
    
    print("=== INDIVIDUAL REWARD COMPONENTS (No Action) ===")
    
    try:
        collision_reward = env._calculate_collision_reward()
        print(f"Collision reward: {collision_reward:.8f}")
    except Exception as e:
        print(f"Collision reward error: {e}")
    
    try:
        fuel_reward = env._calculate_fuel_reward(action)
        print(f"Fuel reward: {fuel_reward:.8f}")
    except Exception as e:
        print(f"Fuel reward error: {e}")
    
    try:
        orbit_reward = env._calculate_orbit_reward()
        print(f"Orbit reward: {orbit_reward:.8f}")
    except Exception as e:
        print(f"Orbit reward error: {e}")
    
    try:
        mission_reward = env._calculate_mission_reward()
        print(f"Mission reward: {mission_reward:.8f}")
    except Exception as e:
        print(f"Mission reward error: {e}")
    
    # Check reward weights
    print(f"\n=== REWARD WEIGHTS ===")
    if hasattr(env, 'reward_config') and 'weights' in env.reward_config:
        weights = env.reward_config['weights']
        print(f"Collision weight: {weights.get('collision', 'Not found')}")
        print(f"Fuel weight: {weights.get('fuel', 'Not found')}")
        print(f"Orbit weight: {weights.get('orbit', 'Not found')}")
        print(f"Mission weight: {weights.get('mission', 'Not found')}")
    
    # Calculate total manually
    try:
        collision_reward = env._calculate_collision_reward()
        fuel_reward = env._calculate_fuel_reward(action)
        orbit_reward = env._calculate_orbit_reward()
        mission_reward = env._calculate_mission_reward()
        
        weights = env.reward_config['weights']
        
        weighted_collision = weights['collision'] * collision_reward
        weighted_fuel = weights['fuel'] * fuel_reward
        weighted_orbit = weights['orbit'] * orbit_reward
        weighted_mission = weights['mission'] * mission_reward
        
        manual_total = weighted_collision + weighted_fuel + weighted_orbit + weighted_mission
        
        print(f"\n=== WEIGHTED COMPONENTS ===")
        print(f"Weighted collision: {weights['collision']} * {collision_reward:.8f} = {weighted_collision:.8f}")
        print(f"Weighted fuel: {weights['fuel']} * {fuel_reward:.8f} = {weighted_fuel:.8f}")
        print(f"Weighted orbit: {weights['orbit']} * {orbit_reward:.8f} = {weighted_orbit:.8f}")
        print(f"Weighted mission: {weights['mission']} * {mission_reward:.8f} = {weighted_mission:.8f}")
        print(f"Manual total: {manual_total:.8f}")
        
        # Get actual environment reward
        obs, actual_reward, done, truncated, info = env.step(action)
        print(f"Actual env reward: {actual_reward:.8f}")
        
        print(f"\n=== ANALYSIS ===")
        largest_component = max([
            ("Collision", abs(weighted_collision)),
            ("Fuel", abs(weighted_fuel)),
            ("Orbit", abs(weighted_orbit)),
            ("Mission", abs(weighted_mission))
        ], key=lambda x: x[1])
        
        print(f"Largest component: {largest_component[0]} with magnitude {largest_component[1]:.8f}")
        
        if largest_component[1] > 1.0:
            print(f"⚠️  WARNING: {largest_component[0]} component is dominating (magnitude > 1.0)")
            print(f"⚠️  This may overwhelm the fuel reward differences (~0.000001 to 0.00002)")
        
    except Exception as e:
        print(f"Error in manual calculation: {e}")
        import traceback
        traceback.print_exc()
    
    env.close()

if __name__ == "__main__":
    analyze_reward_components()