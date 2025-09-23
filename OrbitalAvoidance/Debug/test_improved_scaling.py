#!/usr/bin/env python3
"""
Test the improved reward scaling for better RL training signal.
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_improved_scaling():
    """Test the improved reward scaling."""
    print("=== TESTING IMPROVED REWARD SCALING ===")
    
    from rl_training_env import create_rl_environment
    
    env = create_rl_environment("scenarios/leo_debris_avoidance_fixed.json")
    
    test_cases = [
        {"action": np.array([0.0, 0.0, 0.0]), "name": "No Action"},
        {"action": np.array([0.1, 0.0, 0.0]), "name": "Small Thrust"},
        {"action": np.array([0.5, 0.0, 0.0]), "name": "Medium Thrust"},
        {"action": np.array([1.0, 0.0, 0.0]), "name": "Max Thrust"}
    ]
    
    results = []
    
    for test_case in test_cases:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        action = test_case["action"]
        name = test_case["name"]
        
        obs, total_reward, done, truncated, info = env.step(action)
        
        # Get components
        collision_reward = env._calculate_collision_reward()
        fuel_reward = env._calculate_fuel_reward(action)
        orbit_reward = env._calculate_orbit_reward()
        
        results.append({
            'name': name,
            'action_mag': np.linalg.norm(action),
            'total': total_reward,
            'collision': collision_reward,
            'fuel': fuel_reward,
            'orbit': orbit_reward
        })
        
        print(f"{name}:")
        print(f"  Total: {total_reward:.6f}")
        print(f"  Collision: {collision_reward:.6f}")
        print(f"  Fuel: {fuel_reward:.6f}")
        print(f"  Orbit: {orbit_reward:.6f}")
        print()
    
    # Analysis
    total_rewards = [r['total'] for r in results]
    fuel_rewards = [r['fuel'] for r in results]
    
    total_range = max(total_rewards) - min(total_rewards)
    fuel_range = max(fuel_rewards) - min(fuel_rewards)
    
    print("=== IMPROVED SCALING ANALYSIS ===")
    print(f"Total reward range: {min(total_rewards):.6f} to {max(total_rewards):.6f}")
    print(f"Total variation: {total_range:.6f}")
    print(f"Relative variation: {(total_range/max(total_rewards))*100:.3f}%")
    print()
    print(f"Fuel reward range: {min(fuel_rewards):.6f} to {max(fuel_rewards):.6f}")
    print(f"Fuel variation: {fuel_range:.6f}")
    print(f"Fuel contribution: {(fuel_range/total_range)*100:.1f}%")
    
    # Assessment
    print("\n=== TRAINING EFFECTIVENESS ===")
    min_good_variation = 0.01  # 1% is generally good for RL
    
    if total_range >= min_good_variation:
        print(f"✅ GOOD: Total variation ({total_range:.6f}) >= {min_good_variation}")
        print("   Should enable effective RL training!")
    elif total_range >= 0.001:
        print(f"⚠️  MARGINAL: Total variation ({total_range:.6f}) may work but could be slow")
    else:
        print(f"❌ INSUFFICIENT: Total variation ({total_range:.6f}) too small")
    
    # Test multi-step accumulation
    print("\n=== MULTI-STEP TEST ===")
    strategies = ['conservative', 'aggressive']
    
    for strategy in strategies:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        total_accumulated = 0.0
        steps = 5
        
        for step in range(steps):
            if strategy == 'conservative':
                action = np.array([0.0, 0.0, 0.0])
            else:
                action = np.array([1.0, 0.0, 0.0])
            
            obs, reward, done, truncated, info = env.step(action)
            total_accumulated += reward
            
            if done or truncated:
                break
        
        avg_reward = total_accumulated / (step + 1)
        print(f"{strategy:12s}: {total_accumulated:.6f} total, {avg_reward:.6f} avg")
    
    env.close()

if __name__ == "__main__":
    test_improved_scaling()