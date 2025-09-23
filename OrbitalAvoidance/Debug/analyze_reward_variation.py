#!/usr/bin/env python3
"""
Analyze reward variation to determine if it's sufficient for effective RL training.
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_reward_variation():
    """Analyze reward variation across different scenarios to assess training effectiveness."""
    print("=== REWARD VARIATION ANALYSIS FOR RL TRAINING ===")
    
    from rl_training_env import create_rl_environment
    
    env = create_rl_environment("scenarios/leo_debris_avoidance_fixed.json")
    
    # Test multiple scenarios
    scenarios = []
    
    print("1. Testing Action Magnitude Variation...")
    action_magnitudes = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, 0.2, ..., 1.0
    for magnitude in action_magnitudes:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        action = np.array([magnitude, 0.0, 0.0])
        obs, reward, done, truncated, info = env.step(action)
        
        scenarios.append({
            'type': 'action_magnitude',
            'parameter': magnitude,
            'total_reward': reward,
            'fuel_reward': env._calculate_fuel_reward(action),
            'orbit_reward': env._calculate_orbit_reward(),
            'collision_reward': env._calculate_collision_reward()
        })
    
    print("2. Testing Multi-Step Sequences...")
    # Test accumulated reward over multiple steps
    multi_step_rewards = []
    for strategy in ['conservative', 'aggressive', 'random']:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        total_accumulated = 0.0
        steps = 10
        
        for step in range(steps):
            if strategy == 'conservative':
                action = np.array([0.0, 0.0, 0.0])  # No action
            elif strategy == 'aggressive':
                action = np.array([1.0, 0.0, 0.0])  # Max thrust
            else:  # random
                action = np.random.uniform(-1, 1, 3)
            
            obs, reward, done, truncated, info = env.step(action)
            total_accumulated += reward
            
            if done or truncated:
                break
        
        multi_step_rewards.append({
            'strategy': strategy,
            'steps': step + 1,
            'total_accumulated': total_accumulated,
            'average_per_step': total_accumulated / (step + 1)
        })
    
    print("3. Testing Action Direction Variation...")
    directions = ['x+', 'x-', 'y+', 'y-', 'z+', 'z-', 'none']
    direction_rewards = []
    for direction in directions:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        if direction == 'x+':
            action = np.array([0.5, 0.0, 0.0])
        elif direction == 'x-':
            action = np.array([-0.5, 0.0, 0.0])
        elif direction == 'y+':
            action = np.array([0.0, 0.5, 0.0])
        elif direction == 'y-':
            action = np.array([0.0, -0.5, 0.0])
        elif direction == 'z+':
            action = np.array([0.0, 0.0, 0.5])
        elif direction == 'z-':
            action = np.array([0.0, 0.0, -0.5])
        else:  # none
            action = np.array([0.0, 0.0, 0.0])
        
        obs, reward, done, truncated, info = env.step(action)
        direction_rewards.append({
            'direction': direction,
            'total_reward': reward
        })
    
    # Analysis
    print("\n=== ANALYSIS RESULTS ===")
    
    # 1. Action magnitude variation
    action_rewards = [s['total_reward'] for s in scenarios if s['type'] == 'action_magnitude']
    action_range = max(action_rewards) - min(action_rewards)
    
    print(f"\n1. Action Magnitude Reward Variation:")
    print(f"   Reward range: {min(action_rewards):.6f} to {max(action_rewards):.6f}")
    print(f"   Total variation: {action_range:.6f}")
    print(f"   Relative variation: {(action_range/max(action_rewards))*100:.3f}%")
    
    # Show detailed breakdown for key points
    print(f"   Key points:")
    for i, magnitude in enumerate([0.0, 0.5, 1.0]):
        idx = int(magnitude * 10)  # Convert to index
        scenario = scenarios[idx]
        print(f"     Magnitude {magnitude:.1f}: {scenario['total_reward']:.6f}")
    
    # 2. Multi-step analysis
    print(f"\n2. Multi-Step Strategy Comparison:")
    for result in multi_step_rewards:
        print(f"   {result['strategy']:12s}: {result['total_accumulated']:.6f} total, {result['average_per_step']:.6f} avg/step")
    
    strategy_rewards = [r['total_accumulated'] for r in multi_step_rewards]
    strategy_range = max(strategy_rewards) - min(strategy_rewards)
    print(f"   Strategy variation: {strategy_range:.6f}")
    
    # 3. Direction analysis
    print(f"\n3. Action Direction Rewards:")
    dir_rewards = [r['total_reward'] for r in direction_rewards]
    for result in direction_rewards:
        print(f"   {result['direction']:4s}: {result['total_reward']:.6f}")
    
    direction_range = max(dir_rewards) - min(dir_rewards)
    print(f"   Direction variation: {direction_range:.6f}")
    
    # Overall assessment
    print(f"\n=== TRAINING EFFECTIVENESS ASSESSMENT ===")
    
    # Check if variation is sufficient for RL
    print(f"\nReward Signal Analysis:")
    print(f"• Single-step variation: {action_range:.6f}")
    print(f"• Multi-step variation: {strategy_range:.6f}")
    print(f"• Direction variation: {direction_range:.6f}")
    
    # Typical RL requirements
    min_meaningful_variation = 0.001  # Typical threshold for RL to learn
    optimal_variation = 0.1  # Good range for stable learning
    
    print(f"\nRL Training Assessment:")
    if action_range < min_meaningful_variation:
        print(f"❌ INSUFFICIENT: Single-step variation ({action_range:.6f}) < {min_meaningful_variation}")
        print("   Agent may not learn to distinguish between actions")
    elif action_range < optimal_variation:
        print(f"⚠️  MARGINAL: Single-step variation ({action_range:.6f}) is small but may work")
        print("   Learning might be slow, consider increasing fuel penalty weight")
    else:
        print(f"✅ SUFFICIENT: Single-step variation ({action_range:.6f}) should enable learning")
    
    if strategy_range < min_meaningful_variation * 10:  # Multi-step should have more variation
        print(f"❌ INSUFFICIENT: Multi-step variation ({strategy_range:.6f}) too small")
        print("   Agent may not learn long-term strategy differences")
    else:
        print(f"✅ SUFFICIENT: Multi-step variation ({strategy_range:.6f}) should enable strategy learning")
    
    # Fuel component analysis
    fuel_rewards = [s['fuel_reward'] for s in scenarios if s['type'] == 'action_magnitude']
    fuel_range = max(fuel_rewards) - min(fuel_rewards)
    print(f"\nFuel Reward Component:")
    print(f"• Fuel reward range: {min(fuel_rewards):.6f} to {max(fuel_rewards):.6f}")
    print(f"• Fuel variation: {fuel_range:.6f}")
    print(f"• Fuel contribution to total: {(fuel_range/action_range)*100:.1f}%")
    
    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    if action_range < optimal_variation:
        current_fuel_weight = env.reward_config['fuel']['efficiency_weight']
        recommended_weight = current_fuel_weight * (optimal_variation / action_range)
        print(f"• Consider increasing fuel efficiency_weight from {current_fuel_weight} to {recommended_weight:.3f}")
        
        current_bonus = env.reward_config['fuel']['conservation_bonus']
        recommended_bonus = current_bonus * (optimal_variation / action_range)
        print(f"• Consider increasing fuel conservation_bonus from {current_bonus} to {recommended_bonus:.6f}")
    else:
        print("• Current reward scaling appears appropriate for RL training")
    
    print(f"• Multi-step variation is {'adequate' if strategy_range > min_meaningful_variation * 10 else 'insufficient'}")
    
    env.close()

if __name__ == "__main__":
    analyze_reward_variation()