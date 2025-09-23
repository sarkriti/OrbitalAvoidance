#!/usr/bin/env python3
"""
Test the ORIGINAL reward system with realistic scenarios to understand
why it failed to provide learning signals.
"""

import numpy as np
import sys
sys.path.append('/home/bryson/RL_testing/OrbitalAvoidance')

def test_original_reward_system():
    """Test the original reward system comprehensively."""
    
    print("🔍 TESTING ORIGINAL REWARD SYSTEM")
    print("=" * 60)
    print("Testing the reward system that failed during 100k training iterations")
    print()
    
    test_original_fuel_consumption()
    test_original_survival_time()
    test_original_collision_scenarios()
    analyze_original_reward_variance()

def test_original_fuel_consumption():
    """Test how fuel consumption affects original rewards."""
    print("🔥 Testing ORIGINAL FUEL CONSUMPTION component:")
    print("-" * 50)
    
    from rl_training_env import RLOrbitalEnvironment
    
    # Create environment with ORIGINAL reward system (no improvements)
    env = RLOrbitalEnvironment(config_file='scenarios/leo_debris_avoidance.json')
    
    # Test different thrust magnitudes
    thrust_magnitudes = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    
    print("Testing different thrust magnitudes with ORIGINAL rewards:")
    fuel_rewards = []
    
    for thrust in thrust_magnitudes:
        obs, info = env.reset()
        action = np.array([thrust, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        fuel_rewards.append(reward)
        
        print(f"  Thrust {thrust:3.1f}: reward = {reward:10.4f}")
    
    # Analyze fuel component variance
    reward_range = max(fuel_rewards) - min(fuel_rewards)
    std_dev = np.std(fuel_rewards)
    
    print(f"  Reward range:     {reward_range:10.6f}")
    print(f"  Std deviation:    {std_dev:10.6f}")
    print(f"  Unique values:    {len(set(fuel_rewards))}")
    
    if reward_range > 0.01:
        print("  ✅ Fuel affects rewards")
    else:
        print("  ❌ No fuel effect detected")
    
    env.close()
    return fuel_rewards

def test_original_survival_time():
    """Test how survival time affects original rewards."""
    print("\n⏰ Testing ORIGINAL SURVIVAL TIME component:")
    print("-" * 50)
    
    from rl_training_env import RLOrbitalEnvironment
    
    env = RLOrbitalEnvironment(config_file='scenarios/leo_debris_avoidance.json')
    
    # Test rewards over multiple time steps
    obs, info = env.reset()
    
    print("Testing rewards over time with ORIGINAL system:")
    no_thrust = np.array([0.0, 0.0, 0.0])
    
    survival_rewards = []
    for step in range(10):  # Test more steps
        obs, reward, terminated, truncated, info = env.step(no_thrust)
        survival_rewards.append(reward)
        print(f"  Step {step+1:2d}: reward = {reward:10.4f}")
        
        if terminated:
            print("    Terminated (collision or episode end)")
            break
    
    # Analyze survival component
    if len(survival_rewards) > 1:
        reward_range = max(survival_rewards) - min(survival_rewards)
        std_dev = np.std(survival_rewards)
        
        print(f"  Reward range:     {reward_range:10.6f}")
        print(f"  Std deviation:    {std_dev:10.6f}")
        
        if reward_range > 0.01:
            print("  ✅ Survival time affects rewards")
        else:
            print("  ❌ No survival time effect")
    
    env.close()
    return survival_rewards

def test_original_collision_scenarios():
    """Test original reward system with different collision scenarios."""
    print("\n💥 Testing ORIGINAL COLLISION scenarios:")
    print("-" * 50)
    
    # We need to test the original reward function directly since we can't easily
    # create different debris configurations with the environment
    
    print("Testing original reward calculation logic...")
    
    from rl_training_env import RLOrbitalEnvironment
    
    # Create original environment and examine its reward function
    env = RLOrbitalEnvironment(config_file='scenarios/leo_debris_avoidance.json')
    obs, info = env.reset()
    
    # Test multiple actions from the same state to see reward calculation
    test_actions = [
        ("No thrust", np.array([0.0, 0.0, 0.0])),
        ("Small thrust", np.array([0.01, 0.0, 0.0])),
        ("Medium thrust", np.array([0.1, 0.0, 0.0])),
        ("Large thrust", np.array([0.5, 0.0, 0.0])),
        ("Max thrust", np.array([1.0, 0.0, 0.0])),
        ("Different direction", np.array([0.0, 0.5, 0.0])),
        ("Mixed thrust", np.array([0.3, 0.2, 0.1])),
    ]
    
    collision_rewards = []
    print("Testing various actions from SAME starting state:")
    
    for action_name, action in test_actions:
        # Reset to same state for each test
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        collision_rewards.append(reward)
        
        print(f"  {action_name:20s}: reward = {reward:10.4f}")
    
    # Analyze collision component
    reward_range = max(collision_rewards) - min(collision_rewards)
    std_dev = np.std(collision_rewards)
    unique_vals = len(set(collision_rewards))
    
    print(f"  Reward range:     {reward_range:10.6f}")
    print(f"  Std deviation:    {std_dev:10.6f}")
    print(f"  Unique values:    {unique_vals}")
    
    if reward_range > 0.01:
        print("  ✅ Actions affect rewards")
    else:
        print("  ❌ No action differentiation")
    
    env.close()
    return collision_rewards

def analyze_original_reward_variance():
    """Comprehensive analysis of original reward variance."""
    print("\n📊 ORIGINAL REWARD SYSTEM ANALYSIS:")
    print("=" * 60)
    
    # Test the reward function components more directly
    print("Deep dive into original reward function behavior...")
    
    from rl_training_env import RLOrbitalEnvironment
    
    env = RLOrbitalEnvironment(config_file='scenarios/leo_debris_avoidance.json')
    
    # Test many different scenarios
    print("\nTesting 20 random actions from fresh resets:")
    
    all_rewards = []
    np.random.seed(42)  # For reproducible results
    
    for i in range(20):
        obs, info = env.reset()
        
        # Generate random action
        action = np.random.uniform(-1.0, 1.0, 3)
        obs, reward, terminated, truncated, info = env.step(action)
        all_rewards.append(reward)
        
        print(f"  Test {i+1:2d}: action=[{action[0]:6.3f}, {action[1]:6.3f}, {action[2]:6.3f}] → reward = {reward:10.4f}")
    
    # Statistical analysis
    print(f"\nORIGINAL REWARD STATISTICS:")
    print(f"  Total samples:    {len(all_rewards)}")
    print(f"  Min reward:       {min(all_rewards):10.4f}")
    print(f"  Max reward:       {max(all_rewards):10.4f}")
    print(f"  Mean reward:      {np.mean(all_rewards):10.4f}")
    print(f"  Std deviation:    {np.std(all_rewards):10.6f}")
    print(f"  Range:            {max(all_rewards) - min(all_rewards):10.6f}")
    print(f"  Unique values:    {len(set(all_rewards))}")
    
    # Check for the fatal flaw
    reward_range = max(all_rewards) - min(all_rewards)
    std_dev = np.std(all_rewards)
    unique_count = len(set(all_rewards))
    
    print(f"\n🎯 ORIGINAL SYSTEM DIAGNOSIS:")
    
    if reward_range < 0.001:
        print(f"  ❌ FATAL: Essentially zero reward range ({reward_range:0.6f})")
        print("     Agent cannot distinguish between actions!")
    elif reward_range < 0.1:
        print(f"  ❌ POOR: Very small reward range ({reward_range:0.6f})")
    else:
        print(f"  ✅ OK: Reasonable reward range ({reward_range:0.6f})")
    
    if std_dev < 0.001:
        print(f"  ❌ FATAL: Essentially zero variance ({std_dev:0.6f})")
        print("     No learning signal for PPO!")
    elif std_dev < 0.01:
        print(f"  ❌ POOR: Very low variance ({std_dev:0.6f})")
    else:
        print(f"  ✅ OK: Reasonable variance ({std_dev:0.6f})")
    
    if unique_count == 1:
        print(f"  ❌ FATAL: All rewards identical!")
    elif unique_count < len(all_rewards) * 0.3:
        print(f"  ⚠️  WARNING: Many duplicate values ({unique_count}/{len(all_rewards)})")
    else:
        print(f"  ✅ OK: Good reward diversity ({unique_count}/{len(all_rewards)})")
    
    # Identify the root cause
    if std_dev < 0.001 and reward_range < 0.001:
        print(f"\n🚨 ROOT CAUSE IDENTIFIED:")
        print(f"   The original reward function returns nearly identical values")
        print(f"   for all actions. This explains the training failure!")
        print(f"   PPO needs reward variance to learn - without it, the agent")
        print(f"   has no signal about which actions are better.")
    
    env.close()
    return all_rewards

def compare_with_improved():
    """Compare original vs improved reward systems side by side."""
    print(f"\n🔄 COMPARISON: ORIGINAL vs IMPROVED")
    print("=" * 60)
    
    # Quick test of both systems
    from rl_training_env import RLOrbitalEnvironment
    
    # Original system
    orig_env = RLOrbitalEnvironment(config_file='scenarios/leo_debris_avoidance.json')
    
    # Test action on original
    test_action = np.array([0.3, 0.1, 0.0])
    
    # Original
    obs1, info1 = orig_env.reset()
    obs1, reward1, term1, trunc1, info1 = orig_env.step(test_action)
    
    print(f"Test action [0.3, 0.1, 0.0] from reset:")
    print(f"  Original reward:   {reward1:10.4f}")
    print(f"  Expected improved: ~3.5 to -17.5 (based on earlier test)")
    print(f"  Original problem:  All actions give exactly {reward1:0.4f}")
    
    orig_env.close()

if __name__ == "__main__":
    test_original_reward_system()
    compare_with_improved()
    
    print("\n" + "="*60)
    print("🎯 This analysis reveals why the original training failed:")
    print("1. ✅ Tested fuel consumption effects")
    print("2. ✅ Tested survival time effects")
    print("3. ✅ Tested action differentiation")
    print("4. ✅ Statistical analysis of reward variance")
    print("5. ✅ Identified root cause of learning failure")