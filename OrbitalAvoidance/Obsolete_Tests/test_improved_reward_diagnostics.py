#!/usr/bin/env python3
"""
Test the improved reward function against the same diagnostic criteria
that identified the original broken reward function.
"""

import numpy as np
import sys
sys.path.append('/home/bryson/RL_testing/OrbitalAvoidance')

def test_improved_reward_function():
    """Test if improved reward function passes all diagnostic checks."""
    
    print("🧪 TESTING IMPROVED REWARD FUNCTION")
    print("=" * 60)
    print("Running the same diagnostic tests that identified the broken reward...")
    
    # Create environment with improved rewards
    from rl_training_env import RLOrbitalEnvironment
    
    reward_config = {
        'use_improved_rewards': True,
        'collision_penalty': -50.0,
        'fuel_cost_weight': 0.01,
        'safety_distance_km': 5.0,
        'survival_reward': 0.5,
        'distance_reward_scale': 1.0
    }
    
    env = RLOrbitalEnvironment(
        config_file='scenarios/leo_debris_avoidance_fixed.json',
        reward_config=reward_config
    )
    
    obs, info = env.reset()
    
    print("\n🔍 REWARD VARIATION TEST")
    print("-" * 40)
    
    # Test the exact same actions that revealed the original problem
    actions_to_test = [
        ("No thrust", np.array([0.0, 0.0, 0.0])),
        ("Small thrust X", np.array([0.1, 0.0, 0.0])),
        ("Medium thrust X", np.array([0.5, 0.0, 0.0])),
        ("Large thrust X", np.array([1.0, 0.0, 0.0])),
        ("Small thrust Y", np.array([0.0, 0.1, 0.0])),
        ("Small thrust Z", np.array([0.0, 0.0, 0.1])),
        ("Random thrust", np.array([0.3, -0.2, 0.4])),
    ]
    
    rewards = []
    
    print("Testing different actions:")
    for action_name, action in actions_to_test:
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        print(f"  {action_name:15s}: reward = {reward:8.4f}")
        
        if terminated:
            print(f"    ⚠️  Terminated! (likely collision)")
            obs, info = env.reset()
    
    # Apply the same diagnostic criteria
    print(f"\n📊 REWARD STATISTICS:")
    print(f"  Min reward:    {min(rewards):8.4f}")
    print(f"  Max reward:    {max(rewards):8.4f}")
    print(f"  Mean reward:   {np.mean(rewards):8.4f}")
    print(f"  Std reward:    {np.std(rewards):8.4f}")
    print(f"  Range:         {max(rewards) - min(rewards):8.4f}")
    
    print(f"\n🧪 DIAGNOSTIC CHECKS:")
    
    # Check 1: Range too small
    reward_range = max(rewards) - min(rewards)
    if reward_range < 0.1:
        print("❌ FAIL: Reward range is too small (<0.1)")
        print(f"   Range: {reward_range:.6f}")
    else:
        print("✅ PASS: Reward range is sufficient")
        print(f"   Range: {reward_range:.4f} (> 0.1)")
    
    # Check 2: Rewards too large
    mean_reward = abs(np.mean(rewards))
    if mean_reward > 100:
        print("❌ FAIL: Rewards are too large (mean > 100)")
        print(f"   Mean: {mean_reward:.4f}")
    else:
        print("✅ PASS: Reward scale is reasonable")
        print(f"   Mean: {mean_reward:.4f} (< 100)")
    
    # Check 3: No variance (THE CRITICAL TEST!)
    reward_std = np.std(rewards)
    if reward_std < 0.01:
        print("❌ FAIL: Reward variance is too low")
        print(f"   Std: {reward_std:.6f} - No learning signal!")
    else:
        print("✅ PASS: Reward has sufficient variance")
        print(f"   Std: {reward_std:.4f} (> 0.01)")
    
    # Check 4: All rewards identical (the fatal flaw)
    unique_rewards = len(set(np.round(rewards, 4)))
    if unique_rewards == 1:
        print("❌ FAIL: All rewards are identical!")
        print("   This provides zero learning signal")
    else:
        print("✅ PASS: Rewards show variation")
        print(f"   {unique_rewards} unique reward values found")
    
    print(f"\n🎯 COMPARISON WITH ORIGINAL:")
    print(f"  Original reward: ALL actions = -1298.5192 (std = 0.0000)")
    print(f"  Improved reward: Range {min(rewards):.1f} to {max(rewards):.1f} (std = {np.std(rewards):.4f})")
    
    # Overall assessment
    all_checks_pass = (
        reward_range >= 0.1 and 
        mean_reward <= 100 and 
        reward_std >= 0.01 and 
        unique_rewards > 1
    )
    
    print(f"\n🏆 OVERALL RESULT:")
    if all_checks_pass:
        print("✅ SUCCESS: Improved reward function PASSES all diagnostic tests!")
        print("   The reward function now provides clear learning signals.")
        print("   PPO training should be able to learn with this reward function.")
    else:
        print("❌ FAILURE: Improved reward function still has issues")
        print("   Additional tuning may be needed.")
    
    env.close()
    return all_checks_pass

if __name__ == "__main__":
    success = test_improved_reward_function()
    sys.exit(0 if success else 1)