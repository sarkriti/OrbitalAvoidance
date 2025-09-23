#!/usr/bin/env python3
"""
Test the improved reward function against the same diagnostic criteria
that revealed the original reward function was broken.
"""

import numpy as np
import sys
sys.path.append('/home/bryson/RL_testing/OrbitalAvoidance')

def test_improved_reward_function():
    """Test improved reward function with the same diagnostic criteria."""
    
    print("🧪 TESTING IMPROVED REWARD FUNCTION")
    print("=" * 50)
    print("Running the same diagnostic tests that revealed")
    print("the original reward function was broken...")
    print()
    
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
        max_debris_tracked=10,
        reward_config=reward_config
    )
    
    print("✅ Environment created with improved rewards")
    
    # Test 1: Reward Variation Analysis (same as diagnostic script)
    print("\n🔍 TEST 1: REWARD VARIATION ANALYSIS")
    print("=" * 40)
    
    obs, info = env.reset()
    
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
    
    reward_ranges = []
    
    print("Testing reward function with different actions:")
    for action_name, action in actions_to_test:
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        reward_ranges.append(reward)
        
        print(f"  {action_name:15s}: reward = {reward:8.4f}")
        
        if terminated:
            print(f"    ⚠️  Terminated! (likely collision)")
            obs, info = env.reset()
    
    # Apply the same diagnostic criteria
    print(f"\nReward Statistics:")
    print(f"  Min reward:    {min(reward_ranges):8.4f}")
    print(f"  Max reward:    {max(reward_ranges):8.4f}")
    print(f"  Mean reward:   {np.mean(reward_ranges):8.4f}")
    print(f"  Std reward:    {np.std(reward_ranges):8.4f}")
    print(f"  Range:         {max(reward_ranges) - min(reward_ranges):8.4f}")
    
    # Test 2: Apply the same failure criteria that detected the original problem
    print(f"\n🔍 TEST 2: DIAGNOSTIC FAILURE CRITERIA")
    print("=" * 40)
    
    # Check 1: Range too small (original diagnostic criteria)
    range_val = max(reward_ranges) - min(reward_ranges)
    if range_val < 0.1:
        print("❌ FAILED: Reward range is too small (<0.1)")
        print("   This makes it hard for the agent to distinguish good/bad actions")
        failed_range = True
    else:
        print(f"✅ PASSED: Reward range ({range_val:.4f}) is sufficient (>0.1)")
        failed_range = False
    
    # Check 2: Rewards too large (original diagnostic criteria)
    mean_abs = abs(np.mean(reward_ranges))
    if mean_abs > 100:
        print(f"❌ FAILED: Rewards are too large (mean abs = {mean_abs:.2f} > 100)")
        print("   This can cause numerical instability")
        failed_magnitude = True
    else:
        print(f"✅ PASSED: Reward magnitude ({mean_abs:.2f}) is reasonable (<100)")
        failed_magnitude = False
    
    # Check 3: No variance (THE KILLER that broke the original)
    std_val = np.std(reward_ranges)
    if std_val < 0.01:
        print(f"❌ FAILED: Reward variance is too low ({std_val:.6f})")
        print("   All actions give similar rewards - no learning signal")
        failed_variance = True
    else:
        print(f"✅ PASSED: Reward variance ({std_val:.4f}) provides learning signal")
        failed_variance = False
    
    # Test 3: Additional quality checks
    print(f"\n🔍 TEST 3: QUALITY ASSESSMENT")
    print("=" * 40)
    
    # Check for distinct reward values
    unique_rewards = len(set(reward_ranges))
    print(f"  Unique reward values: {unique_rewards}/{len(reward_ranges)}")
    
    if unique_rewards == 1:
        print("❌ CRITICAL: All rewards are identical!")
    elif unique_rewards < 3:
        print("⚠️  WARNING: Very few distinct reward values")
    else:
        print("✅ GOOD: Multiple distinct reward values")
    
    # Check for reasonable scaling
    reward_range = max(reward_ranges) - min(reward_ranges)
    if reward_range > 10:
        print(f"  Reward scaling: Good range ({reward_range:.2f})")
    else:
        print(f"  Reward scaling: Moderate range ({reward_range:.2f})")
    
    env.close()
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT")
    print("=" * 40)
    
    total_failures = sum([failed_range, failed_magnitude, failed_variance])
    
    if total_failures == 0:
        print("🎉 SUCCESS: Improved reward function PASSES all diagnostic tests!")
        print("   This reward function should enable successful RL training.")
        print(f"   • Reward range: {range_val:.4f} (good variation)")
        print(f"   • Reward magnitude: {mean_abs:.2f} (reasonable scale)")
        print(f"   • Reward variance: {std_val:.4f} (strong learning signal)")
        return True
    else:
        print(f"❌ FAILURE: {total_failures}/3 diagnostic tests failed")
        print("   This reward function may still have issues.")
        return False

if __name__ == "__main__":
    success = test_improved_reward_function()
    sys.exit(0 if success else 1)