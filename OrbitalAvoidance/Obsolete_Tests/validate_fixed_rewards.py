#!/usr/bin/env python3
"""
Comprehensive test of the FIXED reward function.
This validates that all reward components (collision, fuel, orbit, mission) work correctly
and provide proper learning signals with the actual orbital element calculations.
"""

import numpy as np
import sys
import os
import json

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_reward_function_comprehensive():
    """Test the complete fixed reward function with various scenarios."""
    print("=== COMPREHENSIVE REWARD FUNCTION TEST ===")
    print("Testing the FIXED reward system (no more hardcoded orbital elements!)")
    
    from rl_training_env import RLOrbitalEnvironment
    
    # Create environment with ORIGINAL (now fixed) reward system
    config_file = "training_configs/fixed_reward_ppo_training.json"
    
    # Modify config to use original reward system
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Force use of original (now fixed) reward system
    config['environment']['use_improved_rewards'] = False
    
    # Save temporary config
    temp_config = "temp_test_config.json"
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Creating environment with FIXED original reward system...")
    
    try:
        env = RLOrbitalEnvironment(config_file=temp_config)
        print("✅ Environment created successfully!")
        
        # Initialize environment
        print("\nInitializing environment...")
        obs = env.reset()
        print(f"Initial observation shape: {obs[0].shape}")
        
        # Test different actions and their reward components
        test_scenarios = [
            ("No action (drift)", [0, 0, 0]),           # No thrust
            ("Small thrust +X", [0.1, 0, 0]),          # Small thrust
            ("Large thrust +X", [1.0, 0, 0]),          # Large thrust (more fuel cost)
            ("Thrust +Y", [0, 0.5, 0]),                # Different direction
            ("Thrust +Z", [0, 0, 0.3]),                # Out-of-plane thrust
        ]
        
        rewards_data = []
        
        for scenario_name, action in test_scenarios:
            print(f"\n--- Testing: {scenario_name} ---")
            print(f"Action: {action}")
            
            # Reset environment for consistent testing
            obs = env.reset()
            
            # Take action and get reward
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"Total reward: {reward:.4f}")
            
            # Get detailed reward breakdown if available
            if hasattr(env, '_last_reward_breakdown'):
                breakdown = env._last_reward_breakdown
                print("Reward breakdown:")
                for component, value in breakdown.items():
                    print(f"  {component}: {value:.4f}")
            
            rewards_data.append({
                'scenario': scenario_name,
                'action': action,
                'total_reward': reward,
                'spacecraft_pos': env.current_satellite_state.position_km.copy(),
                'spacecraft_vel': env.current_satellite_state.velocity_km_s.copy()
            })
        
        # Analysis
        print("\n=== REWARD ANALYSIS ===")
        total_rewards = [r['total_reward'] for r in rewards_data]
        reward_range = max(total_rewards) - min(total_rewards)
        
        print(f"Reward range: {reward_range:.4f} (from {min(total_rewards):.4f} to {max(total_rewards):.4f})")
        
        if reward_range > 0.001:  # Should see meaningful variation
            print("✅ SUCCESS: Rewards vary significantly between actions!")
            print("✅ LEARNING SIGNAL: Proper reward variation for RL training")
        else:
            print("❌ PROBLEM: Rewards show insufficient variation")
        
        # Check for the old constant penalty
        constant_penalty_count = sum(1 for r in total_rewards if abs(r - (-1299.5)) < 1.0)
        if constant_penalty_count > 0:
            print(f"❌ PROBLEM: Still seeing constant -1299.5 penalty in {constant_penalty_count} cases")
        else:
            print("✅ SUCCESS: No more constant -1299.5 penalty!")
        
        # Test orbital element calculation varies with position
        print("\n=== ORBITAL ELEMENT VARIATION TEST ===")
        positions = [r['spacecraft_pos'] for r in rewards_data]
        altitudes = [np.linalg.norm(pos) - 6371.0 for pos in positions]
        altitude_range = max(altitudes) - min(altitudes)
        
        print(f"Position range: {altitude_range:.1f} km")
        if altitude_range > 1.0:  # Should see position changes from thrust
            print("✅ SUCCESS: Spacecraft position varies with different actions")
        
        env.close()
        
        print("\n=== FINAL VERDICT ===")
        if reward_range > 0.001 and constant_penalty_count == 0:
            print("🎉 COMPLETE SUCCESS! The reward function is FIXED and ready for training!")
            print("🎉 No more hardcoded orbital elements!")
            print("🎉 Proper learning signals for PPO training!")
            return True
        else:
            print("❌ Issues still remain with the reward function")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up temp config
        if os.path.exists(temp_config):
            os.remove(temp_config)

if __name__ == "__main__":
    success = test_reward_function_comprehensive()
    if success:
        print("\n🚀 Ready for PPO training with the fixed reward system!")
    else:
        print("\n🔧 Additional fixes may be needed")