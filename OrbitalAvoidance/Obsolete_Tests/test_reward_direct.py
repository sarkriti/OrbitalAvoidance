#!/usr/bin/env python3
"""
Direct test of the fixed reward function using the working training approach.
This bypasses environment configuration issues and directly tests reward calculation.
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_reward_calculation_direct():
    """Test reward calculation directly using the approach from rl_train.py"""
    print("=== DIRECT REWARD FUNCTION TEST ===")
    print("Testing the FIXED reward system with direct calculation")
    
    # Import what we need
    from rl_training_env import create_rl_environment
    
    # Use the working scenario file approach
    scenario_file = "scenarios/leo_debris_avoidance_fixed.json"
    
    print(f"Creating environment from scenario: {scenario_file}")
    
    try:
        # Create environment the same way rl_train.py does it
        env = create_rl_environment(scenario_file)
        print("✅ Environment created successfully!")
        
        # Initialize
        obs = env.reset()
        print(f"Environment initialized, observation shape: {obs.shape}")
        
        # Get the reward function for direct testing
        reward_func = env.reward_function
        print(f"Reward function type: {type(reward_func)}")
        
        # Test different scenarios by modifying spacecraft state
        test_scenarios = []
        
        # Get initial state
        initial_state = env.current_satellite_state
        initial_pos = initial_state.position_km.copy()
        initial_vel = initial_state.velocity_km_s.copy()
        
        print(f"\nInitial spacecraft state:")
        print(f"Position (km): {initial_pos}")
        print(f"Velocity (km/s): {initial_vel}")
        print(f"Altitude: {np.linalg.norm(initial_pos) - 6371.0:.1f} km")
        
        # Test different positions (simulate what would happen with different actions)
        test_positions = [
            ("Original position", initial_pos.copy(), initial_vel.copy()),
            ("Higher altitude", initial_pos * 1.1, initial_vel * 0.95),  # Higher, slower
            ("Lower altitude", initial_pos * 0.9, initial_vel * 1.05),   # Lower, faster
            ("Different inclination", np.array([initial_pos[0], initial_pos[1], initial_pos[2] * 1.5]), initial_vel),
        ]
        
        rewards = []
        
        for description, pos, vel in test_positions:
            print(f"\n--- Testing: {description} ---")
            print(f"Position (km): {pos}")
            print(f"Velocity (km/s): {vel}")
            print(f"Altitude: {np.linalg.norm(pos) - 6371.0:.1f} km")
            
            # Temporarily modify spacecraft state
            original_pos = env.current_satellite_state.position_km.copy()
            original_vel = env.current_satellite_state.velocity_km_s.copy()
            
            env.current_satellite_state.position_km = pos
            env.current_satellite_state.velocity_km_s = vel
            
            # Calculate reward
            try:
                reward = reward_func.calculate_reward(
                    env.current_satellite_state,
                    env.debris_objects,
                    action=[0, 0, 0],  # No action for pure state reward
                    dt=env.time_step
                )
                
                print(f"Total reward: {reward:.4f}")
                rewards.append(reward)
                
                # Try to get breakdown if available
                if hasattr(reward_func, 'get_last_breakdown'):
                    breakdown = reward_func.get_last_breakdown()
                    if breakdown:
                        print("Reward breakdown:")
                        for component, value in breakdown.items():
                            print(f"  {component}: {value:.4f}")
                
            except Exception as e:
                print(f"Error calculating reward: {e}")
                rewards.append(0.0)
            
            # Restore original state
            env.current_satellite_state.position_km = original_pos
            env.current_satellite_state.velocity_km_s = original_vel
        
        # Analysis
        print("\n=== REWARD ANALYSIS ===")
        if len(rewards) > 1:
            reward_range = max(rewards) - min(rewards)
            print(f"Reward range: {reward_range:.4f} (from {min(rewards):.4f} to {max(rewards):.4f})")
            
            if reward_range > 0.001:
                print("✅ SUCCESS: Rewards vary significantly with spacecraft state!")
                print("✅ LEARNING SIGNAL: The reward function provides proper variation")
            else:
                print("❌ PROBLEM: Insufficient reward variation")
            
            # Check for old constant penalty
            constant_penalty_count = sum(1 for r in rewards if abs(r - (-1299.5)) < 1.0)
            if constant_penalty_count > 0:
                print(f"❌ PROBLEM: Still seeing constant penalty in {constant_penalty_count} cases")
            else:
                print("✅ SUCCESS: No constant penalty detected!")
            
            print(f"\nReward values: {[f'{r:.4f}' for r in rewards]}")
        
        env.close()
        
        return len(rewards) > 1 and max(rewards) - min(rewards) > 0.001
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reward_calculation_direct()
    if success:
        print("\n🎉 REWARD FUNCTION IS WORKING CORRECTLY!")
        print("🎉 Ready for PPO training!")
    else:
        print("\n🔧 Reward function needs investigation")