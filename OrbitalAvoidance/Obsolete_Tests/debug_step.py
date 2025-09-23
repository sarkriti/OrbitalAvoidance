#!/usr/bin/env python3
"""
Debug the step method to see what's happening to the reward.
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_step_method():
    """Debug the step method reward calculation."""
    print("=== DEBUGGING STEP METHOD ===")
    
    from rl_training_env import create_rl_environment
    
    env = create_rl_environment("scenarios/leo_debris_avoidance_fixed.json")
    env.reset()
    
    action = np.array([0.0, 0.0, 0.0])
    
    print("=== BEFORE STEP ===")
    reward_func = env.reward_function
    direct_reward = reward_func(action)
    print(f"Direct reward function call: {direct_reward:.8f}")
    
    # Let's manually trace through the step method
    print(f"\n=== MANUAL STEP TRACE ===")
    
    # Save original state
    original_pos = env.current_satellite_state.position_km.copy()
    
    # Convert action to base environment format (this is what step() does)
    thrust_newtons = action * env.main_satellite.max_thrust_n
    base_action = thrust_newtons
    
    print(f"Base action: {base_action}")
    
    # Call base environment step
    base_obs, base_reward, base_terminated, base_truncated, base_info = env.base_env.step(base_action)
    
    print(f"Base reward: {base_reward:.8f}")
    print(f"Base terminated: {base_terminated}")
    print(f"Base info: {base_info}")
    
    # Update states (this is what step() does)
    satellite_dict = env.base_env.satellites[env.main_satellite_name]
    env.current_satellite_state.update_from_dict(satellite_dict)
    
    # Check if position changed
    new_pos = env.current_satellite_state.position_km
    position_changed = not np.allclose(original_pos, new_pos, atol=1e-6)
    print(f"Position changed: {position_changed}")
    if position_changed:
        print(f"  Original position: {original_pos}")
        print(f"  New position: {new_pos}")
    
    # Calculate reward after state update
    post_update_reward = env.reward_function(action)
    print(f"Reward after state update: {post_update_reward:.8f}")
    
    # Now do the actual step
    print(f"\n=== ACTUAL STEP ===")
    env.reset()  # Reset for clean test
    obs, actual_reward, done, truncated, info = env.step(action)
    
    print(f"Actual step reward: {actual_reward:.8f}")
    
    # The discrepancy might be that the environment is using base_reward instead of our reward
    print(f"\n=== COMPARISON ===")
    print(f"Our reward function: {direct_reward:.8f}")
    print(f"Base environment reward: {base_reward:.8f}")  
    print(f"Actual step reward: {actual_reward:.8f}")
    
    if abs(actual_reward - base_reward) < 0.0001:
        print("❌ PROBLEM: Environment is returning base_reward instead of our reward!")
        print("❌ The custom reward function is being ignored!")
    elif abs(actual_reward - direct_reward) < 0.0001:
        print("✅ Environment is using our reward function correctly")
    else:
        print("⚠️  Unexpected reward source")
    
    env.close()

if __name__ == "__main__":
    debug_step_method()