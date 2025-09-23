#!/usr/bin/env python3
"""
Check which reward function is actually being used.
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_reward_function():
    """Check which reward function is being used."""
    print("=== CHECKING REWARD FUNCTION USAGE ===")
    
    from rl_training_env import create_rl_environment
    
    env = create_rl_environment("scenarios/leo_debris_avoidance_fixed.json")
    env.reset()
    
    print(f"Environment reward config:")
    print(f"  use_improved_rewards: {env.reward_config.get('use_improved_rewards', 'Not found')}")
    
    # Check the reward function
    reward_func = env.reward_function
    print(f"Reward function: {reward_func}")
    print(f"Function name: {getattr(reward_func, '__name__', 'No name')}")
    
    # Try to call it and see what happens
    action = np.array([0.0, 0.0, 0.0])
    
    print(f"\n=== TESTING REWARD FUNCTION CALL ===")
    try:
        reward = reward_func(action)
        print(f"Reward function result: {reward:.8f}")
        
        # Check if this is the improved reward function
        if 'improved' in str(reward_func).lower():
            print("⚠️  Using ImprovedRewardFunction")
        else:
            print("✅ Using original reward function")
            
    except Exception as e:
        print(f"Error calling reward function: {e}")
        import traceback
        traceback.print_exc()
    
    # Let's also check if we can force the original reward function
    print(f"\n=== TESTING FORCED ORIGINAL REWARD ===")
    
    # Temporarily disable improved rewards
    original_flag = env.reward_config.get('use_improved_rewards', False)
    env.reward_config['use_improved_rewards'] = False
    
    # Recreate reward function
    original_reward_func = env._create_reward_function()
    print(f"Forced original reward function: {original_reward_func}")
    
    try:
        original_reward = original_reward_func(action)
        print(f"Original reward result: {original_reward:.8f}")
    except Exception as e:
        print(f"Error with original reward: {e}")
        import traceback
        traceback.print_exc()
    
    # Restore original flag
    env.reward_config['use_improved_rewards'] = original_flag
    
    env.close()

if __name__ == "__main__":
    check_reward_function()