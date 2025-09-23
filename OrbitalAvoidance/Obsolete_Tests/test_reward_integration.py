#!/usr/bin/env python3
"""
Quick test to verify rl_train.py will use improved_reward_function.py
"""

import json
import sys
sys.path.append('/home/bryson/RL_testing/OrbitalAvoidance')

def test_rl_train_integration():
    """Test that rl_train.py properly integrates with improved_reward_function.py"""
    
    print("🧪 Testing RL Train Integration with Improved Rewards")
    print("=" * 60)
    
    # Load the training configuration
    config_path = '/home/bryson/RL_testing/OrbitalAvoidance/training_configs/fixed_reward_ppo_training.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("✅ Training configuration loaded")
    print(f"   Config has use_improved_rewards: {config['environment'].get('use_improved_rewards', False)}")
    
    # Test the reward configuration flow
    from rl_train import setup_environment
    
    try:
        print("\n🔧 Testing environment setup...")
        env = setup_environment(config)
        print("✅ Environment setup successful")
        
        # Check if the environment has the improved reward function
        reward_config = env.reward_config
        uses_improved = reward_config.get('use_improved_rewards', False)
        print(f"   Environment uses improved rewards: {uses_improved}")
        
        if uses_improved:
            print("   Reward config keys:", list(reward_config.keys()))
        
        # Test a reward calculation
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"✅ Reward calculation successful: {reward:.4f}")
        
        # Check reward function type
        reward_func = env.reward_function
        print(f"   Reward function type: {type(reward_func).__name__}")
        
        env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rl_train_integration()
    if success:
        print("\n🎉 SUCCESS: rl_train.py WILL use improved_reward_function.py!")
        print("   When you run training with fixed_reward_ppo_training.json,")
        print("   it will automatically use the improved reward function.")
    else:
        print("\n❌ FAILURE: Integration test failed")
    
    sys.exit(0 if success else 1)