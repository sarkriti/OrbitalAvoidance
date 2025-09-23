#!/usr/bin/env python3
"""
Test the complete improved training system.
"""

import json
import sys
import os

# Add the current directory to Python path
sys.path.append('/home/bryson/RL_testing/OrbitalAvoidance')

def test_improved_training():
    """Test the improved training configuration."""
    
    print("🧪 Testing Improved Training System")
    print("=" * 50)
    
    # Test 1: Load the configuration
    try:
        config_path = '/home/bryson/RL_testing/OrbitalAvoidance/training_configs/fixed_reward_ppo_training.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
        print(f"   Algorithm: {config['algorithm']}")
        print(f"   Learning rate: {config['hyperparameters']['learning_rate']}")
        print(f"   Use improved rewards: {config['environment']['use_improved_rewards']}")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False
    
    # Test 2: Check scenario file exists
    try:
        scenario_path = f"/home/bryson/RL_testing/OrbitalAvoidance/{config['environment']['scenario_file']}"
        if os.path.exists(scenario_path):
            print("✅ Scenario file exists")
        else:
            print(f"❌ Scenario file missing: {scenario_path}")
            return False
    except Exception as e:
        print(f"❌ Scenario check failed: {e}")
        return False
    
    # Test 3: Test environment creation
    try:
        from rl_training_env import RLOrbitalEnvironment
        
        # Create environment with improved rewards - use direct parameters
        reward_config = config['environment']['reward_config'].copy()
        reward_config['use_improved_rewards'] = config['environment']['use_improved_rewards']
        
        env = RLOrbitalEnvironment(
            config_file=config['environment']['scenario_file'],
            max_debris_tracked=10,
            reward_config=reward_config
        )
        
        print("✅ Environment created successfully")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        
        # Test environment step
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"✅ Environment step successful")
        print(f"   Reward: {reward:.4f}")
        print(f"   Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
        
        env.close()
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test training algorithm setup
    try:
        from rl_algorithms import RLAlgorithmFactory
        
        # Create a new environment for algorithm
        env = RLOrbitalEnvironment(
            config_file=config['environment']['scenario_file'],
            max_debris_tracked=10,
            reward_config=reward_config
        )
        
        algorithm = RLAlgorithmFactory.create_algorithm(
            algorithm_name=config['algorithm'],
            env=env,
            config=config
        )
        
        print("✅ Algorithm created successfully")
        print(f"   Algorithm type: {type(algorithm).__name__}")
        
        env.close()
        
    except Exception as e:
        print(f"❌ Algorithm test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All tests passed! The improved training system is ready.")
    print("\nTo start training, run:")
    print(f"   python3 rl_train.py --config {config_path}")
    
    return True

if __name__ == "__main__":
    success = test_improved_training()
    sys.exit(0 if success else 1)