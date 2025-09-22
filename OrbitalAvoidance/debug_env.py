#!/usr/bin/env python3
"""
Debug script to test environment initialization and basic operations.
"""

import sys
import traceback
from rl_training_env import RLTrainingEnvironment

def test_environment():
    """Test basic environment operations."""
    try:
        print("🔧 Creating environment...")
        env = RLTrainingEnvironment(
            scenario_file="../scenarios/leo_debris_avoidance.json"
        )
        print("✓ Environment created successfully")
        
        print("🔄 Testing environment reset...")
        obs = env.reset()
        print(f"✓ Environment reset successfully, observation shape: {obs[0].shape if isinstance(obs, tuple) else obs.shape}")
        
        print("🎯 Testing environment step...")
        action = env.action_space.sample()
        print(f"  Sampled action: {action}")
        
        step_result = env.step(action)
        print(f"✓ Environment step successful, got {len(step_result)} elements")
        
        obs, reward, terminated, truncated, info = step_result
        print(f"  Observation shape: {obs.shape}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Info keys: {list(info.keys()) if isinstance(info, dict) else 'Not a dict'}")
        
        print("🎉 All basic environment operations successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error during environment testing: {e}")
        print("🔍 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting environment debug test...")
    success = test_environment()
    if success:
        print("✅ Environment test completed successfully!")
        sys.exit(0)
    else:
        print("❌ Environment test failed!")
        sys.exit(1)