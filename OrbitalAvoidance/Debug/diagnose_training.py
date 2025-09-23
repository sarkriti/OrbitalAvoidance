#!/usr/bin/env python3
"""
RL Training Diagnostic Tool

This script analyzes your RL training setup to identify why the model isn't learning.
It checks reward function behavior, environment dynamics, observation spaces, and more.
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_reward_function():
    """Analyze the reward function behavior."""
    print("🔍 ANALYZING REWARD FUNCTION")
    print("=" * 50)
    
    from rl_training_env import create_rl_environment
    
    # Create environment
    env = create_rl_environment('scenarios/leo_debris_avoidance.json')
    
    # Reset environment
    obs, info = env.reset()
    
    # Test different actions and their rewards
    print("Testing reward function with different actions:")
    
    actions_to_test = [
        ("No thrust", np.array([0.0, 0.0, 0.0])),
        ("Small thrust X", np.array([0.1, 0.0, 0.0])),
        ("Medium thrust X", np.array([0.5, 0.0, 0.0])),
        ("Large thrust X", np.array([1.0, 0.0, 0.0])),
        ("Small thrust Y", np.array([0.0, 0.1, 0.0])),
        ("Small thrust Z", np.array([0.0, 0.0, 0.1])),
        ("Random thrust", np.random.uniform(-0.5, 0.5, 3)),
    ]
    
    reward_ranges = []
    
    for action_name, action in actions_to_test:
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        reward_ranges.append(reward)
        
        print(f"  {action_name:15s}: reward = {reward:8.4f}")
        
        if terminated:
            print(f"    ⚠️  Terminated! (likely collision)")
            obs, info = env.reset()
    
    # Analyze reward statistics
    print(f"\nReward Statistics:")
    print(f"  Min reward:    {min(reward_ranges):8.4f}")
    print(f"  Max reward:    {max(reward_ranges):8.4f}")
    print(f"  Mean reward:   {np.mean(reward_ranges):8.4f}")
    print(f"  Std reward:    {np.std(reward_ranges):8.4f}")
    print(f"  Range:         {max(reward_ranges) - min(reward_ranges):8.4f}")
    
    # Check if rewards are too sparse or have wrong scale
    if max(reward_ranges) - min(reward_ranges) < 0.1:
        print("❌ PROBLEM: Reward range is too small (<0.1)")
        print("   This makes it hard for the agent to distinguish good/bad actions")
    
    if abs(np.mean(reward_ranges)) > 100:
        print("❌ PROBLEM: Rewards are too large (mean > 100)")
        print("   This can cause numerical instability")
    
    if np.std(reward_ranges) < 0.01:
        print("❌ PROBLEM: Reward variance is too low")
        print("   All actions give similar rewards - no learning signal")
    
    return reward_ranges

def analyze_observation_space():
    """Analyze the observation space."""
    print("\n🔍 ANALYZING OBSERVATION SPACE")
    print("=" * 50)
    
    from rl_training_env import create_rl_environment
    
    env = create_rl_environment('scenarios/leo_debris_avoidance.json')
    obs, info = env.reset()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    
    # Check observation statistics
    print(f"\nObservation Statistics:")
    print(f"  Min value:     {np.min(obs):8.4f}")
    print(f"  Max value:     {np.max(obs):8.4f}")
    print(f"  Mean value:    {np.mean(obs):8.4f}")
    print(f"  Std value:     {np.std(obs):8.4f}")
    print(f"  Range:         {np.max(obs) - np.min(obs):8.4f}")
    
    # Check for common issues
    if np.any(np.isnan(obs)):
        print("❌ PROBLEM: NaN values in observations")
    
    if np.any(np.isinf(obs)):
        print("❌ PROBLEM: Infinite values in observations")
    
    if np.max(obs) > 1000 or np.min(obs) < -1000:
        print("❌ PROBLEM: Observation values are very large")
        print("   Consider normalizing observations")
    
    if np.std(obs) < 0.01:
        print("❌ PROBLEM: Very low observation variance")
        print("   Environment may not be changing enough")
    
    # Test observation changes
    print(f"\nTesting observation dynamics:")
    obs1 = obs.copy()
    
    # Take a few steps
    for i in range(3):
        action = np.random.uniform(-0.1, 0.1, 3)  # Small random action
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    obs2 = obs.copy()
    obs_change = np.mean(np.abs(obs2 - obs1))
    
    print(f"  Observation change over 3 steps: {obs_change:.6f}")
    
    if obs_change < 1e-6:
        print("❌ PROBLEM: Observations barely change")
        print("   Environment dynamics may be broken")

def analyze_action_space():
    """Analyze the action space."""
    print("\n🔍 ANALYZING ACTION SPACE")
    print("=" * 50)
    
    from rl_training_env import create_rl_environment
    
    env = create_rl_environment('scenarios/leo_debris_avoidance.json')
    
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    
    # Test action effects
    obs, info = env.reset()
    initial_pos = obs[:3] if len(obs) >= 3 else obs[:1]  # First few obs components
    
    print(f"\nTesting action effects:")
    print(f"  Initial obs (first 3): {initial_pos}")
    
    # Test different action magnitudes
    for magnitude in [0.01, 0.1, 0.5, 1.0]:
        env.reset()
        action = np.array([magnitude, 0.0, 0.0])
        
        positions = []
        for step in range(10):
            obs, reward, terminated, truncated, info = env.step(action)
            positions.append(obs[:3] if len(obs) >= 3 else obs[:1])
            if terminated or truncated:
                break
        
        if len(positions) > 1:
            pos_change = np.mean(np.abs(positions[-1] - positions[0]))
            print(f"  Action magnitude {magnitude:4.2f}: position change = {pos_change:.6f}")
        
        if terminated:
            print(f"    Terminated after {len(positions)} steps")

def analyze_environment_stability():
    """Check if environment is stable and deterministic."""
    print("\n🔍 ANALYZING ENVIRONMENT STABILITY")
    print("=" * 50)
    
    from rl_training_env import create_rl_environment
    
    env = create_rl_environment('scenarios/leo_debris_avoidance.json')
    
    # Test determinism
    print("Testing determinism with same seed...")
    
    # Run 1
    obs1, info1 = env.reset(seed=42)
    action = np.array([0.1, 0.1, 0.1])
    obs1_step, reward1, _, _, _ = env.step(action)
    
    # Run 2 with same seed
    obs2, info2 = env.reset(seed=42)
    obs2_step, reward2, _, _, _ = env.step(action)
    
    obs_diff = np.mean(np.abs(obs1_step - obs2_step))
    reward_diff = abs(reward1 - reward2)
    
    print(f"  Observation difference: {obs_diff:.10f}")
    print(f"  Reward difference:      {reward_diff:.10f}")
    
    if obs_diff > 1e-10:
        print("❌ PROBLEM: Environment is not deterministic")
        print("   This makes training unstable")
    else:
        print("✅ Environment is deterministic")
    
    # Test episode length
    print(f"\nTesting episode lengths...")
    env.reset(seed=123)
    
    episode_lengths = []
    for episode in range(5):
        env.reset()
        steps = 0
        done = False
        
        while not done and steps < 1000:
            action = np.random.uniform(-0.1, 0.1, 3)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
        
        episode_lengths.append(steps)
        print(f"  Episode {episode + 1}: {steps} steps")
    
    avg_length = np.mean(episode_lengths)
    print(f"  Average episode length: {avg_length:.1f}")
    
    if avg_length < 10:
        print("❌ PROBLEM: Episodes are too short")
        print("   Agent doesn't have time to learn")
    elif avg_length < 50:
        print("⚠️  WARNING: Episodes might be too short for complex learning")

def analyze_hyperparameters():
    """Analyze the hyperparameters from the config."""
    print("\n🔍 ANALYZING HYPERPARAMETERS")
    print("=" * 50)
    
    # Load the training config
    import json
    config_path = "training_configs/example_ppo_training.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("Current PPO hyperparameters:")
        ppo_config = config.get('ppo', {})
        
        learning_rate = ppo_config.get('learning_rate', 0.0003)
        n_steps = ppo_config.get('n_steps', 2048)
        batch_size = ppo_config.get('batch_size', 64)
        n_epochs = ppo_config.get('n_epochs', 10)
        
        print(f"  Learning rate:  {learning_rate}")
        print(f"  N steps:        {n_steps}")
        print(f"  Batch size:     {batch_size}")
        print(f"  N epochs:       {n_epochs}")
        
        # Check for common issues
        if learning_rate > 0.001:
            print("⚠️  WARNING: Learning rate might be too high")
            print("   Try 1e-4 or 3e-5 for stable learning")
        
        if learning_rate < 1e-6:
            print("⚠️  WARNING: Learning rate might be too low")
            print("   Learning will be very slow")
        
        if n_steps < 512:
            print("⚠️  WARNING: n_steps might be too low")
            print("   PPO needs enough experience for each update")
        
        if batch_size > n_steps:
            print("❌ PROBLEM: batch_size > n_steps")
            print("   This is invalid for PPO")
        
        # Training config
        training_config = config.get('training', {})
        total_timesteps = training_config.get('total_timesteps', 100000)
        eval_freq = training_config.get('eval_freq', 25000)
        
        print(f"\nTraining configuration:")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Eval frequency:  {eval_freq:,}")
        
        if total_timesteps < 50000:
            print("⚠️  WARNING: total_timesteps might be too low")
            print("   Complex tasks need more training")
        
    except FileNotFoundError:
        print(f"❌ Could not find config file: {config_path}")

def run_full_diagnosis():
    """Run complete diagnostic analysis."""
    print("🚀 RL TRAINING DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    print("This tool will analyze your training setup to identify")
    print("why your PPO model isn't learning after 100k iterations.")
    print("=" * 60)
    
    try:
        # Run all analyses
        reward_ranges = analyze_reward_function()
        analyze_observation_space()
        analyze_action_space()
        analyze_environment_stability()
        analyze_hyperparameters()
        
        # Summary and recommendations
        print("\n🎯 DIAGNOSIS SUMMARY & RECOMMENDATIONS")
        print("=" * 60)
        
        print("Based on the analysis, here are the likely issues:")
        print()
        
        # Check reward function
        if len(reward_ranges) > 0:
            reward_range = max(reward_ranges) - min(reward_ranges)
            if reward_range < 0.1:
                print("❌ CRITICAL: Reward function has insufficient variation")
                print("   🔧 FIX: Increase reward differences between good/bad actions")
                print("   📝 Example: Make collision penalty -100, success reward +50")
            
            if abs(np.mean(reward_ranges)) > 100:
                print("❌ CRITICAL: Reward scale is too large")
                print("   🔧 FIX: Scale rewards to range [-10, +10]")
        
        print("\n🔧 RECOMMENDED FIXES:")
        print("1. Increase reward variation (collision: -100, success: +50)")
        print("2. Reduce learning rate to 1e-4 or 3e-5")
        print("3. Increase training to 500k-1M timesteps")
        print("4. Add reward shaping for intermediate behaviors")
        print("5. Normalize observations if values are > 1000")
        print("6. Test with shorter episodes (~100-500 steps)")
        
        print("\n📊 NEXT STEPS:")
        print("1. Run: python3 create_better_training_config.py")
        print("2. Train with improved config")
        print("3. Monitor TensorBoard for learning curves")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_full_diagnosis()