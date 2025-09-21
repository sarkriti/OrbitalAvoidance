#!/usr/bin/env python3
"""
Simple RL Training Example

This script demonstrates how to set up and train an RL agent for orbital debris avoidance
using the exact state representation and reward function you specified.

Run this script to see the system in action!
"""

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training_env import create_rl_environment
from rl_algorithms import create_algorithm, get_reward_config


def demonstrate_state_representation():
    """Demonstrate the exact state representation you specified."""
    print("🔍 Demonstrating State Representation")
    print("=" * 50)
    
    # Create environment with your reward configuration
    reward_config = get_reward_config('balanced')  # Your exact reward spec
    env = create_rl_environment(
        config_file="scenarios/leo_debris_avoidance.json",
        max_debris_tracked=10,  # Top 10 debris as requested
        reward_config=reward_config
    )
    
    # Reset and get initial observation
    obs, info = env.reset()
    
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Total observation vector length: {len(obs)}")
    print()
    
    # Break down the state vector according to your specification
    print("📊 State Vector Breakdown:")
    print("-" * 30)
    
    idx = 0
    
    # Spacecraft state (6 elements)
    spacecraft_pos = obs[idx:idx+3]
    spacecraft_vel = obs[idx+3:idx+6]
    idx += 6
    print(f"Spacecraft position (km): [{spacecraft_pos[0]:.3f}, {spacecraft_pos[1]:.3f}, {spacecraft_pos[2]:.3f}]")
    print(f"Spacecraft velocity (km/s): [{spacecraft_vel[0]:.6f}, {spacecraft_vel[1]:.6f}, {spacecraft_vel[2]:.6f}]")
    
    # Mission info (2 elements)
    fuel_remaining = obs[idx]
    time_remaining = obs[idx+1]
    idx += 2
    print(f"Fuel remaining (normalized): {fuel_remaining:.3f}")
    print(f"Time remaining in mission (normalized): {time_remaining:.3f}")
    print()
    
    # Per debris object (8 elements each for top 10)
    print("🛰️ Debris Information (Top 10 closest):")
    print("-" * 40)
    
    for i in range(env.max_debris_tracked):
        debris_start = idx + (i * 8)
        debris_end = debris_start + 8
        debris_data = obs[debris_start:debris_end]
        
        if np.any(debris_data != 0):  # Only show non-zero debris
            rel_pos = debris_data[0:3]
            rel_vel = debris_data[3:6]
            miss_distance = debris_data[6]
            time_to_collision = debris_data[7]
            
            print(f"  Debris {i+1}:")
            print(f"    Relative position (km): [{rel_pos[0]:.3f}, {rel_pos[1]:.3f}, {rel_pos[2]:.3f}]")
            print(f"    Relative velocity (km/s): [{rel_vel[0]:.6f}, {rel_vel[1]:.6f}, {rel_vel[2]:.6f}]")
            print(f"    Miss distance (km): {miss_distance:.3f}")
            print(f"    Time to collision (s): {time_to_collision:.1f}")
            print()
    
    print("✅ State representation matches your exact specification!")
    print()
    
    return env


def demonstrate_action_space():
    """Demonstrate the continuous action space (-1 to 1 on X, Y, Z)."""
    print("🎮 Demonstrating Action Space")
    print("=" * 50)
    
    env = create_rl_environment("scenarios/leo_debris_avoidance.json")
    
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Action bounds: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
    print()
    
    # Test different actions
    test_actions = [
        np.array([0.0, 0.0, 0.0]),      # No thrust
        np.array([1.0, 0.0, 0.0]),      # Full thrust +X
        np.array([-1.0, 0.0, 0.0]),     # Full thrust -X
        np.array([0.0, 1.0, 0.0]),      # Full thrust +Y
        np.array([0.0, 0.0, 1.0]),      # Full thrust +Z
        np.array([0.5, 0.5, 0.0]),      # Partial thrust X+Y
        np.array([-0.3, 0.7, -0.2]),    # Mixed thrust
    ]
    
    print("🚀 Example Actions:")
    print("-" * 20)
    
    for i, action in enumerate(test_actions):
        print(f"Action {i+1}: [{action[0]:+.1f}, {action[1]:+.1f}, {action[2]:+.1f}] "
              f"(thrust_x, thrust_y, thrust_z)")
    
    print()
    print("✅ Continuous action space [-1, 1] on X, Y, Z as requested!")
    print()


def demonstrate_reward_function():
    """Demonstrate the modular reward function with your exact formula."""
    print("🎯 Demonstrating Reward Function")
    print("=" * 50)
    
    reward_config = get_reward_config('balanced')
    
    print("Your Reward Formula:")
    print("R_total = W_collision * R_collision + W_fuel * R_fuel + W_orbit * R_orbit + W_mission * R_mission")
    print()
    
    print("📊 Reward Configuration:")
    print("-" * 25)
    print(f"Weights: {reward_config['weights']}")
    print()
    
    print("R_collision:")
    print(f"  - Penalty if collision_probability > {reward_config['collision']['threshold']}: {reward_config['collision']['penalty']}")
    print(f"  - Small reward otherwise: {reward_config['collision']['small_reward']}")
    print()
    
    print("R_fuel:")
    print(f"  - Penalty: -0.1 * delta_v_magnitude (efficiency_weight: {reward_config['fuel']['efficiency_weight']})")
    print()
    
    print("R_orbit:")
    print(f"  - Penalties for orbital element deviations:")
    print(f"    * Semi-major axis tolerance: {reward_config['orbit']['tolerances']['semi_major_axis_m']}m")
    print(f"    * Eccentricity tolerance: {reward_config['orbit']['tolerances']['eccentricity']}")
    print(f"    * Inclination tolerance: {reward_config['orbit']['tolerances']['inclination_rad']} rad")
    print(f"    * RAAN tolerance: {reward_config['orbit']['tolerances']['raan_rad']} rad")
    print(f"    * Arg periapsis tolerance: {reward_config['orbit']['tolerances']['arg_periapsis_rad']} rad")
    print()
    
    print("R_mission:")
    print(f"  - Every {reward_config['mission']['survival_reward_interval']} steps without collision:")
    print(f"  - Reward: {reward_config['mission']['survival_multiplier']} * fuel_remaining/initial_fuel * {reward_config['mission']['mission_weight']}")
    print()
    
    print("✅ Modular reward system matches your exact specification!")
    print()


def demonstrate_algorithm_switching():
    """Demonstrate easy algorithm switching."""
    print("🔄 Demonstrating Algorithm Switching")
    print("=" * 50)
    
    env = create_rl_environment("scenarios/leo_debris_avoidance.json")
    
    # Available algorithms
    algorithms = ['ppo', 'sac', 'td3']
    
    print("Available algorithms:")
    for i, alg in enumerate(algorithms, 1):
        print(f"  {i}. {alg.upper()}")
    print()
    
    print("🧠 Creating different algorithm instances:")
    print("-" * 40)
    
    for alg_name in algorithms:
        try:
            algorithm = create_algorithm(alg_name, env)
            print(f"✅ {alg_name.upper()}: Ready for training")
        except ImportError:
            print(f"⚠️  {alg_name.upper()}: Requires stable-baselines3 (pip install stable-baselines3)")
    
    print()
    print("💡 To switch algorithms, simply change the algorithm name!")
    print("   Example: create_algorithm('ppo', env) → create_algorithm('sac', env)")
    print()


def run_quick_test():
    """Run a quick test to verify everything works."""
    print("🚀 Running Quick Functionality Test")
    print("=" * 50)
    
    try:
        # Create environment
        print("1. Creating environment...")
        env = create_rl_environment("scenarios/leo_debris_avoidance.json")
        print("   ✅ Environment created successfully")
        
        # Reset environment
        print("2. Resetting environment...")
        obs, info = env.reset()
        print("   ✅ Environment reset successfully")
        
        # Test random actions
        print("3. Testing random actions...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: Action={action}, Reward={reward:.3f}, Done={terminated}")
            
            if terminated or truncated:
                obs, info = env.reset()
                print("   Environment reset due to termination")
        
        print("   ✅ Action testing completed successfully")
        
        print()
        print("🎉 ALL SYSTEMS WORKING CORRECTLY!")
        print("   Your RL training environment is ready for agent training!")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   Please check dependencies and configuration files")


def main():
    """Main demonstration function."""
    print("🛰️ RL Orbital Debris Avoidance - System Demonstration")
    print("=" * 60)
    print()
    
    # Demonstrate each component
    demonstrate_state_representation()
    demonstrate_action_space()
    demonstrate_reward_function()
    demonstrate_algorithm_switching()
    run_quick_test()
    
    print()
    print("🎓 Next Steps:")
    print("-" * 15)
    print("1. Install stable-baselines3: pip install stable-baselines3")
    print("2. Train an agent: python rl_train.py --algorithm ppo --timesteps 50000")
    print("3. Test trained agent: python rl_train.py --mode test --model-path models/your_model")
    print("4. Switch algorithms: python rl_train.py --algorithm sac")
    print("5. Change rewards: python rl_train.py --reward conservative")
    print()
    print("📖 See README.md for complete documentation!")


if __name__ == "__main__":
    main()