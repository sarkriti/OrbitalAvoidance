#!/usr/bin/env python3
"""
Debug the fuel reward calculation specifically.
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_fuel_reward():
    """Debug why fuel rewards are identical."""
    print("=== DEBUGGING FUEL REWARD CALCULATION ===")
    
    from rl_training_env import create_rl_environment
    
    # Create environment
    env = create_rl_environment("scenarios/leo_debris_avoidance_fixed.json")
    env.reset()
    
    # Check reward configuration
    print(f"Reward config: {getattr(env, 'reward_config', 'Not found')}")
    
    # Check satellite properties
    print(f"Main satellite: {getattr(env, 'main_satellite', 'Not found')}")
    if hasattr(env, 'main_satellite'):
        print(f"Max thrust: {env.main_satellite.max_thrust_n} N")
    
    # Check spacecraft state
    spacecraft_state = env.current_satellite_state
    print(f"Spacecraft mass: {getattr(spacecraft_state, 'mass_kg', 'Not found')} kg")
    
    # Test fuel reward calculation directly with different actions
    test_actions = [
        ("No thrust", np.array([0.0, 0.0, 0.0])),
        ("Small thrust", np.array([0.1, 0.0, 0.0])),
        ("Medium thrust", np.array([0.5, 0.0, 0.0])),
        ("Large thrust", np.array([1.0, 0.0, 0.0])),
        ("Max thrust", np.array([1.0, 1.0, 1.0])),
    ]
    
    print(f"\n=== TESTING FUEL REWARD CALCULATION ===")
    
    for description, action in test_actions:
        try:
            fuel_reward = env._calculate_fuel_reward(action)
            thrust_magnitude = np.linalg.norm(action)
            
            print(f"{description}: action={action}")
            print(f"  Thrust magnitude: {thrust_magnitude:.3f}")
            print(f"  Fuel reward: {fuel_reward:.8f}")
            
            # Manual calculation for verification
            if hasattr(env, 'main_satellite') and hasattr(env.main_satellite, 'max_thrust_n'):
                thrust_n = thrust_magnitude * env.main_satellite.max_thrust_n
                if hasattr(spacecraft_state, 'mass_kg'):
                    dt = env.config.physics.time_step_seconds
                    delta_v_ms = (thrust_n / spacecraft_state.mass_kg) * dt
                    delta_v_kms = delta_v_ms / 1000.0
                    
                    print(f"  Manual calc: thrust_n={thrust_n:.1f}, delta_v={delta_v_ms:.6f}m/s, delta_v_km/s={delta_v_kms:.9f}")
                    
                    if hasattr(env, 'reward_config') and 'fuel' in env.reward_config:
                        efficiency_weight = env.reward_config['fuel']['efficiency_weight']
                        manual_reward = -efficiency_weight * delta_v_kms
                        print(f"  Manual reward: -{efficiency_weight} * {delta_v_kms:.9f} = {manual_reward:.8f}")
            
        except Exception as e:
            print(f"Error calculating fuel reward for {description}: {e}")
    
    # Test what happens in actual environment step
    print(f"\n=== TESTING ENVIRONMENT STEP ===")
    
    for description, action in test_actions[:3]:  # Test first 3
        env.reset()
        obs, reward, done, truncated, info = env.step(action)
        print(f"{description}: Total reward = {reward:.8f}")
    
    env.close()

if __name__ == "__main__":
    debug_fuel_reward()