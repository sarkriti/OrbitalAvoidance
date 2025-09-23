#!/usr/bin/env python3
"""
Debug the reward function to understand why rewards are identical
and why we're still seeing 400km hardcoded values.
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_reward_issues():
    """Debug both the hardcoded value and identical reward issues."""
    print("=== DEBUGGING REWARD FUNCTION ISSUES ===")
    
    from rl_training_env import create_rl_environment
    
    # Create environment
    env = create_rl_environment("scenarios/leo_debris_avoidance_fixed.json")
    print("✅ Environment created")
    
    # Initialize
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    # Check what reward function is actually being used
    print(f"\n=== REWARD FUNCTION ANALYSIS ===")
    print(f"Reward function type: {type(env.reward_function)}")
    print(f"Reward function: {env.reward_function}")
    
    # Check if this is a method or external function
    if hasattr(env.reward_function, '__self__'):
        print(f"Reward function is bound to: {type(env.reward_function.__self__)}")
    
    # Check environment attributes for reward configuration
    if hasattr(env, 'config'):
        if hasattr(env.config, 'environment'):
            print(f"Environment config: {env.config.environment}")
        if hasattr(env.config, 'reward_config'):
            print(f"Reward config: {env.config.reward_config}")
    
    # Test the _state_to_orbit_elements function directly
    print(f"\n=== TESTING _state_to_orbit_elements FUNCTION ===")
    
    spacecraft_state = env.current_satellite_state
    initial_pos = spacecraft_state.position_km.copy()
    initial_vel = spacecraft_state.velocity_km_s.copy()
    
    print(f"Current spacecraft position: {initial_pos}")
    print(f"Current spacecraft velocity: {initial_vel}")
    print(f"Current altitude: {np.linalg.norm(initial_pos) - 6371.0:.1f} km")
    
    # Test the function directly
    try:
        elements = env._state_to_orbit_elements(spacecraft_state)
        computed_altitude = elements.semi_major_axis_km - 6371.0
        print(f"\nComputed orbital elements:")
        print(f"  Semi-major axis: {elements.semi_major_axis_km:.1f} km")
        print(f"  Computed altitude: {computed_altitude:.1f} km")
        print(f"  Eccentricity: {elements.eccentricity:.6f}")
        print(f"  Inclination: {elements.inclination_deg:.1f}°")
        
        if abs(computed_altitude - 400.0) < 10.0:
            print("❌ PROBLEM: Still computing ~400km altitude!")
            print("❌ This suggests the conversion function may not be working correctly")
        else:
            print("✅ SUCCESS: Altitude calculation working correctly")
            
    except Exception as e:
        print(f"❌ Error in _state_to_orbit_elements: {e}")
        import traceback
        traceback.print_exc()
    
    # Test the physics engine directly
    print(f"\n=== TESTING PHYSICS ENGINE DIRECTLY ===")
    try:
        physics = env.physics
        print(f"Physics engine type: {type(physics)}")
        
        # Test the cartesian_to_orbit_elements method directly
        if hasattr(physics, 'cartesian_to_orbit_elements'):
            print("✅ Physics engine has cartesian_to_orbit_elements method")
            
            # Test with the current position
            direct_elements = physics.cartesian_to_orbit_elements(
                initial_pos, initial_vel, 59945.0
            )
            direct_altitude = direct_elements.semi_major_axis_km - 6371.0
            print(f"Direct physics conversion altitude: {direct_altitude:.1f} km")
            
            if abs(direct_altitude - 400.0) < 10.0:
                print("❌ PROBLEM: Physics engine also returning ~400km!")
                print("❌ The issue may be in the conversion implementation")
            else:
                print("✅ SUCCESS: Physics engine conversion working")
                
        else:
            print("❌ PROBLEM: Physics engine missing cartesian_to_orbit_elements method")
            
    except Exception as e:
        print(f"❌ Error testing physics engine: {e}")
        import traceback
        traceback.print_exc()
    
    # Test reward calculation components
    print(f"\n=== TESTING REWARD COMPONENTS ===")
    
    # Check if we can access individual reward components
    if hasattr(env, '_calculate_collision_reward'):
        try:
            collision_reward = env._calculate_collision_reward()
            print(f"Collision reward: {collision_reward:.4f}")
        except Exception as e:
            print(f"Error calculating collision reward: {e}")
    
    if hasattr(env, '_calculate_fuel_reward'):
        try:
            fuel_reward = env._calculate_fuel_reward([0, 0, 0])
            print(f"Fuel reward (no action): {fuel_reward:.4f}")
            
            fuel_reward_thrust = env._calculate_fuel_reward([0.5, 0, 0])
            print(f"Fuel reward (thrust): {fuel_reward_thrust:.4f}")
            
            if abs(fuel_reward - fuel_reward_thrust) < 0.0001:
                print("❌ PROBLEM: Fuel reward identical for different actions!")
            else:
                print("✅ SUCCESS: Fuel reward varies with action")
                
        except Exception as e:
            print(f"Error calculating fuel reward: {e}")
    
    if hasattr(env, '_calculate_orbit_reward'):
        try:
            orbit_reward = env._calculate_orbit_reward()
            print(f"Orbit reward: {orbit_reward:.4f}")
            
            if abs(orbit_reward - (-1299.5)) < 1.0:
                print("❌ PROBLEM: Still seeing constant orbit penalty!")
            else:
                print("✅ SUCCESS: Orbit reward not constant penalty")
                
        except Exception as e:
            print(f"Error calculating orbit reward: {e}")
    
    if hasattr(env, '_calculate_mission_reward'):
        try:
            mission_reward = env._calculate_mission_reward()
            print(f"Mission reward: {mission_reward:.4f}")
        except Exception as e:
            print(f"Error calculating mission reward: {e}")
    
    # Test what happens with different actions
    print(f"\n=== TESTING ACTION RESPONSE ===")
    
    actions_to_test = [
        ("No thrust", [0, 0, 0]),
        ("Small thrust +X", [0.1, 0, 0]),
        ("Large thrust +X", [1.0, 0, 0]),
        ("Thrust -X", [-0.5, 0, 0]),
    ]
    
    for description, action in actions_to_test:
        env.reset()  # Reset for each test
        obs, reward, done, truncated, info = env.step(action)
        print(f"{description} {action} -> Reward: {reward:.6f}")
    
    env.close()
    
    print(f"\n=== CONCLUSIONS ===")
    print("1. Check if the environment is using the correct reward function")
    print("2. Verify the physics engine cartesian_to_orbit_elements implementation")  
    print("3. Ensure the reward formula is properly implemented with all components")

if __name__ == "__main__":
    debug_reward_issues()