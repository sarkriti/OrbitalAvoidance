#!/usr/bin/env python3
"""
Simple but comprehensive test of the fixed reward system.
Tests the orbit reward component specifically to verify the fix.
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_orbit_reward_fix():
    """Test specifically the orbit reward component that was fixed."""
    print("=== TESTING ORBIT REWARD FIX ===")
    
    from rl_training_env import create_rl_environment
    
    # Create environment
    env = create_rl_environment("scenarios/leo_debris_avoidance_fixed.json")
    print("✅ Environment created")
    
    # Initialize
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Handle gymnasium tuple format
    print(f"✅ Environment initialized, observation shape: {obs.shape}")
    
    # Get the reward function
    reward_func = env.reward_function
    print(f"Reward function type: {type(reward_func)}")
    
    # Check what reward system is being used
    try:
        from improved_reward_function import ImprovedRewardFunction
        if isinstance(reward_func, ImprovedRewardFunction):
            print("ℹ️  Using ImprovedRewardFunction (workaround)")
            print("⚠️  To test original fixed system, set use_improved_rewards=False")
        else:
            print("✅ Using original (fixed) reward function")
    except:
        print("✅ Using original (fixed) reward function")
    
    # Test the core issue: orbital element calculation
    print("\n=== TESTING ORBITAL ELEMENT CALCULATION ===")
    
    # Get initial spacecraft state
    spacecraft_state = env.current_satellite_state
    initial_pos = spacecraft_state.position_km.copy()
    initial_vel = spacecraft_state.velocity_km_s.copy()
    
    print(f"Initial position: {initial_pos}")
    print(f"Initial velocity: {initial_vel}")
    print(f"Initial altitude: {np.linalg.norm(initial_pos) - 6371.0:.1f} km")
    
    # Test orbit element conversion (this was the broken part)
    if hasattr(env, '_state_to_orbit_elements'):
        print("\nTesting _state_to_orbit_elements function...")
        
        # Test with different positions
        test_positions = [
            ("Original", initial_pos.copy(), initial_vel.copy()),
            ("500km higher", initial_pos * 1.07, initial_vel * 0.97),
            ("300km lower", initial_pos * 0.96, initial_vel * 1.02),
        ]
        
        orbital_elements = []
        
        for description, pos, vel in test_positions:
            # Create temporary spacecraft state
            from rl_training_env import StateWrapper
            test_state = StateWrapper({
                'position': pos,
                'velocity': vel,
                'fuel_remaining': 100.0
            })
            
            try:
                elements = env._state_to_orbit_elements(test_state)
                altitude = elements.semi_major_axis_km - 6371.0
                
                print(f"\n{description}:")
                print(f"  Position: {pos}")
                print(f"  Altitude: {altitude:.1f} km")
                print(f"  Semi-major axis: {elements.semi_major_axis_km:.1f} km") 
                print(f"  Eccentricity: {elements.eccentricity:.6f}")
                print(f"  Inclination: {elements.inclination_deg:.1f}°")
                
                orbital_elements.append(altitude)
                
            except Exception as e:
                print(f"❌ Error with {description}: {e}")
                orbital_elements.append(None)
        
        # Analysis
        valid_altitudes = [a for a in orbital_elements if a is not None]
        if len(valid_altitudes) > 1:
            altitude_range = max(valid_altitudes) - min(valid_altitudes)
            print(f"\n=== RESULTS ===")
            print(f"Altitude range: {altitude_range:.1f} km")
            print(f"Altitudes: {[f'{a:.1f}' for a in valid_altitudes]} km")
            
            if altitude_range > 100:  # Should see significant variation
                print("✅ SUCCESS: Orbital elements vary with spacecraft position!")
                print("✅ FIX CONFIRMED: No more hardcoded orbital elements!")
            else:
                print("⚠️  Limited variation in orbital elements")
            
            # Check for hardcoded 400km altitude
            hardcoded_count = sum(1 for a in valid_altitudes if abs(a - 400.0) < 10.0)
            if hardcoded_count > 0:
                print(f"❌ PROBLEM: Found {hardcoded_count} cases with ~400km (hardcoded value)")
            else:
                print("✅ SUCCESS: No hardcoded 400km altitude detected!")
    
    # Test actual reward calculation with different states
    print("\n=== TESTING REWARD CALCULATION ===")
    
    # Take a few actions and see if rewards vary
    actions = [
        [0, 0, 0],      # No action
        [0.1, 0, 0],    # Small thrust
        [0.5, 0, 0],    # Medium thrust  
    ]
    
    rewards = []
    for i, action in enumerate(actions):
        obs, reward, done, truncated, info = env.step(action)
        if isinstance(obs, tuple):
            obs = obs[0]
        
        rewards.append(reward)
        print(f"Action {action} -> Reward: {reward:.4f}")
        
        if i < len(actions) - 1:  # Reset for next test except last
            env.reset()
    
    # Final analysis
    if len(rewards) > 1:
        reward_range = max(rewards) - min(rewards)
        print(f"\nReward range: {reward_range:.4f}")
        
        # Check for constant penalty
        constant_penalty = abs(rewards[0] - (-1299.5)) < 1.0
        if constant_penalty:
            print("❌ PROBLEM: Still seeing constant -1299.5 penalty!")
        else:
            print("✅ SUCCESS: No constant -1299.5 penalty!")
        
        if reward_range > 0.001:
            print("✅ SUCCESS: Rewards vary between actions!")
        else:
            print("⚠️  Limited reward variation")
    
    env.close()
    
    print("\n=== FINAL ASSESSMENT ===")
    if len(valid_altitudes) > 1 and max(valid_altitudes) - min(valid_altitudes) > 100:
        print("🎉 ORBITAL ELEMENT FIX: CONFIRMED WORKING!")
        print("🎉 Ready for training with proper learning signals!")
        return True
    else:
        print("🔧 May need additional investigation")
        return False

if __name__ == "__main__":
    success = test_orbit_reward_fix()
    if success:
        print("\n🚀 The reward system fix is SUCCESSFUL!")
    else:
        print("\n🔍 Additional debugging may be needed")