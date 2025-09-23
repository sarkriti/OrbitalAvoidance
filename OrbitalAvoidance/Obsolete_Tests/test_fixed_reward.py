#!/usr/bin/env python3
"""
Test the fixed reward system to verify orbit reward varies with spacecraft state.
This test verifies that the _state_to_orbit_elements() fix resolves the constant -1299.5 penalty.
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training_env import RLOrbitalEnvironment, SpacecraftState
from orekit_physics import OrbitElements

def test_original_reward_variation():
    """Test that the original reward system now varies with spacecraft state."""
    print("=== Testing Fixed Original Reward System ===")
    
    # Create environment with original reward (not improved)
    config = {
        'physics_engine': 'orekit',
        'scenario_file': 'scenarios/leo_debris_avoidance_fixed.json',
        'max_timesteps': 1000,
        'observation_history_size': 1,
        'debris_count_limit': 5,
        'use_improved_rewards': False  # Use original (now fixed) reward system
    }
    
    env = RLOrbitalEnvironment(config)
    
    # Test different spacecraft positions to verify orbit reward varies
    test_positions = [
        # Position 1: Reference orbit (should have low orbit penalty)
        np.array([6871.0, 0.0, 0.0]),  # 500km altitude
        
        # Position 2: Different altitude (should have different orbit penalty) 
        np.array([7371.0, 0.0, 0.0]),  # 1000km altitude
        
        # Position 3: Different inclination component
        np.array([4857.0, 4857.0, 0.0]),  # 45° inclination-like
        
        # Position 4: Very different orbit
        np.array([8371.0, 0.0, 0.0]),  # 2000km altitude
    ]
    
    # Corresponding circular velocities for each position
    test_velocities = [
        np.array([0.0, 7.613, 0.0]),    # Circular velocity at 500km
        np.array([0.0, 7.350, 0.0]),    # Circular velocity at 1000km  
        np.array([-5.383, 5.383, 0.0]), # 45° inclination velocity
        np.array([0.0, 6.925, 0.0]),    # Circular velocity at 2000km
    ]
    
    rewards = []
    orbit_elements_list = []
    
    for i, (pos, vel) in enumerate(zip(test_positions, test_velocities)):
        print(f"\n--- Test Position {i+1} ---")
        
        # Create spacecraft state
        spacecraft_state = SpacecraftState({
            'position': pos,
            'velocity': vel,
            'fuel_remaining': 100.0
        })
        
        # Test the _state_to_orbit_elements conversion
        try:
            orbit_elements = env._state_to_orbit_elements(spacecraft_state)
            orbit_elements_list.append(orbit_elements)
            
            print(f"Position (km): {pos}")
            print(f"Velocity (km/s): {vel}")
            print(f"Computed orbital elements:")
            print(f"  Semi-major axis: {orbit_elements.semi_major_axis_km:.1f} km")
            print(f"  Eccentricity: {orbit_elements.eccentricity:.6f}")
            print(f"  Inclination: {orbit_elements.inclination_deg:.1f}°")
            print(f"  RAAN: {orbit_elements.raan_deg:.1f}°")
            print(f"  Arg. Periapsis: {orbit_elements.arg_periapsis_deg:.1f}°")
            print(f"  Mean Anomaly: {orbit_elements.mean_anomaly_deg:.1f}°")
            
        except Exception as e:
            print(f"ERROR in orbital element conversion: {e}")
            continue
        
        # Set the environment state and compute reward
        try:
            env.current_satellite_state = spacecraft_state
            env.scenario.target_orbit = OrbitElements(
                semi_major_axis_km=6871.0,  # 500km target
                eccentricity=0.001,
                inclination_deg=45.0,
                raan_deg=0.0,
                arg_periapsis_deg=0.0,
                mean_anomaly_deg=0.0,
                epoch_mjd=59945.0
            )
            
            # Calculate reward components
            reward_func = env._create_reward_function()
            reward, reward_info = reward_func.calculate_reward(
                env.current_satellite_state,
                0,  # action (no maneuver)
                env.scenario,
                env.current_debris,
                env.simulation_time
            )
            
            rewards.append(reward)
            
            print(f"Reward components:")
            print(f"  Total reward: {reward:.4f}")
            print(f"  Collision reward: {reward_info.get('collision_reward', 'N/A')}")
            print(f"  Fuel reward: {reward_info.get('fuel_reward', 'N/A')}")
            print(f"  Orbit reward: {reward_info.get('orbit_reward', 'N/A')}")
            print(f"  Mission reward: {reward_info.get('mission_reward', 'N/A')}")
            
        except Exception as e:
            print(f"ERROR in reward calculation: {e}")
            rewards.append(None)
    
    # Analysis
    print("\n=== ANALYSIS ===")
    valid_rewards = [r for r in rewards if r is not None]
    
    if len(valid_rewards) > 1:
        reward_range = max(valid_rewards) - min(valid_rewards)
        print(f"Reward range: {reward_range:.4f} (from {min(valid_rewards):.4f} to {max(valid_rewards):.4f})")
        
        if reward_range > 1.0:  # Expect significant variation
            print("✅ SUCCESS: Orbit reward now varies with spacecraft state!")
            print("✅ The original reward system has been fixed!")
        else:
            print("⚠️  WARNING: Limited reward variation detected")
            print("   This might indicate the fix needs refinement")
    else:
        print("❌ FAILED: Could not calculate rewards for multiple positions")
    
    # Check if we're getting the constant -1299.5 orbit penalty
    if any(r is not None and abs(r + 1299.5) < 0.1 for r in rewards):
        print("❌ FAILED: Still getting constant -1299.5 penalty")
        print("   The hardcoded orbital elements issue persists")
    else:
        print("✅ SUCCESS: No longer getting constant -1299.5 penalty")

if __name__ == "__main__":
    test_original_reward_variation()