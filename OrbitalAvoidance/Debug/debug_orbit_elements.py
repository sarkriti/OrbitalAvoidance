#!/usr/bin/env python3
"""
Debug the orbital element calculation to see what's causing the large penalty.
"""

import numpy as np
import sys
import os
import math

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_orbit_calculation():
    """Debug orbital element calculation step by step."""
    print("=== DEBUGGING ORBITAL ELEMENT CALCULATION ===")
    
    from rl_training_env import create_rl_environment
    
    env = create_rl_environment("scenarios/leo_debris_avoidance_fixed.json")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    print(f"=== REFERENCE ORBITAL ELEMENTS ===")
    ref = env.reference_orbit
    print(f"Semi-major axis: {ref.semi_major_axis_km:.6f} km")
    print(f"Eccentricity: {ref.eccentricity:.6f}")
    print(f"Inclination: {ref.inclination_deg:.6f}°")
    print(f"RAAN: {ref.raan_deg:.6f}°")
    print(f"Arg Periapsis: {ref.arg_periapsis_deg:.6f}°")
    print(f"Mean Anomaly: {ref.mean_anomaly_deg:.6f}°")
    
    # Check initial orbit reward (should be 0)
    initial_orbit_reward = env._calculate_orbit_reward()
    print(f"\nInitial orbit reward: {initial_orbit_reward:.6f}")
    
    # Get initial orbital elements for comparison
    initial_elements = env._state_to_orbit_elements(env.current_satellite_state)
    print(f"\n=== INITIAL ORBITAL ELEMENTS (should match reference) ===")
    print(f"Semi-major axis: {initial_elements.semi_major_axis_km:.6f} km")
    print(f"Eccentricity: {initial_elements.eccentricity:.6f}")
    print(f"Inclination: {initial_elements.inclination_deg:.6f}°")
    print(f"RAAN: {initial_elements.raan_deg:.6f}°")
    print(f"Arg Periapsis: {initial_elements.arg_periapsis_deg:.6f}°")
    print(f"Mean Anomaly: {initial_elements.mean_anomaly_deg:.6f}°")
    
    # Check differences
    print(f"\n=== DIFFERENCES (Initial vs Reference) ===")
    print(f"Semi-major axis: {abs(initial_elements.semi_major_axis_km - ref.semi_major_axis_km)*1000:.1f} m")
    print(f"Eccentricity: {abs(initial_elements.eccentricity - ref.eccentricity):.6f}")
    print(f"Inclination: {abs(initial_elements.inclination_deg - ref.inclination_deg):.6f}°")
    print(f"RAAN: {abs(initial_elements.raan_deg - ref.raan_deg):.6f}°")
    print(f"Arg Periapsis: {abs(initial_elements.arg_periapsis_deg - ref.arg_periapsis_deg):.6f}°")
    print(f"Mean Anomaly: {abs(initial_elements.mean_anomaly_deg - ref.mean_anomaly_deg):.6f}°")
    
    # Take one step with no action
    print(f"\n=== TAKING ONE STEP (No Action) ===")
    action = np.array([0.0, 0.0, 0.0])
    obs, reward, done, truncated, info = env.step(action)
    
    # Get orbital elements after step
    post_step_elements = env._state_to_orbit_elements(env.current_satellite_state)
    print(f"\n=== ORBITAL ELEMENTS AFTER STEP ===")
    print(f"Semi-major axis: {post_step_elements.semi_major_axis_km:.6f} km")
    print(f"Eccentricity: {post_step_elements.eccentricity:.6f}")
    print(f"Inclination: {post_step_elements.inclination_deg:.6f}°")
    print(f"RAAN: {post_step_elements.raan_deg:.6f}°")
    print(f"Arg Periapsis: {post_step_elements.arg_periapsis_deg:.6f}°")
    print(f"Mean Anomaly: {post_step_elements.mean_anomaly_deg:.6f}°")
    
    # Check differences after step
    print(f"\n=== DIFFERENCES (After Step vs Reference) ===")
    a_diff_m = abs(post_step_elements.semi_major_axis_km - ref.semi_major_axis_km) * 1000
    e_diff = abs(post_step_elements.eccentricity - ref.eccentricity)
    i_diff_deg = abs(post_step_elements.inclination_deg - ref.inclination_deg)
    raan_diff_deg = abs(post_step_elements.raan_deg - ref.raan_deg)
    argp_diff_deg = abs(post_step_elements.arg_periapsis_deg - ref.arg_periapsis_deg)
    ma_diff_deg = abs(post_step_elements.mean_anomaly_deg - ref.mean_anomaly_deg)
    
    print(f"Semi-major axis: {a_diff_m:.1f} m")
    print(f"Eccentricity: {e_diff:.6f}")
    print(f"Inclination: {i_diff_deg:.6f}°")
    print(f"RAAN: {raan_diff_deg:.6f}°")
    print(f"Arg Periapsis: {argp_diff_deg:.6f}°")
    print(f"Mean Anomaly: {ma_diff_deg:.6f}°")
    
    # Check tolerances
    print(f"\n=== TOLERANCE ANALYSIS ===")
    tolerances = {
        'semi_major_axis_m': 100.0,
        'eccentricity': 0.01,
        'inclination_rad': 0.01,
        'raan_rad': 0.01,
        'arg_periapsis_rad': 0.01
    }
    
    # Convert degree differences to radians for angular elements
    i_diff_rad = math.radians(i_diff_deg)
    raan_diff_rad = math.radians(raan_diff_deg)
    argp_diff_rad = math.radians(argp_diff_deg)
    
    print(f"Semi-major axis: {a_diff_m:.1f} m (tolerance: {tolerances['semi_major_axis_m']} m) -> {'EXCEEDED' if a_diff_m > tolerances['semi_major_axis_m'] else 'OK'}")
    print(f"Eccentricity: {e_diff:.6f} (tolerance: {tolerances['eccentricity']}) -> {'EXCEEDED' if e_diff > tolerances['eccentricity'] else 'OK'}")
    print(f"Inclination: {i_diff_rad:.6f} rad (tolerance: {tolerances['inclination_rad']} rad) -> {'EXCEEDED' if i_diff_rad > tolerances['inclination_rad'] else 'OK'}")
    print(f"RAAN: {raan_diff_rad:.6f} rad (tolerance: {tolerances['raan_rad']} rad) -> {'EXCEEDED' if raan_diff_rad > tolerances['raan_rad'] else 'OK'}")
    print(f"Arg Periapsis: {argp_diff_rad:.6f} rad (tolerance: {tolerances['arg_periapsis_rad']} rad) -> {'EXCEEDED' if argp_diff_rad > tolerances['arg_periapsis_rad'] else 'OK'}")
    
    # Calculate manual orbit reward
    deviations = []
    if a_diff_m > tolerances['semi_major_axis_m']:
        deviations.append((a_diff_m - tolerances['semi_major_axis_m']) / tolerances['semi_major_axis_m'])
    if e_diff > tolerances['eccentricity']:
        deviations.append((e_diff - tolerances['eccentricity']) / tolerances['eccentricity'])
    if i_diff_rad > tolerances['inclination_rad']:
        deviations.append((i_diff_rad - tolerances['inclination_rad']) / tolerances['inclination_rad'])
    if raan_diff_rad > tolerances['raan_rad']:
        deviations.append((raan_diff_rad - tolerances['raan_rad']) / tolerances['raan_rad'])
    if argp_diff_rad > tolerances['arg_periapsis_rad']:
        deviations.append((argp_diff_rad - tolerances['arg_periapsis_rad']) / tolerances['arg_periapsis_rad'])
    
    manual_orbit_reward = -sum(deviations)
    actual_orbit_reward = env._calculate_orbit_reward()
    
    print(f"\n=== REWARD CALCULATION ===")
    print(f"Deviations: {deviations}")
    print(f"Sum of deviations: {sum(deviations):.6f}")
    print(f"Manual orbit reward: {manual_orbit_reward:.6f}")
    print(f"Actual orbit reward: {actual_orbit_reward:.6f}")
    print(f"Total reward: {reward:.6f}")
    
    env.close()

if __name__ == "__main__":
    debug_orbit_calculation()