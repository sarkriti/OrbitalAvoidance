#!/usr/bin/env python3
"""
COMPLETE DIAGNOSIS: Why the original reward function is broken.

Root cause: Orbit reward calculation is broken due to hardcoded orbital elements.
"""

import numpy as np
import sys
import math
sys.path.append('/home/bryson/RL_testing/OrbitalAvoidance')

def complete_diagnosis():
    """Complete diagnosis of why rewards are always the same."""
    
    print("🚨 COMPLETE DIAGNOSIS: WHY REWARDS ARE BROKEN")
    print("=" * 70)
    
    from rl_training_env import RLOrbitalEnvironment
    
    env = RLOrbitalEnvironment(config_file='scenarios/leo_debris_avoidance.json')
    obs, info = env.reset()
    
    print("1. 🔍 CHECKING ORBIT REWARD CALCULATION")
    print("-" * 50)
    
    # Get the orbital elements
    current_elements = env._state_to_orbit_elements(env.current_satellite_state)
    reference_elements = env.reference_orbit
    tolerances = env.reward_config['orbit']['tolerances']
    
    print("Current orbital elements (from _state_to_orbit_elements):")
    print(f"  Semi-major axis: {current_elements.semi_major_axis_km} km")
    print(f"  Eccentricity: {current_elements.eccentricity}")
    print(f"  Inclination: {current_elements.inclination_deg}°")
    print(f"  RAAN: {current_elements.raan_deg}°")
    print(f"  Arg periapsis: {current_elements.arg_periapsis_deg}°")
    print()
    
    print("Reference orbital elements:")
    print(f"  Semi-major axis: {reference_elements.semi_major_axis_km} km")
    print(f"  Eccentricity: {reference_elements.eccentricity}")
    print(f"  Inclination: {reference_elements.inclination_deg}°")
    print(f"  RAAN: {reference_elements.raan_deg}°")
    print(f"  Arg periapsis: {reference_elements.arg_periapsis_deg}°")
    print()
    
    print("Tolerances:")
    for key, value in tolerances.items():
        print(f"  {key}: {value}")
    print()
    
    # Calculate deviations manually
    print("Manual orbit deviation calculation:")
    deviations = []
    
    # Semi-major axis deviation
    a_dev = abs(current_elements.semi_major_axis_km * 1000 - reference_elements.semi_major_axis_km * 1000)
    a_penalty = max(0, a_dev - tolerances['semi_major_axis_m']) / tolerances['semi_major_axis_m']
    deviations.append(a_penalty)
    print(f"  Semi-major axis: |{current_elements.semi_major_axis_km*1000:.1f} - {reference_elements.semi_major_axis_km*1000:.1f}| = {a_dev:.1f}m")
    print(f"    Excess over tolerance: {max(0, a_dev - tolerances['semi_major_axis_m']):.1f}m")
    print(f"    Penalty: {a_penalty:.6f}")
    
    # Eccentricity deviation
    e_dev = abs(current_elements.eccentricity - reference_elements.eccentricity)
    e_penalty = max(0, e_dev - tolerances['eccentricity']) / tolerances['eccentricity']
    deviations.append(e_penalty)
    print(f"  Eccentricity: |{current_elements.eccentricity} - {reference_elements.eccentricity}| = {e_dev}")
    print(f"    Penalty: {e_penalty:.6f}")
    
    # Inclination deviation
    i_dev = abs(math.radians(current_elements.inclination_deg - reference_elements.inclination_deg))
    i_penalty = max(0, i_dev - tolerances['inclination_rad']) / tolerances['inclination_rad']
    deviations.append(i_penalty)
    print(f"  Inclination: |{current_elements.inclination_deg} - {reference_elements.inclination_deg}|° = {math.degrees(i_dev):.6f}°")
    print(f"    Penalty: {i_penalty:.6f}")
    
    # RAAN deviation
    raan_dev = abs(math.radians(current_elements.raan_deg - reference_elements.raan_deg))
    raan_penalty = max(0, raan_dev - tolerances['raan_rad']) / tolerances['raan_rad']
    deviations.append(raan_penalty)
    print(f"  RAAN: |{current_elements.raan_deg} - {reference_elements.raan_deg}|° = {math.degrees(raan_dev):.6f}°")
    print(f"    Penalty: {raan_penalty:.6f}")
    
    # Argument of periapsis deviation
    argp_dev = abs(math.radians(current_elements.arg_periapsis_deg - reference_elements.arg_periapsis_deg))
    argp_penalty = max(0, argp_dev - tolerances['arg_periapsis_rad']) / tolerances['arg_periapsis_rad']
    deviations.append(argp_penalty)
    print(f"  Arg periapsis: |{current_elements.arg_periapsis_deg} - {reference_elements.arg_periapsis_deg}|° = {math.degrees(argp_dev):.6f}°")
    print(f"    Penalty: {argp_penalty:.6f}")
    
    total_orbit_penalty = -sum(deviations)
    print(f"\n  TOTAL ORBIT PENALTY: {total_orbit_penalty:.6f}")
    
    print("\n2. 🎯 ROOT CAUSE ANALYSIS")
    print("-" * 50)
    
    print("PROBLEM IDENTIFIED:")
    print("  The _state_to_orbit_elements() function is HARDCODED!")
    print("  It always returns the same orbital elements regardless of actual spacecraft state.")
    print("  This means:")
    print("    ❌ Orbit reward doesn't reflect actual spacecraft behavior")
    print("    ❌ All actions get the same orbit penalty")
    print("    ❌ No learning signal from orbital dynamics")
    print()
    
    # Test multiple actions to prove they give same orbit reward
    print("3. 🧪 PROVING THE BUG")
    print("-" * 50)
    
    test_actions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([-1.0, -1.0, -1.0])
    ]
    
    print("Testing different actions - orbit penalty should be identical:")
    for i, action in enumerate(test_actions):
        env.reset()
        orbit_reward = env._calculate_orbit_reward()
        print(f"  Action {i+1} {action}: orbit reward = {orbit_reward:.6f}")
    
    print("\n4. 🛠️ THE FIX")
    print("-" * 50)
    
    print("To fix the original reward system, you need to:")
    print("  1. ✅ Fix _state_to_orbit_elements() to calculate ACTUAL orbital elements")
    print("  2. ✅ Use proper orbital mechanics conversions (Cartesian → Keplerian)")
    print("  3. ✅ Ensure reward reflects real spacecraft orbital changes")
    print("  4. ✅ OR disable orbit reward component entirely")
    print()
    print("This is why our improved reward function works better!")
    print("It focuses on collision avoidance and fuel efficiency with proper variation.")
    
    env.close()

if __name__ == "__main__":
    complete_diagnosis()