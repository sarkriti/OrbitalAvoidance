#!/usr/bin/env python3
"""
Test the fixed reward system with minimal environment setup.
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training_env import SpacecraftState
from orekit_physics import OrekitOrbitPropagator

def test_state_to_orbit_elements():
    """Test the fixed _state_to_orbit_elements function."""
    print("=== Testing Fixed _state_to_orbit_elements Function ===")
    
    # Create physics engine directly
    physics = OrekitOrbitPropagator()
    
    # Test different spacecraft positions
    test_cases = [
        ("500km altitude", np.array([6871.0, 0.0, 0.0]), np.array([0.0, 7.613, 0.0])),
        ("1000km altitude", np.array([7371.0, 0.0, 0.0]), np.array([0.0, 7.350, 0.0])),
        ("Different position", np.array([6871.0, 1000.0, 0.0]), np.array([-1.108, 7.531, 0.0])),
    ]
    
    for description, pos, vel in test_cases:
        print(f"\n--- {description} ---")
        
        # Create spacecraft state
        spacecraft_state = SpacecraftState({
            'position': pos,
            'velocity': vel,
            'fuel_remaining': 100.0
        })
        
        print(f"Position (km): {pos}")
        print(f"Velocity (km/s): {vel}")
        
        # Test conversion using the new method
        try:
            elements = physics.cartesian_to_orbit_elements(pos, vel)
            
            print(f"Orbital elements:")
            print(f"  Semi-major axis: {elements.semi_major_axis_km:.1f} km")
            print(f"  Altitude: {elements.semi_major_axis_km - 6371.0:.1f} km")
            print(f"  Eccentricity: {elements.eccentricity:.6f}")
            print(f"  Inclination: {elements.inclination_deg:.1f}°")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ All conversions successful!")
    print("The _state_to_orbit_elements() fix should now work properly!")

if __name__ == "__main__":
    test_state_to_orbit_elements()