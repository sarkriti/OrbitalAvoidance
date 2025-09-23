#!/usr/bin/env python3
"""
Simple test of the new cartesian_to_orbit_elements conversion function.
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orekit_physics import OrekitOrbitPropagator

def test_conversion():
    """Test the new Cartesian-to-Keplerian conversion."""
    print("=== Testing Cartesian-to-Keplerian Conversion ===")
    
    # Create physics engine
    physics = OrekitOrbitPropagator()
    
    # Test with a simple circular orbit at 500km altitude
    position_km = np.array([6871.0, 0.0, 0.0])  # 500km altitude
    velocity_km_s = np.array([0.0, 7.613, 0.0])  # Approximate circular velocity
    
    print(f"Input position (km): {position_km}")
    print(f"Input velocity (km/s): {velocity_km_s}")
    
    try:
        elements = physics.cartesian_to_orbit_elements(position_km, velocity_km_s)
        
        print(f"\nComputed orbital elements:")
        print(f"  Semi-major axis: {elements.semi_major_axis_km:.1f} km")
        print(f"  Eccentricity: {elements.eccentricity:.6f}")
        print(f"  Inclination: {elements.inclination_deg:.1f}°")
        print(f"  RAAN: {elements.raan_deg:.1f}°")
        print(f"  Arg. Periapsis: {elements.arg_periapsis_deg:.1f}°")
        print(f"  Mean Anomaly: {elements.mean_anomaly_deg:.1f}°")
        
        # Check if the altitude makes sense
        altitude_km = elements.semi_major_axis_km - 6371.0
        print(f"  Altitude: {altitude_km:.1f} km")
        
        print("✅ Conversion successful!")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_conversion()