#!/usr/bin/env python3
"""
Test the cartesian_to_orbit_elements function directly with debug output.
"""

import numpy as np
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_conversion_debug():
    """Test conversion with detailed debug output."""
    print("=== DEBUGGING CARTESIAN TO ORBITAL CONVERSION ===")
    
    from orekit_physics import OrekitOrbitPropagator
    
    physics = OrekitOrbitPropagator()
    
    # Test case: Current spacecraft position from the debug
    pos_km = np.array([6764.229, 0.0, 0.0])
    vel_km_s = np.array([0.0, 4.77058582, 6.01898134])
    
    print(f"Input position (km): {pos_km}")
    print(f"Input velocity (km/s): {vel_km_s}")
    print(f"Input altitude: {np.linalg.norm(pos_km) - 6371.0:.1f} km")
    print(f"Input speed: {np.linalg.norm(vel_km_s):.3f} km/s")
    
    try:
        elements = physics.cartesian_to_orbit_elements(pos_km, vel_km_s)
        
        print(f"\nConverted elements:")
        print(f"  Semi-major axis: {elements.semi_major_axis_km:.6f} km")
        print(f"  Eccentricity: {elements.eccentricity:.6f}")
        print(f"  Inclination: {elements.inclination_deg:.6f}°")
        print(f"  RAAN: {elements.raan_deg:.6f}°")
        print(f"  Arg Periapsis: {elements.arg_periapsis_deg:.6f}°")
        print(f"  Mean Anomaly: {elements.mean_anomaly_deg:.6f}°")
        print(f"  Computed altitude: {elements.semi_major_axis_km - 6371.0:.1f} km")
        
        # Check if these look like hardcoded values
        hardcoded_indicators = [
            abs(elements.semi_major_axis_km - 6771.0) < 0.1,  # 400km altitude
            abs(elements.eccentricity - 0.001) < 0.0001,      # 0.001 ecc
            abs(elements.inclination_deg - 51.6) < 0.1,       # 51.6° inc
        ]
        
        if any(hardcoded_indicators):
            print("\n❌ WARNING: Results look suspiciously like hardcoded values!")
            print("❌ Expected different values based on actual position/velocity")
        else:
            print("\n✅ Results look reasonable for the given position/velocity")
            
        # Test with a clearly different position
        print(f"\n=== TESTING WITH CLEARLY DIFFERENT POSITION ===")
        pos_km_2 = np.array([7500.0, 0.0, 0.0])  # Much higher altitude
        vel_km_s_2 = np.array([0.0, 7.0, 0.0])   # Different velocity
        
        print(f"Input position 2 (km): {pos_km_2}")
        print(f"Input velocity 2 (km/s): {vel_km_s_2}")
        print(f"Input altitude 2: {np.linalg.norm(pos_km_2) - 6371.0:.1f} km")
        
        elements_2 = physics.cartesian_to_orbit_elements(pos_km_2, vel_km_s_2)
        
        print(f"\nConverted elements 2:")
        print(f"  Semi-major axis: {elements_2.semi_major_axis_km:.6f} km")
        print(f"  Computed altitude: {elements_2.semi_major_axis_km - 6371.0:.1f} km")
        print(f"  Eccentricity: {elements_2.eccentricity:.6f}")
        print(f"  Inclination: {elements_2.inclination_deg:.6f}°")
        
        # Compare the two results
        alt_diff = abs((elements_2.semi_major_axis_km - 6371.0) - (elements.semi_major_axis_km - 6371.0))
        print(f"\nAltitude difference: {alt_diff:.1f} km")
        
        if alt_diff < 10:
            print("❌ MAJOR PROBLEM: Altitudes are nearly identical despite very different inputs!")
            print("❌ The conversion function is definitely not working correctly")
        else:
            print("✅ Altitudes differ appropriately")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_conversion_debug()