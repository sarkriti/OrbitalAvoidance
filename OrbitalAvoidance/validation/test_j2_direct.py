#!/usr/bin/env python3
"""Direct test of J2 effects to debug the perturbation issue."""

import logging
import math
import numpy as np
from orekit_physics import OrekitOrbitPropagator, convert_to_orbit_elements

logging.basicConfig(level=logging.INFO)

def test_j2_effects():
    """Test J2 effects by comparing propagation with and without gravity field."""
    print("🧪 Direct J2 Effects Test")
    print("=" * 50)
    
    # Create propagator
    propagator = OrekitOrbitPropagator()
    
    # Create an orbit that should show strong J2 effects (maximum at ~63° inclination)
    elements = convert_to_orbit_elements(
        altitude_km=800,
        inclination_deg=60,  # Better angle for J2 effects
        eccentricity=0.001,
        raan_deg=0.0
    )
    
    print(f"Test orbit: 800km altitude, 60° inclination")
    print(f"Semi-major axis: {elements.semi_major_axis_km:.1f} km")
    print(f"Inclination: {elements.inclination_deg:.1f}°")
    print(f"Eccentricity: {elements.eccentricity:.6f}")
    
    # Test multiple time periods
    test_periods = [1, 7, 30]  # 1 day, 1 week, 1 month
    
    for days in test_periods:
        print(f"\n--- {days} day(s) propagation ---")
        propagation_seconds = days * 24 * 3600
        
        try:
            # Get initial position and velocity
            pos_initial, vel_initial = propagator.propagate_orbit(elements, 0)
            
            # Get final position and velocity  
            pos_final, vel_final = propagator.propagate_orbit(elements, propagation_seconds)
            
            # Calculate angular momentum vectors (normal to orbital plane)
            h_initial = np.cross(pos_initial, vel_initial)
            h_final = np.cross(pos_final, vel_final)
            
            # Normalize
            h_initial_norm = h_initial / np.linalg.norm(h_initial)
            h_final_norm = h_final / np.linalg.norm(h_final)
            
            # Calculate angle between planes
            cos_angle = np.dot(h_initial_norm, h_final_norm)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to avoid numerical issues
            plane_change_rad = math.acos(cos_angle)
            plane_change_deg = math.degrees(plane_change_rad)
            
            print(f"  Orbital plane change: {plane_change_deg:.3f}°")
            
            # Calculate expected J2 drift rate
            # For polar orbit: dΩ/dt ≈ -1.5 * J2 * (Re/a)² * n * cos(i)
            # Where J2 ≈ 1.08263e-3, Re = 6378.137 km, n = mean motion
            J2 = 1.08263e-3
            Re = 6378.137  # km
            a = elements.semi_major_axis_km  # Already in km
            
            # Mean motion (rad/s)
            mu = 398600.4418  # km³/s²
            n = math.sqrt(mu / a**3)
            
            # RAAN drift rate (rad/s) for 60° inclination
            inclination_for_calc = math.radians(60)  
            omega_dot = -1.5 * J2 * (Re/a)**2 * n * math.cos(inclination_for_calc)
            
            # Convert to degrees per day
            omega_dot_deg_per_day = math.degrees(omega_dot) * 24 * 3600
            expected_drift_deg = omega_dot_deg_per_day * days
            
            print(f"  Expected drift (theory): {expected_drift_deg:.3f}°")
            print(f"  Drift rate: {plane_change_deg/days:.3f}°/day")
            
            if plane_change_deg > 0.01:
                print(f"  ✅ J2 effects detected!")
            else:
                print(f"  ❌ No significant J2 effects")
                
        except Exception as e:
            print(f"  ❌ Error during {days}-day propagation: {e}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_j2_effects()