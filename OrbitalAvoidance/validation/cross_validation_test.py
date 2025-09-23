#!/usr/bin/env python3
"""
Cross-validation test against known orbital data.
Compares simulation results with published orbital mechanics results.
"""

import numpy as np
import math
from singleton_propagator import get_singleton_propagator
from orekit_physics import convert_to_orbit_elements

def test_against_known_orbits():
    """Test against well-documented orbital mechanics cases."""
    print("🔬 CROSS-VALIDATION AGAINST KNOWN ORBITS")
    print("=" * 60)
    
    physics = get_singleton_propagator()
    
    # Test cases from orbital mechanics literature
    test_cases = [
        {
            "name": "ISS Orbit (Typical)",
            "altitude_km": 408,
            "inclination_deg": 51.64,
            "eccentricity": 0.0001,
            "expected_period_min": 92.68,  # Known ISS orbital period
            "tolerance_min": 0.5
        },
        {
            "name": "GPS Orbit (MEO)",
            "altitude_km": 20200,
            "inclination_deg": 55.0,
            "eccentricity": 0.02,
            "expected_period_min": 718.2,  # 11.97 hours
            "tolerance_min": 5.0
        },
        {
            "name": "Geostationary Orbit",
            "altitude_km": 35786,
            "inclination_deg": 0.0,
            "eccentricity": 0.001,
            "expected_period_min": 1436.1,  # 23.93 hours
            "tolerance_min": 10.0
        },
        {
            "name": "Molniya Orbit (HEO)",
            "altitude_km": 26560,  # Apogee altitude approximation
            "inclination_deg": 63.4,
            "eccentricity": 0.72,
            "expected_period_min": 718.2,  # 12 hours
            "tolerance_min": 20.0
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n🛰️ Testing: {case['name']}")
        
        elements = convert_to_orbit_elements(
            altitude_km=case['altitude_km'],
            inclination_deg=case['inclination_deg'],
            eccentricity=case['eccentricity']
        )
        
        # Calculate orbital period using our physics engine
        calculated_period = physics.get_orbital_period(elements)
        calculated_period_min = calculated_period / 60.0
        
        # Compare with expected period
        expected_period_min = case['expected_period_min']
        period_error = abs(calculated_period_min - expected_period_min)
        period_error_percent = (period_error / expected_period_min) * 100
        
        print(f"  Expected period: {expected_period_min:.2f} min")
        print(f"  Calculated period: {calculated_period_min:.2f} min")
        print(f"  Error: {period_error:.2f} min ({period_error_percent:.3f}%)")
        
        # Tolerance check
        tolerance = case['tolerance_min']
        if period_error <= tolerance:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        print(f"  Status: {status} (tolerance: ±{tolerance:.1f} min)")
        
        results.append({
            'name': case['name'],
            'expected': expected_period_min,
            'calculated': calculated_period_min,
            'error': period_error,
            'error_percent': period_error_percent,
            'passed': period_error <= tolerance
        })
    
    # Test J2 perturbation effects
    print(f"\n🌍 Testing J2 Perturbation Effects")
    print("-" * 40)
    
    # Known J2 effect: regression of nodes for ISS
    elements = convert_to_orbit_elements(altitude_km=408, inclination_deg=51.64)
    
    # Propagate for one day and measure orbital plane drift
    pos_0, vel_0 = physics.propagate_orbit(elements, 0)
    pos_1, vel_1 = physics.propagate_orbit(elements, 24 * 3600)
    
    # Calculate angular momentum vectors (orbit normal)
    h_0 = np.cross(pos_0, vel_0)
    h_1 = np.cross(pos_1, vel_1)
    
    h_0_norm = h_0 / np.linalg.norm(h_0)
    h_1_norm = h_1 / np.linalg.norm(h_1)
    
    # Calculate drift angle
    drift_angle = math.degrees(math.acos(np.clip(np.dot(h_0_norm, h_1_norm), -1, 1)))
    
    # Expected J2 drift for ISS: ~4-5°/day
    expected_drift_range = (3.0, 6.0)
    
    print(f"  Orbital plane drift (1 day): {drift_angle:.3f}°")
    print(f"  Expected range: {expected_drift_range[0]}-{expected_drift_range[1]}°")
    
    if expected_drift_range[0] <= drift_angle <= expected_drift_range[1]:
        print(f"  J2 Effect: ✅ PASS")
        j2_passed = True
    else:
        print(f"  J2 Effect: ❌ FAIL")
        j2_passed = False
    
    # Summary
    print(f"\n📊 CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for r in results if r['passed'])
    total_tests = len(results)
    
    print(f"Orbital period tests: {passed_tests}/{total_tests} passed")
    print(f"J2 perturbation test: {'PASS' if j2_passed else 'FAIL'}")
    
    max_error = max(r['error_percent'] for r in results)
    print(f"Maximum period error: {max_error:.3f}%")
    
    overall_passed = (passed_tests == total_tests) and j2_passed
    
    if overall_passed:
        print("✅ CROSS-VALIDATION: All tests passed - simulation validated against known data")
    else:
        print("⚠️  CROSS-VALIDATION: Some tests failed - review orbital mechanics implementation")
    
    return overall_passed

if __name__ == "__main__":
    test_against_known_orbits()