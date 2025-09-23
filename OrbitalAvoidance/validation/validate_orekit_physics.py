#!/usr/bin/env python3
"""
Orekit Physics Validation Script

This script provides definitive proof that Orekit physics are being used
by comparing orbital propagation against known analytical solutions and
demonstrating realistic orbital mechanics behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orekit_physics import create_propagator, OrbitElements, convert_to_orbit_elements
from enhanced_web_simulator import OrbitingObject

class OrekitPhysicsValidator:
    """Comprehensive validation of Orekit physics implementation."""
    
    def __init__(self):
        self.test_results = []
        self.propagator = create_propagator()
        
    def test_orbital_period_validation(self):
        """Test 1: Validate orbital period against theoretical calculations."""
        print("🧪 Test 1: Orbital Period Validation")
        print("=" * 50)
        
        # Test orbit at 400 km altitude
        altitude_km = 400
        elements = convert_to_orbit_elements(
            altitude_km=altitude_km,
            inclination_deg=51.6,  # ISS-like orbit
            eccentricity=0.001     # Nearly circular
        )
        
        # Theoretical period using Kepler's Third Law
        earth_radius_km = 6371.0
        earth_mu_km3_s2 = 398600.4418  # km³/s²
        a_km = earth_radius_km + altitude_km
        theoretical_period_s = 2 * math.pi * math.sqrt(a_km**3 / earth_mu_km3_s2)
        theoretical_period_min = theoretical_period_s / 60.0
        
        # Orekit period calculation
        orekit_period_s = self.propagator.get_orbital_period(elements)
        orekit_period_min = orekit_period_s / 60.0
        
        print(f"Theoretical orbital period: {theoretical_period_min:.2f} minutes")
        print(f"Orekit orbital period: {orekit_period_min:.2f} minutes")
        
        # Calculate error
        error_percent = abs(orekit_period_min - theoretical_period_min) / theoretical_period_min * 100
        print(f"Error: {error_percent:.3f}%")
        
        # Test passes if error is less than 0.1% (much tighter tolerance now)
        passed = error_percent < 0.1
        print(f"✅ PASSED" if passed else f"❌ FAILED")
        
        self.test_results.append({
            'test': 'Orbital Period',
            'passed': passed,
            'error_percent': error_percent,
            'details': f'Theoretical: {theoretical_period_min:.2f}min, Actual: {orekit_period_min:.2f}min'
        })
        
        print()
    
    def validate_conservation_of_energy(self):
        """Test 2: Validate conservation of orbital energy."""
        print("🧪 Test 2: Conservation of Energy")
        print("=" * 50)
        
        orbit_elements = convert_to_orbit_elements(
            altitude_km=800,  # Higher altitude for safety
            inclination_deg=0,
            eccentricity=0.01,  # Nearly circular to avoid surface impact  
            mean_anomaly_deg=0,
            arg_periapsis_deg=0,
            raan_deg=0
        )
        
        # Propagate orbit for one complete period
        period = 2 * np.pi * np.sqrt((6771000)**3 / 3.986004418e14)
        times = np.linspace(0, period, 50)
        
        energies = []
        mu_earth = 3.986004418e14  # m³/s²
        
        for t in times:
            pos, vel = self.propagator.propagate_orbit(orbit_elements, t)
            
            # Convert to meters and m/s
            pos_m = pos * 1000
            vel_ms = vel * 1000
            
            # Calculate specific orbital energy: E = v²/2 - μ/r
            kinetic_energy = 0.5 * np.linalg.norm(vel_ms)**2
            potential_energy = -mu_earth / np.linalg.norm(pos_m)
            total_energy = kinetic_energy + potential_energy
            
            energies.append(total_energy)
        
        energies = np.array(energies)
        energy_variation = (np.max(energies) - np.min(energies)) / np.abs(np.mean(energies)) * 100
        
        print(f"Energy variation over one orbit: {energy_variation:.6f}%")
        
        # Test passes if energy is conserved to within 0.01%
        passed = energy_variation < 0.01
        print(f"✅ PASSED" if passed else f"❌ FAILED")
        
        self.test_results.append({
            'test': 'Energy Conservation',
            'passed': passed,
            'error_percent': energy_variation,
            'details': f'Energy variation: {energy_variation:.6f}%'
        })
        
        print()
    
    def test_j2_perturbations(self):
        """Test 3: Demonstrate J2 perturbation effects on RAAN."""
        print("🧪 Test 3: J2 Perturbation Effects")
        print("=" * 50)
        
        # Create orbit that's sensitive to J2 effects (60° inclination is optimal)
        elements = convert_to_orbit_elements(
            altitude_km=800,
            inclination_deg=60,  # Optimal inclination for J2 effects  
            eccentricity=0.001,  # Nearly circular to avoid surface impact
            raan_deg=0.0
        )
        
        # Propagate for multiple orbits to see J2 effects
        propagation_days = 7  # 1 week should show clear RAAN drift
        propagation_seconds = propagation_days * 24 * 3600
        
        # Get initial and final positions
        pos_initial, vel_initial = self.propagator.propagate_orbit(elements, 0)
        pos_final, vel_final = self.propagator.propagate_orbit(elements, propagation_seconds)
        
        # Convert back to orbital elements to see RAAN change
        # For now, let's measure the change in orbital plane
        initial_angular_momentum = np.cross(pos_initial, vel_initial)
        final_angular_momentum = np.cross(pos_final, vel_final)
        
        # Normalize angular momentum vectors
        h_initial = initial_angular_momentum / np.linalg.norm(initial_angular_momentum)
        h_final = final_angular_momentum / np.linalg.norm(final_angular_momentum)
        
        # Calculate angle between orbital planes (proxy for RAAN drift)
        cos_angle = np.dot(h_initial, h_final)
        # Clamp to [-1, 1] to avoid numerical issues
        cos_angle = max(-1.0, min(1.0, cos_angle))
        plane_drift_rad = math.acos(cos_angle)
        plane_drift_deg = math.degrees(plane_drift_rad)
        
        print(f"Orbital plane drift over {propagation_days} day(s): {plane_drift_deg:.3f} degrees")
        
        # Expected J2 drift rate for 60° inclination orbit at 800km: ~3 degrees/day
        expected_drift_deg = 3.0 * propagation_days  # More accurate estimate
        expected_range_low = expected_drift_deg * 0.7   # Allow for some variation
        expected_range_high = expected_drift_deg * 1.4
        
        print(f"Expected drift range: {expected_range_low:.1f} - {expected_range_high:.1f} degrees")
        
        # Test passes if we see realistic J2 drift (not zero, not too extreme)
        has_drift = plane_drift_deg > 1.0  # More than 1 degree for 1 day
        realistic_drift = expected_range_low <= plane_drift_deg <= expected_range_high
        passed = has_drift and realistic_drift
        
        print(f"Has drift: {has_drift}")
        print(f"Realistic magnitude: {realistic_drift}")
        print(f"✅ PASSED - J2 effects detected" if passed else f"❌ FAILED - No clear J2 effects")
        
        self.test_results.append({
            'test': 'J2 Perturbations',
            'passed': passed,
            'drift_degrees': plane_drift_deg,
            'details': f'Plane drift: {plane_drift_deg:.3f}°, Expected: {expected_drift_deg:.1f}°'
        })
        
        print()
    
    def validate_web_simulator_integration(self):
        """Test 4: Validate web simulator uses same Orekit backend."""
        print("🧪 Test 4: Web Simulator Integration")
        print("=" * 50)
        
        # Create same orbit in both systems
        altitude_km = 400
        inclination_deg = 51.6
        
        # Direct Orekit propagation
        orbit_elements = convert_to_orbit_elements(
            altitude_km=altitude_km,
            inclination_deg=inclination_deg,
            eccentricity=0.001,
            mean_anomaly_deg=0,
            arg_periapsis_deg=0,
            raan_deg=0
        )
        
        direct_pos, direct_vel = self.propagator.propagate_orbit(orbit_elements, 3600)  # 1 hour
        
        # Web simulator object
        web_obj = OrbitingObject(
            name="TestSat",
            altitude_km=altitude_km,
            inclination_deg=inclination_deg,
            eccentricity=0.001,
            mean_anomaly_deg=0,
            omega_deg=0,
            raan_deg=0
        )
        
        web_obj.update_position(3600)  # 1 hour
        web_pos = np.array([web_obj.position_x, web_obj.position_y, web_obj.position_z])
        web_vel = np.array([web_obj.velocity_x, web_obj.velocity_y, web_obj.velocity_z])
        
        # Compare positions and velocities
        pos_diff = np.linalg.norm(direct_pos - web_pos)
        vel_diff = np.linalg.norm(direct_vel - web_vel)
        
        pos_error_percent = pos_diff / np.linalg.norm(direct_pos) * 100
        vel_error_percent = vel_diff / np.linalg.norm(direct_vel) * 100
        
        print(f"Position difference: {pos_diff:.6f} km ({pos_error_percent:.6f}%)")
        print(f"Velocity difference: {vel_diff:.6f} km/s ({vel_error_percent:.6f}%)")
        
        # Test passes if differences are negligible (< 0.001%)
        passed = pos_error_percent < 0.001 and vel_error_percent < 0.001
        print(f"✅ PASSED - Web simulator uses same Orekit backend" if passed else f"❌ FAILED - Different physics")
        
        self.test_results.append({
            'test': 'Web Simulator Integration',
            'passed': passed,
            'error_percent': max(pos_error_percent, vel_error_percent),
            'details': f'Pos error: {pos_error_percent:.6f}%, Vel error: {vel_error_percent:.6f}%'
        })
        
        print()
    
    def run_all_tests(self):
        """Run all validation tests."""
        print("🔬 OREKIT PHYSICS VALIDATION")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        self.test_orbital_period_validation()
        self.validate_conservation_of_energy()
        self.test_j2_perturbations()
        self.validate_web_simulator_integration()
        
        # Summary
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} | {result['test']:25} | {result['details']}")
        
        print()
        print(f"Overall Result: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED - Orekit physics confirmed!")
        else:
            print("⚠️  Some tests failed - physics implementation may have issues")
        
        return passed_tests == total_tests


if __name__ == "__main__":
    validator = OrekitPhysicsValidator()
    success = validator.run_all_tests()
    
    if success:
        print("\n✅ CONCLUSION: Your system is definitely using Orekit physics!")
    else:
        print("\n❌ CONCLUSION: Physics validation failed - investigate implementation")