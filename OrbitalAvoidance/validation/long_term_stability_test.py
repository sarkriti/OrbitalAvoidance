#!/usr/bin/env python3
"""
Long-term stability test for research-grade orbital mechanics validation.
Tests propagation accuracy over days, weeks, and months.
"""

import numpy as np
import time
import logging
from singleton_propagator import get_singleton_propagator
from orekit_physics import convert_to_orbit_elements

logging.basicConfig(level=logging.INFO)

def test_long_term_propagation():
    """Test orbital propagation stability over extended periods."""
    print("🕐 LONG-TERM STABILITY TEST")
    print("=" * 60)
    
    physics = get_singleton_propagator()
    
    # Test different orbital regimes
    test_orbits = [
        ("LEO", 400, 51.6, 0.001),      # ISS-like
        ("MEO", 20200, 55, 0.01),       # GPS-like
        ("GEO", 35786, 0, 0.001),       # Geostationary
        ("HEO", 26560, 63.4, 0.72),     # Molniya-like
    ]
    
    time_periods = [
        ("1 day", 24 * 3600),
        ("1 week", 7 * 24 * 3600),
        ("1 month", 30 * 24 * 3600),
        ("1 year", 365 * 24 * 3600),
    ]
    
    results = []
    
    for orbit_name, altitude, inclination, eccentricity in test_orbits:
        print(f"\n🛰️ Testing {orbit_name} orbit ({altitude} km, {inclination}°, e={eccentricity})")
        
        elements = convert_to_orbit_elements(
            altitude_km=altitude,
            inclination_deg=inclination,
            eccentricity=eccentricity
        )
        
        # Get initial orbital energy
        pos_0, vel_0 = physics.propagate_orbit(elements, 0)
        mu = 398600.4418  # km³/s²
        energy_0 = 0.5 * np.linalg.norm(vel_0)**2 - mu / np.linalg.norm(pos_0)
        
        for period_name, time_seconds in time_periods:
            try:
                start_time = time.time()
                pos_t, vel_t = physics.propagate_orbit(elements, time_seconds)
                end_time = time.time()
                
                # Calculate energy drift
                energy_t = 0.5 * np.linalg.norm(vel_t)**2 - mu / np.linalg.norm(pos_t)
                energy_error = abs(energy_t - energy_0) / abs(energy_0) * 100
                
                # Calculate position accuracy
                r_0 = np.linalg.norm(pos_0)
                r_t = np.linalg.norm(pos_t)
                radius_change = abs(r_t - r_0)
                
                computation_time = end_time - start_time
                
                print(f"  {period_name:8s}: Energy drift = {energy_error:.6f}%, "
                      f"Radius change = {radius_change:.3f} km, "
                      f"Computation = {computation_time:.3f}s")
                
                results.append({
                    'orbit': orbit_name,
                    'period': period_name,
                    'energy_error': energy_error,
                    'radius_change': radius_change,
                    'computation_time': computation_time
                })
                
                # Research-grade thresholds
                if energy_error > 1.0:  # 1% energy drift threshold
                    print(f"    ⚠️  WARNING: High energy drift for {period_name}")
                
            except Exception as e:
                print(f"  {period_name:8s}: ❌ FAILED - {e}")
    
    # Summary analysis
    print(f"\n📊 SUMMARY")
    print("=" * 60)
    max_energy_error = max(r['energy_error'] for r in results)
    max_computation_time = max(r['computation_time'] for r in results)
    
    print(f"Maximum energy drift: {max_energy_error:.6f}%")
    print(f"Maximum computation time: {max_computation_time:.3f}s")
    
    if max_energy_error < 0.1:
        print("✅ EXCELLENT: Long-term stability verified")
    elif max_energy_error < 1.0:
        print("✅ GOOD: Acceptable long-term stability")
    else:
        print("⚠️  WARNING: Long-term stability needs improvement")

if __name__ == "__main__":
    test_long_term_propagation()