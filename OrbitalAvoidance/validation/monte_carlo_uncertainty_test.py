#!/usr/bin/env python3
"""
Monte Carlo uncertainty analysis for research-grade validation.
Tests simulation robustness against parameter uncertainties.
"""

import numpy as np
import matplotlib.pyplot as plt
from singleton_propagator import get_singleton_propagator
from orekit_physics import convert_to_orbit_elements, SpacecraftProperties
import logging

logging.basicConfig(level=logging.WARNING)  # Reduce log spam

def monte_carlo_uncertainty_analysis(n_samples=1000):
    """Perform Monte Carlo analysis of orbital propagation uncertainty."""
    print("🎲 MONTE CARLO UNCERTAINTY ANALYSIS")
    print("=" * 60)
    
    physics = get_singleton_propagator()
    
    # Nominal orbit parameters
    nominal_altitude = 400  # km
    nominal_inclination = 51.6  # degrees
    
    # Uncertainty ranges (realistic for spacecraft operations)
    altitude_std = 2.0      # ±2 km altitude uncertainty
    inclination_std = 0.1   # ±0.1° inclination uncertainty
    mass_std = 50.0         # ±50 kg mass uncertainty
    
    print(f"Running {n_samples} Monte Carlo samples...")
    print(f"Uncertainties: ±{altitude_std} km altitude, ±{inclination_std}° inclination, ±{mass_std} kg mass")
    
    results = []
    propagation_time = 24 * 3600  # 1 day
    
    for i in range(n_samples):
        if i % 100 == 0:
            print(f"  Sample {i}/{n_samples}")
        
        # Sample uncertain parameters
        altitude = np.random.normal(nominal_altitude, altitude_std)
        inclination = np.random.normal(nominal_inclination, inclination_std)
        mass = np.random.normal(1000, mass_std)
        
        # Ensure physical bounds
        altitude = max(200, min(2000, altitude))  # 200-2000 km
        inclination = max(0, min(180, inclination))  # 0-180°
        mass = max(100, mass)  # At least 100 kg
        
        try:
            elements = convert_to_orbit_elements(
                altitude_km=altitude,
                inclination_deg=inclination,
                eccentricity=0.001
            )
            
            spacecraft_props = SpacecraftProperties(mass_kg=mass)
            
            # Propagate initial and final states
            pos_0, vel_0 = physics.propagate_orbit(elements, 0)
            pos_t, vel_t = physics.propagate_orbit(elements, propagation_time)
            
            # Calculate metrics
            r_0 = np.linalg.norm(pos_0)
            r_t = np.linalg.norm(pos_t)
            v_0 = np.linalg.norm(vel_0)
            v_t = np.linalg.norm(vel_t)
            
            # Store results
            results.append({
                'altitude_in': altitude,
                'inclination_in': inclination,
                'mass_in': mass,
                'radius_0': r_0,
                'radius_t': r_t,
                'velocity_0': v_0,
                'velocity_t': v_t,
                'radius_change': r_t - r_0,
                'velocity_change': v_t - v_0
            })
            
        except Exception as e:
            print(f"    Sample {i} failed: {e}")
    
    # Statistical analysis
    results = np.array([(r['radius_change'], r['velocity_change']) for r in results])
    
    radius_changes = results[:, 0]
    velocity_changes = results[:, 1]
    
    print(f"\n📊 UNCERTAINTY ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Successful samples: {len(results)}/{n_samples}")
    print(f"\nRadius change after 1 day:")
    print(f"  Mean: {np.mean(radius_changes):.3f} ± {np.std(radius_changes):.3f} km")
    print(f"  Range: [{np.min(radius_changes):.3f}, {np.max(radius_changes):.3f}] km")
    
    print(f"\nVelocity change after 1 day:")
    print(f"  Mean: {np.mean(velocity_changes)*1000:.3f} ± {np.std(velocity_changes)*1000:.3f} m/s")
    print(f"  Range: [{np.min(velocity_changes)*1000:.3f}, {np.max(velocity_changes)*1000:.3f}] m/s")
    
    # Research-grade assessment
    radius_std = np.std(radius_changes)
    velocity_std = np.std(velocity_changes) * 1000  # Convert to m/s
    
    print(f"\n🎯 UNCERTAINTY ASSESSMENT:")
    if radius_std < 1.0 and velocity_std < 10.0:
        print("✅ EXCELLENT: Low sensitivity to parameter uncertainties")
    elif radius_std < 5.0 and velocity_std < 50.0:
        print("✅ GOOD: Acceptable uncertainty propagation")
    else:
        print("⚠️  WARNING: High sensitivity to uncertainties")
    
    return results

if __name__ == "__main__":
    monte_carlo_uncertainty_analysis()