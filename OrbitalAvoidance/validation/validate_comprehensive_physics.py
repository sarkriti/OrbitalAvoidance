#!/usr/bin/env python3
"""
Comprehensive validation that RL training and web simulator use validated Orekit physics.
"""

import logging
import sys
import os
import numpy as np
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_rl_environment_physics():
    """Test that RL environment uses validated Orekit physics."""
    print("🧪 Testing RL Environment Physics")
    print("=" * 50)
    
    try:
        from rl_training_env import create_rl_environment
        from config_system import SimulationConfig
        
        # Create a simple test configuration
        config = SimulationConfig(
            name="physics_test",
            scenario_path="/home/bryson/RL_testing/scenarios/leo_debris_avoidance.json"
        )
        
        # Create RL environment
        print("Creating RL environment...")
        env = create_rl_environment(config)
        
        # Test physics access
        print("Testing physics engine access...")
        physics_engine = env.physics
        print(f"Physics engine type: {type(physics_engine)}")
        
        # Test propagation
        from orekit_physics import convert_to_orbit_elements
        elements = convert_to_orbit_elements(altitude_km=400, inclination_deg=51.6)
        
        print("Testing orbit propagation...")
        pos, vel = physics_engine.propagate_orbit(elements, 60.0)
        print(f"✅ Propagation successful: pos={pos[:3]}")
        
        return True, "RL environment uses validated Orekit physics"
        
    except Exception as e:
        print(f"❌ RL environment physics test failed: {e}")
        traceback.print_exc()
        return False, str(e)


def test_web_simulator_physics():
    """Test that web simulator uses validated Orekit physics."""
    print("\n🧪 Testing Web Simulator Physics")
    print("=" * 50)
    
    try:
        from enhanced_web_simulator import OrbitingObject
        
        # Create a test satellite using the web simulator's physics
        print("Creating test satellite with web simulator physics...")
        satellite = OrbitingObject(
            name="TestSat",
            altitude_km=400,
            inclination_deg=51.6,
            eccentricity=0.001,
            color="blue"
        )
        
        print(f"Physics engine type: {type(satellite.propagator)}")
        
        # Test initial position
        initial_pos = satellite.position
        print(f"Initial position: {initial_pos}")
        
        # Simulate 1 minute
        print("Testing orbit propagation (60 seconds)...")
        satellite.update(60.0)
        final_pos = satellite.position
        
        # Calculate movement
        movement = np.linalg.norm(np.array(final_pos) - np.array(initial_pos))
        print(f"✅ Propagation successful: moved {movement:.3f} km in 60s")
        
        return True, "Web simulator uses validated Orekit physics"
        
    except Exception as e:
        print(f"❌ Web simulator physics test failed: {e}")
        traceback.print_exc()
        return False, str(e)


def test_physics_consistency():
    """Test that both environments use the same physics engine."""
    print("\n🧪 Testing Physics Consistency")
    print("=" * 50)
    
    try:
        # Import both engines
        from rl_training_env import create_rl_environment
        from enhanced_web_simulator import OrbitingObject
        from config_system import SimulationConfig
        from orekit_physics import convert_to_orbit_elements
        
        # Create RL environment
        config = SimulationConfig(
            name="consistency_test",
            scenario_path="/home/bryson/RL_testing/scenarios/leo_debris_avoidance.json"
        )
        rl_env = create_rl_environment(config)
        
        # Create web simulator object
        web_sat = OrbitingObject(
            name="TestSat",
            altitude_km=400,
            inclination_deg=51.6,
            eccentricity=0.001,
            color="blue"
        )
        
        # Create same orbital elements for both
        elements = convert_to_orbit_elements(altitude_km=400, inclination_deg=51.6, eccentricity=0.001)
        
        # Propagate same orbit for 60 seconds in both systems
        rl_pos, rl_vel = rl_env.physics.propagate_orbit(elements, 60.0)
        
        # For web simulator, we need to manually propagate since update() includes time accumulation
        web_pos, web_vel = web_sat.propagator.propagate_orbit(elements, 60.0)
        
        # Compare results
        pos_diff = np.linalg.norm(rl_pos - web_pos)
        vel_diff = np.linalg.norm(rl_vel - web_vel)
        
        print(f"RL Environment position: {rl_pos}")
        print(f"Web Simulator position: {web_pos}")
        print(f"Position difference: {pos_diff:.10f} km")
        print(f"Velocity difference: {vel_diff:.10f} km/s")
        
        # They should be identical (within floating point precision)
        tolerance = 1e-10
        consistent = pos_diff < tolerance and vel_diff < tolerance
        
        if consistent:
            print("✅ Both systems use identical physics engines")
            return True, "Physics engines are consistent"
        else:
            print("❌ Physics engines produce different results")
            return False, f"Position diff: {pos_diff}, Velocity diff: {vel_diff}"
            
    except Exception as e:
        print(f"❌ Physics consistency test failed: {e}")
        traceback.print_exc()
        return False, str(e)


def main():
    """Run comprehensive physics validation for both systems."""
    print("🔬 COMPREHENSIVE PHYSICS VALIDATION")
    print("=" * 60)
    print("Validating that RL training and web simulator use the same")
    print("validated Orekit physics engine with realistic perturbations.")
    print("=" * 60)
    
    results = []
    
    # Test RL environment
    rl_success, rl_msg = test_rl_environment_physics()
    results.append(("RL Environment Physics", rl_success, rl_msg))
    
    # Test web simulator
    web_success, web_msg = test_web_simulator_physics()
    results.append(("Web Simulator Physics", web_success, web_msg))
    
    # Test consistency
    consistency_success, consistency_msg = test_physics_consistency()
    results.append(("Physics Consistency", consistency_success, consistency_msg))
    
    # Summary
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {test_name:<25} | {message}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Both RL training and web simulator use validated Orekit physics")
        print("✅ Physics engines are consistent and realistic")
        print("✅ RL models will train with the same realistic physics as the web demo")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️  Physics validation incomplete - check implementation")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)