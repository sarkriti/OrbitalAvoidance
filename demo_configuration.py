#!/usr/bin/env python3
"""
Orbital Avoidance Simulator - Configuration Demo

This script demonstrates how easy it is to modify the orbital environment
using JSON configuration files. You can quickly change satellites, debris,
physics models, and reward functions without touching any code.
"""

import json
import numpy as np
from OrbitalAvoidance.configurable_env import ConfigurableOrbitalEnvironment, list_scenarios


def demo_scenario_switching():
    """Demonstrate switching between different scenarios."""
    print("🚀 Orbital Avoidance Simulator - Configuration Demo")
    print("=" * 60)
    
    # List available scenarios
    scenarios = list_scenarios()
    print(f"Available scenarios: {scenarios}")
    print()
    
    # Test each scenario
    for scenario_name in scenarios[:3]:  # Test first 3 scenarios
        try:
            print(f"📡 Testing scenario: {scenario_name}")
            
            # Load environment
            env = ConfigurableOrbitalEnvironment(config_file=f'scenarios/{scenario_name}.json')
            
            config = env.config
            print(f"   Description: {config.description}")
            print(f"   Satellites: {len(config.satellites)}")
            print(f"   Debris fields: {len(config.debris_fields)}")
            print(f"   Static debris: {len(config.static_debris)}")
            print(f"   Physics timestep: {config.physics.time_step_seconds}s")
            print(f"   Collision threshold: {config.physics.collision_threshold_km}km")
            print(f"   Action space: {env.action_space}")
            print(f"   Observation space: {env.observation_space.shape}")
            
            # Run a few steps
            obs, info = env.reset()
            total_reward = 0
            
            for step in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            print(f"   Sample run reward: {total_reward:.3f}")
            print(f"   Final fuel levels: {info['fuel_remaining']}")
            
            env.close()
            print("   ✅ Scenario test successful!")
            print()
            
        except Exception as e:
            print(f"   ❌ Scenario test failed: {e}")
            print()


def demo_custom_scenario():
    """Demonstrate creating a custom scenario on-the-fly."""
    print("🛠️  Creating Custom Scenario")
    print("-" * 30)
    
    # Create a custom scenario with different parameters
    custom_config = {
        "scenario_name": "Custom_Test_Scenario",
        "description": "Custom scenario created programmatically",
        "satellites": [
            {
                "name": "TestSat_1",
                "orbit": {
                    "altitude_km": 450,
                    "inclination_deg": 28.5,  # Different inclination
                    "eccentricity": 0.005,
                    "raan_deg": 45,
                    "arg_periapsis_deg": 90,
                    "mean_anomaly_deg": 0
                },
                "mass_kg": 500,  # Smaller satellite
                "is_controllable": True,
                "max_thrust_n": 0.5,  # Lower thrust
                "fuel_capacity_kg": 50,
                "collision_radius_m": 3.0
            },
            {
                "name": "TestSat_2",
                "orbit": {
                    "altitude_km": 450,
                    "inclination_deg": 28.5,
                    "eccentricity": 0.005,
                    "raan_deg": 225,  # Opposite side of orbit
                    "arg_periapsis_deg": 90,
                    "mean_anomaly_deg": 180
                },
                "mass_kg": 500,
                "is_controllable": True,
                "max_thrust_n": 0.5,
                "fuel_capacity_kg": 50,
                "collision_radius_m": 3.0
            }
        ],
        "debris_fields": [
            {
                "num_objects": 10,  # Fewer debris for testing
                "altitude_range_km": [440, 460],
                "inclination_range_deg": [25, 32],
                "eccentricity_range": [0, 0.01],
                "mass_range_kg": [1, 20],
                "size_range_m": [0.1, 1.0],
                "distribution_type": "uniform"
            }
        ],
        "static_debris": [],
        "physics": {
            "propagator_type": "orekit_keplerian",
            "time_step_seconds": 30,  # Smaller timestep
            "simulation_duration_seconds": 1800,  # 30 minutes
            "collision_threshold_km": 0.0005  # Tighter collision detection
        },
        "rewards": {
            "reward_type": "multi_objective",
            "collision_penalty": -500,  # Lower penalty for testing
            "fuel_cost_weight": 0.2,
            "safety_distance_km": 0.5,
            "close_approach_penalty_weight": 5.0,
            "mission_success_reward": 200.0
        },
        "observations": {
            "include_relative_positions": True,
            "include_relative_velocities": True,
            "include_fuel_levels": True,
            "include_time_to_collision": False,
            "max_tracked_objects": 5,
            "observation_radius_km": 25.0
        },
        "random_seed": 789
    }
    
    # Save custom scenario
    with open('scenarios/custom_demo.json', 'w') as f:
        json.dump(custom_config, f, indent=2)
    
    print("📄 Custom scenario configuration saved to scenarios/custom_demo.json")
    
    # Test the custom scenario
    try:
        env = ConfigurableOrbitalEnvironment(config_file='scenarios/custom_demo.json')
        
        print(f"✅ Custom scenario loaded: {env.config.scenario_name}")
        print(f"   Multi-satellite setup: {len(env.config.satellites)} satellites")
        print(f"   Action space (dict): {env.action_space}")
        print(f"   Observation space: {env.observation_space.shape}")
        
        # Test reset and a few steps
        obs, info = env.reset()
        print(f"   Reset successful, {len(env.satellites)} satellites, {len(env.debris_objects)} debris")
        
        # Sample multi-satellite actions
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            fuel_status = [f"{name}:{fuel:.1f}kg" for name, fuel in info['fuel_remaining'].items()]
            print(f"   Step {step}: reward={reward:.3f}, fuel=[{', '.join(fuel_status)}]")
            
            if terminated or truncated:
                break
        
        env.close()
        print("   ✅ Custom scenario test successful!")
        
    except Exception as e:
        print(f"   ❌ Custom scenario test failed: {e}")
        import traceback
        traceback.print_exc()


def demo_configuration_flexibility():
    """Show examples of different configuration options."""
    print("⚙️  Configuration Flexibility Examples")
    print("-" * 40)
    
    examples = [
        {
            "name": "High-precision GEO scenario",
            "changes": {
                "altitude": "35,786 km (geostationary)",
                "physics": "Include solar radiation pressure",
                "timestep": "5 minutes (longer for GEO)",
                "duration": "24 hours (full GEO period)"
            }
        },
        {
            "name": "Dense LEO debris field", 
            "changes": {
                "debris": "500+ objects in single field",
                "distribution": "Realistic debris size distribution",
                "collision_detection": "High-precision ellipsoidal",
                "reward": "Fuel efficiency focused"
            }
        },
        {
            "name": "Formation flying",
            "changes": {
                "satellites": "4-8 satellites in tight formation",
                "constraints": "Maintain relative positions",
                "observations": "Include formation metrics",
                "actions": "Coordinated thrust commands"
            }
        },
        {
            "name": "Debris mitigation",
            "changes": {
                "objective": "Clear debris from orbit",
                "satellites": "Debris collection spacecraft",
                "rewards": "Debris removal bonus",
                "physics": "Contact dynamics modeling"
            }
        }
    ]
    
    for example in examples:
        print(f"📋 {example['name']}:")
        for key, value in example['changes'].items():
            print(f"   {key}: {value}")
        print()
    
    print("💡 All of these scenarios can be created by simply editing JSON files!")
    print("   No code changes required - just modify the configuration.")


if __name__ == "__main__":
    try:
        # Run all demos
        demo_scenario_switching()
        print()
        demo_custom_scenario() 
        print()
        demo_configuration_flexibility()
        
        print("\n🎉 Configuration system demo completed successfully!")
        print("\n📚 Summary:")
        print("✅ Multiple scenario templates available")
        print("✅ Easy JSON-based configuration")
        print("✅ Flexible satellite and debris setup")  
        print("✅ Configurable physics and rewards")
        print("✅ Multi-satellite support")
        print("✅ Modular, data-driven design")
        print("\n🚀 Ready for advanced orbital mechanics research!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()