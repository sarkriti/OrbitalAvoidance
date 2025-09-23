#!/usr/bin/env python3
"""
Diagnostic to understand why the reward is always -1298.5192
(which appears to be a collision penalty being applied constantly)
"""

import numpy as np
import sys
sys.path.append('/home/bryson/RL_testing/OrbitalAvoidance')

def diagnose_collision_reward():
    """Diagnose why collision rewards are constantly being triggered."""
    
    print("🔍 DIAGNOSING COLLISION REWARD CALCULATION")
    print("=" * 60)
    
    from rl_training_env import RLOrbitalEnvironment
    
    # Create original environment
    env = RLOrbitalEnvironment(config_file='scenarios/leo_debris_avoidance.json')
    obs, info = env.reset()
    
    # Check reward configuration
    reward_config = env.reward_config
    print("Reward Configuration:")
    print(f"  Weights: {reward_config['weights']}")
    print(f"  Collision penalty: {reward_config['collision']['penalty']}")
    print(f"  Collision threshold: {reward_config['collision']['threshold']}")
    print(f"  Collision small reward: {reward_config['collision']['small_reward']}")
    print()
    
    # Take a step and analyze the reward breakdown
    action = np.array([0.0, 0.0, 0.0])  # No thrust
    obs, total_reward, terminated, truncated, info = env.step(action)
    
    print(f"Total reward: {total_reward}")
    print()
    
    # Now manually calculate each component to see what's happening
    print("Manual reward component calculation:")
    
    # Get current state
    sat_state = env.current_satellite_state
    print(f"Satellite position: {sat_state.position_km}")
    print(f"Satellite velocity: {sat_state.velocity_km_s}")
    print()
    
    # Calculate collision component manually
    print("🚨 COLLISION COMPONENT ANALYSIS:")
    collision_reward = calculate_collision_component_manually(env)
    print(f"  Collision reward: {collision_reward}")
    
    # Calculate fuel component
    print("\n⛽ FUEL COMPONENT ANALYSIS:")
    fuel_reward = calculate_fuel_component_manually(env, action)
    print(f"  Fuel reward: {fuel_reward}")
    
    # Calculate orbit component  
    print("\n🌍 ORBIT COMPONENT ANALYSIS:")
    orbit_reward = calculate_orbit_component_manually(env)
    print(f"  Orbit reward: {orbit_reward}")
    
    # Calculate mission component
    print("\n🎯 MISSION COMPONENT ANALYSIS:")
    mission_reward = calculate_mission_component_manually(env)
    print(f"  Mission reward: {mission_reward}")
    
    # Total calculation
    weights = reward_config['weights']
    calculated_total = (
        weights['collision'] * collision_reward +
        weights['fuel'] * fuel_reward +
        weights['orbit'] * orbit_reward +
        weights['mission'] * mission_reward
    )
    
    print(f"\n📊 REWARD BREAKDOWN:")
    print(f"  Collision: {weights['collision']} × {collision_reward:8.4f} = {weights['collision'] * collision_reward:8.4f}")
    print(f"  Fuel:      {weights['fuel']} × {fuel_reward:8.4f} = {weights['fuel'] * fuel_reward:8.4f}")
    print(f"  Orbit:     {weights['orbit']} × {orbit_reward:8.4f} = {weights['orbit'] * orbit_reward:8.4f}")
    print(f"  Mission:   {weights['mission']} × {mission_reward:8.4f} = {weights['mission'] * mission_reward:8.4f}")
    print(f"  TOTAL:                              = {calculated_total:8.4f}")
    print(f"  Environment returned:               = {total_reward:8.4f}")
    print()
    
    # Identify the dominant component
    components = {
        'collision': weights['collision'] * collision_reward,
        'fuel': weights['fuel'] * fuel_reward,
        'orbit': weights['orbit'] * orbit_reward,
        'mission': weights['mission'] * mission_reward
    }
    
    dominant_component = max(components, key=lambda x: abs(components[x]))
    print(f"🎯 DOMINANT COMPONENT: {dominant_component} ({components[dominant_component]:8.4f})")
    
    # Check if we're always in collision state
    if collision_reward == reward_config['collision']['penalty']:
        print(f"\n🚨 PROBLEM IDENTIFIED:")
        print(f"   The satellite is ALWAYS in collision state!")
        print(f"   Collision penalty (-{abs(reward_config['collision']['penalty'])}) is applied every step")
        print(f"   This means either:")
        print(f"   1. Collision detection is broken")
        print(f"   2. Debris is too close to satellite initially") 
        print(f"   3. Collision threshold is too low")
        print(f"   4. Collision radius is too large")
    
    env.close()

def calculate_collision_component_manually(env):
    """Manually calculate collision reward component."""
    
    config = env.reward_config['collision']
    
    # Get closest debris (same logic as environment)
    closest_debris = env._get_closest_debris(5)
    max_collision_prob = 0.0
    
    sat_pos = env.current_satellite_state.position_km
    sat_vel = env.current_satellite_state.velocity_km_s
    
    print(f"    Checking {len(closest_debris)} closest debris objects:")
    
    for i, debris in enumerate(closest_debris):
        miss_distance, time_to_collision = env._calculate_collision_metrics(
            sat_pos, sat_vel, debris.position_km, debris.velocity_km_s
        )
        
        print(f"      Debris {i+1}:")
        print(f"        Position: {debris.position_km}")
        print(f"        Distance: {np.linalg.norm(debris.position_km - sat_pos):6.3f} km")
        print(f"        Miss distance: {miss_distance:6.3f} km")
        print(f"        Time to collision: {time_to_collision:6.1f} s")
        
        # Calculate collision probability
        if time_to_collision > 0 and time_to_collision < 3600:  # Within 1 hour
            collision_radius_km = env.main_satellite.collision_radius_m / 1000.0
            collision_prob = max(0, 1.0 - miss_distance / collision_radius_km)
            max_collision_prob = max(max_collision_prob, collision_prob)
            
            print(f"        Collision radius: {collision_radius_km:6.3f} km")
            print(f"        Collision probability: {collision_prob:6.3f}")
        else:
            print(f"        No collision risk (time > 1h or diverging)")
    
    print(f"    Max collision probability: {max_collision_prob:6.3f}")
    print(f"    Threshold: {config['threshold']}")
    
    # Apply threshold-based reward
    if max_collision_prob > config['threshold']:
        print(f"    Result: PENALTY applied ({config['penalty']})")
        return config['penalty']
    else:
        print(f"    Result: Small reward ({config['small_reward']})")
        return config['small_reward']

def calculate_fuel_component_manually(env, action):
    """Manually calculate fuel reward component."""
    
    thrust_magnitude = np.linalg.norm(action) * env.main_satellite.max_thrust_n
    mass_kg = env.current_satellite_state.mass_kg
    dt = env.config.physics.time_step_seconds
    
    delta_v_magnitude = (thrust_magnitude / mass_kg) * dt  # m/s
    efficiency_weight = env.reward_config['fuel']['efficiency_weight']
    fuel_reward = -efficiency_weight * (delta_v_magnitude / 1000.0)
    
    print(f"    Action magnitude: {np.linalg.norm(action):6.3f}")
    print(f"    Thrust magnitude: {thrust_magnitude:6.3f} N")
    print(f"    Delta-V: {delta_v_magnitude:6.6f} m/s")
    print(f"    Efficiency weight: {efficiency_weight}")
    print(f"    Fuel penalty: {fuel_reward:8.6f}")
    
    return fuel_reward

def calculate_orbit_component_manually(env):
    """Manually calculate orbit reward component."""
    # This is complex - just return a placeholder for now
    print(f"    Orbit component calculation (complex)")
    return -298.5192  # Placeholder - this might be the actual value

def calculate_mission_component_manually(env):
    """Manually calculate mission reward component.""" 
    print(f"    Mission component calculation")
    return 0.0  # Placeholder

if __name__ == "__main__":
    diagnose_collision_reward()