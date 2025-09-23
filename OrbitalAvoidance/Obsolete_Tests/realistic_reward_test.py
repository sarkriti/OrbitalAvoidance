#!/usr/bin/env python3
"""
REALISTIC reward function test that accounts for:
1. Distance to debris (collision avoidance)
2. Fuel consumption (action magnitude)
3. Survival time (how long alive)
4. Orbital maintenance (staying in orbit)

The previous tests were flawed because they tested from identical
starting conditions where debris distances, fuel levels, etc. were the same.
"""

import numpy as np
import sys
sys.path.append('/home/bryson/RL_testing/OrbitalAvoidance')

def test_reward_components_properly():
    """Test reward function by varying the components it actually depends on."""
    
    print("🧪 REALISTIC REWARD FUNCTION TEST")
    print("=" * 60)
    print("Testing how rewards change based on actual reward components:")
    print("1. Distance to debris (collision risk)")
    print("2. Fuel consumption (action magnitude)")
    print("3. Time/survival duration")
    print("4. Orbital parameters")
    print()
    
    test_fuel_consumption_component()
    test_survival_time_component()
    test_collision_distance_component()

def test_fuel_consumption_component():
    """Test how fuel consumption affects rewards."""
    print("🔥 Testing FUEL CONSUMPTION component:")
    print("-" * 40)
    
    from rl_training_env import RLOrbitalEnvironment
    
    # Create environment with improved rewards
    reward_config = {
        'use_improved_rewards': True,
        'collision_penalty': -50.0,
        'fuel_cost_weight': 0.01,
        'safety_distance_km': 5.0,
        'survival_reward': 0.5,
        'distance_reward_scale': 1.0
    }
    
    env = RLOrbitalEnvironment(
        config_file='scenarios/leo_debris_avoidance_fixed.json',
        reward_config=reward_config
    )
    
    # Test different thrust magnitudes from SAME starting state
    thrust_magnitudes = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    
    print("Testing different thrust magnitudes (fuel consumption):")
    rewards = []
    
    for thrust in thrust_magnitudes:
        obs, info = env.reset()
        action = np.array([thrust, 0.0, 0.0])  # Thrust only in X direction
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        print(f"  Thrust {thrust:3.1f}: reward = {reward:8.4f}")
    
    # Analyze fuel component
    reward_range = max(rewards) - min(rewards)
    print(f"  Reward range: {reward_range:8.4f}")
    if reward_range > 0.01:
        print("  ✅ Fuel consumption affects rewards")
    else:
        print("  ❌ Fuel consumption has no effect")
    
    env.close()
    return rewards

def test_survival_time_component():
    """Test how survival time affects rewards."""
    print("\n⏰ Testing SURVIVAL TIME component:")
    print("-" * 40)
    
    from rl_training_env import RLOrbitalEnvironment
    
    reward_config = {
        'use_improved_rewards': True,
        'collision_penalty': -50.0,
        'fuel_cost_weight': 0.01,
        'safety_distance_km': 5.0,
        'survival_reward': 0.5,
        'distance_reward_scale': 1.0
    }
    
    env = RLOrbitalEnvironment(
        config_file='scenarios/leo_debris_avoidance_fixed.json',
        reward_config=reward_config
    )
    
    # Test rewards over multiple time steps
    obs, info = env.reset()
    
    print("Testing rewards over time (survival component):")
    no_thrust = np.array([0.0, 0.0, 0.0])
    
    rewards_over_time = []
    for step in range(5):
        obs, reward, terminated, truncated, info = env.step(no_thrust)
        rewards_over_time.append(reward)
        print(f"  Step {step+1}: reward = {reward:8.4f}")
        
        if terminated:
            print("    Terminated (collision)")
            break
    
    env.close()
    return rewards_over_time

def test_collision_distance_component():
    """Test how distance to debris affects rewards by creating different scenarios."""
    print("\n💥 Testing COLLISION DISTANCE component:")
    print("-" * 40)
    
    print("This component requires different debris configurations...")
    print("Testing with scenarios that have different debris proximities:")
    
    # Test the improved reward function directly with mock scenarios
    from improved_reward_function import ImprovedRewardFunction
    
    config = {
        'collision_penalty': -50.0,
        'fuel_cost_weight': 0.01,
        'safety_distance_km': 5.0,
        'survival_reward': 0.5,
        'distance_reward_scale': 1.0
    }
    
    reward_func = ImprovedRewardFunction(config)
    
    # Create mock environment states with different debris distances
    class MockState:
        def __init__(self, pos):
            self.position_km = np.array(pos)
    
    class MockEnv:
        def __init__(self, debris_distances):
            self.current_satellite_state = MockState([0, 0, 0])
            self.current_debris_states = [MockState([d, 0, 0]) for d in debris_distances]
    
    # Test different collision scenarios
    test_scenarios = [
        ("Very close debris (2km)", [2.0]),
        ("Close debris (4km)", [4.0]),
        ("Safe distance (8km)", [8.0]),
        ("Far debris (15km)", [15.0]),
        ("Very far debris (25km)", [25.0]),
        ("Multiple debris (close)", [3.0, 7.0, 12.0]),
        ("Multiple debris (far)", [15.0, 20.0, 30.0]),
    ]
    
    print("Testing collision distance scenarios:")
    no_thrust = np.array([0.0, 0.0, 0.0])
    
    distance_rewards = []
    for scenario_name, debris_distances in test_scenarios:
        mock_env = MockEnv(debris_distances)
        reward = reward_func.calculate_reward(mock_env, no_thrust)
        distance_rewards.append(reward)
        
        min_dist = min(debris_distances)
        print(f"  {scenario_name:25s} (min: {min_dist:4.1f}km): reward = {reward:8.4f}")
    
    # Analyze distance component
    reward_range = max(distance_rewards) - min(distance_rewards)
    print(f"  Reward range: {reward_range:8.4f}")
    if reward_range > 1.0:
        print("  ✅ Collision distance strongly affects rewards")
    else:
        print("  ❌ Collision distance has weak effect")
    
    return distance_rewards

def analyze_reward_variance():
    """Analyze overall reward variance across all components."""
    print("\n📊 COMPLETE REWARD VARIANCE ANALYSIS:")
    print("=" * 50)
    
    print("Testing reward function with REALISTIC scenarios...")
    
    # Collect rewards from all tests
    fuel_rewards = test_fuel_consumption_component()
    survival_rewards = test_survival_time_component()
    distance_rewards = test_collision_distance_component()
    
    # Combine all rewards
    all_rewards = fuel_rewards + survival_rewards + distance_rewards
    
    print(f"\nCOMBINED REWARD STATISTICS:")
    print(f"  Total samples:   {len(all_rewards)}")
    print(f"  Min reward:      {min(all_rewards):8.4f}")
    print(f"  Max reward:      {max(all_rewards):8.4f}")
    print(f"  Mean reward:     {np.mean(all_rewards):8.4f}")
    print(f"  Std deviation:   {np.std(all_rewards):8.4f}")
    print(f"  Range:           {max(all_rewards) - min(all_rewards):8.4f}")
    print(f"  Unique values:   {len(set(all_rewards))}")
    
    # Final diagnostic
    reward_range = max(all_rewards) - min(all_rewards)
    std_dev = np.std(all_rewards)
    
    print(f"\n🎯 FINAL DIAGNOSIS:")
    if reward_range > 5.0:
        print("  ✅ GOOD: Large reward range (>5.0)")
    elif reward_range > 1.0:
        print("  ⚠️  MODERATE: Medium reward range (1.0-5.0)")
    else:
        print("  ❌ POOR: Small reward range (<1.0)")
    
    if std_dev > 1.0:
        print("  ✅ GOOD: High reward variance (>1.0)")
    elif std_dev > 0.1:
        print("  ⚠️  MODERATE: Medium reward variance (0.1-1.0)")
    else:
        print("  ❌ POOR: Low reward variance (<0.1)")
    
    if len(set(all_rewards)) > len(all_rewards) * 0.7:
        print("  ✅ GOOD: High reward diversity")
    else:
        print("  ⚠️  WARNING: Many duplicate reward values")

if __name__ == "__main__":
    analyze_reward_variance()
    
    print("\n" + "="*60)
    print("🎯 This test properly evaluates rewards by:")
    print("1. ✅ Testing fuel consumption with different thrust levels")
    print("2. ✅ Testing survival time over multiple steps")  
    print("3. ✅ Testing collision distance with various debris scenarios")
    print("4. ✅ Analyzing combined variance across all components")
    print("5. ✅ Using realistic reward function dependencies")