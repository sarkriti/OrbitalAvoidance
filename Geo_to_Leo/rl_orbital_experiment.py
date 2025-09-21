#!/usr/bin/env python3
"""
RL Orbital Transfer Learning Experiment
=====================================

This demonstrates why RL is valuable for orbital mechanics:
Compare RL-learned trajectories vs traditional Hohmann transfers
"""

import numpy as np
import matplotlib.pyplot as plt
from satellite_env import SatelliteOrbitControlEnv
from stable_baselines3 import PPO
import time

def compare_transfer_methods():
    """Compare RL vs traditional orbital transfer methods"""
    
    print("🚀 RL vs Traditional Orbital Transfer Comparison")
    print("=" * 50)
    
    # Create environment
    env = SatelliteOrbitControlEnv()
    
    # Traditional Hohmann Transfer
    print("\n📊 Traditional Hohmann Transfer:")
    hohmann_dv = calculate_hohmann_dv(env.initial_altitude, env.target_altitude)
    print(f"   Theoretical ΔV: {hohmann_dv:.1f} m/s")
    print(f"   Transfer time: ~5.25 hours (fixed)")
    print(f"   Fuel efficiency: Standard")
    
    # Test if we have a trained model
    try:
        model = PPO.load("satellite_rl_model_final")
        print("\n🧠 RL-Learned Transfer:")
        
        # Test RL performance
        obs, _ = env.reset()
        total_dv = 0
        steps = 0
        
        for _ in range(1000):  # Max steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_dv += np.linalg.norm(action) * env.dt * 1000  # Convert to m/s
            steps += 1
            
            if done or truncated:
                break
        
        transfer_time = steps * env.dt / 3600  # hours
        print(f"   Learned ΔV: {total_dv:.1f} m/s")
        print(f"   Transfer time: {transfer_time:.2f} hours")
        
        # Calculate efficiency
        efficiency = hohmann_dv / total_dv * 100
        print(f"   Fuel efficiency: {efficiency:.1f}% of Hohmann")
        
        if efficiency > 100:
            print("   🎉 RL BEATS TRADITIONAL METHOD!")
            print("   RL discovered more efficient trajectory")
        elif efficiency > 90:
            print("   ✅ RL performance is competitive")
        else:
            print("   📈 RL needs more training")
            
    except FileNotFoundError:
        print("\n⚠️  No trained RL model found")
        print("   Run satellite_rl_training.py first to train a model")
    
    # Explain why RL can be better
    print("\n🔬 Why RL Can Outperform Traditional Methods:")
    print("   • Learns optimal timing for burns")
    print("   • Adapts to specific mission constraints")
    print("   • Discovers non-intuitive trajectories")
    print("   • Handles equipment failures automatically")
    print("   • Optimizes for multiple objectives simultaneously")

def calculate_hohmann_dv(r1_alt, r2_alt):
    """Calculate theoretical Hohmann transfer ΔV"""
    # Earth parameters
    R_earth = 6371000  # m
    mu = 3.986e14  # m³/s²
    
    # Orbit radii
    r1 = R_earth + r1_alt * 1000  # m
    r2 = R_earth + r2_alt * 1000  # m
    
    # Hohmann transfer ΔV
    v1 = np.sqrt(mu / r1)  # Initial circular velocity
    v2 = np.sqrt(mu / r2)  # Final circular velocity
    
    # Transfer orbit velocities
    v_transfer_1 = np.sqrt(mu * (2/r1 - 2/(r1+r2)))  # At perigee
    v_transfer_2 = np.sqrt(mu * (2/r2 - 2/(r1+r2)))  # At apogee
    
    # Total ΔV
    dv1 = abs(v_transfer_1 - v1)  # Departure burn
    dv2 = abs(v2 - v_transfer_2)  # Insertion burn
    
    return (dv1 + dv2)  # m/s

def explain_rl_advantages():
    """Explain specific advantages of RL for orbital mechanics"""
    
    print("\n🎯 Specific RL Advantages for Space Missions:")
    print("=" * 50)
    
    advantages = [
        {
            "scenario": "Fuel-Optimal Transfers",
            "traditional": "Use fixed Hohmann/bi-elliptic transfers",
            "rl_advantage": "Learn custom trajectories for specific fuel constraints",
            "impact": "10-20% fuel savings = months of extra mission time"
        },
        {
            "scenario": "Multi-Satellite Coordination", 
            "traditional": "Plan each satellite independently",
            "rl_advantage": "Learn coordinated maneuvers to avoid collisions",
            "impact": "Enable large constellations (Starlink, etc.)"
        },
        {
            "scenario": "Equipment Failures",
            "traditional": "Ground control replanning (hours/days)",
            "rl_advantage": "Instant adaptation using remaining thrusters",
            "impact": "Mission survival vs. total loss"
        },
        {
            "scenario": "Complex Gravity Fields",
            "traditional": "Simplified mathematical models",
            "rl_advantage": "Learn from full n-body gravitational dynamics", 
            "impact": "Precise asteroid/moon missions"
        },
        {
            "scenario": "Real-Time Debris Avoidance",
            "traditional": "Pre-planned avoidance maneuvers",
            "rl_advantage": "Dynamic collision avoidance with minimal fuel",
            "impact": "Safe operation in cluttered orbital environment"
        }
    ]
    
    for i, adv in enumerate(advantages, 1):
        print(f"\n{i}. {adv['scenario']}:")
        print(f"   Traditional: {adv['traditional']}")
        print(f"   RL Advantage: {adv['rl_advantage']}")
        print(f"   Impact: {adv['impact']}")

if __name__ == "__main__":
    compare_transfer_methods()
    explain_rl_advantages()
    
    print("\n🚀 Want to see RL in action?")
    print("   Run: python satellite_rl_training.py")
    print("   Then: python rl_orbital_experiment.py")