#!/usr/bin/env python3
"""
Create Better Training Configuration

This script creates an improved training configuration that fixes the issues
identified in the diagnostic analysis.
"""

import json
import os
from pathlib import Path

def create_fixed_reward_config():
    """Create a better reward configuration."""
    return {
        "reward_type": "multi_objective_fixed",
        "collision_penalty": -50.0,  # Reduced from -1000
        "fuel_cost_weight": 0.01,    # Much smaller fuel penalty
        "safety_distance_km": 5.0,   # Larger safety zone
        "close_approach_penalty_weight": 2.0,  # Moderate penalty
        "mission_success_reward": 20.0,  # Good positive reward
        "survival_reward": 0.5,      # Small reward for staying alive
        "distance_reward_scale": 1.0  # Reward for maintaining distance
    }

def create_improved_ppo_config():
    """Create improved PPO hyperparameters."""
    return {
        "learning_rate": 0.00005,  # Much lower learning rate
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,  # Encourage exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
            "activation_fn": "tanh"
        }
    }

def create_better_training_config():
    """Create complete improved training configuration."""
    
    config = {
        "algorithm": "ppo",
        "scenario_file": "../scenarios/leo_debris_avoidance_fixed.json",
        "reward_function": "balanced_fixed",
        "training": {
            "total_timesteps": 1000000,  # More training time
            "checkpoint_freq": 50000,
            "eval_freq": 10000,  # More frequent evaluation
            "reward_threshold": 15.0,  # Stop when achieving good performance
            "save_best_only": True
        },
        "environment": {
            "max_debris_tracked": 5,  # Simpler environment to start
            "render_mode": None,
            "normalize_observations": True,  # Fix large observation values
            "episode_length": 200  # Shorter episodes
        },
        "output": {
            "log_dir": "./training_logs",
            "save_dir": "./trained_models", 
            "experiment_name": "debris_avoidance_ppo_fixed"
        },
        "ppo": create_improved_ppo_config()
    }
    
    return config

def create_fixed_scenario():
    """Create a fixed scenario configuration."""
    
    scenario = {
        "scenario_name": "LEO_Debris_Avoidance_Fixed",
        "description": "Fixed LEO debris avoidance with better reward structure",
        "satellites": [
            {
                "name": "MainSatellite",
                "orbit": {
                    "altitude_km": 400,
                    "inclination_deg": 51.6,
                    "eccentricity": 0.001,
                    "raan_deg": 0,
                    "arg_periapsis_deg": 0,
                    "mean_anomaly_deg": 0,
                    "epoch_mjd": 59945.0
                },
                "mass_kg": 1000,
                "drag_area_m2": 10,
                "drag_coefficient": 2.2,
                "radiation_area_m2": 10,
                "radiation_coefficient": 1.8,
                "is_controllable": True,
                "max_thrust_n": 5.0,  # Increased thrust for better control
                "fuel_capacity_kg": 200,  # More fuel
                "collision_radius_m": 10.0  # Larger collision radius
            }
        ],
        "debris_fields": [
            {
                "num_objects": 5,  # Fewer debris for simpler learning
                "altitude_range_km": [390, 410],
                "inclination_range_deg": [45, 60],
                "eccentricity_range": [0, 0.02],
                "mass_range_kg": [1, 50],
                "size_range_m": [0.5, 2.0],
                "distribution_type": "uniform"
            }
        ],
        "static_debris": [
            {
                "name": "LargeDebris_1",
                "orbit": {
                    "altitude_km": 405,
                    "inclination_deg": 51.6,
                    "eccentricity": 0.002,
                    "raan_deg": 10,
                    "arg_periapsis_deg": 45,
                    "mean_anomaly_deg": 180
                },
                "mass_kg": 200,
                "drag_area_m2": 3,
                "is_controllable": False,
                "max_thrust_n": 0,
                "fuel_capacity_kg": 0,
                "collision_radius_m": 5.0
            }
        ],
        "physics": {
            "propagator_type": "orekit_numerical",
            "time_step_seconds": 30,  # Larger time steps for faster training
            "simulation_duration_seconds": 6000,  # 1.67 hours max episode
            "include_drag": True,
            "include_solar_radiation": True,
            "include_j2_perturbation": True,
            "collision_detection_method": "spherical",
            "collision_threshold_km": 0.01
        },
        "rewards": create_fixed_reward_config(),
        "observations": {
            "include_relative_positions": True,
            "include_relative_velocities": True,
            "include_fuel_levels": True,
            "include_time_to_collision": True,
            "max_tracked_objects": 5,  # Smaller observation space
            "observation_radius_km": 50.0,  # Smaller observation radius
            "normalize_observations": True  # Normalize large values
        },
        "random_seed": 42
    }
    
    return scenario

def main():
    """Create all improved configuration files."""
    print("🔧 Creating Improved Training Configuration")
    print("=" * 50)
    
    # Create improved training config
    config = create_better_training_config()
    config_path = "training_configs/improved_ppo_training.json"
    
    os.makedirs("training_configs", exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created improved training config: {config_path}")
    
    # Create fixed scenario
    scenario = create_fixed_scenario()
    scenario_path = "scenarios/leo_debris_avoidance_fixed.json"
    
    os.makedirs("scenarios", exist_ok=True)
    
    with open(scenario_path, 'w') as f:
        json.dump(scenario, f, indent=2)
    
    print(f"✅ Created fixed scenario: {scenario_path}")
    
    print("\n🎯 KEY IMPROVEMENTS MADE:")
    print("1. ✅ Fixed reward function:")
    print("   - Reduced collision penalty: -1000 → -50")
    print("   - Added survival reward: +0.5 per step")
    print("   - Added distance-based rewards")
    print("   - Much smaller fuel penalties")
    
    print("\n2. ✅ Improved hyperparameters:")
    print("   - Reduced learning rate: 3e-4 → 5e-5")
    print("   - Added exploration bonus (ent_coef=0.01)")
    print("   - Better network architecture")
    
    print("\n3. ✅ Simplified environment:")
    print("   - Fewer debris objects (20 → 5)")
    print("   - Smaller observation space")
    print("   - Shorter episodes (3600s → 6000s max)")
    print("   - Larger collision radius for easier avoidance")
    
    print("\n4. ✅ Better training setup:")
    print("   - More timesteps: 500k → 1M")
    print("   - More frequent evaluation")
    print("   - Observation normalization")
    
    print("\n🚀 READY TO TRAIN!")
    print("Run this command to start improved training:")
    print(f"python3 rl_train.py --config {config_path}")
    
    print("\n📊 Monitor progress with:")
    print("tensorboard --logdir training_logs/")
    
    # Create a quick start script
    start_script = f"""#!/bin/bash
# Quick start script for improved training

echo "🚀 Starting improved PPO training..."
echo "Key fixes: reward function, hyperparameters, simpler environment"

cd /home/bryson/RL_testing/OrbitalAvoidance

python3 rl_train.py --config {config_path}

echo "✅ Training completed!"
echo "📊 View results with: tensorboard --logdir training_logs/"
"""
    
    with open("start_improved_training.sh", 'w') as f:
        f.write(start_script)
    
    os.chmod("start_improved_training.sh", 0o755)
    print(f"✅ Created quick start script: start_improved_training.sh")

if __name__ == "__main__":
    main()