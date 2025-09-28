#!/usr/bin/env python3
"""
Simple Collision/Near-Miss Counter for Orbital Debris Avoidance
Runs for 10,000+ steps and counts misses/near-misses with high debris density
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np

from orbital_avoidance import create_rl_environment
from stable_baselines3 import PPO, SAC, TD3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("collision_counter")

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}


def count_collisions_and_misses(model_path, algo, env, total_steps, near_miss_threshold_km=0.050):
    """Run model for specified steps and count collisions/near-misses."""
    model_cls = ALGOS[algo.lower()]
    model = model_cls.load(model_path, env=env)

    results = {
        "total_steps": 0,
        "collisions": 0,
        "near_misses": 0,
        "collision_steps": [],  # Step numbers when collisions occurred
        "near_miss_steps": [],  # Step numbers when near-misses occurred
        "min_distances": [],    # Track all minimum distances
        "episodes": 0
    }

    # Reset environment
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    step_count = 0
    episode_count = 1

    logger.info(f"Starting {total_steps} step collision test...")
    logger.info(f"Near-miss threshold: {near_miss_threshold_km} km")
    
    while step_count < total_steps:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1

        # Calculate minimum distance to all debris
        if hasattr(env, 'current_debris_states') and hasattr(env, 'current_satellite_state'):
            sat_pos = env.current_satellite_state.position_km
            min_distance = float('inf')
            
            for debris in env.current_debris_states:
                distance = np.linalg.norm(debris.position_km - sat_pos)
                min_distance = min(min_distance, distance)
            
            results["min_distances"].append(min_distance)
            
            # Get collision radii
            sat_collision_radius = getattr(env.current_satellite_state, 'collision_radius', 0.001)
            debris_collision_radius = 0.001  # Default 1m
            if env.current_debris_states:
                debris_collision_radius = getattr(env.current_debris_states[0], 'collision_radius', 0.001)
            
            total_collision_radius = sat_collision_radius + debris_collision_radius
            
            # Check for collision
            if min_distance <= total_collision_radius:
                results["collisions"] += 1
                results["collision_steps"].append(step_count)
                logger.warning(f"COLLISION at step {step_count}! Distance: {min_distance:.6f} km")
            
            # Check for near-miss (closer than threshold but not collision)
            elif min_distance <= near_miss_threshold_km:
                results["near_misses"] += 1
                results["near_miss_steps"].append(step_count)
                if len(results["near_miss_steps"]) % 100 == 0:  # Log every 100th near-miss
                    logger.info(f"Near-miss #{len(results['near_miss_steps'])} at step {step_count}, distance: {min_distance:.6f} km")

        # Log progress every 1000 steps
        if step_count % 1000 == 0:
            logger.info(f"Step {step_count}/{total_steps} - Collisions: {results['collisions']}, Near-misses: {results['near_misses']}")

        # Handle episode resets
        if done or truncated:
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            episode_count += 1

    results["total_steps"] = step_count
    results["episodes"] = episode_count
    
    # Calculate statistics
    if results["min_distances"]:
        results["avg_min_distance"] = float(np.mean(results["min_distances"]))
        results["std_min_distance"] = float(np.std(results["min_distances"]))
        results["closest_approach"] = float(np.min(results["min_distances"]))
    
    results["collision_rate"] = results["collisions"] / step_count
    results["near_miss_rate"] = results["near_misses"] / step_count
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to evaluation config with high debris density")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--algo", type=str, choices=['ppo', 'sac', 'td3'], required=True,
                        help="Algorithm used for training")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Total steps to run (default: 10000)")
    parser.add_argument("--near-miss-threshold", type=float, default=0.050,
                        help="Near-miss threshold in km (default: 0.050)")
    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        logger.error(f"Model file {args.model} not found!")
        return

    # Load config
    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Create high-density debris environment
    env_kwargs = {
        'config_file': args.config,
        'max_debris_tracked': cfg.get("max_debris_tracked", 50),  # Track more debris
        'episode_horizon': args.steps + 1000,  # Longer episodes
        'render_mode': None
    }
    
    env = create_rl_environment(**env_kwargs)
    
    logger.info(f"Environment created with {len(env._base_env.debris_objects)} debris objects")
    logger.info(f"Testing model: {args.model}")
    logger.info(f"Algorithm: {args.algo.upper()}")

    # Run collision counting test
    results = count_collisions_and_misses(
        args.model, 
        args.algo, 
        env, 
        args.steps,
        args.near_miss_threshold
    )

    # Print results
    print("\n" + "="*60)
    print("COLLISION/NEAR-MISS TEST RESULTS")
    print("="*60)
    print(f"Total Steps:           {results['total_steps']:,}")
    print(f"Episodes:              {results['episodes']}")
    print(f"Collisions:            {results['collisions']}")
    print(f"Near-misses:           {results['near_misses']}")
    print(f"Collision Rate:        {results['collision_rate']:.6f} per step")
    print(f"Near-miss Rate:        {results['near_miss_rate']:.6f} per step")
    
    if 'avg_min_distance' in results:
        print(f"Avg Min Distance:      {results['avg_min_distance']:.6f} km")
        print(f"Closest Approach:      {results['closest_approach']:.6f} km")
    
    print(f"Near-miss Threshold:   {args.near_miss_threshold} km")
    
    # Safety assessment
    total_events = results['collisions'] + results['near_misses']
    if total_events > 0:
        collision_percentage = (results['collisions'] / total_events) * 100
        print(f"\nSafety Analysis:")
        print(f"  Total Risk Events:   {total_events}")
        print(f"  Collision %:         {collision_percentage:.2f}%")
        print(f"  Near-miss %:         {100-collision_percentage:.2f}%")
    
    # Save detailed results
    output_file = f"collision_test_results_{args.algo}_{args.steps}steps.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    env.close()


if __name__ == "__main__":
    main()