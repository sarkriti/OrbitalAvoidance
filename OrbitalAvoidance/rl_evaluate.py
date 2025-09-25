#!/usr/bin/env python3
"""
RL Evaluation Script for Orbital Debris Avoidance
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np

from orbital_avoidance import create_rl_environment
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rl_eval")

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}


def evaluate_model(model_path, algo, env, episodes, max_steps):
    """Evaluate a trained RL model on the given environment."""
    model_cls = ALGOS[algo.lower()]
    model = model_cls.load(model_path, env=env)

    results = {"collisions": 0, "near_misses": 0, "rewards": []}

    for ep in range(episodes):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        truncated = False
        ep_reward = 0
        collision_free_steps = 0

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            collision_free_steps += 1

            if info.get("collision", False):
                results["collisions"] += 1
                break
            if info.get("near_miss", False):
                results["near_misses"] += 1

            if done or truncated:
                break

        results["rewards"].append(ep_reward)

    results["avg_reward"] = float(np.mean(results["rewards"]))
    results["std_reward"] = float(np.std(results["rewards"]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to evaluation JSON config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Setup environment with optional physics fidelity flags
    physics_flags = cfg.get("physics", {
        "gravity": True,
        "drag": True,
        "collision_detection": True,
        "orbital_perturbations": True
    })
    env = create_rl_environment(cfg["scenario_file"], physics=physics_flags)

    # Handle VecNormalize if stats are provided
    vecnormalize_path = cfg.get("vecnormalize_stats_path")
    if vecnormalize_path:
        if Path(vecnormalize_path).exists() and isinstance(env, VecNormalize):
            env.load(vecnormalize_path)
            env.training = False  # Ensure eval mode
            logger.info(f"Loaded VecNormalize stats from {vecnormalize_path}")

    for model_cfg in cfg["models"]:
        path = Path(model_cfg["path"])
        if not path.exists():
            logger.warning(f"Model file {path} not found, skipping.")
            continue

        algo = model_cfg["algo"]
        episodes = model_cfg.get("episodes", 20)
        max_steps = model_cfg.get("max_steps", 1000)

        logger.info(f"Evaluating {algo.upper()} model from {path}")
        res = evaluate_model(path, algo, env, episodes, max_steps)
        logger.info(f"Results:\n{json.dumps(res, indent=2)}")


if __name__ == "__main__":
    main()
