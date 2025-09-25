#!/usr/bin/env python3
"""
RL Training Script for Orbital Debris Avoidance (Modernized)

This script provides a complete training pipeline for reinforcement learning
agents to control satellites in orbital debris fields.

Features:
- Easy algorithm switching (PPO, SAC, TD3)
- Configurable reward functions from src
- Optional physics fidelity flags
- Automatic logging and checkpointing
- Resume training from checkpoints
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
import logging
import gymnasium as gym

# Ensure src directory is on path for modern package imports
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / 'src'
if SRC_DIR.exists():
    sys.path.insert(0, SRC_DIR.as_posix())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from orbital_avoidance import (
        RLOrbitalEnvironment, create_rl_environment,
        RLAlgorithmFactory, TrainingManager, RewardFunctionManager
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import orbital_avoidance package. Ensure src/ is on PYTHONPATH or installed.\nOriginal error: {e}"
    )

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    VEC_ENV_AVAILABLE = True
except ImportError:
    VEC_ENV_AVAILABLE = False

# -------------------------------------------------------------
# Config resolution helpers
# -------------------------------------------------------------
def resolve_config_path(path_str: str, kind: str) -> str:
    p = Path(path_str)
    if p.exists():
        return p.as_posix()
    
    # New default locations
    if kind == 'scenario':
        candidates = [REPO_ROOT / 'configs' / 'scenarios' / p.name]
    elif kind == 'training':
        candidates = [REPO_ROOT / 'configs' / 'training' / p.name]
    else:
        raise ValueError(f"Unknown config kind: {kind}")

    # Legacy folders
    legacy_dirs = [REPO_ROOT / 'scenarios', REPO_ROOT / 'training_configs']
    for d in legacy_dirs:
        candidates.append(d / p.name)

    for c in candidates:
        if c.exists():
            logger.info(f"Resolved {kind} path '{path_str}' -> '{c}'")
            return c.as_posix()

    raise FileNotFoundError(f"Could not resolve {kind} config path: {path_str}. Tried: {[str(c) for c in candidates]}")

# -------------------------------------------------------------
# Training config helpers
# -------------------------------------------------------------
def load_training_config(config_file: str) -> Dict[str, Any]:
    """Load training configuration JSON or scenario file."""
    resolved_path = resolve_config_path(config_file, kind='training')
    with open(resolved_path, 'r') as f:
        config = json.load(f)
    
    # Default training config if missing
    if 'training' not in config:
        config['training'] = {
            'total_timesteps': 50000,
            'checkpoint_freq': 10000,
            'eval_freq': 5000,
            'save_best_only': True
        }
    return config

def create_training_config(args) -> Dict[str, Any]:
    """Construct training config from CLI arguments."""
    config = {
        'algorithm': args.algorithm,
        'scenario_file': args.scenario,
        'reward_function': args.reward,
        'training': {
            'total_timesteps': args.timesteps,
            'checkpoint_freq': args.checkpoint_freq,
            'eval_freq': args.eval_freq,
            'save_best_only': not args.save_all_checkpoints
        },
        'environment': {
            'scenario_file': args.scenario,
            'max_debris_tracked': args.max_debris,
            'episode_horizon': args.episode_horizon,
            'render_mode': None,
            'physics': {
                'gravity': not args.no_gravity,
                'drag': not args.no_drag,
                'collision_detection': not args.no_collision,
                'orbital_perturbations': not args.no_perturbations
            }
        },
        'output': {
            'log_dir': args.log_dir,
            'save_dir': args.save_dir,
            'experiment_name': args.experiment_name
        }
    }

    # PPO hyperparameters
    if args.algorithm == 'ppo':
        config['ppo'] = {
            'learning_rate': args.learning_rate,
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'clip_range': args.clip_range,
            'gae_lambda': args.gae_lambda,
            'gamma': args.gamma
        }

    return config

# -------------------------------------------------------------
# Environment setup
# -------------------------------------------------------------
def setup_environment(config: Dict[str, Any]) -> RLOrbitalEnvironment:
    """Setup RL environment with optional physics fidelity."""
    scenario_path = resolve_config_path(config['environment']['scenario_file'], kind='scenario')
    env_params = {
        'config_file': scenario_path,
        'max_debris_tracked': config['environment'].get('max_debris_tracked', 10),
        'render_mode': config['environment'].get('render_mode'),
        'reward_config': None
    }

    env = create_rl_environment(**env_params)

    # TimeLimit wrapper
    horizon = config['environment'].get('episode_horizon', 0)
    if horizon > 0:
        try:
            from gymnasium.wrappers import TimeLimit
            env = TimeLimit(env, max_episode_steps=horizon)
        except Exception as e:
            logger.warning(f"Failed to apply TimeLimit: {e}")

    # VecNormalize (if requested)
    norm_cfg = config.get('normalization', {})
    if norm_cfg.get('enabled') and VEC_ENV_AVAILABLE:
        env_vec = DummyVecEnv([lambda: env])
        env = VecNormalize(env_vec,
                           training=True,
                           norm_obs=norm_cfg.get('norm_obs', True),
                           norm_reward=norm_cfg.get('norm_reward', True),
                           clip_obs=norm_cfg.get('clip_obs', 10.0),
                           clip_reward=norm_cfg.get('clip_reward', 10.0),
                           gamma=config.get('ppo', {}).get('gamma', 0.99))
        logger.info("VecNormalize enabled")

    return env

# -------------------------------------------------------------
# Algorithm setup
# -------------------------------------------------------------
def setup_algorithm(env: RLOrbitalEnvironment, config: Dict[str, Any]):
    algorithm_name = config['algorithm']
    algorithm = RLAlgorithmFactory.create_algorithm(
        algorithm_name=algorithm_name,
        env=env,
        config=config
    )
    return algorithm

# -------------------------------------------------------------
# Training
# -------------------------------------------------------------
def train_agent(config: Dict[str, Any]):
    env = setup_environment(config)
    algorithm = setup_algorithm(env, config)
    output_config = config['output']
    log_dir = Path(output_config['log_dir']) / output_config['experiment_name']
    save_dir = Path(output_config['save_dir']) / output_config['experiment_name']

    training_manager = TrainingManager(
        algorithm=algorithm,
        log_dir=str(log_dir),
        save_dir=str(save_dir),
        eval_env=None
    )

    training_cfg = config['training']
    training_info = training_manager.train(
        total_timesteps=training_cfg['total_timesteps'],
        checkpoint_freq=training_cfg['checkpoint_freq'],
        eval_freq=training_cfg['eval_freq'],
        reward_threshold=training_cfg.get('reward_threshold'),
        save_best_only=training_cfg.get('save_best_only', True)
    )

    # Save final config
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("Training complete")
    return training_info

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train RL agent for orbital debris avoidance')

    parser.add_argument('--algorithm', choices=['ppo', 'sac', 'td3'], default='ppo')
    parser.add_argument('--scenario', default='scenarios/leo_debris_avoidance.json')
    parser.add_argument('--reward', choices=['balanced', 'conservative', 'aggressive'], default='balanced')
    parser.add_argument('--config', type=str)

    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--n-steps', type=int, default=2048)
    parser.add_argument('--episode-horizon', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--max-debris', type=int, default=10)
    parser.add_argument('--log-dir', default='./logs')
    parser.add_argument('--save-dir', default='./models')
    parser.add_argument('--experiment-name', default='orbital_rl_experiment')

    # Physics fidelity flags
    parser.add_argument('--no-gravity', action='store_true', help='Disable gravity')
    parser.add_argument('--no-drag', action='store_true', help='Disable drag')
    parser.add_argument('--no-collision', action='store_true', help='Disable collision detection')
    parser.add_argument('--no-perturbations', action='store_true', help='Disable orbital perturbations')

    parser.add_argument('--save-all-checkpoints', action='store_true')

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = load_training_config(args.config)
    else:
        config = create_training_config(args)

    # Train
    train_agent(config)

if __name__ == "__main__":
    main()
