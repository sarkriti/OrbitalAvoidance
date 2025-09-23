#!/usr/bin/env python3
"""
RL Training Script for Orbital Debris Avoidance

This script provides a complete training pipeline for reinforcement learning
agents to control satellites in orbital debris fields.

Usage:
    python rl_train.py --algorithm ppo --scenario scenarios/leo_debris_avoidance.json --reward balanced
    python rl_train.py --algorithm sac --scenario scenarios/multi_satellite_constellation.json --reward aggressive
    python rl_train.py --config training_configs/custom_training.json

Features:
- Easy algorithm switching (PPO, SAC, TD3)
- Configurable reward functions
- Automatic logging and checkpointing
- Evaluation and visualization
- Resume training from checkpoints
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import logging

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training_env import RLOrbitalEnvironment, create_rl_environment
from rl_algorithms import RLAlgorithmFactory, TrainingManager, RewardFunctionManager
from config_system import ConfigurationLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_config(config_file: str) -> Dict[str, Any]:
    """Load training configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def create_training_config(args) -> Dict[str, Any]:
    """Create training configuration from command line arguments."""
    # Determine save strategy
    save_best_only = not args.save_all_checkpoints  # Default is True unless explicitly saving all
    
    config = {
        'algorithm': args.algorithm,
        'scenario_file': args.scenario,
        'reward_function': args.reward,
        'training': {
            'total_timesteps': args.timesteps,
            'checkpoint_freq': args.checkpoint_freq,
            'eval_freq': args.eval_freq,
            'reward_threshold': args.reward_threshold,
            'save_best_only': save_best_only
        },
        'environment': {
            'max_debris_tracked': args.max_debris,
            'render_mode': None
        },
        'output': {
            'log_dir': args.log_dir,
            'save_dir': args.save_dir,
            'experiment_name': args.experiment_name
        }
    }
    
    # Add algorithm-specific configuration
    if args.algorithm == 'ppo':
        config['ppo'] = {
            'learning_rate': args.learning_rate,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
        }
    elif args.algorithm == 'sac':
        config['sac'] = {
            'learning_rate': args.learning_rate,
            'buffer_size': 1000000,
            'batch_size': 256,
        }
    elif args.algorithm == 'td3':
        config['td3'] = {
            'learning_rate': args.learning_rate,
            'buffer_size': 1000000,
            'batch_size': 100,
        }
    
    return config


def setup_environment(config: Dict[str, Any]) -> RLOrbitalEnvironment:
    """Setup the RL training environment."""
    logger.info(f"Setting up environment with scenario: {config['environment']['scenario_file']}")
    
    # Get reward configuration - support both old and new formats
    reward_config = None
    if 'environment' in config and 'use_improved_rewards' in config['environment']:
        # New format with improved rewards
        reward_config = config['environment'].copy()
    elif 'reward_function' in config:
        # Old format
        reward_manager = RewardFunctionManager()
        reward_config = reward_manager.get_reward_config(config['reward_function'])
    
    # Create environment with proper parameters
    env_params = {
        'config_file': config['environment']['scenario_file'],
        'max_debris_tracked': config['environment'].get('max_debris_tracked', 10),
        'render_mode': config['environment'].get('render_mode', None)
    }
    
    if reward_config:
        env_params['reward_config'] = reward_config
    
    env = create_rl_environment(**env_params)
    
    logger.info(f"Environment created successfully")
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    
    return env


def setup_algorithm(env: RLOrbitalEnvironment, config: Dict[str, Any]) -> Any:
    """Setup the RL algorithm."""
    algorithm_name = config['algorithm']
    logger.info(f"Setting up {algorithm_name.upper()} algorithm")
    
    algorithm = RLAlgorithmFactory.create_algorithm(
        algorithm_name=algorithm_name,
        env=env,
        config=config
    )
    
    logger.info(f"Algorithm created successfully")
    return algorithm


def train_agent(config: Dict[str, Any]) -> Dict[str, Any]:
    """Main training function."""
    logger.info("Starting RL training session")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Setup environment
    env = setup_environment(config)
    
    # Setup evaluation environment - needed for best model saving
    training_config = config['training']
    save_best_only = training_config.get('save_best_only', True)
    
    if save_best_only:
        logger.info("Creating evaluation environment for best model selection")
        eval_env = setup_environment(config)
    else:
        logger.info("Skipping evaluation environment (checkpoint-based saving)")
        eval_env = None
    
    # Setup algorithm
    algorithm = setup_algorithm(env, config)
    
    # Setup training manager
    output_config = config['output']
    log_dir = Path(output_config['log_dir']) / output_config['experiment_name']
    save_dir = Path(output_config['save_dir']) / output_config['experiment_name']
    
    training_manager = TrainingManager(
        algorithm=algorithm,
        log_dir=str(log_dir),
        save_dir=str(save_dir),
        eval_env=eval_env
    )
    
    # Start training
    training_config = config['training']
    logger.info(f"Starting training for {training_config['total_timesteps']} timesteps")
    
    # Check for save_best_only option
    save_best_only = training_config.get('save_best_only', True)  # Default to True
    if save_best_only:
        logger.info("Save mode: Best model only (based on evaluation performance)")
    else:
        logger.info("Save mode: All checkpoint models")
    
    training_info = training_manager.train(
        total_timesteps=training_config['total_timesteps'],
        checkpoint_freq=training_config['checkpoint_freq'],
        eval_freq=training_config['eval_freq'],
        reward_threshold=training_config.get('reward_threshold'),
        save_best_only=save_best_only
    )
    
    logger.info("Training completed successfully")
    logger.info(f"Training info: {training_info}")
    
    # Save final configuration
    config_save_path = save_dir / "training_config.json"
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return training_info


def test_agent(model_path: str, scenario_file: str, num_episodes: int = 10) -> Dict[str, Any]:
    """Test a trained agent."""
    logger.info(f"Testing agent from {model_path}")
    
    # Setup environment
    env = create_rl_environment(config_file=scenario_file)
    
    # Load algorithm (determine type from model file)
    # This is a simplified approach - in practice you'd save algorithm type with model
    try:
        from stable_baselines3 import PPO, SAC, TD3
        
        # Try loading as different algorithm types
        model = None
        for AlgorithmClass in [PPO, SAC, TD3]:
            try:
                model = AlgorithmClass.load(model_path, env=env)
                algorithm_name = AlgorithmClass.__name__
                break
            except:
                continue
        
        if model is None:
            raise ValueError("Could not load model with any known algorithm")
            
        logger.info(f"Loaded {algorithm_name} model successfully")
        
    except ImportError:
        raise ImportError("stable-baselines3 required for testing")
    
    # Run test episodes
    results = {
        'episodes': [],
        'total_reward': 0,
        'successful_episodes': 0,
        'average_fuel_remaining': 0,
        'average_collision_free_steps': 0
    }
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        collision_free_steps = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            collision_free_steps += 1
            done = terminated or truncated
        
        # Record episode results
        episode_result = {
            'episode': episode + 1,
            'total_reward': episode_reward,
            'collision_free_steps': collision_free_steps,
            'fuel_remaining_percent': info.get('fuel_remaining_percent', 0),
            'successful': not terminated  # Successful if not terminated by collision
        }
        
        results['episodes'].append(episode_result)
        results['total_reward'] += episode_reward
        if episode_result['successful']:
            results['successful_episodes'] += 1
        results['average_fuel_remaining'] += episode_result['fuel_remaining_percent']
        results['average_collision_free_steps'] += collision_free_steps
        
        logger.info(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                   f"Steps={collision_free_steps}, Fuel={episode_result['fuel_remaining_percent']:.1f}%")
    
    # Calculate averages
    results['average_reward'] = results['total_reward'] / num_episodes
    results['success_rate'] = results['successful_episodes'] / num_episodes
    results['average_fuel_remaining'] /= num_episodes
    results['average_collision_free_steps'] /= num_episodes
    
    logger.info(f"Test Results:")
    logger.info(f"  Average Reward: {results['average_reward']:.2f}")
    logger.info(f"  Success Rate: {results['success_rate']:.2%}")
    logger.info(f"  Average Fuel Remaining: {results['average_fuel_remaining']:.1f}%")
    logger.info(f"  Average Collision-Free Steps: {results['average_collision_free_steps']:.1f}")
    
    return results


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description='Train RL agent for orbital debris avoidance')
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                       help='Mode: train a new agent or test an existing one')
    
    # Training configuration
    parser.add_argument('--algorithm', choices=['ppo', 'sac', 'td3'], default='ppo',
                       help='RL algorithm to use')
    parser.add_argument('--scenario', default='scenarios/leo_debris_avoidance.json',
                       help='Scenario configuration file')
    parser.add_argument('--reward', choices=['balanced', 'conservative', 'aggressive'], default='balanced',
                       help='Reward function configuration')
    parser.add_argument('--config', type=str,
                       help='Training configuration JSON file (overrides other options)')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--checkpoint-freq', type=int, default=10000,
                       help='Checkpoint save frequency')
    parser.add_argument('--eval-freq', type=int, default=5000,
                       help='Evaluation frequency')
    parser.add_argument('--reward-threshold', type=float,
                       help='Stop training when this reward is reached')
    parser.add_argument('--save-best-only', action='store_true', default=False,
                       help='Only save the best model based on evaluation performance (default: True)')
    parser.add_argument('--save-all-checkpoints', action='store_true', default=False,
                       help='Save all checkpoint models during training (overrides save-best-only)')
    
    # Environment parameters
    parser.add_argument('--max-debris', type=int, default=10,
                       help='Maximum debris objects to track in state')
    
    # Output parameters
    parser.add_argument('--log-dir', default='./logs',
                       help='Directory for training logs')
    parser.add_argument('--save-dir', default='./models',
                       help='Directory for saving models')
    parser.add_argument('--experiment-name', default='orbital_rl_experiment',
                       help='Name for this experiment')
    
    # Testing parameters
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model for testing')
    parser.add_argument('--test-episodes', type=int, default=10,
                       help='Number of episodes for testing')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            # Load or create training configuration
            if args.config:
                config = load_training_config(args.config)
            else:
                config = create_training_config(args)
            
            # Start training
            training_info = train_agent(config)
            
            print("\n" + "="*50)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*50)
            print(f"Algorithm: {config['algorithm'].upper()}")
            print(f"Total timesteps: {config['training']['total_timesteps']:,}")
            print(f"Scenario: {config['scenario_file']}")
            print(f"Reward function: {config['reward_function']}")
            print(f"Models saved to: {config['output']['save_dir']}/{config['output']['experiment_name']}")
            print(f"Logs saved to: {config['output']['log_dir']}/{config['output']['experiment_name']}")
            
        elif args.mode == 'test':
            if not args.model_path:
                print("Error: --model-path required for testing mode")
                return
            
            # Test trained agent
            results = test_agent(
                model_path=args.model_path,
                scenario_file=args.scenario,
                num_episodes=args.test_episodes
            )
            
            print("\n" + "="*50)
            print("TESTING COMPLETED!")
            print("="*50)
            print(f"Model: {args.model_path}")
            print(f"Episodes: {args.test_episodes}")
            print(f"Average Reward: {results['average_reward']:.2f}")
            print(f"Success Rate: {results['success_rate']:.1%}")
            print(f"Average Fuel Remaining: {results['average_fuel_remaining']:.1f}%")
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()