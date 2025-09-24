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
- Automatic CPU/GPU detection and optimization
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import logging

# Ensure src directory is on path for new package layout
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / 'src'
if SRC_DIR.exists():
    sys.path.insert(0, SRC_DIR.as_posix())

# Legacy self-directory also (for wrappers / backward compatibility)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Require the new unified package (src/ should be on PYTHONPATH or installed)
    from orbital_avoidance import (
        RLOrbitalEnvironment, create_rl_environment,
        RLAlgorithmFactory, TrainingManager, RewardFunctionManager,
        ConfigurationLoader
    )
except ImportError as e:
    raise ImportError(
        "Failed to import the modern 'orbital_avoidance' package.\n"
        "Make sure you run this script with the repository 'src' directory on PYTHONPATH (e.g. `PYTHONPATH=./src python rl_train.py`)"
        " or install the package in editable mode: `pip install -e src`.\n"
        f"Original import error: {e}"
    )

# -------------------------------------------------------------
# Path resolution helpers (scenarios & training configs)
# -------------------------------------------------------------
LEGACY_SCENARIO_DIRS = [
    REPO_ROOT / 'scenarios',
    REPO_ROOT / 'OrbitalAvoidance' / 'scenarios'
]
NEW_SCENARIO_DIR = REPO_ROOT / 'configs' / 'scenarios'
LEGACY_TRAINING_DIRS = [REPO_ROOT / 'training_configs']
NEW_TRAINING_DIR = REPO_ROOT / 'configs' / 'training'

def resolve_config_path(path_str: str, kind: str) -> str:
    """Resolve scenario or training config path with backward compatibility.

    Args:
        path_str: User-supplied path (may be relative like 'scenarios/foo.json')
        kind: 'scenario' or 'training'
    Returns:
        Absolute path to existing file.
    Raises:
        FileNotFoundError if not found in any known location.
    """
    p = Path(path_str)
    # If already exists as given
    if p.exists():
        return p.as_posix()
    candidates = []
    if kind == 'scenario':
        # If user passed legacy relative 'scenarios/...'
        if p.parts and p.parts[0] == 'scenarios':
            candidates.append(NEW_SCENARIO_DIR / '/'.join(p.parts[1:]))
        # Strip any leading scenarios/ and try in new dir
        candidates.append(NEW_SCENARIO_DIR / p.name)
        # Legacy dirs
        for d in LEGACY_SCENARIO_DIRS:
            candidates.append(d / p.name)
    elif kind == 'training':
        if p.parts and p.parts[0] in ('training_configs', 'configs'):
            # Replace prefix with new path
            if p.parts[0] == 'training_configs':
                candidates.append(NEW_TRAINING_DIR / '/'.join(p.parts[1:]))
        candidates.append(NEW_TRAINING_DIR / p.name)
        for d in LEGACY_TRAINING_DIRS:
            candidates.append(d / p.name)
    else:
        raise ValueError(f"Unknown config kind: {kind}")
    for c in candidates:
        if c.exists():
            if str(c).startswith(str(NEW_SCENARIO_DIR)) or str(c).startswith(str(NEW_TRAINING_DIR)):
                logger.info(f"Resolved {kind} path '{path_str}' -> '{c}' (new location)")
            else:
                logger.info(f"Resolved {kind} path '{path_str}' -> '{c}' (legacy location)")
            return c.as_posix()
    raise FileNotFoundError(f"Could not resolve {kind} config path: {path_str}. Tried: {[c.as_posix() for c in candidates]}")

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    VEC_ENV_AVAILABLE = True
except ImportError:
    VEC_ENV_AVAILABLE = False

try:
    from hardware_detection import print_hardware_info, configure_ml_frameworks
    HARDWARE_DETECTION_AVAILABLE = True
except ImportError:
    HARDWARE_DETECTION_AVAILABLE = False
    print("Warning: Hardware detection not available. Using default settings.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_config(config_file: str) -> Dict[str, Any]:
    """Load training configuration from JSON file."""
    # Try to resolve as training config first, then as scenario
    try:
        resolved = resolve_config_path(config_file, kind='training')
    except FileNotFoundError:
        # If not found as training config, try as scenario
        resolved = resolve_config_path(config_file, kind='scenario')
    
    with open(resolved, 'r') as f:
        config = json.load(f)
    
    # If this is a scenario file (doesn't have training section), add defaults
    if 'training' not in config:
        config.update({
            'algorithm': 'ppo',
            'scenario_file': config_file,
            'reward_function': 'balanced',
            'training': {
                'total_timesteps': 50000,
                'checkpoint_freq': 10000,
                'eval_freq': 5000,
                'reward_threshold': None,
                'save_best_only': True
            },
            'environment': {
                'max_debris_tracked': 10,
                'render_mode': None,
                'scenario_file': config_file,
                'episode_horizon': 1000
            },
            'output': {
                'log_dir': 'training_logs',
                'save_dir': 'trained_models',
                'experiment_name': 'scenario_training'
            },
            'ppo': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'clip_range': 0.2,
                'gae_lambda': 0.95,
                'gamma': 0.99,
            }
        })
    
    return config


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
            'render_mode': None,
            'scenario_file': args.scenario,
            'episode_horizon': args.episode_horizon
        },
        'output': {
            'log_dir': args.log_dir,
            'save_dir': args.save_dir,
            'experiment_name': args.experiment_name
        },
        # Hardware configuration
        'prefer_gpu': getattr(args, 'prefer_gpu', True),
        'gpu_memory_fraction': getattr(args, 'gpu_memory_fraction', 0.8),
        'max_threads': getattr(args, 'max_threads', None)
    }
    
    # Add algorithm-specific configuration
    if args.algorithm == 'ppo':
        config['ppo'] = {
            'learning_rate': args.learning_rate,
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'clip_range': args.clip_range,
            'gae_lambda': args.gae_lambda,
            'gamma': args.gamma,
            # Optional advanced PPO hyperparameters
            'ent_coef': getattr(args, 'ent_coef', None) if getattr(args, 'ent_coef', None) is not None else 0.0,
            'vf_coef': getattr(args, 'vf_coef', None) if getattr(args, 'vf_coef', None) is not None else 0.5,
            'max_grad_norm': getattr(args, 'max_grad_norm', None) if getattr(args, 'max_grad_norm', None) is not None else 0.5,
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
    
    # Handle both formats: scenario files directly and training configs with environment sections
    if 'environment' in config:
        # Full training config format
        scenario_file = config['environment']['scenario_file']
        max_debris = config['environment'].get('max_debris_tracked', 10)
        render_mode = config['environment'].get('render_mode', None)
        env_config = config['environment']
    else:
        # Simple scenario file format - use scenario_file at top level with defaults
        scenario_file = config.get('scenario_file', config.get('config', 'scenarios/leo_debris_avoidance.json'))
        max_debris = config.get('max_debris_tracked', 10)
        render_mode = config.get('render_mode', None)
        env_config = config
    
    logger.info(f"Setting up environment with scenario: {scenario_file}")
    
    # Get reward configuration - support both old and new formats
    reward_config = None
    if 'use_improved_rewards' in env_config:
        # New format with improved rewards
        reward_config = env_config.copy()
    elif 'reward_function' in config:
        # Old format
        reward_manager = RewardFunctionManager()
        reward_config = reward_manager.get_reward_config(config['reward_function'])
    
    # Create environment with proper parameters
    scenario_path = resolve_config_path(scenario_file, kind='scenario')
    env_params = {
        'config_file': scenario_path,
        'max_debris_tracked': max_debris,
        'render_mode': render_mode
    }
    
    if reward_config:
        env_params['reward_config'] = reward_config
    
    env = create_rl_environment(**env_params)

    # Wrap with Monitor BEFORE any vector/normalization wrappers so SB3 logs ep_rew_mean
    try:
        from stable_baselines3.common.monitor import Monitor
        import os
        log_dir = config.get('output', {}).get('log_dir', './logs')
        os.makedirs(log_dir, exist_ok=True)
        # Use a log file to track episode statistics
        monitor_file = os.path.join(log_dir, "monitor.csv")
        env = Monitor(env, filename=monitor_file)
        logger.info(f"Environment wrapped with Monitor, logging to: {monitor_file}")
    except Exception as e:
        logger.warning(f"Could not wrap env with Monitor: {e}")

    # Optional TimeLimit wrapper for shorter episodes (for better ep_rew_mean signal)
    episode_horizon = config.get('environment', {}).get('episode_horizon')
    if episode_horizon and episode_horizon > 0:
        try:
            import gymnasium as gym
            from gymnasium.wrappers import TimeLimit
            env = TimeLimit(env, max_episode_steps=episode_horizon)
            logger.info(f"Applied TimeLimit wrapper with horizon={episode_horizon}")
        except Exception as e:
            logger.warning(f"Failed to apply TimeLimit wrapper: {e}")

    # Optionally wrap in VecNormalize (only at training time, not for evaluation unless stats loaded)
    norm_cfg = config.get('normalization', {})
    if norm_cfg.get('enabled') and VEC_ENV_AVAILABLE:
        # Use DummyVecEnv wrapper
        env_vec = DummyVecEnv([lambda: env])
        env = VecNormalize(
            env_vec,
            training=True,
            norm_obs=norm_cfg.get('norm_obs', True),
            norm_reward=norm_cfg.get('norm_reward', True),
            clip_obs=norm_cfg.get('clip_obs', 10.0),
            clip_reward=norm_cfg.get('clip_reward', 10.0),
            gamma=config.get('ppo', {}).get('gamma', 0.99)
        )
        logger.info(f"VecNormalize enabled (obs={norm_cfg.get('norm_obs', True)} reward={norm_cfg.get('norm_reward', True)})")
    else:
        if norm_cfg.get('enabled') and not VEC_ENV_AVAILABLE:
            logger.warning("VecNormalize requested but stable_baselines3 not providing vec_env modules")
    
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
        # For eval env we disable normalization training mode
        eval_env = setup_environment(config)
        # If VecNormalize was used, set eval_env training False
        try:
            from stable_baselines3.common.vec_env import VecNormalize as VN
            if isinstance(env, VN):
                # Create a separate stats copy for evaluation (shared stats) if needed
                eval_env.training = False
        except Exception:
            pass
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

    # If using VecNormalize, save statistics
    try:
        from stable_baselines3.common.vec_env import VecNormalize as VN
        if isinstance(env, VN):
            stats_path = Path(output_config['save_dir']) / output_config['experiment_name'] / 'vecnormalize_stats.pkl'
            env.save(stats_path.as_posix())
            logger.info(f"Saved VecNormalize statistics to {stats_path}")
    except Exception as e:
        logger.warning(f"Could not save VecNormalize stats: {e}")
    
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
    
    # Initialize hardware detection and optimization
    if HARDWARE_DETECTION_AVAILABLE:
        print("🔍 Detecting and configuring hardware resources...")
        try:
            hardware_config = print_hardware_info()
            configure_ml_frameworks(hardware_config)
            print("✅ Hardware configuration complete!")
        except Exception as e:
            logger.warning(f"Hardware detection failed, using defaults: {e}")
    else:
        print("⚠️  Hardware detection not available, using default CPU settings")
    
    parser = argparse.ArgumentParser(description='Train RL agent for orbital debris avoidance')
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                       help='Mode: train a new agent or test an existing one')
    
    # Hardware configuration
    parser.add_argument('--prefer-gpu', action='store_true', default=True,
                       help='Prefer GPU over CPU when available (default: True)')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.8,
                       help='Fraction of GPU memory to use (0.1-1.0, default: 0.8)')
    parser.add_argument('--max-threads', type=int,
                       help='Maximum number of CPU threads to use (default: auto-detect)')
    
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
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='PPO rollout size (reduce e.g. 256 for faster early feedback)')
    parser.add_argument('--episode-horizon', type=int, default=0,
                       help='Force episode truncation at this many steps (0=disable)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='PPO batch size')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='PPO epochs per update (consider 5 for noisy rewards)')
    # Advanced PPO hyperparameters (optional)
    parser.add_argument('--ent-coef', type=float, default=None,
                       help='Entropy coefficient for PPO (optional)')
    parser.add_argument('--vf-coef', type=float, default=None,
                       help='Value function coefficient for PPO (optional)')
    parser.add_argument('--max-grad-norm', type=float, default=None,
                       help='Max gradient norm for optimizer (optional)')
    parser.add_argument('--clip-range', type=float, default=0.2,
                       help='PPO clip range')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    # Normalization flags
    parser.add_argument('--normalize', action='store_true', default=False,
                       help='Enable observation & reward normalization (VecNormalize)')
    parser.add_argument('--norm-obs', action='store_true', default=False,
                       help='Normalize observations (requires --normalize)')
    parser.add_argument('--norm-reward', action='store_true', default=False,
                       help='Normalize rewards (requires --normalize)')
    parser.add_argument('--clip-obs', type=float, default=10.0,
                       help='Observation clip value for normalization')
    parser.add_argument('--clip-reward', type=float, default=10.0,
                       help='Reward clip value for normalization')
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
                if args.normalize:
                    config['normalization'] = {
                        'enabled': True,
                        'norm_obs': args.norm_obs or True,  # default True when enabled
                        'norm_reward': args.norm_reward or True,  # default True when enabled
                        'clip_obs': args.clip_obs,
                        'clip_reward': args.clip_reward
                    }
            
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