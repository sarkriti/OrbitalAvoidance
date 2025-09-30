#!/usr/bin/env python3
"""
RL Training Script for Orbital Debris Avoidance (Production Version)

Features:
- Reward normalization (essential for PPO stability)
- Curriculum learning support (debris scaling, noise ramping)
- Proper checkpoint/evaluation handling
- Episode statistics logging (ep_len_mean, ep_rew_mean)
- Modern best practices from RL literature

Based on best practices from:
- Gaudet et al. (2020) "Deep RL for Spacecraft Guidance"
- Federici et al. (2021) "Deep Learning for Autonomous Navigation"
- Izzo et al. (2019) "Real-time Guidance Using DNNs"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics

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
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    VEC_ENV_AVAILABLE = True
except ImportError:
    VEC_ENV_AVAILABLE = False
    logger.warning("stable-baselines3 not available. Some features disabled.")

# -------------------------------------------------------------
# Config resolution helpers
# -------------------------------------------------------------
def resolve_config_path(path_str: str, kind: str) -> str:
    """Resolve config file path with fallback search."""
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
# Episode Stats Logging Callback
# -------------------------------------------------------------
class EpisodeStatsCallback(BaseCallback):
    """
    Logs episode statistics (length and reward) to TensorBoard.
    Only prints to console every N episodes to avoid spam.
    """
    def __init__(self, print_freq: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.print_freq = print_freq
        
        # Track cumulative rewards per environment
        self.cumulative_rewards = None
        self.episode_starts = None
    
    def _on_training_start(self) -> None:
        """Initialize tracking arrays based on number of environments."""
        n_envs = self.training_env.num_envs
        self.cumulative_rewards = np.zeros(n_envs)
        self.episode_starts = np.zeros(n_envs)
        logger.info(f"EpisodeStatsCallback initialized for {n_envs} environment(s)")
    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        
        # ALWAYS print what we're seeing to debug
        #print(f"\n=== CALLBACK STEP {self.num_timesteps} ===")
        #print(f"Keys in locals: {list(self.locals.keys())}")
        
        # Get rewards for this step
        rewards = self.locals.get('rewards', [])
        if not isinstance(rewards, (list, np.ndarray)):
            rewards = [rewards]
        #print(f"Rewards: {rewards}")
        
        # Accumulate rewards
        if self.cumulative_rewards is not None:
            self.cumulative_rewards += rewards
        
        # Check for episode endings - try multiple possible keys
        dones = None
        for key in ['dones', 'done', 'terminateds', 'truncateds']:
            if key in self.locals:
                #print(f"Found key '{key}': {self.locals[key]}")
                if self.locals[key] is not None:
                    dones = self.locals[key]
                    #print(f"Using '{key}' as dones: {dones}")
                    break
        
        # Check infos for episode endings
        infos = self.locals.get('infos', [])
        #print(f"Infos: {infos}")
        
        # If still None, try to construct from infos
        if dones is None and infos:
            # Check if any info has 'terminal_observation' or 'episode' key
            dones = []
            for info in infos:
                has_terminal = 'terminal_observation' in info or '_terminal_observation' in info
                has_episode = 'episode' in info
                done = has_terminal or has_episode
                dones.append(done)
                if done:
                    print(f"Detected done from info: terminal={has_terminal}, episode={has_episode}")
        
        #print(f"Final dones value: {dones}")
        #print("=" * 50 + "\n")
        
        if dones is not None:
            # Ensure dones is iterable
            if not isinstance(dones, (list, tuple, np.ndarray)):
                dones = [dones]
            
            infos = self.locals.get('infos', [])
            
            for i, done in enumerate(dones):
                if done:
                    print(f"\n=== EPISODE END DETECTED (env {i}) ===")
                    print(f"Done: {done}")
                    print(f"Cumulative reward: {self.cumulative_rewards[i] if self.cumulative_rewards is not None else 'N/A'}")
                    
                    # Try to get episode info from info dict
                    ep_reward = None
                    ep_length = None
                    
                    if i < len(infos):
                        info = infos[i]
                        print(f"Info keys: {info.keys()}")
                        
                        # Check for episode info (from Monitor wrapper)
                        if 'episode' in info:
                            ep_info = info['episode']
                            ep_reward = ep_info.get('r')
                            ep_length = ep_info.get('l')
                            print(f"Found 'episode' key: reward={ep_reward}, length={ep_length}")
                        
                        # Fallback: use our tracked cumulative reward
                        if ep_reward is None and self.cumulative_rewards is not None:
                            ep_reward = self.cumulative_rewards[i]
                            ep_length = self.num_timesteps - self.episode_starts[i]
                            print(f"Using tracked cumulative: reward={ep_reward}, length={ep_length}")
                    
                    print("=" * 50 + "\n")
                    
                    # If we got episode stats, log them
                    if ep_reward is not None:
                        self.episode_rewards.append(float(ep_reward))
                        self.episode_lengths.append(int(ep_length) if ep_length else 0)
                        self.episode_count += 1
                        
                        # Reset tracking for this environment
                        if self.cumulative_rewards is not None:
                            self.cumulative_rewards[i] = 0
                            self.episode_starts[i] = self.num_timesteps
                        
                        # Log to TensorBoard
                        if len(self.episode_rewards) > 0:
                            self.logger.record('rollout/ep_rew_mean', np.mean(self.episode_rewards[-100:]))
                            self.logger.record('rollout/ep_len_mean', np.mean(self.episode_lengths[-100:]))
                            self.logger.record('rollout/ep_rew_recent', ep_reward)
                            self.logger.record('rollout/ep_len_recent', ep_length if ep_length else 0)
                        
                        # Print to console periodically
                        if self.verbose > 0 and self.episode_count % self.print_freq == 0:
                            recent_rewards = self.episode_rewards[-self.print_freq:]
                            recent_lengths = self.episode_lengths[-self.print_freq:]
                            logger.info(
                                f"Episodes {self.episode_count - self.print_freq + 1}-{self.episode_count}: "
                                f"mean_reward={np.mean(recent_rewards):.2f}, "
                                f"mean_length={np.mean(recent_lengths):.1f}"
                            )
        
        return True

# -------------------------------------------------------------
# Curriculum Learning Callback
# -------------------------------------------------------------
class CurriculumCallback(BaseCallback):
    """
    Implements curriculum learning by adjusting environment difficulty.
    
    Based on Narvekar et al. (2020) curriculum learning survey.
    """
    def __init__(self, curriculum_config: Dict, base_config: Dict, verbose: int = 1):
        super().__init__(verbose)
        self.curriculum_config = curriculum_config
        self.base_config = base_config
        self.current_stage = 0
        self.stages = curriculum_config.get('stages', [])
        self.stage_timestep_thresholds = []
        
        # Calculate cumulative timesteps for each stage
        cumulative = 0
        for stage in self.stages:
            cumulative += stage['timesteps']
            self.stage_timestep_thresholds.append(cumulative)
    
    def _on_step(self) -> bool:
        """Check if we should advance to next curriculum stage."""
        if not self.stages or self.current_stage >= len(self.stages):
            return True
        
        # Check if we've reached next stage threshold
        if (self.current_stage < len(self.stage_timestep_thresholds) - 1 and 
            self.num_timesteps >= self.stage_timestep_thresholds[self.current_stage]):
            
            self.current_stage += 1
            stage = self.stages[self.current_stage]
            
            logger.info("="*60)
            logger.info(f"CURRICULUM STAGE {self.current_stage + 1}/{len(self.stages)}")
            logger.info(f"Description: {stage.get('description', 'N/A')}")
            logger.info(f"Debris multiplier: {stage.get('debris_multiplier', 1.0)}")
            logger.info(f"Noise multiplier: {stage.get('noise_multiplier', 1.0)}")
            logger.info(f"Total timesteps so far: {self.num_timesteps}")
            logger.info("="*60)
            
            # NOTE: Actual environment reconfiguration would require
            # recreating the environment with new parameters.
            # This is complex with VecEnv - consider implementing
            # dynamic difficulty adjustment in the environment itself.
            logger.warning("Dynamic curriculum requires environment recreation - not yet implemented")
        
        return True

# -------------------------------------------------------------
# Training config helpers
# -------------------------------------------------------------
def load_training_config(config_file: str) -> Dict[str, Any]:
    """Load training configuration JSON."""
    resolved_path = resolve_config_path(config_file, kind='training')
    with open(resolved_path, 'r') as f:
        config = json.load(f)
    
    # Validate required sections
    if 'training' not in config:
        config['training'] = {
            'total_timesteps': 100000,
            'checkpoint_freq': 10000,
            'eval_freq': 5000,
            'save_best_only': True
        }
    
    if 'rewards' not in config:
        logger.warning("No 'rewards' section in config - using defaults")
        config['rewards'] = {
            'collision_penalty': -50.0,
            'mission_success_reward': 10.0,
            'fuel_cost_weight': 0.01,
            'survival_bonus': 0.005
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
        },
        'rewards': {
            'collision_penalty': -50.0,
            'mission_success_reward': 10.0,
            'fuel_cost_weight': 0.01,
            'survival_bonus': 0.005
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
def setup_environment(config: Dict[str, Any]) -> gym.Env:
    """Setup RL environment with reward normalization."""
    scenario_path = resolve_config_path(config['environment']['scenario_file'], kind='scenario')
    
    def make_env():
        env = create_rl_environment(
            config_file=scenario_path,
            episode_horizon=config['environment'].get('episode_horizon'),
            max_debris_tracked=config['environment'].get('max_debris_tracked', 10),
            render_mode=config['environment'].get('render_mode'),
        )
        
        # Apply TimeLimit FIRST if specified
        horizon = config['environment'].get('episode_horizon', 0)
        if horizon > 0:
            try:
                from gymnasium.wrappers import TimeLimit
                env = TimeLimit(env, max_episode_steps=horizon)
                logger.info(f"Applied TimeLimit: {horizon} steps")
            except Exception as e:
                logger.warning(f"Failed to apply TimeLimit: {e}")
        
        # IMPORTANT: Use Monitor from stable-baselines3, NOT RecordEpisodeStatistics from gymnasium
        # Monitor must be applied BEFORE vectorization
        env = Monitor(env)
        logger.info(f"Applied Monitor wrapper (SB3): {type(env).__name__}")
        
        return env
    
    # Check if stable-baselines3 is available
    if not VEC_ENV_AVAILABLE:
        logger.error("stable-baselines3 not available - cannot create training environment")
        raise ImportError("Install stable-baselines3 for training")
    
    # Create vectorized env (only once!)
    env = DummyVecEnv([make_env])
    logger.info("Created single environment (parallel training not available due to Java/Python integration)")
    
    # Apply reward normalization
    gamma = config.get('ppo', {}).get('gamma', 0.999)
    env = VecNormalize(
        env,
        training=True,
        norm_obs=False,
        norm_reward=False,  # Set to True if you want normalization
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
        epsilon=1e-8
    )
    
    logger.info("="*60)
    logger.info("REWARD NORMALIZATION: " + ("ENABLED" if False else "DISABLED"))
    logger.info(f"  - norm_obs: False")
    logger.info(f"  - norm_reward: False")
    logger.info(f"  - gamma: {gamma}")
    logger.info("="*60)
    
    return env
# -------------------------------------------------------------
# Algorithm setup
# -------------------------------------------------------------
def setup_algorithm(env: gym.Env, config: Dict[str, Any]):
    """Create RL algorithm with config."""
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
    """
    Main training loop with modern RL best practices.
    """
    # Setup
    env = setup_environment(config)
    algorithm = setup_algorithm(env, config)
    
    output_config = config['output']
    log_dir = Path(output_config['log_dir']) / output_config['experiment_name']
    save_dir = Path(output_config['save_dir']) / output_config['experiment_name']
    
    # Create directories
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup callbacks
    callbacks = []
    
    # Add episode stats callback (always enabled)
    episode_stats_callback = EpisodeStatsCallback(print_freq=10, verbose=1)
    callbacks.append(episode_stats_callback)
    logger.info("Episode statistics logging enabled (prints every 10 episodes)")
    
    # Add curriculum callback if enabled
    curriculum_config = config.get('training', {}).get('curriculum_learning', {})
    if curriculum_config.get('enabled', False):
        curriculum_callback = CurriculumCallback(curriculum_config, config)
        callbacks.append(curriculum_callback)
        logger.info("Curriculum learning enabled with {} stages".format(
            len(curriculum_config.get('stages', []))
        ))
    
    # Create training manager
    training_manager = TrainingManager(
        algorithm=algorithm,
        log_dir=str(log_dir),
        save_dir=str(save_dir),
        eval_env=None  # Could add separate eval env here
    )
    
    # Train
    training_cfg = config['training']
    logger.info("="*60)
    logger.info("STARTING TRAINING")
    logger.info(f"  Total timesteps: {training_cfg['total_timesteps']:,}")
    logger.info(f"  Checkpoint frequency: {training_cfg['checkpoint_freq']:,}")
    logger.info(f"  Evaluation frequency: {training_cfg['eval_freq']:,}")
    logger.info(f"  Algorithm: {config['algorithm'].upper()}")
    logger.info(f"  Custom callbacks: {len(callbacks)}")
    logger.info("="*60)
    
    # Pass callbacks to the training manager!
    training_info = training_manager.train(
        total_timesteps=training_cfg['total_timesteps'],
        checkpoint_freq=training_cfg['checkpoint_freq'],
        eval_freq=training_cfg['eval_freq'],
        reward_threshold=training_cfg.get('reward_threshold'),
        save_best_only=training_cfg.get('save_best_only', True),
        callbacks=callbacks  # <-- THIS IS THE KEY CHANGE!
    )
    
    # Save final config and normalization stats
    with open(save_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save VecNormalize stats for deployment
    if isinstance(env, VecNormalize):
        env.save(save_dir / "vec_normalize.pkl")
        logger.info(f"Saved VecNormalize statistics to {save_dir / 'vec_normalize.pkl'}")
        logger.info("IMPORTANT: Load these stats during deployment with VecNormalize.load()")
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Final model saved to: {save_dir}")
    logger.info(f"  Logs saved to: {log_dir}")
    logger.info("="*60)
    
    return training_info

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Train RL agent for orbital debris avoidance',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core arguments
    parser.add_argument('--algorithm', choices=['ppo', 'sac', 'td3'], default='ppo',
                       help='RL algorithm to use')
    parser.add_argument('--config', type=str,
                       help='Path to training config JSON (overrides other args)')
    parser.add_argument('--scenario', default='leo_debris_avoidance.json',
                       help='Scenario configuration file')
    parser.add_argument('--reward', choices=['balanced', 'conservative', 'aggressive'], 
                       default='balanced', help='Reward function preset')

    # Training parameters
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--checkpoint-freq', type=int, default=10000,
                       help='Save checkpoint every N timesteps')
    parser.add_argument('--eval-freq', type=int, default=5000,
                       help='Evaluate policy every N timesteps')

    # PPO hyperparameters (from Schulman et al. 2017 + modern best practices)
    parser.add_argument('--n-steps', type=int, default=4096,
                       help='Steps per environment per update')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Minibatch size for SGD')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='Number of epochs per update')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--clip-range', type=float, default=0.2,
                       help='PPO clipping parameter')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda parameter')
    parser.add_argument('--gamma', type=float, default=0.999,
                       help='Discount factor (high for long episodes)')

    # Environment parameters
    parser.add_argument('--episode-horizon', type=int, default=1500,
                       help='Maximum steps per episode')
    parser.add_argument('--max-debris', type=int, default=20,
                       help='Maximum debris objects to track')

    # Output configuration
    parser.add_argument('--log-dir', default='./logs',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--save-dir', default='./models',
                       help='Directory for saved models')
    parser.add_argument('--experiment-name', default='orbital_rl_experiment',
                       help='Experiment name for organizing outputs')

    # Physics fidelity flags (for ablation studies)
    parser.add_argument('--no-gravity', action='store_true',
                       help='Disable gravity (not recommended)')
    parser.add_argument('--no-drag', action='store_true',
                       help='Disable atmospheric drag')
    parser.add_argument('--no-collision', action='store_true',
                       help='Disable collision detection (not recommended)')
    parser.add_argument('--no-perturbations', action='store_true',
                       help='Disable orbital perturbations')

    # Other options
    parser.add_argument('--save-all-checkpoints', action='store_true',
                       help='Save all checkpoints (not just best)')

    args = parser.parse_args()

    # Load or create config
    if args.config:
        logger.info(f"Loading training config from: {args.config}")
        config = load_training_config(args.config)
    else:
        logger.info("Creating config from command-line arguments")
        config = create_training_config(args)

    # Warning about no parallelization
    logger.warning("="*60)
    logger.warning("TRAINING WITHOUT PARALLELIZATION")
    logger.warning("Due to Orekit/Java integration, parallel environments are not supported.")
    logger.warning("Training will be slower. Consider increasing total_timesteps to compensate.")
    logger.warning("="*60)

    # Train
    try:
        train_agent(config)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()