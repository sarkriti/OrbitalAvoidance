"""Legacy shim for rl_algorithms re-exporting modern implementations."""
from __future__ import annotations
import sys
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

warnings.warn(
    "Importing rl_algorithms from legacy location is deprecated. Use 'orbital_avoidance.rl_algorithms' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from orbital_avoidance.rl_algorithms import *  # noqa: F401,F403
#!/usr/bin/env python3
"""
Modular RL Algorithm Framework

This module provides a unified interface for different reinforcement learning
algorithms, making it easy to switch between PPO, SAC, TD3, A2C, etc.

Features:
- Common interface for all algorithms
- Easy algorithm switching via configuration
- Automatic hyperparameter management
- Built-in logging and monitoring
- Checkpoint saving/loading
"""

from __future__ import annotations

import os
import json
import time
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List, Tuple
import numpy as np
import gymnasium as gym
from pathlib import Path

# Add parent directory to path for hardware detection import
sys.path.append(str(Path(__file__).parent.parent))

try:
    from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold,
        CallbackList, BaseCallback
    )
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("Warning: stable-baselines3 not available. Install with: pip install stable-baselines3")

try:
    from hardware_detection import get_optimal_config, configure_ml_frameworks, HardwareConfig
    HARDWARE_DETECTION_AVAILABLE = True
except ImportError:
    HARDWARE_DETECTION_AVAILABLE = False
    print("Warning: hardware_detection module not available. Using default CPU settings.")

import logging
logger = logging.getLogger(__name__)
"""Backward compatibility wrapper for RL algorithms module.

The implementation moved to `orbital_avoidance.algo.rl_algorithms`.
"""
import warnings
warnings.warn(
    "Importing 'rl_algorithms' from the old location is deprecated. Use 'from orbital_avoidance.algo import rl_algorithms' instead.",
    DeprecationWarning,
    stacklevel=2
)
from orbital_avoidance.algo.rl_algorithms import *  # noqa: F401,F403
            verbose=1
        )
        
        logger.info(f"SAC model created with batch_size={model_config['batch_size']}, buffer_size={model_config['buffer_size']}, device={getattr(self, 'device', 'cpu')}")
        
        return self.model
    
    def train(self, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """Train SAC model."""
        if self.model is None:
            self.create_model()
            
        start_time = time.time()
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        training_time = time.time() - start_time
        
        training_info = {
            'algorithm': 'SAC',
            'total_timesteps': total_timesteps,
            'training_time': training_time,
            'final_reward': list(getattr(self.model, 'ep_info_buffer', []))
        }
        
        self.training_history.append(training_info)
        return training_info
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict action using trained SAC model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str) -> None:
        """Save SAC model."""
        if self.model is not None:
            self.model.save(path)
            history_path = Path(path).parent / f"{Path(path).stem}_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load SAC model."""
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable-baselines3 required for SAC")
        self.model = SAC.load(path, env=self.env)


class TD3Algorithm(RLAlgorithmBase):
    """Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm wrapper."""
    
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        super().__init__(env, config)
        self.default_config = {
            'learning_rate': 1e-3,
            'buffer_size': 1000000,
            'learning_starts': 100,
            'batch_size': 100,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'policy_delay': 2,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                'net_arch': [400, 300]
            }
        }
    
    def create_model(self) -> TD3:
        """Create TD3 model."""
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable-baselines3 required for TD3")
            
        model_config = {**self.default_config, **self.config.get('td3', {})}
        
        # Apply hardware-specific optimizations
        if hasattr(self, 'hardware_config') and self.hardware_config:
            # Adjust batch size and buffer size based on available memory
            if self.hardware_config.use_gpu and self.hardware_config.gpu_info:
                gpu_memory_gb = self.hardware_config.gpu_info[0].memory_total_mb / 1024
                if gpu_memory_gb >= 8:
                    model_config['batch_size'] = min(256, model_config['batch_size'] * 2)
                    model_config['buffer_size'] = min(2000000, model_config['buffer_size'] * 2)
                elif gpu_memory_gb < 4:
                    model_config['batch_size'] = max(64, model_config['batch_size'] // 2)
                    model_config['buffer_size'] = max(500000, model_config['buffer_size'] // 2)
            else:
                # For CPU: adjust based on available RAM
                ram_gb = self.hardware_config.cpu_info.memory_available_gb
                if ram_gb >= 16:
                    model_config['batch_size'] = min(256, model_config['batch_size'] * 2)
                elif ram_gb < 8:
                    model_config['batch_size'] = max(64, model_config['batch_size'] // 2)
                    model_config['buffer_size'] = max(500000, model_config['buffer_size'] // 2)
            
            # Set device in policy kwargs if using GPU
            if self.hardware_config.use_gpu:
                if 'policy_kwargs' not in model_config:
                    model_config['policy_kwargs'] = {}
                model_config['policy_kwargs']['device'] = self.device
        
        self.model = TD3(
            policy=model_config['policy'],
            env=self.env,
            learning_rate=model_config['learning_rate'],
            buffer_size=model_config['buffer_size'],
            learning_starts=model_config['learning_starts'],
            batch_size=model_config['batch_size'],
            tau=model_config['tau'],
            gamma=model_config['gamma'],
            train_freq=model_config['train_freq'],
            gradient_steps=model_config['gradient_steps'],
            policy_delay=model_config['policy_delay'],
            target_policy_noise=model_config['target_policy_noise'],
            target_noise_clip=model_config['target_noise_clip'],
            policy_kwargs=model_config['policy_kwargs'],
            verbose=1
        )
        
        logger.info(f"TD3 model created with batch_size={model_config['batch_size']}, buffer_size={model_config['buffer_size']}, device={getattr(self, 'device', 'cpu')}")
        
        return self.model
    
    def train(self, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """Train TD3 model."""
        if self.model is None:
            self.create_model()
            
        start_time = time.time()
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        training_time = time.time() - start_time
        
        training_info = {
            'algorithm': 'TD3',
            'total_timesteps': total_timesteps,
            'training_time': training_time,
            'final_reward': list(getattr(self.model, 'ep_info_buffer', []))
        }
        
        self.training_history.append(training_info)
        return training_info
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict action using trained TD3 model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str) -> None:
        """Save TD3 model."""
        if self.model is not None:
            self.model.save(path)
            history_path = Path(path).parent / f"{Path(path).stem}_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load TD3 model."""
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable-baselines3 required for TD3")
        self.model = TD3.load(path, env=self.env)


class RLAlgorithmFactory:
    """Factory class for creating RL algorithm instances."""
    
    _algorithms = {
        'ppo': PPOAlgorithm,
        'sac': SACAlgorithm,
        'td3': TD3Algorithm,
    }
    
    @classmethod
    def create_algorithm(cls, algorithm_name: str, env: gym.Env, config: Dict[str, Any]) -> RLAlgorithmBase:
        """
        Create an RL algorithm instance.
        
        Args:
            algorithm_name: Name of algorithm ('ppo', 'sac', 'td3')
            env: Gym environment
            config: Algorithm configuration dictionary
            
        Returns:
            RLAlgorithmBase instance
        """
        algorithm_name = algorithm_name.lower()
        
        if algorithm_name not in cls._algorithms:
            available = ', '.join(cls._algorithms.keys())
            raise ValueError(f"Unknown algorithm '{algorithm_name}'. Available: {available}")
        
        algorithm_class = cls._algorithms[algorithm_name]
        return algorithm_class(env, config)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """Get list of available algorithm names."""
        return list(cls._algorithms.keys())
    
    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: type) -> None:
        """Register a new algorithm class."""
        cls._algorithms[name] = algorithm_class


class TrainingManager:
    """Manages RL training process with callbacks, logging, and evaluation."""
    
    def __init__(self, 
                 algorithm: RLAlgorithmBase,
                 log_dir: str = "./logs",
                 save_dir: str = "./models",
                 eval_env: Optional[gym.Env] = None):
        """
        Initialize training manager.
        
        Args:
            algorithm: RL algorithm instance
            log_dir: Directory for training logs
            save_dir: Directory for saving models
            eval_env: Optional evaluation environment
        """
        self.algorithm = algorithm
        self.log_dir = Path(log_dir)
        self.save_dir = Path(save_dir)
        self.eval_env = eval_env
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_callbacks(self, 
                       checkpoint_freq: int = 10000,
                       eval_freq: int = 5000,
                       reward_threshold: Optional[float] = None,
                       save_best_only: bool = True) -> CallbackList:
        """Setup training callbacks."""
        callbacks = []
        
        # Only add checkpoint callback if not saving best only
        if not save_best_only:
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(self.save_dir),
                name_prefix='model_checkpoint'
            )
            callbacks.append(checkpoint_callback)
        
        # Evaluation callback - this will save the best model
        if self.eval_env is not None:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=str(self.save_dir),
                log_path=str(self.log_dir),
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
                verbose=1  # Print when new best model is found
            )
            callbacks.append(eval_callback)
        else:
            # If no eval environment, we need at least some saving mechanism
            if save_best_only:
                logger.warning("save_best_only=True but no eval_env provided. Adding checkpoint callback.")
                checkpoint_callback = CheckpointCallback(
                    save_freq=checkpoint_freq,
                    save_path=str(self.save_dir),
                    name_prefix='model_checkpoint'
                )
                callbacks.append(checkpoint_callback)
        
        # Stop training on reward threshold
        if reward_threshold is not None:
            stop_callback = StopTrainingOnRewardThreshold(
                reward_threshold=reward_threshold,
                verbose=1
            )
            callbacks.append(stop_callback)
        
        return CallbackList(callbacks)
    
    def train(self, 
              total_timesteps: int,
              checkpoint_freq: int = 10000,
              eval_freq: int = 5000,
              reward_threshold: Optional[float] = None,
              save_best_only: bool = True) -> Dict[str, Any]:
        """
        Train the RL algorithm with full management.
        
        Args:
            total_timesteps: Total training timesteps
            checkpoint_freq: Frequency of model checkpoints (ignored if save_best_only=True)
            eval_freq: Frequency of evaluation
            reward_threshold: Stop training when this reward is reached
            save_best_only: If True, only save the best model based on evaluation performance
            
        Returns:
            Training information dictionary
        """
        # Setup logging
        if hasattr(self.algorithm.model, 'set_logger'):
            logger_obj = configure(str(self.log_dir), ["stdout", "csv", "tensorboard"])
            self.algorithm.model.set_logger(logger_obj)
        
        # Setup callbacks
        callbacks = self.setup_callbacks(checkpoint_freq, eval_freq, reward_threshold, save_best_only)
        
        # Train
        training_info = self.algorithm.train(
            total_timesteps=total_timesteps,
            callback=callbacks
        )
        
        # Save final model only if not using best-only saving or no eval environment
        if not save_best_only or self.eval_env is None:
            final_model_path = self.save_dir / "final_model"
            self.algorithm.save(str(final_model_path))
            logger.info(f"Final model saved to: {final_model_path}")
        else:
            logger.info(f"Best model saved during training to: {self.save_dir}/best_model.zip")
        
        return training_info


class RewardFunctionManager:
    """Manages different reward function configurations."""
    
    def __init__(self):
        self.reward_functions = {}
        self._register_default_functions()
    
    def _register_default_functions(self):
        """Register default reward function configurations."""
        
        # Conservative reward (prioritizes safety)
        self.reward_functions['conservative'] = {
            'weights': {
                'collision': 5.0,    # High collision penalty
                'fuel': 0.5,         # Low fuel penalty
                'orbit': 2.0,        # Medium orbit maintenance
                'mission': 1.0       # Standard mission reward
            },
            'collision': {
                'penalty': -2000.0,
                'threshold': 0.05,   # Lower threshold
                'small_reward': 2.0
            },
            'fuel': {
                'efficiency_weight': 0.05
            }
        }
        
        # Aggressive reward (prioritizes mission completion)
        self.reward_functions['aggressive'] = {
            'weights': {
                'collision': 2.0,    # Lower collision penalty
                'fuel': 2.0,         # Higher fuel penalty
                'orbit': 0.5,        # Lower orbit maintenance
                'mission': 3.0       # High mission reward
            },
            'collision': {
                'penalty': -500.0,
                'threshold': 0.2,    # Higher threshold
                'small_reward': 0.5
            },
            'fuel': {
                'efficiency_weight': 0.2
            }
        }
        
        # Balanced reward (your original specification)
        self.reward_functions['balanced'] = {
            'weights': {
                'collision': 1.0,
                'fuel': 1.0,
                'orbit': 1.0,
                'mission': 1.0
            },
            'collision': {
                'penalty': -1000.0,
                'threshold': 0.1,
                'small_reward': 1.0
            },
            'fuel': {
                'efficiency_weight': 0.1
            },
            'orbit': {
                'tolerances': {
                    'semi_major_axis_m': 100.0,
                    'eccentricity': 0.01,
                    'inclination_rad': 0.01,
                    'raan_rad': 0.01,
                    'arg_periapsis_rad': 0.01
                }
            },
            'mission': {
                'survival_reward_interval': 500,
                'survival_multiplier': 3.0,
                'mission_weight': 1.0
            }
        }
    
    def get_reward_config(self, name: str) -> Dict[str, Any]:
        """Get reward function configuration by name."""
        if name not in self.reward_functions:
            available = ', '.join(self.reward_functions.keys())
            raise ValueError(f"Unknown reward function '{name}'. Available: {available}")
        
        return self.reward_functions[name].copy()
    
    def register_reward_function(self, name: str, config: Dict[str, Any]) -> None:
        """Register a new reward function configuration."""
        self.reward_functions[name] = config
    
    def get_available_reward_functions(self) -> List[str]:
        """Get list of available reward function names."""
        return list(self.reward_functions.keys())


# Convenience functions for easy usage
def create_algorithm(algorithm_name: str, env: gym.Env, config: Dict[str, Any] = None) -> RLAlgorithmBase:
    """Convenience function to create an algorithm."""
    config = config or {}
    return RLAlgorithmFactory.create_algorithm(algorithm_name, env, config)


def get_reward_config(reward_name: str = 'balanced') -> Dict[str, Any]:
    """Convenience function to get reward configuration."""
    manager = RewardFunctionManager()
    return manager.get_reward_config(reward_name)