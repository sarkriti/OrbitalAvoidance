"""RL algorithms (migrated and refactored package version)."""

from __future__ import annotations
import os
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
from pathlib import Path
import numpy as np
import gymnasium as gym

# --- Stable Baselines 3 and Hardware Detection Imports ---
try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold, CallbackList, BaseCallback
    )
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    STABLE_BASELINES_AVAILABLE = True
except ImportError:  # pragma: no cover
    # Define dummy classes for type hinting if SB3 is not available
    BaseAlgorithm = object
    STABLE_BASELINES_AVAILABLE = False

try:
    from hardware_detection import get_optimal_config, configure_ml_frameworks
    HARDWARE_DETECTION_AVAILABLE = True
except ImportError:  # pragma: no cover
    HARDWARE_DETECTION_AVAILABLE = False

logger = logging.getLogger(__name__)


class RLAlgorithmBase(ABC):
    """
    Refactored abstract base class for RL algorithms.
    Handles the entire model lifecycle (creation, training, saving, loading).
    """
    def __init__(self,
                 env: gym.Env,
                 config: Dict[str, Any],
                 model_class: Type[BaseAlgorithm],
                 algorithm_name: str):
        self.env = env
        self.config = config
        self.model_class = model_class
        self.algorithm_name = algorithm_name
        self.model: Optional[BaseAlgorithm] = None
        self.training_history = []
        self.hardware_config = None
        self.device = 'cpu'
        self._setup_hardware()

    def _setup_hardware(self):  # pragma: no cover
        if HARDWARE_DETECTION_AVAILABLE:
            try:
                self.hardware_config = get_optimal_config(
                    prefer_gpu=self.config.pop('prefer_gpu', True),
                    memory_fraction=self.config.pop('gpu_memory_fraction', 0.8),
                    max_threads=self.config.pop('max_threads', None)
                )
                configure_ml_frameworks(self.hardware_config)
                self.device = self.hardware_config.torch_device
                logger.info(f"Hardware configured to use device: {self.device}")
            except Exception as e:
                logger.warning(f"Hardware auto-configuration failed: {e}. Defaulting to 'cpu'.")
                self.device = 'cpu'
        else:
            self.device = self.config.pop('device', 'cpu')

    def create_model(self) -> BaseAlgorithm:
        """Creates the Stable-Baselines3 model instance."""
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError(f"stable-baselines3 is required to use the {self.algorithm_name} algorithm.")

        # Unpack the config directly. SB3 models ignore unknown keys.
        self.model = self.model_class(
            env=self.env,
            verbose=1,
            device=self.device,
            **self.config
        )
        return self.model

    def train(self, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """Generic training loop, now located in the base class."""
        if self.model is None:
            self.create_model()

        start_time = time.time()
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        training_duration = time.time() - start_time

        # Get final reward info from the Monitor wrapper's buffer
        final_reward_mean = None
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            final_reward_mean = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]).item()

        info = {
            'algorithm': self.algorithm_name,
            'total_timesteps': total_timesteps,
            'training_time_seconds': round(training_duration, 2),
            'final_mean_reward': final_reward_mean
        }
        self.training_history.append(info)
        return info

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generic prediction, now located in the base class."""
        if self.model is None:
            raise ValueError("Model is not created or loaded. Call create_model() or load() first.")
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str):
        """Generic model and history saving, now located in the base class."""
        if self.model:
            self.model.save(path)
            history_path = Path(path).with_suffix(".history.json")
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            logger.info(f"Saved model to {path} and history to {history_path}")

    def load(self, path: str):
        """Generic model loading, now located in the base class."""
        self.model = self.model_class.load(path, env=self.env)
        logger.info(f"Loaded model from {path}")


class PPOAlgorithm(RLAlgorithmBase):
    """Simplified PPO implementation. Only defines hyperparameters."""
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        default_config = {
            'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64,
            'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95,
            'clip_range': 0.2, 'ent_coef': 0.0, 'vf_coef': 0.5, 'max_grad_norm': 0.5,
            'use_sde': False, 'policy': 'MlpPolicy',
            # Larger policy and value networks for PPO (recommended starting point)
            'policy_kwargs': {'net_arch': dict(pi=[512, 512, 256], vf=[512, 512, 256])}
        }
        final_config = {**default_config, **config.get('ppo', {})}

        super().__init__(
            env=env,
            config=final_config,
            model_class=PPO,
            algorithm_name='PPO'
        )


class SACAlgorithm(RLAlgorithmBase):
    """Simplified SAC implementation. Only defines hyperparameters."""
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        default_config = {
            'learning_rate': 3e-4, 'buffer_size': 1_000_000, 'learning_starts': 100,
            'batch_size': 256, 'tau': 0.005, 'gamma': 0.99, 'train_freq': 1,
            'gradient_steps': 1, 'ent_coef': 'auto', 'target_update_interval': 1,
            'target_entropy': 'auto', 'use_sde': False, 'policy': 'MlpPolicy',
            # Larger shared actor-critic network for SAC
            'policy_kwargs': {'net_arch': [512, 512, 256]}
        }
        final_config = {**default_config, **config.get('sac', {})}

        super().__init__(
            env=env,
            config=final_config,
            model_class=SAC,
            algorithm_name='SAC'
        )


class TD3Algorithm(RLAlgorithmBase):
    """Simplified TD3 implementation. Only defines hyperparameters."""
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        default_config = {
            'learning_rate': 1e-3, 'buffer_size': 1_000_000, 'learning_starts': 100,
            'batch_size': 100, 'tau': 0.005, 'gamma': 0.99,
            'train_freq': 1, 'gradient_steps': 1, 'policy_delay': 2,
            'target_policy_noise': 0.2, 'target_noise_clip': 0.5, 'policy': 'MlpPolicy',
            # Increase TD3 actor-critic sizes from [400,300] to [512,512,256]
            'policy_kwargs': {'net_arch': [512, 512, 256]}
        }
        final_config = {**default_config, **config.get('td3', {})}

        super().__init__(
            env=env,
            config=final_config,
            model_class=TD3,
            algorithm_name='TD3'
        )


class RLAlgorithmFactory:
    """Factory for creating RL algorithm instances by name."""
    _algorithms = {
        'ppo': PPOAlgorithm,
        'sac': SACAlgorithm,
        'td3': TD3Algorithm
    }

    @classmethod
    def create_algorithm(cls, algorithm_name: str, env: gym.Env, config: Dict[str, Any]) -> RLAlgorithmBase:
        name = algorithm_name.lower()
        if name not in cls._algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {list(cls._algorithms.keys())}")
        return cls._algorithms[name](env, config)

    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        return list(cls._algorithms.keys())


class TrainingManager:
    """Manages the training process, including callbacks and logging."""
    def __init__(self,
                 algorithm: RLAlgorithmBase,
                 log_dir: str = "./logs",
                 save_dir: str = "./models",
                 eval_env: Optional[gym.Env] = None):
        self.algorithm = algorithm
        self.log_dir = Path(log_dir)
        self.save_dir = Path(save_dir)
        self.eval_env = eval_env
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def setup_callbacks(self,
                        checkpoint_freq: int,
                        eval_freq: int,
                        reward_threshold: Optional[float],
                        save_best_only: bool) -> CallbackList:
        callbacks = []

        # Checkpoint callback to save model periodically
        if not save_best_only:
            callbacks.append(CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(self.save_dir),
                name_prefix='model_checkpoint'
            ))

        # Evaluation callback to save the best model based on performance
        if self.eval_env is not None:
            callbacks.append(EvalCallback(
                self.eval_env,
                best_model_save_path=str(self.save_dir),
                log_path=str(self.log_dir),
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
                verbose=1
            ))

        # Callback to stop training once a reward threshold is met
        if reward_threshold is not None:
            callbacks.append(StopTrainingOnRewardThreshold(
                reward_threshold=reward_threshold,
                verbose=1
            ))

        return CallbackList(callbacks)

    def train(self,
              total_timesteps: int,
              checkpoint_freq: int = 10000,
              eval_freq: int = 5000,
              reward_threshold: Optional[float] = None,
              save_best_only: bool = True) -> Dict[str, Any]:
        """
        Main training function.
        Note: The environment should be wrapped with `stable_baselines3.common.monitor.Monitor`
        for proper logging of episode rewards and lengths.
        """
        # Set up the logger for SB3 (stdout, csv, tensorboard)
        if self.algorithm.model is None:
            self.algorithm.create_model()
        self.algorithm.model.set_logger(configure(str(self.log_dir), ["stdout", "csv", "tensorboard"]))

        callbacks = self.setup_callbacks(checkpoint_freq, eval_freq, reward_threshold, save_best_only)
        info = self.algorithm.train(total_timesteps=total_timesteps, callback=callbacks)

        # Always save the final model for completeness
        final_model_path = self.save_dir / "final_model"
        self.algorithm.save(str(final_model_path))
        return info

# --- Helper Functions and Public API ---
def create_algorithm(algorithm_name: str, env: gym.Env, config: Dict[str, Any] = None) -> RLAlgorithmBase:
    """Convenience function to create an algorithm instance."""
    return RLAlgorithmFactory.create_algorithm(algorithm_name, env, config or {})

 
class RewardFunctionManager:
    """Manages different reward function configurations.

    Backwards-compatible manager returning simple dicts describing reward weights
    and thresholds. This mirrors the legacy API so other modules can import
    `RewardFunctionManager` from this module.
    """
    def __init__(self):
        self.reward_functions: Dict[str, Dict[str, Any]] = {}
        self._register_default_functions()

    def _register_default_functions(self):
        # Conservative reward (prioritizes safety)
        self.reward_functions['conservative'] = {
            'weights': {
                'collision': 5.0,
                'fuel': 0.5,
                'orbit': 2.0,
                'mission': 1.0
            },
            'collision': {
                'penalty': -2000.0,
                'threshold': 0.05,
                'small_reward': 2.0
            },
            'fuel': {'efficiency_weight': 0.05}
        }

        # Aggressive reward (prioritizes mission completion)
        self.reward_functions['aggressive'] = {
            'weights': {
                'collision': 2.0,
                'fuel': 2.0,
                'orbit': 0.5,
                'mission': 3.0
            },
            'collision': {
                'penalty': -500.0,
                'threshold': 0.2,
                'small_reward': 0.5
            },
            'fuel': {'efficiency_weight': 0.2}
        }

        # Balanced reward (default)
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
            'fuel': {'efficiency_weight': 0.1},
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
        if name not in self.reward_functions:
            available = ', '.join(self.reward_functions.keys())
            raise ValueError(f"Unknown reward function '{name}'. Available: {available}")
        # Return a shallow copy to avoid accidental external mutation
        return dict(self.reward_functions[name])

    def register_reward_function(self, name: str, config: Dict[str, Any]) -> None:
        self.reward_functions[name] = config

    def get_available_reward_functions(self) -> List[str]:
        return list(self.reward_functions.keys())

__all__ = [
    'RLAlgorithmBase', 'PPOAlgorithm', 'SACAlgorithm', 'TD3Algorithm',
    'RLAlgorithmFactory', 'TrainingManager', 'create_algorithm', 'RewardFunctionManager'
]