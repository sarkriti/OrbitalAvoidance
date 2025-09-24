"""RL algorithms (migrated package version)."""

from __future__ import annotations
import os, json, time, sys, logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List, Tuple
from pathlib import Path
import numpy as np
import gymnasium as gym

try:
	from stable_baselines3 import PPO, SAC, TD3
	from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
	from stable_baselines3.common.callbacks import (
		CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold, CallbackList
	)
	from stable_baselines3.common.logger import configure
	STABLE_BASELINES_AVAILABLE = True
except ImportError:  # pragma: no cover
	STABLE_BASELINES_AVAILABLE = False

try:
	from hardware_detection import get_optimal_config, configure_ml_frameworks
	HARDWARE_DETECTION_AVAILABLE = True
except ImportError:  # pragma: no cover
	HARDWARE_DETECTION_AVAILABLE = False

logger = logging.getLogger(__name__)


class RLAlgorithmBase(ABC):
	def __init__(self, env: gym.Env, config: Dict[str, Any]):
		self.env = env
		self.config = config
		self.model = None
		self.training_history = []
		self.hardware_config = None
		self._setup_hardware()

	def _setup_hardware(self):  # pragma: no cover
		if HARDWARE_DETECTION_AVAILABLE:
			try:
				self.hardware_config = get_optimal_config(
					prefer_gpu=self.config.get('prefer_gpu', True),
					memory_fraction=self.config.get('gpu_memory_fraction', 0.8),
					max_threads=self.config.get('max_threads')
				)
				configure_ml_frameworks(self.hardware_config)
				self.device = self.hardware_config.torch_device
			except Exception as e:
				logger.warning(f"Hardware config failed: {e}")
				self.device = 'cpu'
		else:
			self.device = 'cpu'

	@abstractmethod
	def create_model(self): ...
	@abstractmethod
	def train(self, total_timesteps: int, **kwargs): ...
	@abstractmethod
	def predict(self, observation: np.ndarray, deterministic: bool = True): ...
	@abstractmethod
	def save(self, path: str): ...
	@abstractmethod
	def load(self, path: str): ...


class PPOAlgorithm(RLAlgorithmBase):
	def __init__(self, env: gym.Env, config: Dict[str, Any]):
		super().__init__(env, config)
		self.default_config = {
			'learning_rate': 3e-4,'n_steps': 2048,'batch_size': 64,'n_epochs': 10,'gamma': 0.99,
			'gae_lambda': 0.95,'clip_range': 0.2,'ent_coef': 0.0,'vf_coef': 0.5,'max_grad_norm': 0.5,
			'use_sde': False,'policy': 'MlpPolicy','policy_kwargs': {'net_arch': dict(pi=[256,256], vf=[256,256])}
		}

	def create_model(self):
		if not STABLE_BASELINES_AVAILABLE:
			raise ImportError("stable-baselines3 required for PPO")
		cfg = {**self.default_config, **self.config.get('ppo', {})}
		kwargs = dict(policy=cfg['policy'], env=self.env, learning_rate=cfg['learning_rate'], n_steps=cfg['n_steps'],
					  batch_size=cfg['batch_size'], n_epochs=cfg['n_epochs'], gamma=cfg['gamma'], gae_lambda=cfg['gae_lambda'],
					  clip_range=cfg['clip_range'], ent_coef=cfg['ent_coef'], vf_coef=cfg['vf_coef'],
					  max_grad_norm=cfg['max_grad_norm'], use_sde=cfg['use_sde'], policy_kwargs=cfg['policy_kwargs'], verbose=1)
		if getattr(self, 'hardware_config', None) and self.hardware_config.use_gpu:
			kwargs['device'] = self.device
		from stable_baselines3 import PPO
		self.model = PPO(**kwargs)
		return self.model

	def train(self, total_timesteps: int, **kwargs):
		if self.model is None:
			self.create_model()
		start = time.time(); self.model.learn(total_timesteps=total_timesteps, **kwargs)
		info = {'algorithm':'PPO','total_timesteps':total_timesteps,'training_time':time.time()-start,
				'final_reward': list(getattr(self.model,'ep_info_buffer', []))}
		self.training_history.append(info); return info

	def predict(self, observation: np.ndarray, deterministic: bool = True):
		if self.model is None:
			raise ValueError("Model not trained")
		return self.model.predict(observation, deterministic=deterministic)

	def save(self, path: str):
		if self.model:
			self.model.save(path)
			with open(Path(path).with_name(f"{Path(path).stem}_history.json"), 'w') as f:
				json.dump(self.training_history, f, indent=2)

	def load(self, path: str):
		from stable_baselines3 import PPO
		self.model = PPO.load(path, env=self.env)


class SACAlgorithm(RLAlgorithmBase):
	def __init__(self, env: gym.Env, config: Dict[str, Any]):
		super().__init__(env, config)
		self.default_config = {
			'learning_rate':3e-4,'buffer_size':1000000,'learning_starts':100,'batch_size':256,'tau':0.005,
			'gamma':0.99,'train_freq':1,'gradient_steps':1,'ent_coef':'auto','target_update_interval':1,
			'target_entropy':'auto','use_sde':False,'policy':'MlpPolicy','policy_kwargs':{'net_arch':[256,256]}
		}

	def create_model(self):
		if not STABLE_BASELINES_AVAILABLE:
			raise ImportError("stable-baselines3 required for SAC")
		cfg = {**self.default_config, **self.config.get('sac', {})}
		from stable_baselines3 import SAC
		self.model = SAC(policy=cfg['policy'], env=self.env, learning_rate=cfg['learning_rate'], buffer_size=cfg['buffer_size'],
						 learning_starts=cfg['learning_starts'], batch_size=cfg['batch_size'], tau=cfg['tau'], gamma=cfg['gamma'],
						 train_freq=cfg['train_freq'], gradient_steps=cfg['gradient_steps'], ent_coef=cfg['ent_coef'],
						 target_update_interval=cfg['target_update_interval'], target_entropy=cfg['target_entropy'],
						 use_sde=cfg['use_sde'], policy_kwargs=cfg['policy_kwargs'], verbose=1)
		return self.model

	def train(self, total_timesteps: int, **kwargs):
		if self.model is None:
			self.create_model()
		start=time.time(); self.model.learn(total_timesteps=total_timesteps, **kwargs)
		info={'algorithm':'SAC','total_timesteps':total_timesteps,'training_time':time.time()-start,'final_reward':list(getattr(self.model,'ep_info_buffer', []))}
		self.training_history.append(info); return info

	def predict(self, observation: np.ndarray, deterministic: bool = True):
		if self.model is None: raise ValueError("Model not trained")
		return self.model.predict(observation, deterministic=deterministic)

	def save(self, path: str):
		if self.model:
			self.model.save(path)
			with open(Path(path).with_name(f"{Path(path).stem}_history.json"), 'w') as f:
				json.dump(self.training_history, f, indent=2)

	def load(self, path: str):
		from stable_baselines3 import SAC
		self.model = SAC.load(path, env=self.env)


class TD3Algorithm(RLAlgorithmBase):
	def __init__(self, env: gym.Env, config: Dict[str, Any]):
		super().__init__(env, config)
		self.default_config = {
			'learning_rate':1e-3,'buffer_size':1000000,'learning_starts':100,'batch_size':100,'tau':0.005,'gamma':0.99,
			'train_freq':1,'gradient_steps':1,'policy_delay':2,'target_policy_noise':0.2,'target_noise_clip':0.5,'policy':'MlpPolicy',
			'policy_kwargs':{'net_arch':[400,300]}
		}

	def create_model(self):
		if not STABLE_BASELINES_AVAILABLE: raise ImportError("stable-baselines3 required for TD3")
		cfg={**self.default_config, **self.config.get('td3', {})}
		from stable_baselines3 import TD3
		self.model = TD3(policy=cfg['policy'], env=self.env, learning_rate=cfg['learning_rate'], buffer_size=cfg['buffer_size'],
						 learning_starts=cfg['learning_starts'], batch_size=cfg['batch_size'], tau=cfg['tau'], gamma=cfg['gamma'],
						 train_freq=cfg['train_freq'], gradient_steps=cfg['gradient_steps'], policy_delay=cfg['policy_delay'],
						 target_policy_noise=cfg['target_policy_noise'], target_noise_clip=cfg['target_noise_clip'],
						 policy_kwargs=cfg['policy_kwargs'], verbose=1)
		return self.model

	def train(self, total_timesteps: int, **kwargs):
		if self.model is None: self.create_model()
		start=time.time(); self.model.learn(total_timesteps=total_timesteps, **kwargs)
		info={'algorithm':'TD3','total_timesteps':total_timesteps,'training_time':time.time()-start,'final_reward':list(getattr(self.model,'ep_info_buffer', []))}
		self.training_history.append(info); return info

	def predict(self, observation: np.ndarray, deterministic: bool = True):
		if self.model is None: raise ValueError("Model not trained")
		return self.model.predict(observation, deterministic=deterministic)

	def save(self, path: str):
		if self.model:
			self.model.save(path)
			with open(Path(path).with_name(f"{Path(path).stem}_history.json"), 'w') as f:
				json.dump(self.training_history, f, indent=2)

	def load(self, path: str):
		from stable_baselines3 import TD3
		self.model = TD3.load(path, env=self.env)


class RLAlgorithmFactory:
	_algorithms = {'ppo': PPOAlgorithm, 'sac': SACAlgorithm, 'td3': TD3Algorithm}

	@classmethod
	def create_algorithm(cls, algorithm_name: str, env: gym.Env, config: Dict[str, Any]):
		name = algorithm_name.lower()
		if name not in cls._algorithms:
			raise ValueError(f"Unknown algorithm {algorithm_name}")
		return cls._algorithms[name](env, config)

	@classmethod
	def get_available_algorithms(cls) -> List[str]:
		return list(cls._algorithms.keys())


class TrainingManager:
	def __init__(self, algorithm: RLAlgorithmBase, log_dir: str = "./logs", save_dir: str = "./models", eval_env: Optional[gym.Env] = None):
		self.algorithm = algorithm
		self.log_dir = Path(log_dir); self.save_dir = Path(save_dir); self.eval_env = eval_env
		self.log_dir.mkdir(parents=True, exist_ok=True); self.save_dir.mkdir(parents=True, exist_ok=True)

	def setup_callbacks(self, checkpoint_freq: int = 10000, eval_freq: int = 5000, reward_threshold: Optional[float] = None, save_best_only: bool = True):
		from stable_baselines3.common.callbacks import BaseCallback
		
		class EpisodeRewardLogger(BaseCallback):
			def __init__(self, verbose=0):
				super().__init__(verbose)
				self.episode_rewards = []
				self.episode_lengths = []
			
			def _on_step(self) -> bool:
				# Check if episode ended
				if len(self.locals.get('dones', [])) > 0 and any(self.locals['dones']):
					# Get episode info from Monitor wrapper
					if 'infos' in self.locals:
						for info in self.locals['infos']:
							if 'episode' in info:
								ep_rew = info['episode']['r']
								ep_len = info['episode']['l']
								self.episode_rewards.append(ep_rew)
								self.episode_lengths.append(ep_len)
								print(f"Episode completed - Reward: {ep_rew:.2f}, Length: {ep_len}")
								
								# Log recent episode statistics 
								if len(self.episode_rewards) >= 5:
									recent_mean = sum(self.episode_rewards[-5:]) / 5
									print(f"Recent 5 episodes mean reward: {recent_mean:.2f}")
				return True
		
		callbacks = []
		
		# Add episode reward logger
		callbacks.append(EpisodeRewardLogger(verbose=1))
		
		if not save_best_only:
			callbacks.append(CheckpointCallback(save_freq=checkpoint_freq, save_path=str(self.save_dir), name_prefix='model_checkpoint'))
		if self.eval_env is not None:
			callbacks.append(EvalCallback(self.eval_env, best_model_save_path=str(self.save_dir), log_path=str(self.log_dir), eval_freq=eval_freq, deterministic=True, render=False, verbose=1))
		if reward_threshold is not None:
			callbacks.append(StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1))
		from stable_baselines3.common.callbacks import CallbackList
		return CallbackList(callbacks)

	def train(self, total_timesteps: int, checkpoint_freq: int = 10000, eval_freq: int = 5000, reward_threshold: Optional[float] = None, save_best_only: bool = True):
		if hasattr(self.algorithm.model, 'set_logger'):
			self.algorithm.model.set_logger(configure(str(self.log_dir), ["stdout","csv","tensorboard"]))
		callbacks = self.setup_callbacks(checkpoint_freq, eval_freq, reward_threshold, save_best_only)
		info = self.algorithm.train(total_timesteps=total_timesteps, callback=callbacks)
		if not save_best_only or self.eval_env is None:
			self.algorithm.save(str(self.save_dir / "final_model"))
		return info


class RewardFunctionManager:
	def __init__(self):
		self.reward_functions = {}
		self.reward_functions['balanced'] = {
			'weights': {'collision':1.0,'fuel':1.0,'orbit':1.0,'mission':1.0},
			'collision': {'penalty':-1000.0,'threshold':0.1,'small_reward':1.0},
			'fuel': {'efficiency_weight':0.1},
			'orbit': {'tolerances': {'semi_major_axis_m':100.0,'eccentricity':0.01,'inclination_rad':0.01,'raan_rad':0.01,'arg_periapsis_rad':0.01}},
			'mission': {'survival_reward_interval':500,'survival_multiplier':3.0,'mission_weight':1.0}
		}

	def get_reward_config(self, name: str) -> Dict[str, Any]:
		if name not in self.reward_functions:
			raise ValueError(name)
		return self.reward_functions[name].copy()


def create_algorithm(algorithm_name: str, env: gym.Env, config: Dict[str, Any] = None):
	return RLAlgorithmFactory.create_algorithm(algorithm_name, env, config or {})


def get_reward_config(reward_name: str = 'balanced'):
	return RewardFunctionManager().get_reward_config(reward_name)


__all__ = [
	'RLAlgorithmBase','PPOAlgorithm','SACAlgorithm','TD3Algorithm','RLAlgorithmFactory','TrainingManager','RewardFunctionManager','create_algorithm','get_reward_config'
]
