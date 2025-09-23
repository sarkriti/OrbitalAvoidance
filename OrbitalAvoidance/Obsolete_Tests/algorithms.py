"""
Research-Grade Agent Framework for Orbital Debris Avoidance

Supports multiple agent architectures:
- Classical control methods
- Reinforcement Learning (SB3, custom)
- Large Language Models (GPT, Claude, etc.)
- Neural networks (LSTM, Transformers)
- Hybrid approaches

All agents implement the standard Policy interface for seamless comparison.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
import numpy as np
import logging
import json


@dataclass
class AgentMetrics:
    """Container for agent performance metrics."""
    
    total_reward: float = 0.0
    total_delta_v: float = 0.0
    min_distance_achieved: float = float('inf')
    collision_count: int = 0
    warning_count: int = 0
    steps_taken: int = 0
    
    def reset(self):
        """Reset all metrics."""
        self.total_reward = 0.0
        self.total_delta_v = 0.0
        self.min_distance_achieved = float('inf')
        self.collision_count = 0
        self.warning_count = 0
        self.steps_taken = 0
    
    def update(self, reward: float, delta_v: float, min_distance: float, 
               collision: bool, warning: bool):
        """Update metrics with step data."""
        self.total_reward += reward
        self.total_delta_v += delta_v
        self.min_distance_achieved = min(self.min_distance_achieved, min_distance)
        if collision:
            self.collision_count += 1
        if warning:
            self.warning_count += 1
        self.steps_taken += 1


class Policy(ABC):
    """Base class for all avoidance policies.
    
    This abstract base class defines the standard interface for all agents.
    All implementations must provide:
    - act(obs, info) -> action
    - reset() for episode initialization
    - get_name() for identification
    """
    
    def __init__(self, name: str = "Unknown Policy"):
        self.name = name
        self.metrics = AgentMetrics()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Compute action for given observation and info.
        
        Args:
            observation: Environment observation (satellite state + K-nearest debris)
            info: Additional information dict containing:
                - min_distance_km: Minimum distance to any debris
                - collision: Boolean collision flag
                - warning: Boolean warning flag
                - total_dv_m_s: Cumulative delta-v used
                
        Returns:
            action: RTN delta-v command in m/s, shape (3,)
        """
        pass
    
    def reset(self):
        """Reset agent for new episode."""
        self.metrics.reset()
    
    def get_name(self) -> str:
        """Get agent name for identification."""
        return self.name
    
    def get_metrics(self) -> AgentMetrics:
        """Get current performance metrics."""
        return self.metrics
    
    def update_metrics(self, reward: float, info: Dict[str, Any]):
        """Update performance metrics after each step."""
        self.metrics.update(
            reward=reward,
            delta_v=info.get('step_dv_m_s', 0.0),
            min_distance=info.get('min_distance_km', float('inf')),
            collision=info.get('collision', False),
            warning=info.get('warning', False)
        )


# =====================================================================================
# CLASSICAL CONTROL AGENTS
# =====================================================================================

class ZeroPolicy(Policy):
    """No-action baseline policy."""
    
    def __init__(self):
        super().__init__("Zero Policy")
    
    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)


class RandomPolicy(Policy):
    """Random action policy for baseline comparison."""
    
    def __init__(self, action_scale: float = 0.1, seed: Optional[int] = None):
        super().__init__("Random Policy")
        self.action_scale = action_scale
        self.rng = np.random.default_rng(seed)
    
    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        # Random actions scaled to reasonable range
        return self.rng.normal(0, self.action_scale, size=3).astype(np.float32)


class ReactiveKeepoutPolicy(Policy):
    """Reactive rule-based policy that maintains safe distance."""
    
    def __init__(self, warning_distance_km: float = 5.0, critical_distance_km: float = 1.0,
                 gain_warning: float = 0.2, gain_critical: float = 0.5):
        super().__init__("Reactive Keepout")
        self.warning_distance = warning_distance_km
        self.critical_distance = critical_distance_km
        self.gain_warning = gain_warning
        self.gain_critical = gain_critical
    
    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        min_distance = info.get('min_distance_km', float('inf'))
        
        if min_distance < self.critical_distance:
            # Emergency avoidance: strong radial and along-track maneuver
            gain = self.gain_critical
        elif min_distance < self.warning_distance:
            # Preventive avoidance: moderate maneuver
            gain = self.gain_warning
        else:
            # Safe distance: no action
            return np.zeros(3, dtype=np.float32)
        
        # Simple radial + along-track boost
        # In practice, this should use relative geometry to the nearest debris
        return np.array([gain, gain * 0.5, 0.0], dtype=np.float32)


class ProportionalNavigationPolicy(Policy):
    """Proportional navigation guidance for collision avoidance."""
    
    def __init__(self, navigation_constant: float = 3.0, engagement_distance_km: float = 10.0):
        super().__init__("Proportional Navigation")
        self.nav_constant = navigation_constant
        self.engagement_distance = engagement_distance_km
    
    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        min_distance = info.get('min_distance_km', float('inf'))
        
        if min_distance > self.engagement_distance:
            return np.zeros(3, dtype=np.float32)
        
        # Simplified PN implementation
        # In reality, this needs line-of-sight rate and relative velocity
        los_rate = 0.1  # Placeholder - would compute from debris geometry
        relative_speed = 7.5  # km/s orbital speed approximation
        
        # PN command (perpendicular to LOS)
        acceleration = self.nav_constant * relative_speed * los_rate
        
        # Convert to RTN frame (simplified)
        return np.array([0.0, acceleration * 1000, 0.0], dtype=np.float32)  # m/s


# =====================================================================================
# REINFORCEMENT LEARNING AGENTS
# =====================================================================================

class SB3Agent(Policy):
    """Stable-Baselines3 reinforcement learning agent wrapper."""
    
    def __init__(self, model_path: str, algorithm: str = "PPO"):
        super().__init__(f"SB3 {algorithm}")
        self.model_path = model_path
        self.algorithm = algorithm
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained SB3 model."""
        try:
            from stable_baselines3 import PPO, SAC, TD3, A2C
            
            model_classes = {
                "PPO": PPO,
                "SAC": SAC,
                "TD3": TD3,
                "A2C": A2C
            }
            
            if self.algorithm in model_classes:
                self.model = model_classes[self.algorithm].load(self.model_path)
                self.logger.info(f"Loaded {self.algorithm} model from {self.model_path}")
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")
                
        except ImportError:
            self.logger.error("Stable-Baselines3 not installed. Install with: pip install stable-baselines3")
            self.model = None
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        if self.model is None:
            self.logger.warning("Model not loaded, returning zero action")
            return np.zeros(3, dtype=np.float32)
        
        action, _ = self.model.predict(observation, deterministic=True)
        return action.astype(np.float32)


class CustomRLAgent(Policy):
    """Wrapper for custom reinforcement learning models."""
    
    def __init__(self, model, name: str = "Custom RL Agent"):
        super().__init__(name)
        self.model = model
    
    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        # Assume model has a predict or forward method
        if hasattr(self.model, 'predict'):
            action = self.model.predict(observation)
        elif hasattr(self.model, 'forward'):
            import torch
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                action = self.model.forward(obs_tensor).squeeze(0).numpy()
        else:
            self.logger.error("Model does not have predict or forward method")
            return np.zeros(3, dtype=np.float32)
        
        return action.astype(np.float32)


# =====================================================================================
# LARGE LANGUAGE MODEL AGENTS
# =====================================================================================

class LLMAgent(Policy):
    """Large Language Model agent for orbital debris avoidance."""
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        super().__init__(f"LLM {model_name}")
        self.model_name = model_name
        self.api_key = api_key
        self.conversation_history = []
        self.step_count = 0
        
        # Initialize API client
        self._init_client()
    
    def _init_client(self):
        """Initialize the LLM API client."""
        try:
            if "gpt" in self.model_name.lower():
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            elif "claude" in self.model_name.lower():
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            else:
                self.logger.warning(f"Unsupported model: {self.model_name}")
                self.client = None
        except ImportError as e:
            self.logger.error(f"Required package not installed: {e}")
            self.client = None
    
    def _create_prompt(self, observation: np.ndarray, info: Dict[str, Any]) -> str:
        """Create prompt for the LLM based on current state."""
        # Extract key information from observation
        # obs format: [altitude, speed, dx1, dy1, dz1, dvx1, dvy1, dvz1, ...]
        altitude = observation[0]
        speed = observation[1]
        
        # Get debris information (K-nearest)
        k_nearest = len(observation[2:]) // 6
        debris_info = []
        for i in range(k_nearest):
            start_idx = 2 + i * 6
            dx, dy, dz = observation[start_idx:start_idx+3]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            debris_info.append(f"Debris {i+1}: {distance:.2f} km away")
        
        prompt = f"""You are controlling a satellite in orbit to avoid debris collisions.

Current situation:
- Satellite altitude: {altitude:.1f} km
- Satellite speed: {speed:.3f} km/s  
- Minimum distance to debris: {info.get('min_distance_km', 'unknown'):.3f} km
- Total delta-v used: {info.get('total_dv_m_s', 0):.2f} m/s

Nearby debris:
{chr(10).join(debris_info)}

You must output a delta-v command in RTN coordinates (Radial, Tangential, Normal) in m/s.
- Radial: positive = away from Earth
- Tangential: positive = in direction of orbital motion  
- Normal: positive = perpendicular to orbital plane

Constraints:
- Maximum delta-v per step: 0.5 m/s per axis
- Collision occurs if distance < 0.2 km
- Warning zone is < 5 km

Output format: [radial_dv, tangential_dv, normal_dv]
Example: [0.1, -0.2, 0.0]

Your command:"""
        
        return prompt
    
    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        if self.client is None:
            self.logger.warning("LLM client not initialized, using fallback")
            return np.zeros(3, dtype=np.float32)
        
        # Limit LLM calls for performance (every N steps)
        if self.step_count % 5 != 0:
            return np.zeros(3, dtype=np.float32)
        
        self.step_count += 1
        
        try:
            prompt = self._create_prompt(observation, info)
            
            if "gpt" in self.model_name.lower():
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.1
                )
                response_text = response.choices[0].message.content
            else:
                # Handle other models...
                response_text = "[0.0, 0.0, 0.0]"
            
            # Parse response
            action = self._parse_response(response_text)
            return action
            
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            return np.zeros(3, dtype=np.float32)
    
    def _parse_response(self, response: str) -> np.ndarray:
        """Parse LLM response to extract action."""
        try:
            # Look for array-like pattern [x, y, z]
            import re
            pattern = r'\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]'
            match = re.search(pattern, response)
            
            if match:
                action = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
                # Clip to reasonable range
                action = np.clip(action, -0.5, 0.5)
                return action.astype(np.float32)
            else:
                self.logger.warning(f"Could not parse LLM response: {response}")
                return np.zeros(3, dtype=np.float32)
                
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return np.zeros(3, dtype=np.float32)


# =====================================================================================
# NEURAL NETWORK AGENTS
# =====================================================================================

class LSTMAgent(Policy):
    """LSTM-based agent for sequential decision making."""
    
    def __init__(self, model_path: str, sequence_length: int = 10):
        super().__init__("LSTM Agent")
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.observation_history = []
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the LSTM model."""
        try:
            import torch
            import torch.nn as nn
            
            # Define LSTM model architecture
            class LSTMPolicy(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, output_size):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    output = self.fc(lstm_out[:, -1, :])  # Use last timestep
                    return output
            
            # Load model (this would be customized based on your architecture)
            self.model = torch.load(self.model_path, map_location='cpu')
            self.model.eval()
            self.logger.info(f"Loaded LSTM model from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load LSTM model: {e}")
            self.model = None
    
    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        if self.model is None:
            return np.zeros(3, dtype=np.float32)
        
        # Maintain observation history
        self.observation_history.append(observation)
        if len(self.observation_history) > self.sequence_length:
            self.observation_history.pop(0)
        
        # Need full sequence for LSTM
        if len(self.observation_history) < self.sequence_length:
            return np.zeros(3, dtype=np.float32)
        
        try:
            import torch
            
            # Prepare sequence input
            sequence = np.array(self.observation_history)
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                action = self.model(sequence_tensor).squeeze(0).numpy()
            
            return action.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"LSTM inference failed: {e}")
            return np.zeros(3, dtype=np.float32)
    
    def reset(self):
        """Reset agent and clear history."""
        super().reset()
        self.observation_history.clear()


# =====================================================================================
# UTILITY FUNCTIONS
# =====================================================================================

def create_agent_from_config(config: Dict[str, Any]) -> Policy:
    """Factory function to create agents from configuration."""
    
    agent_type = config.get('type', 'zero')
    
    if agent_type == 'zero':
        return ZeroPolicy()
    
    elif agent_type == 'random':
        return RandomPolicy(
            action_scale=config.get('action_scale', 0.1),
            seed=config.get('seed', None)
        )
    
    elif agent_type == 'reactive':
        return ReactiveKeepoutPolicy(
            warning_distance_km=config.get('warning_distance', 5.0),
            critical_distance_km=config.get('critical_distance', 1.0),
            gain_warning=config.get('gain_warning', 0.2),
            gain_critical=config.get('gain_critical', 0.5)
        )
    
    elif agent_type == 'proportional_navigation':
        return ProportionalNavigationPolicy(
            navigation_constant=config.get('nav_constant', 3.0),
            engagement_distance_km=config.get('engagement_distance', 10.0)
        )
    
    elif agent_type == 'sb3':
        return SB3Agent(
            model_path=config['model_path'],
            algorithm=config.get('algorithm', 'PPO')
        )
    
    elif agent_type == 'llm':
        return LLMAgent(
            model_name=config.get('model_name', 'gpt-4'),
            api_key=config.get('api_key', None)
        )
    
    elif agent_type == 'lstm':
        return LSTMAgent(
            model_path=config['model_path'],
            sequence_length=config.get('sequence_length', 10)
        )
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def compare_agents(agents: List[Policy], env, num_episodes: int = 10) -> Dict[str, Dict[str, float]]:
    """Compare multiple agents over multiple episodes."""
    
    results = {}
    
    for agent in agents:
        agent_results = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            agent.reset()
            
            episode_reward = 0
            episode_steps = 0
            
            while True:
                action = agent.act(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                if terminated or truncated:
                    break
            
            agent_results.append({
                'reward': episode_reward,
                'steps': episode_steps,
                'final_distance': info.get('min_distance_km', 0),
                'total_dv': info.get('total_dv_m_s', 0)
            })
        
        # Calculate statistics
        rewards = [r['reward'] for r in agent_results]
        distances = [r['final_distance'] for r in agent_results]
        delta_vs = [r['total_dv'] for r in agent_results]
        
        results[agent.get_name()] = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_final_distance': np.mean(distances),
            'mean_delta_v': np.mean(delta_vs),
            'success_rate': sum(1 for d in distances if d > 0.2) / len(distances)
        }
    
    return results
