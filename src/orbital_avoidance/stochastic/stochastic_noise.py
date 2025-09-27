#!/usr/bin/env python3
"""
stochastic_noise.py

Configurable stochastic noise injection for orbital debris avoidance simulations.
Provides realistic uncertainty modeling for observations, dynamics, and control.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StochasticNoiseManager:
    """Manages all stochastic noise sources in the simulation."""
    
    def __init__(self, stochasticity_config: Dict[str, Any]):
        self.config = stochasticity_config
        self.enabled = stochasticity_config.get('enabled', False)
        
        if not self.enabled:
            logger.info("Stochasticity disabled")
            return
            
        # Parse noise configurations
        self.observation_noise = self._parse_noise_config(
            stochasticity_config.get('observation_noise', {})
        )
        self.dynamics_noise = self._parse_noise_config(
            stochasticity_config.get('dynamics_noise', {})
        )
        self.control_noise = self._parse_noise_config(
            stochasticity_config.get('control_noise', {})
        )
        self.initial_state_noise = self._parse_noise_config(
            stochasticity_config.get('initial_state_noise', {})
        )
        
        # Environmental uncertainties
        self.environmental = stochasticity_config.get('environmental', {})
        
        logger.info("StochasticNoiseManager initialized")
    
    def _parse_noise_config(self, noise_config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate noise configuration."""
        if not noise_config.get('enabled', False):
            return {'enabled': False}
            
        return {
            'enabled': True,
            'type': noise_config.get('type', 'gaussian'),
            'std_dev': noise_config.get('std_dev', 0.01),
            'bias': noise_config.get('bias', 0.0),
            'correlation_time': noise_config.get('correlation_time', None),
            'scaling_factors': noise_config.get('scaling_factors', {})
        }
    
    def apply_observation_noise(self, observation: np.ndarray) -> np.ndarray:
        """Apply noise to observations (sensor uncertainty)."""
        if not self.enabled or not self.observation_noise['enabled']:
            return observation
            
        noise_config = self.observation_noise
        
        if noise_config['type'] == 'gaussian':
            noise = np.random.normal(0, noise_config['std_dev'], observation.shape)
        elif noise_config['type'] == 'uniform':
            bound = noise_config['std_dev'] * np.sqrt(3)  # Same variance as gaussian
            noise = np.random.uniform(-bound, bound, observation.shape)
        else:
            noise = np.zeros_like(observation)
            
        # Apply bias
        bias = noise_config.get('bias', 0.0)
        if bias != 0.0:
            noise += bias
            
        # Scale noise for different observation components
        scaling = noise_config.get('scaling_factors', {})
        if scaling:
            noise = self._apply_component_scaling(noise, scaling, 'observation')
            
        return observation + noise
    
    def apply_dynamics_noise(self, position: np.ndarray, velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply noise to dynamics propagation (process uncertainty)."""
        if not self.enabled or not self.dynamics_noise['enabled']:
            return position, velocity
            
        noise_config = self.dynamics_noise
        
        # Position noise (measurement/GPS uncertainty)
        pos_noise = self._generate_noise(noise_config, position.shape)
        
        # Velocity noise (smaller than position)
        vel_std = noise_config['std_dev'] * 0.1  # Typically much smaller
        vel_noise = np.random.normal(0, vel_std, velocity.shape)
        
        # Apply scaling
        scaling = noise_config.get('scaling_factors', {})
        if scaling:
            pos_noise = self._apply_component_scaling(pos_noise, scaling, 'position')
            vel_noise = self._apply_component_scaling(vel_noise, scaling, 'velocity')
            
        return position + pos_noise, velocity + vel_noise
    
    def apply_control_noise(self, action: np.ndarray) -> np.ndarray:
        """Apply noise to control commands (actuator uncertainty)."""
        if not self.enabled or not self.control_noise['enabled']:
            return action
            
        noise_config = self.control_noise
        noise = self._generate_noise(noise_config, action.shape)
        
        # Control noise is typically multiplicative for thrusters
        if noise_config.get('multiplicative', True):
            return action * (1.0 + noise)
        else:
            return action + noise
    
    def apply_initial_state_noise(self, position: np.ndarray, velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply noise to initial orbital states."""
        if not self.enabled or not self.initial_state_noise['enabled']:
            return position, velocity
            
        noise_config = self.initial_state_noise
        
        # Orbital element uncertainties are more realistic
        pos_noise = self._generate_noise(noise_config, position.shape)
        vel_noise = self._generate_noise(noise_config, velocity.shape) * 0.01  # Much smaller for velocity
        
        scaling = noise_config.get('scaling_factors', {})
        if scaling:
            pos_noise = self._apply_component_scaling(pos_noise, scaling, 'position')
            vel_noise = self._apply_component_scaling(vel_noise, scaling, 'velocity')
            
        return position + pos_noise, velocity + vel_noise
    
    def get_environmental_perturbation(self, component: str) -> float:
        """Get environmental uncertainty factors (atmospheric density, solar flux, etc.)."""
        if not self.enabled:
            return 1.0
            
        env_config = self.environmental.get(component, {})
        if not env_config.get('enabled', False):
            return 1.0
            
        uncertainty = env_config.get('uncertainty_fraction', 0.1)
        return np.random.normal(1.0, uncertainty)
    
    def _generate_noise(self, noise_config: Dict[str, Any], shape: Tuple) -> np.ndarray:
        """Generate noise based on configuration."""
        noise_type = noise_config.get('type', 'gaussian')
        std_dev = noise_config['std_dev']
        bias = noise_config.get('bias', 0.0)
        
        if noise_type == 'gaussian':
            noise = np.random.normal(bias, std_dev, shape)
        elif noise_type == 'uniform':
            bound = std_dev * np.sqrt(3)
            noise = np.random.uniform(bias - bound, bias + bound, shape)
        elif noise_type == 'laplace':
            # Heavy-tailed noise for rare but large errors
            scale = std_dev / np.sqrt(2)
            noise = np.random.laplace(bias, scale, shape)
        else:
            logger.warning(f"Unknown noise type: {noise_type}, using gaussian")
            noise = np.random.normal(bias, std_dev, shape)
            
        return noise
    
    def _apply_component_scaling(self, noise: np.ndarray, scaling: Dict[str, float], component_type: str) -> np.ndarray:
        """Apply different scaling factors to different components of the noise vector."""
        if component_type == 'observation':
            # Example: different noise for position vs velocity vs fuel
            # This would depend on your specific observation structure
            if 'position' in scaling and len(noise) >= 3:
                noise[:3] *= scaling['position']
            if 'velocity' in scaling and len(noise) >= 6:
                noise[3:6] *= scaling['velocity']
            if 'fuel' in scaling and len(noise) >= 7:
                noise[6] *= scaling['fuel']
                
        elif component_type in ['position', 'velocity']:
            # Apply uniform scaling to all components
            scale_factor = scaling.get(component_type, 1.0)
            noise *= scale_factor
            
        return noise
    
    def reset_correlated_noise(self):
        """Reset any correlated noise states (for episode resets)."""
        # If you implement correlated noise, reset state here
        pass


def create_default_stochasticity_config() -> Dict[str, Any]:
    """Create a default stochasticity configuration."""
    return {
        "enabled": True,
        "observation_noise": {
            "enabled": True,
            "type": "gaussian",
            "std_dev": 0.001,
            "bias": 0.0,
            "scaling_factors": {
                "position": 1.0,
                "velocity": 0.5,
                "fuel": 0.01
            }
        },
        "dynamics_noise": {
            "enabled": True,
            "type": "gaussian", 
            "std_dev": 0.01,
            "scaling_factors": {
                "position": 1.0,
                "velocity": 0.1
            }
        },
        "control_noise": {
            "enabled": True,
            "type": "gaussian",
            "std_dev": 0.05,
            "multiplicative": True
        },
        "initial_state_noise": {
            "enabled": True,
            "type": "gaussian",
            "std_dev": 0.1,
            "scaling_factors": {
                "position": 1.0,
                "velocity": 0.01
            }
        },
        "environmental": {
            "atmospheric_density": {
                "enabled": True,
                "uncertainty_fraction": 0.15
            },
            "solar_flux": {
                "enabled": True,
                "uncertainty_fraction": 0.1
            }
        }
    }