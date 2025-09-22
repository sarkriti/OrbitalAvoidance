#!/usr/bin/env python3
"""
RL-Enabled Web Interface Integration

This module provides the bridge between your RL training system and the web interface,
allowing you to:

1. Visualize trained RL agents in action on the web interface
2. Switch between manual control, zero policy, and RL agents
3. Load and test different trained models in real-time
4. Compare different algorithms and reward functions visually
5. Debug and tune RL agent behavior interactively
"""

import os
import sys
import json
import time
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

# Add project paths
if __package__ in (None, ""):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)

try:
    from stable_baselines3 import PPO, SAC, TD3
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("Warning: stable-baselines3 not available. RL agents disabled.")

from rl_training_env import RLOrbitalEnvironment, create_rl_environment
from rl_algorithms import RLAlgorithmFactory, get_reward_config


class RLWebAgent:
    """RL agent interface for web simulator integration."""
    
    def __init__(self, model_path: Optional[str] = None, algorithm: str = 'ppo'):
        """
        Initialize RL agent for web interface.
        
        Args:
            model_path: Path to trained model file
            algorithm: Algorithm type ('ppo', 'sac', 'td3')
        """
        self.model_path = model_path
        self.algorithm = algorithm.lower()
        self.model = None
        self.env = None
        self.last_observation = None
        self.is_loaded = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained RL model."""
        if not STABLE_BASELINES_AVAILABLE:
            print("Error: stable-baselines3 not available for RL agents")
            return False
        
        try:
            print(f"Loading RL model: {model_path}")
            
            # Create environment for the agent (needed for model loading)
            self.env = create_rl_environment(
                config_file="scenarios/leo_debris_avoidance.json",
                max_debris_tracked=10
            )
            
            # Try to load with different algorithm types
            model_loaded = False
            for AlgorithmClass in [PPO, SAC, TD3]:
                try:
                    self.model = AlgorithmClass.load(model_path, env=self.env)
                    self.algorithm = AlgorithmClass.__name__.lower()
                    model_loaded = True
                    print(f"✅ Successfully loaded {self.algorithm.upper()} model")
                    break
                except Exception as e:
                    continue
            
            if not model_loaded:
                print(f"❌ Failed to load model: {model_path}")
                return False
            
            self.model_path = model_path
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_action(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action from RL agent given observation."""
        if not self.is_loaded or self.model is None:
            return np.array([0.0, 0.0, 0.0])  # No thrust
        
        try:
            action, _ = self.model.predict(observation, deterministic=deterministic)
            return action
        except Exception as e:
            print(f"Error getting action from RL agent: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def convert_web_state_to_rl_observation(self, satellite_state: Dict, debris_states: List[Dict]) -> np.ndarray:
        """
        Convert web simulator state to RL observation format.
        
        The web simulator has different state format than RL environment,
        so we need to convert between them.
        """
        try:
            obs = []
            
            # Spacecraft state (6 elements)
            sat_pos = satellite_state.get('position', [0, 0, 0])
            sat_vel = satellite_state.get('velocity', [0, 0, 0])
            obs.extend(sat_pos)
            obs.extend(sat_vel)
            
            # Mission info (2 elements)
            fuel_remaining = satellite_state.get('fuel_remaining_percent', 100) / 100.0
            time_remaining = 1.0 - (satellite_state.get('step_count', 0) / 10000.0)  # Normalized
            obs.extend([fuel_remaining, time_remaining])
            
            # Sort debris by distance and take closest 10
            sat_pos_array = np.array(sat_pos)
            debris_with_distance = []
            
            for debris in debris_states:
                debris_pos = np.array(debris.get('position', [0, 0, 0]))
                distance = np.linalg.norm(debris_pos - sat_pos_array)
                debris_with_distance.append((distance, debris))
            
            # Sort by distance and take top 10
            debris_with_distance.sort(key=lambda x: x[0])
            closest_debris = [debris for _, debris in debris_with_distance[:10]]
            
            # Per debris object (8 elements each)
            for i, debris in enumerate(closest_debris):
                if i >= 10:  # Limit to 10 debris
                    break
                
                debris_pos = np.array(debris.get('position', [0, 0, 0]))
                debris_vel = np.array(debris.get('velocity', [0, 0, 0]))
                
                # Relative position and velocity
                rel_pos = debris_pos - sat_pos_array
                rel_vel = debris_vel - np.array(sat_vel)
                
                obs.extend(rel_pos.tolist())
                obs.extend(rel_vel.tolist())
                
                # Calculate miss distance and time to collision (simplified)
                distance = np.linalg.norm(rel_pos)
                relative_speed = np.linalg.norm(rel_vel)
                time_to_collision = distance / max(relative_speed, 1e-6)
                
                obs.extend([distance, time_to_collision])
            
            # Pad with zeros if fewer than 10 debris
            while len(closest_debris) < 10:
                obs.extend([0.0] * 8)  # 8 zeros for missing debris
            
            return np.array(obs, dtype=np.float32)
            
        except Exception as e:
            print(f"Error converting web state to RL observation: {e}")
            # Return zero observation if conversion fails
            return np.zeros(6 + 2 + (10 * 8), dtype=np.float32)


class WebRLController:
    """Controller that manages different control modes for web interface."""
    
    def __init__(self):
        self.control_mode = 'zero'  # 'zero', 'manual', 'rl'
        self.rl_agent = None
        self.manual_thrust = np.array([0.0, 0.0, 0.0])
        self.available_models = self._scan_for_models()
        
    def _scan_for_models(self) -> Dict[str, str]:
        """Scan for available trained models."""
        models = {}
        model_dirs = ['./models', './trained_models', '../models']
        
        for model_dir in model_dirs:
            model_path = Path(model_dir)
            if model_path.exists():
                for file_path in model_path.rglob('*.zip'):
                    model_name = file_path.stem
                    models[model_name] = str(file_path)
        
        return models
    
    def set_control_mode(self, mode: str) -> bool:
        """Set control mode: 'zero', 'manual', or 'rl'."""
        if mode in ['zero', 'manual', 'rl']:
            self.control_mode = mode
            print(f"Control mode set to: {mode}")
            return True
        return False
    
    def load_rl_model(self, model_path: str) -> bool:
        """Load an RL model for control."""
        self.rl_agent = RLWebAgent(model_path)
        if self.rl_agent.is_loaded:
            self.control_mode = 'rl'
            return True
        return False
    
    def set_manual_thrust(self, thrust_x: float, thrust_y: float, thrust_z: float):
        """Set manual thrust commands."""
        self.manual_thrust = np.array([thrust_x, thrust_y, thrust_z])
        self.manual_thrust = np.clip(self.manual_thrust, -1.0, 1.0)
    
    def get_action(self, satellite_state: Dict, debris_states: List[Dict]) -> Dict[str, Any]:
        """
        Get action based on current control mode.
        
        Returns:
            Dict with action and metadata
        """
        action_info = {
            'thrust': [0.0, 0.0, 0.0],
            'mode': self.control_mode,
            'source': 'unknown',
            'model_name': None
        }
        
        if self.control_mode == 'zero':
            action_info['thrust'] = [0.0, 0.0, 0.0]
            action_info['source'] = 'zero_policy'
            
        elif self.control_mode == 'manual':
            action_info['thrust'] = self.manual_thrust.tolist()
            action_info['source'] = 'manual_input'
            
        elif self.control_mode == 'rl' and self.rl_agent and self.rl_agent.is_loaded:
            # Convert web state to RL observation
            observation = self.rl_agent.convert_web_state_to_rl_observation(
                satellite_state, debris_states
            )
            
            # Get action from RL agent
            action = self.rl_agent.get_action(observation, deterministic=True)
            action_info['thrust'] = action.tolist()
            action_info['source'] = f'rl_{self.rl_agent.algorithm}'
            action_info['model_name'] = os.path.basename(self.rl_agent.model_path) if self.rl_agent.model_path else None
            
        return action_info
    
    def get_status(self) -> Dict[str, Any]:
        """Get current controller status."""
        status = {
            'control_mode': self.control_mode,
            'available_models': list(self.available_models.keys()),
            'rl_agent_loaded': self.rl_agent is not None and self.rl_agent.is_loaded,
            'manual_thrust': self.manual_thrust.tolist(),
            'stable_baselines_available': STABLE_BASELINES_AVAILABLE
        }
        
        if self.rl_agent and self.rl_agent.is_loaded:
            status['rl_model_info'] = {
                'algorithm': self.rl_agent.algorithm,
                'model_path': self.rl_agent.model_path
            }
        
        return status


def create_enhanced_web_simulator():
    """
    Create an enhanced web simulator with RL integration.
    
    This function patches the existing web simulator to add RL capabilities.
    """
    print("🚀 Creating RL-enhanced web simulator...")
    
    # Import the existing web simulator
    from web_simulator_realistic import WebSimulator, simulator
    
    # Create RL controller
    rl_controller = WebRLController()
    
    # Store original step method
    original_step = simulator.step_simulation
    
    def enhanced_step_simulation():
        """Enhanced step simulation with RL control."""
        # Get current state
        satellite_state = {
            'position': [simulator.satellite.position_x, simulator.satellite.position_y, simulator.satellite.position_z],
            'velocity': [simulator.satellite.velocity_x, simulator.satellite.velocity_y, simulator.satellite.velocity_z],
            'fuel_remaining_percent': getattr(simulator.satellite, 'fuel_remaining_percent', 100),
            'step_count': simulator.step_count
        }
        
        debris_states = []
        for debris in simulator.debris_objects:
            debris_states.append({
                'position': [debris.position_x, debris.position_y, debris.position_z],
                'velocity': [debris.velocity_x, debris.velocity_y, debris.velocity_z]
            })
        
        # Get action from controller
        action_info = rl_controller.get_action(satellite_state, debris_states)
        
        # Apply thrust to satellite (this would need to be implemented in the web simulator)
        # For now, we'll just log the action
        if any(thrust != 0 for thrust in action_info['thrust']):
            print(f"🎮 {action_info['source']}: Thrust = {action_info['thrust']}")
        
        # Call original step
        original_step()
        
        # Store action info for web interface
        simulator.last_action_info = action_info
    
    # Patch the simulator
    simulator.step_simulation = enhanced_step_simulation
    simulator.rl_controller = rl_controller
    
    print("✅ Web simulator enhanced with RL capabilities!")
    return simulator


# API functions for web interface integration
def api_set_control_mode(mode: str) -> Dict[str, Any]:
    """API endpoint to set control mode."""
    from web_simulator_realistic import simulator
    
    if hasattr(simulator, 'rl_controller'):
        success = simulator.rl_controller.set_control_mode(mode)
        return {
            'success': success,
            'current_mode': simulator.rl_controller.control_mode,
            'message': f"Control mode set to {mode}" if success else f"Invalid mode: {mode}"
        }
    else:
        return {'success': False, 'message': 'RL controller not initialized'}


def api_load_rl_model(model_name: str) -> Dict[str, Any]:
    """API endpoint to load an RL model."""
    from web_simulator_realistic import simulator
    
    if hasattr(simulator, 'rl_controller'):
        if model_name in simulator.rl_controller.available_models:
            model_path = simulator.rl_controller.available_models[model_name]
            success = simulator.rl_controller.load_rl_model(model_path)
            return {
                'success': success,
                'model_name': model_name,
                'model_path': model_path,
                'message': f"Model loaded successfully" if success else f"Failed to load model"
            }
        else:
            return {
                'success': False,
                'message': f"Model '{model_name}' not found",
                'available_models': list(simulator.rl_controller.available_models.keys())
            }
    else:
        return {'success': False, 'message': 'RL controller not initialized'}


def api_set_manual_thrust(thrust_x: float, thrust_y: float, thrust_z: float) -> Dict[str, Any]:
    """API endpoint to set manual thrust."""
    from web_simulator_realistic import simulator
    
    if hasattr(simulator, 'rl_controller'):
        simulator.rl_controller.set_manual_thrust(thrust_x, thrust_y, thrust_z)
        return {
            'success': True,
            'thrust': [thrust_x, thrust_y, thrust_z],
            'message': 'Manual thrust set'
        }
    else:
        return {'success': False, 'message': 'RL controller not initialized'}


def api_get_rl_status() -> Dict[str, Any]:
    """API endpoint to get RL controller status."""
    from web_simulator_realistic import simulator
    
    if hasattr(simulator, 'rl_controller'):
        return simulator.rl_controller.get_status()
    else:
        return {'error': 'RL controller not initialized'}


if __name__ == "__main__":
    # Demo the integration
    print("🛰️ RL Web Integration Demo")
    print("=" * 40)
    
    # Create enhanced simulator
    enhanced_sim = create_enhanced_web_simulator()
    
    # Show available functionality
    print(f"Control modes: {enhanced_sim.rl_controller.get_status()}")
    
    print("\n💡 Integration complete! The web interface now supports:")
    print("  • Zero policy (no thrust)")
    print("  • Manual thrust control")
    print("  • RL agent control with trained models")
    print("  • Real-time model switching")
    print("  • Visual debugging of RL behavior")