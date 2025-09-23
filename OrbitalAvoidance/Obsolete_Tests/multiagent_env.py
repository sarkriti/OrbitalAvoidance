#!/usr/bin/env python3
"""
PettingZoo Multi-Agent Wrapper for Orbital Avoidance Environment

This module wraps the existing Gymnasium environment to enable multi-agent
reinforcement learning using the PettingZoo API.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any, Union
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
import logging

from configurable_env import ConfigurableOrbitalEnvironment
from algorithms import Policy


class MultiAgentOrbitalAvoidance(ParallelEnv):
    """
    Multi-agent orbital avoidance environment using PettingZoo.
    
    Each satellite is an independent agent that can take actions to avoid
    collisions with debris and other satellites.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "multiagent_orbital_avoidance_v1",
    }
    
    def __init__(self, 
                 num_satellites: int = 2,
                 num_debris: int = 8,
                 env_config: Optional[EnvConfig] = None,
                 render_mode: Optional[str] = None):
        """
        Initialize multi-agent orbital avoidance environment.
        
        Args:
            num_satellites: Number of controllable satellites (agents)
            num_debris: Number of debris objects
            env_config: Environment configuration
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.num_satellites = num_satellites
        self.num_debris = num_debris
        self.render_mode = render_mode
        
        # Create base environment for reference
        self.base_env = make_env(seed=42)
        
        # Agent names
        self.possible_agents = [f"satellite_{i}" for i in range(num_satellites)]
        self.agents = self.possible_agents.copy()
        
        # Observation and action spaces (same for all agents)
        self._setup_spaces()
        
        # Internal state
        self.timestep = 0
        self.max_timesteps = 1000
        self._agent_positions = {}
        self._agent_velocities = {}
        self._debris_positions = []
        self._last_actions = {agent: np.zeros(3) for agent in self.agents}
        
        logging.info(f"Created multi-agent environment with {num_satellites} satellites and {num_debris} debris")
    
    def _setup_spaces(self):
        """Setup observation and action spaces for all agents."""
        # Each agent observes:
        # - Own position (3D)
        # - Own velocity (3D) 
        # - Relative positions to nearest debris/satellites (k*3D)
        # - Relative velocities to nearest debris/satellites (k*3D)
        
        k_nearest = 5  # Number of nearest objects to observe
        obs_dim = 3 + 3 + k_nearest * 3 + k_nearest * 3  # pos + vel + rel_pos + rel_vel
        
        self.observation_spaces = {
            agent: spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(obs_dim,), 
                dtype=np.float32
            )
            for agent in self.possible_agents
        }
        
        # Action space: 3D acceleration commands (Δv in RTN frame)
        self.action_spaces = {
            agent: spaces.Box(
                low=-0.1,   # Max 0.1 km/s acceleration per axis
                high=0.1,
                shape=(3,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }
    
    @property
    def observation_space(self):
        """Return observation space for current agents."""
        return {agent: self.observation_spaces[agent] for agent in self.agents}
    
    @property
    def action_space(self):
        """Return action space for current agents."""
        return {agent: self.action_spaces[agent] for agent in self.agents}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment and return initial observations."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset internal state
        self.agents = self.possible_agents.copy()
        self.timestep = 0
        
        # Generate initial satellite positions/velocities
        self._initialize_satellites()
        
        # Generate debris field
        self._initialize_debris()
        
        # Get initial observations
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def _initialize_satellites(self):
        """Initialize satellite positions and velocities."""
        self._agent_positions = {}
        self._agent_velocities = {}
        
        for i, agent in enumerate(self.agents):
            # Distribute satellites in different orbital planes
            altitude_km = 400 + i * 20  # Slightly different altitudes
            inclination_deg = 51.6 + i * 5  # Different inclinations
            
            # Initial position (simplified circular orbit)
            angle = (2 * np.pi * i) / len(self.agents)
            radius = 6371 + altitude_km
            
            self._agent_positions[agent] = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.0
            ])
            
            # Orbital velocity
            v_orbital = np.sqrt(398600.4 / radius)  # km/s
            self._agent_velocities[agent] = np.array([
                -v_orbital * np.sin(angle),
                v_orbital * np.cos(angle),
                0.0
            ])
    
    def _initialize_debris(self):
        """Initialize debris field."""
        self._debris_positions = []
        
        for i in range(self.num_debris):
            # Random debris positions
            altitude_km = 350 + np.random.uniform(0, 100)
            angle = np.random.uniform(0, 2 * np.pi)
            inclination = np.random.uniform(-10, 10)
            
            radius = 6371 + altitude_km
            
            pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                radius * np.sin(np.radians(inclination)) * 100
            ])
            
            self._debris_positions.append(pos)
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        observations = {}
        
        for agent in self.agents:
            obs = self._get_agent_observation(agent)
            observations[agent] = obs
        
        return observations
    
    def _get_agent_observation(self, agent: str) -> np.ndarray:
        """Get observation for a specific agent."""
        agent_pos = self._agent_positions[agent]
        agent_vel = self._agent_velocities[agent]
        
        # Collect all other object positions
        other_positions = []
        other_velocities = []
        
        # Add other satellites
        for other_agent in self.agents:
            if other_agent != agent:
                other_positions.append(self._agent_positions[other_agent])
                other_velocities.append(self._agent_velocities[other_agent])
        
        # Add debris (with zero velocity assumption)
        for debris_pos in self._debris_positions:
            other_positions.append(debris_pos)
            other_velocities.append(np.zeros(3))
        
        # Calculate relative positions and distances
        if other_positions:
            other_positions = np.array(other_positions)
            other_velocities = np.array(other_velocities)
            
            rel_positions = other_positions - agent_pos
            distances = np.linalg.norm(rel_positions, axis=1)
            
            # Get k nearest objects
            k = min(5, len(distances))
            nearest_indices = np.argsort(distances)[:k]
            
            nearest_rel_pos = rel_positions[nearest_indices]
            nearest_rel_vel = other_velocities[nearest_indices] - agent_vel
            
            # Pad if necessary
            if len(nearest_rel_pos) < 5:
                padding_pos = np.zeros((5 - len(nearest_rel_pos), 3))
                padding_vel = np.zeros((5 - len(nearest_rel_vel), 3))
                nearest_rel_pos = np.vstack([nearest_rel_pos, padding_pos])
                nearest_rel_vel = np.vstack([nearest_rel_vel, padding_vel])
        else:
            nearest_rel_pos = np.zeros((5, 3))
            nearest_rel_vel = np.zeros((5, 3))
        
        # Construct observation vector
        obs = np.concatenate([
            agent_pos,                    # Own position (3)
            agent_vel,                    # Own velocity (3)
            nearest_rel_pos.flatten(),    # Relative positions (15)
            nearest_rel_vel.flatten()     # Relative velocities (15)
        ])
        
        return obs.astype(np.float32)
    
    def step(self, actions: Dict[str, np.ndarray]):
        """Step the environment with actions from all agents."""
        self.timestep += 1
        
        # Store actions
        for agent in self.agents:
            if agent in actions:
                self._last_actions[agent] = np.array(actions[agent])
        
        # Update satellite positions based on actions
        self._update_satellite_dynamics(actions)
        
        # Calculate rewards
        rewards = self._calculate_rewards(actions)
        
        # Check for terminations
        terminations = self._check_terminations()
        truncations = self._check_truncations()
        
        # Get new observations
        observations = self._get_observations()
        
        # Information for each agent
        infos = {agent: self._get_agent_info(agent) for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _update_satellite_dynamics(self, actions: Dict[str, np.ndarray]):
        """Update satellite positions and velocities based on actions."""
        dt = 60.0  # 1 minute time step
        
        for agent in self.agents:
            if agent in actions:
                # Apply action as acceleration (simplified dynamics)
                acceleration = np.array(actions[agent])
                
                # Update velocity
                self._agent_velocities[agent] += acceleration * dt
                
                # Update position
                self._agent_positions[agent] += self._agent_velocities[agent] * dt
        
        # Also update debris positions (simplified orbital motion)
        self._update_debris_dynamics(dt)
    
    def _update_debris_dynamics(self, dt: float):
        """Update debris positions with simplified orbital dynamics."""
        for i, debris_pos in enumerate(self._debris_positions):
            # Simple circular orbital motion
            radius = np.linalg.norm(debris_pos)
            if radius > 0:
                v_orbital = np.sqrt(398600.4 / radius)  # km/s
                angular_velocity = v_orbital / radius
                
                # Rotate position
                angle = angular_velocity * dt
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                
                x, y, z = debris_pos
                new_x = x * cos_a - y * sin_a
                new_y = x * sin_a + y * cos_a
                
                self._debris_positions[i] = np.array([new_x, new_y, z])
    
    def _calculate_rewards(self, actions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate rewards for all agents."""
        rewards = {}
        
        for agent in self.agents:
            reward = 0.0
            
            # Base survival reward
            reward += 1.0
            
            # Penalty for fuel usage (action magnitude)
            if agent in actions:
                fuel_penalty = -np.linalg.norm(actions[agent]) * 10.0
                reward += fuel_penalty
            
            # Collision avoidance reward
            min_distance = self._get_min_distance_to_objects(agent)
            if min_distance < 50.0:  # Close approach penalty
                proximity_penalty = -100.0 * (50.0 - min_distance) / 50.0
                reward += proximity_penalty
            
            if min_distance < 10.0:  # Collision penalty
                reward -= 1000.0
            
            rewards[agent] = reward
        
        return rewards
    
    def _get_min_distance_to_objects(self, agent: str) -> float:
        """Get minimum distance from agent to all other objects."""
        agent_pos = self._agent_positions[agent]
        min_dist = float('inf')
        
        # Check distance to other satellites
        for other_agent in self.agents:
            if other_agent != agent:
                dist = np.linalg.norm(agent_pos - self._agent_positions[other_agent])
                min_dist = min(min_dist, dist)
        
        # Check distance to debris
        for debris_pos in self._debris_positions:
            dist = np.linalg.norm(agent_pos - debris_pos)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _check_terminations(self) -> Dict[str, bool]:
        """Check if any agents should terminate (collision, etc.)."""
        terminations = {}
        
        for agent in self.agents:
            terminated = False
            
            # Check for collision
            min_distance = self._get_min_distance_to_objects(agent)
            if min_distance < 1.0:  # 1 km collision threshold
                terminated = True
            
            # Check if orbit decayed (too low altitude)
            altitude = np.linalg.norm(self._agent_positions[agent]) - 6371.0
            if altitude < 150.0:  # 150 km minimum altitude
                terminated = True
            
            terminations[agent] = terminated
        
        return terminations
    
    def _check_truncations(self) -> Dict[str, bool]:
        """Check if episode should be truncated (time limit, etc.)."""
        truncated = self.timestep >= self.max_timesteps
        return {agent: truncated for agent in self.agents}
    
    def _get_agent_info(self, agent: str) -> Dict[str, Any]:
        """Get information for a specific agent."""
        agent_pos = self._agent_positions[agent]
        altitude = np.linalg.norm(agent_pos) - 6371.0
        min_distance = self._get_min_distance_to_objects(agent)
        
        return {
            "altitude_km": altitude,
            "min_distance_km": min_distance,
            "position": agent_pos.copy(),
            "velocity": self._agent_velocities[agent].copy(),
            "last_action": self._last_actions[agent].copy(),
            "timestep": self.timestep
        }
    
    def render(self):
        """Render the environment (placeholder for now)."""
        if self.render_mode == "human":
            print(f"Timestep: {self.timestep}")
            for agent in self.agents:
                info = self._get_agent_info(agent)
                print(f"{agent}: altitude={info['altitude_km']:.1f}km, "
                     f"min_distance={info['min_distance_km']:.1f}km")
        
        return None
    
    def close(self):
        """Close the environment."""
        pass


def make_multiagent_env(num_satellites: int = 2, 
                       num_debris: int = 8,
                       env_config: Optional[EnvConfig] = None) -> MultiAgentOrbitalAvoidance:
    """Factory function to create multi-agent orbital avoidance environment."""
    return MultiAgentOrbitalAvoidance(
        num_satellites=num_satellites,
        num_debris=num_debris,
        env_config=env_config
    )


def make_aec_env(num_satellites: int = 2, 
                num_debris: int = 8,
                env_config: Optional[EnvConfig] = None):
    """Create AEC (Agent-Environment-Cycle) version of the environment."""
    parallel_env = make_multiagent_env(num_satellites, num_debris, env_config)
    aec_env = parallel_to_aec(parallel_env)
    return aec_env


if __name__ == "__main__":
    # Test the multi-agent environment
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Multi-Agent Orbital Avoidance Environment")
    
    # Create environment
    env = make_multiagent_env(num_satellites=2, num_debris=5)
    
    # Reset environment
    observations, infos = env.reset(seed=42)
    print(f"Initial observations: {list(observations.keys())}")
    print(f"Observation shape: {observations['satellite_0'].shape}")
    
    # Run a few steps
    for step in range(5):
        # Random actions
        actions = {
            agent: env.action_space[agent].sample() 
            for agent in env.agents
        }
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        print(f"\nStep {step + 1}:")
        for agent in env.agents:
            print(f"  {agent}: reward={rewards[agent]:.2f}, "
                 f"altitude={infos[agent]['altitude_km']:.1f}km")
        
        # Check if any agent terminated
        if any(terms.values()):
            print("Some agents terminated!")
            break
    
    env.close()
    print("Multi-agent test completed!")