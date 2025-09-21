#!/usr/bin/env python3
"""
Satellite Control Environment for Reinforcement Learning

This creates a custom Gymnasium environment for satellite orbital control
using poliastro for orbital mechanics and stable_baselines3 for RL algorithms.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.util import norm
import matplotlib.pyplot as plt


class SatelliteOrbitControlEnv(gym.Env):
    """
    Custom Environment for Satellite Orbital Control
    
    The goal is to control a satellite's orbit through thrust maneuvers
    to reach and maintain a target orbit.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Define action space: [thrust_x, thrust_y, thrust_z] in m/s
        # Normalized between -1 and 1, scaled in step()
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(3,), 
            dtype=np.float32
        )
        
        # Define observation space: 
        # [position_x, position_y, position_z, velocity_x, velocity_y, velocity_z,
        #  target_pos_x, target_pos_y, target_pos_z, target_vel_x, target_vel_y, target_vel_z]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(12,), 
            dtype=np.float32
        )
        
        # Environment parameters
        self.max_thrust = 0.1  # m/s maximum thrust per action
        self.max_steps = 1000
        self.dt = 60.0  # time step in seconds (1 minute)
        
        # Target orbit parameters (geostationary orbit example)
        self.target_altitude = 35786 * u.km  # GEO altitude
        self.target_orbit = None
        
        # Current state
        self.current_orbit = None
        self.step_count = 0
        self.current_time = None
        
        # Initialize environment
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize random starting orbit (Low Earth Orbit)
        alt_start = 400 + np.random.uniform(-100, 100)  # 300-500 km altitude
        ecc_start = np.random.uniform(0.0, 0.1)  # Low eccentricity
        inc_start = np.random.uniform(0, 30)  # Inclination 0-30 degrees
        
        self.current_orbit = Orbit.circular(
            Earth, 
            alt=alt_start * u.km,
            inc=inc_start * u.deg
        )
        
        # Define target orbit (Geostationary)
        self.target_orbit = Orbit.circular(
            Earth,
            alt=self.target_altitude,
            inc=0 * u.deg  # Equatorial orbit
        )
        
        self.step_count = 0
        self.current_time = Time.now()
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in the environment"""
        
        # Scale action to actual thrust values
        thrust = action * self.max_thrust  # m/s
        
        # Apply maneuver if thrust is significant
        if np.linalg.norm(thrust) > 1e-6:
            # Convert thrust to velocity change
            dv = thrust * u.m / u.s
            
            # Create maneuver
            try:
                maneuver = Maneuver.impulse(dv)
                self.current_orbit = self.current_orbit.apply_maneuver(maneuver)
            except Exception as e:
                # If maneuver fails, apply penalty but continue
                reward = -100.0
                done = True
                return self._get_observation(), reward, done, False, {"error": str(e)}
        
        # Propagate orbit forward in time
        self.current_time = self.current_time + self.dt * u.s
        try:
            self.current_orbit = self.current_orbit.propagate(self.current_time)
        except Exception:
            # Orbit may have decayed or become invalid
            reward = -100.0
            done = True
            return self._get_observation(), reward, done, False, {"error": "Orbit decay"}
        
        # Calculate reward
        reward = self._calculate_reward()
        self._last_reward = reward  # Store for visualization
        
        # Update total reward tracking
        if not hasattr(self, '_total_reward'):
            self._total_reward = 0
        self._total_reward += reward
        
        # Check if episode is done
        self.step_count += 1
        done = self.step_count >= self.max_steps
        truncated = False
        
        # Check for mission success (within tolerance of target orbit)
        if self._is_target_reached():
            reward += 100.0  # Bonus for reaching target
            done = True
        
        return self._get_observation(), reward, done, truncated, {}
    
    def _get_observation(self):
        """Get current observation state"""
        
        # Current orbit state vectors
        r_current = self.current_orbit.r.to(u.km).value  # position in km
        v_current = self.current_orbit.v.to(u.km/u.s).value  # velocity in km/s
        
        # Target orbit state vectors
        r_target = self.target_orbit.r.to(u.km).value
        v_target = self.target_orbit.v.to(u.km/u.s).value
        
        # Combine into observation vector
        obs = np.concatenate([
            r_current,
            v_current, 
            r_target,
            v_target
        ]).astype(np.float32)
        
        return obs
    
    def _calculate_reward(self):
        """Calculate reward based on proximity to target orbit"""
        
        # Position difference (km)
        r_diff = np.linalg.norm(
            self.current_orbit.r.to(u.km).value - 
            self.target_orbit.r.to(u.km).value
        )
        
        # Velocity difference (km/s)
        v_diff = np.linalg.norm(
            self.current_orbit.v.to(u.km/u.s).value - 
            self.target_orbit.v.to(u.km/u.s).value
        )
        
        # Reward is negative of combined distance (closer = higher reward)
        # Scale to reasonable values
        position_reward = -r_diff / 1000.0  # Scale down position difference
        velocity_reward = -v_diff * 100.0    # Scale up velocity difference
        
        # Small positive reward for staying in valid orbit
        stability_reward = 1.0
        
        total_reward = position_reward + velocity_reward + stability_reward
        
        return total_reward
    
    def _is_target_reached(self, position_tolerance=500*u.km, velocity_tolerance=0.1*u.km/u.s):
        """Check if satellite has reached target orbit within tolerance"""
        
        r_diff = norm(self.current_orbit.r - self.target_orbit.r)
        v_diff = norm(self.current_orbit.v - self.target_orbit.v)
        
        return (r_diff < position_tolerance) and (v_diff < velocity_tolerance)
    
    def render(self, mode='human'):
        """Render the environment with visual feedback"""
        if mode == 'human':
            print(f"Step: {self.step_count}")
            print(f"Current altitude: {self.current_orbit.a.to(u.km):.1f}")
            print(f"Target altitude: {self.target_orbit.a.to(u.km):.1f}")
            print(f"Altitude difference: {(self.current_orbit.a - self.target_orbit.a).to(u.km):.1f}")
            print("-" * 40)
        elif mode == 'visual':
            # Use the visualizer for graphical output
            try:
                from satellite_visualizer import SatelliteVisualizer
                viz = SatelliteVisualizer()
                fig = viz.create_mission_dashboard(
                    self.current_orbit, self.target_orbit,
                    step=self.step_count, reward=self._last_reward, 
                    total_reward=getattr(self, '_total_reward', 0)
                )
                import matplotlib.pyplot as plt
                plt.show(block=False)
                plt.pause(0.1)
                return fig
            except ImportError:
                print("Visual rendering requires satellite_visualizer.py")
                self.render('human')  # Fallback to text
        
        # Store last reward for visualization
        if not hasattr(self, '_last_reward'):
            self._last_reward = 0
    
    def close(self):
        """Clean up environment"""
        pass


def test_environment():
    """Test the satellite environment"""
    
    print("Testing Satellite Control Environment...")
    
    # Create environment
    env = SatelliteOrbitControlEnv()
    
    # Test random actions
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    for i in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Step {i+1}: Reward = {reward:.2f}")
        env.render()
        
        if done:
            print("Episode finished!")
            break
    
    env.close()
    print("Environment test completed!")


def train_satellite_controller():
    """Train an RL agent to control the satellite"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    
    print("Training Satellite Control Agent...")
    
    # Create and check environment
    env = SatelliteOrbitControlEnv()
    check_env(env)
    print("Environment check passed!")
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
        tensorboard_log="./satellite_tensorboard/"
    )
    
    # Train the agent
    print("Starting training...")
    model.learn(total_timesteps=100000)
    
    # Save the model
    model.save("satellite_control_ppo")
    print("Model saved as 'satellite_control_ppo'")
    
    # Test the trained agent
    print("Testing trained agent...")
    obs, _ = env.reset()
    
    for i in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Test Step {i+1}: Reward = {reward:.2f}")
        env.render()
        
        if done:
            print("Test episode finished!")
            break
    
    env.close()
    print("Training completed!")


if __name__ == "__main__":
    print("Satellite Control Environment")
    print("=" * 50)
    
    choice = input("Choose option:\n1. Test environment\n2. Train RL agent\n3. Both\nEnter choice (1/2/3): ")
    
    if choice == "1":
        test_environment()
    elif choice == "2":
        train_satellite_controller()
    elif choice == "3":
        test_environment()
        print("\n" + "="*50 + "\n")
        train_satellite_controller()
    else:
        print("Invalid choice. Running test...")
        test_environment()