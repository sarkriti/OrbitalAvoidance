#!/usr/bin/env python3
"""
RL Satellite Training Visualizer

Shows how the RL agent learns to perform orbital transfers,
with real-time training visualization and mission success tracking.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.util import norm
import warnings
warnings.filterwarnings('ignore')

# Import environments
try:
    from satellite_env import SatelliteOrbitControlEnv as SatelliteEnv
except ImportError:
    from test_physics_env import TestHighFidelitySatelliteEnv as SatelliteEnv

class RLTrainingVisualizer:
    """Visualize RL agent learning orbital transfers"""
    
    def __init__(self, figsize=(16, 10)):
        self.figsize = figsize
        self.earth_radius = 6371.0  # km
        
    def visualize_agent_learning(self, model_path=None, num_episodes=3):
        """Show how trained agent performs orbital transfers"""
        
        print("🤖 RL AGENT ORBITAL TRANSFER VISUALIZATION")
        print("=" * 45)
        
        # Load trained model if available
        if model_path:
            try:
                from stable_baselines3 import PPO
                model = PPO.load(model_path)
                print(f"✅ Loaded trained model: {model_path}")
                use_trained_agent = True
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
                print("Using random agent instead...")
                use_trained_agent = False
        else:
            print("🎲 Using random agent for demonstration")
            use_trained_agent = False
        
        # Create environment
        env = SatelliteEnv()
        
        # Run episodes and visualize
        for episode in range(num_episodes):
            print(f"\n🛰️ Episode {episode + 1}/{num_episodes}")
            
            obs, _ = env.reset()
            
            # Store trajectory data
            trajectory = []
            rewards = []
            actions_taken = []
            
            total_reward = 0
            step = 0
            
            while step < 100:  # Limit steps for visualization
                
                if use_trained_agent:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample() * 0.1  # Small random actions
                
                obs, reward, done, truncated, info = env.step(action)
                
                # Store data for visualization
                if hasattr(env, 'current_orbit'):
                    current_orbit = env.current_orbit
                    target_orbit = env.target_orbit
                    
                    current_pos = current_orbit.r.to(u.km).value
                    target_pos = target_orbit.r.to(u.km).value
                    current_alt = (norm(current_orbit.r) - Earth.R).to(u.km).value
                    target_alt = (norm(target_orbit.r) - Earth.R).to(u.km).value
                    
                    trajectory.append({
                        'step': step,
                        'position': current_pos,
                        'target_position': target_pos,
                        'altitude': current_alt,
                        'target_altitude': target_alt,
                        'reward': reward,
                        'action': action.copy()
                    })
                
                rewards.append(reward)
                actions_taken.append(action.copy())
                total_reward += reward
                step += 1
                
                if done or truncated:
                    break
            
            print(f"   Total reward: {total_reward:.2f}")
            print(f"   Steps taken: {step}")
            
            # Create visualization for this episode
            if trajectory:
                self._plot_episode_trajectory(trajectory, episode + 1, 
                                            use_trained_agent)
        
        env.close()
        print("\n✅ RL training visualization complete!")
    
    def _plot_episode_trajectory(self, trajectory, episode_num, is_trained):
        """Plot the trajectory for one episode"""
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=self.figsize)
        
        # 3D trajectory plot
        ax_3d = fig.add_subplot(221, projection='3d')
        self._plot_3d_trajectory(ax_3d, trajectory, episode_num, is_trained)
        
        # Altitude vs time
        ax_alt = fig.add_subplot(222)
        self._plot_altitude_progress(ax_alt, trajectory)
        
        # Reward vs time
        ax_reward = fig.add_subplot(223)
        self._plot_reward_progress(ax_reward, trajectory)
        
        # Action analysis
        ax_action = fig.add_subplot(224)
        self._plot_action_analysis(ax_action, trajectory)
        
        agent_type = "Trained Agent" if is_trained else "Random Agent"
        fig.suptitle(f'Episode {episode_num}: {agent_type} Orbital Transfer', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_3d_trajectory(self, ax, trajectory, episode_num, is_trained):
        """Plot 3D orbital trajectory"""
        
        # Draw Earth
        self._draw_earth_sphere(ax)
        
        # Extract positions
        positions = np.array([step['position'] for step in trajectory])
        target_pos = trajectory[0]['target_position']
        
        # Plot trajectory
        color = 'blue' if is_trained else 'orange'
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
               color=color, linewidth=3, alpha=0.8, label='Satellite Path')
        
        # Mark start and end
        ax.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 
                  color='green', s=100, label='Start', marker='o')
        ax.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], 
                  color='red', s=100, label='End', marker='s')
        
        # Mark target orbit (approximate circle)
        target_radius = np.linalg.norm(target_pos)
        theta = np.linspace(0, 2*np.pi, 50)
        target_x = target_radius * np.cos(theta)
        target_y = target_radius * np.sin(theta)
        target_z = np.zeros_like(theta)
        ax.plot(target_x, target_y, target_z, 'r--', alpha=0.5, 
               linewidth=2, label='Target Orbit')
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('3D Orbital Trajectory')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = max(np.max(np.abs(positions)), target_radius) * 1.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
    
    def _plot_altitude_progress(self, ax, trajectory):
        """Plot altitude vs time"""
        
        steps = [step['step'] for step in trajectory]
        altitudes = [step['altitude'] for step in trajectory]
        target_alt = trajectory[0]['target_altitude']
        
        ax.plot(steps, altitudes, 'b-', linewidth=2, label='Current Altitude')
        ax.axhline(y=target_alt, color='r', linestyle='--', linewidth=2, 
                  label=f'Target Altitude ({target_alt:.0f} km)')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Altitude (km)')
        ax.set_title('Altitude Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add improvement percentage
        initial_error = abs(altitudes[0] - target_alt)
        final_error = abs(altitudes[-1] - target_alt)
        if initial_error > 0:
            improvement = (initial_error - final_error) / initial_error * 100
            ax.text(0.7, 0.9, f'Improvement: {improvement:.1f}%', 
                   transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    def _plot_reward_progress(self, ax, trajectory):
        """Plot reward vs time"""
        
        steps = [step['step'] for step in trajectory]
        rewards = [step['reward'] for step in trajectory]
        cumulative_rewards = np.cumsum(rewards)
        
        ax.plot(steps, rewards, 'g-', linewidth=1, alpha=0.7, label='Step Reward')
        ax.plot(steps, cumulative_rewards, 'b-', linewidth=2, label='Cumulative Reward')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add final score
        final_reward = cumulative_rewards[-1]
        ax.text(0.7, 0.1, f'Final Score: {final_reward:.1f}', 
               transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
    
    def _plot_action_analysis(self, ax, trajectory):
        """Plot action analysis"""
        
        actions = np.array([step['action'] for step in trajectory])
        steps = [step['step'] for step in trajectory]
        
        # Plot thrust components
        ax.plot(steps, actions[:, 0], label='Thrust X', alpha=0.8)
        ax.plot(steps, actions[:, 1], label='Thrust Y', alpha=0.8)
        ax.plot(steps, actions[:, 2], label='Thrust Z', alpha=0.8)
        
        # Plot thrust magnitude
        thrust_magnitude = np.linalg.norm(actions, axis=1)
        ax.plot(steps, thrust_magnitude, 'k--', linewidth=2, label='Total Thrust')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Thrust Command')
        ax.set_title('Control Actions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add fuel usage estimate
        total_fuel = np.sum(thrust_magnitude)
        ax.text(0.7, 0.9, f'Fuel Used: {total_fuel:.2f}', 
               transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='lightyellow'))
    
    def _draw_earth_sphere(self, ax, alpha=0.3):
        """Draw Earth sphere"""
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = self.earth_radius * np.outer(np.cos(u), np.sin(v))
        y = self.earth_radius * np.outer(np.sin(u), np.sin(v))
        z = self.earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, alpha=alpha, color='lightblue')
    
    def compare_random_vs_trained(self, model_path):
        """Compare random agent vs trained agent performance"""
        
        print("🆚 RANDOM vs TRAINED AGENT COMPARISON")
        print("=" * 40)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Random Agent vs Trained Agent Comparison', 
                    fontsize=16, fontweight='bold')
        
        # Test both agents
        agents = [
            {'name': 'Random Agent', 'use_model': False, 'color': 'orange'},
            {'name': 'Trained Agent', 'use_model': True, 'color': 'blue'}
        ]
        
        for i, agent_config in enumerate(agents):
            print(f"\n🤖 Testing {agent_config['name']}...")
            
            # Load model if needed
            model = None
            if agent_config['use_model']:
                try:
                    from stable_baselines3 import PPO
                    model = PPO.load(model_path)
                except Exception as e:
                    print(f"Could not load model: {e}")
                    continue
            
            # Run test episode
            env = SatelliteEnv()
            obs, _ = env.reset()
            
            trajectory = []
            total_reward = 0
            
            for step in range(50):
                if model:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample() * 0.1
                
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                if hasattr(env, 'current_orbit'):
                    current_alt = (norm(env.current_orbit.r) - Earth.R).to(u.km).value
                    target_alt = (norm(env.target_orbit.r) - Earth.R).to(u.km).value
                    
                    trajectory.append({
                        'step': step,
                        'altitude': current_alt,
                        'target_altitude': target_alt,
                        'reward': reward
                    })
                
                if done or truncated:
                    break
            
            env.close()
            
            # Plot results
            if trajectory:
                steps = [t['step'] for t in trajectory]
                altitudes = [t['altitude'] for t in trajectory]
                rewards = [t['reward'] for t in trajectory]
                target_alt = trajectory[0]['target_altitude']
                
                # Altitude plot
                axes[i, 0].plot(steps, altitudes, color=agent_config['color'], linewidth=2)
                axes[i, 0].axhline(y=target_alt, color='red', linestyle='--', alpha=0.7)
                axes[i, 0].set_title(f"{agent_config['name']} - Altitude Progress")
                axes[i, 0].set_xlabel('Step')
                axes[i, 0].set_ylabel('Altitude (km)')
                axes[i, 0].grid(True, alpha=0.3)
                
                # Reward plot
                cumulative_rewards = np.cumsum(rewards)
                axes[i, 1].plot(steps, cumulative_rewards, color=agent_config['color'], linewidth=2)
                axes[i, 1].set_title(f"{agent_config['name']} - Cumulative Reward")
                axes[i, 1].set_xlabel('Step')
                axes[i, 1].set_ylabel('Cumulative Reward')
                axes[i, 1].grid(True, alpha=0.3)
                
                print(f"   Final reward: {total_reward:.2f}")
                print(f"   Altitude error: {abs(altitudes[-1] - target_alt):.1f} km")
        
        plt.tight_layout()
        plt.show()


def demo_rl_visualization():
    """Demonstrate RL training visualization"""
    
    print("🤖 RL SATELLITE TRAINING VISUALIZER")
    print("=" * 40)
    
    visualizer = RLTrainingVisualizer()
    
    # Check for trained models
    import os
    model_files = []
    if os.path.exists("models/"):
        model_files = [f for f in os.listdir("models/") if f.endswith('.zip')]
    
    if model_files:
        print(f"\n📦 Found {len(model_files)} trained models:")
        for i, model in enumerate(model_files):
            print(f"   {i+1}. {model}")
        
        print("\nChoose visualization:")
        print("1. Show trained agent performance")
        print("2. Compare random vs trained agent")
        print("3. Show random agent only")
        
        try:
            choice = input("Enter choice (1-3): ").strip()
        except KeyboardInterrupt:
            choice = "3"
        
        if choice == "1":
            model_path = f"models/{model_files[0].replace('.zip', '')}"
            visualizer.visualize_agent_learning(model_path, num_episodes=2)
        elif choice == "2":
            model_path = f"models/{model_files[0].replace('.zip', '')}"
            visualizer.compare_random_vs_trained(model_path)
        else:
            visualizer.visualize_agent_learning(None, num_episodes=2)
    else:
        print("\n📝 No trained models found. Showing random agent...")
        visualizer.visualize_agent_learning(None, num_episodes=2)


if __name__ == "__main__":
    demo_rl_visualization()