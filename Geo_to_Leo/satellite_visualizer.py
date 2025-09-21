#!/usr/bin/env python3
"""
Satellite Orbit Visualization Tools

This module provides visualization capabilities for satellite orbits,
training progress, and real-time mission monitoring.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import warnings
warnings.filterwarnings('ignore')

# Try to import poliastro plotting, but use fallback if not available
try:
    from poliastro.plotting import OrbitPlotter3D
    POLIASTRO_PLOTTING_AVAILABLE = True
except ImportError:
    POLIASTRO_PLOTTING_AVAILABLE = False
    print("Note: poliastro plotting not available, using matplotlib-only visualization")


class SatelliteVisualizer:
    """Visualize satellite orbits and mission progress"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        self.earth_radius = 6371.0  # km
        
    def plot_earth_3d(self, ax):
        """Add Earth sphere to 3D plot"""
        # Create Earth sphere
        u_sphere = np.linspace(0, 2 * np.pi, 50)
        v_sphere = np.linspace(0, np.pi, 50)
        x_earth = self.earth_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
        y_earth = self.earth_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
        z_earth = self.earth_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
        
        ax.plot_surface(x_earth, y_earth, z_earth, alpha=0.3, color='blue')
        
    def visualize_orbits_3d(self, current_orbit, target_orbit, title="Satellite Mission"):
        """Create 3D visualization of current and target orbits"""
        
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Plot Earth
        self.plot_earth_3d(self.ax)
        
        # Generate orbit points by propagating over time
        try:
            # Create time array for one orbital period
            n_points = 100
            period_current = current_orbit.period.to(u.s).value
            period_target = target_orbit.period.to(u.s).value
            
            # Use the longer period to show both orbits completely
            max_period = max(period_current, period_target)
            times = np.linspace(0, max_period, n_points) * u.s
            
            current_positions = []
            target_positions = []
            
            base_time = Time.now()
            
            for t in times:
                try:
                    # Current orbit positions
                    curr_orbit_t = current_orbit.propagate(base_time + t)
                    curr_pos = curr_orbit_t.r.to(u.km).value
                    current_positions.append(curr_pos)
                    
                    # Target orbit positions  
                    tgt_orbit_t = target_orbit.propagate(base_time + t)
                    tgt_pos = tgt_orbit_t.r.to(u.km).value
                    target_positions.append(tgt_pos)
                    
                except Exception as e:
                    # If propagation fails, use the last known position
                    if current_positions:
                        current_positions.append(current_positions[-1])
                        target_positions.append(target_positions[-1])
                    else:
                        # Use current positions as fallback
                        current_positions.append(current_orbit.r.to(u.km).value)
                        target_positions.append(target_orbit.r.to(u.km).value)
            
            current_positions = np.array(current_positions)
            target_positions = np.array(target_positions)
            
            # Plot orbits
            self.ax.plot(current_positions[:, 0], current_positions[:, 1], current_positions[:, 2], 
                        'r-', linewidth=2, label='Current Orbit', alpha=0.8)
            self.ax.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 
                        'g--', linewidth=2, label='Target Orbit', alpha=0.8)
            
            # Mark current position
            curr_pos = current_orbit.r.to(u.km).value
            self.ax.scatter(curr_pos[0], curr_pos[1], curr_pos[2], 
                          color='red', s=100, label='Satellite', zorder=10)
            
            # Mark target position
            tgt_pos = target_orbit.r.to(u.km).value
            self.ax.scatter(tgt_pos[0], tgt_pos[1], tgt_pos[2], 
                          color='green', s=100, label='Target', zorder=10)
            
            # Styling
            self.ax.set_xlabel('X (km)')
            self.ax.set_ylabel('Y (km)')
            self.ax.set_zlabel('Z (km)')
            self.ax.set_title(title)
            self.ax.legend()
            
            # Set equal aspect ratio
            max_range = max(
                np.max(np.abs(current_positions)) if len(current_positions) > 0 else 10000,
                np.max(np.abs(target_positions)) if len(target_positions) > 0 else 10000
            )
            self.ax.set_xlim([-max_range, max_range])
            self.ax.set_ylim([-max_range, max_range])
            self.ax.set_zlim([-max_range, max_range])
            
        except Exception as e:
            print(f"Orbit visualization error: {e}")
            print("Showing simplified orbit visualization...")
            
            # Fallback: show simple circular approximations
            curr_r = np.linalg.norm(current_orbit.r.to(u.km).value)
            tgt_r = np.linalg.norm(target_orbit.r.to(u.km).value)
            
            # Simple circular orbits
            theta = np.linspace(0, 2*np.pi, 100)
            
            # Current orbit (assume circular in XY plane)
            curr_x = curr_r * np.cos(theta)
            curr_y = curr_r * np.sin(theta)
            curr_z = np.zeros_like(theta)
            
            # Target orbit
            tgt_x = tgt_r * np.cos(theta) 
            tgt_y = tgt_r * np.sin(theta)
            tgt_z = np.zeros_like(theta)
            
            self.ax.plot(curr_x, curr_y, curr_z, 'r-', linewidth=2, label='Current Orbit')
            self.ax.plot(tgt_x, tgt_y, tgt_z, 'g--', linewidth=2, label='Target Orbit')
            
            # Current positions
            curr_pos = current_orbit.r.to(u.km).value
            self.ax.scatter(curr_pos[0], curr_pos[1], curr_pos[2], 
                          color='red', s=100, label='Satellite')
            
            max_range = max(curr_r, tgt_r) * 1.2
            self.ax.set_xlim([-max_range, max_range])
            self.ax.set_ylim([-max_range, max_range])
            self.ax.set_zlim([-max_range, max_range])
            
            self.ax.set_xlabel('X (km)')
            self.ax.set_ylabel('Y (km)')
            self.ax.set_zlabel('Z (km)')
            self.ax.set_title(title)
            self.ax.legend()
            
        plt.tight_layout()
        return self.fig, self.ax
    
    def plot_training_progress(self, rewards, episode_lengths, window_size=100):
        """Plot training progress with smoothed curves"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        episodes = np.arange(len(rewards))
        
        # Smooth rewards
        if len(rewards) > window_size:
            smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            smooth_episodes = episodes[window_size-1:]
            ax1.plot(smooth_episodes, smoothed_rewards, 'b-', linewidth=2, label=f'Smoothed ({window_size} ep)')
        
        ax1.plot(episodes, rewards, 'lightblue', alpha=0.3, label='Raw Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Episode lengths
        if len(episode_lengths) > window_size:
            smoothed_lengths = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(smooth_episodes, smoothed_lengths, 'g-', linewidth=2, label=f'Smoothed ({window_size} ep)')
        
        ax2.plot(episodes, episode_lengths, 'lightgreen', alpha=0.3, label='Raw Length')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_mission_dashboard(self, current_orbit, target_orbit, step, reward, total_reward):
        """Create a real-time mission dashboard"""
        
        fig = plt.figure(figsize=(15, 10))
        
        # 3D orbit plot (top left)
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2, projection='3d')
        self.plot_earth_3d(ax1)
        
        # Plot current position
        curr_pos = current_orbit.r.to(u.km).value
        ax1.scatter(curr_pos[0], curr_pos[1], curr_pos[2], 
                   color='red', s=100, label='Satellite')
        
        # Quick orbit traces
        try:
            # Generate a short trace of the orbit
            period_seconds = current_orbit.period.to(u.s).value
            trace_duration = min(period_seconds / 4, 3600)  # Quarter orbit or 1 hour max
            times = np.linspace(0, trace_duration, 25) * u.s
            curr_trace = []
            base_time = Time.now()
            
            for t in times:
                try:
                    pos = current_orbit.propagate(base_time + t).r.to(u.km).value
                    curr_trace.append(pos)
                except:
                    # If propagation fails, skip this point
                    continue
                    
            if len(curr_trace) > 1:
                curr_trace = np.array(curr_trace)
                ax1.plot(curr_trace[:, 0], curr_trace[:, 1], curr_trace[:, 2], 'r-', alpha=0.6, linewidth=1)
        except Exception as e:
            print(f"Orbit trace error: {e}")
            pass
            
        ax1.set_title('Current Mission Status')
        ax1.set_xlabel('X (km)')
        ax1.set_ylabel('Y (km)')
        ax1.set_zlabel('Z (km)')
        
        # Mission parameters (top right)
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        ax2.axis('off')
        
        # Calculate mission metrics
        altitude_current = (np.linalg.norm(curr_pos) - self.earth_radius)
        altitude_target = (np.linalg.norm(target_orbit.r.to(u.km).value) - self.earth_radius)
        altitude_error = abs(altitude_current - altitude_target)
        
        velocity_current = np.linalg.norm(current_orbit.v.to(u.km/u.s).value)
        velocity_target = np.linalg.norm(target_orbit.v.to(u.km/u.s).value)
        velocity_error = abs(velocity_current - velocity_target)
        
        mission_text = f"""
MISSION STATUS
Step: {step}
Current Reward: {reward:.2f}
Total Reward: {total_reward:.1f}

ORBITAL PARAMETERS
Current Alt: {altitude_current:.1f} km
Target Alt: {altitude_target:.1f} km
Alt Error: {altitude_error:.1f} km

Current Vel: {velocity_current:.2f} km/s
Target Vel: {velocity_target:.2f} km/s
Vel Error: {velocity_error:.3f} km/s

SUCCESS CRITERIA
Alt Error < 500 km
Vel Error < 0.1 km/s
        """
        
        ax2.text(0.05, 0.95, mission_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Altitude tracking (bottom left)
        ax3 = plt.subplot2grid((3, 3), (2, 0))
        ax3.axhline(y=altitude_target, color='g', linestyle='--', label='Target')
        ax3.scatter(step, altitude_current, color='red', s=50)
        ax3.set_ylabel('Altitude (km)')
        ax3.set_xlabel('Step')
        ax3.set_title('Altitude Tracking')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Velocity tracking (bottom center)
        ax4 = plt.subplot2grid((3, 3), (2, 1))
        ax4.axhline(y=velocity_target, color='g', linestyle='--', label='Target')
        ax4.scatter(step, velocity_current, color='red', s=50)
        ax4.set_ylabel('Velocity (km/s)')
        ax4.set_xlabel('Step')
        ax4.set_title('Velocity Tracking')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Reward tracking (bottom right)
        ax5 = plt.subplot2grid((3, 3), (2, 2))
        ax5.scatter(step, reward, color='blue', s=50)
        ax5.set_ylabel('Reward')
        ax5.set_xlabel('Step')
        ax5.set_title('Reward Progress')
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def demo_visualization():
    """Demonstrate the visualization capabilities"""
    
    print("=== Satellite Visualization Demo ===")
    
    # Create sample orbits
    current_orbit = Orbit.circular(Earth, alt=400 * u.km, inc=51.6 * u.deg)  # ISS-like
    target_orbit = Orbit.circular(Earth, alt=35786 * u.km, inc=0 * u.deg)   # GEO
    
    # Create visualizer
    viz = SatelliteVisualizer()
    
    # Show 3D orbit comparison
    print("Creating 3D orbit visualization...")
    fig1, ax1 = viz.visualize_orbits_3d(current_orbit, target_orbit, 
                                       "Mission: ISS to Geostationary Orbit")
    plt.show()
    
    # Show mission dashboard
    print("Creating mission dashboard...")
    fig2 = viz.create_mission_dashboard(current_orbit, target_orbit, 
                                       step=150, reward=-45.2, total_reward=-1250.7)
    plt.show()
    
    # Show sample training progress
    print("Creating training progress visualization...")
    # Generate sample training data
    episodes = 200
    rewards = -1000 + 800 * (1 - np.exp(-np.arange(episodes)/50)) + np.random.normal(0, 50, episodes)
    lengths = 1000 * np.exp(-np.arange(episodes)/100) + 100 + np.random.normal(0, 20, episodes)
    
    fig3 = viz.plot_training_progress(rewards, lengths)
    plt.show()
    
    print("Visualization demo completed!")


if __name__ == "__main__":
    print("Satellite Visualization Tools")
    print("=" * 40)
    
    choice = input("Choose demo:\n1. Visualization demo\n2. Test with environment\nEnter choice (1/2): ")
    
    if choice == "1":
        demo_visualization()
    elif choice == "2":
        print("Testing with satellite environment...")
        try:
            from satellite_env import SatelliteOrbitControlEnv
            
            env = SatelliteOrbitControlEnv()
            viz = SatelliteVisualizer()
            
            obs, _ = env.reset()
            
            # Show initial state
            fig = viz.create_mission_dashboard(
                env.current_orbit, env.target_orbit, 
                step=0, reward=0, total_reward=0
            )
            plt.show()
            
            # Take a few steps and show progress
            total_reward = 0
            for step in range(5):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                if step % 2 == 0:  # Show every other step
                    fig = viz.create_mission_dashboard(
                        env.current_orbit, env.target_orbit,
                        step=step+1, reward=reward, total_reward=total_reward
                    )
                    plt.show()
                
                if done:
                    break
            
            env.close()
            
        except ImportError:
            print("satellite_env.py not found. Run demo_visualization instead.")
            demo_visualization()
    else:
        demo_visualization()