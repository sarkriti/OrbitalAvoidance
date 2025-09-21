#!/usr/bin/env python3
"""
Simple Single-Plot Real-Time Satellite Visualization

This creates ONE plot window that updates to show:
- Satellite position moving in real-time
- Mission progress metrics
- Simple and stable visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from satellite_env import SatelliteOrbitControlEnv
import time

class SimpleSatelliteMonitor:
    """Simple real-time satellite mission monitor"""
    
    def __init__(self):
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('🛰️ Real-Time Satellite Mission Monitor', fontsize=14, fontweight='bold')
        
        # Flatten axes for easier access
        self.ax_3d = self.fig.add_subplot(2, 2, 1, projection='3d')
        self.ax_alt = self.axes[0, 1]
        self.ax_reward = self.axes[1, 0] 
        self.ax_info = self.axes[1, 1]
        
        # Remove the default subplot that's now replaced by 3D
        self.axes[0, 0].remove()
        
        # Data storage
        self.steps = []
        self.altitudes = []
        self.rewards = []
        self.positions = []
        self.target_altitude = None
        
        # Setup plots
        self.setup_plots()
        plt.ion()  # Interactive mode
        
    def setup_plots(self):
        """Initialize plot settings"""
        
        # 3D plot setup
        self.ax_3d.set_xlabel('X (km)')
        self.ax_3d.set_ylabel('Y (km)')
        self.ax_3d.set_zlabel('Z (km)')
        self.ax_3d.set_title('Satellite Position')
        
        # Altitude plot
        self.ax_alt.set_xlabel('Step')
        self.ax_alt.set_ylabel('Altitude (km)')
        self.ax_alt.set_title('Altitude Progress')
        self.ax_alt.grid(True, alpha=0.3)
        
        # Reward plot
        self.ax_reward.set_xlabel('Step')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.set_title('Mission Score')
        self.ax_reward.grid(True, alpha=0.3)
        
        # Info display
        self.ax_info.axis('off')
        self.ax_info.set_title('Mission Status')
        
        plt.tight_layout()
    
    def update(self, env, step, reward, total_reward):
        """Update all displays with current mission state"""
        
        try:
            # Get current data
            curr_pos = env.current_orbit.r.to('km').value
            curr_alt = np.linalg.norm(curr_pos) - 6371  # Earth radius
            
            if self.target_altitude is None:
                tgt_pos = env.target_orbit.r.to('km').value
                self.target_altitude = np.linalg.norm(tgt_pos) - 6371
            
            # Store data
            self.steps.append(step)
            self.altitudes.append(curr_alt)
            self.rewards.append(reward)
            self.positions.append(curr_pos.copy())
            
            # Update 3D view
            self.update_3d_view(env)
            
            # Update altitude plot
            self.update_altitude_plot()
            
            # Update reward plot
            self.update_reward_plot()
            
            # Update info display
            self.update_info(env, step, reward, total_reward)
            
            # Refresh display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"Update error: {e}")
    
    def update_3d_view(self, env):
        """Update 3D satellite view"""
        
        self.ax_3d.clear()
        
        # Draw simple Earth
        earth_radius = 6371
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
        y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
        z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.ax_3d.plot_surface(x_earth, y_earth, z_earth, alpha=0.3, color='lightblue')
        
        # Current and target positions
        curr_pos = env.current_orbit.r.to('km').value
        tgt_pos = env.target_orbit.r.to('km').value
        
        # Plot positions
        self.ax_3d.scatter(curr_pos[0], curr_pos[1], curr_pos[2], 
                          color='red', s=100, label='Satellite')
        self.ax_3d.scatter(tgt_pos[0], tgt_pos[1], tgt_pos[2], 
                          color='green', s=100, label='Target')
        
        # Trail of recent positions
        if len(self.positions) > 1:
            trail = np.array(self.positions[-5:])  # Last 5 positions
            self.ax_3d.plot(trail[:, 0], trail[:, 1], trail[:, 2], 
                           'orange', alpha=0.7, linewidth=2, label='Trail')
        
        # Set equal view
        curr_r = np.linalg.norm(curr_pos)
        tgt_r = np.linalg.norm(tgt_pos)
        max_r = max(curr_r, tgt_r) * 1.1
        
        self.ax_3d.set_xlim([-max_r, max_r])
        self.ax_3d.set_ylim([-max_r, max_r])
        self.ax_3d.set_zlim([-max_r, max_r])
        
        self.ax_3d.set_xlabel('X (km)')
        self.ax_3d.set_ylabel('Y (km)')
        self.ax_3d.set_zlabel('Z (km)')
        self.ax_3d.set_title('🛰️ Satellite Position')
        self.ax_3d.legend()
    
    def update_altitude_plot(self):
        """Update altitude tracking plot"""
        
        self.ax_alt.clear()
        
        # Plot altitude progress
        self.ax_alt.plot(self.steps, self.altitudes, 'b-', linewidth=2, label='Current')
        
        # Target line
        if self.target_altitude is not None:
            self.ax_alt.axhline(y=self.target_altitude, color='g', 
                               linestyle='--', linewidth=2, label='Target')
        
        self.ax_alt.set_xlabel('Step')
        self.ax_alt.set_ylabel('Altitude (km)')
        self.ax_alt.set_title('Altitude Progress')
        self.ax_alt.grid(True, alpha=0.3)
        self.ax_alt.legend()
    
    def update_reward_plot(self):
        """Update reward tracking plot"""
        
        self.ax_reward.clear()
        
        # Plot rewards
        self.ax_reward.plot(self.steps, self.rewards, 'purple', linewidth=2)
        
        # Add trend line if enough data
        if len(self.steps) > 10:
            z = np.polyfit(self.steps, self.rewards, 1)
            p = np.poly1d(z)
            self.ax_reward.plot(self.steps, p(self.steps), 'red', 
                               linestyle='--', alpha=0.7, label='Trend')
            self.ax_reward.legend()
        
        self.ax_reward.set_xlabel('Step')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.set_title('Mission Score')
        self.ax_reward.grid(True, alpha=0.3)
    
    def update_info(self, env, step, reward, total_reward):
        """Update mission information display"""
        
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Calculate mission metrics
        curr_pos = env.current_orbit.r.to('km').value
        tgt_pos = env.target_orbit.r.to('km').value
        
        curr_alt = np.linalg.norm(curr_pos) - 6371
        tgt_alt = np.linalg.norm(tgt_pos) - 6371
        alt_error = abs(curr_alt - tgt_alt)
        
        # Mission status
        alt_success = alt_error < 500  # km tolerance
        progress = max(0, min(100, (1 - alt_error/tgt_alt) * 100))
        
        info_text = f"""MISSION STATUS
Step: {step}
Current Reward: {reward:.1f}
Total Score: {total_reward:.1f}

ALTITUDE
Current: {curr_alt:.0f} km
Target: {tgt_alt:.0f} km
Error: {alt_error:.0f} km

PROGRESS
Mission: {progress:.1f}% complete
Status: {'🎯 SUCCESS!' if alt_success else '🚀 In Progress'}

CONTROLS
Close window to stop"""
        
        color = 'lightgreen' if alt_success else 'lightblue'
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))


def run_simple_demo():
    """Run simple real-time demo"""
    
    print("🛰️ Simple Real-Time Satellite Monitor")
    print("=" * 50)
    print("ONE window will show satellite mission in real-time!")
    print("Close the window to stop the demo.\n")
    
    # Create environment and monitor
    env = SatelliteOrbitControlEnv()
    monitor = SimpleSatelliteMonitor()
    
    obs, _ = env.reset()
    total_reward = 0
    
    print(f"Mission: {env.current_orbit.a.to('km'):.0f} → {env.target_orbit.a.to('km'):.0f}")
    print("Watch the satellite move in real-time...\n")
    
    try:
        for step in range(200):
            # Take action
            action = env.action_space.sample() * 0.2  # Smaller actions for smoother demo
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Update monitor
            monitor.update(env, step+1, reward, total_reward)
            
            print(f"Step {step+1:3d}: Alt={np.linalg.norm(env.current_orbit.r.to('km').value)-6371:6.0f}km, "
                  f"Reward={reward:6.1f}, Total={total_reward:8.1f}", end='\r')
            
            # Check if window is still open
            if not plt.fignum_exists(monitor.fig.number):
                print("\nWindow closed - stopping demo")
                break
            
            if done:
                print(f"\n🎯 Mission completed in {step+1} steps!")
                input("Press Enter to continue or close window to exit...")
                
            # Control update speed
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Demo stopped")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    finally:
        env.close()
        plt.ioff()
        print("Demo ended. Close window when done viewing.")
        plt.show()


if __name__ == "__main__":
    run_simple_demo()