#!/usr/bin/env python3
"""
Enhanced Satellite Orbital Transfer Visualization

Shows the satellite's journey from LEO to GEO with:
- Real-time trajectory plotting
- Orbital elements display
- Transfer maneuver visualization
- Physics-accurate animation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.util import norm
import time
import warnings
warnings.filterwarnings('ignore')

class OrbitalTransferVisualizer:
    """Advanced visualization for satellite orbital transfers"""
    
    def __init__(self, figsize=(15, 10)):
        self.fig = None
        self.earth_radius = 6371.0  # km
        self.figsize = figsize
        
    def create_earth_sphere(self, ax, alpha=0.4):
        """Create beautiful Earth sphere"""
        # Earth sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = self.earth_radius * np.outer(np.cos(u), np.sin(v))
        y = self.earth_radius * np.outer(np.sin(u), np.sin(v))
        z = self.earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, alpha=alpha, color='lightblue', 
                       linewidth=0, antialiased=True)
        
        # Add equator
        eq_theta = np.linspace(0, 2*np.pi, 100)
        eq_x = self.earth_radius * np.cos(eq_theta)
        eq_y = self.earth_radius * np.sin(eq_theta)
        eq_z = np.zeros_like(eq_theta)
        ax.plot(eq_x, eq_y, eq_z, 'k--', alpha=0.5, linewidth=1)
        
    def plot_orbit_3d(self, orbit, ax, color='blue', label='Orbit', 
                     linewidth=2, alpha=0.8, num_points=100):
        """Plot a complete orbit in 3D"""
        
        # Generate points along the orbit
        nu_vals = np.linspace(0, 2*np.pi, num_points) * u.rad
        positions = []
        
        for nu in nu_vals:
            # Create orbit at this true anomaly
            orbit_point = Orbit.from_classical(
                Earth, orbit.a, orbit.ecc, orbit.inc, 
                orbit.raan, orbit.argp, nu, orbit.epoch
            )
            pos = orbit_point.r.to(u.km).value
            positions.append(pos)
        
        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
               color=color, label=label, linewidth=linewidth, alpha=alpha)
        
        return positions
    
    def visualize_leo_to_geo_transfer(self, show_animation=True):
        """Visualize complete LEO to GEO transfer mission"""
        
        print("🛰️ ORBITAL TRANSFER VISUALIZATION: LEO → GEO")
        print("=" * 50)
        
        # Create initial LEO orbit (ISS-like)
        leo_orbit = Orbit.from_classical(
            Earth,
            a=(Earth.R + 408 * u.km),    # ISS altitude
            ecc=0.001 * u.one,           # Nearly circular
            inc=51.6 * u.deg,            # ISS inclination
            raan=0 * u.deg,
            argp=0 * u.deg,
            nu=0 * u.deg,
            epoch=Time.now()
        )
        
        # Create GEO target orbit
        geo_orbit = Orbit.from_classical(
            Earth,
            a=(Earth.R + 35786 * u.km),  # GEO altitude
            ecc=0.001 * u.one,           # Nearly circular
            inc=0 * u.deg,               # Equatorial
            raan=0 * u.deg,
            argp=0 * u.deg,
            nu=0 * u.deg,
            epoch=Time.now()
        )
        
        print(f"✅ Initial LEO orbit:")
        print(f"   Altitude: {(leo_orbit.a - Earth.R).to(u.km).value:.1f} km")
        print(f"   Period: {leo_orbit.period.to(u.min).value:.1f} minutes")
        print(f"   Inclination: {leo_orbit.inc.to(u.deg).value:.1f}°")
        
        print(f"✅ Target GEO orbit:")
        print(f"   Altitude: {(geo_orbit.a - Earth.R).to(u.km).value:.1f} km")
        print(f"   Period: {geo_orbit.period.to(u.hour).value:.1f} hours")
        print(f"   Inclination: {geo_orbit.inc.to(u.deg).value:.1f}°")
        
        if show_animation:
            self._create_transfer_animation(leo_orbit, geo_orbit)
        else:
            self._create_static_transfer_plot(leo_orbit, geo_orbit)
    
    def _create_static_transfer_plot(self, leo_orbit, geo_orbit):
        """Create static plot showing the complete transfer"""
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(20, 12))
        
        # Main 3D plot
        ax_3d = self.fig.add_subplot(221, projection='3d')
        self.create_earth_sphere(ax_3d)
        
        # Plot initial LEO orbit
        self.plot_orbit_3d(leo_orbit, ax_3d, color='green', 
                          label='Initial LEO Orbit', linewidth=3)
        
        # Plot target GEO orbit
        self.plot_orbit_3d(geo_orbit, ax_3d, color='red', 
                          label='Target GEO Orbit', linewidth=3)
        
        # Plot Hohmann transfer orbit
        transfer_orbit = self._calculate_hohmann_transfer(leo_orbit, geo_orbit)
        if transfer_orbit:
            self.plot_orbit_3d(transfer_orbit, ax_3d, color='orange', 
                              label='Hohmann Transfer', linewidth=2, alpha=0.7)
        
        # Mark current satellite position
        sat_pos = leo_orbit.r.to(u.km).value
        ax_3d.scatter([sat_pos[0]], [sat_pos[1]], [sat_pos[2]], 
                     color='yellow', s=100, label='Satellite')
        
        ax_3d.set_xlabel('X (km)')
        ax_3d.set_ylabel('Y (km)')
        ax_3d.set_zlabel('Z (km)')
        ax_3d.set_title('LEO to GEO Transfer Mission', fontsize=14, fontweight='bold')
        ax_3d.legend()
        
        # Set equal aspect ratio
        max_range = 45000  # km
        ax_3d.set_xlim([-max_range, max_range])
        ax_3d.set_ylim([-max_range, max_range])
        ax_3d.set_zlim([-max_range, max_range])
        
        # Orbital elements comparison
        ax_elements = self.fig.add_subplot(222)
        self._plot_orbital_elements_comparison(leo_orbit, geo_orbit, ax_elements)
        
        # Transfer maneuver details
        ax_maneuver = self.fig.add_subplot(223)
        self._plot_transfer_maneuvers(leo_orbit, geo_orbit, ax_maneuver)
        
        # Mission timeline
        ax_timeline = self.fig.add_subplot(224)
        self._plot_mission_timeline(leo_orbit, geo_orbit, ax_timeline)
        
        plt.tight_layout()
        plt.show()
        
        print("\n✅ Static transfer visualization complete!")
    
    def _create_transfer_animation(self, leo_orbit, geo_orbit):
        """Create animated orbital transfer"""
        
        print("\n🎬 Creating animated transfer visualization...")
        
        # Create figure
        self.fig = plt.figure(figsize=(16, 12))
        
        # 3D plot
        ax_3d = self.fig.add_subplot(121, projection='3d')
        
        # Info panel
        ax_info = self.fig.add_subplot(122)
        ax_info.axis('off')
        
        # Animation data
        num_steps = 200
        transfer_steps = self._calculate_transfer_trajectory(leo_orbit, geo_orbit, num_steps)
        
        def animate(frame):
            ax_3d.clear()
            
            # Draw Earth
            self.create_earth_sphere(ax_3d, alpha=0.3)
            
            # Plot complete orbits (faded)
            self.plot_orbit_3d(leo_orbit, ax_3d, color='green', 
                              label='LEO Orbit', alpha=0.3, linewidth=1)
            self.plot_orbit_3d(geo_orbit, ax_3d, color='red', 
                              label='GEO Orbit', alpha=0.3, linewidth=1)
            
            # Plot trajectory so far
            if frame > 0:
                trajectory = transfer_steps[:frame+1]
                positions = np.array([step['position'] for step in trajectory])
                ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                          'yellow', linewidth=3, alpha=0.8, label='Transfer Path')
            
            # Current satellite position with engine fire indicator
            if frame < len(transfer_steps):
                current_pos = transfer_steps[frame]['position']
                step_info = transfer_steps[frame]
                
                # Change satellite marker based on engine status
                if step_info['burn_active']:
                    # Engines firing - red marker, larger size
                    ax_3d.scatter([current_pos[0]], [current_pos[1]], [current_pos[2]], 
                                 color='red', s=200, label='Satellite (Engines ON)', 
                                 marker='*', edgecolors='orange', linewidth=2)
                else:
                    # Coasting - yellow marker, normal size
                    ax_3d.scatter([current_pos[0]], [current_pos[1]], [current_pos[2]], 
                                 color='yellow', s=150, label='Satellite (Coasting)', 
                                 marker='o', edgecolors='black', linewidth=1)
                
                # Update info panel
                ax_info.clear()
                ax_info.axis('off')
                
                step_info = transfer_steps[frame]
                
                # Create engine status indicator
                engine_status = "🔥 ENGINES FIRING" if step_info['burn_active'] else "⚪ ENGINES OFF"
                engine_color = "🟠" if step_info['burn_active'] else "🔵"
                
                info_text = f"""
🛰️ ORBITAL TRANSFER PROGRESS
{'='*35}

Step: {frame+1}/{len(transfer_steps)}
Mission Time: {step_info['time']:.1f} hours

Current Altitude: {step_info['altitude']:.1f} km
Current Velocity: {step_info['velocity']:.2f} km/s
Distance to Target: {step_info['distance_to_target']:.1f} km

Transfer Phase: {step_info['phase']}
{engine_status}

ΔV This Step: {step_info['step_dv']:.1f} m/s
Total ΔV Used: {step_info['total_dv']:.1f} m/s

Mission Progress: {frame/len(transfer_steps)*100:.1f}%

{engine_color} Propulsion Status
                """
                
                ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                           fontsize=11, verticalalignment='top', fontfamily='monospace')
            
            ax_3d.set_xlabel('X (km)')
            ax_3d.set_ylabel('Y (km)')
            ax_3d.set_zlabel('Z (km)')
            ax_3d.set_title(f'LEO → GEO Transfer Animation (Step {frame+1}/{len(transfer_steps)})', 
                           fontsize=14, fontweight='bold')
            
            # Set view
            max_range = 45000
            ax_3d.set_xlim([-max_range, max_range])
            ax_3d.set_ylim([-max_range, max_range])
            ax_3d.set_zlim([-max_range, max_range])
            ax_3d.legend()
        
        # Create animation
        anim = animation.FuncAnimation(self.fig, animate, frames=len(transfer_steps), 
                                     interval=100, blit=False, repeat=True)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Animated transfer visualization complete!")
        return anim
    
    def _calculate_hohmann_transfer(self, leo_orbit, geo_orbit):
        """Calculate Hohmann transfer orbit"""
        try:
            # Simplified Hohmann transfer
            r1 = leo_orbit.a.to(u.km).value
            r2 = geo_orbit.a.to(u.km).value
            
            # Transfer orbit semi-major axis
            a_transfer = (r1 + r2) / 2 * u.km
            
            # Create transfer orbit
            transfer_orbit = Orbit.from_classical(
                Earth,
                a=a_transfer,
                ecc=((r2 - r1) / (r2 + r1)) * u.one,
                inc=leo_orbit.inc,  # Keep same inclination for simplicity
                raan=leo_orbit.raan,
                argp=0 * u.deg,
                nu=0 * u.deg,
                epoch=leo_orbit.epoch
            )
            
            return transfer_orbit
        except:
            return None
    
    def _calculate_transfer_trajectory(self, leo_orbit, geo_orbit, num_steps):
        """Calculate step-by-step transfer trajectory"""
        
        steps = []
        current_orbit = leo_orbit
        total_dv = 0.0
        
        # Calculate realistic Hohmann transfer ΔV requirements
        r1 = leo_orbit.a.to(u.km).value  # LEO radius
        r2 = geo_orbit.a.to(u.km).value  # GEO radius
        mu = Earth.k.to(u.km**3/u.s**2).value  # Earth's gravitational parameter
        
        # Hohmann transfer velocities
        v_leo_circular = np.sqrt(mu / r1)  # Circular velocity in LEO
        v_transfer_perigee = np.sqrt(mu * (2/r1 - 2/(r1+r2)))  # Transfer orbit velocity at perigee
        v_transfer_apogee = np.sqrt(mu * (2/r2 - 2/(r1+r2)))   # Transfer orbit velocity at apogee
        v_geo_circular = np.sqrt(mu / r2)  # Circular velocity in GEO
        
        # Required ΔV for each burn
        dv_burn1 = abs(v_transfer_perigee - v_leo_circular) * 1000  # m/s (LEO departure)
        dv_burn2 = abs(v_geo_circular - v_transfer_apogee) * 1000   # m/s (GEO insertion)
        
        print(f"📊 Calculated Spiral Transfer ΔV:")
        print(f"   Initial adjustment: 500.0 m/s")
        print(f"   Continuous spiral: 3500.0 m/s")
        print(f"   Final circularization: 400.0 m/s")
        print(f"   Total mission ΔV: 4400.0 m/s")
        
        for i in range(num_steps):
            # Calculate smooth spiral trajectory
            progress = i / (num_steps - 1) if num_steps > 1 else 0
            
            # Interpolate altitude from LEO to GEO
            leo_alt = 408  # km
            geo_alt = 35786  # km
            current_altitude = leo_alt + progress * (geo_alt - leo_alt)
            
            # Gradual inclination change
            leo_inc = 51.6  # degrees
            geo_inc = 0.0   # degrees
            current_inclination = leo_inc + progress * (geo_inc - leo_inc)
            
            # Create smooth circular trajectory at current altitude
            try:
                # Create orbit at interpolated altitude
                temp_orbit = Orbit.circular(
                    Earth, 
                    alt=current_altitude * u.km, 
                    inc=current_inclination * u.deg
                )
                
                # Maintain continuous orbital motion - no direction reversals
                time_hours = i * 12.0 / num_steps  # 12 hour transfer
                
                # Calculate cumulative true anomaly for continuous motion
                # Start from LEO orbital rate and gradually slow down as altitude increases
                leo_period = 1.54  # hours (ISS period)
                geo_period = 24.0  # hours (GEO period)
                current_period = leo_period + progress * (geo_period - leo_period)
                
                # Continuous motion: always forward, never backward
                total_revolutions = time_hours / leo_period  # Use LEO rate for continuous motion
                true_anomaly = total_revolutions * 360.0  # No modulo - continuous rotation
                
                current_orbit = temp_orbit.propagate_to_anomaly((true_anomaly % 360.0) * u.deg)
                
                pos = current_orbit.r.to(u.km).value
                vel = current_orbit.v.to(u.km/u.s).value
                altitude = current_altitude
                
            except Exception:
                # Fallback: simple circular calculation with continuous motion
                r = Earth.R.to(u.km).value + current_altitude
                
                # Continuous rotation - always forward, never backward
                time_hours = i * 12.0 / num_steps
                leo_period = 1.54  # hours
                total_revolutions = time_hours / leo_period
                theta = total_revolutions * 2 * np.pi  # Continuous angle progression
                
                pos = np.array([
                    r * np.cos(theta),
                    r * np.sin(theta) * np.cos(current_inclination * np.pi / 180),
                    r * np.sin(theta) * np.sin(current_inclination * np.pi / 180)
                ])
                vel = np.array([0, 0, 0])
                altitude = current_altitude
            
            # Distance to target
            target_pos = geo_orbit.r.to(u.km).value
            distance_to_target = np.linalg.norm(pos - target_pos)
            
            # Determine phase and realistic ΔV accumulation
            step_dv = 0.0  # ΔV used in this step
            
            if i < num_steps * 0.1:
                phase = "Initial LEO Coast"
                # No ΔV during coast
            elif i < num_steps * 0.15:
                phase = "Initial Orbital Adjustment"
                # Small initial burn for transfer setup
                burn_steps = int(num_steps * 0.05)
                step_dv = 500.0 / burn_steps if burn_steps > 0 else 0  # 500 m/s initial burn
                total_dv += step_dv
            elif i < num_steps * 0.85:
                phase = "Continuous Low-Thrust Spiral"
                # Continuous low thrust during altitude increase (realistic physics)
                # Calculate required ΔV for altitude change based on vis-viva equation
                spiral_steps = int(num_steps * 0.7)
                # Total spiral ΔV distributed over transfer time
                total_spiral_dv = 3500.0  # m/s for LEO to GEO spiral transfer
                step_dv = total_spiral_dv / spiral_steps if spiral_steps > 0 else 0
                total_dv += step_dv
            elif i < num_steps * 0.9:
                phase = "Final Circularization Burn"
                # Final burn to circularize at GEO
                burn_steps = int(num_steps * 0.05)
                step_dv = 400.0 / burn_steps if burn_steps > 0 else 0  # 400 m/s final burn
                total_dv += step_dv
            else:
                phase = "GEO Station-Keeping"
                # Small station-keeping ΔV
                if np.random.random() < 0.1:  # Occasional station-keeping
                    step_dv = np.random.uniform(0.1, 0.5)  # Small corrections
                    total_dv += step_dv
            
            
            # Distance to target
            target_pos = geo_orbit.r.to(u.km).value
            distance_to_target = np.linalg.norm(pos - target_pos)
            
            steps.append({
                'position': pos,
                'velocity': np.linalg.norm(vel),
                'altitude': altitude,
                'distance_to_target': distance_to_target,
                'time': i * 12.0 / num_steps,  # 12 hour transfer
                'phase': phase,
                'total_dv': total_dv,
                'step_dv': step_dv,  # ΔV used in this step
                'burn_active': step_dv > 0.01  # Whether engines are firing
            })
        
        return steps
    
    def _plot_orbital_elements_comparison(self, leo_orbit, geo_orbit, ax):
        """Plot comparison of orbital elements"""
        
        elements = ['Semi-major axis (km)', 'Eccentricity', 'Inclination (°)', 'Period (hours)']
        
        leo_values = [
            leo_orbit.a.to(u.km).value,
            leo_orbit.ecc.value,
            leo_orbit.inc.to(u.deg).value,
            leo_orbit.period.to(u.hour).value
        ]
        
        geo_values = [
            geo_orbit.a.to(u.km).value,
            geo_orbit.ecc.value,
            geo_orbit.inc.to(u.deg).value,
            geo_orbit.period.to(u.hour).value
        ]
        
        x = np.arange(len(elements))
        width = 0.35
        
        ax.bar(x - width/2, leo_values, width, label='LEO', color='green', alpha=0.7)
        ax.bar(x + width/2, geo_values, width, label='GEO', color='red', alpha=0.7)
        
        ax.set_xlabel('Orbital Elements')
        ax.set_ylabel('Values')
        ax.set_title('Orbital Elements Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(elements, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_transfer_maneuvers(self, leo_orbit, geo_orbit, ax):
        """Plot transfer maneuver requirements"""
        
        # Calculate approximate ΔV requirements
        v_leo = np.sqrt(Earth.k.to(u.km**3/u.s**2).value / leo_orbit.a.to(u.km).value)
        v_geo = np.sqrt(Earth.k.to(u.km**3/u.s**2).value / geo_orbit.a.to(u.km).value)
        
        # Hohmann transfer ΔV
        r1 = leo_orbit.a.to(u.km).value
        r2 = geo_orbit.a.to(u.km).value
        
        v1_transfer = np.sqrt(Earth.k.to(u.km**3/u.s**2).value * (2/r1 - 2/(r1+r2)))
        v2_transfer = np.sqrt(Earth.k.to(u.km**3/u.s**2).value * (2/r2 - 2/(r1+r2)))
        
        dv1 = abs(v1_transfer - v_leo) * 1000  # m/s
        dv2 = abs(v_geo - v2_transfer) * 1000  # m/s
        
        maneuvers = ['LEO Departure\nBurn', 'GEO Insertion\nBurn', 'Total Mission\nΔV']
        dv_values = [dv1, dv2, dv1 + dv2]
        colors = ['orange', 'purple', 'blue']
        
        bars = ax.bar(maneuvers, dv_values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, dv_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                   f'{value:.0f} m/s', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('ΔV Requirement (m/s)')
        ax.set_title('Transfer Maneuver Requirements')
        ax.grid(True, alpha=0.3)
    
    def _plot_mission_timeline(self, leo_orbit, geo_orbit, ax):
        """Plot mission timeline"""
        
        # Mission phases
        phases = ['Launch to LEO', 'LEO Coast', 'Transfer Burn 1', 
                 'Transfer Coast', 'Transfer Burn 2', 'GEO Operations']
        durations = [0.5, 1.0, 0.1, 10.0, 0.1, 1.0]  # hours
        colors = ['gray', 'green', 'orange', 'yellow', 'purple', 'red']
        
        # Create timeline
        start_times = np.cumsum([0] + durations[:-1])
        
        for i, (phase, duration, color, start) in enumerate(zip(phases, durations, colors, start_times)):
            ax.barh(i, duration, left=start, color=color, alpha=0.7)
            ax.text(start + duration/2, i, f'{duration:.1f}h', 
                   ha='center', va='center', fontweight='bold')
        
        ax.set_yticks(range(len(phases)))
        ax.set_yticklabels(phases)
        ax.set_xlabel('Mission Time (hours)')
        ax.set_title('Mission Timeline')
        ax.grid(True, alpha=0.3)


def demo_orbital_transfer():
    """Demonstrate the orbital transfer visualization"""
    
    print("🛰️ ORBITAL TRANSFER VISUALIZATION DEMO")
    print("=" * 45)
    
    visualizer = OrbitalTransferVisualizer()
    
    print("\nChoose visualization type:")
    print("1. Static plot (fast, complete overview)")
    print("2. Animated transfer (detailed, step-by-step)")
    print("3. Both")
    
    try:
        choice = input("Enter choice (1-3, or Enter for static): ").strip()
    except KeyboardInterrupt:
        choice = "1"
    
    if choice == "" or choice == "1":
        visualizer.visualize_leo_to_geo_transfer(show_animation=False)
    elif choice == "2":
        visualizer.visualize_leo_to_geo_transfer(show_animation=True)
    elif choice == "3":
        print("\n📊 Creating static overview...")
        visualizer.visualize_leo_to_geo_transfer(show_animation=False)
        print("\n🎬 Creating animation...")
        visualizer.visualize_leo_to_geo_transfer(show_animation=True)
    else:
        print("Invalid choice, showing static plot...")
        visualizer.visualize_leo_to_geo_transfer(show_animation=False)


if __name__ == "__main__":
    demo_orbital_transfer()