#!/usr/bin/env python3
"""
Enhanced Web Simulator with Full RL Integration

Features:
1. Full RL model control of satellite
2. JSON-based configuration for all parameters
3. Orekit physics for all objects
4. Real-time model switching and visualization

Usage:
    python enhanced_web_simulator.py [--config scenario.json] [--port 8080]
"""

import os
import sys
import json
import time
import math
import random
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
from pathlib import Path
import numpy as np

# Add project path
if __package__ in (None, ""):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Import RL and physics modules
try:
    from stable_baselines3 import PPO, SAC, TD3
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("Warning: stable-baselines3 not available. RL agents disabled.")

from OrbitalAvoidance.rl_training_env import create_rl_environment
from OrbitalAvoidance.orekit_physics import (
    create_propagator, OrbitElements, convert_to_orbit_elements
)
import numpy as np


class SimpleRLController:
    """Simple RL controller for the enhanced web simulator."""
    
    def __init__(self):
        self.control_mode = 'zero'  # 'zero', 'manual', 'rl'
        self.rl_model = None
        self.manual_thrust = np.array([0.0, 0.0, 0.0])
        self.available_models = self._scan_for_models()
        self.model_path = None
        
    def _scan_for_models(self):
        """Scan for available trained models."""
        models = {}
        
        # Look in the correct trained_models directory
        model_dirs = [
            './trained_models',
            './trained_models/debris_avoidance_ppo',
            './trained_models/minimal_test_ppo'
        ]
        
        for model_dir in model_dirs:
            model_path = Path(model_dir)
            if model_path.exists():
                for file_path in model_path.rglob('*.zip'):
                    if 'model' in file_path.name:
                        model_name = file_path.name
                        models[model_name] = str(file_path)
                        print(f"📦 Found model: {model_name}")
        
        print(f"✅ Found {len(models)} trained models")
        return models
    
    def load_rl_model(self, model_path):
        """Load an RL model."""
        if not STABLE_BASELINES_AVAILABLE:
            print("❌ stable-baselines3 not available")
            return False
            
        try:
            print(f"🤖 Loading RL model: {model_path}")
            self.rl_model = PPO.load(model_path)
            self.model_path = model_path
            self.control_mode = 'rl'
            print(f"✅ RL model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading RL model: {e}")
            return False
    
    def set_control_mode(self, mode):
        """Set control mode."""
        if mode in ['zero', 'manual', 'rl']:
            self.control_mode = mode
            print(f"🎮 Control mode set to: {mode}")
            return True
        return False
    
    def set_manual_thrust(self, thrust_x, thrust_y, thrust_z):
        """Set manual thrust."""
        self.manual_thrust = np.array([thrust_x, thrust_y, thrust_z])
        self.manual_thrust = np.clip(self.manual_thrust, -1.0, 1.0)
    
    def get_action(self, satellite_state, debris_states):
        """Get action based on control mode."""
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
            
        elif self.control_mode == 'rl' and self.rl_model is not None:
            # Create RL observation from simulator state
            obs = self._create_rl_observation(satellite_state, debris_states)
            action, _ = self.rl_model.predict(obs, deterministic=True)
            action_info['thrust'] = action.tolist()
            action_info['source'] = 'rl_model'
            action_info['model_name'] = os.path.basename(self.model_path) if self.model_path else None
            
        return action_info
    
    def _create_rl_observation(self, satellite_state, debris_states):
        """Create RL observation from web simulator state."""
        obs = []
        
        # Satellite position and velocity (6 elements)
        sat_pos = satellite_state.get('position', [0, 0, 0])
        sat_vel = satellite_state.get('velocity', [0, 0, 0])
        obs.extend(sat_pos)
        obs.extend(sat_vel)
        
        # Mission info (2 elements)
        fuel_remaining = satellite_state.get('fuel_remaining_percent', 100) / 100.0
        time_remaining = 1.0  # Simplified
        obs.extend([fuel_remaining, time_remaining])
        
        # Debris relative positions and velocities (up to 10 debris, 8 elements each)
        sat_pos_array = np.array(sat_pos)
        sat_vel_array = np.array(sat_vel)
        
        for i in range(min(10, len(debris_states))):
            debris = debris_states[i]
            debris_pos = np.array(debris.get('position', [0, 0, 0]))
            debris_vel = np.array(debris.get('velocity', [0, 0, 0]))
            
            # Relative position and velocity
            rel_pos = debris_pos - sat_pos_array
            rel_vel = debris_vel - sat_vel_array
            
            obs.extend(rel_pos.tolist())
            obs.extend(rel_vel.tolist())
            
            # Distance and collision time
            distance = np.linalg.norm(rel_pos)
            rel_speed = np.linalg.norm(rel_vel)
            time_to_collision = distance / max(rel_speed, 1e-6)
            
            obs.extend([distance, time_to_collision])
        
        # Pad with zeros if fewer than 10 debris
        while len(debris_states) < 10:
            obs.extend([0.0] * 8)
            debris_states.append({'position': [0, 0, 0], 'velocity': [0, 0, 0]})
        
        return np.array(obs, dtype=np.float32)
    
    def get_status(self):
        """Get controller status."""
        return {
            'control_mode': self.control_mode,
            'available_models': list(self.available_models.keys()),
            'rl_agent_loaded': self.rl_model is not None,
            'manual_thrust': self.manual_thrust.tolist(),
            'stable_baselines_available': STABLE_BASELINES_AVAILABLE,
            'rl_model_info': {
                'algorithm': 'PPO',
                'model_path': self.model_path
            } if self.rl_model else None
        }


class OrbitingObject:
    """Individual orbiting object with high-precision Orekit-based orbital mechanics."""
    
    def __init__(self, name, altitude_km, inclination_deg=0, eccentricity=0.0, 
                 mean_anomaly_deg=0, omega_deg=0, raan_deg=0):
        self.name = name
        self.altitude_km = altitude_km
        self.inclination_deg = inclination_deg
        self.eccentricity = eccentricity
        self.mean_anomaly_deg = mean_anomaly_deg
        self.omega_deg = omega_deg  # Argument of periapsis
        self.raan_deg = raan_deg    # Right ascension of ascending node
        
        # Create Orekit propagator (STRICT MODE - NO FALLBACKS)
        print(f"🚀 Creating {name} with Orekit backend...")
        self.propagator = create_propagator()
        self.orbit_elements = convert_to_orbit_elements(
            altitude_km=altitude_km,
            inclination_deg=inclination_deg,
            eccentricity=eccentricity,
            mean_anomaly_deg=mean_anomaly_deg,
            arg_periapsis_deg=omega_deg,
            raan_deg=raan_deg
        )
        
        # Initialize position and velocity
        initial_position, initial_velocity = self.propagator.propagate_orbit(
            self.orbit_elements, 0.0
        )
        
        self.position_x = initial_position[0]
        self.position_y = initial_position[1] 
        self.position_z = initial_position[2]
        self.velocity_x = initial_velocity[0]
        self.velocity_y = initial_velocity[1]
        self.velocity_z = initial_velocity[2]
        
        # Fuel and thrust properties (for controllable satellites)
        self.fuel_remaining_kg = 100.0
        self.initial_fuel_kg = 100.0  # Track initial fuel for percentage calculation
        self.max_thrust_n = 1.0
        self.is_controllable = name.lower() == "satellite"
        
        print(f"✅ {name} initialized at altitude {altitude_km} km")
    
    def update_position(self, simulation_time_s):
        """Update position using high-precision Orekit orbital mechanics."""
        try:
            new_position, new_velocity = self.propagator.propagate_orbit(
                self.orbit_elements, simulation_time_s
            )
            
            self.position_x = new_position[0]
            self.position_y = new_position[1] 
            self.position_z = new_position[2]
            self.velocity_x = new_velocity[0]
            self.velocity_y = new_velocity[1]
            self.velocity_z = new_velocity[2]
            
        except Exception as e:
            print(f"⚠️ Warning: Orekit propagation failed for {self.name}: {e}")
            # Keep current position if propagation fails
    
    def apply_thrust(self, thrust_x, thrust_y, thrust_z, time_step_s):
        """Apply thrust to modify orbital trajectory."""
        if not self.is_controllable or self.fuel_remaining_kg <= 0:
            return
        
        # Calculate fuel consumption
        thrust_magnitude = math.sqrt(thrust_x**2 + thrust_y**2 + thrust_z**2)
        if thrust_magnitude > self.max_thrust_n:
            # Normalize thrust to max thrust
            thrust_x = thrust_x * self.max_thrust_n / thrust_magnitude
            thrust_y = thrust_y * self.max_thrust_n / thrust_magnitude 
            thrust_z = thrust_z * self.max_thrust_n / thrust_magnitude
            thrust_magnitude = self.max_thrust_n
        
        # Simplified fuel consumption (proportional to thrust)
        fuel_consumption_rate = 0.1  # kg per Newton per second
        fuel_consumed = thrust_magnitude * fuel_consumption_rate * time_step_s
        self.fuel_remaining_kg = max(0, self.fuel_remaining_kg - fuel_consumed)
        
        # Apply delta-v (simplified - affects velocity directly)
        mass_kg = 1000  # Assume 1000 kg satellite
        delta_v_x = (thrust_x / mass_kg) * time_step_s / 1000  # Convert to km/s
        delta_v_y = (thrust_y / mass_kg) * time_step_s / 1000
        delta_v_z = (thrust_z / mass_kg) * time_step_s / 1000
        
        self.velocity_x += delta_v_x
        self.velocity_y += delta_v_y
        self.velocity_z += delta_v_z
        
        # Update orbital elements based on new velocity (simplified)
        # In reality, you'd convert Cartesian to Keplerian elements
        self.orbit_elements = convert_to_orbit_elements(
            altitude_km=self.altitude_km,  # Keep original for now
            inclination_deg=self.inclination_deg,
            eccentricity=self.eccentricity,
            mean_anomaly_deg=self.mean_anomaly_deg,
            arg_periapsis_deg=self.omega_deg,
            raan_deg=self.raan_deg
        )


class EnhancedWebSimulator:
    """Enhanced web-based orbital avoidance simulator with RL integration."""
    
    def __init__(self, config_file="scenarios/leo_debris_avoidance.json"):
        print("Initializing enhanced web simulator with RL integration...")
        
        # Load RAW configuration from JSON file BEFORE creating environment
        self.config_file = config_file
        self.raw_config = self.load_raw_configuration()
        print(f"✅ Loaded raw configuration from {config_file}")
        
        # Create environment for physics reference (this will use processed config)
        self.env = create_rl_environment(config_file=config_file)
        self.obs, self.info = self.env.reset()
        
        # Initialize RL controller
        self.rl_controller = SimpleRLController()
        
        # Simulation state
        self.running = False
        self.step_count = 0
        self.simulation_time_s = 0
        self.time_step_s = self.raw_config["physics"]["time_step_seconds"]  # Use JSON time step
        self.speed_multiplier = 1.0  # 1x, 2x, or 4x speed
        self.total_close_encounters = 0
        self.last_encounter_step = -1
        self.total_collisions = 0
        self.last_collision_step = -1
        
        # Start automatic stepping thread
        self._stop_auto_step = False
        self._auto_step_thread = threading.Thread(target=self._auto_step_loop, daemon=True)
        self._auto_step_thread.start()
        
        # Action tracking for visualization
        self.last_action_info = {
            'thrust': [0.0, 0.0, 0.0],
            'mode': 'zero',
            'source': 'zero_policy',
            'model_name': None
        }
        
        # Create objects from JSON configuration
        self.create_objects_from_config()
        
        print("✅ Enhanced simulator initialized successfully")
    
    def load_raw_configuration(self):
        """Load raw configuration from JSON file without processing."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"❌ Error loading configuration from {self.config_file}: {e}")
            raise
    
    def load_configuration(self):
        """Load configuration from JSON file."""
        # This method is now deprecated - use load_raw_configuration instead
        return self.load_raw_configuration()
    
    def create_objects_from_config(self):
        """Create satellite and debris objects from JSON configuration."""
        print("🚀 Creating objects from JSON configuration...")
        
        # Create satellite from JSON
        satellite_config = self.raw_config["satellites"][0]
        orbit = satellite_config["orbit"]
        
        # Handle different orbit parameter formats
        earth_radius_km = 6371.0
        if "altitude_km" in orbit:
            altitude_km = orbit["altitude_km"]
        else:
            # Convert semi_major_axis to altitude (altitude = semi_major_axis - Earth_radius)
            altitude_km = orbit["semi_major_axis_km"] - earth_radius_km
        
        self.satellite = OrbitingObject(
            name=satellite_config["name"],
            altitude_km=altitude_km,
            inclination_deg=orbit["inclination_deg"],
            eccentricity=orbit["eccentricity"],
            mean_anomaly_deg=orbit.get("mean_anomaly_deg", orbit.get("true_anomaly_deg", 0)),
            omega_deg=orbit.get("arg_periapsis_deg", orbit.get("argument_of_perigee_deg", 0)),
            raan_deg=orbit["raan_deg"]
        )
        
        # Set satellite properties from JSON
        self.satellite.fuel_remaining_kg = satellite_config["fuel_capacity_kg"]
        self.satellite.initial_fuel_kg = satellite_config["fuel_capacity_kg"]  # Track initial fuel
        self.satellite.max_thrust_n = satellite_config.get("thrust_capacity_n", 1.0)
        self.satellite.is_controllable = satellite_config["is_controllable"]
        
        print(f"✅ Created satellite: {satellite_config['name']} at {altitude_km:.1f} km")
        
        # Create debris objects from JSON configuration
        self.debris_objects = []
        
        # Debug: Print what we're working with
        print(f"🔍 Raw config keys: {list(self.raw_config.keys())}")
        print(f"🔍 debris_fields: {self.raw_config.get('debris_fields', 'NOT FOUND')}")
        print(f"🔍 static_debris: {self.raw_config.get('static_debris', 'NOT FOUND')}")
        print(f"🔍 debris: {self.raw_config.get('debris', 'NOT FOUND')}")
        
        # Handle debris_fields for generating multiple random debris
        for debris_field in self.raw_config.get("debris_fields", []):
            num_objects = debris_field["num_objects"]
            altitude_range = debris_field["altitude_range_km"]
            inclination_range = debris_field["inclination_range_deg"]
            eccentricity_range = debris_field["eccentricity_range"]
            
            print(f"🌪️ Creating {num_objects} debris objects from field...")
            
            for i in range(num_objects):
                # Generate random orbital parameters within ranges
                altitude = random.uniform(altitude_range[0], altitude_range[1])
                inclination = random.uniform(inclination_range[0], inclination_range[1])
                eccentricity = random.uniform(eccentricity_range[0], eccentricity_range[1])
                mean_anomaly = random.uniform(0, 360)
                omega = random.uniform(0, 360)
                raan = random.uniform(0, 360)
                
                debris = OrbitingObject(
                    name=f"FieldDebris_{len(self.debris_objects) + 1}",
                    altitude_km=altitude,
                    inclination_deg=inclination,
                    eccentricity=eccentricity,
                    mean_anomaly_deg=mean_anomaly,
                    omega_deg=omega,
                    raan_deg=raan
                )
                debris.is_controllable = False
                self.debris_objects.append(debris)
        
        # Handle static debris array from JSON (current format)
        debris_list = self.raw_config.get("debris", [])
        for i, debris_config in enumerate(debris_list):
            orbit = debris_config["orbit"]
            # Convert semi_major_axis to altitude
            altitude_km = orbit["semi_major_axis_km"] - earth_radius_km
            
            debris = OrbitingObject(
                name=debris_config["name"],
                altitude_km=altitude_km,
                inclination_deg=orbit["inclination_deg"],
                eccentricity=orbit["eccentricity"],
                mean_anomaly_deg=orbit.get("true_anomaly_deg", 0),
                omega_deg=orbit.get("argument_of_perigee_deg", 0),
                raan_deg=orbit["raan_deg"]
            )
            debris.is_controllable = False
            self.debris_objects.append(debris)
            print(f"✅ Created debris: {debris.name} at {altitude_km:.1f} km")
        
        # Handle static_debris array (if present)
        static_debris_list = self.raw_config.get("static_debris", [])
        for debris_config in static_debris_list:
            orbit = debris_config["orbit"]
            # Handle different orbit parameter formats
            if "altitude_km" in orbit:
                altitude_km = orbit["altitude_km"]
            else:
                altitude_km = orbit["semi_major_axis_km"] - earth_radius_km
            
            debris = OrbitingObject(
                name=debris_config["name"],
                altitude_km=altitude_km,
                inclination_deg=orbit["inclination_deg"],
                eccentricity=orbit["eccentricity"],
                mean_anomaly_deg=orbit.get("mean_anomaly_deg", orbit.get("true_anomaly_deg", 0)),
                omega_deg=orbit.get("arg_periapsis_deg", orbit.get("argument_of_perigee_deg", 0)),
                raan_deg=orbit["raan_deg"]
            )
            debris.is_controllable = False
            self.debris_objects.append(debris)
            print(f"✅ Created static debris: {debris.name} at {altitude_km:.1f} km")
        
        print(f"✅ Created {len(self.debris_objects)} total debris objects from configuration")
    
    def _auto_step_loop(self):
        """Automatic stepping loop that runs in a separate thread."""
        step_count = 0
        while not self._stop_auto_step:
            if self.running:
                try:
                    # Calculate steps per batch based on speed multiplier
                    steps_to_run = int(self.speed_multiplier)
                    
                    for _ in range(steps_to_run):
                        self.step_simulation()
                        step_count += 1
                    
                    if step_count % 10 == 0:  # Print every 10 steps
                        print(f"⚡ Auto-stepping: {step_count} steps, simulation time: {self.simulation_time_s:.1f}s, speed: {self.speed_multiplier}x")
                    
                    # Base delay is 0.5s for 1x speed, adjusted for speed multiplier
                    base_delay = 0.5
                    adjusted_delay = base_delay / self.speed_multiplier
                    time.sleep(adjusted_delay)
                    
                except Exception as e:
                    print(f"❌ Error in auto-step {step_count}: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)  # Wait longer on error
            else:
                time.sleep(0.1)  # Check every 100ms when paused
    
    def get_simulation_state(self):
        """Get current simulation state including RL action info."""
        # Calculate distances and encounters
        current_encounters = 0
        current_collisions = 0
        closest_distance = float('inf')
        
        for debris in self.debris_objects:
            distance = math.sqrt(
                (self.satellite.position_x - debris.position_x)**2 +
                (self.satellite.position_y - debris.position_y)**2 +
                (self.satellite.position_z - debris.position_z)**2
            )
            
            closest_distance = min(closest_distance, distance)
            
            if distance < 50:  # Close encounter threshold
                current_encounters += 1
            if distance < 10:  # Collision threshold  
                current_collisions += 1
        
        # Update encounter counters (avoid double counting)
        if current_encounters > 0 and self.last_encounter_step != self.step_count:
            self.total_close_encounters += current_encounters
            self.last_encounter_step = self.step_count
        
        if current_collisions > 0 and self.last_collision_step != self.step_count:
            self.total_collisions += current_collisions
            self.last_collision_step = self.step_count
        
        # Satellite state
        satellite_data = {
            'x': self.satellite.position_x,
            'y': self.satellite.position_y,
            'z': self.satellite.position_z,
            'vx': self.satellite.velocity_x,
            'vy': self.satellite.velocity_y,
            'vz': self.satellite.velocity_z,
            'fuel_remaining_kg': self.satellite.fuel_remaining_kg,
            'fuel_remaining_percent': (self.satellite.fuel_remaining_kg / self.satellite.initial_fuel_kg) * 100
        }
        
        # Debris states
        debris_data = []
        for debris in self.debris_objects:
            debris_data.append({
                'name': debris.name,
                'x': debris.position_x,
                'y': debris.position_y,
                'z': debris.position_z,
                'vx': debris.velocity_x,
                'vy': debris.velocity_y,
                'vz': debris.velocity_z
            })
        
        return {
            'satellite': satellite_data,
            'debris': debris_data,
            'simulation_time': self.simulation_time_s,
            'step_count': self.step_count,
            'running': self.running,
            'speed_multiplier': self.speed_multiplier,
            'encounters_and_collisions': self.total_close_encounters + self.total_collisions,
            'total_close_encounters': self.total_close_encounters,
            'total_collisions': self.total_collisions,
            'closest_distance': closest_distance,
            'last_action': self.last_action_info,
            'rl_status': self.rl_controller.get_status()
        }
    
    def step_simulation(self):
        """Execute one simulation step with RL integration."""
        if not self.running:
            return False
            
        print(f"🔧 Step {self.step_count}: Running={self.running}, Time={self.simulation_time_s:.1f}s")
        
        # Get current state for RL agent
        satellite_state = {
            'position': [self.satellite.position_x, self.satellite.position_y, self.satellite.position_z],
            'velocity': [self.satellite.velocity_x, self.satellite.velocity_y, self.satellite.velocity_z],
            'fuel_remaining_percent': (self.satellite.fuel_remaining_kg / self.satellite.initial_fuel_kg) * 100,
            'step_count': self.step_count
        }
        
        debris_states = []
        for debris in self.debris_objects:
            debris_states.append({
                'position': [debris.position_x, debris.position_y, debris.position_z],
                'velocity': [debris.velocity_x, debris.velocity_y, debris.velocity_z]
            })
        
        # Get action from RL controller
        action_info = self.rl_controller.get_action(satellite_state, debris_states)
        self.last_action_info = action_info
        
        # Apply thrust to satellite
        thrust = action_info['thrust']
        self.satellite.apply_thrust(thrust[0], thrust[1], thrust[2], self.time_step_s)
        
        # Update simulation time FIRST
        self.simulation_time_s += self.time_step_s
        
        # Update all object positions with new time
        self.satellite.update_position(self.simulation_time_s)
        for debris in self.debris_objects:
            debris.update_position(self.simulation_time_s)
        
        # Increment step count AFTER updating positions
        old_step = self.step_count
        self.step_count += 1
        print(f"✅ Step completed: {old_step} → {self.step_count} (time: {self.simulation_time_s}s)")
        
        # Log significant actions
        if any(abs(t) > 0.1 for t in thrust):
            print(f"🎮 Step {self.step_count}: {action_info['source']} → Thrust = {thrust}")
        
        return True
    
    def reset_simulation(self):
        """Reset simulation to initial state."""
        print("🔄 Resetting simulation...")
        self.step_count = 0
        self.simulation_time_s = 0
        self.total_close_encounters = 0
        self.total_collisions = 0
        self.last_encounter_step = -1
        self.last_collision_step = -1
        
        # Reset satellite fuel to initial capacity from config
        satellite_config = self.raw_config["satellites"][0]
        self.satellite.fuel_remaining_kg = satellite_config["fuel_capacity_kg"]
        self.satellite.initial_fuel_kg = satellite_config["fuel_capacity_kg"]
        
        # Reset all positions to initial time
        self.satellite.update_position(0.0)
        for debris in self.debris_objects:
            debris.update_position(0.0)
        
        print("✅ Simulation reset")
    
    def set_speed_multiplier(self, multiplier):
        """Set simulation speed multiplier."""
        if multiplier in [1.0, 2.0, 4.0]:
            self.speed_multiplier = multiplier
            print(f"🏃 Speed set to {multiplier}x")
            return True
        else:
            print(f"❌ Invalid speed multiplier: {multiplier}. Must be 1.0, 2.0, or 4.0")
            return False
    
    def toggle_running(self):
        """Toggle simulation running state."""
        self.running = not self.running
        print(f"🔄 Simulation {'started' if self.running else 'paused'}")
        print(f"   Step count: {self.step_count}")
        print(f"   Current time: {self.simulation_time_s}s")
        print(f"   Debris count: {len(self.debris_objects)}")


# Global enhanced simulator instance - will be initialized in main()
simulator = None


class EnhancedWebHandler(BaseHTTPRequestHandler):
    """Enhanced HTTP request handler with RL integration."""
    
    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path
        query_params = parse_qs(urlparse(self.path).query)
        
        if path == "/":
            self._serve_html()
        elif path == "/api/state":
            self._serve_state()
        elif path == "/api/step":
            simulator.step_simulation()
            self._serve_state()
        elif path == "/api/toggle":
            simulator.toggle_running()
            self._serve_state()
        elif path == "/api/reset":
            simulator.reset_simulation()
            self._serve_state()
        elif path == "/api/rl_status":
            self._serve_rl_status()
        elif path == "/api/set_control_mode":
            mode = query_params.get('mode', ['zero'])[0]
            self._handle_set_control_mode(mode)
        elif path == "/api/load_model":
            model_name = query_params.get('model', [''])[0]
            self._handle_load_model(model_name)
        elif path == "/api/set_thrust":
            # Manual thrust control
            thrust_x = float(query_params.get('x', [0])[0])
            thrust_y = float(query_params.get('y', [0])[0])
            thrust_z = float(query_params.get('z', [0])[0])
            self._handle_set_thrust(thrust_x, thrust_y, thrust_z)
        elif path == "/api/set_speed":
            # Speed control
            speed = float(query_params.get('speed', [1.0])[0])
            self._handle_set_speed(speed)
        else:
            self.send_error(404)
    
    def _serve_html(self):
        """Serve the enhanced HTML interface with RL controls."""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛰️ RL-Enhanced Orbital Avoidance Simulator</title>
    <script src="https://unpkg.com/three@0.145.0/build/three.min.js"></script>
    <script src="https://unpkg.com/three@0.145.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            overflow: hidden;
        }
        
        #container {
            display: flex;
            height: 100vh;
        }
        
        #visualization {
            flex: 1;
            position: relative;
        }
        
        .controls {
            width: 400px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 20px;
            overflow-y: auto;
            box-shadow: -5px 0 15px rgba(0,0,0,0.3);
            border-left: 2px solid #0f4c75;
        }
        
        .control-section {
            background: rgba(15, 76, 117, 0.1);
            border: 1px solid rgba(15, 76, 117, 0.3);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .control-section h3 {
            margin: 0 0 15px 0;
            color: #3282b8;
            border-bottom: 2px solid #3282b8;
            padding-bottom: 8px;
        }
        
        .button {
            background: linear-gradient(135deg, #3282b8 0%, #0f4c75 100%);
            color: white;
            border: none;
            padding: 10px 15px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        
        .button:hover {
            background: linear-gradient(135deg, #0f4c75 0%, #3282b8 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(50, 130, 184, 0.3);
        }
        
        .button.active {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        }
        
        .thrust-control {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .thrust-axis {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .thrust-axis label {
            width: 20px;
            font-weight: bold;
        }
        
        .thrust-axis input {
            flex: 1;
            height: 30px;
            background: #1a1a2e;
            border: 1px solid #3282b8;
            border-radius: 4px;
            color: #ffffff;
        }
        
        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            font-size: 14px;
        }
        
        .metric {
            background: rgba(0,0,0,0.3);
            padding: 8px;
            border-radius: 4px;
            border-left: 3px solid #3282b8;
        }
        
        .metric-value {
            font-weight: bold;
            color: #3282b8;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-active { background-color: #28a745; }
        .status-inactive { background-color: #dc3545; }
        .status-warning { background-color: #ffc107; }
        
        select {
            background: #1a1a2e;
            color: #ffffff;
            border: 1px solid #3282b8;
            border-radius: 4px;
            padding: 8px;
            width: 100%;
            margin-top: 5px;
        }
        
        .action-display {
            background: rgba(40, 167, 69, 0.1);
            border: 1px solid #28a745;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
            font-family: monospace;
            font-size: 12px;
        }
        
        .rl-info {
            background: rgba(255, 193, 7, 0.1);
            border: 1px solid #ffc107;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="visualization"></div>
        
        <div class="controls">
            <h2>🎮 RL Control Center</h2>
            
            <!-- Control Mode Selection -->
            <div class="control-section">
                <h3>Control Mode</h3>
                <button class="button" id="mode-zero" onclick="setControlMode('zero')">
                    <span class="status-indicator status-active" id="status-zero"></span>Zero Policy
                </button>
                <button class="button" id="mode-manual" onclick="setControlMode('manual')">
                    <span class="status-indicator status-inactive" id="status-manual"></span>Manual Control
                </button>
                <button class="button" id="mode-rl" onclick="setControlMode('rl')">
                    <span class="status-indicator status-inactive" id="status-rl"></span>RL Agent
                </button>
            </div>
            
            <!-- RL Model Selection -->
            <div class="control-section">
                <h3>🧠 RL Models</h3>
                <select id="model-select" onchange="loadSelectedModel()">
                    <option value="">Select a trained model...</option>
                </select>
                <div class="rl-info" id="rl-info">
                    No RL agent loaded
                </div>
            </div>
            
            <!-- Manual Thrust Controls -->
            <div class="control-section">
                <h3>🚀 Manual Thrust</h3>
                <div class="thrust-control">
                    <div class="thrust-axis">
                        <label>X:</label>
                        <input type="range" id="thrust-x" min="-1" max="1" step="0.1" value="0" 
                               oninput="updateManualThrust()">
                        <span id="thrust-x-value">0.0</span>
                    </div>
                    <div class="thrust-axis">
                        <label>Y:</label>
                        <input type="range" id="thrust-y" min="-1" max="1" step="0.1" value="0" 
                               oninput="updateManualThrust()">
                        <span id="thrust-y-value">0.0</span>
                    </div>
                    <div class="thrust-axis">
                        <label>Z:</label>
                        <input type="range" id="thrust-z" min="-1" max="1" step="0.1" value="0" 
                               oninput="updateManualThrust()">
                        <span id="thrust-z-value">0.0</span>
                    </div>
                    <button class="button" onclick="resetThrust()">Reset Thrust</button>
                </div>
            </div>
            
            <!-- Current Action Display -->
            <div class="control-section">
                <h3>📊 Current Action</h3>
                <div class="action-display" id="action-display">
                    Source: zero_policy<br>
                    Thrust: [0.000, 0.000, 0.000]
                </div>
            </div>
            
            <!-- Simulation Controls -->
            <div class="control-section">
                <h3>⚙️ Simulation</h3>
                <button class="button" onclick="toggleSimulation()" id="toggle-btn">▶️ Start</button>
                <button class="button" onclick="stepSimulation()">⏭️ Step</button>
                <button class="button" onclick="resetSimulation()">🔄 Reset</button>
            </div>
            
            <!-- Speed Controls -->
            <div class="control-section">
                <h3>🏃 Speed Control</h3>
                <button class="button" id="speed-toggle" onclick="toggleSpeed()">
                    <span id="speed-icon">🐌</span> <span id="speed-text">1x Normal</span>
                </button>
                <div style="margin-top: 8px; font-size: 11px; color: #666;">
                    Click to cycle: 1x → 2x → 4x
                </div>
            </div>
            
            <!-- Metrics -->
            <div class="control-section">
                <h3>📈 Metrics</h3>
                <div class="metrics">
                    <div class="metric">
                        <div>Step</div>
                        <div class="metric-value" id="step-count">0</div>
                    </div>
                    <div class="metric">
                        <div>Fuel (%)</div>
                        <div class="metric-value" id="fuel-percent">100</div>
                    </div>
                    <div class="metric">
                        <div>Encounters</div>
                        <div class="metric-value" id="encounters">0</div>
                    </div>
                    <div class="metric">
                        <div>Closest (km)</div>
                        <div class="metric-value" id="closest-distance">∞</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let scene, camera, renderer, controls;
        let satellite, debris = [];
        let currentData = null;
        let currentControlMode = 'zero';
        
        function init() {
            // Create scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000011);
            
            // Create camera with proper aspect ratio
            const container = document.getElementById('visualization');
            const aspect = container.clientWidth / container.clientHeight;
            camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 100000);
            camera.position.set(15000, 10000, 15000);
            camera.lookAt(0, 0, 0);
            
            // Create renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            
            // Handle window resize to maintain proper aspect ratio
            window.addEventListener('resize', () => {
                const newAspect = container.clientWidth / container.clientHeight;
                camera.aspect = newAspect;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            });
            
            container.appendChild(renderer.domElement);
            
            // Add orbit controls
            if (typeof THREE.OrbitControls !== 'undefined') {
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.screenSpacePanning = false;
                controls.minDistance = 7000;
                controls.maxDistance = 30000;
            }
            
            // Create Earth
            const earthGeometry = new THREE.SphereGeometry(6371, 32, 32);
            const earthMaterial = new THREE.MeshLambertMaterial({
                color: 0x4477aa,
                transparent: true,
                opacity: 0.9
            });
            const earth = new THREE.Mesh(earthGeometry, earthMaterial);
            scene.add(earth);
            
            // Add lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6); // Increased ambient light
            scene.add(ambientLight);
            
            // Primary directional light (sun)
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(50000, 50000, 50000);
            directionalLight.castShadow = true;
            scene.add(directionalLight);
            
            // Secondary directional light from opposite side (fill light)
            const fillLight = new THREE.DirectionalLight(0x6699ff, 0.3);
            fillLight.position.set(-30000, -20000, -30000);
            scene.add(fillLight);
            
            // Additional hemisphere light for even better visibility
            const hemisphereLight = new THREE.HemisphereLight(0x87ceeb, 0x262626, 0.4);
            scene.add(hemisphereLight);
            
            // Create satellite
            const satelliteGeometry = new THREE.BoxGeometry(100, 100, 100);
            const satelliteMaterial = new THREE.MeshLambertMaterial({ 
                color: 0x00ff00,
                emissive: 0x002200  // Add slight glow
            });
            satellite = new THREE.Mesh(satelliteGeometry, satelliteMaterial);
            scene.add(satellite);
            
            // Create debris objects (dynamic count based on configuration)
            debris = [];
            for (let i = 0; i < 50; i++) {  // Max 50 debris for performance
                const debrisGeometry = new THREE.SphereGeometry(50, 8, 8);
                const debrisMaterial = new THREE.MeshLambertMaterial({ 
                    color: 0xff4444,
                    emissive: 0x220000  // Add slight glow
                });
                const debrisObj = new THREE.Mesh(debrisGeometry, debrisMaterial);
                debrisObj.visible = false;  // Initially hidden
                debris.push(debrisObj);
                scene.add(debrisObj);
            }
            
            // Start update loop
            fetchData();
            animate();
            setInterval(fetchData, 500);
            
            // Load available models
            loadAvailableModels();
        }
        
        function animate() {
            requestAnimationFrame(animate);
            
            if (controls) {
                controls.update();
            }
            
            renderer.render(scene, camera);
        }
        
        function fetchData() {
            fetch('/api/state')
                .then(response => response.json())
                .then(data => {
                    currentData = data;
                    updateVisualization(data);
                    updateUI(data);
                })
                .catch(error => console.error('Error:', error));
        }
        
        function updateVisualization(data) {
            // Update satellite position
            satellite.position.set(data.satellite.x, data.satellite.y, data.satellite.z);
            
            // Update debris positions (handle dynamic count)
            for (let i = 0; i < debris.length; i++) {
                if (i < data.debris.length) {
                    debris[i].position.set(
                        data.debris[i].x,
                        data.debris[i].y,
                        data.debris[i].z
                    );
                    debris[i].visible = true;
                } else {
                    debris[i].visible = false;  // Hide unused debris objects
                }
            }
        }
        
        function updateUI(data) {
            // Update metrics
            document.getElementById('step-count').textContent = data.step_count;
            document.getElementById('fuel-percent').textContent = data.satellite.fuel_remaining_percent.toFixed(1);
            document.getElementById('encounters').textContent = data.encounters_and_collisions;
            document.getElementById('closest-distance').textContent = 
                data.closest_distance === Infinity ? '∞' : data.closest_distance.toFixed(1);
            
            // Update action display
            const actionDisplay = document.getElementById('action-display');
            const action = data.last_action;
            actionDisplay.innerHTML = 
                `Source: ${action.source}<br>` +
                `Thrust: [${action.thrust.map(t => t.toFixed(3)).join(', ')}]` +
                (action.model_name ? `<br>Model: ${action.model_name}` : '');
            
            // Update control mode indicators
            updateControlModeUI(data.rl_status.control_mode);
            
            // Update RL info
            updateRLInfo(data.rl_status);
            
            // Update toggle button
            document.getElementById('toggle-btn').textContent = data.running ? '⏸️ Pause' : '▶️ Start';
            
            // Update speed UI
            if (data.speed_multiplier !== undefined) {
                updateSpeedUI(data.speed_multiplier);
            }
        }
        
        function updateControlModeUI(mode) {
            currentControlMode = mode;
            
            // Reset all indicators
            document.getElementById('status-zero').className = 'status-indicator status-inactive';
            document.getElementById('status-manual').className = 'status-indicator status-inactive';
            document.getElementById('status-rl').className = 'status-indicator status-inactive';
            
            // Reset all buttons
            document.getElementById('mode-zero').className = 'button';
            document.getElementById('mode-manual').className = 'button';
            document.getElementById('mode-rl').className = 'button';
            
            // Set active mode
            document.getElementById(`status-${mode}`).className = 'status-indicator status-active';
            document.getElementById(`mode-${mode}`).className = 'button active';
        }
        
        function updateRLInfo(rlStatus) {
            const rlInfo = document.getElementById('rl-info');
            
            if (rlStatus.rl_agent_loaded) {
                const modelInfo = rlStatus.rl_model_info;
                rlInfo.innerHTML = 
                    `✅ Agent Loaded<br>` +
                    `Algorithm: ${modelInfo.algorithm.toUpperCase()}<br>` +
                    `Model: ${modelInfo.model_path.split('/').pop()}`;
                rlInfo.style.borderColor = '#28a745';
                rlInfo.style.backgroundColor = 'rgba(40, 167, 69, 0.1)';
            } else {
                rlInfo.innerHTML = '❌ No RL agent loaded';
                rlInfo.style.borderColor = '#dc3545';
                rlInfo.style.backgroundColor = 'rgba(220, 53, 69, 0.1)';
            }
        }
        
        function loadAvailableModels() {
            fetch('/api/rl_status')
                .then(response => response.json())
                .then(data => {
                    const select = document.getElementById('model-select');
                    select.innerHTML = '<option value="">Select a trained model...</option>';
                    
                    data.available_models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = model;
                        select.appendChild(option);
                    });
                })
                .catch(error => console.error('Error loading models:', error));
        }
        
        function setControlMode(mode) {
            fetch(`/api/set_control_mode?mode=${mode}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateControlModeUI(mode);
                    }
                })
                .catch(error => console.error('Error setting control mode:', error));
        }
        
        function loadSelectedModel() {
            const modelName = document.getElementById('model-select').value;
            if (modelName) {
                fetch(`/api/load_model?model=${encodeURIComponent(modelName)}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert(`Model "${modelName}" loaded successfully!`);
                        } else {
                            alert(`Failed to load model: ${data.message}`);
                        }
                    })
                    .catch(error => console.error('Error loading model:', error));
            }
        }
        
        function updateManualThrust() {
            const thrustX = document.getElementById('thrust-x').value;
            const thrustY = document.getElementById('thrust-y').value;
            const thrustZ = document.getElementById('thrust-z').value;
            
            document.getElementById('thrust-x-value').textContent = parseFloat(thrustX).toFixed(1);
            document.getElementById('thrust-y-value').textContent = parseFloat(thrustY).toFixed(1);
            document.getElementById('thrust-z-value').textContent = parseFloat(thrustZ).toFixed(1);
            
            if (currentControlMode === 'manual') {
                fetch(`/api/set_thrust?x=${thrustX}&y=${thrustY}&z=${thrustZ}`)
                    .catch(error => console.error('Error setting thrust:', error));
            }
        }
        
        function resetThrust() {
            document.getElementById('thrust-x').value = 0;
            document.getElementById('thrust-y').value = 0;
            document.getElementById('thrust-z').value = 0;
            updateManualThrust();
        }
        
        function toggleSimulation() {
            fetch('/api/toggle')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('toggle-btn').textContent = 
                        data.running ? '⏸️ Pause' : '▶️ Start';
                })
                .catch(error => console.error('Error:', error));
        }
        
        function stepSimulation() {
            fetch('/api/step')
                .catch(error => console.error('Error:', error));
        }
        
        function resetSimulation() {
            fetch('/api/reset')
                .catch(error => console.error('Error:', error));
        }
        
        function toggleSpeed() {
            // Get current speed and cycle to next
            let currentSpeed = currentData ? currentData.speed_multiplier : 1.0;
            let nextSpeed;
            
            if (currentSpeed === 1.0) {
                nextSpeed = 2.0;
            } else if (currentSpeed === 2.0) {
                nextSpeed = 4.0;
            } else {
                nextSpeed = 1.0;
            }
            
            setSpeed(nextSpeed);
        }
        
        function setSpeed(speedMultiplier) {
            fetch(`/api/set_speed?speed=${speedMultiplier}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateSpeedUI(data.speed_multiplier);
                    } else {
                        console.error('Failed to set speed:', data.message);
                    }
                })
                .catch(error => console.error('Error setting speed:', error));
        }
        
        function updateSpeedUI(speedMultiplier) {
            const speedIcon = document.getElementById('speed-icon');
            const speedText = document.getElementById('speed-text');
            const speedButton = document.getElementById('speed-toggle');
            
            // Update icon, text, and button color based on speed
            if (speedMultiplier === 1.0) {
                speedIcon.textContent = '🐌';
                speedText.textContent = '1x Normal';
                speedButton.style.background = 'linear-gradient(135deg, #3282b8 0%, #0f4c75 100%)';
            } else if (speedMultiplier === 2.0) {
                speedIcon.textContent = '🏃';
                speedText.textContent = '2x Fast';
                speedButton.style.background = 'linear-gradient(135deg, #28a745 0%, #20c997 100%)';
            } else if (speedMultiplier === 4.0) {
                speedIcon.textContent = '🚀';
                speedText.textContent = '4x Turbo';
                speedButton.style.background = 'linear-gradient(135deg, #dc3545 0%, #e83e8c 100%)';
            }
        }
        
        // Handle window resize
        window.addEventListener('resize', () => {
            const container = document.getElementById('visualization');
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });
        
        // Initialize when page loads
        window.addEventListener('load', init);
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def _serve_state(self):
        """Serve current simulation state as JSON."""
        state = simulator.get_simulation_state()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(state).encode())
    
    def _serve_rl_status(self):
        """Serve RL controller status."""
        status = simulator.rl_controller.get_status()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())
    
    def _handle_set_control_mode(self, mode):
        """Handle control mode change."""
        success = simulator.rl_controller.set_control_mode(mode)
        response = {
            'success': success,
            'current_mode': simulator.rl_controller.control_mode,
            'message': f"Control mode set to {mode}" if success else f"Invalid mode: {mode}"
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def _handle_load_model(self, model_name):
        """Handle RL model loading."""
        if model_name in simulator.rl_controller.available_models:
            model_path = simulator.rl_controller.available_models[model_name]
            success = simulator.rl_controller.load_rl_model(model_path)
            response = {
                'success': success,
                'model_name': model_name,
                'model_path': model_path,
                'message': "Model loaded successfully" if success else "Failed to load model"
            }
        else:
            response = {
                'success': False,
                'message': f"Model '{model_name}' not found",
                'available_models': list(simulator.rl_controller.available_models.keys())
            }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def _handle_set_thrust(self, thrust_x, thrust_y, thrust_z):
        """Handle manual thrust setting."""
        simulator.rl_controller.set_manual_thrust(thrust_x, thrust_y, thrust_z)
        response = {
            'success': True,
            'thrust': [thrust_x, thrust_y, thrust_z],
            'message': 'Manual thrust set'
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def _handle_set_speed(self, speed):
        """Handle speed multiplier setting."""
        success = simulator.set_speed_multiplier(speed)
        response = {
            'success': success,
            'speed_multiplier': simulator.speed_multiplier,
            'message': f'Speed set to {speed}x' if success else f'Invalid speed: {speed}'
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())


def run_enhanced_server(port=8080, config_file="scenarios/leo_debris_avoidance.json"):
    """Run the enhanced web server."""
    global simulator
    
    print(f"🚀 Starting enhanced RL web simulator on port {port}")
    print(f"📋 Using configuration: {config_file}")
    print(f"🌐 Open your browser to: http://localhost:{port}")
    print()
    
    # Initialize simulator with config file
    simulator = EnhancedWebSimulator(config_file)
    
    print("Features:")
    print("  • 🎮 Manual thrust control")
    print("  • 🧠 RL agent integration")
    print("  • 🔄 Real-time model switching")
    print("  • 📊 Action visualization")
    print("  • ⚡ Live performance metrics")
    print("  • 📋 JSON configuration support")
    print()
    
    try:
        server = HTTPServer(('localhost', port), EnhancedWebHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        if simulator:
            simulator._stop_auto_step = True
    except Exception as e:
        print(f"❌ Server error: {e}")


def main():
    """Main entry point for the enhanced web simulator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Web Simulator with RL Integration')
    parser.add_argument('--config', default='scenarios/leo_debris_avoidance.json',
                       help='Configuration file path (default: scenarios/leo_debris_avoidance.json)')
    parser.add_argument('--port', type=int, default=8080,
                       help='Server port (default: 8080)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🛰️ ENHANCED RL ORBITAL AVOIDANCE SIMULATOR")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Port: {args.port}")
    print("=" * 60)
    
    run_enhanced_server(port=args.port, config_file=args.config)


if __name__ == "__main__":
    main()