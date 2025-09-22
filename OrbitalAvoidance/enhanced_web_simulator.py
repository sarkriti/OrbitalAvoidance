#!/usr/bin/env python3
"""
Enhanced Web Simulator with RL Integration

This enhanced version of your web simulator adds:
1. RL agent control integration
2. Manual thrust controls  
3. Real-time model switching
4. RL vs manual performance comparison
5. Action visualization and debugging

Usage:
    python enhanced_web_simulator.py
    
Then visit: http://localhost:8080
"""

import os
import sys
import json
import time
import math
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

# Add project path
if __package__ in (None, ""):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from OrbitalAvoidance.rl_training_env import create_rl_environment
from OrbitalAvoidance.algorithms import ZeroPolicy
from OrbitalAvoidance.orekit_physics import (
    create_propagator, OrbitElements, convert_to_orbit_elements
)
from OrbitalAvoidance.rl_web_integration import WebRLController, RLWebAgent
import numpy as np


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
    
    def __init__(self):
        print("Initializing enhanced web simulator with RL integration...")
        
                # Create environment for physics reference  
        self.env = create_rl_environment(config_file="scenarios/leo_debris_avoidance.json")
        self.obs, self.info = self.env.reset()
        
        # Initialize RL controller
        self.rl_controller = WebRLController()
        
        # Simulation state
        self.running = False
        self.step_count = 0
        self.simulation_time_s = 0
        self.time_step_s = 20.0  # 20 second time steps
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
        
        # Create realistic orbiting objects
        self.satellite = OrbitingObject(
            name="Satellite",
            altitude_km=400,
            inclination_deg=51.6,
            eccentricity=0.001,
            mean_anomaly_deg=0,
            raan_deg=0
        )
        
        # Create debris with varying orbital parameters, well separated from satellite
        self.debris_objects = []
        satellite_altitude = 400  # km
        debris_altitudes = [320, 350, 380, 450, 480, 520, 550, 580]  # Well separated altitudes
        
        for i in range(8):  # 8 debris objects
            debris = OrbitingObject(
                name=f"Debris_{i}",
                altitude_km=debris_altitudes[i],
                inclination_deg=51.6 + (i - 4) * 8,  # Much more varied inclinations (-28 to +24 degrees)
                eccentricity=0.001 + i * 0.004,  # More varying eccentricity
                mean_anomaly_deg=i * 45,  # Spread around orbit every 45 degrees
                omega_deg=i * 25,  # Vary argument of periapsis
                raan_deg=i * 40  # Much more different orbital planes
            )
            self.debris_objects.append(debris)
        
        print("✅ Enhanced simulator initialized successfully")
    
    def _auto_step_loop(self):
        """Automatic stepping loop that runs in a separate thread."""
        import time
        while not self._stop_auto_step:
            if self.running:
                try:
                    self.step_simulation()
                    time.sleep(0.1)  # 10 FPS update rate
                except Exception as e:
                    print(f"Error in auto-step: {e}")
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
            'fuel_remaining_percent': (self.satellite.fuel_remaining_kg / 100.0) * 100
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
            'encounters_and_collisions': self.total_close_encounters + self.total_collisions,
            'total_close_encounters': self.total_close_encounters,
            'total_collisions': self.total_collisions,
            'closest_distance': closest_distance,
            'last_action': self.last_action_info,
            'rl_status': self.rl_controller.get_status()
        }
    
    def step_simulation(self):
        """Execute one simulation step with RL integration."""
        print(f"🔧 Step {self.step_count}: Running={self.running}")
        
        # Get current state for RL agent
        satellite_state = {
            'position': [self.satellite.position_x, self.satellite.position_y, self.satellite.position_z],
            'velocity': [self.satellite.velocity_x, self.satellite.velocity_y, self.satellite.velocity_z],
            'fuel_remaining_percent': (self.satellite.fuel_remaining_kg / 100.0) * 100,
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
        
        # Update all object positions
        self.satellite.update_position(self.simulation_time_s)
        for debris in self.debris_objects:
            debris.update_position(self.simulation_time_s)
        
        # Update simulation time and step count
        self.simulation_time_s += self.time_step_s
        self.step_count += 1
        
        # Log significant actions
        if any(abs(t) > 0.1 for t in thrust):
            print(f"🎮 Step {self.step_count}: {action_info['source']} → Thrust = {thrust}")
    
    def reset_simulation(self):
        """Reset simulation to initial state."""
        print("🔄 Resetting simulation...")
        self.step_count = 0
        self.simulation_time_s = 0
        self.total_close_encounters = 0
        self.total_collisions = 0
        self.last_encounter_step = -1
        self.last_collision_step = -1
        
        # Reset satellite
        self.satellite.fuel_remaining_kg = 100.0
        
        # Reset all positions
        self.satellite.update_position(0.0)
        for debris in self.debris_objects:
            debris.update_position(0.0)
        
        print("✅ Simulation reset")
    
    def toggle_running(self):
        """Toggle simulation running state."""
        self.running = not self.running
        print(f"🔄 Simulation {'started' if self.running else 'paused'}")
        print(f"   Step count: {self.step_count}")
        print(f"   Current time: {self.simulation_time_s}s")


# Global enhanced simulator instance
simulator = EnhancedWebSimulator()


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
            const earthMaterial = new THREE.MeshPhongMaterial({
                color: 0x4477aa,
                shininess: 100
            });
            const earth = new THREE.Mesh(earthGeometry, earthMaterial);
            scene.add(earth);
            
            // Add lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
            directionalLight.position.set(50000, 50000, 50000);
            directionalLight.castShadow = true;
            scene.add(directionalLight);
            
            // Create satellite
            const satelliteGeometry = new THREE.BoxGeometry(100, 100, 100);
            const satelliteMaterial = new THREE.MeshPhongMaterial({ color: 0x00ff00 });
            satellite = new THREE.Mesh(satelliteGeometry, satelliteMaterial);
            scene.add(satellite);
            
            // Create debris objects
            for (let i = 0; i < 8; i++) {
                const debrisGeometry = new THREE.SphereGeometry(50, 8, 8);
                const debrisMaterial = new THREE.MeshPhongMaterial({ color: 0xff4444 });
                const debrisObj = new THREE.Mesh(debrisGeometry, debrisMaterial);
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
            
            // Update debris positions
            for (let i = 0; i < Math.min(debris.length, data.debris.length); i++) {
                debris[i].position.set(
                    data.debris[i].x,
                    data.debris[i].y,
                    data.debris[i].z
                );
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


def run_enhanced_server(port=8080):
    """Run the enhanced web server."""
    print(f"🚀 Starting enhanced RL web simulator on port {port}")
    print(f"🌐 Open your browser to: http://localhost:{port}")
    print()
    print("Features:")
    print("  • 🎮 Manual thrust control")
    print("  • 🧠 RL agent integration")
    print("  • 🔄 Real-time model switching")
    print("  • 📊 Action visualization")
    print("  • ⚡ Live performance metrics")
    print()
    
    try:
        server = HTTPServer(('localhost', port), EnhancedWebHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")


def main():
    """Main entry point for the enhanced web simulator."""
    run_enhanced_server()


if __name__ == "__main__":
    run_enhanced_server()