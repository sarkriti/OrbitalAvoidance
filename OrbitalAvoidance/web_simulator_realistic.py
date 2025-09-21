#!/usr/bin/env python3
"""
Web-based Orbital Avoidance Simulator with Realistic Orbital Mechanics

Features realistic orbital physics where each object has independent orbital parameters.
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

from OrbitalAvoidance.research_grade_env import make_env
from OrbitalAvoidance.algorithms import ZeroPolicy
from OrbitalAvoidance.orekit_physics import (
    create_propagator, OrbitElements, convert_to_orbit_elements
)
from OrbitalAvoidance.multiagent_env import make_multiagent_env
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
        print(f"✓ Created {name} with OREKIT: altitude={altitude_km}km")
        
        # Current state
        self.current_time_s = 0
        self.position_km = self.calculate_position(0)
        
    def calculate_position(self, time_s):
        """Calculate position at given time using Orekit (STRICT MODE)."""
        # Use Orekit for high-precision propagation
        pos, vel = self.propagator.propagate_orbit(self.orbit_elements, time_s)
        return pos.tolist()
    
    def _calculate_position_simple(self, time_s):
        """REMOVED - No fallback mechanics allowed."""
        raise RuntimeError("❌ Simple mechanics fallback has been disabled. Orekit is required.")
        # Mean anomaly at time t
        M = math.radians(self.mean_anomaly_deg) + self.mean_motion_rad_s * time_s
        
        # Solve Kepler's equation for eccentric anomaly (simplified for low eccentricity)
        E = M + self.eccentricity * math.sin(M)  # First order approximation
        
        # True anomaly
        nu = 2 * math.atan2(
            math.sqrt(1 + self.eccentricity) * math.sin(E / 2),
            math.sqrt(1 - self.eccentricity) * math.cos(E / 2)
        )
        
        # Distance from Earth center
        r = self.semi_major_axis_km * (1 - self.eccentricity * math.cos(E))
        
        # Position in orbital plane
        x_orbital = r * math.cos(nu)
        y_orbital = r * math.sin(nu)
        z_orbital = 0
        
        # Rotate to Earth-centered inertial coordinates
        inc_rad = math.radians(self.inclination_deg)
        raan_rad = math.radians(self.raan_deg)
        omega_rad = math.radians(self.omega_deg)
        
        # Apply rotations (simplified - just inclination and RAAN)
        x = x_orbital * math.cos(raan_rad) - y_orbital * math.sin(raan_rad) * math.cos(inc_rad)
        y = x_orbital * math.sin(raan_rad) + y_orbital * math.cos(raan_rad) * math.cos(inc_rad)
        z = y_orbital * math.sin(inc_rad)
        
        return [x, y, z]
    
    def update(self, dt_s):
        """Update object position by time step."""
        self.current_time_s += dt_s
        self.position_km = self.calculate_position(self.current_time_s)
        return self.position_km


class WebSimulator:
    """Web-based orbital avoidance simulator with realistic orbital mechanics."""
    
    def __init__(self):
        print("Initializing web simulator with realistic orbital mechanics...")
        
        # Create environment for physics reference (but we'll use our own mechanics)
        self.env = make_env(seed=42)
        self.obs, self.info = self.env.reset()
        self.agent = ZeroPolicy()
        
        # Simulation state
        self.running = False
        self.step_count = 0
        self.simulation_time_s = 0
        self.time_step_s = 20.0  # 20 second time steps
        self.total_close_encounters = 0  # Persistent counter for total encounters
        self.last_encounter_step = -1  # Track when last encounter occurred to avoid double counting
        self.total_collisions = 0  # Persistent counter for collisions < 10km
        self.last_collision_step = -1  # Track when last collision occurred
        
        # Create realistic orbiting objects with different parameters
        self.satellite = OrbitingObject(
            name="Satellite",
            altitude_km=400,
            inclination_deg=51.6,
            eccentricity=0.001,
            mean_anomaly_deg=0,
            raan_deg=0
        )
        
        # Create debris with varying orbital parameters for realistic motion
        self.debris_objects = []
        for i in range(8):  # 8 debris objects
            debris = OrbitingObject(
                name=f"Debris_{i}",
                altitude_km=380 + i * 20,  # Different altitudes (380-520 km)
                inclination_deg=51.6 + (i - 4) * 3,  # Different inclinations
                eccentricity=0.002 + i * 0.001,  # Varying eccentricity
                mean_anomaly_deg=i * 45,  # Spread around orbit
                raan_deg=i * 25  # Different orbital planes
            )
            self.debris_objects.append(debris)
        
        print(f"✓ Created satellite and {len(self.debris_objects)} debris objects with realistic orbits")
        print("✓ Environment ready")
    
    def get_current_state(self):
        """Get current simulation state with realistic orbital positions."""
        # Get current positions
        sat_pos = self.satellite.position_km
        debris_positions = [debris.position_km for debris in self.debris_objects]
        
        # Calculate distances
        min_distance = 999.0
        all_distances = []
        current_close_encounters = 0
        current_collisions = 0
        
        if sat_pos and debris_positions:
            sat_np = np.array(sat_pos)
            for debris_pos in debris_positions:
                deb_np = np.array(debris_pos)
                distance = np.linalg.norm(sat_np - deb_np)
                all_distances.append(distance)
                if distance < min_distance:
                    min_distance = distance
                if distance <= 50.0:
                    current_close_encounters += 1
                if distance <= 10.0:  # Collision threshold
                    current_collisions += 1
        
        # Update total encounters if we have a new close encounter and it's not the same step
        if current_close_encounters > 0 and self.last_encounter_step != self.step_count:
            self.total_close_encounters += 1
            self.last_encounter_step = self.step_count
            print(f"🚨 CLOSE ENCOUNTER at step {self.step_count}! Min distance: {min_distance:.1f}km")
        
        # Update total collisions if we have a new collision and it's not the same step
        if current_collisions > 0 and self.last_collision_step != self.step_count:
            self.total_collisions += 1
            self.last_collision_step = self.step_count
            print(f"💥 COLLISION at step {self.step_count}! CRITICAL: {min_distance:.1f}km")
        
        # Debug output every 10 steps
        if self.step_count % 10 == 0:
            print(f"Step {self.step_count}: Satellite at altitude {np.linalg.norm(sat_pos) - 6371:.1f}km")
            if len(all_distances) > 0:
                print(f"  Distances: {[f'{d:.1f}' for d in all_distances[:3]]}... min={min_distance:.1f}km")
                print(f"  Total encounters: {self.total_close_encounters}, Total collisions: {self.total_collisions}")
        
        return {
            "step": self.step_count,
            "running": self.running,
            "altitude": np.linalg.norm(sat_pos) - 6371.0,  # Altitude above Earth surface
            "satellite": sat_pos,
            "debris": debris_positions,
            "min_distance": float(min_distance),
            "close_encounters": self.total_close_encounters,  # Use total persistent counter
            "collisions": self.total_collisions,  # Add collision counter
            "current_encounters": current_close_encounters,  # Current step encounters for debugging
            "current_collisions": current_collisions,  # Current step collisions for debugging
            "all_distances_km": all_distances,
            "simulation_time_s": self.simulation_time_s
        }
    
    def step_simulation(self):
        """Execute one simulation step with realistic orbital mechanics."""
        if not self.running:
            return True  # Don't step if paused
            
        # Update all objects
        self.satellite.update(self.time_step_s)
        for debris in self.debris_objects:
            debris.update(self.time_step_s)
        
        # Update simulation state
        self.step_count += 1
        self.simulation_time_s += self.time_step_s
        
        # Minimal debug output
        if self.step_count % 20 == 0:
            print(f"Simulation time: {self.simulation_time_s/60:.1f} minutes")
        
        return True  # Continue simulation
    
    def reset_simulation(self):
        """Reset simulation to initial state."""
        self.step_count = 0
        self.simulation_time_s = 0
        self.running = False
        self.total_close_encounters = 0  # Reset encounter counter
        self.last_encounter_step = -1  # Reset encounter tracking
        self.total_collisions = 0  # Reset collision counter
        self.last_collision_step = -1  # Reset collision tracking
        
        # Reset all objects to initial positions
        self.satellite.current_time_s = 0
        self.satellite.position_km = self.satellite.calculate_position(0)
        
        for debris in self.debris_objects:
            debris.current_time_s = 0
            debris.position_km = debris.calculate_position(0)
        
        print("Simulation reset")
    
    def toggle_running(self):
        """Toggle simulation running state."""
        self.running = not self.running
        print(f"Simulation {'started' if self.running else 'paused'}")


# Global simulator instance
simulator = WebSimulator()


class WebHandler(BaseHTTPRequestHandler):
    """HTTP request handler for web interface."""
    
    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path
        
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
        else:
            self.send_error(404)
    
    def _serve_html(self):
        """Serve the main HTML interface."""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛰️ Realistic Orbital Avoidance Simulator</title>
    <script src="https://unpkg.com/three@0.145.0/build/three.min.js"></script>
    <script src="https://unpkg.com/three@0.145.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #0c0c1e 0%, #1a1a2e 50%, #16213e 100%);
            font-family: 'Segoe UI', Arial, sans-serif;
            color: white;
            overflow: hidden;
        }
        
        .header {
            position: absolute;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 100;
            text-align: center;
        }
        
        .header h1 {
            margin: 0;
            font-size: 24px;
            background: linear-gradient(45deg, #00d4ff, #ff6b35);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }
        
        .controls {
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 100;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .btn {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            backdrop-filter: blur(10px);
            color: white;
        }
        
        .btn-play {
            background: linear-gradient(45deg, #28a745, #20c997);
            box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
        }
        
        .btn-play:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(40, 167, 69, 0.6);
        }
        
        .btn-step {
            background: linear-gradient(45deg, #17a2b8, #6f42c1);
            box-shadow: 0 4px 15px rgba(23, 162, 184, 0.4);
        }
        
        .btn-step:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(23, 162, 184, 0.6);
        }
        
        .btn-reset {
            background: linear-gradient(45deg, #dc3545, #fd7e14);
            box-shadow: 0 4px 15px rgba(220, 53, 69, 0.4);
        }
        
        .btn-reset:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(220, 53, 69, 0.6);
        }
        
        .status {
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 100;
            background: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            min-width: 250px;
        }
        
        .status h3 {
            margin: 0 0 15px 0;
            color: #00d4ff;
            font-size: 18px;
        }
        
        .status-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 14px;
        }
        
        .status-label {
            color: #aaa;
        }
        
        .status-value {
            color: #fff;
            font-weight: bold;
        }
        
        .warning {
            color: #ff6b35 !important;
            animation: pulse 1s infinite;
        }
        
        .collision {
            color: #ff1744 !important;
            animation: flash 0.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        @keyframes flash {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.2; }
        }
        
        .instructions {
            position: absolute;
            top: 150px;
            right: 20px;
            z-index: 100;
            background: rgba(0, 0, 0, 0.7);
            padding: 15px;
            border-radius: 10px;
            font-size: 14px;
            max-width: 250px;
            backdrop-filter: blur(10px);
        }
        
        .instructions h3 {
            margin: 0 0 10px 0;
            color: #00d4ff;
        }
        
        .instructions ul {
            margin: 0;
            padding-left: 20px;
        }
        
        .instructions li {
            margin-bottom: 5px;
        }
        
        #canvas-container {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;  /* Keep 3D canvas behind UI elements */
        }
        
        .counter-display {
            position: absolute;
            top: 220px;
            left: 20px;
            z-index: 1000;
            background: rgba(0, 30, 50, 0.95);
            border: 2px solid #00d4ff;
            border-radius: 12px;
            padding: 15px;
            min-width: 180px;
            text-align: center;
            backdrop-filter: blur(10px);
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
        }
        
        .counter-section {
            margin-bottom: 15px;
        }
        
        .counter-section:last-child {
            margin-bottom: 0;
        }
        
        .counter-title {
            color: #00d4ff;
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 5px;
        }
        
        .counter-value {
            font-size: 28px;
            font-weight: bold;
            color: #ffffff;
            text-shadow: 0 0 8px rgba(255, 255, 255, 0.4);
            margin: 5px 0;
        }
        
        .counter-subtitle {
            color: #aaaaaa;
            font-size: 12px;
            margin-top: 3px;
        }
        
        .counter-warning {
            color: #ff6b35 !important;
            animation: pulse 1s infinite;
        }
        
        .counter-danger {
            color: #ff1744 !important;
            animation: flash 0.5s infinite;
        }
        
        .counter-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, #00d4ff, transparent);
            margin: 12px 0;
            opacity: 0.5;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛰️ Realistic Orbital Mechanics Simulator</h1>
    </div>
    
    <div class="controls">
        <button id="playBtn" class="btn btn-play" onclick="toggleSimulation()">▶ Play</button>
        <button class="btn btn-step" onclick="stepSimulation()">⏭ Step</button>
        <button class="btn btn-reset" onclick="resetSimulation()">🔄 Reset</button>
    </div>
    
    <div class="counter-display">
        <div class="counter-section">
            <div class="counter-title">🚨 Close Encounters</div>
            <div id="encounter-counter" class="counter-value">0</div>
            <div class="counter-subtitle">Total encounters &lt; 50km</div>
        </div>
        
        <div class="counter-divider"></div>
        
        <div class="counter-section">
            <div class="counter-title">💥 Collisions</div>
            <div id="collision-counter" class="counter-value">0</div>
            <div class="counter-subtitle">Critical zone &lt; 10km</div>
        </div>
        
        <div class="counter-divider"></div>
        
        <div class="counter-section">
            <div class="counter-subtitle">Current min: <span id="current-distance">999.0</span> km</div>
        </div>
    </div>
    
    <div class="status">
        <h3>🛰️ Mission Status</h3>
        <div class="status-item">
            <span class="status-label">Step:</span>
            <span id="step-count" class="status-value">0</span>
        </div>
        <div class="status-item">
            <span class="status-label">Status:</span>
            <span id="status" class="status-value">Paused</span>
        </div>
        <div class="status-item">
            <span class="status-label">Altitude:</span>
            <span id="altitude" class="status-value">400.0 km</span>
        </div>
        <div class="status-item">
            <span class="status-label">Min Distance:</span>
            <span id="min-distance" class="status-value">999.0 km</span>
        </div>
        <div class="status-item">
            <span class="status-label">Close Encounters:</span>
            <span id="encounters" class="status-value">0</span>
        </div>
        <div class="status-item">
            <span class="status-label">Sim Time:</span>
            <span id="sim-time" class="status-value">0.0 min</span>
        </div>
    </div>
    
    <div class="instructions">
        <h3>🎮 Controls</h3>
        <ul>
            <li>🖱️ Mouse: Rotate view</li>
            <li>🎡 Wheel: Zoom in/out</li>
            <li>▶️ Play: Auto advance</li>
            <li>⏭️ Step: Manual advance</li>
            <li>🔄 Reset: Restart</li>
        </ul>
        <p><strong>Realistic Physics:</strong><br/>
        Each object follows independent Keplerian orbits with different periods and inclinations.</p>
    </div>
    
    <div id="canvas-container"></div>
    
    <script>
        let scene, camera, renderer, controls;
        let earth, satellite, debrisObjects = [];
        let animationTimer = null;
        let isRunning = false;
        let lastEncounterCount = 0;  // Track encounters for celebrations
        
        // Initialize 3D scene
        function init3D() {
            console.log('Initializing realistic 3D orbital mechanics...');
            
            // Create scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000510);
            
            // Create camera
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 50000);
            camera.position.set(0, 0, 15000);
            
            // Create renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            
            const container = document.getElementById('canvas-container');
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
            createEarth();
            
            // Create stars
            createStars();
            
            // Add lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
            scene.add(ambientLight);
            
            const sunLight = new THREE.DirectionalLight(0xffffff, 1);
            sunLight.position.set(10000, 0, 0);
            sunLight.castShadow = true;
            scene.add(sunLight);
            
            console.log('✓ 3D scene initialized with realistic orbital mechanics');
            
            // Start render loop
            animate();
            
            // Initial state fetch
            fetchState();
        }
        
        function createEarth() {
            const earthGeometry = new THREE.SphereGeometry(6371, 64, 32);
            const earthMaterial = new THREE.MeshPhongMaterial({
                color: 0x2194ce,
                shininess: 0.8,
                transparent: true,
                opacity: 0.9
            });
            earth = new THREE.Mesh(earthGeometry, earthMaterial);
            earth.castShadow = true;
            earth.receiveShadow = true;
            scene.add(earth);
        }
        
        function createStars() {
            const starsGeometry = new THREE.BufferGeometry();
            const starsMaterial = new THREE.PointsMaterial({
                color: 0xffffff,
                size: 2,
                sizeAttenuation: false
            });
            
            const starsVertices = [];
            for (let i = 0; i < 2000; i++) {
                const x = (Math.random() - 0.5) * 40000;
                const y = (Math.random() - 0.5) * 40000;
                const z = (Math.random() - 0.5) * 40000;
                starsVertices.push(x, y, z);
            }
            
            starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starsVertices, 3));
            const stars = new THREE.Points(starsGeometry, starsMaterial);
            scene.add(stars);
        }
        
        function updateDisplay(data) {
            console.log('Updating display with realistic orbital data:', {
                step: data.step,
                running: data.running,
                min_distance: data.min_distance,
                sim_time: data.simulation_time_s
            });
            
            // Update UI
            document.getElementById('step-count').textContent = data.step;
            document.getElementById('min-distance').textContent = data.min_distance.toFixed(1) + ' km';
            document.getElementById('encounters').textContent = data.close_encounters;
            document.getElementById('altitude').textContent = data.altitude.toFixed(1) + ' km';
            document.getElementById('sim-time').textContent = (data.simulation_time_s / 60).toFixed(1) + ' min';
            
            // Update prominent counter displays
            document.getElementById('encounter-counter').textContent = data.close_encounters;
            document.getElementById('current-distance').textContent = data.min_distance.toFixed(1);
            document.getElementById('collision-counter').textContent = data.collisions;
            
            // Celebration effect for new encounters
            if (data.close_encounters > lastEncounterCount) {
                console.log('🚨 NEW CLOSE ENCOUNTER DETECTED!');
                const counter = document.getElementById('encounter-counter');
                counter.style.transform = 'scale(1.3)';
                counter.style.textShadow = '0 0 20px #ff1744';
                setTimeout(() => {
                    counter.style.transform = 'scale(1)';
                    counter.style.textShadow = '0 0 10px rgba(255, 255, 255, 0.5)';
                }, 1000);
                lastEncounterCount = data.close_encounters;
            }
            
            // Collision celebration effect
            if (data.collisions > (window.lastCollisionCount || 0)) {
                console.log('💥 NEW COLLISION DETECTED!');
                const collisionCounter = document.getElementById('collision-counter');
                collisionCounter.style.transform = 'scale(1.5)';
                collisionCounter.style.textShadow = '0 0 30px #ff1744';
                setTimeout(() => {
                    collisionCounter.style.transform = 'scale(1)';
                    collisionCounter.style.textShadow = '0 0 10px rgba(255, 255, 255, 0.5)';
                }, 1500);
                window.lastCollisionCount = data.collisions;
            }
            
            // Update status
            const statusEl = document.getElementById('status');
            const minDistEl = document.getElementById('min-distance');
            const encountersEl = document.getElementById('encounters');
            const prominentCounter = document.getElementById('encounter-counter');
            const collisionCounter = document.getElementById('collision-counter');
            
            if (data.running) {
                statusEl.textContent = 'Running';
                statusEl.className = 'status-value';
            } else {
                statusEl.textContent = 'Paused';
                statusEl.className = 'status-value';
            }
            
            // Color code distance warnings with enhanced feedback
            if (data.min_distance < 10) {
                // COLLISION ZONE - RED CRITICAL
                minDistEl.className = 'status-value collision';
                encountersEl.className = 'status-value collision';
                prominentCounter.className = 'counter-value counter-danger';
                collisionCounter.className = 'collision-value collision-critical';
                minDistEl.style.animation = 'flash 0.2s infinite';
            } else if (data.min_distance < 50) {
                // CLOSE ENCOUNTER - RED
                minDistEl.className = 'status-value collision';
                encountersEl.className = 'status-value collision';
                prominentCounter.className = 'counter-value counter-danger';
                collisionCounter.className = 'collision-value';
                minDistEl.style.animation = 'flash 0.5s infinite';
            } else if (data.min_distance < 100) {
                // WARNING ZONE - ORANGE
                minDistEl.className = 'status-value warning';
                encountersEl.className = 'status-value';
                prominentCounter.className = 'counter-value counter-warning';
                collisionCounter.className = 'collision-value';
                minDistEl.style.animation = 'pulse 1.5s infinite';
            } else {
                // SAFE ZONE - WHITE
                minDistEl.className = 'status-value';
                encountersEl.className = 'status-value';
                prominentCounter.className = 'counter-value';
                collisionCounter.className = 'collision-value';
                minDistEl.style.animation = 'none';
            }
            
            // Update play button
            const playBtn = document.getElementById('playBtn');
            if (data.running) {
                playBtn.textContent = '⏸ Pause';
                playBtn.className = 'btn btn-play';
            } else {
                playBtn.textContent = '▶ Play';
                playBtn.className = 'btn btn-play';
            }
            
            // Update 3D objects
            update3DObjects(data);
        }
        
        function update3DObjects(data) {
            // Update satellite
            if (data.satellite && satellite) {
                satellite.position.set(data.satellite[0], data.satellite[1], data.satellite[2]);
            } else if (data.satellite && !satellite) {
                // Create satellite
                const satGeometry = new THREE.SphereGeometry(100, 16, 8);
                const satMaterial = new THREE.MeshPhongMaterial({ color: 0x00ff00, emissive: 0x002200 });
                satellite = new THREE.Mesh(satGeometry, satMaterial);
                satellite.position.set(data.satellite[0], data.satellite[1], data.satellite[2]);
                scene.add(satellite);
            }
            
            // Update debris
            data.debris.forEach((pos, i) => {
                if (debrisObjects[i]) {
                    debrisObjects[i].position.set(pos[0], pos[1], pos[2]);
                } else {
                    // Create debris object
                    const debGeometry = new THREE.SphereGeometry(50, 8, 6);
                    const debMaterial = new THREE.MeshPhongMaterial({ color: 0xff4444, emissive: 0x220000 });
                    const debris = new THREE.Mesh(debGeometry, debMaterial);
                    debris.position.set(pos[0], pos[1], pos[2]);
                    debrisObjects[i] = debris;
                    scene.add(debris);
                }
            });
        }
        
        function animate() {
            requestAnimationFrame(animate);
            
            // Rotate Earth slowly
            if (earth) {
                earth.rotation.y += 0.002;
            }
            
            if (controls) {
                controls.update();
            }
            
            if (renderer && scene && camera) {
                renderer.render(scene, camera);
            }
        }
        
        function onWindowResize() {
            if (camera && renderer) {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }
        }
        
        window.addEventListener('resize', onWindowResize, false);
        
        // API functions
        function fetchState() {
            fetch('/api/state')
                .then(response => response.json())
                .then(data => updateDisplay(data))
                .catch(error => console.error('Error:', error));
        }
        
        function toggleSimulation() {
            console.log('Toggle button clicked, current isRunning:', isRunning);
            fetch('/api/toggle')
                .then(response => response.json())
                .then(data => {
                    console.log('Toggle response:', data);
                    updateDisplay(data);
                    isRunning = data.running;
                    if (isRunning && !animationTimer) {
                        console.log('Starting animation');
                        startAnimation();
                    } else if (!isRunning && animationTimer) {
                        console.log('Stopping animation');
                        stopAnimation();
                    }
                })
                .catch(error => console.error('Toggle error:', error));
        }
        
        function stepSimulation() {
            console.log('Step simulation called');
            fetch('/api/step')
                .then(response => response.json())
                .then(data => {
                    console.log('Step response received');
                    updateDisplay(data);
                })
                .catch(error => console.error('Step error:', error));
        }
        
        function resetSimulation() {
            stopAnimation();
            fetch('/api/reset')
                .then(response => response.json())
                .then(data => updateDisplay(data));
        }
        
        function startAnimation() {
            console.log('Starting animation timer');
            if (animationTimer) {
                console.log('Animation already running, stopping previous timer');
                clearInterval(animationTimer);
            }
            animationTimer = setInterval(() => {
                console.log('Animation tick, isRunning:', isRunning);
                if (isRunning) {
                    console.log('Calling stepSimulation from animation');
                    stepSimulation();
                } else {
                    console.log('Not running, stopping animation');
                    stopAnimation();
                }
            }, 1000); // 1 second intervals for stable animation
        }
        
        function stopAnimation() {
            if (animationTimer) {
                console.log('Stopping animation timer');
                clearInterval(animationTimer);
                animationTimer = null;
            }
        }
        
        // Initialize when page loads
        window.addEventListener('DOMContentLoaded', function() {
            console.log('DOM loaded, initializing realistic orbital simulator...');
            try {
                init3D();
            } catch (error) {
                console.error('Failed to initialize 3D scene:', error);
            }
        });
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
        state = simulator.get_current_state()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(state).encode())
    
    def log_message(self, format, *args):
        # Suppress HTTP request logging
        pass


def run_web_server(port=8080):
    """Run the web server."""
    server = HTTPServer(('localhost', port), WebHandler)
    print(f"🌐 Realistic orbital simulator running at: http://localhost:{port}")
    print("Open this URL in your web browser to use the simulator")
    print("Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server...")
        server.shutdown()


def main():
    """Main function."""
    print("=" * 60)
    print("REALISTIC ORBITAL AVOIDANCE SIMULATOR")
    print("=" * 60)
    print("Features independent orbital mechanics for each object!")
    print("Each satellite and debris follows realistic Keplerian orbits.")
    print("=" * 60)
    
    run_web_server()


if __name__ == "__main__":
    main()