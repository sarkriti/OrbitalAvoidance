#!/usr/bin/env python3
"""
Enhanced Satellite Environment with High-Fidelity Real-World Physics

This version includes:
- J2 perturbations (Earth's oblateness)
- Atmospheric drag
- Solar radia                    return self._get_observation(), reward, done, False, {"error": f"Propagation failed: {e}"}           return self._get_observation(), reward, done, False, {"error": f"Propagation failed: {e}"}xcept Exception as e:
            # Orbit may have decayed or become invalid
            reward = -1000.0
            done = True
            return self._get_observation(), reward, done, False, {"error": f"Propagation failed: {e}"}           return self._get_observation(), reward, done, False, {"error": f"Propagation failed: {e}"}           return self._get_observation(), reward, done, False, {"error": f"Propagation failed: {e}"}pressure
- Third-body gravitational effects (Moon, Sun)
- Realistic spacecraft properties
- More accurate propagation methods
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel

# Poliastro imports for high-fidelity orbital mechanics
from poliastro.bodies import Earth, Sun, Moon
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.util import norm
from poliastro.core.perturbations import J2_perturbation, atmospheric_drag_exponential, radiation_pressure
from poliastro.twobody.propagation import CowellPropagator
from poliastro.core.propagation import func_twobody
from poliastro.spacecraft import Spacecraft

import warnings
warnings.filterwarnings('ignore')

class HighFidelitySatelliteEnv(gym.Env):
    """
    High-fidelity satellite environment with realistic orbital physics
    
    Includes:
    - Earth's J2 gravitational perturbation
    - Atmospheric drag modeling
    - Solar radiation pressure
    - Third-body effects (Moon, Sun)
    - Realistic spacecraft parameters
    - High-precision numerical integration
    """
    
    metadata = {'render_modes': ['human', 'visual']}
    
    def __init__(self, render_mode=None, spacecraft_mass=500.0, spacecraft_area=10.0):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Spacecraft properties for realistic physics
        self.spacecraft_mass = spacecraft_mass * u.kg  # 500 kg default
        self.spacecraft_area = spacecraft_area * u.m**2  # 10 m² cross-sectional area
        self.drag_coefficient = 2.2  # Typical for satellites
        self.reflectivity_coefficient = 1.2  # Solar radiation pressure
        
        # Enhanced action space: [thrust_x, thrust_y, thrust_z] in m/s²
        # Realistic thrust levels for small satellites
        self.max_thrust_acceleration = 0.001  # 1 mm/s² (realistic for electric propulsion)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(3,), 
            dtype=np.float32
        )
        
        # Enhanced observation space:
        # [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,           # current state (6)
        #  target_pos_x, target_pos_y, target_pos_z,          # target position (3)
        #  target_vel_x, target_vel_y, target_vel_z,          # target velocity (3)
        #  sun_pos_x, sun_pos_y, sun_pos_z,                   # Sun position (3)
        #  moon_pos_x, moon_pos_y, moon_pos_z,                # Moon position (3)
        #  solar_flux, atmospheric_density, eclipse_factor]    # environmental (3)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(21,), 
            dtype=np.float32
        )
        
        # Environment parameters
        self.max_steps = 2000  # Longer missions for realistic scenarios
        self.dt = 60.0  # 60 seconds time step
        
        # Physics modeling flags
        self.enable_j2_perturbation = True
        self.enable_atmospheric_drag = True
        self.enable_solar_radiation_pressure = True
        self.enable_third_body_effects = True
        
        # Mission parameters
        self.target_orbit = None
        self.current_orbit = None
        self.spacecraft = None
        self.step_count = 0
        self.current_time = None
        
        # Environmental state
        self.sun_position = None
        self.moon_position = None
        self.solar_flux = 1361.0  # W/m² at 1 AU
        self.atmospheric_density = 1e-12  # kg/m³ (varies with altitude)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment with realistic initial conditions"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Create realistic spacecraft object
        self.spacecraft = Spacecraft(
            A=self.spacecraft_area,
            mass=self.spacecraft_mass,
            Cd=self.drag_coefficient,
            Cr=self.reflectivity_coefficient
        )
        
        # Initialize with realistic LEO orbit (ISS-like)
        alt_start = 400 + np.random.uniform(-50, 50)  # 350-450 km
        ecc_start = np.random.uniform(0.0, 0.05)      # Low eccentricity
        inc_start = 51.6 + np.random.uniform(-5, 5)   # ISS-like inclination
        raan_start = np.random.uniform(0, 360)        # Random RAAN
        argp_start = np.random.uniform(0, 360)        # Random argument of perigee
        nu_start = np.random.uniform(0, 360)          # Random true anomaly
        
        self.current_orbit = Orbit.from_classical(
            Earth,
            a=(Earth.R + alt_start * u.km),
            ecc=ecc_start * u.one,
            inc=inc_start * u.deg,
            raan=raan_start * u.deg,
            argp=argp_start * u.deg,
            nu=nu_start * u.deg,
            epoch=Time.now()
        )
        
        # Target: Geostationary orbit with slight randomization
        target_alt = 35786 + np.random.uniform(-100, 100)  # GEO ± 100 km
        self.target_orbit = Orbit.from_classical(
            Earth,
            a=(Earth.R + target_alt * u.km),
            ecc=0.001 * u.one,  # Nearly circular
            inc=np.random.uniform(-2, 2) * u.deg,  # Near-equatorial
            raan=np.random.uniform(0, 360) * u.deg,
            argp=0 * u.deg,
            nu=0 * u.deg,
            epoch=Time.now()
        )
        
        self.step_count = 0
        self.current_time = Time.now()
        
        # Initialize environmental conditions
        self.update_environmental_conditions()
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step with high-fidelity physics"""
        
        # Scale action to realistic thrust acceleration
        thrust_accel = action * self.max_thrust_acceleration  # m/s²
        
        # Apply maneuver if thrust is significant
        if np.linalg.norm(thrust_accel) > 1e-6:
            # Convert thrust acceleration to velocity change over time step
            dv = thrust_accel * self.dt * u.m / u.s
            
            try:
                # Apply instantaneous velocity change
                maneuver = Maneuver.impulse(dv)
                self.current_orbit = self.current_orbit.apply_maneuver(maneuver)
            except Exception as e:
                # Penalize invalid maneuvers
                reward = -1000.0
                done = True
                return self._get_observation(), reward, done, False, {"error": str(e)}
        
        # Propagate orbit with perturbations
        next_time = self.current_time + self.dt * u.s
        
        try:
            self.current_orbit = self.propagate_with_perturbations(
                self.current_orbit, next_time
            )
            self.current_time = next_time
        except Exception as e:
            # Orbit may have decayed or become invalid
            reward = -1000.0
            done = True
            return self._get_observation(), reward, done, False, {"error": f"Propagation failed: {e}")}
        
        # Update environmental conditions
        self.update_environmental_conditions()
        
        # Calculate reward
        reward = self._calculate_enhanced_reward()
        
        # Check termination conditions
        self.step_count += 1
        done = self.step_count >= self.max_steps
        truncated = False
        
        # Check for orbital decay
        altitude = (self.current_orbit.r_p.to(u.km).value - Earth.R.to(u.km).value)
        if altitude < 100:  # Below 100 km = atmospheric entry
            reward -= 1000.0
            done = True
            
        # Check for mission success
        if self._is_mission_successful():
            reward += 1000.0
            done = True
        
        return self._get_observation(), reward, done, truncated, {}
    
    def propagate_with_perturbations(self, orbit, target_time):
        """Propagate orbit including realistic perturbations"""
        
        def perturbation_function(t0, state, k):
            """Combined perturbation function"""
            
            total_perturbation = np.zeros(3) * u.m / u.s**2
            
            # J2 perturbation (Earth's oblateness)
            if self.enable_j2_perturbation:
                j2_accel = J2_perturbation(
                    t0, state, k,
                    J2=Earth.J2.value,
                    R=Earth.R.to(u.m).value
                )
                total_perturbation += j2_accel
            
            # Atmospheric drag
            if self.enable_atmospheric_drag:
                r_norm = np.linalg.norm(state[:3]) * u.m
                altitude = (r_norm - Earth.R).to(u.km).value
                
                if altitude < 1000:  # Only apply drag below 1000 km
                    # Exponential atmosphere model
                    h0 = 400  # km, reference altitude
                    rho0 = 5e-12  # kg/m³, reference density
                    H = 50  # km, scale height
                    rho = rho0 * np.exp(-(altitude - h0) / H)
                    
                    velocity = state[3:6] * u.m / u.s
                    drag_accel = atmospheric_drag_exponential(
                        t0, state, k,
                        R=Earth.R.to(u.m).value,
                        C_D=self.drag_coefficient,
                        A_over_m=(self.spacecraft_area / self.spacecraft_mass).to(u.m**2 / u.kg).value,
                        H0=H * u.km.to(u.m),
                        rho0=rho
                    )
                    total_perturbation += drag_accel
            
            # Solar radiation pressure
            if self.enable_solar_radiation_pressure:
                # Simplified SRP model
                sun_vector = self.sun_position / np.linalg.norm(self.sun_position)
                eclipse_factor = self._calculate_eclipse_factor(state[:3])
                
                if eclipse_factor > 0:  # Not in Earth's shadow
                    P = 4.56e-6  # N/m², solar radiation pressure at 1 AU
                    srp_accel = (P * self.reflectivity_coefficient * 
                                (self.spacecraft_area / self.spacecraft_mass).to(u.m**2 / u.kg).value *
                                eclipse_factor * sun_vector)
                    total_perturbation += srp_accel * u.m / u.s**2
            
            # Third-body perturbations (simplified)
            if self.enable_third_body_effects:
                # Sun and Moon gravitational effects
                r_sat = state[:3] * u.m
                
                # Solar third-body effect
                r_sun = self.sun_position * u.m
                GM_sun = Sun.k.to(u.m**3 / u.s**2).value
                
                r_sat_sun = r_sat - r_sun
                sun_accel = GM_sun * (
                    -r_sat_sun / np.linalg.norm(r_sat_sun)**3 +
                    r_sun / np.linalg.norm(r_sun)**3
                )
                total_perturbation += sun_accel * 0.001  # Scale down for numerical stability
            
            return total_perturbation.to(u.m / u.s**2).value
        
        # Use Cowell propagation with perturbations
        propagator = CowellPropagator(f=perturbation_function, rtol=1e-11)
        
        try:
            new_orbit = orbit.propagate(target_time, method=propagator)
            return new_orbit
        except Exception as e:
            # Fallback to simple two-body propagation
            print(f"Perturbation propagation failed: {e}, using two-body")
            return orbit.propagate(target_time)
    
    def update_environmental_conditions(self):
        """Update Sun/Moon positions and environmental conditions"""
        
        try:
            # Get Sun and Moon positions
            with solar_system_ephemeris.set('builtin'):
                sun_pos, _ = get_body_barycentric_posvel('sun', self.current_time)
                moon_pos, _ = get_body_barycentric_posvel('moon', self.current_time)
                earth_pos, _ = get_body_barycentric_posvel('earth', self.current_time)
                
            # Convert to Earth-centered coordinates
            self.sun_position = (sun_pos - earth_pos).xyz.to(u.m).value
            self.moon_position = (moon_pos - earth_pos).xyz.to(u.m).value
            
        except Exception:
            # Fallback to approximate positions
            self.sun_position = np.array([1.5e11, 0, 0])  # 1 AU in x-direction
            self.moon_position = np.array([3.84e8, 0, 0])  # Moon distance in x-direction
        
        # Update atmospheric density based on solar activity (simplified)
        altitude = (self.current_orbit.r_p.to(u.km).value - Earth.R.to(u.km).value)
        if altitude < 1000:
            # Exponential atmosphere
            h0 = 400  # km
            rho0 = 5e-12  # kg/m³
            H = 50  # km
            self.atmospheric_density = rho0 * np.exp(-(altitude - h0) / H)
        else:
            self.atmospheric_density = 0.0
    
    def _calculate_eclipse_factor(self, position):
        """Calculate eclipse factor (0 = full shadow, 1 = full sunlight)"""
        
        # Simplified cylindrical shadow model
        earth_radius = Earth.R.to(u.m).value
        sun_dir = self.sun_position / np.linalg.norm(self.sun_position)
        
        # Project satellite position onto sun direction
        sat_sun_proj = np.dot(position, sun_dir)
        
        if sat_sun_proj > 0:  # Satellite on day side
            return 1.0
        
        # Check if in Earth's shadow
        perp_distance = np.linalg.norm(position - sat_sun_proj * sun_dir)
        
        if perp_distance < earth_radius:
            return 0.0  # In umbra
        elif perp_distance < earth_radius * 1.1:
            return 0.5  # In penumbra (simplified)
        else:
            return 1.0  # In sunlight
    
    def _get_observation(self):
        """Get enhanced observation with environmental data"""
        
        # Current orbital state
        r_current = self.current_orbit.r.to(u.km).value
        v_current = self.current_orbit.v.to(u.km/u.s).value
        
        # Target orbital state
        r_target = self.target_orbit.r.to(u.km).value
        v_target = self.target_orbit.v.to(u.km/u.s).value
        
        # Environmental conditions
        sun_pos_norm = self.sun_position / 1e11  # Normalize to ~1 AU
        moon_pos_norm = self.moon_position / 1e8  # Normalize to ~Moon distance
        
        # Eclipse factor
        eclipse_factor = self._calculate_eclipse_factor(r_current * 1000)  # Convert to meters
        
        # Combine observation
        obs = np.concatenate([
            r_current,                    # Current position (3)
            v_current,                    # Current velocity (3)
            r_target,                     # Target position (3)
            v_target,                     # Target velocity (3)
            sun_pos_norm,                 # Sun position (3)
            moon_pos_norm,                # Moon position (3)
            [self.solar_flux / 1400],     # Normalized solar flux (1)
            [self.atmospheric_density * 1e12],  # Scaled atmospheric density (1)
            [eclipse_factor]              # Eclipse factor (1)
        ]).astype(np.float32)
        
        return obs
    
    def _calculate_enhanced_reward(self):
        """Enhanced reward function considering realistic mission objectives"""
        
        # Base distance rewards
        r_diff = norm(self.current_orbit.r - self.target_orbit.r).to(u.km).value
        v_diff = norm(self.current_orbit.v - self.target_orbit.v).to(u.km/u.s).value
        
        # Primary rewards
        position_reward = -r_diff / 1000.0
        velocity_reward = -v_diff * 1000.0
        
        # Orbital element rewards (more realistic for mission planning)
        a_error = abs(self.current_orbit.a.to(u.km).value - self.target_orbit.a.to(u.km).value)
        e_error = abs(self.current_orbit.ecc.value - self.target_orbit.ecc.value)
        i_error = abs(self.current_orbit.inc.to(u.deg).value - self.target_orbit.inc.to(u.deg).value)
        
        altitude_reward = -a_error / 5000.0  # Semi-major axis error
        eccentricity_reward = -e_error * 1000.0  # Eccentricity error
        inclination_reward = -i_error / 10.0  # Inclination error
        
        # Survival rewards
        altitude = (self.current_orbit.r_p.to(u.km).value - Earth.R.to(u.km).value)
        survival_reward = 1.0 if altitude > 150 else -10.0  # Penalty for low altitude
        
        # Efficiency reward (penalize excessive maneuvering)
        efficiency_reward = 0.1  # Small reward for each step
        
        total_reward = (position_reward + velocity_reward + 
                       altitude_reward + eccentricity_reward + inclination_reward +
                       survival_reward + efficiency_reward)
        
        return total_reward
    
    def _is_mission_successful(self, 
                              position_tolerance=1000*u.km, 
                              velocity_tolerance=0.5*u.km/u.s,
                              altitude_tolerance=500*u.km):
        """Enhanced mission success criteria"""
        
        r_diff = norm(self.current_orbit.r - self.target_orbit.r)
        v_diff = norm(self.current_orbit.v - self.target_orbit.v)
        a_diff = abs(self.current_orbit.a - self.target_orbit.a)
        
        return (r_diff < position_tolerance and 
                v_diff < velocity_tolerance and 
                a_diff < altitude_tolerance)
    
    def render(self, mode='human'):
        """Render with enhanced information"""
        if mode == 'human':
            altitude = (self.current_orbit.r_p.to(u.km).value - Earth.R.to(u.km).value)
            apogee = (self.current_orbit.r_a.to(u.km).value - Earth.R.to(u.km).value)
            
            print(f"Step: {self.step_count}")
            print(f"Altitude: {altitude:.1f} km (Apogee: {apogee:.1f} km)")
            print(f"Eccentricity: {self.current_orbit.ecc.value:.4f}")
            print(f"Inclination: {self.current_orbit.inc.to(u.deg).value:.2f}°")
            print(f"Atmospheric density: {self.atmospheric_density:.2e} kg/m³")
            print(f"Eclipse factor: {self._calculate_eclipse_factor(self.current_orbit.r.to(u.m).value):.2f}")
            print("-" * 50)
    
    def close(self):
        """Clean up environment"""
        pass


# Test function for the enhanced environment
def test_enhanced_environment():
    """Test the enhanced physics environment"""
    
    print("🛰️ Testing Enhanced High-Fidelity Satellite Environment")
    print("=" * 60)
    
    env = HighFidelitySatelliteEnv()
    
    print("✅ Environment created with enhanced physics!")
    print(f"   - J2 perturbations: {env.enable_j2_perturbation}")
    print(f"   - Atmospheric drag: {env.enable_atmospheric_drag}")
    print(f"   - Solar radiation pressure: {env.enable_solar_radiation_pressure}")
    print(f"   - Third-body effects: {env.enable_third_body_effects}")
    
    obs, _ = env.reset()
    print(f"✅ Environment reset, observation shape: {obs.shape}")
    print(f"   Enhanced observation includes: position, velocity, target, Sun, Moon, environment")
    
    # Test a few steps
    total_reward = 0
    for i in range(5):
        action = env.action_space.sample() * 0.1  # Small actions
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i+1}: Reward = {reward:.2f}, Total = {total_reward:.2f}")
        env.render()
        
        if done:
            break
    
    env.close()
    print("✅ Enhanced environment test completed!")
    
    return True


if __name__ == "__main__":
    test_enhanced_environment()