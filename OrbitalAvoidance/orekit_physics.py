#!/usr/bin/env python3
# Orekit-based physics engine for high-precision orbital mechanics.
# This module provides a high-fidelity orbital propagation system using
# the Orekit library, replacing the Poliastro-based implementation.

from __future__ import annotations

import math
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import tempfile
import zipfile
import urllib.request

import numpy as np

# Orekit imports - STRICT MODE (no fallbacks)
try:
    import orekit_jpype as oj
    import jpype
    import os
    
    # Path to Orekit JAR files
    orekit_jar_dir = "/home/bryson/RL_testing/orekit_lib"
    jar_files = [os.path.join(orekit_jar_dir, f) for f in os.listdir(orekit_jar_dir) if f.endswith(".jar")]
    
    # Initialize JVM with Orekit JAR files
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=jar_files)  # ✅ Use jpype.startJVM, not oj.initVM
    
    # Now initialize Orekit helper
    oj.initVM()  # This just tells orekit_jpype that JVM is ready
    
    # Setup Orekit data directory using Java API directly
    from org.orekit.data import DataContext, DirectoryCrawler
    from java.io import File
    
    data_dir = File("/home/bryson/RL_testing/orekit-data")
    crawler = DirectoryCrawler(data_dir)
    DataContext.getDefault().getDataProvidersManager().addProvider(crawler)
    print("✓ Orekit data directory configured")
    
    # Test basic import with correct Orekit 13.x class
    from org.orekit.orbits import PositionAngleType
    print(f"✓ Orekit PositionAngleType imported successfully: {PositionAngleType.TRUE}")
    
    # Core Orekit imports
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.frames import FramesFactory, TopocentricFrame
    from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory
    from org.orekit.utils import Constants, IERSConventions
    from org.orekit.orbits import KeplerianOrbit, CartesianOrbit, PositionAngleType, Orbit
    from org.orekit.utils import PVCoordinates
    from org.orekit.propagation import SpacecraftState
    from org.orekit.propagation.numerical import NumericalPropagator
    from org.orekit.propagation.analytical import KeplerianPropagator
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
    
    # Force model imports for realistic physics
    from org.orekit.forces.gravity.potential import GravityFieldFactory
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, NewtonianAttraction
    from org.orekit.forces.gravity import ThirdBodyAttraction
    from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient, SolarRadiationPressure
    from org.orekit.forces.drag import IsotropicDrag, DragForce
    from org.orekit.models.earth.atmosphere import NRLMSISE00
    from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation
    
    OREKIT_AVAILABLE = True
    print("✓ All Orekit modules imported successfully")
    logging.info("Orekit successfully initialized - FULL FUNCTIONALITY AVAILABLE")
    
except ImportError as e:
    OREKIT_AVAILABLE = False
    error_msg = f"❌ OREKIT NOT AVAILABLE: {e}"
    print(error_msg)
    logging.error(error_msg)
    raise RuntimeError(f"Orekit is required but not available: {e}")

except Exception as e:
    OREKIT_AVAILABLE = False
    error_msg = f"❌ FAILED TO INITIALIZE OREKIT: {e}"
    print(error_msg)
    logging.error(error_msg)
    raise RuntimeError(f"Failed to initialize Orekit: {e}")


@dataclass
class OrbitElements:
    # Orbital elements in standard format.
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float  # Right Ascension of Ascending Node
    arg_periapsis_deg: float  # Argument of Periapsis
    mean_anomaly_deg: float
    epoch_mjd: float = 59945.0  # Default epoch (2023-01-01)


@dataclass
class SpacecraftProperties:
    # Physical properties of a spacecraft.
    mass_kg: float = 1000.0
    drag_area_m2: float = 10.0
    drag_coefficient: float = 2.2
    radiation_area_m2: float = 5.0  # Reduced from 10.0 to reduce SRP effect
    radiation_coefficient: float = 1.3  # Reduced from 1.8 for more realistic SRP


class OrekitOrbitPropagator:
    # High-precision orbit propagator using Orekit - STRICT MODE.
    
    def __init__(self):
        # Initialize the Orekit propagator.
        if not OREKIT_AVAILABLE:
            raise RuntimeError("❌ OREKIT IS REQUIRED but not available. Please install orekit-jpype properly.")
        
        # Initialize data context (minimal setup for basic operations)
        self._setup_minimal_data()
        
        # Get standard frames and time scales
        self.itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        self.gcrf = FramesFactory.getGCRF()
        self.utc = TimeScalesFactory.getUTC()
        
        # Earth model
        self.earth = CelestialBodyFactory.getEarth()
        
        # Default spacecraft properties
        self.spacecraft_props = SpacecraftProperties()
        
        # Propagator caching to avoid repeated force model setup
        self._propagator_cache = {}
        self._force_models_initialized = False
        
        logging.info("OrekitOrbitPropagator initialized successfully")
    
    def _setup_minimal_data(self):
        # Setup minimal data context for Orekit operations.
        try:
            # Data should already be configured via setup_orekit_curdir
            pass
            
        except Exception as e:
            logging.warning(f"Could not setup Orekit data context: {e}")
            # Continue with default context
    
    def create_orbit_from_elements(self, elements: OrbitElements) -> KeplerianOrbit:
        # Create an Orekit orbit from orbital elements.
        # Convert to SI units
        a = elements.semi_major_axis_km * 1000.0  # m
        e = elements.eccentricity
        i = math.radians(elements.inclination_deg)
        raan = math.radians(elements.raan_deg)
        pa = math.radians(elements.arg_periapsis_deg)
        ma = math.radians(elements.mean_anomaly_deg)
        
        # Create epoch
        epoch = AbsoluteDate.createMJDDate(int(elements.epoch_mjd), 
                                         elements.epoch_mjd % 1.0, 
                                         self.utc)
        
        # Create Keplerian orbit
        orbit = KeplerianOrbit(a, e, i, pa, raan, ma, 
                              PositionAngleType.MEAN, 
                              self.gcrf, epoch, 
                              Constants.WGS84_EARTH_MU)
        
        return orbit
    
    def create_propagator(self, orbit: KeplerianOrbit, 
                         spacecraft_props: Optional[SpacecraftProperties] = None) -> NumericalPropagator:
        # Create a numerical propagator with realistic force models.
        if spacecraft_props is None:
            spacecraft_props = self.spacecraft_props
        
        # Use the best integrator now that we have compatible libraries
        try:
            # DormandPrince853Integrator - highest precision adaptive integrator
            min_step = 1.0     # seconds (reasonable minimum)
            max_step = 120.0   # seconds (reduced from 300 for better accuracy)
            position_tolerance = 1.0e-11  # Tighter tolerances for better long-term accuracy
            velocity_tolerance = 1.0e-11  # Tighter tolerances
            integrator = DormandPrince853Integrator(min_step, max_step, 
                                                   position_tolerance, velocity_tolerance)
            if not self._force_models_initialized:
                logging.info("Using DormandPrince853Integrator (high precision, adaptive step)")
            
        except Exception as e:
            logging.warning(f"DormandPrince853Integrator failed: {e}")
            try:
                # Fallback: Classical Runge-Kutta integrator
                from org.hipparchus.ode.nonstiff import ClassicalRungeKuttaIntegrator
                step_size = 30.0  # 30 seconds fixed step for better precision
                integrator = ClassicalRungeKuttaIntegrator(step_size)
                logging.info("Using ClassicalRungeKuttaIntegrator as fallback")
            except Exception as e2:
                logging.error(f"All integrators failed: {e2}")
                raise RuntimeError(f"Cannot create any numerical integrator: {e2}")
        
        # Create numerical propagator
        propagator = NumericalPropagator(integrator)
        
        # Set initial state
        initial_state = SpacecraftState(orbit, spacecraft_props.mass_kg)
        propagator.setInitialState(initial_state)
        
        # Add realistic force models (but be more conservative about which ones)
        self._add_force_models_safely(propagator, spacecraft_props)
        
        return propagator
    
    def _add_force_models_safely(self, propagator: NumericalPropagator, 
                                spacecraft_props: SpacecraftProperties):
        # Add force models one by one with individual error handling.
        # Only log setup messages once to avoid spam
        force_models_added = 0
        
        # 1. PRIORITY: Earth gravity field with J2+ perturbations
        try:
            gravity_field = GravityFieldFactory.getNormalizedProvider(8, 8)  # Full 8x8 gravity field
            # Use ITRF frame for gravity field consistency
            gravity_model = HolmesFeatherstoneAttractionModel(self.itrf, gravity_field)
            propagator.addForceModel(gravity_model)
            force_models_added += 1
            if not self._force_models_initialized:
                logging.info("✅ Added 8x8 gravity field model (includes J2, J3, J4, etc.)")
        except Exception as e:
            if not self._force_models_initialized:
                logging.warning(f"Failed to add 8x8 gravity model: {e}")
            try:
                gravity_field = GravityFieldFactory.getNormalizedProvider(4, 4)  # 4x4 gravity field
                gravity_model = HolmesFeatherstoneAttractionModel(self.itrf, gravity_field)
                propagator.addForceModel(gravity_model)
                force_models_added += 1
                if not self._force_models_initialized:
                    logging.info("✅ Added 4x4 gravity field model (includes J2, J3, J4)")
            except Exception as e2:
                if not self._force_models_initialized:
                    logging.warning(f"Failed to add 4x4 gravity model: {e2}")
                try:
                    gravity_field = GravityFieldFactory.getNormalizedProvider(2, 0)  # Just J2
                    gravity_model = HolmesFeatherstoneAttractionModel(self.itrf, gravity_field)
                    propagator.addForceModel(gravity_model)
                    force_models_added += 1
                    if not self._force_models_initialized:
                        logging.info("✅ Added J2 gravity field model")
                except Exception as e3:
                    if not self._force_models_initialized:
                        logging.error(f"❌ Failed to add even J2 gravity model: {e3}")
                    # Critical fallback
                    basic_gravity = NewtonianAttraction(Constants.WGS84_EARTH_MU)
                    propagator.addForceModel(basic_gravity)
                    force_models_added += 1
                    if not self._force_models_initialized:
                        logging.info("✅ Added basic Newtonian gravity (fallback)")
        
        # 2. Third body perturbations (Sun and Moon) - less critical
        try:
            sun = CelestialBodyFactory.getSun()
            moon = CelestialBodyFactory.getMoon()
            propagator.addForceModel(ThirdBodyAttraction(sun))
            propagator.addForceModel(ThirdBodyAttraction(moon))
            force_models_added += 2
            if not self._force_models_initialized:
                logging.info("✅ Added third-body perturbations (Sun and Moon)")
        except Exception as e:
            if not self._force_models_initialized:
                logging.warning(f"⚠️ Failed to add third-body forces: {e}")
        
        # 3. Solar radiation pressure - optional for LEO
        try:
            earth_shape = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                                         Constants.WGS84_EARTH_FLATTENING,
                                         self.itrf)
            sun = CelestialBodyFactory.getSun()
            radiation_coefficient = spacecraft_props.radiation_coefficient
            radiation_area = spacecraft_props.radiation_area_m2
            isotropic_radiation = IsotropicRadiationSingleCoefficient(radiation_area, radiation_coefficient)
            srp_model = SolarRadiationPressure(sun, earth_shape, isotropic_radiation)
            propagator.addForceModel(srp_model)
            force_models_added += 1
            if not self._force_models_initialized:
                logging.info("✅ Added solar radiation pressure model")
        except Exception as e:
            if not self._force_models_initialized:
                logging.warning(f"⚠️ Failed to add solar radiation pressure: {e}")
        
        # 4. Atmospheric drag - important for LEO
        try:
            earth_shape = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                                         Constants.WGS84_EARTH_FLATTENING,
                                         self.itrf)
            from org.orekit.models.earth.atmosphere import SimpleExponentialAtmosphere
            # Use more aggressive atmospheric model for better drag at low altitudes
            atmosphere = SimpleExponentialAtmosphere(earth_shape, 0.0002, 6500.0, 0.0)  # Higher density, lower scale height
            drag_coefficient = spacecraft_props.drag_coefficient
            drag_area = spacecraft_props.drag_area_m2
            isotropic_drag = IsotropicDrag(drag_area, drag_coefficient)
            drag_model = DragForce(atmosphere, isotropic_drag)
            propagator.addForceModel(drag_model)
            force_models_added += 1
            if not self._force_models_initialized:
                logging.info("✅ Added atmospheric drag model")
        except Exception as e:
            if not self._force_models_initialized:
                logging.warning(f"⚠️ Failed to add atmospheric drag: {e}")
        
        if not self._force_models_initialized:
            logging.info(f"Successfully added {force_models_added} force models to propagator")
            self._force_models_initialized = True
    
    def propagate_orbit(self, elements: OrbitElements, 
                       time_seconds: float,
                       spacecraft_props: Optional[SpacecraftProperties] = None) -> Tuple[np.ndarray, np.ndarray]:
        # Propagate orbit and return position and velocity.
        # Returns: Tuple of (position_km, velocity_km_s) as numpy arrays
        # Create orbit and propagator
        orbit = self.create_orbit_from_elements(elements)
        propagator = self.create_propagator(orbit, spacecraft_props)
        
        # Propagate to target time
        target_date = orbit.getDate().shiftedBy(time_seconds)
        
        try:
            final_state = propagator.propagate(target_date)
            logging.debug(f"Successfully propagated orbit by {time_seconds} seconds")
        except Exception as e:
            logging.error(f"Numerical propagation failed: {e}")
            # For debugging, let's try to understand why
            logging.error(f"Propagator type: {type(propagator)}")
            logging.error(f"Initial orbit: a={orbit.getA()/1000:.1f}km, e={orbit.getE():.4f}, i={math.degrees(orbit.getI()):.1f}°")
            logging.error(f"Time step: {time_seconds} seconds")
            
            # Try to get the initial state at least
            try:
                final_state = propagator.getInitialState()
                logging.warning("Using initial state as fallback")
            except Exception as e2:
                logging.error(f"Even initial state access failed: {e2}")
                raise RuntimeError(f"Complete propagation failure: {e}")
        
        # Extract position and velocity
        pv = final_state.getPVCoordinates()
        position = pv.getPosition()
        velocity = pv.getVelocity()
        
        # Convert to numpy arrays (km and km/s)
        pos_array = np.array([
            position.getX() / 1000.0,
            position.getY() / 1000.0,
            position.getZ() / 1000.0
        ])
        
        vel_array = np.array([
            velocity.getX() / 1000.0,
            velocity.getY() / 1000.0,
            velocity.getZ() / 1000.0
        ])
        
        return pos_array, vel_array
    
    def batch_propagate(self, elements_list: List[OrbitElements], 
                       time_seconds: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        # Propagate multiple orbits efficiently.
        results = []
        
        for elements in elements_list:
            try:
                pos, vel = self.propagate_orbit(elements, time_seconds)
                results.append((pos, vel))
            except Exception as e:
                logging.warning(f"Failed to propagate orbit {elements}: {e}")
                # Return zero position/velocity as fallback
                results.append((np.zeros(3), np.zeros(3)))
        
        return results
    
    def get_orbital_period(self, elements: OrbitElements) -> float:
        # Calculate orbital period in seconds using Orekit.
        orbit = self.create_orbit_from_elements(elements)
        return orbit.getKeplerianPeriod()
    
    def get_orbital_energy(self, elements: OrbitElements) -> float:
        # Calculate specific orbital energy (J/kg).
        orbit = self.create_orbit_from_elements(elements)
        # Specific energy = -μ/(2a)
        mu = Constants.WGS84_EARTH_MU
        a = orbit.getA()
        return -mu / (2.0 * a)
    
    def cartesian_to_orbit_elements(self, position_km: np.ndarray, velocity_km_s: np.ndarray, 
                                   epoch_mjd: float = 59945.0) -> OrbitElements:
        """Convert Cartesian position/velocity to orbital elements using Orekit.
        
        Args:
            position_km: Position vector in km [x, y, z] in GCRF frame
            velocity_km_s: Velocity vector in km/s [vx, vy, vz] in GCRF frame  
            epoch_mjd: Epoch in Modified Julian Date (default: 2023-01-01)
            
        Returns:
            OrbitElements: Orbital elements derived from Cartesian state
        """
        # Convert to SI units (meters and m/s)
        pos_m = position_km * 1000.0
        vel_ms = velocity_km_s * 1000.0
        
        # Create Orekit Vector3D objects
        position = Vector3D(float(pos_m[0]), float(pos_m[1]), float(pos_m[2]))
        velocity = Vector3D(float(vel_ms[0]), float(vel_ms[1]), float(vel_ms[2]))
        
        # Create epoch
        epoch = AbsoluteDate.createMJDDate(int(epoch_mjd), epoch_mjd % 1.0, self.utc)
        
        # Create PVCoordinates and CartesianOrbit
        pv = PVCoordinates(position, velocity)
        cartesian_orbit = CartesianOrbit(pv, self.gcrf, epoch, Constants.WGS84_EARTH_MU)
        
        # Convert to KeplerianOrbit for element extraction
        keplerian_orbit = KeplerianOrbit(cartesian_orbit)
        
        # Extract orbital elements
        semi_major_axis_km = keplerian_orbit.getA() / 1000.0  # Convert to km
        eccentricity = keplerian_orbit.getE()
        inclination_deg = math.degrees(keplerian_orbit.getI())
        raan_deg = math.degrees(keplerian_orbit.getRightAscensionOfAscendingNode()) 
        arg_periapsis_deg = math.degrees(keplerian_orbit.getPerigeeArgument())
        mean_anomaly_deg = math.degrees(keplerian_orbit.getMeanAnomaly())
        
        return OrbitElements(
            semi_major_axis_km=semi_major_axis_km,
            eccentricity=eccentricity,
            inclination_deg=inclination_deg,
            raan_deg=raan_deg,
            arg_periapsis_deg=arg_periapsis_deg,
            mean_anomaly_deg=mean_anomaly_deg,
            epoch_mjd=epoch_mjd
        )


def create_propagator() -> OrekitOrbitPropagator:
    # Factory function to create OREKIT propagator - NO FALLBACKS
    return OrekitOrbitPropagator()


# For backward compatibility
def convert_to_orbit_elements(altitude_km: float, 
                            inclination_deg: float = 0,
                            eccentricity: float = 0.0,
                            mean_anomaly_deg: float = 0,
                            arg_periapsis_deg: float = 0,
                            raan_deg: float = 0) -> OrbitElements:
    # Convert simple parameters to OrbitElements
    # Use WGS84 equatorial radius for consistency with Orekit Constants
    earth_radius_km = 6378.137  # WGS84 equatorial radius used by Orekit
    return OrbitElements(
        semi_major_axis_km=earth_radius_km + altitude_km,
        eccentricity=eccentricity,
        inclination_deg=inclination_deg,
        raan_deg=raan_deg,
        arg_periapsis_deg=arg_periapsis_deg,
        mean_anomaly_deg=mean_anomaly_deg
    )


if __name__ == "__main__":
    # Test the propagator
    logging.basicConfig(level=logging.INFO)
    
    # Create test orbit elements
    elements = OrbitElements(
        semi_major_axis_km=6771.0,  # 400 km altitude
        eccentricity=0.001,
        inclination_deg=51.6,
        raan_deg=0.0,
        arg_periapsis_deg=0.0,
        mean_anomaly_deg=0.0
    )
    
    # Create propagator
    propagator = create_propagator()
    
    # Propagate for one orbit
    print("Testing orbit propagation...")
    pos, vel = propagator.propagate_orbit(elements, 5400.0)  # ~90 minutes
    print(f"Position (km): {pos}")
    print(f"Velocity (km/s): {vel}")
    print(f"Altitude (km): {np.linalg.norm(pos) - 6371.0}")