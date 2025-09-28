"""Orekit physics module (migrated to package structure).

Original path: OrbitalAvoidance/orekit_physics.py
This version lives under orbital_avoidance.physics. Adjusted path discovery
to account for new src/ layout. API preserved.
"""

from __future__ import annotations

import math
import os
import sys
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import concurrent.futures
import multiprocessing
from pathlib import Path

import numpy as np

# Determine project root (assumes this file is at src/orbital_avoidance/physics)
_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent  # orbital_avoidance
_SRC_ROOT = _PKG_ROOT.parent  # src
_PROJECT_ROOT = _SRC_ROOT.parent  # repository root

# Add project root to sys.path for hardware_detection if present
if str(_PROJECT_ROOT) not in sys.path:
	sys.path.append(str(_PROJECT_ROOT))

try:
	from hardware_detection import get_optimal_config
	HARDWARE_DETECTION_AVAILABLE = True
except ImportError:
	HARDWARE_DETECTION_AVAILABLE = False

# ---- Orekit JVM Initialization (unchanged logic, path adjusted) ----
try:
	import orekit_jpype as oj
	import jpype

	orekit_jar_dir = _PROJECT_ROOT / "orekit_lib"
	if not orekit_jar_dir.is_dir():
		# Fallback: look one level up (in case layout differs)
		alt = _PROJECT_ROOT.parent / "orekit_lib"
		if alt.is_dir():
			orekit_jar_dir = alt
	jar_files = [str(p) for p in orekit_jar_dir.glob("*.jar")]
	if not jpype.isJVMStarted():
		jpype.startJVM(classpath=jar_files)
	oj.initVM()

	from org.orekit.data import DataContext, DirectoryCrawler
	from java.io import File

	candidate_dirs = [
		_PROJECT_ROOT / "orekit-data",
		_PROJECT_ROOT.parent / "orekit-data",
		Path(os.environ.get("OREKIT_DATA_PATH", "")),
	]
	candidate_dirs = [d for d in candidate_dirs if d and d.is_dir()]
	data_registered = False
	for d in candidate_dirs:
		try:
			if any(d.iterdir()):
				crawler = DirectoryCrawler(File(str(d)))
				mgr = DataContext.getDefault().getDataProvidersManager()
				mgr.clearProviders()
				mgr.addProvider(crawler)
				print(f"✓ Orekit data directory configured: {d}")
				data_registered = True
				break
		except Exception as e:  # pragma: no cover
			logging.warning(f"Failed to register Orekit data at {d}: {e}")
	if not data_registered:
		print("⚠️ No valid Orekit data directory found with files. Proceeding with limited defaults.")

	from org.orekit.orbits import PositionAngleType
	print(f"✓ Orekit PositionAngleType imported successfully: {PositionAngleType.TRUE}")

	# Core imports
	from org.orekit.time import AbsoluteDate, TimeScalesFactory
	from org.orekit.frames import FramesFactory
	from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
	from org.orekit.utils import Constants, IERSConventions
	from org.orekit.orbits import KeplerianOrbit, CartesianOrbit
	from org.orekit.utils import PVCoordinates
	from org.orekit.propagation import SpacecraftState
	from org.orekit.propagation.numerical import NumericalPropagator
	from org.hipparchus.geometry.euclidean.threed import Vector3D
	from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
	from org.orekit.forces.gravity.potential import GravityFieldFactory
	from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, NewtonianAttraction, ThirdBodyAttraction
	from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient, SolarRadiationPressure
	from org.orekit.forces.drag import IsotropicDrag, DragForce

	OREKIT_AVAILABLE = True
except Exception as e:  # pragma: no cover
	OREKIT_AVAILABLE = False
	logging.error(f"Failed to initialize Orekit: {e}")
	raise


@dataclass
class OrbitElements:
	semi_major_axis_km: float
	eccentricity: float
	inclination_deg: float
	raan_deg: float
	arg_periapsis_deg: float
	mean_anomaly_deg: float
	epoch_mjd: float = 59945.0


@dataclass
class SpacecraftProperties:
	mass_kg: float = 1000.0
	drag_area_m2: float = 10.0
	drag_coefficient: float = 2.2
	radiation_area_m2: float = 5.0
	radiation_coefficient: float = 1.3


class OrekitOrbitPropagator:
	def __init__(self):
		if not OREKIT_AVAILABLE:
			raise RuntimeError("Orekit not available")
		self._setup_minimal_data()
		self.itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
		self.gcrf = FramesFactory.getGCRF()
		self.utc = TimeScalesFactory.getUTC()
		self.earth = CelestialBodyFactory.getEarth()
		self.spacecraft_props = SpacecraftProperties()
		self._cached_propagators = {}
		self._force_models_initialized = False
		self._persistent = {}
		self.hardware_config = None
		if HARDWARE_DETECTION_AVAILABLE:
			try:
				self.hardware_config = get_optimal_config()
			except Exception as e:  # pragma: no cover
				logging.warning(f"Hardware detection failed: {e}")

	def _setup_minimal_data(self):  # pragma: no cover (setup)
		try:
			from org.orekit.data import DataContext, DirectoryCrawler
			from java.io import File
			candidates = [
				_PROJECT_ROOT / "orekit-data",
				_PROJECT_ROOT.parent / "orekit-data",
				Path(os.environ.get("OREKIT_DATA_PATH", "")),
			]
			candidates = [c for c in candidates if c and c.is_dir()]
			mgr = DataContext.getDefault().getDataProvidersManager()
			for c in candidates:
				try:
					if any(c.iterdir()):
						mgr.clearProviders()
						mgr.addProvider(DirectoryCrawler(File(str(c))))
						return
				except Exception:
					continue
		except Exception as e:
			logging.warning(f"Could not re-register Orekit data: {e}")

	def create_orbit_from_elements(self, elements: OrbitElements) -> KeplerianOrbit:
		a = elements.semi_major_axis_km * 1000.0
		e = elements.eccentricity
		inc = math.radians(elements.inclination_deg)
		raan = math.radians(elements.raan_deg)
		argp = math.radians(elements.arg_periapsis_deg)
		ma = math.radians(elements.mean_anomaly_deg)
		epoch = self._mjd_to_date(elements.epoch_mjd)
		return KeplerianOrbit(a, e, inc, argp, raan, ma, PositionAngleType.MEAN, self.gcrf, epoch, Constants.WGS84_EARTH_MU)

	def _mjd_to_date(self, mjd: float):
		return AbsoluteDate.createMJDDate(int(mjd), mjd % 1.0, self.utc)

	def create_propagator(self, orbit: KeplerianOrbit, spacecraft_props: Optional[SpacecraftProperties] = None):
		if spacecraft_props is None:
			spacecraft_props = self.spacecraft_props
		try:
			integrator = DormandPrince853Integrator(1.0, 120.0, 1e-11, 1e-11)
		except Exception as e:  # pragma: no cover
			raise RuntimeError(f"Failed integrator: {e}")
		propagator = NumericalPropagator(integrator)
		propagator.setInitialState(SpacecraftState(orbit, spacecraft_props.mass_kg))
		self._add_force_models_safely(propagator, spacecraft_props)
		return propagator

	def _add_force_models_safely(self, propagator, spacecraft_props):
		added = 0
		try:
			gravity_field = GravityFieldFactory.getNormalizedProvider(8, 8)
			propagator.addForceModel(HolmesFeatherstoneAttractionModel(self.itrf, gravity_field))
			added += 1
		except Exception:
			propagator.addForceModel(NewtonianAttraction(Constants.WGS84_EARTH_MU))
			added += 1
		try:
			sun = CelestialBodyFactory.getSun(); moon = CelestialBodyFactory.getMoon()
			propagator.addForceModel(ThirdBodyAttraction(sun)); propagator.addForceModel(ThirdBodyAttraction(moon))
			added += 2
		except Exception:
			pass
		try:
			earth_shape = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, self.itrf)
			sun = CelestialBodyFactory.getSun()
			iso = IsotropicRadiationSingleCoefficient(spacecraft_props.radiation_area_m2, spacecraft_props.radiation_coefficient)
			propagator.addForceModel(SolarRadiationPressure(sun, earth_shape, iso))
			added += 1
		except Exception:
			pass
		try:
			earth_shape = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, self.itrf)
			from org.orekit.models.earth.atmosphere import SimpleExponentialAtmosphere
			atmosphere = SimpleExponentialAtmosphere(earth_shape, 0.0002, 6500.0, 0.0)
			drag = IsotropicDrag(spacecraft_props.drag_area_m2, spacecraft_props.drag_coefficient)
			propagator.addForceModel(DragForce(atmosphere, drag))
			added += 1
		except Exception:
			pass
		if not self._force_models_initialized:
			logging.info(f"Force models added: {added}")
			self._force_models_initialized = True

	def propagate_orbit(self, elements: OrbitElements, time_seconds: float, spacecraft_props: Optional[SpacecraftProperties] = None):
		key = (
			round(elements.semi_major_axis_km, 6), round(elements.eccentricity, 8),
			round(elements.inclination_deg, 6), round(elements.raan_deg, 6),
			round(elements.arg_periapsis_deg, 6), round(elements.mean_anomaly_deg, 6)
		)
		entry = self._cached_propagators.get(key)
		if entry is None:
			orbit = self.create_orbit_from_elements(elements)
			propagator = self.create_propagator(orbit, spacecraft_props)
			entry = {'propagator': propagator, 'last_date': orbit.getDate()}
			self._cached_propagators[key] = entry
		target_date = entry['last_date'].shiftedBy(time_seconds)
		final_state = entry['propagator'].propagate(target_date)
		entry['last_date'] = target_date
		pv = final_state.getPVCoordinates()
		pos = pv.getPosition(); vel = pv.getVelocity()
		pos_arr = np.array([pos.getX(), pos.getY(), pos.getZ()]) / 1000.0
		vel_arr = np.array([vel.getX(), vel.getY(), vel.getZ()]) / 1000.0
		return pos_arr, vel_arr

	# Persistent API
	def create_persistent_object(self, key: str, elements: OrbitElements, spacecraft_props: Optional[SpacecraftProperties] = None):
		if key in self._persistent:
			return
		orbit = self.create_orbit_from_elements(elements)
		propagator = self.create_propagator(orbit, spacecraft_props)
		state = propagator.getInitialState()
		self._persistent[key] = {'propagator': propagator, 'state': state}

	def propagate_persistent_delta(self, key: str, delta_t: float):
		if key not in self._persistent:
			raise KeyError(f"No persistent object {key}")
		entry = self._persistent[key]
		target = entry['state'].getDate().shiftedBy(delta_t)
		new_state = entry['propagator'].propagate(target)
		entry['state'] = new_state
		pv = new_state.getPVCoordinates()
		p = pv.getPosition(); v = pv.getVelocity()
		return np.array([p.getX(), p.getY(), p.getZ()])/1000.0, np.array([v.getX(), v.getY(), v.getZ()])/1000.0

	def apply_instantaneous_delta_v(self, key: str, delta_v_km_s: np.ndarray):
		if key not in self._persistent:
			raise KeyError(key)
		entry = self._persistent[key]
		pv = entry['state'].getPVCoordinates()
		
		# Create new velocity vector
		new_vel = Vector3D(
			pv.getVelocity().getX() + delta_v_km_s[0]*1000.0,
			pv.getVelocity().getY() + delta_v_km_s[1]*1000.0,
			pv.getVelocity().getZ() + delta_v_km_s[2]*1000.0,
		)
		
		# Create new PVCoordinates
		new_pv = PVCoordinates(pv.getPosition(), new_vel)
		
		# Create new SpacecraftState using the simpler constructor
		# Option 1: Create from orbit (recommended)
		current_date = entry['state'].getDate()
		current_frame = entry['state'].getFrame()
		
		# Convert PV to CartesianOrbit then to KeplerianOrbit
		cart_orbit = CartesianOrbit(new_pv, current_frame, current_date, Constants.WGS84_EARTH_MU)
		kep_orbit = KeplerianOrbit(cart_orbit)
		
		# Create new state with mass
		entry['state'] = SpacecraftState(kep_orbit, entry['state'].getMass())

	def cartesian_to_orbit_elements(self, position_km: np.ndarray, velocity_km_s: np.ndarray, epoch_mjd: float = 59945.0) -> OrbitElements:
		pos_m = position_km * 1000.0; vel_ms = velocity_km_s * 1000.0
		position_vec = Vector3D(float(pos_m[0]), float(pos_m[1]), float(pos_m[2]))
		velocity_vec = Vector3D(float(vel_ms[0]), float(vel_ms[1]), float(vel_ms[2]))
		pv = PVCoordinates(position_vec, velocity_vec)
		epoch = self._mjd_to_date(epoch_mjd)
		cart = CartesianOrbit(pv, self.gcrf, epoch, Constants.WGS84_EARTH_MU)
		kep = KeplerianOrbit(cart)
		elements = OrbitElements(
			semi_major_axis_km=kep.getA()/1000.0,
			eccentricity=kep.getE(),
			inclination_deg=math.degrees(kep.getI()),
			raan_deg=math.degrees(kep.getRightAscensionOfAscendingNode()),
			arg_periapsis_deg=math.degrees(kep.getPerigeeArgument()),
			mean_anomaly_deg=math.degrees(kep.getMeanAnomaly()),
			epoch_mjd=epoch_mjd
		)
		self._cached_propagators.clear()
		return elements

	def get_orbital_period(self, elements: OrbitElements) -> float:
		orbit = self.create_orbit_from_elements(elements)
		return orbit.getKeplerianPeriod()

	def get_orbital_energy(self, elements: OrbitElements) -> float:
		orbit = self.create_orbit_from_elements(elements)
		mu = Constants.WGS84_EARTH_MU
		return -mu / (2.0 * orbit.getA())


def create_propagator() -> OrekitOrbitPropagator:
	return OrekitOrbitPropagator()


def convert_to_orbit_elements(altitude_km: float, inclination_deg: float = 0, eccentricity: float = 0.0,
							  mean_anomaly_deg: float = 0, arg_periapsis_deg: float = 0, raan_deg: float = 0) -> OrbitElements:
	earth_radius_km = 6378.137
	return OrbitElements(
		semi_major_axis_km=earth_radius_km + altitude_km,
		eccentricity=eccentricity,
		inclination_deg=inclination_deg,
		raan_deg=raan_deg,
		arg_periapsis_deg=arg_periapsis_deg,
		mean_anomaly_deg=mean_anomaly_deg
	)


__all__ = [
	'OrekitOrbitPropagator', 'OrbitElements', 'SpacecraftProperties', 'create_propagator', 'convert_to_orbit_elements'
]
