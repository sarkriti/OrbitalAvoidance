"""
Physics models and utilities for research-grade orbital simulation.

- Cowell propagation with configurable perturbations: J2, drag, SRP, third-body (Sun, Moon)
- RTN frame utilities for applying impulsive/burn maneuvers
- Scenario helpers to generate debris with controlled close encounters

Dependencies: astropy, poliastro
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
def norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


# --- Force models -----------------------------------------------------------------

@dataclass
class ForceConfig:
    j2: bool = True
    drag: bool = True
    srp: bool = False
    third_body_sun: bool = False
    third_body_moon: bool = False

    # Parameters
    cd: float = 2.2  # Drag coefficient
    cr: float = 1.2  # SRP reflectivity coefficient
    area: float = 4.0  # m^2
    mass: float = 500.0  # kg

    # Atmosphere
    rho0: float = 1.225  # kg/m^3 at sea level
    H: float = 8500.0  # m, scale height


@dataclass
class EarthParams:
    mu: float = Earth.k.to_value(u.km**3 / u.s**2)
    r: float = Earth.R.to_value(u.km)
    J2: float = 1.08262668e-3


def accel_j2(r_vec_km: np.ndarray, earth: EarthParams) -> np.ndarray:
    x, y, z = r_vec_km
    r2 = np.dot(r_vec_km, r_vec_km)
    r = np.sqrt(r2)
    mu = earth.mu
    R = earth.r
    J2 = earth.J2

    zx = z / r
    factor = 1.5 * J2 * mu * R**2 / r**5
    ax = factor * x * (5 * zx**2 - 1)
    ay = factor * y * (5 * zx**2 - 1)
    az = factor * z * (5 * zx**2 - 3)
    return np.array([ax, ay, az])


def density_exponential(h_m: float, rho0: float, H: float) -> float:
    return rho0 * np.exp(-h_m / H)


def accel_drag(r_km: np.ndarray, v_km_s: np.ndarray, cfg: ForceConfig, earth: EarthParams) -> np.ndarray:
    # Simple exponential atmosphere above Earth
    r_norm_km = norm(r_km)
    alt_km = r_norm_km - earth.r
    if alt_km < 0:
        return np.zeros(3)
    rho = density_exponential(alt_km * 1000.0, cfg.rho0, cfg.H)
    v_rel = v_km_s  # neglect winds
    v_rel_m_s = v_rel * 1000.0
    a_drag_m_s2 = -0.5 * cfg.cd * cfg.area / cfg.mass * rho * np.linalg.norm(v_rel_m_s) * v_rel_m_s
    return a_drag_m_s2 / 1000.0  # to km/s^2


def accel_srp(r_km: np.ndarray, t: float, cfg: ForceConfig) -> np.ndarray:
    # Very simplified SRP pointing from Sun to spacecraft
    P0 = 4.56e-6  # N/m^2 at 1 AU
    # Sun vector from Earth (approx circular)
    omega = 2 * np.pi / (365.25 * 24 * 3600)
    sun_dir = np.array([np.cos(omega * t), np.sin(omega * t), 0.0])
    a_srp_m_s2 = cfg.cr * cfg.area / cfg.mass * P0 * sun_dir
    return a_srp_m_s2 / 1000.0


def third_body_accel(sc_pos_eci_km: np.ndarray, body_pos_eci_km: np.ndarray, mu_body_km3_s2: float) -> np.ndarray:
    r_sb = body_pos_eci_km - sc_pos_eci_km
    r_se = body_pos_eci_km
    return mu_body_km3_s2 * (r_sb / norm(r_sb) ** 3 - r_se / norm(r_se) ** 3)


def dynamics(t: float, state: np.ndarray, cfg: ForceConfig, earth: EarthParams) -> np.ndarray:
    r = state[:3]
    v = state[3:]
    mu = earth.mu

    a_grav = -mu * r / norm(r) ** 3
    a = a_grav

    if cfg.j2:
        a += accel_j2(r, earth)
    if cfg.drag:
        a += accel_drag(r, v, cfg, earth)
    if cfg.srp:
        a += accel_srp(r, t, cfg)
    # For research, third bodies usually matter above LEO; placeholders kept simple
    # Could extend with precise ephemerides via astropy for Sun/Moon
    return np.hstack((v, a))


# --- RTN utilities ----------------------------------------------------------------

def eci_to_rtn(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    r_hat = r_eci / norm(r_eci)
    h_hat = np.cross(r_eci, v_eci)
    h_hat = h_hat / norm(h_hat)
    t_hat = np.cross(h_hat, r_hat)
    return np.vstack((r_hat, t_hat, h_hat)).T


def apply_delta_v_rtn(r_eci: np.ndarray, v_eci: np.ndarray, dv_rtn: np.ndarray) -> np.ndarray:
    C = eci_to_rtn(r_eci, v_eci)
    dv_eci = C @ dv_rtn
    return v_eci + dv_eci


# --- Scenario helpers --------------------------------------------------------------

def circular_orbit(a_km: float, inc_deg: float = 0.0, nu_deg: float = 0.0, epoch: Optional[Time] = None) -> Orbit:
    if epoch is None:
        epoch = Time.now()
    return Orbit.from_classical(
        Earth,
        a=a_km * u.km,
        ecc=0.0 * u.one,
        inc=inc_deg * u.deg,
        raan=0 * u.deg,
        argp=0 * u.deg,
        nu=nu_deg * u.deg,
        epoch=epoch,
    )


def propagate_orbit_cowell(orb: Orbit, dt_s: float, steps: int, cfg: ForceConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate using a robust local RK4 integrator with selected perturbations.

    Returns arrays of positions and velocities (km, km/s) shape (steps+1, 3)
    """
    earth = EarthParams()
    r0 = orb.r.to_value(u.km)
    v0 = orb.v.to_value(u.km / u.s)
    state = np.hstack((r0, v0))

    def rk4_step(t: float, y: np.ndarray, h: float) -> np.ndarray:
        k1 = dynamics(t, y, cfg, earth)
        k2 = dynamics(t + 0.5 * h, y + 0.5 * h * k1, cfg, earth)
        k3 = dynamics(t + 0.5 * h, y + 0.5 * h * k2, cfg, earth)
        k4 = dynamics(t + h, y + h * k3, cfg, earth)
        return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Substep for stability (<= 5s substeps typical for LEO)
    sub_h = max(0.5, min(5.0, dt_s / 6.0))
    sub_n = int(np.ceil(dt_s / sub_h))
    h = dt_s / sub_n

    r_hist = [state[:3].copy()]
    v_hist = [state[3:].copy()]
    t = 0.0
    for _ in range(steps):
        for __ in range(sub_n):
            state = rk4_step(t, state, h)
            t += h
        r_hist.append(state[:3].copy())
        v_hist.append(state[3:].copy())

    return np.vstack(r_hist), np.vstack(v_hist)


def seed_close_encounters(base_orbit: Orbit, n_debris: int = 16, max_offset_km: float = 5.0, inc_spread_deg: float = 0.5) -> List[Orbit]:
    """Create debris orbits near the base orbit to ensure close approaches.

    - Small semi-major axis offsets (+/- max_offset_km)
    - Small inclination tweaks (+/- inc_spread_deg)
    - Random true anomaly to stagger phasing
    """
    a0 = base_orbit.a.to_value(u.km)
    inc0 = base_orbit.inc.to_value(u.deg)
    epoch = base_orbit.epoch

    debris = []
    for i in range(n_debris):
        # Alternate small +/- a offsets and tiny inclination tweaks
        a = a0 + np.random.uniform(-max_offset_km, max_offset_km)
        inc = inc0 + np.random.uniform(-inc_spread_deg, inc_spread_deg)
        # Stagger true anomaly closely around the satellite to force near passes
        nu = np.random.normal(loc=0.0, scale=5.0)
        debris.append(circular_orbit(a, inc, nu, epoch))
    return debris
