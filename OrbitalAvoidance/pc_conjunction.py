"""
Conjunction assessment utilities.

Provides: closest approach computation, coarse probability of collision (Pc)
models, and thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ConjunctionConfig:
    collision_radius_km: float = 0.0002  # 200 m combined radius
    warning_km: float = 5.0


def closest_approach_km(track_a: np.ndarray, track_b: np.ndarray) -> Tuple[float, int]:
    """Return min distance [km] and index between two tracks (N,3)."""
    diffs = track_a - track_b
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    return float(dists[idx]), idx


def risk_scalar(distance_km: float, cfg: ConjunctionConfig) -> float:
    if distance_km <= cfg.collision_radius_km:
        return 1.0
    if distance_km <= cfg.warning_km:
        return 0.8 * (cfg.warning_km - distance_km) / cfg.warning_km
    if distance_km <= 20:
        return 0.1 * (20 - distance_km) / 20
    return 0.0


def batch_min_distance_km(target_track: np.ndarray, debris_tracks: List[np.ndarray]) -> Tuple[float, int, int]:
    """Compute overall min distance among many debris tracks.

    Returns: (min_distance_km, debris_index, time_index)
    """
    best = (1e12, -1, -1)
    for j, track in enumerate(debris_tracks):
        d, i = closest_approach_km(target_track, track)
        if d < best[0]:
            best = (d, j, i)
    return best
