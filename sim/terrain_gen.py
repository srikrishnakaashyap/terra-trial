"""Turns TerrainParams into a heightfield + scattered rock colliders.

Terrain spans TERRAIN_SIZE_M x TERRAIN_SIZE_M, centered at the origin (this
matches PyBullet's default heightfield centering). The rover starts near the
low-x edge and drives toward +x; completion_pct is measured against
TERRAIN_SIZE_M.

The RNG is seeded from a hash of the params (not a random draw), so the same
TerrainParams always produce the same terrain — required for a repeatable
demo and an apples-to-apples baseline comparison.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from contracts.schemas import TerrainParams

TERRAIN_SIZE_M = 30.0
RESOLUTION = 128
_BASE_AMPLITUDE_M = 0.6  # broad rolling undulation at roughness=1.0
_GRAIN_AMPLITUDE_M = 0.15  # finer, higher-frequency surface texture
_MAX_ROCKS = 220
_SPAWN_CLEAR_M = 3.0  # keep rocks off the rover's spawn strip
_CLUSTER_FRACTION = 0.15  # fraction of rocks that become cluster centers
_CLUSTER_SPREAD_M = 2.0  # stddev of scatter around a cluster center


@dataclass(frozen=True)
class Rock:
    position: tuple[float, float, float]
    radius: float  # bounding radius — used for physics/placement math regardless of shape
    shape: str  # "sphere" | "box"
    half_extents: tuple[float, float, float]  # used when shape == "box"; == (radius,)*3 for sphere
    yaw_deg: float  # box rotation about z; ignored for sphere


def terrain_seed(params: TerrainParams) -> int:
    key = (
        f"{params.slope_deg:.4f}|{params.rock_density:.4f}|{params.roughness:.4f}|"
        f"{params.rock_size_range_m[0]:.4f}|{params.rock_size_range_m[1]:.4f}"
    )
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big")


def generate_terrain(params: TerrainParams) -> tuple[np.ndarray, list[Rock]]:
    """Return (heights[n,n] float32 meters, [Rock, ...])."""
    rng = np.random.default_rng(terrain_seed(params))
    n = RESOLUTION

    # Two noise scales, not one: a broad rolling base plus a finer grain
    # layer on top, so terrain reads as textured ground rather than a
    # single smooth bump field — real regolith looks like both at once.
    base = _fractal_noise(rng, n, octaves=5) * params.roughness * _BASE_AMPLITUDE_M
    grain = _fractal_noise(rng, n, octaves=7) * params.roughness * _GRAIN_AMPLITUDE_M
    heights = base + grain
    heights += _linear_slope(n, TERRAIN_SIZE_M, params.slope_deg)

    rocks = _scatter_rocks(rng, params, heights)
    return heights.astype(np.float32), rocks


def _fractal_noise(rng: np.random.Generator, n: int, octaves: int = 4) -> np.ndarray:
    field = np.zeros((n, n), dtype=np.float64)
    amplitude = 1.0
    total_amplitude = 0.0
    for octave in range(octaves):
        scale = 2**octave
        coarse = rng.normal(size=(max(n // scale, 2), max(n // scale, 2)))
        field += amplitude * _resize_bilinear(coarse, n)
        total_amplitude += amplitude
        amplitude *= 0.5
    field /= total_amplitude
    return (field - field.min()) / (field.max() - field.min() + 1e-9) - 0.5


def _resize_bilinear(arr: np.ndarray, n: int) -> np.ndarray:
    x_old = np.linspace(0, 1, arr.shape[0])
    y_old = np.linspace(0, 1, arr.shape[1])
    x_new = np.linspace(0, 1, n)
    y_new = np.linspace(0, 1, n)
    tmp = np.array([np.interp(x_new, x_old, arr[:, j]) for j in range(arr.shape[1])]).T
    return np.array([np.interp(y_new, y_old, tmp[i, :]) for i in range(tmp.shape[0])])


def _linear_slope(n: int, size_m: float, slope_deg: float) -> np.ndarray:
    """Ramp that rises along x (columns) — the rover's drive direction —
    and is constant along y (rows), so the rover climbs into the slope
    instead of spawning parked on a sideways tilt."""
    rise_per_m = np.tan(np.radians(slope_deg))
    axis = np.linspace(0, size_m, n)
    return np.tile(axis * rise_per_m, (n, 1))


def _scatter_rocks(rng: np.random.Generator, params: TerrainParams, heights: np.ndarray) -> list[Rock]:
    """Cluster rocks around a handful of outcrop centers rather than
    scattering them uniformly — real rock fields bunch up near outcrops,
    not Poisson-uniform. Sizes skew toward small with occasional boulders
    (Beta distribution), and shapes mix spheres with randomly-yawed boxes
    so the field doesn't read as a tray of ball bearings."""
    n = heights.shape[0]
    cell = TERRAIN_SIZE_M / n
    count = int(params.rock_density * _MAX_ROCKS)
    if count == 0:
        return []
    min_col = _SPAWN_CLEAR_M / cell

    n_clusters = max(1, int(count * _CLUSTER_FRACTION))
    cluster_centers = [(rng.uniform(min_col, n - 1), rng.uniform(0, n - 1)) for _ in range(n_clusters)]
    spread_cells = _CLUSTER_SPREAD_M / cell

    lo_r, hi_r = params.rock_size_range_m
    rocks: list[Rock] = []
    for _ in range(count):
        cx, cy = cluster_centers[rng.integers(0, n_clusters)]
        col = int(np.clip(rng.normal(cx, spread_cells), min_col, n - 1))
        row = int(np.clip(rng.normal(cy, spread_cells), 0, n - 1))

        radius = float(lo_r + rng.beta(1.5, 4.0) * (hi_r - lo_r))
        x = col * cell - TERRAIN_SIZE_M / 2
        y = row * cell - TERRAIN_SIZE_M / 2

        if rng.random() < 0.5:
            shape = "sphere"
            half_extents = (radius, radius, radius)
        else:
            shape = "box"
            aspect = rng.uniform(0.7, 1.3, size=3)
            half_extents = tuple(float(radius * a) for a in aspect)

        z = float(heights[row, col]) + half_extents[2] * 0.6
        rocks.append(
            Rock(
                position=(x, y, z),
                radius=radius,
                shape=shape,
                half_extents=half_extents,
                yaw_deg=float(rng.uniform(0, 360)),
            )
        )
    return rocks
