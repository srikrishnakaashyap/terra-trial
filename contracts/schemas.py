"""Data contracts shared across sim/, agents/, orchestrator.py, and dashboard/.

Freeze this file first: every other module imports from here, so changing a
field name or range later means touching the whole tree. Field names and
ranges here follow the build plan's contract spec exactly (see doc section 4)
so agent prompts, sim code, and the dashboard all speak the same shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SuspensionType(str, Enum):
    RIGID = "rigid"
    ROCKER_BOGIE = "rocker_bogie"
    INDEPENDENT = "independent"


class Outcome(str, Enum):
    COMPLETED = "completed"
    STALLED = "stalled"
    TIPPED = "tipped"
    TIMEOUT = "timeout"


# --------------------------------------------------------------------------
# Rover config — user/preset input (contracts/schemas.py owns validation;
# this is a system boundary, so out-of-range values raise rather than clamp).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RoverConfig:
    name: str
    wheel_count: int  # 4 or 6
    wheel_diameter_m: float  # (0.1, 0.6]
    ground_clearance_m: float  # (0.05, 0.5]
    suspension_type: SuspensionType
    mass_kg: float  # [10, 500]
    max_motor_torque_nm: float  # [1, 100]

    def __post_init__(self) -> None:
        if self.wheel_count not in (4, 6):
            raise ValueError(f"wheel_count must be 4 or 6, got {self.wheel_count}")
        if not (0.1 <= self.wheel_diameter_m <= 0.6):
            raise ValueError(f"wheel_diameter_m out of range: {self.wheel_diameter_m}")
        if not (0.05 <= self.ground_clearance_m <= 0.5):
            raise ValueError(f"ground_clearance_m out of range: {self.ground_clearance_m}")
        if not (10.0 <= self.mass_kg <= 500.0):
            raise ValueError(f"mass_kg out of range: {self.mass_kg}")
        if not (1.0 <= self.max_motor_torque_nm <= 100.0):
            raise ValueError(f"max_motor_torque_nm out of range: {self.max_motor_torque_nm}")


def validate_rover_config(raw: dict) -> RoverConfig:
    """Build a RoverConfig from a raw dict (preset JSON or a form submission).

    This is a system boundary — invalid input raises ValueError rather than
    being clamped, unlike terrain params below.
    """
    return RoverConfig(
        name=str(raw["name"]),
        wheel_count=int(raw["wheel_count"]),
        wheel_diameter_m=float(raw["wheel_diameter_m"]),
        ground_clearance_m=float(raw["ground_clearance_m"]),
        suspension_type=SuspensionType(raw["suspension_type"]),
        mass_kg=float(raw["mass_kg"]),
        max_motor_torque_nm=float(raw["max_motor_torque_nm"]),
    )


# --------------------------------------------------------------------------
# Rover redesign — Rover Design Agent output (co-evolution mode only, see
# CoevolveOrchestrator). Same clamp-never-reject treatment as terrain: the
# agent is modifying an existing design, so an unset/malformed field falls
# back to the PREVIOUS design's value for that field, not a fixed default —
# clamp_rover_config needs the prior RoverConfig, unlike clamp_terrain_params.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RoverRedesign:
    rover_config: RoverConfig
    targets_weakness: str
    reasoning: str


def clamp_rover_config(raw: dict, base: RoverConfig) -> RoverConfig:
    """Clamp a raw (possibly malformed) Rover Design Agent output into valid
    ranges, falling back to `base`'s value per-field rather than a fixed
    default — a redesign is a diff against the previous rover, not a fresh
    proposal from scratch."""
    wheel_count = base.wheel_count
    try:
        candidate = int(raw.get("wheel_count", base.wheel_count))
        if candidate in (4, 6):
            wheel_count = candidate
    except (TypeError, ValueError):
        pass

    try:
        suspension_type = SuspensionType(raw.get("suspension_type", base.suspension_type.value))
    except ValueError:
        suspension_type = base.suspension_type

    return RoverConfig(
        name=str(raw.get("name", base.name)),
        wheel_count=wheel_count,
        wheel_diameter_m=_clamp_float(raw.get("wheel_diameter_m"), 0.1, 0.6, base.wheel_diameter_m),
        ground_clearance_m=_clamp_float(raw.get("ground_clearance_m"), 0.05, 0.5, base.ground_clearance_m),
        suspension_type=suspension_type,
        mass_kg=_clamp_float(raw.get("mass_kg"), 10.0, 500.0, base.mass_kg),
        max_motor_torque_nm=_clamp_float(raw.get("max_motor_torque_nm"), 1.0, 100.0, base.max_motor_torque_nm),
    )


# --------------------------------------------------------------------------
# Planet profile — loaded from contracts/planet_profiles.json.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TerrainBias:
    """Suggested ranges the Terrain Agent should stay near for this planet."""

    slope_deg: tuple[float, float]
    rock_density: tuple[float, float]
    roughness: tuple[float, float]


@dataclass(frozen=True)
class Palette:
    ground: tuple[float, float, float]
    sky: tuple[float, float, float]


@dataclass(frozen=True)
class PlanetProfile:
    name: str
    gravity_ms2: float
    surface_friction: float
    terrain_bias: TerrainBias
    palette: Palette


def load_planet_profiles(path: Path) -> dict[str, PlanetProfile]:
    """Load contracts/planet_profiles.json into name -> PlanetProfile."""
    raw = json.loads(path.read_text())
    profiles: dict[str, PlanetProfile] = {}
    for name, data in raw.items():
        bias = data["terrain_bias"]
        pal = data["palette"]
        profiles[name] = PlanetProfile(
            name=name,
            gravity_ms2=float(data["gravity_ms2"]),
            surface_friction=float(data["surface_friction"]),
            terrain_bias=TerrainBias(
                slope_deg=(bias["slope_deg"][0], bias["slope_deg"][1]),
                rock_density=(bias["rock_density"][0], bias["rock_density"][1]),
                roughness=(bias["roughness"][0], bias["roughness"][1]),
            ),
            palette=Palette(
                ground=(pal["ground"][0], pal["ground"][1], pal["ground"][2]),
                sky=(pal["sky"][0], pal["sky"][1], pal["sky"][2]),
            ),
        )
    return profiles


# --------------------------------------------------------------------------
# Terrain params — Terrain Agent output. Clamp, never reject: a clamped
# agent output keeps the demo running and is honest in the logs; a rejected
# one stalls it. TerrainParams itself carries no validation — the only path
# that produces one is clamp_terrain_params() below, which is the single
# source of truth for valid ranges.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TerrainParams:
    slope_deg: float  # [0, 40]
    rock_density: float  # [0, 1]
    rock_size_range_m: tuple[float, float]  # each in [0.01, 0.8]
    roughness: float  # [0, 1]
    targets_weakness: str
    reasoning: str


def _clamp_float(value: object, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def clamp_terrain_params(raw: dict) -> TerrainParams:
    """Clamp a raw (possibly malformed) Terrain Agent output into valid ranges."""
    slope_deg = _clamp_float(raw.get("slope_deg"), 0.0, 40.0, 10.0)
    rock_density = _clamp_float(raw.get("rock_density"), 0.0, 1.0, 0.2)
    roughness = _clamp_float(raw.get("roughness"), 0.0, 1.0, 0.2)

    raw_range = raw.get("rock_size_range_m")
    try:
        lo_r = _clamp_float(raw_range[0], 0.01, 0.8, 0.05)
        hi_r = _clamp_float(raw_range[1], 0.01, 0.8, 0.3)
    except (TypeError, IndexError, KeyError):
        lo_r, hi_r = 0.05, 0.3
    if lo_r > hi_r:
        lo_r, hi_r = hi_r, lo_r

    return TerrainParams(
        slope_deg=slope_deg,
        rock_density=rock_density,
        rock_size_range_m=(lo_r, hi_r),
        roughness=roughness,
        targets_weakness=str(raw.get("targets_weakness", "unknown")),
        reasoning=str(raw.get("reasoning", "")),
    )


# --------------------------------------------------------------------------
# Trial metrics — sim output.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SimMetrics:
    completion_pct: float
    distance_m: float
    max_tilt_deg: float
    max_wheel_slip_pct: float
    time_to_stall_s: float | None  # None if the rover never stalled
    outcome: Outcome
    wall_clock_s: float


# --------------------------------------------------------------------------
# Critic output.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CriticOutput:
    diagnosis: str
    primary_weakness: str
    confidence: str  # "low" | "medium" | "high"
    evidence: list[str]
    next_target: str


# --------------------------------------------------------------------------
# Run history / state, persisted to runs/<run_id>/run_state.json.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    terrain_params: TerrainParams
    metrics: SimMetrics
    critique: CriticOutput
    # Co-evolution mode only (CoevolveOrchestrator) — None in escalation mode.
    # rover_config is the design USED for this iteration's trial; redesign is
    # what the Rover Design Agent proposed in response, which becomes next
    # iteration's rover_config.
    rover_config: RoverConfig | None = None
    redesign: RoverRedesign | None = None


@dataclass
class RunState:
    run_id: str
    planet: PlanetProfile
    rover_config: RoverConfig
    status: str = "pending"  # pending | running | complete | failed
    iterations: list[IterationResult] = field(default_factory=list)
    final_report: str | None = None
