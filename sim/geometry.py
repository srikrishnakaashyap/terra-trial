"""Engine-agnostic rover geometry derived from RoverConfig.

Wheel layout, chassis sizing, and mass split are physics-engine-independent
— every SimInterface backend (PyBullet, MuJoCo, and Isaac Sim tomorrow)
derives its rover from these same numbers, so swapping backends can never
silently change what rover gets simulated.
"""

from __future__ import annotations

from contracts.schemas import RoverConfig

CHASSIS_HALF_HEIGHT_M = 0.08
_TRACK_WIDTH_FACTOR = 3.0  # track width = wheel_diameter_m * this
_AXLE_SPACING_FACTOR = 2.5  # spacing between axles = wheel_diameter_m * this
_WHEEL_MASS_FRACTION = 0.05  # fraction of total mass carried by each wheel

# Trial drive policy (build plan section 6, Step 3): "drive forward at
# constant target velocity", not open-loop max torque. A velocity target
# with a torque cap is what actually produces the traction-limited failure
# modes the metrics are meant to capture (wheel slip, stall) — open-loop
# full torque just slams the reaction torque into the chassis on every
# trial and reliably wheelies it regardless of terrain.
TARGET_SPEED_MPS = 1.0


def target_wheel_angular_velocity(config: RoverConfig) -> float:
    return TARGET_SPEED_MPS / (config.wheel_diameter_m / 2)


def wheel_layout(config: RoverConfig) -> list[tuple[float, float]]:
    """[(x, y), ...] wheel-center offsets from the chassis center, in meters."""
    axles = config.wheel_count // 2
    axle_spacing = config.wheel_diameter_m * _AXLE_SPACING_FACTOR
    wheelbase = axle_spacing * (axles - 1)
    half_track = config.wheel_diameter_m * _TRACK_WIDTH_FACTOR / 2

    layout = []
    for axle in range(axles):
        x = wheelbase / 2 - axle * axle_spacing
        for y in (half_track, -half_track):
            layout.append((x, y))
    return layout


def wheel_center_local_z(config: RoverConfig) -> float:
    """Wheel-center height relative to the chassis center (negative = below)."""
    return -CHASSIS_HALF_HEIGHT_M - config.ground_clearance_m + config.wheel_diameter_m / 2


def chassis_rest_height(config: RoverConfig) -> float:
    """Chassis-center height above ground when all wheels touch down."""
    return CHASSIS_HALF_HEIGHT_M + config.ground_clearance_m


def chassis_half_extents(config: RoverConfig) -> tuple[float, float, float]:
    axles = config.wheel_count // 2
    wheel_radius = config.wheel_diameter_m / 2
    return (
        config.wheel_diameter_m * _AXLE_SPACING_FACTOR * max(axles - 1, 1) / 2 + wheel_radius,
        config.wheel_diameter_m * _TRACK_WIDTH_FACTOR / 2 - wheel_radius * 0.3,
        CHASSIS_HALF_HEIGHT_M,
    )


def wheel_mass_kg(config: RoverConfig) -> float:
    return config.mass_kg * _WHEEL_MASS_FRACTION


def chassis_mass_kg(config: RoverConfig) -> float:
    return config.mass_kg - wheel_mass_kg(config) * config.wheel_count
