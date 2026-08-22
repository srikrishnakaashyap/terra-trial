"""SimInterface implementation backed by MuJoCo (Contract A: run_trial()).

Primary backend for this project: pybullet has no prebuilt wheel and its
vendored zlib fails to compile against the macOS SDK on this dev machine
(see README), while mujoco ships a working wheel and has strong ARM
support. sim/pybullet_sim.py is kept as an alternate backend behind the
same SimInterface in case pybullet builds cleanly elsewhere (e.g. the
GB10's toolchain) — swapping is a one-line change in main.py's backend
registry, same as swapping in sim/isaac_sim.py tomorrow.

A trial builds a throwaway MJCF (MuJoCo XML) model from RoverConfig +
PlanetProfile + TerrainParams, runs it to completion, and discards it —
matching the "own world setup/teardown per call" contract in interface.py.
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext

import mujoco
import mujoco.viewer
import numpy as np

from contracts.schemas import Outcome, PlanetProfile, RoverConfig, SimMetrics, TerrainParams
from sim.geometry import (
    CHASSIS_HALF_HEIGHT_M,
    chassis_half_extents,
    chassis_mass_kg,
    target_wheel_angular_velocity,
    wheel_center_local_z,
    wheel_layout,
    wheel_mass_kg,
)
from sim.interface import SimInterface
from sim.terrain_gen import TERRAIN_SIZE_M, Rock, generate_terrain

_DT = 1 / 60
_SETTLE_STEPS = 100
_MAX_TRIAL_S = 30.0
_MAX_STEPS = int(_MAX_TRIAL_S / _DT)
_STALL_VELOCITY_MPS = 0.02
_STALL_GRACE_STEPS = 120  # ~2s at 60Hz before a slow rover counts as stalled
_ROLLOVER_TILT_DEG = 55.0
_START_MARGIN_M = 2.0
_THROTTLE = 1.0
_SPINUP_GRACE_STEPS = 30  # ignore wheel-slip readings during initial speed ramp-up
_WHEEL_JOINT_PREFIX = "wheel_"


class MuJoCoSim(SimInterface):
    def __init__(self, gui: bool = False) -> None:
        self._gui = gui

    def run_trial(
        self,
        rover_config: RoverConfig,
        planet: PlanetProfile,
        terrain_params: TerrainParams,
    ) -> SimMetrics:
        wall_clock_start = time.perf_counter()

        heights, rocks = generate_terrain(terrain_params)
        xml, hfield_z_base, hfield_elevation_z = _build_mjcf(rover_config, planet, heights, rocks)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        n = heights.shape[0]
        model.hfield_data[: n * n] = ((heights - hfield_z_base) / hfield_elevation_z).flatten()

        start_x = -TERRAIN_SIZE_M / 2 + _START_MARGIN_M
        start_z = _sample_height(heights, start_x, 0.0) + _rest_height(rover_config) + 0.05
        chassis_joint_adr = model.joint("chassis").qposadr[0]
        data.qpos[chassis_joint_adr : chassis_joint_adr + 3] = [start_x, 0.0, start_z]
        mujoco.mj_forward(model, data)

        n_wheels = rover_config.wheel_count
        wheel_actuator_ids = [model.actuator(f"{_WHEEL_JOINT_PREFIX}{i}").id for i in range(n_wheels)]
        wheel_dof_adrs = [model.joint(f"{_WHEEL_JOINT_PREFIX}{i}").dofadr[0] for i in range(n_wheels)]
        wheel_radius = rover_config.wheel_diameter_m / 2
        target_velocity = _THROTTLE * target_wheel_angular_velocity(rover_config)

        distance_m = 0.0
        max_tilt_deg = 0.0
        max_wheel_slip_pct = 0.0
        time_to_stall_s = None
        slow_steps = 0
        outcome = Outcome.TIMEOUT
        sim_elapsed_s = 0.0

        chassis_body_id = model.body("chassis").id

        # launch_passive requires the script be run under `mjpython`, not
        # `python`, on macOS (Cocoa's main-thread GUI restriction) — mujoco
        # raises a clear RuntimeError itself if that's not the case.
        viewer_cm = mujoco.viewer.launch_passive(model, data) if self._gui else nullcontext(None)
        with viewer_cm as viewer:
            if viewer is not None:
                # MuJoCo's default free camera auto-frames the whole model's
                # bounding box — dominated by the 30m terrain plane, which
                # makes a ~0.5m rover an invisible speck. Chase-cam on the
                # rover instead; lookat is updated every frame below.
                viewer.cam.distance = 4.0
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20
                viewer.cam.lookat[:] = data.xpos[chassis_body_id]

            for _ in range(_SETTLE_STEPS):
                if not _advance(model, data, viewer):
                    break

            last_pos = data.xpos[chassis_body_id].copy()
            final_x = last_pos[0]

            for _step in range(_MAX_STEPS):
                for actuator_id in wheel_actuator_ids:
                    data.ctrl[actuator_id] = target_velocity
                if not _advance(model, data, viewer):
                    break  # viewer window closed mid-trial
                sim_elapsed_s += _DT
                if viewer is not None:
                    viewer.cam.lookat[:] = data.xpos[chassis_body_id]

                pos = data.xpos[chassis_body_id]
                quat = data.xquat[chassis_body_id]  # (w, x, y, z)
                roll, pitch = _quat_to_roll_pitch(quat)
                tilt_deg = max(abs(roll), abs(pitch)) * 180 / math.pi
                max_tilt_deg = max(max_tilt_deg, tilt_deg)
                final_x = pos[0]

                step_dist = math.dist(pos, last_pos)
                distance_m += step_dist
                body_speed = step_dist / _DT
                last_pos = pos.copy()

                if _step >= _SPINUP_GRACE_STEPS:
                    wheel_speed = _mean_wheel_surface_speed(data, wheel_dof_adrs, wheel_radius)
                    if wheel_speed > 1e-6:
                        slip_pct = max(0.0, min(1.0, 1.0 - body_speed / wheel_speed)) * 100
                        max_wheel_slip_pct = max(max_wheel_slip_pct, slip_pct)

                if tilt_deg >= _ROLLOVER_TILT_DEG:
                    outcome = Outcome.TIPPED
                    break

                if final_x - start_x >= TERRAIN_SIZE_M - 2 * _START_MARGIN_M:
                    outcome = Outcome.COMPLETED
                    break

                if body_speed < _STALL_VELOCITY_MPS:
                    slow_steps += 1
                    if slow_steps == _STALL_GRACE_STEPS:
                        if time_to_stall_s is None:
                            time_to_stall_s = sim_elapsed_s
                        outcome = Outcome.STALLED
                        break
                else:
                    slow_steps = 0

        completion_pct = max(0.0, min(1.0, (final_x - start_x) / (TERRAIN_SIZE_M - 2 * _START_MARGIN_M))) * 100

        return SimMetrics(
            completion_pct=completion_pct,
            distance_m=distance_m,
            max_tilt_deg=max_tilt_deg,
            max_wheel_slip_pct=max_wheel_slip_pct,
            time_to_stall_s=time_to_stall_s,
            outcome=outcome,
            wall_clock_s=time.perf_counter() - wall_clock_start,
        )


def _advance(model: mujoco.MjModel, data: mujoco.MjData, viewer: mujoco.viewer.Handle | None) -> bool:
    """One physics step. With a viewer, paces to real time (a "completed"
    trial otherwise finishes in ~0.03s of wall-clock compute — too fast to
    watch) and syncs the window. Returns False if the viewer was closed."""
    if viewer is None:
        mujoco.mj_step(model, data)
        return True

    if not viewer.is_running():
        return False
    step_start = time.perf_counter()
    mujoco.mj_step(model, data)
    viewer.sync()
    remaining = _DT - (time.perf_counter() - step_start)
    if remaining > 0:
        time.sleep(remaining)
    return True


def _rest_height(config: RoverConfig) -> float:
    return CHASSIS_HALF_HEIGHT_M + config.ground_clearance_m


def _sample_height(heights: np.ndarray, x: float, y: float) -> float:
    n = heights.shape[0]
    cell = TERRAIN_SIZE_M / n
    col = int(np.clip((x + TERRAIN_SIZE_M / 2) / cell, 0, n - 1))
    row = int(np.clip((y + TERRAIN_SIZE_M / 2) / cell, 0, n - 1))
    return float(heights[row, col])


def _quat_to_roll_pitch(quat: np.ndarray) -> tuple[float, float]:
    """quat is MuJoCo's (w, x, y, z) order. Returns (roll, pitch) in radians."""
    w, x, y, z = quat
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


def _mean_wheel_surface_speed(data: mujoco.MjData, wheel_dof_adrs: list[int], wheel_radius: float) -> float:
    speeds = [abs(data.qvel[dof_adr]) * wheel_radius for dof_adr in wheel_dof_adrs]
    return sum(speeds) / len(speeds) if speeds else 0.0


def _rock_geom_xml(rock: Rock, friction: float, ground_rgb: str) -> str:
    x, y, z = rock.position
    friction_attr = f'friction="{friction} {friction} {friction}"'
    rgba_attr = f'rgba="{ground_rgb} 1"'
    if rock.shape == "sphere":
        return f'    <geom type="sphere" size="{rock.radius:.4f}" pos="{x:.4f} {y:.4f} {z:.4f}" {friction_attr} {rgba_attr}/>'
    hx, hy, hz = rock.half_extents
    return (
        f'    <geom type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}" pos="{x:.4f} {y:.4f} {z:.4f}" '
        f'euler="0 0 {rock.yaw_deg:.2f}" {friction_attr} {rgba_attr}/>'
    )


def _build_mjcf(
    rover_config: RoverConfig,
    planet: PlanetProfile,
    heights: np.ndarray,
    rocks: list[Rock],
) -> tuple[str, float, float]:
    """Returns (xml, hfield_z_base, hfield_elevation_z) — the latter two are
    needed by the caller to write model.hfield_data in normalized [0, 1]."""
    n = heights.shape[0]
    half_size = TERRAIN_SIZE_M / 2

    z_min = float(heights.min())
    elevation_z = max(float(heights.max()) - z_min, 1e-3)
    friction = planet.surface_friction
    ground_rgb = " ".join(f"{c:.3f}" for c in planet.palette.ground)

    rock_geoms = "\n".join(_rock_geom_xml(rock, friction, ground_rgb) for rock in rocks)

    half_x, half_y, half_z = chassis_half_extents(rover_config)
    wheel_radius = rover_config.wheel_diameter_m / 2
    wheel_half_len = rover_config.wheel_diameter_m * 0.125
    wheel_z = wheel_center_local_z(rover_config)
    chassis_mass = chassis_mass_kg(rover_config)
    wheel_mass = wheel_mass_kg(rover_config)

    wheel_bodies = []
    actuators = []
    for i, (wx, wy) in enumerate(wheel_layout(rover_config)):
        joint_name = f"{_WHEEL_JOINT_PREFIX}{i}"
        # euler="90 0 0" rotates the body-local z axis onto world -y, so both
        # the hinge axis (local z) and the cylinder's roll axis (local z)
        # end up aligned with the world y axis (sign handled by the motor's
        # gear below) — wheels spin about the world y axis.
        wheel_bodies.append(
            f'''    <body name="wheel_body_{i}" pos="{wx:.4f} {wy:.4f} {wheel_z:.4f}" euler="90 0 0">
      <joint name="{joint_name}" type="hinge" axis="0 0 1" damping="0.01"/>
      <geom type="cylinder" size="{wheel_radius:.4f} {wheel_half_len:.4f}" mass="{wheel_mass:.4f}" friction="1 1 1"/>
    </body>'''
        )
        # velocity (not motor/torque) actuator: ctrl is a target angular
        # velocity, force-capped at max_motor_torque_nm — matches the build
        # plan's "drive forward at constant target velocity" and makes
        # traction (not an instant full-torque impulse) the limiting factor.
        # gear=-1: euler="90 0 0" maps the body-local z axis (the hinge axis
        # below) onto world -y, not +y — negating the gear keeps positive
        # ctrl meaning "drive toward +x" without special-casing sign at the
        # call site, matching PyBullet's positive-throttle-forward convention.
        torque_cap = rover_config.max_motor_torque_nm
        actuators.append(
            f'    <velocity name="{joint_name}" joint="{joint_name}" gear="-1" kv="30" '
            f'forcerange="-{torque_cap:.4f} {torque_cap:.4f}"/>'
        )

    xml = f"""<mujoco model="terra_trial">
  <compiler angle="degree"/>
  <option timestep="{_DT}" gravity="0 0 -{planet.gravity_ms2}"/>
  <asset>
    <hfield name="terrain" nrow="{n}" ncol="{n}" size="{half_size} {half_size} {elevation_z} 0.1"/>
  </asset>
  <worldbody>
    <geom type="hfield" hfield="terrain" pos="0 0 {z_min}" friction="{friction} {friction} {friction}" rgba="{ground_rgb} 1"/>
{rock_geoms}
    <body name="chassis" pos="0 0 1">
      <joint name="chassis" type="free"/>
      <geom type="box" size="{half_x:.4f} {half_y:.4f} {half_z:.4f}" mass="{chassis_mass:.4f}"/>
{chr(10).join(wheel_bodies)}
    </body>
  </worldbody>
  <actuator>
{chr(10).join(actuators)}
  </actuator>
</mujoco>"""

    return xml, z_min, elevation_z
