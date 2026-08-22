"""SimInterface implementation backed by PyBullet (Contract A: run_trial())."""

from __future__ import annotations

import math
import time

import numpy as np
import pybullet as p

from contracts.schemas import Outcome, PlanetProfile, RoverConfig, SimMetrics, TerrainParams
from sim.geometry import chassis_rest_height
from sim.interface import SimInterface
from sim.terrain_gen import RESOLUTION, TERRAIN_SIZE_M, generate_terrain
from sim.vehicle import apply_drive, build_rover

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


class PyBulletSim(SimInterface):
    def __init__(self, gui: bool = False) -> None:
        self._gui = gui

    def run_trial(
        self,
        rover_config: RoverConfig,
        planet: PlanetProfile,
        terrain_params: TerrainParams,
    ) -> SimMetrics:
        wall_clock_start = time.perf_counter()

        mode = p.GUI if self._gui else p.DIRECT
        client_id = p.connect(mode)
        try:
            p.setGravity(0, 0, -planet.gravity_ms2, physicsClientId=client_id)
            p.setTimeStep(_DT, physicsClientId=client_id)

            heights, rocks = generate_terrain(terrain_params)
            cell = TERRAIN_SIZE_M / RESOLUTION
            terrain_shape = p.createCollisionShape(
                p.GEOM_HEIGHTFIELD,
                meshScale=[cell, cell, 1.0],
                heightfieldData=heights.flatten().tolist(),
                numHeightfieldRows=RESOLUTION,
                numHeightfieldColumns=RESOLUTION,
                physicsClientId=client_id,
            )
            terrain_id = p.createMultiBody(0, terrain_shape, physicsClientId=client_id)
            p.changeDynamics(terrain_id, -1, lateralFriction=planet.surface_friction, physicsClientId=client_id)

            for rock in rocks:
                if rock.shape == "sphere":
                    rock_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=rock.radius, physicsClientId=client_id)
                    orn = (0, 0, 0, 1)
                else:
                    rock_shape = p.createCollisionShape(
                        p.GEOM_BOX, halfExtents=rock.half_extents, physicsClientId=client_id
                    )
                    orn = p.getQuaternionFromEuler([0, 0, math.radians(rock.yaw_deg)])
                rock_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=rock_shape,
                    basePosition=rock.position,
                    baseOrientation=orn,
                    physicsClientId=client_id,
                )
                p.changeDynamics(rock_id, -1, lateralFriction=planet.surface_friction, physicsClientId=client_id)

            start_x = -TERRAIN_SIZE_M / 2 + _START_MARGIN_M
            start_z = float(_sample_height(heights, start_x, 0.0)) + chassis_rest_height(rover_config) + 0.05
            body_id, wheel_links = build_rover(client_id, rover_config, start_pos=(start_x, 0.0, start_z))
            wheel_radius = rover_config.wheel_diameter_m / 2

            for _ in range(_SETTLE_STEPS):
                p.stepSimulation(physicsClientId=client_id)

            distance_m = 0.0
            max_tilt_deg = 0.0
            max_wheel_slip_pct = 0.0
            time_to_stall_s = None
            slow_steps = 0
            outcome = Outcome.TIMEOUT
            sim_elapsed_s = 0.0

            last_pos = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)[0]
            final_x = last_pos[0]

            for step in range(_MAX_STEPS):
                apply_drive(client_id, body_id, wheel_links, rover_config, _THROTTLE)
                p.stepSimulation(physicsClientId=client_id)
                sim_elapsed_s += _DT

                pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
                roll, pitch, _ = p.getEulerFromQuaternion(orn)
                tilt_deg = max(abs(roll), abs(pitch)) * 180 / math.pi
                max_tilt_deg = max(max_tilt_deg, tilt_deg)
                final_x = pos[0]

                step_dist = math.dist(pos, last_pos)
                distance_m += step_dist
                last_pos = pos

                linear_vel, _ = p.getBaseVelocity(body_id, physicsClientId=client_id)
                body_speed = math.hypot(linear_vel[0], linear_vel[1])
                if step >= _SPINUP_GRACE_STEPS:
                    wheel_speed = _mean_wheel_surface_speed(client_id, body_id, wheel_links, wheel_radius)
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
        finally:
            p.disconnect(physicsClientId=client_id)


def _sample_height(heights: np.ndarray, x: float, y: float) -> float:
    n = heights.shape[0]
    cell = TERRAIN_SIZE_M / n
    col = int(np.clip((x + TERRAIN_SIZE_M / 2) / cell, 0, n - 1))
    row = int(np.clip((y + TERRAIN_SIZE_M / 2) / cell, 0, n - 1))
    return float(heights[row, col])


def _mean_wheel_surface_speed(client_id: int, body_id: int, wheel_links: list[int], wheel_radius: float) -> float:
    speeds = []
    for link in wheel_links:
        joint_state = p.getJointState(body_id, link, physicsClientId=client_id)
        angular_velocity = joint_state[1]
        speeds.append(abs(angular_velocity) * wheel_radius)
    return sum(speeds) / len(speeds) if speeds else 0.0
