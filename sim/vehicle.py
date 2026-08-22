"""Builds a PyBullet multibody rover from a RoverConfig.

Geometry comes from sim/geometry.py (shared with the MuJoCo backend) —
this module only turns those numbers into PyBullet API calls.
"""

from __future__ import annotations

import pybullet as p

from contracts.schemas import RoverConfig
from sim.geometry import (
    CHASSIS_HALF_HEIGHT_M,
    chassis_half_extents,
    chassis_mass_kg,
    target_wheel_angular_velocity,
    wheel_center_local_z,
    wheel_layout,
    wheel_mass_kg,
)


def build_rover(client_id: int, config: RoverConfig, start_pos: tuple[float, float, float]) -> tuple[int, list[int]]:
    """Create the rover body. Returns (body_id, wheel_link_indices)."""
    wheel_radius = config.wheel_diameter_m / 2
    layout = wheel_layout(config)
    wheel_center_z = wheel_center_local_z(config)

    chassis_shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=chassis_half_extents(config), physicsClientId=client_id
    )
    wheel_shape = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=wheel_radius,
        height=config.wheel_diameter_m * 0.25,
        physicsClientId=client_id,
    )

    n_wheels = len(layout)
    wheel_mass = wheel_mass_kg(config)
    chassis_mass = chassis_mass_kg(config)
    wheel_positions = [(x, y, wheel_center_z) for x, y in layout]

    body_id = p.createMultiBody(
        baseMass=chassis_mass,
        baseCollisionShapeIndex=chassis_shape,
        basePosition=start_pos,
        linkMasses=[wheel_mass] * n_wheels,
        linkCollisionShapeIndices=[wheel_shape] * n_wheels,
        linkVisualShapeIndices=[-1] * n_wheels,
        linkPositions=wheel_positions,
        linkOrientations=[p.getQuaternionFromEuler([1.5708, 0, 0])] * n_wheels,
        linkInertialFramePositions=[(0, 0, 0)] * n_wheels,
        linkInertialFrameOrientations=[(0, 0, 0, 1)] * n_wheels,
        linkParentIndices=[0] * n_wheels,
        linkJointTypes=[p.JOINT_REVOLUTE] * n_wheels,
        linkJointAxis=[(0, 1, 0)] * n_wheels,
        physicsClientId=client_id,
    )

    for link in range(n_wheels):
        p.changeDynamics(body_id, link, lateralFriction=1.0, physicsClientId=client_id)

    return body_id, list(range(n_wheels))


def apply_drive(client_id: int, body_id: int, wheel_links: list[int], config: RoverConfig, throttle: float) -> None:
    """Drive all wheels toward a target angular velocity, torque-capped at
    max_motor_torque_nm. throttle in [-1, 1] scales the target velocity —
    this is what makes traction (not just an instant torque impulse) the
    limiting factor on hard terrain."""
    target_velocity = throttle * target_wheel_angular_velocity(config)
    for link in wheel_links:
        p.setJointMotorControl2(
            body_id,
            link,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=target_velocity,
            force=config.max_motor_torque_nm,
            physicsClientId=client_id,
        )
