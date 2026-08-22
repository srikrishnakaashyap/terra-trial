"""SimInterface stub that returns canned metrics instead of running physics.

Lets the agent/orchestrator loop be built and tested end to end before the
PyBullet backend exists, and lets it keep being tested without a GPU/PyBullet
install at all. Metrics are a deterministic function of terrain difficulty
(not random) so a test run is reproducible: harder terrain relative to the
rover's torque produces worse outcomes.
"""

from __future__ import annotations

from contracts.schemas import Outcome, PlanetProfile, RoverConfig, SimMetrics, TerrainParams
from sim.interface import SimInterface


class FakeSim(SimInterface):
    def run_trial(
        self,
        rover_config: RoverConfig,
        planet: PlanetProfile,
        terrain_params: TerrainParams,
    ) -> SimMetrics:
        difficulty = (
            terrain_params.slope_deg / 40.0 * 0.4
            + terrain_params.rock_density * 0.4
            + terrain_params.roughness * 0.2
        )
        torque_margin = min(1.0, rover_config.max_motor_torque_nm / 100.0)
        strain = max(0.0, min(1.0, difficulty - torque_margin * 0.3))

        completion_pct = max(0.0, (1.0 - strain) * 100)
        max_tilt_deg = 10.0 + strain * 50.0
        max_wheel_slip_pct = strain * 100.0

        if max_tilt_deg >= 55.0:
            outcome = Outcome.TIPPED
            time_to_stall_s = None
        elif completion_pct >= 95.0:
            outcome = Outcome.COMPLETED
            time_to_stall_s = None
        elif strain > 0.5:
            outcome = Outcome.STALLED
            time_to_stall_s = 5.0 + (1.0 - strain) * 10.0
        else:
            outcome = Outcome.TIMEOUT
            time_to_stall_s = None

        return SimMetrics(
            completion_pct=completion_pct,
            distance_m=completion_pct / 100 * 26.0,
            max_tilt_deg=max_tilt_deg,
            max_wheel_slip_pct=max_wheel_slip_pct,
            time_to_stall_s=time_to_stall_s,
            outcome=outcome,
            wall_clock_s=0.05,
        )
