"""STRETCH: Isaac Sim backend implementing the same SimInterface as pybullet_sim.py.

Not implemented yet — requires the Isaac Sim Python environment (omniverse
kit), which isn't a pip-installable dependency of this project. Swap this in
via main.py once that environment is available. See the build plan section 8
for the spike rules (Member C only, 90-minute time box, after the PyBullet
demo runs end to end).
"""

from __future__ import annotations

from contracts.schemas import PlanetProfile, RoverConfig, SimMetrics, TerrainParams
from sim.interface import SimInterface


class IsaacSim(SimInterface):
    def __init__(self, gui: bool = False) -> None:
        self._gui = gui

    def run_trial(
        self,
        rover_config: RoverConfig,
        planet: PlanetProfile,
        terrain_params: TerrainParams,
    ) -> SimMetrics:
        raise NotImplementedError("Isaac Sim backend is a stretch goal, not yet implemented")
