"""Contract A. pybullet_sim.py, fake_sim.py, and (later) isaac_sim.py all
implement this. The orchestrator must never import a concrete backend
directly — only this ABC — so swapping simulators is a one-line change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from contracts.schemas import PlanetProfile, RoverConfig, SimMetrics, TerrainParams


class SimInterface(ABC):
    @abstractmethod
    def run_trial(
        self,
        rover_config: RoverConfig,
        planet: PlanetProfile,
        terrain_params: TerrainParams,
    ) -> SimMetrics:
        """Run one full episode and return its metrics.

        Implementations own their own world setup/teardown per call — the
        caller does not manage a reset/step/close lifecycle.
        """
