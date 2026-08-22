"""Drives one run: terrain_agent proposes terrain, sim.run_trial() executes
it, critic_agent diagnoses the result, repeat, then report_agent summarizes.
Zero LLM calls happen here — this is pure control flow (see build plan
section 1.3). State is persisted to runs/<run_id>/run_state.json after every
iteration so the dashboard can tail it live.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from agents.critic_agent import CriticAgent
from agents.model_client import ModelClient
from agents.report_agent import ReportAgent
from agents.rover_design_agent import RoverDesignAgent
from agents.terrain_agent import TerrainAgent
from contracts.schemas import IterationResult, Outcome, PlanetProfile, RoverConfig, RunState
from sim.interface import SimInterface

RUNS_DIR = Path(__file__).parent / "runs"


class Orchestrator:
    def __init__(
        self,
        sim: SimInterface,
        model_client: ModelClient,
        planet: PlanetProfile,
        rover_config: RoverConfig,
        max_iterations: int = 4,
    ) -> None:
        self._sim = sim
        self._terrain_agent = TerrainAgent(model_client)
        self._critic_agent = CriticAgent(model_client)
        self._report_agent = ReportAgent(model_client)
        self._max_iterations = max_iterations

        run_id = time.strftime("%Y%m%d-%H%M%S")
        self._run_dir = RUNS_DIR / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._state = RunState(run_id=run_id, planet=planet, rover_config=rover_config)

    def run(self) -> RunState:
        self._state.status = "running"
        self._save()

        try:
            for i in range(1, self._max_iterations + 1):
                self._log(f"iteration {i}/{self._max_iterations}: Terrain Agent proposing terrain...")
                terrain_params = self._terrain_agent.propose(
                    self._state.planet, self._state.rover_config, self._state.iterations
                )
                self._log(
                    f"iteration {i}/{self._max_iterations}: terrain -> slope={terrain_params.slope_deg:.1f}deg "
                    f"rock_density={terrain_params.rock_density:.2f} roughness={terrain_params.roughness:.2f}"
                )

                self._log(f"iteration {i}/{self._max_iterations}: running trial in sim...")
                metrics = self._sim.run_trial(self._state.rover_config, self._state.planet, terrain_params)
                self._log(
                    f"iteration {i}/{self._max_iterations}: outcome={metrics.outcome.value} "
                    f"completion={metrics.completion_pct:.0f}%"
                )

                self._log(f"iteration {i}/{self._max_iterations}: Critic Agent diagnosing...")
                critique = self._critic_agent.diagnose(self._state.rover_config, terrain_params, metrics)
                self._log(
                    f"iteration {i}/{self._max_iterations}: critic -> {critique.primary_weakness} "
                    f"({critique.confidence} confidence)"
                )

                self._state.iterations.append(
                    IterationResult(
                        iteration=i,
                        terrain_params=terrain_params,
                        metrics=metrics,
                        critique=critique,
                    )
                )
                self._save()

                if metrics.outcome == Outcome.TIPPED and i >= 2:
                    self._log("hard failure found, stopping early")
                    break

            self._log("Report Agent summarizing run...")
            self._state.final_report = self._report_agent.summarize(
                self._state.rover_config, self._state.planet, self._state.iterations
            )
            self._state.status = "complete"
        except Exception:
            self._state.status = "failed"
            raise
        finally:
            self._save()

        return self._state

    def _log(self, msg: str) -> None:
        print(f"[{self._state.run_id}] {msg}", flush=True)

    def _save(self) -> None:
        (self._run_dir / "run_state.json").write_text(json.dumps(asdict(self._state), indent=2, default=str))


class CoevolveOrchestrator:
    """Adversarial co-evolution: the Terrain Agent tries to break the
    current rover, the Rover Design Agent redesigns it in response, and the
    *rover* evolves iteration to iteration — not just the terrain. Kept as
    a separate class from Orchestrator (not a flag/subclass) so the proven
    single-agent escalation loop is untouched by this addition; both share
    the same contracts, sim backend, and every agent but the new one.
    """

    def __init__(
        self,
        sim: SimInterface,
        model_client: ModelClient,
        planet: PlanetProfile,
        initial_rover_config: RoverConfig,
        max_iterations: int = 4,
    ) -> None:
        self._sim = sim
        self._terrain_agent = TerrainAgent(model_client)
        self._critic_agent = CriticAgent(model_client)
        self._rover_design_agent = RoverDesignAgent(model_client)
        self._report_agent = ReportAgent(model_client)
        self._max_iterations = max_iterations

        run_id = time.strftime("%Y%m%d-%H%M%S")
        self._run_dir = RUNS_DIR / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._state = RunState(run_id=run_id, planet=planet, rover_config=initial_rover_config)

    def run(self) -> RunState:
        self._state.status = "running"
        self._save()

        current_rover = self._state.rover_config
        try:
            for i in range(1, self._max_iterations + 1):
                self._log(f"iteration {i}/{self._max_iterations}: rover={current_rover.name} — Terrain Agent proposing terrain...")
                terrain_params = self._terrain_agent.propose(self._state.planet, current_rover, self._state.iterations)
                self._log(
                    f"iteration {i}/{self._max_iterations}: terrain -> slope={terrain_params.slope_deg:.1f}deg "
                    f"rock_density={terrain_params.rock_density:.2f} roughness={terrain_params.roughness:.2f}"
                )

                self._log(f"iteration {i}/{self._max_iterations}: running trial in sim...")
                metrics = self._sim.run_trial(current_rover, self._state.planet, terrain_params)
                self._log(
                    f"iteration {i}/{self._max_iterations}: outcome={metrics.outcome.value} "
                    f"completion={metrics.completion_pct:.0f}%"
                )

                self._log(f"iteration {i}/{self._max_iterations}: Critic Agent diagnosing...")
                critique = self._critic_agent.diagnose(current_rover, terrain_params, metrics)
                self._log(
                    f"iteration {i}/{self._max_iterations}: critic -> {critique.primary_weakness} "
                    f"({critique.confidence} confidence)"
                )

                self._log(f"iteration {i}/{self._max_iterations}: Rover Design Agent redesigning...")
                redesign = self._rover_design_agent.redesign(current_rover, terrain_params, metrics, critique)
                self._log(
                    f"iteration {i}/{self._max_iterations}: redesign -> {redesign.rover_config.name} "
                    f"(targeting {redesign.targets_weakness})"
                )

                self._state.iterations.append(
                    IterationResult(
                        iteration=i,
                        terrain_params=terrain_params,
                        metrics=metrics,
                        critique=critique,
                        rover_config=current_rover,
                        redesign=redesign,
                    )
                )
                self._save()

                current_rover = redesign.rover_config  # evolve for next iteration

                if metrics.outcome == Outcome.TIPPED and i >= 2:
                    self._log("hard failure found, stopping early")
                    break

            self._log("Report Agent summarizing run...")
            self._state.final_report = self._report_agent.summarize(current_rover, self._state.planet, self._state.iterations)
            self._state.status = "complete"
        except Exception:
            self._state.status = "failed"
            raise
        finally:
            self._save()

        return self._state

    def _log(self, msg: str) -> None:
        print(f"[{self._state.run_id}] {msg}", flush=True)

    def _save(self) -> None:
        (self._run_dir / "run_state.json").write_text(json.dumps(asdict(self._state), indent=2, default=str))
