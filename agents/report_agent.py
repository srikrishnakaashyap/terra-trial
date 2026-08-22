"""Writes the final human-readable report for a completed run."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from agents.model_client import ModelClient
from contracts.schemas import IterationResult, PlanetProfile, RoverConfig

_PROMPT_PATH = Path(__file__).parent / "prompts" / "report_agent.txt"


class ReportAgent:
    def __init__(self, client: ModelClient) -> None:
        self._client = client
        self._system_prompt = _PROMPT_PATH.read_text()

    def summarize(
        self,
        rover_config: RoverConfig,
        planet: PlanetProfile,
        history: list[IterationResult],
    ) -> str:
        user_prompt = (
            f"Final rover config: {json.dumps(dataclasses.asdict(rover_config), indent=2)}\n\n"
            f"Planet: {json.dumps(dataclasses.asdict(planet), indent=2)}\n\n"
            f"Run history: {json.dumps([_compact_iteration(r) for r in history], indent=2)}"
        )
        return self._client.generate(self._system_prompt, user_prompt)


def _compact_iteration(r: IterationResult) -> dict:
    """A trimmed summary, not the full dataclass — the full nested history
    (especially in co-evolve mode, which adds a rover_config + redesign per
    iteration) is large enough to push real inference calls toward timeout
    for no benefit; the report doesn't need rock_size_range_m, wall_clock_s,
    or the full evidence list to write a good summary."""
    entry: dict = {
        "iteration": r.iteration,
        "terrain": {
            "slope_deg": r.terrain_params.slope_deg,
            "rock_density": r.terrain_params.rock_density,
            "roughness": r.terrain_params.roughness,
        },
        "outcome": r.metrics.outcome,
        "completion_pct": round(r.metrics.completion_pct, 1),
        "max_wheel_slip_pct": round(r.metrics.max_wheel_slip_pct, 1),
        "max_tilt_deg": round(r.metrics.max_tilt_deg, 1),
        "critic_diagnosis": r.critique.diagnosis,
        "critic_primary_weakness": r.critique.primary_weakness,
    }
    if r.rover_config is not None:
        entry["rover_used"] = {
            "name": r.rover_config.name,
            "max_motor_torque_nm": r.rover_config.max_motor_torque_nm,
            "mass_kg": r.rover_config.mass_kg,
            "wheel_diameter_m": r.rover_config.wheel_diameter_m,
            "ground_clearance_m": r.rover_config.ground_clearance_m,
            "suspension_type": r.rover_config.suspension_type,
        }
    if r.redesign is not None:
        entry["redesign_reasoning"] = r.redesign.reasoning
    return entry
