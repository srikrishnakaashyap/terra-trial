"""Proposes the next TerrainParams to test, given rover config and run history."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from agents.model_client import ModelClient
from contracts.schemas import IterationResult, PlanetProfile, RoverConfig, TerrainParams, clamp_terrain_params

_PROMPT_PATH = Path(__file__).parent / "prompts" / "terrain_agent.txt"

_RESPONSE_SCHEMA = {
    "slope_deg": "float, 0-40",
    "rock_density": "float, 0-1",
    "rock_size_range_m": "[float, float], each 0.01-0.8",
    "roughness": "float, 0-1",
    "targets_weakness": "string",
    "reasoning": "string",
}


class TerrainAgent:
    def __init__(self, client: ModelClient) -> None:
        self._client = client
        self._system_prompt = _PROMPT_PATH.read_text()

    def propose(
        self,
        planet: PlanetProfile,
        rover_config: RoverConfig,
        history: list[IterationResult],
    ) -> TerrainParams:
        user_prompt = (
            f"Planet: {json.dumps(dataclasses.asdict(planet), indent=2)}\n\n"
            f"Rover config: {json.dumps(dataclasses.asdict(rover_config), indent=2)}\n\n"
            f"History: {json.dumps(_history_summary(history), indent=2) if history else 'none yet, this is iteration 1'}"
        )
        raw = self._client.complete(self._system_prompt, user_prompt, _RESPONSE_SCHEMA)
        return clamp_terrain_params(raw)


def _history_summary(history: list[IterationResult]) -> list[dict]:
    return [
        {
            "iteration": r.iteration,
            "terrain": dataclasses.asdict(r.terrain_params),
            "outcome": r.metrics.outcome,
            "completion_pct": r.metrics.completion_pct,
            "max_wheel_slip_pct": r.metrics.max_wheel_slip_pct,
            "max_tilt_deg": r.metrics.max_tilt_deg,
            "critic_diagnosis": r.critique.diagnosis,
            "critic_next_target": r.critique.next_target,
        }
        for r in history
    ]
