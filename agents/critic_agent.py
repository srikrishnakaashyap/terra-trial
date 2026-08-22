"""Diagnoses why one trial ended the way it did, and names the next thing to attack."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from agents.model_client import ModelClient
from contracts.schemas import CriticOutput, RoverConfig, SimMetrics, TerrainParams

_PROMPT_PATH = Path(__file__).parent / "prompts" / "critic_agent.txt"

_RESPONSE_SCHEMA = {
    "diagnosis": "string",
    "primary_weakness": "string",
    "confidence": "'low' | 'medium' | 'high'",
    "evidence": "list of strings, each citing a specific metric value",
    "next_target": "string",
}


class CriticAgent:
    def __init__(self, client: ModelClient) -> None:
        self._client = client
        self._system_prompt = _PROMPT_PATH.read_text()

    def diagnose(
        self,
        rover_config: RoverConfig,
        terrain_params: TerrainParams,
        metrics: SimMetrics,
    ) -> CriticOutput:
        user_prompt = (
            f"Rover config: {json.dumps(dataclasses.asdict(rover_config), indent=2)}\n\n"
            f"Terrain: {json.dumps(dataclasses.asdict(terrain_params), indent=2)}\n\n"
            f"Metrics: {json.dumps(dataclasses.asdict(metrics), indent=2)}"
        )
        raw = self._client.complete(self._system_prompt, user_prompt, _RESPONSE_SCHEMA)
        return CriticOutput(
            diagnosis=str(raw.get("diagnosis", "")),
            primary_weakness=str(raw.get("primary_weakness", "unknown")),
            confidence=str(raw.get("confidence", "low")),
            evidence=[str(e) for e in raw.get("evidence", [])],
            next_target=str(raw.get("next_target", "")),
        )
