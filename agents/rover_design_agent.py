"""Co-evolution mode only. Watches how the current rover failed and proposes
a redesign to survive it — the adversarial counterpart to the Terrain Agent.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from agents.model_client import ModelClient
from contracts.schemas import (
    CriticOutput,
    RoverConfig,
    RoverRedesign,
    SimMetrics,
    TerrainParams,
    clamp_rover_config,
)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "rover_design_agent.txt"

_RESPONSE_SCHEMA = {
    "name": "string",
    "wheel_count": "4 | 6",
    "wheel_diameter_m": "float, 0.1-0.6",
    "ground_clearance_m": "float, 0.05-0.5",
    "suspension_type": "'rigid' | 'rocker_bogie' | 'independent'",
    "mass_kg": "float, 10-500",
    "max_motor_torque_nm": "float, 1-100",
    "targets_weakness": "string",
    "reasoning": "string",
}


class RoverDesignAgent:
    def __init__(self, client: ModelClient) -> None:
        self._client = client
        self._system_prompt = _PROMPT_PATH.read_text()

    def redesign(
        self,
        rover_config: RoverConfig,
        terrain_params: TerrainParams,
        metrics: SimMetrics,
        critique: CriticOutput,
    ) -> RoverRedesign:
        user_prompt = (
            f"Current rover config: {json.dumps(dataclasses.asdict(rover_config), indent=2)}\n\n"
            f"Terrain it just attempted: {json.dumps(dataclasses.asdict(terrain_params), indent=2)}\n\n"
            f"Metrics: {json.dumps(dataclasses.asdict(metrics), indent=2)}\n\n"
            f"Critic's diagnosis: {json.dumps(dataclasses.asdict(critique), indent=2)}"
        )
        raw = self._client.complete(self._system_prompt, user_prompt, _RESPONSE_SCHEMA)
        new_config = clamp_rover_config(raw, base=rover_config)
        return RoverRedesign(
            rover_config=new_config,
            targets_weakness=str(raw.get("targets_weakness", critique.primary_weakness)),
            reasoning=str(raw.get("reasoning", "")),
        )
