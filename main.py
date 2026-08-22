"""Entry point + rover presets + sim/model backend registries.

Backends are imported lazily inside each factory so a missing/broken
dependency for one backend (e.g. pybullet failing to build, or NemoClaw not
being installed on this machine) can't break the others. Swapping backends —
including dropping in sim/isaac_sim.py or agents/model_client.py's
NemoClawModelClient tomorrow once they're implemented — is a flag, no code
changes elsewhere; that's what freezing SimInterface and ModelClient buys.
"""

import argparse
from pathlib import Path
from typing import Callable

from agents.model_client import ModelClient
from contracts.schemas import RoverConfig, SuspensionType, load_planet_profiles
from orchestrator import CoevolveOrchestrator, Orchestrator
from sim.interface import SimInterface

PLANET_PROFILES_PATH = Path(__file__).parent / "contracts" / "planet_profiles.json"

# Presets so a demo run never needs values typed live (build plan section 7).
ROVER_PRESETS = {
    "scout": RoverConfig(
        name="Scout-4W",
        wheel_count=4,
        wheel_diameter_m=0.25,
        ground_clearance_m=0.15,
        suspension_type=SuspensionType.RIGID,
        mass_kg=50.0,
        max_motor_torque_nm=15.0,
    ),
    "hauler": RoverConfig(
        name="Hauler-6W",
        wheel_count=6,
        wheel_diameter_m=0.35,
        ground_clearance_m=0.20,
        suspension_type=SuspensionType.INDEPENDENT,
        mass_kg=250.0,
        max_motor_torque_nm=60.0,
    ),
    "explorer": RoverConfig(
        name="Explorer-6W",
        wheel_count=6,
        wheel_diameter_m=0.30,
        ground_clearance_m=0.35,
        suspension_type=SuspensionType.ROCKER_BOGIE,
        mass_kg=120.0,
        max_motor_torque_nm=40.0,
    ),
}


def _mujoco_backend(gui: bool) -> SimInterface:
    from sim.mujoco_sim import MuJoCoSim

    return MuJoCoSim(gui=gui)


def _pybullet_backend(gui: bool) -> SimInterface:
    from sim.pybullet_sim import PyBulletSim

    return PyBulletSim(gui=gui)


def _isaac_backend(gui: bool) -> SimInterface:
    from sim.isaac_sim import IsaacSim

    return IsaacSim(gui=gui)


# name -> (gui: bool) -> SimInterface. Add a new backend by adding one entry
# here; nothing else in the codebase references a concrete backend by name.
SIM_BACKENDS: dict[str, Callable[[bool], SimInterface]] = {
    "mujoco": _mujoco_backend,
    "pybullet": _pybullet_backend,
    "isaac": _isaac_backend,
}


def _anthropic_model(model_name: str | None) -> ModelClient:
    from agents.model_client import AnthropicClient

    return AnthropicClient(model=model_name) if model_name else AnthropicClient()


def _nemoclaw_model(model_name: str | None) -> ModelClient:
    from agents.model_client import NemoClawModelClient

    return NemoClawModelClient(model=model_name)


# name -> (model_name: str | None) -> ModelClient. Add a new one here when
# NemoClawModelClient.generate() is actually implemented (see its docstring).
MODEL_BACKENDS: dict[str, Callable[[str | None], ModelClient]] = {
    "anthropic": _anthropic_model,
    "nemoclaw": _nemoclaw_model,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one terra-trial adversarial terrain campaign.")
    parser.add_argument("--planet", default="mars", choices=["mars", "moon", "europa"])
    parser.add_argument("--rover", default="scout", choices=list(ROVER_PRESETS))
    parser.add_argument("--backend", default="mujoco", choices=list(SIM_BACKENDS))
    parser.add_argument("--model-backend", default="anthropic", choices=list(MODEL_BACKENDS))
    parser.add_argument("--model-name", default=None, help="Override the model name/tag for the chosen model backend")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument(
        "--mode",
        default="escalate",
        choices=["escalate", "coevolve"],
        help="escalate: terrain gets harder against a fixed rover (default, proven). "
        "coevolve: a Rover Design Agent also redesigns the rover each iteration in response "
        "to the Critic's diagnosis — adversarial terrain-vs-rover co-evolution.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show the sim's live viewer, where supported. On macOS with the mujoco backend, "
        "run this script with `mjpython`, not `python` — mujoco's viewer requires it.",
    )
    args = parser.parse_args()

    planet = load_planet_profiles(PLANET_PROFILES_PATH)[args.planet]
    rover_config = ROVER_PRESETS[args.rover]

    sim = SIM_BACKENDS[args.backend](args.gui)
    model_client = MODEL_BACKENDS[args.model_backend](args.model_name)

    if args.mode == "coevolve":
        orchestrator = CoevolveOrchestrator(sim, model_client, planet, rover_config, max_iterations=args.iterations)
    else:
        orchestrator = Orchestrator(sim, model_client, planet, rover_config, max_iterations=args.iterations)

    state = orchestrator.run()
    print(f"Run {state.run_id} finished with status={state.status}")


if __name__ == "__main__":
    main()
