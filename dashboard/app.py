"""Streamlit dashboard that tails runs/<run_id>/run_state.json.

Reads state only — never calls agents or the sim directly (build plan
section 1.1, L4).

Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

RUNS_DIR = Path(__file__).parent.parent / "runs"


def _list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted((d for d in RUNS_DIR.iterdir() if d.is_dir()), reverse=True)


def _load_state(run_dir: Path) -> dict | None:
    state_path = run_dir / "run_state.json"
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text())


def main() -> None:
    st.set_page_config(page_title="terra-trial", layout="wide")
    st.title("terra-trial run dashboard")

    runs = _list_runs()
    if not runs:
        st.info("No runs yet. Kick one off with `python main.py`.")
        return

    run_dir = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name)
    state = _load_state(run_dir)
    if state is None:
        st.warning("run_state.json not written yet for this run.")
        return

    st.subheader(f"Status: {state['status']}")
    col1, col2 = st.columns(2)
    col1.markdown(f"**Planet: {state['planet']['name']}**")
    col1.json(state["planet"])
    col2.markdown(f"**Rover: {state['rover_config']['name']}**")
    col2.json(state["rover_config"])

    iterations = state["iterations"]
    is_coevolve = any(it.get("rover_config") for it in iterations)

    if iterations:
        st.subheader("Difficulty trend")
        st.caption("Proof the terrain agent is escalating adversarially, not sampling randomly.")
        trend = {
            "slope_deg": [it["terrain_params"]["slope_deg"] for it in iterations],
            "rock_density": [it["terrain_params"]["rock_density"] for it in iterations],
            "roughness": [it["terrain_params"]["roughness"] for it in iterations],
        }
        st.line_chart(trend)

    if is_coevolve:
        st.subheader("Rover evolution")
        st.caption("The Rover Design Agent's redesign each iteration — the rover's side of the arms race.")
        rover_trend = {
            "max_motor_torque_nm": [it["rover_config"]["max_motor_torque_nm"] for it in iterations],
            "mass_kg": [it["rover_config"]["mass_kg"] for it in iterations],
            "ground_clearance_m": [it["rover_config"]["ground_clearance_m"] for it in iterations],
        }
        st.line_chart(rover_trend)

    st.subheader("Iterations")
    for it in iterations:
        metrics = it["metrics"]
        critique = it["critique"]
        terrain = it["terrain_params"]
        header = f"#{it['iteration']} — outcome={metrics['outcome']} — completion={metrics['completion_pct']:.0f}%"
        with st.expander(header, expanded=(it is iterations[-1])):
            if is_coevolve and it.get("rover_config"):
                st.markdown(f"**Rover this iteration:** {it['rover_config']['name']}")

            st.markdown(f"**Terrain agent's reasoning:** {terrain['reasoning']}")
            st.markdown(f"*Targeting: {terrain['targets_weakness']}*")

            c1, c2 = st.columns(2)
            c1.markdown("**Terrain**")
            c1.json(terrain)
            c2.markdown("**Metrics**")
            c2.json(metrics)

            st.markdown(f"**Critic diagnosis** (confidence: {critique['confidence']})")
            st.write(critique["diagnosis"])
            st.markdown("**Evidence**")
            for e in critique["evidence"]:
                st.markdown(f"- {e}")
            st.markdown(f"**Next target:** {critique['next_target']}")

            if is_coevolve and it.get("redesign"):
                redesign = it["redesign"]
                st.markdown(f"**Rover Design Agent's redesign** (targeting: {redesign['targets_weakness']})")
                st.write(redesign["reasoning"])
                st.json(redesign["rover_config"])

    if state.get("final_report"):
        st.subheader("Final report")
        st.markdown(state["final_report"])


if __name__ == "__main__":
    main()
