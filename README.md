# terra-trial

Agentic rover trial-by-fire: a Terrain Agent proposes progressively harder
terrain, a physics sim runs the rover on it, a Critic Agent diagnoses why it
broke, and the loop repeats until the rover fails hard or the iteration
budget runs out. A Report Agent then writes up the rover's performance
envelope. Built for the Dell x NVIDIA Hackathon (Aug 22 2026) — target
hardware is a Dell Pro Max with GB10 (ARM64, 128GB unified memory), running
OpenClaw + NVIDIA NemoClaw + OpenShell for local inference.

## Running it

```bash
pip install -e .
export ANTHROPIC_API_KEY=...
python main.py --planet mars --rover scout --iterations 4
```

Or, with a local NemoClaw sandbox already running (`nemoclaw status` to
check):

```bash
python main.py --planet mars --rover scout --iterations 4 --model-backend nemoclaw
```

`--backend {mujoco,pybullet,isaac}` picks the sim (default `mujoco`, see
below); `--model-backend {anthropic,nemoclaw}` picks the LLM backend
(default `anthropic`). `--gui` opens a live viewer, paced to real time so
it's actually watchable (an easy trial otherwise completes in well under a
second of wall-clock compute).

**On macOS with the mujoco backend, `--gui` needs `mjpython`, not
`python`** — mujoco's passive viewer requires it (a Cocoa main-thread GUI
restriction) and raises a clear `RuntimeError` naming this if you forget:

```bash
mjpython main.py --planet mars --rover scout --iterations 1 --gui
```

This only affects the viewer; without `--gui`, plain `python` is fine on
any platform. Runs are written to `runs/<run_id>/run_state.json`; tail them
live with:

```bash
streamlit run dashboard/app.py
```

## Architecture

Two contracts everything else is built around (frozen in
`contracts/schemas.py` and `sim/interface.py` / `agents/model_client.py`):

- **`SimInterface.run_trial(rover_config, planet, terrain_params) ->
  SimMetrics`** — implemented by `sim/mujoco_sim.py` (primary),
  `sim/pybullet_sim.py` (alternate, see below), `sim/fake_sim.py` (canned
  metrics, for testing the agent loop without physics), and
  `sim/isaac_sim.py` (unimplemented stretch backend). Wheel layout, chassis
  sizing, mass split, and drive-speed targets are engine-agnostic and live
  in `sim/geometry.py` / `sim/terrain_gen.py`, shared by every backend, so
  swapping backends can't silently change what rover or terrain is actually
  being simulated.
- **`ModelClient.complete(system_prompt, user_prompt, response_schema) ->
  dict`** (plus a raw `generate()` for the freeform Report Agent) —
  implemented by `agents/model_client.py`'s `AnthropicClient` (cloud) and
  `NemoClawModelClient` (local sandbox, see below). Agents and the
  orchestrator never know which is live; swap with `--model-backend`.

`orchestrator.py` is pure control flow — a fixed `for` loop over
terrain → sim → critique, no LLM calls of its own. The agentic claim rests on
the Terrain ↔ Critic feedback loop, which is genuinely autonomous after
"Run". Terrain params coming back from the Terrain Agent are always clamped,
never rejected, via `clamp_terrain_params()` in `contracts/schemas.py` — a
malformed or out-of-range agent output keeps the demo running instead of
stalling it.

### Swapping sim backends (plug and play)

`main.py`'s `SIM_BACKENDS` is a `name -> (gui) -> SimInterface` registry;
nothing else in the codebase imports a concrete backend by name. Each
factory imports its backend lazily, so one backend's broken/missing
dependency can't break the others. To drop in the real Isaac Sim
implementation tomorrow: implement `sim/isaac_sim.py`'s `run_trial()`
against the existing `SimInterface` ABC (geometry/terrain helpers are
already reusable from `sim/geometry.py` and `sim/terrain_gen.py`), then run
with `--backend isaac`. No other file changes.

### Why MuJoCo, not PyBullet

The build plan's original sim choice was PyBullet. On this dev machine,
`pip install pybullet` fails to build from source: its vendored 1990s-era
zlib defines `#define fdopen(fd, mode) NULL` for an old macOS guard, which
collides with the real `fdopen` declaration in the current macOS SDK's
`_stdio.h` and breaks the parse. `pip install mujoco` installs a working
prebuilt wheel with strong ARM support instead, so it's the default
(`--backend mujoco`). `sim/pybullet_sim.py` is kept working and behind the
same `SimInterface` — install the `pybullet` extra
(`pip install -e ".[pybullet]"`) and pass `--backend pybullet` if it builds
cleanly on a given machine (worth trying fresh on the GB10; this is a
macOS-SDK-specific clash, not guaranteed to reproduce on Linux).

Both backends drive wheels with velocity control, target-speed-capped by
`max_motor_torque_nm` (`sim/geometry.py`'s `target_wheel_angular_velocity`),
not open-loop max torque — open-loop full torque slams the wheel's reaction
torque into the chassis every trial and reliably wheelies/flips it
regardless of terrain, which isn't what the terrain difficulty is supposed
to be testing.

### NemoClaw integration

`NemoClawModelClient` in `agents/model_client.py` is real, verified against
a NemoClaw sandbox installed locally (not yet the GB10 — re-verify there
tomorrow, most of this should transfer since it goes through the CLI, not
anything host-specific). Two things worth knowing before touching it:

- **No Python SDK or reachable HTTP endpoint.** NemoClaw doesn't expose
  either — the sandbox's own OpenAI-compatible route only resolves inside
  the sandbox's network namespace, not from the host. Calls go through
  `nemoclaw <sandbox> agent -m ... --json` as a subprocess (`openshell
  sandbox exec` under the hood) — the CLI *is* the API here.
- **Sessions aren't stateless by default.** Reusing an agent id without a
  fresh `--session-id` continues that session's conversation history
  (confirmed empirically — message count grew across calls). Every
  `generate()` call uses a new random session id to satisfy the "no
  conversation state" part of Contract B, at the cost of losing prompt
  caching across calls.
- **Every call carries real overhead.** The sandbox's OpenClaw agent
  injects its own bootstrap system prompt (AGENTS.md/SOUL.md/tool
  schemas) on top of ours — on the tested sandbox that's ~15K tokens,
  regardless of what we actually ask. Budget for this against a local
  Nano-tier model tomorrow; it's the dominant latency cost, not our prompts.
- **The output token budget is fixed and not overridable per call** (no
  `--max-tokens` flag on the CLI). Nemotron-3-Super-120B (the model behind
  the currently-configured `nvidia-prod` route) opens responses with a
  chain-of-thought preamble even with `thinking: off`, and on a first pass
  this **burned through the entire 4096-token budget mid-reasoning and
  never reached the JSON** — a real failure (`error.kind=incomplete_turn`),
  not a hypothetical one. Fixed by (a) all three prompts now explicitly
  forbid any text before the JSON, and (b) `ModelClient.complete()` now
  retries once on `generate()` raising, not just on unparseable output.
  Re-verify this against whatever model you actually run tomorrow — a
  Nano-tier model may behave differently (better or worse).

Verified end-to-end: a full multi-iteration run (`--model-backend nemoclaw
--backend mujoco`) completed with schema-valid Terrain/Critic output citing
real metrics as evidence, and a coherent final report.

## Layout

```
contracts/    schemas.py (dataclasses + clamp/validate), planet_profiles.json
sim/          interface.py (ABC), geometry.py (shared rover math), terrain_gen.py (shared terrain),
              mujoco_sim.py, pybullet_sim.py, fake_sim.py, isaac_sim.py (stretch), vehicle.py (pybullet body builder)
agents/       model_client.py, terrain_agent.py, critic_agent.py, report_agent.py, prompts/
orchestrator.py   the loop
dashboard/    Streamlit UI, reads run_state.json only
main.py       entry point, rover presets, sim backend registry
runs/         run_state.json + model_calls.jsonl per run
```

## Status

- MuJoCo backend, agent loop, and dashboard are wired end to end against the
  frozen contracts, and verified with a real physics run (not just
  `FakeSim`) across a slope sweep and a full multi-iteration campaign.
- PyBullet backend is implemented and behind the same interface, but
  unverified on this machine (see "Why MuJoCo, not PyBullet" above) — try it
  on the GB10.
- Isaac Sim backend is a stub (`NotImplementedError`) — stretch goal, see
  the build plan section 8 for the spike rules.
- NemoClaw local inference (`NemoClawModelClient`) is implemented and
  verified end-to-end against a locally-installed NemoClaw sandbox (see
  "NemoClaw integration" above) — re-verify against the GB10's install
  tomorrow, since the model actually behind the route will likely change
  from the 120B cloud-routed default to a local Nano-tier model.
