"""Contract B: complete(system_prompt, user_prompt, response_schema) -> dict.

Agents depend only on the ModelClient ABC, never on a concrete adapter, so
the cloud client tonight and the NemoClaw-backed local client tomorrow are a
one-line swap in main.py.

The ABC owns JSON extraction, the one-retry-on-malformed-output policy, and
request/response logging; concrete adapters only implement raw text
generation via generate(). complete() is for the Terrain/Critic agents,
which need structured JSON; generate() is exposed directly for the Report
agent, which returns freeform markdown.
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_DEFAULT_LOG_PATH = Path(__file__).parent.parent / "runs" / "model_calls.jsonl"


def _extract_json(raw: str) -> dict | None:
    """Pull a JSON object out of text that may have markdown fences or prose
    wrapped around it. Returns None if nothing parses."""
    candidates = [raw.strip()]

    fence_match = _JSON_FENCE_RE.search(raw)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class ModelClient(ABC):
    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path = log_path or _DEFAULT_LOG_PATH

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's raw text response for one turn. No conversation state."""

    def complete(self, system_prompt: str, user_prompt: str, response_schema: dict) -> dict:
        # The retry covers two distinct failure modes with one repair prompt:
        # generate() raising (a transport-level failure — timeout, an
        # abandoned turn, a truncated response that never reached the JSON)
        # and generate() succeeding but returning unparseable text. Only the
        # first attempt is caught; a second failure propagates rather than
        # looping.
        raw = "<generate() raised on first attempt>"
        data = None
        try:
            raw = self.generate(system_prompt, user_prompt)
            data = _extract_json(raw)
        except Exception as e:
            raw = f"<generate() raised: {e}>"

        if data is None:
            repair_prompt = (
                f"{user_prompt}\n\n"
                "Your previous response could not be parsed as JSON matching this shape:\n"
                f"{json.dumps(response_schema, indent=2)}\n\n"
                "Respond with ONLY valid JSON matching that shape. No prose, no markdown fences, "
                "no reasoning or explanation before or after it."
            )
            raw = self.generate(system_prompt, repair_prompt)
            data = _extract_json(raw)

        self._log(system_prompt, user_prompt, raw, data)

        if data is None:
            raise ValueError(f"model did not return valid JSON after one retry: {raw!r}")
        return data

    def _log(self, system_prompt: str, user_prompt: str, raw_response: str, parsed: dict | None) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.time(),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parsed_ok": parsed is not None,
        }
        with self._log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")


class AnthropicClient(ModelClient):
    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None, log_path: Path | None = None) -> None:
        super().__init__(log_path=log_path)
        import anthropic

        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class NemoClawModelClient(ModelClient):
    """Inference via a NemoClaw/OpenClaw sandbox, driven through the
    `nemoclaw <sandbox> agent` CLI as a subprocess — NemoClaw doesn't expose
    a Python SDK or a host-reachable HTTP endpoint (the sandbox's own
    OpenAI-compatible route, e.g. https://inference.local/v1, only resolves
    *inside* the sandbox's network namespace; the CLI's `agent` subcommand
    is the sanctioned host-side path, going through `openshell sandbox
    exec`). Verified end-to-end against a real local NemoClaw install.

    Two things this CLI doesn't give us that ModelClient assumes:

    - No separate system/user roles — `-m` is one free-text turn, and the
      sandbox's OpenClaw agent already injects its own large bootstrap
      system prompt (AGENTS.md/SOUL.md/tool schemas, ~15K tokens on the
      sandbox tested against) that we can't replace. We concatenate our
      system_prompt + user_prompt into that one turn; the model just also
      carries OpenClaw's own assistant persona as extra (mostly ignorable)
      context. Expect real per-call latency/token overhead from this,
      especially against a local Nano-tier model — budget for it.
    - Calls are NOT stateless by default: reusing an agent id without a
      fresh --session-id continues that session's conversation history
      (confirmed: message count grew across repeated calls with the same
      id). We generate a new session id per call to satisfy generate()'s
      "no conversation state" contract — at the cost of losing prompt
      caching across calls, which is the tradeoff for correctness here.

    `model` is accepted for interface parity with AnthropicClient but not
    acted on: switching the active model is a sandbox-wide operation
    (`nemoclaw inference set --provider ... --model ...`) that also needs a
    provider, not something safe to trigger implicitly from a constructor.
    Run that yourself before constructing this client if you want a
    different model active.
    """

    def __init__(
        self,
        sandbox: str | None = None,
        agent_id: str = "main",
        model: str | None = None,
        log_path: Path | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        super().__init__(log_path=log_path)
        self._sandbox = sandbox or _default_nemoclaw_sandbox()
        self._agent_id = agent_id
        self._timeout_s = timeout_s
        self._model = model  # informational only, see docstring

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import subprocess
        import uuid

        session_id = str(uuid.uuid4())
        message = f"{system_prompt}\n\n{user_prompt}"
        result = subprocess.run(
            [
                "nemoclaw",
                self._sandbox,
                "agent",
                "-m",
                message,
                "--agent",
                self._agent_id,
                "--session-id",
                session_id,
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=self._timeout_s,
        )
        if result.returncode != 0:
            raise RuntimeError(f"nemoclaw agent call failed (exit {result.returncode}): {result.stderr.strip()}")

        try:
            payload = json.loads(result.stdout)
            return payload["result"]["payloads"][0]["text"]
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise RuntimeError(f"unexpected nemoclaw agent response shape: {e}\nstdout: {result.stdout[:500]!r}") from e


def _default_nemoclaw_sandbox() -> str:
    import subprocess

    result = subprocess.run(["nemoclaw", "list", "--json"], capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise RuntimeError(f"nemoclaw list failed (exit {result.returncode}): {result.stderr.strip()}")
    data = json.loads(result.stdout)
    default = data.get("defaultSandbox")
    if not default:
        raise RuntimeError("no default NemoClaw sandbox found — pass sandbox= explicitly")
    return default
