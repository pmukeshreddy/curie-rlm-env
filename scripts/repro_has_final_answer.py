"""Reproduce the has_final_answer state-propagation bug in isolation.

No GPU, no Docker, no model, no orchestrator. Drives the framework's tool-
dispatch + stop-predicate chain by hand against a tiny StatefulToolEnv
subclass that mirrors how CurieRLMEnv wires submit_answer.

If the bug reproduces here, we can step through with prints / a debugger.
If it does NOT reproduce, the issue is at a higher layer (RLMEnv overrides,
env_worker IPC, concurrent rollouts).

Run with:
    uv run python scripts/repro_has_final_answer.py
"""
from __future__ import annotations

import asyncio
import json

from datasets import Dataset

import verifiers as vf
from verifiers.envs.stateful_tool_env import StatefulToolEnv
from verifiers.types import Messages, State


# Minimal fake assistant message with a `submit_answer` tool call. Shape matches
# what the framework's env_response loop pulls out of last_msg.tool_calls.
class _FakeToolCall:
    id = "tc_repro_1"
    name = "submit_answer"
    arguments = json.dumps({"content": "the canonical answer"})


class _FakeAssistantMsg:
    role = "assistant"
    content = ""
    tool_calls = [_FakeToolCall()]


class MiniEnv(StatefulToolEnv):
    """Minimum repro env: one tool, one stop predicate, both touch state."""

    def __init__(self):
        # Empty placeholder dataset — Environment.__init__ refuses None.
        ds = Dataset.from_list([{"prompt": "ignored", "answer": "ignored"}])
        super().__init__(tools=[], dataset=ds)
        self.probes: list[tuple] = []
        self.add_tool(self.submit_answer, args_to_skip=["state"])

    def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs):
        if tool_name == "submit_answer":
            self.probes.append(("update_tool_args", id(state), "final_answer" in state))
            return {**tool_args, "state": state}
        return tool_args

    async def submit_answer(self, content: str, state: State) -> str:
        self.probes.append(("submit_answer_pre", id(state), "final_answer" in state))
        state["final_answer"] = content
        self.probes.append(("submit_answer_post", id(state), "final_answer" in state))
        return "ok"

    @vf.stop
    async def check_done(self, state: State) -> bool:
        present = "final_answer" in state
        self.probes.append(("check_done", id(state), present))
        return present


async def main() -> None:
    env = MiniEnv()
    state: State = State()
    state["trajectory"] = []

    messages: Messages = [_FakeAssistantMsg()]  # type: ignore[list-item]

    print(f"BEFORE env_response: id(state)={id(state)} keys={list(state.keys())}")
    tool_messages = await env.env_response(messages, state)
    print(f"AFTER env_response:  id(state)={id(state)} final_answer in state? {'final_answer' in state}")
    if "final_answer" in state:
        print(f"  final_answer = {state['final_answer']!r}")

    done = await env.is_completed(state)
    print(f"is_completed → {done}")
    print(f"tool_messages → {tool_messages}")

    print("\nPROBES (where, id(state), final_answer_in_state):")
    for p in env.probes:
        print(f"  {p}")

    print("\n--- VERDICT ---")
    ids = {p[1] for p in env.probes}
    if len(ids) == 1 and "final_answer" in state and done:
        print("HEALTHY: state shared, mutation propagated, stop fired.")
        print("→ Bug does NOT reproduce at the StatefulToolEnv layer.")
        print("→ Widen the repro (try RLMEnv subclass, then concurrent rollouts).")
    elif len(ids) > 1:
        print(f"STATE COPY DETECTED: {len(ids)} distinct id(state) values.")
        print("→ Framework is copying state somewhere in the dispatch chain.")
        print("→ Bug reproduced locally. Trace which probe got a new id().")
    elif "final_answer" not in state or not done:
        print("STATE SHARED but MUTATION LOST.")
        print("→ State is the same object everywhere but the write disappeared.")
        print("→ Look for custom __setitem__, dict-subclass weirdness, or stale read.")


if __name__ == "__main__":
    asyncio.run(main())
