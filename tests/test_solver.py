"""Tests for the MCPTox solver (single-turn attack protocol replay)."""

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ModelName, ModelOutput
from inspect_ai.solver import TaskState

from inspect_evals_mcptox.solver import mcptox_solver


def _make_state() -> TaskState:
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id="s1",
        epoch=1,
        input="Read '/tmp/a.txt'.",
        messages=[ChatMessageUser(content="Read '/tmp/a.txt'.")],
        metadata={"system_prompt": "You have access to these tools: ... qubit ..."},
    )


@pytest.mark.asyncio
async def test_solver_prepends_system_prompt_and_generates_once():
    state = _make_state()
    calls = 0

    async def fake_generate(s: TaskState) -> TaskState:
        nonlocal calls
        calls += 1
        s.output = ModelOutput.from_content("mockllm/model", "ok")
        return s

    solver = mcptox_solver()
    result = await solver(state, fake_generate)

    assert calls == 1
    assert isinstance(result.messages[0], ChatMessageSystem)
    assert result.messages[0].text == "You have access to these tools: ... qubit ..."
    # The original user query is preserved after the system message.
    assert isinstance(result.messages[1], ChatMessageUser)
    assert result.messages[1].text == "Read '/tmp/a.txt'."
