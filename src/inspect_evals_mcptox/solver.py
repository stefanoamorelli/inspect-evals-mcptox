"""Solver that reproduces the MCPTox single-turn attack protocol.

MCPTox presents the agent with a system prompt that lists the server's legitimate
tools together with one poisoned tool, and ends with an instruction to reply with a
single JSON tool call. The agent is then given a benign user query and produces one
tool-call decision, which is scored. This solver replays that exact stimulus: it
prepends the per-sample system prompt and generates a single response.
"""

from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import Generate, Solver, TaskState, solver


@solver
def mcptox_solver() -> Solver:
    """Prepend the MCPTox system prompt and generate a single tool-call response."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        system_prompt = state.metadata["system_prompt"]
        state.messages.insert(0, ChatMessageSystem(content=system_prompt))
        return await generate(state)

    return solve
