"""MCPTox: a benchmark for Tool Poisoning Attacks on real-world MCP servers.

Paper: https://arxiv.org/abs/2508.14925 (AAAI 2026)
Upstream data: https://github.com/zhiqiangwang4/MCPTox-Benchmark

Tool Poisoning hides a malicious instruction inside a tool's *description* (the
metadata an agent reads when planning), rather than in any executed code. Each
sample presents an agent with a benign user query and an MCP server whose tool set
contains one such poisoned tool; the eval measures how often the agent is
manipulated into carrying out the malicious action (the Attack Success Rate).
"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, Model

from inspect_evals_mcptox.dataset import load_mcptox_dataset
from inspect_evals_mcptox.scorer import mcptox_scorer
from inspect_evals_mcptox.solver import mcptox_solver

DEFAULT_MAX_TOKENS = 2048


@task
def mcptox(
    judge_model: str | Model | None = None,
    paradigms: list[str] | None = None,
    security_risks: list[str] | None = None,
    servers: list[str] | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> Task:
    """Tool Poisoning Attack benchmark on real-world MCP servers.

    Args:
        judge_model: Model used to grade outcomes. If ``None``, the ``grader``
            model role is used (falling back to the default model).
        paradigms: Optionally restrict to specific attack paradigms, e.g.
            ``["Template-1"]`` (explicit trigger), ``["Template-2"]`` (implicit
            trigger function hijack), ``["Template-3"]`` (parameter tampering).
        security_risks: Optionally restrict to specific risk categories, e.g.
            ``["Credential Leakage", "Privacy Leakage"]``.
        servers: Optionally restrict to specific MCP servers, e.g. ``["FileSystem"]``.
        shuffle: Shuffle the dataset.
        seed: Random seed used when ``shuffle`` is set.

    Returns:
        An Inspect :class:`~inspect_ai.Task`.
    """
    dataset = load_mcptox_dataset(
        paradigms=paradigms,
        security_risks=security_risks,
        servers=servers,
        shuffle=shuffle,
        seed=seed,
    )

    return Task(
        dataset=dataset,
        solver=mcptox_solver(),
        scorer=mcptox_scorer(judge_model),
        config=GenerateConfig(max_tokens=DEFAULT_MAX_TOKENS),
        version=1,
    )
