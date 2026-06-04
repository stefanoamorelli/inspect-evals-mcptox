"""Dataset loader for the MCPTox tool-poisoning benchmark.

The upstream data file ``response_all.json`` is a nested structure
(``servers`` -> ``malicious_instance`` -> ``datas``) rather than a flat list of
records. This module downloads the pinned file (with checksum verification) and
flattens it into one :class:`~inspect_ai.dataset.Sample` per valid attack instance.
"""

import json
import re
from typing import Any, cast

from inspect_ai.dataset import MemoryDataset, Sample

from inspect_evals_mcptox._assets import (
    CACHE_DIR,
    create_stable_id,
    download_and_verify,
)

# Pinned to a specific commit of the upstream benchmark repo for reproducibility.
DATA_COMMIT = "f85189f9ad12504c197c7f920ab818a40657b1fa"
DATA_URL = (
    "https://raw.githubusercontent.com/zhiqiangwang4/MCPTox-Benchmark/"
    f"{DATA_COMMIT}/response_all.json"
)
# SHA256 of response_all.json at DATA_COMMIT. If the upstream content legitimately
# changes, recompute this value and bump the eval's version.
DATA_SHA256 = "79a90049be931c59e71446d6180b1d7f0d196d123d08a59bc155d142b5041c03"

# The poisoned tool block has the form:
#   "Tool: <name>\nDescription: <description>[\nArguments:\n- ...]"
# Some upstream records use literal "\n" escape sequences instead of real
# newlines, and some append an Arguments schema section after the description.
_TOOL_NAME_RE = re.compile(r"Tool:\s*(?P<name>.+)")
_TOOL_DESC_RE = re.compile(
    r"Description:\s*(?P<description>.*?)(?:\nArguments:|\Z)", re.DOTALL
)


def _parse_poisoned_tool(poisoned_tool: str) -> tuple[str, str]:
    """Split a poisoned tool block into its name and (malicious) description."""
    text = poisoned_tool.replace("\\n", "\n").replace("\\t", "\t").strip()
    name_match = _TOOL_NAME_RE.search(text)
    if name_match is None:
        raise ValueError(f"Could not parse poisoned tool block: {poisoned_tool!r}")
    name = name_match.group("name").strip()
    desc_match = _TOOL_DESC_RE.search(text)
    description = desc_match.group("description").strip() if desc_match else ""
    return name, description


def _download_data() -> dict[str, Any]:
    dest = CACHE_DIR / "response_all.json"
    download_and_verify(DATA_URL, DATA_SHA256, dest)
    with open(dest, encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def _record_to_sample(
    server_name: str, server: dict[str, Any], instance: dict[str, Any], index: int
) -> Sample:
    stimulus = instance["datas"][0]
    poisoned_name, poisoned_description = _parse_poisoned_tool(
        instance["poisoned_tool"]
    )
    instance_metadata = instance.get("metadata", {})
    legitimate_tools = [
        tool for tool in server.get("tool_names", []) if tool != poisoned_name
    ]
    query = stimulus["query"]
    return Sample(
        input=query,
        # The attacker's goal is for the agent to be manipulated ("Success"). The
        # scorer determines via an LLM judge whether that goal was achieved, so the
        # target is informational rather than an exact-match key.
        target="Success",
        # The same (server, poisoned tool, query) triple recurs across distinct
        # attack instances, so the deterministic load index keeps ids unique.
        id=create_stable_id(
            f"{index}|{server_name}|{poisoned_name}|{query}", prefix="mcptox"
        ),
        metadata={
            "server_name": server_name,
            "server_url": server.get("server_url"),
            "poisoned_tool_name": poisoned_name,
            "poisoned_tool_description": poisoned_description,
            "legitimate_tools": legitimate_tools,
            "paradigm": instance_metadata.get("paradigm"),
            "security_risk": instance_metadata.get("security risk"),
            # The full upstream system prompt already embeds the poisoned tool among
            # the legitimate tools and the JSON-only response protocol; it is the
            # exact stimulus used by the benchmark.
            "system_prompt": stimulus["system"],
        },
    )


def load_mcptox_dataset(
    paradigms: list[str] | None = None,
    security_risks: list[str] | None = None,
    servers: list[str] | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> MemoryDataset:
    """Load the MCPTox dataset as a flat list of attack samples.

    Each sample is one *valid* tool-poisoning attack instance: a benign user query
    paired with a server whose tool set contains one poisoned tool. Instances that
    the upstream authors flagged as invalid (``wrong_data``) are skipped, leaving
    1,312 samples.

    Args:
        paradigms: If set, keep only these attack paradigms (e.g. ``["Template-1"]``).
        security_risks: If set, keep only these risk categories (e.g. ``["Privacy Leakage"]``).
        servers: If set, keep only these MCP servers (e.g. ``["FileSystem"]``).
        shuffle: Shuffle the samples.
        seed: Random seed used when ``shuffle`` is set.

    Returns:
        A :class:`~inspect_ai.dataset.MemoryDataset` of attack samples.
    """
    data = _download_data()

    samples: list[Sample] = []
    index = 0
    for server_name, server in data["servers"].items():
        for instance in server["malicious_instance"]:
            if instance.get("wrong_data"):
                continue
            # Assign the deterministic index before applying display filters so a
            # sample's id is independent of which filters are active.
            current_index = index
            index += 1
            instance_metadata = instance.get("metadata", {})
            if servers is not None and server_name not in servers:
                continue
            if (
                paradigms is not None
                and instance_metadata.get("paradigm") not in paradigms
            ):
                continue
            if (
                security_risks is not None
                and instance_metadata.get("security risk") not in security_risks
            ):
                continue
            samples.append(
                _record_to_sample(server_name, server, instance, current_index)
            )

    dataset = MemoryDataset(samples=samples, name="mcptox")
    if shuffle:
        dataset.shuffle(seed=seed)
    return dataset
