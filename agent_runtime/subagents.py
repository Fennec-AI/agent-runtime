"""SubagentDefinition + SUBAGENT_REGISTRY — what kinds of subagents exist.

Each subagent type (e.g., "Explore", "Plan", "general-purpose") is a
definition that SpawnAgentTool can spawn. Users register their own subagent
types via `register_subagent()`.

A SubagentDefinition carries:
  - name             — what the LLM passes as `subagent_type`
  - description      — shown to the LLM in SpawnAgentTool's description
  - system_prompt    — the subagent's own system prompt
  - tools            — optional explicit whitelist of BaseTool instances
  - disallowed_tools — optional blacklist of tool names to strip from
                       inherited tools (e.g., {"Edit", "Write"} for read-only)

Subagents inherit the parent's LLM and tool list by default. Tool overrides:
  - tools is not None        → use exactly this list
  - tools is None            → inherit parent's tools, then drop:
        SpawnAgentTool (always — prevents trivial recursion)
        anything in disallowed_tools (matches Claude Code's pattern,
        see built-in/exploreAgent.ts:67)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.tools import BaseTool


@dataclass(slots=True)
class SubagentDefinition:
    """One kind of subagent that SpawnAgentTool can spawn."""
    name: str
    """The string the LLM uses to request this subagent type."""

    description: str
    """One-line description shown to the parent LLM, explaining when to
    spawn this kind of subagent."""

    system_prompt: str
    """The subagent's own system prompt — sets its persona and constraints."""

    tools: list[BaseTool] | None = field(default=None)
    """Explicit tool whitelist. If None, the subagent inherits the parent's
    tools (with SpawnAgentTool and `disallowed_tools` stripped)."""

    disallowed_tools: set[str] | None = field(default=None)
    """Tool names to strip when inheriting parent's tools. Mirrors Claude
    Code's `disallowedTools` on read-only agents (e.g., Explore, Plan)
    which want everything EXCEPT Edit/Write. Ignored when `tools` is set."""

    model: str | None = field(default=None)
    """Model profile name (see agent_runtime.models). If None, subagent
    inherits the parent's LLM."""


# ── Module-level registry ───────────────────────────────────────────────
SUBAGENT_REGISTRY: dict[str, SubagentDefinition] = {}


def register_subagent(definition: SubagentDefinition) -> None:
    """Register a subagent type. Re-registering an existing name overwrites."""
    SUBAGENT_REGISTRY[definition.name] = definition


def get_subagent(name: str) -> SubagentDefinition | None:
    """Look up a registered subagent by name."""
    return SUBAGENT_REGISTRY.get(name)


def list_subagent_types() -> list[str]:
    """Names of all registered subagents."""
    return list(SUBAGENT_REGISTRY.keys())
