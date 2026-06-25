"""Core data shapes.

We use LangChain's message types directly (HumanMessage, AIMessage,
ToolMessage, SystemMessage from langchain_core.messages) — no wrapper.
Same for the chat model (BaseChatModel) and tools (BaseTool).

This file is intentionally small: one bundle of runtime state
(`RunContext`) plus a couple of NewType IDs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NewType, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from .config import CONFIG

if TYPE_CHECKING:
    from .cancellation import AbortController
    from .permissions import AskCallback, PermissionMode


# ── IDs ──────────────────────────────────────────────────────────────────
# NewType makes these distinct types for static checkers (mypy/pyright catch
# accidental mixing) but plain strs at runtime (zero overhead).
AgentId = NewType("AgentId", str)
ToolUseId = NewType("ToolUseId", str)


# ── RunContext — the one bundle of runtime state ────────────────────────

@dataclass(slots=True)
class RunContext:
    """Everything one agent needs to execute its loop.

    Same object is passed to query(), threaded down to tools via the
    CURRENT_RUN_CONTEXT ContextVar (set by tool_executor before each
    tool.ainvoke), and used by SpawnAgentTool when spawning a child agent
    (to read parent_id, abort, llm, depth, etc.).

    Matches the convention of modern Python agent frameworks
    (Pydantic AI's RunContext, OpenAI Agents SDK's RunContext): one
    bundle for everything, rather than separate per-scope objects.
    """
    agent_id: AgentId
    messages: list[BaseMessage]
    """The conversation history. Mutated across loop iterations as
    assistant messages and tool results are appended."""

    abort: AbortController
    """Cooperative cancellation flag. Checked at the top of each loop
    iteration and before each tool dispatch."""

    tools: list[BaseTool]
    """Tools available to this agent."""

    llm: BaseChatModel
    """The chat model this agent uses. Subagents may use a different
    one if their SubagentDefinition.model resolves to a different
    registered profile."""

    max_turns: int = field(default_factory=lambda: CONFIG.max_turns_default)
    """Safety cap on iterations of the query() while loop."""

    depth: int = 0
    """Recursion depth in the agent tree. 0 = main session; capped by
    SpawnAgentTool against CONFIG.max_agent_depth."""

    parent_id: AgentId | None = None
    """The agent that spawned this one (None for the main session)."""

    file_cache: dict[str, Any] = field(default_factory=dict)
    """Per-run cache shared between tools (e.g., a file's last-read
    mtime). Empty by default."""

    compaction_state: dict[str, Any] = field(default_factory=dict)
    """Per-agent compaction bookkeeping (auto-compaction circuit breaker
    + recompaction tracking). Keys: 'compacted' (bool), 'turn_id' (str),
    'turn_counter' (int), 'consecutive_failures' (int). Empty = never
    compacted. Only the main agent populates this in practice — subagents
    are short-lived and rarely cross the threshold."""

    permission_mode: "PermissionMode" = "default"
    """How tool calls are gated: default / acceptEdits / plan /
    bypassPermissions. See agent_runtime/permissions.py."""

    ask_callback: "AskCallback | None" = None
    """Optional async (tool_name, args, reason) -> bool used to resolve an
    'ask' permission decision (e.g. the CLI's y/n prompt). None = gated
    tools auto-allow in default mode (library-friendly default)."""
