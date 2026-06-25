"""Microcompaction — mechanical tool-result clearing, no LLM (Mechanism C).

Ported from the time-based path of Claude Code's microCompact.ts. The
cached-MC and server-native context-management strategies
(clear_tool_uses_20250919) depend on Anthropic's raw context_management
request param, which langchain ChatAnthropic doesn't expose — those are
deliberately omitted (external CC builds omit them too).

What it does: when the last assistant turn is old enough (gap window),
walk the compactable tool calls, keep the most-recent N, and replace the
CONTENT of the older corresponding ToolMessages with a short placeholder.
This frees tokens without an LLM call. It's lossy but cheap, and it runs
BEFORE auto-compaction so the freed tokens lower the count auto-compaction
sees.

Default OFF (CONFIG.microcompact_enabled). Main-agent only — subagents
(depth > 0) are short-lived and skip it.
"""
from __future__ import annotations

import logging
import time

from langchain_core.messages import AIMessage, ToolMessage

from ..config import CONFIG
from ..types import RunContext
from .tokens import rough_token_estimation


logger = logging.getLogger(__name__)


CLEARED_PLACEHOLDER = "[Old tool result content cleared]"
"""The literal replacement content. CC uses this exact string; we guard
against re-clearing by checking for it."""

# Tool names whose results are safe to clear (read-only / reproducible).
# Uses the actual tool .name values registered in the runtime.
COMPACTABLE_TOOLS = frozenset({
    "Read", "Bash", "Grep", "Glob", "Edit", "Write",
    "BashOutput", "TaskOutput",
})


def microcompact_messages(ctx: RunContext) -> int:
    """Clear stale compactable tool results in place. Returns the
    estimated number of tokens freed (0 if nothing changed / disabled).
    """
    if not CONFIG.microcompact_enabled:
        return 0
    if ctx.depth > 0:
        # Subagents are short-lived; microcompaction targets the long
        # main session.
        return 0

    messages = ctx.messages

    # Gap gate: only fire when the last assistant turn is old enough. We
    # approximate "last assistant timestamp" with a per-ctx marker since
    # LangChain messages don't carry wall-clock time.
    now = time.time()
    last_ts = ctx.compaction_state.get("last_microcompact_ts")
    if last_ts is not None:
        gap_seconds = CONFIG.microcompact_gap_minutes * 60
        if now - last_ts < gap_seconds:
            return 0

    # Collect compactable tool_call ids in conversation order.
    compactable_ids: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") in COMPACTABLE_TOOLS:
                    compactable_ids.append(tc["id"])

    if len(compactable_ids) <= CONFIG.microcompact_keep_recent:
        return 0

    # Keep the most-recent N; everything before is eligible to clear.
    keep = max(1, CONFIG.microcompact_keep_recent)
    clearable = set(compactable_ids[:-keep])

    tokens_freed = 0
    cleared = 0
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.tool_call_id not in clearable:
            continue
        if msg.content == CLEARED_PLACEHOLDER:
            continue  # already cleared — don't double-count
        tokens_freed += rough_token_estimation(msg)
        msg.content = CLEARED_PLACEHOLDER
        cleared += 1

    if cleared:
        ctx.compaction_state["last_microcompact_ts"] = now
        logger.info(
            "microcompaction: cleared %d stale tool results (~%d tokens freed)",
            cleared, tokens_freed,
        )
    return tokens_freed
