"""The agent loop — shared by main agent AND every subagent.

The same function runs at every level of the agent tree. What differs
between agents is the RunContext passed in (agent_id, messages, llm,
tools, abort, depth, parent_id, ...).

The loop structure is final — when we add features (compaction, hooks,
stop-hook handling, etc.), they slot into existing extension points
without restructuring.

Idle-wait policy: if the assistant produces no tool_calls AND there are
background children still running, the loop idle-waits for the next
child to finish (via asyncio.wait FIRST_COMPLETED) rather than ending
the turn. The next iteration drains the freshly-arrived notification
and the model sees it. This is how Claude Code keeps its loop alive
when a turn ends with "spawned, will notify" but the results haven't
landed yet.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
)

from .compaction import auto_compact_if_needed, microcompact_messages
from .registry import AGENT_REGISTRY, drain_inbox, list_children
from .tool_executor import run_tools
from .types import AgentId, RunContext


# Sentinel content yielded when the conversation is past the hard context
# limit and could not be compacted — the turn is refused.
PROMPT_TOO_LONG_MESSAGE = (
    "[context limit reached: the conversation exceeds the model's context "
    "window and could not be compacted. Start a new session or remove "
    "context to continue.]"
)

# Prefix on the SystemMessage query() yields when auto-compaction fires.
# Renderers match this to surface a compaction banner.
COMPACTION_MARKER = "[compaction]"


logger = logging.getLogger(__name__)


async def query(
    ctx: RunContext,
) -> AsyncGenerator[BaseMessageChunk | BaseMessage, None]:
    """The agent loop. Runs until the model produces no more tool calls, or
    until abort / max_turns.

    Yields:
        - AIMessageChunk objects as the model streams (for real-time UI)
        - AIMessage when the assistant turn is complete (already in history)
        - ToolMessage objects as each tool result lands
    """
    turn_count = 0

    while True:
        turn_count += 1

        # ─── 1. DRAIN INBOX ────────────────────────────────────────────
        # Notifications from background children land in this agent's inbox.
        # Each becomes a synthetic user message for the next turn AND is
        # yielded so renderers can surface lifecycle events to the user.
        for notif_msg in _drain_notifications_into(ctx.messages, ctx.agent_id):
            yield notif_msg

        # ─── 2. PRE-TURN HOOKS ────────────────────────────────────────
        # Extension point — may modify messages or short-circuit.

        # ─── 3. COMPACTION ────────────────────────────────────────────
        # Keep the conversation under the model's context window. Order
        # matches Claude Code: mechanical microcompaction first (frees
        # tokens cheaply, no LLM), then threshold-triggered auto-compaction
        # (LLM summarization) sees the reduced count. Both gated by CONFIG
        # flags; both mutate ctx.messages in place.
        snip_freed = microcompact_messages(ctx)
        outcome = await auto_compact_if_needed(ctx, snip_freed)
        if outcome.was_compacted:
            # Surface the compaction so the user/renderer can see it fire.
            # A SystemMessage with the COMPACTION_MARKER prefix is the
            # signal; renderers match the prefix (nothing else yields a
            # SystemMessage out of query()).
            yield SystemMessage(content=(
                f"{COMPACTION_MARKER} summarized history "
                f"(~{outcome.token_count} tokens → {len(ctx.messages)} "
                f"messages, threshold {outcome.threshold})"
            ))
        if outcome.blocking_limit:
            # Past the hard limit and couldn't compact — refuse the turn
            # rather than send a request the API will reject.
            yield AIMessage(content=PROMPT_TOO_LONG_MESSAGE)
            return
        messages_for_query = ctx.messages

        # ─── 4. EXIT CHECKS ───────────────────────────────────────────
        if ctx.abort.aborted:
            return
        if turn_count > ctx.max_turns:
            return

        # ─── 5. STREAM THE MODEL ──────────────────────────────────────
        # Bind tools to the model so it can call them. Stream chunks to
        # the caller (for real-time UI) and accumulate them into a
        # complete AIMessage via the + operator on AIMessageChunk.
        llm = ctx.llm.bind_tools(ctx.tools) if ctx.tools else ctx.llm

        accumulated: AIMessageChunk | None = None
        async for chunk in llm.astream(messages_for_query):
            yield chunk
            accumulated = chunk if accumulated is None else accumulated + chunk

        if accumulated is None:
            # Stream produced nothing — shouldn't happen, but fail safe.
            return

        # Convert the accumulated AIMessageChunk into a concrete AIMessage
        # for the conversation history. AIMessage is the canonical type
        # for storage; the chunk type carries streaming metadata we don't
        # need long-term.
        assistant_msg = AIMessage(
            content=accumulated.content,
            tool_calls=list(accumulated.tool_calls),
            usage_metadata=accumulated.usage_metadata,
        )
        ctx.messages.append(assistant_msg)
        yield assistant_msg

        # ─── 6. EXIT IF NO TOOL CALLS ─────────────────────────────────
        if not assistant_msg.tool_calls:
            # Before exiting, check two things that should keep the loop
            # alive:
            #
            #   (a) Notifications already in the inbox. A child may have
            #       finished WHILE the parent's assistant text was
            #       streaming. By the time we reach this point the
            #       child's task is done(), its entry is unregistered —
            #       but its notification is sitting in pending_messages.
            #       We just need to loop back so the top-of-loop drain
            #       picks it up.
            #
            #   (b) Children still running. We need to idle-wait for one
            #       to finish before we can surface anything.
            entry = AGENT_REGISTRY.get(ctx.agent_id)
            if entry and entry.pending_messages:
                continue

            pending = _pending_children_tasks(ctx.agent_id)
            if pending:
                try:
                    await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED,
                    )
                except asyncio.CancelledError:
                    # Forced cancel (e.g. session shutdown). Bail out
                    # without re-entering the loop.
                    return
                # Loop back: drain_inbox at the top of the next iteration
                # surfaces the just-arrived notification(s) to the model.
                continue

            # No tool_calls AND no pending notifications AND no running
            # children — turn is genuinely over.
            #
            # Extension point: STOP HOOKS can re-engage the loop.
            return

        # ─── 7. PRE-TOOL-USE HOOKS ───────────────────────────────────
        # Extension point — may block specific tool calls.

        # ─── 8. DISPATCH TOOLS ────────────────────────────────────────
        # run_tools partitions by is_concurrency_safe metadata, dispatches
        # under a Semaphore(N), and yields ToolMessages in arrival order.
        async for tool_msg in run_tools(assistant_msg.tool_calls, ctx):
            ctx.messages.append(tool_msg)
            yield tool_msg

        # ─── 9. POST-TOOL-USE HOOKS ──────────────────────────────────
        # Extension point — may inspect / modify results.

        # → next iteration


# ─── Internal helpers ────────────────────────────────────────────────────

def _drain_notifications_into(
    messages: list[BaseMessage],
    agent_id: AgentId,
) -> list[HumanMessage]:
    """Drain inbox notifications, append each as a synthetic user message,
    AND return them so the caller (query loop) can yield them out for
    renderers/observers to see.

    The notification body is already an XML <task-notification> block;
    we don't wrap it again.
    """
    drained: list[HumanMessage] = []
    for notif in drain_inbox(agent_id):
        msg = HumanMessage(content=notif)
        messages.append(msg)
        drained.append(msg)
    return drained


def _pending_children_tasks(agent_id: AgentId) -> list[asyncio.Task]:
    """asyncio.Tasks for this agent's currently-registered, still-running
    background children. Used by the idle-wait policy to keep the loop
    alive when the assistant produced no tool_calls but background
    children haven't reported back yet.

    Filters out:
      - entries whose task is None (sync spawns — they don't have a
        long-running asyncio.Task; they're already inside the parent's
        await anyway)
      - tasks that are already done (their notification has either
        landed in the mailbox or will be picked up next iteration)
    """
    return [
        e.task for e in list_children(agent_id)
        if e.task is not None and not e.task.done()
    ]
