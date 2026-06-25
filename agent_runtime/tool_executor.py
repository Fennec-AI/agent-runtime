"""Tool dispatcher — runs the tool calls emitted by the assistant.

Matches Claude Code's `toolOrchestration.ts` algorithm:

    1. Walk through tool_calls IN DECLARED ORDER.
    2. Group CONSECUTIVE concurrency-safe tools into one batch; each
       unsafe tool gets its own size-1 batch.
    3. Run batches sequentially. Within a multi-item safe batch, run
       all items in parallel under a Semaphore (yielding as each lands).

This preserves the model's declared order while still extracting
parallelism from consecutive safe runs. A read interleaved with mutating
writes stays in its declared position rather than being pulled forward.

Concurrency safety is opt-in via tool metadata:

    @tool
    def my_safe_tool(...): ...
    my_safe_tool.metadata = {"concurrency_safe": True}

Default is UNSAFE (serial). Read-only tools should opt in; mutating
tools (file writes, shell commands) stay serial.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool

from . import hooks
from .config import CONFIG
from .permissions import evaluate_permission
from .types import RunContext


logger = logging.getLogger(__name__)


# ── Runtime context for tools that need it (SpawnAgentTool) ──────────────────
# SpawnAgentTool can't take ctx as a normal kwarg because LangChain's BaseTool
# enforces a fixed schema. Instead, the dispatcher sets this ContextVar
# just before invoking each tool; tools that need ctx (SpawnAgentTool) read it
# via CURRENT_RUN_CONTEXT.get().
#
# ContextVars are async-safe: each asyncio.Task inherits its parent's
# context at creation, and changes within a task don't leak to siblings.
# So when the dispatcher fans out N parallel tasks, each has its own
# CURRENT_RUN_CONTEXT view.
CURRENT_RUN_CONTEXT: contextvars.ContextVar[RunContext | None] = (
    contextvars.ContextVar("CURRENT_RUN_CONTEXT", default=None)
)

# Per-tool-call identifier. SpawnAgentTool reads this when launching a
# background subagent so the eventual completion notification can carry a
# `<tool-use-id>` field — letting the model correlate "this notification"
# with "the spawn call I made N turns ago." Mirrors Claude Code's
# notification format (LocalAgentTask.tsx:252).
#
# Genuinely per-call (a different value for every tool invocation), unlike
# CURRENT_RUN_CONTEXT which is the same across calls in one query loop.
CURRENT_TOOL_USE_ID: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("CURRENT_TOOL_USE_ID", default=None)
)

# Optional observer for subagent events. When set, the runtime invokes
# this callable for every event a subagent's query() loop yields — its
# streaming text chunks, its tool calls, its tool results. Lets the CLI
# (or any other UI layer) surface subagent activity LIVE, instead of
# only seeing the final result. Signature:
#
#     sink(ctx: RunContext, event: BaseMessage) -> None
#
# Default None = no observation (zero overhead). The CLI installs a
# printer at startup that formats events to stdout with a
# "[<subagent_type> <short-id>] " prefix.
#
# Set ONCE at process / session startup; survives across all tool calls
# and spawned tasks (asyncio inherits the ContextVar at task creation).
CURRENT_SUBAGENT_EVENT_SINK: contextvars.ContextVar[Any] = (
    contextvars.ContextVar("CURRENT_SUBAGENT_EVENT_SINK", default=None)
)


@dataclass(slots=True)
class _Batch:
    """A consecutive run of same-safety tool calls.

    safe=True batches may contain many items (run in parallel).
    safe=False batches always contain exactly one item (run alone).
    """
    safe: bool
    items: list[tuple[ToolCall, BaseTool]] = field(default_factory=list)


async def run_tools(
    tool_calls: list[ToolCall],
    ctx: RunContext,
) -> AsyncGenerator[ToolMessage, None]:
    """Dispatch the assistant's tool calls in declared-order batches.

    Order:
      - Batches run sequentially in declared order.
      - Within a multi-item safe batch, results yield in arrival order
        (whichever finishes first), since safe tools have no inter-
        ordering constraints.
    """
    tools_by_name = {t.name: t for t in ctx.tools}
    batches: list[_Batch] = []

    # 1. Resolve names + group consecutive same-safety calls into batches.
    for tc in tool_calls:
        tool = tools_by_name.get(tc["name"])
        if tool is None:
            logger.warning("unknown tool name: %s", tc["name"])
            yield ToolMessage(
                content=f"Tool {tc['name']!r} not found in this agent's tools.",
                tool_call_id=tc["id"],
                status="error",
            )
            continue

        is_safe = _is_concurrency_safe(tool)
        # Merge into the previous batch only if BOTH this and the previous
        # are safe — unsafe tools always get their own size-1 batch.
        if is_safe and batches and batches[-1].safe:
            batches[-1].items.append((tc, tool))
        else:
            batches.append(_Batch(safe=is_safe, items=[(tc, tool)]))

    logger.debug(
        "dispatching %d batches: %s",
        len(batches),
        [(b.safe, len(b.items)) for b in batches],
    )

    # 2. Run each batch in order.
    for batch in batches:
        if batch.safe and len(batch.items) > 1:
            # Parallel batch — fan out under a semaphore, yield as completed.
            sem = asyncio.Semaphore(CONFIG.max_concurrent_tools)

            async def _capped(tc: ToolCall, tool: BaseTool) -> ToolMessage:
                async with sem:
                    return await _execute_tool(tc, tool, ctx)

            pending = {
                asyncio.create_task(_capped(tc, tool)) for tc, tool in batch.items
            }
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    yield task.result()
        else:
            # Single item (either safe alone or unsafe) — just run it.
            for tc, tool in batch.items:
                yield await _execute_tool(tc, tool, ctx)


# ── Helpers ─────────────────────────────────────────────────────────────

async def _execute_tool(
    tc: ToolCall,
    tool: BaseTool,
    ctx: RunContext,
) -> ToolMessage:
    """Invoke a single tool. Always returns a ToolMessage — exceptions
    become error ToolMessages so the next API call sees them as data,
    not crashes.

    Fires hook events around the call:
      pre_tool   — before tool.ainvoke (NOT fired on abort-pre-check)
      post_tool  — always, with status + duration_ms
      on_error   — when tool.ainvoke RAISES (before post_tool)
    """
    if ctx.abort.aborted:
        return ToolMessage(
            content="Tool was aborted before execution.",
            tool_call_id=tc["id"],
            status="error",
        )

    args = tc.get("args", {})
    logger.debug("invoking tool %s(%s)", tool.name, args)

    # ── PERMISSION GATE ───────────────────────────────────────────────
    # Resolve allow/deny (an 'ask' is resolved inside, via ctx.ask_callback).
    # A denial returns an error ToolMessage WITHOUT running the tool — the
    # model sees the denial as data and can adapt.
    decision = await evaluate_permission(
        tool, args, ctx.permission_mode, ctx.ask_callback,
    )
    if decision.behavior == "deny":
        logger.info("permission denied: %s — %s", tool.name, decision.message)
        await hooks.fire(
            "permission_denied",
            tool_name=tool.name,
            args=args,
            tool_call_id=tc["id"],
            agent_id=ctx.agent_id,
            message=decision.message,
        )
        return ToolMessage(
            content=f"Permission denied: {decision.message}",
            tool_call_id=tc["id"],
            status="error",
        )
    if decision.updated_input is not None:
        args = decision.updated_input

    # Fire pre_tool BEFORE setting context vars; hooks may inspect
    # ContextVars on their own but the args we pass are the canonical
    # payload.
    await hooks.fire(
        "pre_tool",
        tool_name=tool.name,
        args=args,
        tool_call_id=tc["id"],
        agent_id=ctx.agent_id,
    )
    started = time.monotonic()

    # Set the runtime context vars so context-aware tools (SpawnAgentTool)
    # can read the parent's RunContext AND the current tool_use_id. Reset
    # both on exit so nothing that runs after this call in the same task
    # sees stale values.
    token_ctx = CURRENT_RUN_CONTEXT.set(ctx)
    token_tui = CURRENT_TOOL_USE_ID.set(tc["id"])
    msg: ToolMessage
    try:
        result = await tool.ainvoke(args)
        content = result if isinstance(result, str) else str(result)
        msg = ToolMessage(content=content, tool_call_id=tc["id"])
        return msg
    except asyncio.CancelledError:
        # Cooperative cancellation — re-raise so the loop can unwind.
        # We do NOT fire post_tool in this branch because the call was
        # CANCELLED, not "completed with error" — semantics differ.
        raise
    except Exception as e:
        logger.exception("tool %s raised", tool.name)
        # Fire on_error before post_tool so handlers can record the
        # exception against their open span.
        await hooks.fire(
            "on_error",
            tool_name=tool.name,
            exception=e,
            tool_call_id=tc["id"],
            agent_id=ctx.agent_id,
        )
        msg = ToolMessage(
            content=f"{type(e).__name__}: {e}",
            tool_call_id=tc["id"],
            status="error",
        )
        return msg
    finally:
        CURRENT_TOOL_USE_ID.reset(token_tui)
        CURRENT_RUN_CONTEXT.reset(token_ctx)

        # post_tool fires for both success and error paths (just not
        # CancelledError, which short-circuited above). msg is None
        # only if we re-raised CancelledError — guard for that.
        try:
            duration_ms = int((time.monotonic() - started) * 1000)
            if "msg" in locals():
                await hooks.fire(
                    "post_tool",
                    tool_name=tool.name,
                    args=args,
                    result=msg.content,
                    tool_call_id=tc["id"],
                    agent_id=ctx.agent_id,
                    status=msg.status or "success",
                    duration_ms=duration_ms,
                )
        except Exception:                               # noqa: BLE001
            # Hook firing in a finally block must never propagate.
            logger.warning("post_tool hook fan-out failed", exc_info=True)


def _is_concurrency_safe(tool: BaseTool) -> bool:
    """A tool is concurrency-safe iff its metadata explicitly says so.

    Default = False (serial). Opt in by setting:
        tool.metadata = {"concurrency_safe": True}
    """
    md: dict[str, Any] | None = tool.metadata
    if md is None:
        return False
    return bool(md.get("concurrency_safe", False))
