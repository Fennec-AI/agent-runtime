"""SpawnAgentTool — the spawn mechanism.

SpawnAgentTool is just another LangChain BaseTool that the model can call.
Its implementation recursively invokes query() with a child RunContext,
producing a complete subagent — same loop, same primitives, different
agent_id and system prompt.

Two spawn modes:
  - Synchronous (default): parent's query() awaits the subagent's full
    run inline; final assistant text comes back as the tool_result.
  - Background (run_in_background=True): subagent runs as an
    asyncio.Task; the tool_result returns immediately with a "spawned"
    acknowledgment, and when the subagent finishes, a structured
    <task-notification> XML block is enqueued into the parent's inbox.

Concurrency:
  SpawnAgentTool is marked concurrency_safe=True. Multiple SpawnAgentTool calls
  in one assistant message will fan out via the dispatcher's safe-batch
  path (parallel under semaphore).

Ctx access:
  SpawnAgentTool needs access to the parent's RunContext (for abort
  propagation, agent_id, depth, LLM) AND the current tool_use_id (for
  the notification's <tool-use-id> field). LangChain's BaseTool interface
  doesn't directly pass either to _arun. We use ContextVars set by the
  tool dispatcher just before invoke — see tool_executor's
  CURRENT_RUN_CONTEXT and CURRENT_TOOL_USE_ID.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .. import hooks
from ..cancellation import create_child_controller
from ..cleanup_registry import register_cleanup
from ..config import CONFIG
from ..models import get_model
from ..registry import (
    AGENT_REGISTRY,
    AgentEntry,
    enqueue_notification,
    get_task_output_path,
    kill_agent,
    new_agent_id,
    register,
    unregister,
)
from ..subagents import SUBAGENT_REGISTRY, get_subagent, list_subagent_types
from ..types import AgentId, RunContext


DEFAULT_SUBAGENT_TYPE = "general-purpose"
"""The subagent type used when the model omits `subagent_type`."""


logger = logging.getLogger(__name__)


# ── Tool's input schema (what the LLM sees) ─────────────────────────────

class SpawnAgentToolInput(BaseModel):
    """Arguments the LLM provides when spawning a subagent.

    Schema mirrors Claude Code's `Agent` tool exactly so the model sees a
    familiar surface. Required: description, prompt. Everything else is
    optional with sensible defaults.
    """

    description: str = Field(
        description="A short (3-5 word) description of the task",
    )
    prompt: str = Field(
        description="The task for the agent to perform",
    )
    subagent_type: str | None = Field(
        default=None,
        description=(
            "The type of specialized agent to use for this task. If omitted, "
            f"the {DEFAULT_SUBAGENT_TYPE!r} agent is used."
        ),
    )
    run_in_background: bool | None = Field(
        default=None,
        description=(
            "Set to true to run this agent in the background. You will be "
            "notified when it completes."
        ),
    )


# ── The Tool ────────────────────────────────────────────────────────────

class SpawnAgentTool(BaseTool):
    """Launch a subagent to perform a sub-task.

    The subagent runs its own query() loop with its own system prompt and
    optional restricted tool list. The result returned is the subagent's
    final assistant text — the parent never sees the subagent's internal
    tool calls or intermediate reasoning.
    """

    name: str = "Agent"
    description: str = ""                              # set in __init__
    args_schema: type[BaseModel] = SpawnAgentToolInput
    # Concurrency-safe — multiple subagent spawns in one turn run in parallel.
    metadata: dict[str, Any] = {"concurrency_safe": True}

    def __init__(self, **kwargs: Any) -> None:
        # Capture the registered subagent types at construction time.
        # If you register new subagent types AFTER this, rebuild the tool.
        kwargs.setdefault("description", _build_description())
        super().__init__(**kwargs)

    async def _arun(
        self,
        description: str,
        prompt: str,
        subagent_type: str | None = None,
        run_in_background: bool | None = None,
    ) -> str:
        """Spawn a subagent. Sync path returns the child's final text;
        background path returns "spawned" immediately and delivers the
        result via a <task-notification> in the parent's inbox.
        """
        # ── Get parent ctx + tool_use_id from the dispatcher's ContextVars ──
        # Lazy import to avoid circular dependency at module load.
        from ..tool_executor import CURRENT_RUN_CONTEXT, CURRENT_TOOL_USE_ID
        parent_ctx = CURRENT_RUN_CONTEXT.get(None)
        if parent_ctx is None:
            return (
                "Internal error: SpawnAgentTool can only be invoked through the "
                "agent_runtime tool executor (parent context unavailable)."
            )
        tool_use_id = CURRENT_TOOL_USE_ID.get(None)

        # ── Depth cap — prevent runaway recursion ──
        if parent_ctx.depth + 1 > CONFIG.max_agent_depth:
            logger.warning(
                "SpawnAgentTool refused spawn: depth %d would exceed max %d",
                parent_ctx.depth + 1, CONFIG.max_agent_depth,
            )
            return (
                f"Agent depth limit reached ({CONFIG.max_agent_depth}). "
                f"Cannot spawn a subagent at depth {parent_ctx.depth + 1}."
            )

        # ── Resolve the subagent type (default if omitted, like Claude Code) ──
        resolved_type = subagent_type or DEFAULT_SUBAGENT_TYPE
        defn = get_subagent(resolved_type)
        if defn is None:
            available = ", ".join(list_subagent_types()) or "(none registered)"
            return (
                f"Unknown subagent_type {resolved_type!r}. "
                f"Available types: {available}."
            )

        # ── Build child's tool list ──
        # Explicit whitelist wins. Otherwise inherit parent's tools,
        # stripping SpawnAgentTool (always — prevents trivial recursion)
        # plus anything in defn.disallowed_tools (mirrors Claude Code's
        # exploreAgent.ts:67 — read-only agents deny Edit/Write/etc).
        if defn.tools is not None:
            child_tools = list(defn.tools)
        else:
            excluded = {self.name}
            if defn.disallowed_tools:
                excluded.update(defn.disallowed_tools)
            child_tools = [t for t in parent_ctx.tools if t.name not in excluded]

        # ── Choose the child's LLM ──
        # If the subagent definition specifies a model profile, look it up
        # in the registry. Otherwise inherit the parent's LLM.
        if defn.model is not None:
            try:
                child_llm = get_model(defn.model)
            except KeyError as e:
                return f"Error: {e}"
        else:
            child_llm = parent_ctx.llm

        # ── Build child's RunContext ──
        child_id = new_agent_id()
        child_abort = create_child_controller(parent_ctx.abort)
        # The child gets a fresh messages list (its own conversation, starts
        # with system + initial user prompt). Built inside
        # _run_subagent_to_completion just before query().
        child_ctx = RunContext(
            agent_id=child_id,
            messages=[],                     # filled in below
            abort=child_abort,
            tools=child_tools,
            llm=child_llm,
            depth=parent_ctx.depth + 1,
            parent_id=parent_ctx.agent_id,
            # Inherit the parent's permission MODE so plan mode (read-only)
            # propagates down the tree. But NOT the ask_callback — a
            # background subagent must not block on a user prompt; its
            # gated tools auto-allow in default mode (and are already
            # restricted by the subagent's tool list), while plan mode
            # still denies them.
            permission_mode=parent_ctx.permission_mode,
            ask_callback=None,
        )

        # Register so observers can see the subagent and so the eventual
        # notification has a registry entry to update. task=None for now;
        # the background branch overwrites it with the live asyncio.Task.
        register(AgentEntry(
            agent_id=child_id,
            abort=child_abort,
            task=None,
            ctx=child_ctx,
            description=description,
            subagent_type=resolved_type,
        ))

        # ── Branch on sync vs background ──
        if run_in_background:
            # Fire-and-forget: create the task, hand back an ack. The
            # finalize wrapper handles unregister + notification on its own.
            #
            # Register a shutdown cleanup so if the process exits before
            # this child finishes, kill_agent fires and unwinds it cleanly
            # (orphan prevention). The cleanup unregisters itself in
            # _run_and_finalize's `finally` so it doesn't fire on a
            # finished agent.
            cleanup_unregister = register_cleanup(
                lambda aid=child_id: kill_agent(aid, reason="shutdown"),
            )

            task = asyncio.create_task(
                _run_and_finalize(
                    child_ctx=child_ctx,
                    system_prompt=defn.system_prompt,
                    prompt=prompt,
                    parent_id=parent_ctx.agent_id,
                    tool_use_id=tool_use_id,
                    cleanup_unregister=cleanup_unregister,
                )
            )
            AGENT_REGISTRY[child_id].task = task
            return (
                f"Spawned background agent {child_id} ({resolved_type!r}). "
                f"You will be notified when it completes."
            )

        # Sync path: await the child inline, return its final text. We
        # don't bother updating status/result/end_time on the entry — sync
        # has no notification consumer, the text rides back as the
        # tool_result, and the entry is gone immediately after.
        try:
            return await _run_subagent_to_completion(
                child_ctx=child_ctx,
                system_prompt=defn.system_prompt,
                prompt=prompt,
            )
        finally:
            unregister(child_id)

    def _run(self, *args: Any, **kwargs: Any) -> str:
        # We're async-only. LangChain calls _run for sync invocations; raise
        # so users know to use the async path.
        raise NotImplementedError("SpawnAgentTool is async-only; use ainvoke().")


# ── Internal helpers ────────────────────────────────────────────────────

async def _run_and_finalize(
    child_ctx: RunContext,
    system_prompt: str,
    prompt: str,
    parent_id: AgentId,
    tool_use_id: str | None,
    cleanup_unregister: Callable[[], None] | None = None,
) -> None:
    """Background spawn lifecycle: run the child's loop to completion,
    capture outcome on its AgentEntry, write final transcript to disk,
    enqueue a <task-notification> into the parent's inbox, unregister.

    Errors are caught here (they map to status='failed' notifications)
    rather than propagated — a background child crashing must not break
    the parent's loop or leave the registry in an inconsistent state.

    The notification is enqueued under an atomic check-and-set on
    entry.notified, mirroring Claude Code's pattern
    (LocalAgentTask.tsx:224). With single-threaded asyncio the race
    window is narrow, but the guard still matters once we add a TaskStop
    tool that may enqueue a 'killed' notification concurrently with
    natural completion.
    """
    child_id = child_ctx.agent_id
    entry = AGENT_REGISTRY.get(child_id)
    if entry is None:
        # Entry vanished before we started — only happens if something
        # else unregistered us, which would be a bug elsewhere.
        logger.warning("background child %s has no registry entry; aborting finalize", child_id)
        if cleanup_unregister is not None:
            cleanup_unregister()
        return

    # on_spawn fires at the moment the child actually starts running its
    # loop — after registration, before the first LLM call. Observability
    # backends use this to open a root span for the child.
    await hooks.fire(
        "on_spawn",
        agent_id=child_id,
        parent_id=parent_id,
        subagent_type=entry.subagent_type or "general-purpose",
        run_in_background=True,
        description=entry.description or "",
    )

    try:
        result = await _run_subagent_to_completion(
            child_ctx=child_ctx,
            system_prompt=system_prompt,
            prompt=prompt,
        )
        # query() returns silently on abort; check the flag to distinguish
        # "completed normally" from "exited because someone aborted us".
        if child_ctx.abort.aborted:
            entry.status = "killed"
        else:
            entry.status = "completed"
            entry.result = result
    except asyncio.CancelledError:
        # Forceful task.cancel() (kill_agent fallback) lands here.
        entry.status = "killed"
        # Don't re-raise — we want the notification to land, not unwind.
    except Exception as exc:                                # pylint: disable=broad-except
        logger.exception("background child %s crashed", child_id)
        entry.status = "failed"
        entry.error = f"{type(exc).__name__}: {exc}"
    finally:
        entry.end_time = time.time()

        # Persist final state to disk so a future TaskOutput tool (or a
        # human inspecting the filesystem) can read it after the entry
        # is unregistered. Best-effort: a disk error here mustn't block
        # the notification.
        #
        # Stored as JSON (not plain text as before) so the disk-fallback
        # path in TaskOutput can preserve the REAL terminal status
        # (completed/failed/killed) instead of always reporting
        # 'archived'. Pre-fix, TaskOutput's disk read could only return
        # 'archived' which confused the model into thinking the task had
        # been cleaned up rather than e.g. completed normally.
        try:
            import json
            duration_ms = (
                int((entry.end_time - entry.start_time) * 1000)
                if entry.end_time is not None
                else None
            )
            disk_payload = {
                "status": entry.status,
                "description": entry.description or "",
                "result": entry.result,
                "error": entry.error,
                "duration_ms": duration_ms,
            }
            get_task_output_path(child_id).write_text(
                json.dumps(disk_payload, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("failed to write task output for %s", child_id, exc_info=True)

        # on_terminate fires BEFORE the entry is unregistered so hook
        # handlers can read final state from the registry if they need
        # to. Awaiting it in the finally block — exceptions are
        # swallowed by hooks.fire itself, so we won't break unwind.
        await hooks.fire(
            "on_terminate",
            agent_id=child_id,
            status=entry.status,
            duration_ms=(
                int((entry.end_time - entry.start_time) * 1000)
                if entry.end_time is not None
                else 0
            ),
            error=entry.error,
        )

        # Atomic check-and-set on notified.
        should_enqueue = not entry.notified
        if should_enqueue:
            entry.notified = True

        if should_enqueue:
            notification = _format_task_notification(entry, tool_use_id)
            enqueue_notification(parent_id, notification)

        unregister(child_id)

        # Remove our shutdown handler — we're done, no orphan to kill.
        if cleanup_unregister is not None:
            cleanup_unregister()


def _format_task_notification(entry: AgentEntry, tool_use_id: str | None) -> str:
    """Build the structured XML notification body that lands in the
    parent's inbox. Tag set mirrors Claude Code's notification format
    (see LocalAgentTask.tsx:252).

    Required tags (always present): task-id, status, summary, output-file.
    Optional tags: tool-use-id (when invoked via a tool call), result
    (only for completed status), usage (when duration is known).
    """
    tool_use_id_line = (
        f"\n<tool-use-id>{tool_use_id}</tool-use-id>" if tool_use_id else ""
    )

    # <summary> — human-readable one-liner, varies by status.
    desc = entry.description or entry.agent_id
    if entry.status == "completed":
        summary = f'Agent "{desc}" completed'
    elif entry.status == "failed":
        summary = f'Agent "{desc}" failed: {entry.error or "Unknown error"}'
    elif entry.status == "killed":
        summary = f'Agent "{desc}" was stopped'
    else:                                           # 'running' shouldn't notify, but handle defensively
        summary = f'Agent "{desc}" status: {entry.status}'

    # <result> — only on success. Failures/kills carry their info in <summary>
    # and the full transcript is in <output-file>.
    result_section = (
        f"\n<result>{entry.result}</result>"
        if entry.status == "completed" and entry.result is not None
        else ""
    )

    # <usage> — duration when end_time is set.
    if entry.end_time is not None:
        duration_ms = int((entry.end_time - entry.start_time) * 1000)
        usage_section = f"\n<usage><duration-ms>{duration_ms}</duration-ms></usage>"
    else:
        usage_section = ""

    output_path = get_task_output_path(entry.agent_id)

    return (
        f"<task-notification>\n"
        f"<task-id>{entry.agent_id}</task-id>{tool_use_id_line}\n"
        f"<output-file>{output_path}</output-file>\n"
        f"<status>{entry.status}</status>\n"
        f"<summary>{summary}</summary>"
        f"{result_section}{usage_section}\n"
        f"</task-notification>"
    )


async def _run_subagent_to_completion(
    child_ctx: RunContext,
    system_prompt: str,
    prompt: str,
) -> str:
    """Run a subagent's query() loop to completion, return final assistant text.

    Also forwards every event the child yields to the CURRENT_SUBAGENT_EVENT_SINK
    callable (if one is installed), so a CLI / UI can surface the child's
    work — tool calls, intermediate streaming, results — LIVE instead of
    only seeing the final answer. Default sink is None (zero-cost no-op).
    """
    # Lazy imports — avoid circular deps at module load.
    from ..query import query
    from ..tool_executor import CURRENT_SUBAGENT_EVENT_SINK

    # Populate the child's messages: system prompt + the parent-supplied prompt.
    child_ctx.messages.extend([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ])

    sink = CURRENT_SUBAGENT_EVENT_SINK.get(None)

    final_text = ""
    async for event in query(child_ctx):
        # Forward to observer (CLI printer, transcript file, etc.). The
        # try/except keeps a buggy sink from breaking the child's loop —
        # observation is best-effort, not load-bearing.
        if sink is not None:
            try:
                sink(child_ctx, event)
            except Exception:                       # noqa: BLE001
                logger.exception("subagent event sink raised")

        # Extract final text from AIMessage (skip chunks — they're streaming).
        if isinstance(event, AIMessage):
            content = event.content
            if isinstance(content, str):
                final_text = content
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                if parts:
                    final_text = "".join(parts)

    return final_text or "(subagent produced no text response)"


def _build_description() -> str:
    """Build SpawnAgentTool's description, listing registered subagent types."""
    base = (
        "Launch a subagent to perform a sub-task. The subagent runs with "
        "its own system prompt and (optionally restricted) tool set. "
        "Returns the subagent's final answer as the tool result."
    )
    if not SUBAGENT_REGISTRY:
        return f"{base}\n\nNo subagent types are currently registered."
    types_list = "\n".join(
        f"  - {d.name}: {d.description}"
        for d in SUBAGENT_REGISTRY.values()
    )
    return f"{base}\n\nAvailable subagent types:\n{types_list}"
