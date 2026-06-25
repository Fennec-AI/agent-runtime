"""Hook pipeline — observation + extension points for the runtime.

Hooks are fire-and-forget observation callbacks scattered through the
agent runtime at the lifecycle points that matter. Register any number
of callbacks per event; they all run in registration order. Exceptions
inside a hook are caught and logged — hooks must NEVER break agent
execution. They observe; they don't control.

Built-in events (more may be added; the contract is "we add events,
never break existing signatures"):

    pre_tool(tool_name, args, tool_call_id, agent_id)
        Fired right before a tool's _arun executes. Receives the
        invoking agent's id and the unique tool_use_id (so spans can
        be matched to results).

    post_tool(tool_name, args, result, tool_call_id, agent_id,
              status, duration_ms)
        Fired after the tool returns (success or error). `status`
        mirrors the ToolMessage status ("success" or "error").
        duration_ms is wall-clock from pre_tool to post_tool.

    on_error(tool_name, exception, tool_call_id, agent_id)
        Fired when a tool's _arun RAISES (not when it returns an error
        string). Fired BEFORE post_tool — both fire on the same call.

    on_spawn(agent_id, parent_id, subagent_type, run_in_background,
             description)
        Fired when a subagent is spawned (sync or background). Fired
        once per spawn; if the spawn fails before the entry is
        registered, this is not fired.

    on_terminate(agent_id, status, duration_ms, error)
        Fired when an agent reaches a terminal state (completed,
        failed, killed). Fired from inside _run_and_finalize's finally
        block, BEFORE the entry is unregistered.

Hook callbacks may be sync OR async — fire() awaits coroutines
automatically. Sync hooks need to be FAST (they run inline on the
agent's hot path); push slow work to a background task if you need it.

Example — register a built-in hook:

    from agent_runtime.hooks import register_hook

    def log_tool(tool_name, status, duration_ms, **_):
        print(f"  [hook] {tool_name}: {status} in {duration_ms}ms")

    register_hook("post_tool", log_tool)
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from typing import Any, Callable


logger = logging.getLogger(__name__)


# Hook callable accepts arbitrary kwargs (the event payload). May be
# sync or async — fire() handles both.
HookFn = Callable[..., Any]


# Registered hooks keyed by event name. defaultdict so register is O(1)
# without an existence check; module-level mutable state is fine here
# (single-process asyncio runtime, same as our registries).
_hooks: dict[str, list[HookFn]] = defaultdict(list)


# ── Registration ────────────────────────────────────────────────────────

def register_hook(event: str, fn: HookFn) -> Callable[[], None]:
    """Register a callback for `event`. Returns an idempotent unregister
    function — call it to drop the hook (subsequent calls are no-ops).

    Multiple hooks per event compose; they fire in registration order.
    """
    _hooks[event].append(fn)

    def unregister() -> None:
        try:
            _hooks[event].remove(fn)
        except ValueError:
            pass                                        # idempotent

    return unregister


def clear_hooks(event: str | None = None) -> None:
    """Remove all hooks (for `event` if specified, else every event).

    Primarily for tests — production code unregisters individual hooks
    via the function returned from register_hook().
    """
    if event is None:
        _hooks.clear()
    else:
        _hooks.pop(event, None)


def hook_count(event: str | None = None) -> int:
    """Registered-hook count (for `event` or total). Introspection."""
    if event is None:
        return sum(len(h) for h in _hooks.values())
    return len(_hooks.get(event, []))


# ── Firing ──────────────────────────────────────────────────────────────

async def fire(event: str, **kwargs: Any) -> None:
    """Fire all hooks for `event`, awaiting any coroutines. Exceptions
    are caught and logged — a misbehaving hook never breaks the agent.

    Use this from async code (the common case — tool execution, spawn
    finalize, etc.). For sync call sites use fire_sync().
    """
    hooks = _hooks.get(event)
    if not hooks:
        return

    # Snapshot the list so a hook can unregister itself (or others) mid-fire
    # without mutating what we're iterating.
    for fn in list(hooks):
        try:
            r = fn(**kwargs)
            if inspect.iscoroutine(r):
                await r
        except Exception:                               # noqa: BLE001
            logger.warning(
                "hook %r raised on event %r", fn, event,
                exc_info=True,
            )


def fire_sync(event: str, **kwargs: Any) -> None:
    """Sync variant — call SYNC hooks inline; coroutines from async
    hooks are scheduled as tasks (fire-and-forget) if a loop is running.

    Use only from sync call sites that can't await. Prefer fire().
    """
    hooks = _hooks.get(event)
    if not hooks:
        return

    for fn in list(hooks):
        try:
            r = fn(**kwargs)
            if inspect.iscoroutine(r):
                try:
                    asyncio.create_task(r, name=f"hook-{event}")
                except RuntimeError:
                    # No running event loop — drop quietly to avoid
                    # the "coroutine was never awaited" warning.
                    r.close()
        except Exception:                               # noqa: BLE001
            logger.warning(
                "hook %r raised on event %r", fn, event,
                exc_info=True,
            )
