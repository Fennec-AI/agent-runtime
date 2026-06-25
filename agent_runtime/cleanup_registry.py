"""Global cleanup registry — shutdown hooks for orphan-prevention.

Background agents register a `kill_agent(agent_id)` callable here at
spawn time. When the agent reaches a terminal state (completed / failed
/ killed), it unregisters. If the process exits before the agent finishes
— or the host Session is closed — `run_all_cleanups()` iterates whatever
callbacks remain and kills the orphans.

This is the third leak-prevention layer Claude Code uses (alongside the
`notified` flag and the AbortController hierarchy). Mirrors
`utils/cleanupRegistry.ts` in Claude Code's source.

Threading: single-threaded asyncio. The internal Set is mutated only by
register/discard — both are atomic in CPython. No locks needed.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable


logger = logging.getLogger(__name__)


# A cleanup is any zero-arg async function. We keep them in a Set so
# (un)registration is O(1) and idempotent.
_CleanupFn = Callable[[], Awaitable[None]]
_cleanup_functions: set[_CleanupFn] = set()


def register_cleanup(fn: _CleanupFn) -> Callable[[], None]:
    """Add `fn` to the cleanup set; return an idempotent unregister
    function.

    `fn` must be an async callable taking no arguments. It runs at most
    once per `run_all_cleanups()` invocation — exceptions are caught and
    logged, never propagated (one bad cleanup shouldn't stop the others).

    The returned unregister is safe to call multiple times; the second
    and later calls are no-ops. Callers should invoke it as soon as their
    work reaches a terminal state to avoid the cleanup firing on a
    finished task.
    """
    _cleanup_functions.add(fn)

    def unregister() -> None:
        _cleanup_functions.discard(fn)

    return unregister


async def run_all_cleanups() -> None:
    """Run every registered cleanup, concurrently. Always drains the
    registry — even if some cleanups raise.

    Called by Session.aclose() and (optionally) by application-level
    shutdown hooks. Each cleanup runs under `asyncio.gather` with
    `return_exceptions=True` so one failure doesn't prevent the others.
    """
    if not _cleanup_functions:
        return

    # Snapshot the set so any cleanup that unregisters itself (or others)
    # during execution can't mutate what we're iterating.
    pending = list(_cleanup_functions)
    _cleanup_functions.clear()

    results = await asyncio.gather(
        *(fn() for fn in pending),
        return_exceptions=True,
    )

    for fn, result in zip(pending, results):
        if isinstance(result, BaseException):
            logger.warning(
                "cleanup function %r raised during shutdown: %s",
                fn, result, exc_info=result,
            )


def cleanup_count() -> int:
    """How many cleanups are currently registered (for introspection / tests)."""
    return len(_cleanup_functions)
