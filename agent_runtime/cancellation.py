"""AbortController — cooperative cancellation primitive.

A small wrapper around asyncio.Event that adds:
  - friendlier method names (abort() vs set(), aborted vs is_set())
  - a `reason` field that carries WHY we aborted
  - hierarchical child controllers (parent abort → child abort, GC-safe via weakref)

Cancellation is COOPERATIVE: setting the flag does nothing by itself. Code
that holds a reference checks `aborted` at safe checkpoints and exits
gracefully (typically with try/finally cleanup).
"""
from __future__ import annotations

import asyncio
import weakref


class AbortController:
    """A latched flag with an `await`-able signal and a reason string."""

    # __weakref__ is required so create_child_controller's weakref.ref(child)
    # works — classes using __slots__ otherwise can't be weak-referenced.
    __slots__ = ("signal", "reason", "__weakref__")

    def __init__(self) -> None:
        self.signal = asyncio.Event()
        self.reason: str | None = None

    def abort(self, reason: str = "") -> None:
        """Set the flag. Subsequent `aborted` reads return True. Wakes all
        coroutines parked on `await self.signal.wait()`."""
        self.reason = reason
        self.signal.set()

    @property
    def aborted(self) -> bool:
        """Non-blocking peek at the flag. Use this in polling loops."""
        return self.signal.is_set()


def create_child_controller(parent: AbortController) -> AbortController:
    """Create a child controller wired to fire when the parent fires.

    Memory-safe: uses weakref so the parent doesn't pin abandoned children.
    When the parent fires, a small background watcher wakes up and propagates
    the abort to the child. When the child is GC'd, the watcher is cancelled
    automatically.

    Use this when spawning a sync subagent that should die with the parent.
    For async (background) subagents, create a plain AbortController() that
    is NOT linked to the parent — they should survive ESC.
    """
    child = AbortController()

    # Fast path: parent is already aborted, no need to set up a watcher.
    if parent.aborted:
        child.abort(reason=parent.reason or "")
        return child

    weak_child = weakref.ref(child)

    async def _propagate() -> None:
        await parent.signal.wait()
        c = weak_child()
        if c is not None and not c.aborted:
            c.abort(reason=parent.reason or "")

    task = asyncio.create_task(_propagate())

    # When the child is garbage-collected, cancel the watcher so it doesn't
    # stay parked in the parent's Event waiter list forever (memory leak).
    weakref.finalize(child, task.cancel)

    return child
