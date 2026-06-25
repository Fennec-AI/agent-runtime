"""Global agent registry — the shared mailbox for cross-agent communication.

This is where running async (background) agents live. The registry serves
three jobs:
    1. Keep Task references alive (prevents GC of pending tasks)
    2. Provide ID → (controller, task, inbox) lookup for cancellation/status
    3. Provide the push-notification mechanism: when a child finishes, it
       enqueues a notification into its parent's inbox; the parent drains
       at the top of its loop iteration.

Threading model: this module assumes single-threaded asyncio. Mutations are
atomic because the GIL + cooperative scheduling means no two coroutines
mutate the dict at the same instant.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from .cancellation import AbortController
from .config import CONFIG
from .types import AgentId

if TYPE_CHECKING:
    from .types import RunContext


logger = logging.getLogger(__name__)


# Match Claude Code's status taxonomy (Task.ts:15-21): running while in
# flight; completed on clean exit; failed on exception; killed on abort.
AgentStatus = Literal["running", "completed", "failed", "killed"]


# ── AgentEntry — what we store per running agent ─────────────────────────
@dataclass(slots=True)
class AgentEntry:
    """One row in the registry. Holds everything needed to operate on a
    running agent: cancel it, await its task, deliver messages to it.

    Observability fields (depth, parent_id, llm, tools, messages, …) are
    reachable through `ctx` — keeping AgentEntry slim and ensuring there's
    one source of truth for that state. `abort` is duplicated here as a
    convenience: it's the same instance as `ctx.abort`, but lifting it to
    the top level keeps cancellation paths one hop shorter.

    The status / result / error / notified / end_time block exists for
    background-spawned agents — their parent reads them via the
    notification mailbox after the child finishes. For sync spawns these
    fields are set briefly during finalize and then the entry is
    unregistered immediately.
    """
    agent_id: AgentId
    abort: AbortController                          # same object as ctx.abort
    task: Optional[asyncio.Task]                    # None for the main session
    ctx: "RunContext"                               # back-ref — depth, parent_id, llm, …
    pending_messages: list[str] = field(default_factory=list)
    description: str = ""
    subagent_type: str = ""                         # e.g. "Explore"; "" for main session
    # ── Lifecycle / completion state (populated by _run_and_finalize) ──
    status: AgentStatus = "running"
    result: Optional[str] = None                    # final assistant text (completed)
    error: Optional[str] = None                     # exception summary (failed)
    notified: bool = False                          # atomic check-and-set guard
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None


# ── Module-level singleton ──────────────────────────────────────────────
AGENT_REGISTRY: dict[AgentId, AgentEntry] = {}


# ── ID generation ────────────────────────────────────────────────────────
def new_agent_id() -> AgentId:
    """Generate a fresh agent ID. Stable, lowercase, URL/filesystem-safe."""
    return AgentId(f"a{uuid.uuid4().hex[:16]}")


# ── Output file path ─────────────────────────────────────────────────────
_TASK_OUTPUT_DIR = Path(tempfile.gettempdir()) / "agent_runtime" / "tasks"


def get_task_output_path(agent_id: AgentId) -> Path:
    """Deterministic on-disk location for this agent's final transcript.

    Mirrors Claude Code's `getTaskOutputPath(taskId)` — the notification's
    `<output-file>` field carries this path so the parent (or a future
    TaskOutput tool) can read the full result long after the AgentEntry
    has been unregistered.

    Creates the parent directory lazily on each call (cheap; mkdir with
    exist_ok=True is idempotent).
    """
    _TASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _TASK_OUTPUT_DIR / f"{agent_id}.output"


# ── Lifecycle helpers ────────────────────────────────────────────────────
def register(entry: AgentEntry) -> None:
    """Add a new agent to the registry. Called when spawning."""
    AGENT_REGISTRY[entry.agent_id] = entry


def unregister(agent_id: AgentId) -> None:
    """Remove an agent. Called after it finishes and cleanup completes."""
    AGENT_REGISTRY.pop(agent_id, None)


def get_agent(agent_id: AgentId) -> Optional[AgentEntry]:
    """Read an agent's current state. Returns None if unknown."""
    return AGENT_REGISTRY.get(agent_id)


def list_running_agents() -> list[AgentEntry]:
    """All agents whose tasks are still running (not yet completed/cancelled)."""
    return [
        e for e in AGENT_REGISTRY.values()
        if e.task is not None and not e.task.done()
    ]


def list_children(parent_id: AgentId) -> list[AgentEntry]:
    """All currently-registered agents whose parent is `parent_id`.

    O(n) scan over AGENT_REGISTRY — fine because n is small (handful of
    agents alive at once). Matches Claude Code's approach: parent_id is
    denormalized onto each entry (via entry.ctx.parent_id) rather than
    maintained as a separate index.
    """
    return [e for e in AGENT_REGISTRY.values() if e.ctx.parent_id == parent_id]


# ── Mailbox: push from children, drain by parents ────────────────────────
def enqueue_notification(parent_id: AgentId, message: str) -> None:
    """Push a notification into a parent's inbox.

    Called by background children when they finish; the parent will see the
    notification at the top of its next loop iteration.

    Silent no-op if the parent doesn't exist (e.g., session already ended).
    This is intentional — we don't want a stale child to crash because its
    parent went away.
    """
    entry = AGENT_REGISTRY.get(parent_id)
    if entry is None:
        return
    entry.pending_messages.append(message)


def drain_inbox(agent_id: AgentId) -> list[str]:
    """Drain all pending notifications for an agent. Returns them in arrival
    order; clears the inbox.

    Called at the top of every loop iteration by query().
    """
    entry = AGENT_REGISTRY.get(agent_id)
    if entry is None:
        return []
    msgs, entry.pending_messages = entry.pending_messages, []
    return msgs


# ── Cancellation helper (used by TaskStop in Step 6) ─────────────────────
async def kill_agent(
    agent_id: AgentId,
    reason: str = "killed",
    timeout: float | None = None,
) -> bool:
    """Cooperative cancellation: signal the flag, wait for clean unwind.

    Falls back to forceful task.cancel() if the agent ignores the flag for
    `timeout` seconds.

    Returns True if the agent was found (regardless of how it died), False
    if no such agent exists.
    """
    entry = AGENT_REGISTRY.get(agent_id)
    if entry is None:
        return False

    entry.abort.abort(reason)

    if entry.task is None:                          # main session — no task to await
        unregister(agent_id)
        return True

    grace = timeout if timeout is not None else CONFIG.kill_grace_period_seconds
    try:
        await asyncio.wait_for(entry.task, timeout=grace)
    except asyncio.TimeoutError:
        # Agent ignored the cooperative signal — force-cancel.
        entry.task.cancel()
        try:
            await entry.task
        except asyncio.CancelledError:
            pass
    except asyncio.CancelledError:
        # Already cancelled by something else; fine.
        pass
    finally:
        unregister(agent_id)
    return True
