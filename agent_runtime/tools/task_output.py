"""TaskOutputTool — read a background task's result / progress.

Mirrors Claude Code's `TaskOutput` tool:

    task_id: str               — which task to inspect
    block:   bool = True       — wait for completion if still running
    timeout: int = 30000ms     — max wait when block=True

Resolution order on lookup:
  1. Registry entry exists → return live status (+ wait if block=True)
  2. No registry entry → try the on-disk output file
     (get_task_output_path(task_id)); the finalize wrapper writes the
     final result/error there before unregistering, so finished agents
     remain readable after cleanup.
  3. Neither → "task not found"

The structured response (formatted string) gives the model:
  status / description / result / error / duration — whichever apply.

Concurrency-safe — pure read.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..registry import AGENT_REGISTRY, get_task_output_path
from ..types import AgentId


logger = logging.getLogger(__name__)


# Bounds on the block timeout. Match Claude Code's range (TaskOutputTool.tsx:33).
DEFAULT_BLOCK_TIMEOUT_MS = 30_000
MAX_BLOCK_TIMEOUT_MS = 600_000


class TaskOutputInput(BaseModel):
    """Arguments for TaskOutput. Mirrors Claude Code's schema."""

    task_id: str = Field(
        description="The task ID to get output from.",
    )
    block: bool = Field(
        default=True,
        description=(
            "Whether to wait for completion if the task is still running. "
            "If False, returns the current state immediately."
        ),
    )
    timeout: int = Field(
        default=DEFAULT_BLOCK_TIMEOUT_MS,
        ge=0, le=MAX_BLOCK_TIMEOUT_MS,
        description=(
            f"Max wait time in milliseconds when block=True "
            f"(default {DEFAULT_BLOCK_TIMEOUT_MS}, max {MAX_BLOCK_TIMEOUT_MS})."
        ),
    )


class TaskOutputTool(BaseTool):
    """Read a background task's status and output by ID."""

    name: str = "TaskOutput"
    description: str = (
        "Read a background task's status and final output by ID. "
        "If `block=True` (default) and the task is still running, waits up "
        "to `timeout` ms for it to finish. If the task already finished "
        "and was cleaned up, reads the final result from its on-disk "
        "output file (written by the spawn lifecycle on terminal). "
        "Returns a structured status report or an error if the task "
        "is unknown."
    )
    args_schema: type[BaseModel] = TaskOutputInput
    metadata: dict[str, Any] = {"concurrency_safe": True}

    async def _arun(
        self,
        task_id: str,
        block: bool = True,
        timeout: int = DEFAULT_BLOCK_TIMEOUT_MS,
    ) -> str:
        agent_id = AgentId(task_id)
        entry = AGENT_REGISTRY.get(agent_id)

        # ── Phase 1: optionally wait for a running task to terminate ──
        # The finalize wrapper unregisters the entry IMMEDIATELY after
        # flipping status, so the entry may vanish while we wait — that
        # case falls through to the disk fallback below.
        if entry is not None and entry.status == "running" and block:
            if entry.task is not None and not entry.task.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(entry.task),
                        timeout=timeout / 1000.0,
                    )
                except asyncio.TimeoutError:
                    entry = AGENT_REGISTRY.get(agent_id)
                    return _format_response(
                        task_id=task_id,
                        status="timeout",
                        description=entry.description if entry else "",
                        note=f"Task still running after {timeout}ms.",
                    )
            entry = AGENT_REGISTRY.get(agent_id)            # re-read

        # ── Phase 2: live entry (running with block=False, or terminal) ──
        if entry is not None:
            return _format_response(
                task_id=task_id,
                status=entry.status,
                description=entry.description,
                result=entry.result,
                error=entry.error,
                duration_ms=_duration_ms(entry),
            )

        # ── Phase 3: disk fallback (entry already cleaned up) ──────────
        # The finalize wrapper writes a JSON payload here right before
        # unregistering, so the real terminal status (completed / failed
        # / killed) is preserved across the unregister boundary.
        output_path = get_task_output_path(agent_id)
        if output_path.exists():
            try:
                raw = output_path.read_text(encoding="utf-8")
            except OSError as e:
                return f"Error reading output file for {task_id!r}: {e}"

            payload = _try_parse_json(raw)
            if payload is not None:
                # Modern JSON payload — return the real status & fields.
                return _format_response(
                    task_id=task_id,
                    status=str(payload.get("status", "unknown")),
                    description=str(payload.get("description") or ""),
                    result=payload.get("result"),
                    error=payload.get("error"),
                    duration_ms=payload.get("duration_ms"),
                )
            # Back-compat: a legacy plain-text file (written by an older
            # spawn_agent finalize that pre-dates the JSON payload).
            # Report it as 'archived' so the caller knows we don't have
            # a structured terminal status for it.
            return _format_response(
                task_id=task_id,
                status="archived",
                description="(retrieved from disk; legacy plain-text format)",
                result=raw,
            )

        # ── Phase 4: truly unknown ─────────────────────────────────────
        return (
            f"Task {task_id!r} not found. No live entry and no on-disk "
            f"output. The id may be invalid."
        )

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("TaskOutputTool is async-only; use ainvoke().")


# ── Helpers ─────────────────────────────────────────────────────────────

def _try_parse_json(raw: str) -> dict[str, Any] | None:
    """Parse the on-disk payload as JSON; return None if it isn't JSON
    or doesn't decode to a dict. Lets us distinguish modern structured
    payloads from legacy plain-text files written before the JSON
    contract was added."""
    try:
        loaded = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _duration_ms(entry: Any) -> int | None:
    if entry.end_time is None:
        return None
    return int((entry.end_time - entry.start_time) * 1000)


def _format_response(
    task_id: str,
    status: str,
    description: str = "",
    result: str | None = None,
    error: str | None = None,
    duration_ms: int | None = None,
    note: str | None = None,
) -> str:
    """Build a model-readable status block.

    Plain text rather than JSON — the model reads it directly without
    needing a parser, same convention as our notification format.
    """
    lines: list[str] = [
        f"task_id: {task_id}",
        f"status: {status}",
    ]
    if description:
        lines.append(f"description: {description}")
    if duration_ms is not None:
        lines.append(f"duration_ms: {duration_ms}")
    if note:
        lines.append(f"note: {note}")
    if result is not None:
        lines.append("--- result ---")
        lines.append(result)
    if error is not None:
        lines.append("--- error ---")
        lines.append(error)
    return "\n".join(lines)
