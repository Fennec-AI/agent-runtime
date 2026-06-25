"""TaskStopTool — let the model kill a running background task by ID.

Mirrors Claude Code's `TaskStop` tool. Required: task_id. The schema uses
the generic `task_id` name (not `agent_id`) because Claude Code's
TaskStopTool is the universal kill surface for ANY background task type
(local_agent, local_bash, remote_agent, in_process_teammate, …). We
only have async agents today, but keeping `task_id` lets us add more
task types later without a schema rename.

Concurrency:
  Concurrency_safe = True. The model may emit TaskStop in a single batch
  alongside Read/Glob/Grep without forcing serial execution.

Why this tool exists separately (vs just calling kill_agent directly):
  - It's the LLM-facing surface for cancellation, with a clear schema
    the model can call.
  - The dispatcher's permission system can deny it per-agent (e.g.,
    "Explore subagents may NOT kill other agents") via the same
    machinery that gates Bash/Edit/etc.
  - Errors come back as ToolMessage(status="error") — the model sees
    "agent already finished" or "unknown id" as data, not exceptions.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..registry import AGENT_REGISTRY, kill_agent
from ..types import AgentId


logger = logging.getLogger(__name__)


class TaskStopInput(BaseModel):
    """Arguments for TaskStop."""
    task_id: str = Field(
        description=(
            "The ID of the background task to stop. For background agents, "
            "this is the agent_id returned by the previous SpawnAgent call."
        ),
    )


class TaskStopTool(BaseTool):
    """Stop a running background task by ID."""

    name: str = "TaskStop"
    description: str = (
        "Stop a running background task (e.g., a subagent spawned with "
        "run_in_background=True). Pass the task_id you received when the "
        "task was spawned. Returns confirmation of the kill, or a clear "
        "error if the task is unknown or already finished. Cancellation "
        "is cooperative — the task gets a brief grace period to unwind "
        "before being forcibly cancelled."
    )
    args_schema: type[BaseModel] = TaskStopInput
    # Concurrency-safe — multiple TaskStop calls in one turn run in parallel.
    metadata: dict[str, Any] = {"concurrency_safe": True}

    async def _arun(self, task_id: str) -> str:
        # Look up the agent FIRST so we can give an accurate response.
        # kill_agent itself returns False on unknown IDs (silent no-op,
        # which is great for retries but unhelpful for the model).
        entry = AGENT_REGISTRY.get(AgentId(task_id))
        if entry is None:
            return (
                f"Task {task_id!r} not found. It may have already finished "
                f"and been cleaned up, or the ID is invalid."
            )

        # Snapshot description + status for the response BEFORE we kill,
        # since kill_agent unregisters the entry on its way out.
        description = entry.description or task_id
        prior_status = entry.status

        if prior_status != "running":
            return (
                f"Task {task_id!r} ({description!r}) is already in "
                f"terminal state {prior_status!r}. Nothing to stop."
            )

        # Fire the cooperative abort + grace-period wait inside kill_agent.
        # Returns True iff the agent was found (which we just confirmed).
        logger.info("TaskStop: killing %s (%s)", task_id, description)
        await kill_agent(AgentId(task_id), reason="killed_by_TaskStop")
        return f"Stopped task {task_id!r} ({description!r})."

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("TaskStopTool is async-only; use ainvoke().")
