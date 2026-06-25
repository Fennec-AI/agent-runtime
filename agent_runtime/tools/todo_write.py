"""TodoWrite — a structured task checklist for the current session.

Faithful port of Claude Code's TodoWriteTool. The model maintains a live
to-do list to plan and track multi-step work. Each call REPLACES the whole
list (the model sends the full updated list every time).

Todo item shape (from utils/todo/types.ts):
    content:    str   — imperative task description ("Run the tests")
    status:     pending | in_progress | completed
    activeForm: str   — present-continuous form shown while in progress
                        ("Running the tests")

The list is stored per-agent in a module-level registry, keyed by the
calling agent's id (read from CURRENT_RUN_CONTEXT). So the main session and
each subagent keep independent lists.

Verification nudge: CC appends a reminder when every item is completed but
none of them mention verification — a gentle push to actually verify the
work rather than declaring done. Ported here.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


TODO_WRITE_TOOL_NAME = "TodoWrite"

TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItem(BaseModel):
    """One task in the checklist."""
    content: str = Field(
        min_length=1,
        description="The task, in imperative form (e.g. 'Run the test suite').",
    )
    status: TodoStatus = Field(
        description="pending, in_progress, or completed.",
    )
    activeForm: str = Field(
        min_length=1,
        description=(
            "Present-continuous form shown while the task is in progress "
            "(e.g. 'Running the test suite')."
        ),
    )


class TodoWriteInput(BaseModel):
    """Arguments for TodoWrite — the full updated list."""
    todos: list[TodoItem] = Field(description="The updated todo list.")


# Per-agent todo lists. Keyed by agent_id; values are lists of plain dicts.
_TODO_LISTS: dict[str, list[dict[str, Any]]] = {}


def get_todos(agent_id: str) -> list[dict[str, Any]]:
    """Current todo list for an agent (empty if none). For tests/inspection."""
    return _TODO_LISTS.get(agent_id, [])


def clear_todos(agent_id: str | None = None) -> None:
    """Clear one agent's list, or all lists. For tests."""
    if agent_id is None:
        _TODO_LISTS.clear()
    else:
        _TODO_LISTS.pop(agent_id, None)


DESCRIPTION = (
    "Create and manage a structured task list for the current session. Use "
    "it proactively for complex multi-step work (3+ steps), when the user "
    "gives multiple tasks, or to track progress on a non-trivial request. "
    "Each call REPLACES the whole list — always send the full updated list. "
    "Mark exactly one task in_progress at a time; mark a task completed "
    "immediately when it's done before starting the next. Skip this tool for "
    "single trivial tasks. Each item needs: content (imperative), status "
    "(pending/in_progress/completed), and activeForm (present-continuous)."
)


class TodoWriteTool(BaseTool):
    """Maintain the session's task checklist."""

    name: str = TODO_WRITE_TOOL_NAME
    description: str = DESCRIPTION
    args_schema: type[BaseModel] = TodoWriteInput
    # Mutates the shared per-agent list — keep it serial (not concurrency
    # safe), so two TodoWrite calls in one turn don't race on the list.
    metadata: dict[str, Any] = {"concurrency_safe": False}

    async def _arun(self, todos: list[Any]) -> str:
        from ..tool_executor import CURRENT_RUN_CONTEXT
        ctx = CURRENT_RUN_CONTEXT.get(None)
        agent_id = str(ctx.agent_id) if ctx is not None else "default"

        new_list = [_normalize(t) for t in todos]
        _TODO_LISTS[agent_id] = new_list

        nudge = _needs_verification_nudge(new_list)
        return _format_todos(new_list, nudge)

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("TodoWriteTool is async-only; use ainvoke().")


# ── helpers ───────────────────────────────────────────────────────────

def _normalize(item: Any) -> dict[str, Any]:
    """Accept either a TodoItem model or a plain dict; return a dict."""
    if isinstance(item, BaseModel):
        return item.model_dump()
    if isinstance(item, dict):
        return {
            "content": item.get("content", ""),
            "status": item.get("status", "pending"),
            "activeForm": item.get("activeForm", item.get("content", "")),
        }
    return {"content": str(item), "status": "pending", "activeForm": str(item)}


def _needs_verification_nudge(todos: list[dict[str, Any]]) -> bool:
    """CC's nudge: all items completed AND none mention verification."""
    if not todos:
        return False
    all_done = all(t.get("status") == "completed" for t in todos)
    mentions_verify = any(
        re.search(r"verif", t.get("content", ""), re.IGNORECASE) for t in todos
    )
    return all_done and not mentions_verify


_STATUS_GLYPH = {"completed": "✔", "in_progress": "▶", "pending": "☐"}


def _format_todos(todos: list[dict[str, Any]], nudge: bool) -> str:
    """Render the list as a readable checklist for the model + the renderer."""
    if not todos:
        return "Todo list cleared (no items)."

    lines = [f"Todos ({len(todos)}):"]
    for t in todos:
        status = t.get("status", "pending")
        glyph = _STATUS_GLYPH.get(status, "☐")
        if status == "in_progress":
            lines.append(f"  {glyph} {t.get('activeForm', t.get('content', ''))}")
        else:
            lines.append(f"  {glyph} {t.get('content', '')}")

    if nudge:
        lines.append("")
        lines.append(
            "All tasks are marked completed. Before finishing, consider "
            "adding a verification step (run tests / build) to confirm the "
            "work actually succeeded."
        )
    return "\n".join(lines)
