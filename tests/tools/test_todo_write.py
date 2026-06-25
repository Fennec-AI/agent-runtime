"""TodoWrite — list replacement, per-agent isolation, verification nudge."""
from __future__ import annotations

import pytest

from agent_runtime.tools import TodoWriteTool
from agent_runtime.tools.todo_write import clear_todos, get_todos


pytestmark = pytest.mark.asyncio


def _todos(*items):
    return [
        {"content": c, "status": s, "activeForm": a}
        for (c, s, a) in items
    ]


async def test_replaces_list_and_returns_checklist(with_run_context, run_context):
    clear_todos()
    tool = TodoWriteTool()
    with with_run_context:
        out = await tool.ainvoke({"todos": _todos(
            ("Write the parser", "in_progress", "Writing the parser"),
            ("Add tests", "pending", "Adding tests"),
        )})
    assert "Todos (2)" in out
    assert "▶ Writing the parser" in out      # in_progress shows activeForm
    assert "☐ Add tests" in out               # pending shows content
    # stored under the agent id
    stored = get_todos(str(run_context.agent_id))
    assert len(stored) == 2
    assert stored[0]["status"] == "in_progress"


async def test_whole_list_replace(with_run_context, run_context):
    clear_todos()
    tool = TodoWriteTool()
    with with_run_context:
        await tool.ainvoke({"todos": _todos(("A", "pending", "Doing A"))})
        await tool.ainvoke({"todos": _todos(
            ("A", "completed", "Doing A"),
            ("B", "in_progress", "Doing B"),
        )})
    stored = get_todos(str(run_context.agent_id))
    assert len(stored) == 2                    # replaced, not appended
    assert stored[0]["status"] == "completed"


async def test_verification_nudge_when_all_done_no_verify(with_run_context):
    clear_todos()
    tool = TodoWriteTool()
    with with_run_context:
        out = await tool.ainvoke({"todos": _todos(
            ("Implement feature", "completed", "Implementing feature"),
            ("Update docs", "completed", "Updating docs"),
        )})
    assert "verification step" in out


async def test_no_nudge_when_a_todo_mentions_verify(with_run_context):
    clear_todos()
    tool = TodoWriteTool()
    with with_run_context:
        out = await tool.ainvoke({"todos": _todos(
            ("Implement feature", "completed", "Implementing feature"),
            ("Verify tests pass", "completed", "Verifying tests"),
        )})
    assert "verification step" not in out


async def test_no_nudge_when_not_all_done(with_run_context):
    clear_todos()
    tool = TodoWriteTool()
    with with_run_context:
        out = await tool.ainvoke({"todos": _todos(
            ("Implement feature", "completed", "Implementing feature"),
            ("Update docs", "pending", "Updating docs"),
        )})
    assert "verification step" not in out


async def test_empty_list_clears(with_run_context, run_context):
    clear_todos()
    tool = TodoWriteTool()
    with with_run_context:
        await tool.ainvoke({"todos": _todos(("A", "pending", "Doing A"))})
        out = await tool.ainvoke({"todos": []})
    assert "cleared" in out.lower()
    assert get_todos(str(run_context.agent_id)) == []
