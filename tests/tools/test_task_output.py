"""TaskOutput tool — JSON disk fallback preserves real terminal status."""
from __future__ import annotations

import json

import pytest

from agent_runtime import TaskOutputTool
from agent_runtime.registry import get_task_output_path
from agent_runtime.types import AgentId


pytestmark = pytest.mark.asyncio


async def test_disk_fallback_returns_real_status_completed():
    """Modern JSON payload → 'completed' status, not 'archived'."""
    tool = TaskOutputTool()
    aid = AgentId("a_completed")
    get_task_output_path(aid).write_text(
        json.dumps({
            "status": "completed",
            "description": "compute the answer",
            "result": "42",
            "error": None,
            "duration_ms": 1234,
        }),
        encoding="utf-8",
    )

    r = await tool.ainvoke({"task_id": str(aid)})

    try:
        assert "status: completed" in r
        assert "archived" not in r
        assert "duration_ms: 1234" in r
        assert "42" in r
    finally:
        get_task_output_path(aid).unlink(missing_ok=True)


async def test_disk_fallback_returns_killed_with_error():
    tool = TaskOutputTool()
    aid = AgentId("a_killed")
    get_task_output_path(aid).write_text(
        json.dumps({
            "status": "killed",
            "description": "long running",
            "result": None,
            "error": None,
            "duration_ms": 500,
        }),
        encoding="utf-8",
    )

    r = await tool.ainvoke({"task_id": str(aid)})

    try:
        assert "status: killed" in r
    finally:
        get_task_output_path(aid).unlink(missing_ok=True)


async def test_disk_fallback_legacy_plaintext_returns_archived():
    """Back-compat: non-JSON files report status=archived."""
    tool = TaskOutputTool()
    aid = AgentId("a_legacy")
    get_task_output_path(aid).write_text(
        "legacy plain text result",
        encoding="utf-8",
    )

    r = await tool.ainvoke({"task_id": str(aid)})

    try:
        assert "status: archived" in r
        assert "legacy plain-text format" in r
        assert "legacy plain text result" in r
    finally:
        get_task_output_path(aid).unlink(missing_ok=True)


async def test_unknown_task_id_returns_not_found():
    tool = TaskOutputTool()
    r = await tool.ainvoke({"task_id": "a_nonexistent_xyz"})
    assert "not found" in r
