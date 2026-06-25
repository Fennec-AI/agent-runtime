"""Permission system — modes, gating, and enforcement in the dispatcher."""
from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from agent_runtime import BashTool, ReadTool, EditTool, WriteTool
from agent_runtime.permissions import evaluate_permission, is_gated, edits_files
from agent_runtime.tool_executor import _execute_tool
from agent_runtime.types import AgentId, RunContext
from agent_runtime.cancellation import AbortController


pytestmark = pytest.mark.asyncio


# ── tool classification ───────────────────────────────────────────────

async def test_gating_classification():
    assert is_gated(BashTool()) is True
    assert is_gated(EditTool()) is True
    assert is_gated(WriteTool()) is True
    assert is_gated(ReadTool()) is False
    assert edits_files(EditTool()) is True
    assert edits_files(WriteTool()) is True
    assert edits_files(BashTool()) is False    # bash is gated but not an edit


# ── evaluate_permission across modes ──────────────────────────────────

async def test_ungated_tool_always_allowed():
    for mode in ("default", "acceptEdits", "plan", "bypassPermissions"):
        r = await evaluate_permission(ReadTool(), {}, mode, None)
        assert r.behavior == "allow", mode


async def test_bypass_allows_everything():
    r = await evaluate_permission(BashTool(), {"command": "rm -rf /"},
                                  "bypassPermissions", None)
    assert r.behavior == "allow"


async def test_plan_mode_denies_gated_tools():
    r = await evaluate_permission(BashTool(), {"command": "ls"}, "plan", None)
    assert r.behavior == "deny"
    assert "plan mode" in r.message.lower()


async def test_accept_edits_allows_edits_but_bash_still_asks():
    # Edit auto-allowed in acceptEdits
    r_edit = await evaluate_permission(EditTool(), {}, "acceptEdits", None)
    assert r_edit.behavior == "allow"
    # Bash is gated but not an edit → falls through to ask; no callback → allow
    r_bash = await evaluate_permission(BashTool(), {}, "acceptEdits", None)
    assert r_bash.behavior == "allow"           # no callback → library default
    # ...but WITH a denying callback, bash is denied even in acceptEdits
    async def deny_cb(name, args, reason): return False
    r_bash2 = await evaluate_permission(BashTool(), {}, "acceptEdits", deny_cb)
    assert r_bash2.behavior == "deny"


async def test_default_mode_no_callback_allows():
    r = await evaluate_permission(BashTool(), {}, "default", None)
    assert r.behavior == "allow"


async def test_default_mode_callback_decides():
    async def approve(name, args, reason): return True
    async def decline(name, args, reason): return False

    assert (await evaluate_permission(BashTool(), {}, "default", approve)).behavior == "allow"
    denied = await evaluate_permission(BashTool(), {}, "default", decline)
    assert denied.behavior == "deny"
    assert "declined" in denied.message.lower()


async def test_broken_callback_denies_safely():
    async def boom(name, args, reason): raise RuntimeError("prompt crashed")
    r = await evaluate_permission(BashTool(), {}, "default", boom)
    assert r.behavior == "deny"


# ── enforcement: a denied tool does NOT run ───────────────────────────

def _ctx(mode, ask_callback=None) -> RunContext:
    return RunContext(
        agent_id=AgentId("a_test"), messages=[], abort=AbortController(),
        tools=[], llm=None, depth=0,
        permission_mode=mode, ask_callback=ask_callback,
    )


async def test_denied_bash_does_not_execute(tmp_path: Path):
    """plan mode blocks Bash — the side effect must NOT happen."""
    marker = tmp_path / "should_not_exist.txt"
    tc = {"id": "c1", "name": "Bash",
          "args": {"command": f"touch {marker}"}}
    msg = await _execute_tool(tc, BashTool(), _ctx("plan"))
    assert msg.status == "error"
    assert "Permission denied" in msg.content
    assert not marker.exists(), "denied Bash still ran its command!"


async def test_allowed_bash_executes_in_default_mode(tmp_path: Path):
    marker = tmp_path / "created.txt"
    tc = {"id": "c2", "name": "Bash",
          "args": {"command": f"touch {marker}"}}
    msg = await _execute_tool(tc, BashTool(), _ctx("default"))
    assert msg.status != "error"                # ran successfully
    assert marker.exists()


async def test_declining_callback_blocks_execution(tmp_path: Path):
    marker = tmp_path / "declined.txt"
    async def decline(name, args, reason): return False
    tc = {"id": "c3", "name": "Bash",
          "args": {"command": f"touch {marker}"}}
    msg = await _execute_tool(tc, BashTool(), _ctx("default", decline))
    assert msg.status == "error"
    assert not marker.exists()
