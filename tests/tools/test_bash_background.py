"""Bash background mode + BashOutput + KillShell — end-to-end lifecycle.

Covers:
  - register, drain, status transitions
  - cursor-based BashOutput (only new content per call)
  - filter regex
  - KillShell via process-group
  - SIGKILL escalation for SIGTERM-ignoring children
"""
from __future__ import annotations

import asyncio
import re
import time

import pytest

from agent_runtime import BashTool, BashOutputTool, KillShellTool
from agent_runtime.bash_shells import get_shell, kill_shell, read_new_output


pytestmark = pytest.mark.asyncio


SHELL_ID_RE = re.compile(r"shell (sh_[a-f0-9]+) ")


def _extract_shell_id(spawn_response: str) -> str:
    m = SHELL_ID_RE.search(spawn_response)
    assert m, f"shell_id not in spawn response: {spawn_response!r}"
    return m.group(1)


async def test_bg_spawn_registers_and_streams_output():
    bash = BashTool()
    r = await bash.ainvoke({
        "command": "echo hello-bg; sleep 0.2; echo done",
        "run_in_background": True,
    })
    sid = _extract_shell_id(r)
    assert get_shell(sid).status == "running"

    await asyncio.sleep(0.5)
    new_out, dropped = await read_new_output(sid, "stdout")
    assert b"hello-bg" in new_out
    assert b"done" in new_out
    assert dropped == 0


async def test_cursor_advances_between_polls():
    """Second read returns only NEW bytes appended since the first."""
    bash = BashTool()
    r = await bash.ainvoke({
        "command": "echo first; sleep 0.3; echo second",
        "run_in_background": True,
    })
    sid = _extract_shell_id(r)

    await asyncio.sleep(0.1)
    first_out, _ = await read_new_output(sid, "stdout")
    assert b"first" in first_out

    await asyncio.sleep(0.4)
    second_out, _ = await read_new_output(sid, "stdout")
    assert b"second" in second_out
    assert b"first" not in second_out, "cursor didn't advance"


async def test_killshell_terminates_running_shell_fast():
    """KillShell on `sleep 30; echo` dies via killpg in ~0ms (not ~2s)."""
    bash = BashTool()
    kill = KillShellTool()
    r = await bash.ainvoke({
        "command": "sleep 30; echo never",
        "run_in_background": True,
    })
    sid = _extract_shell_id(r)
    assert get_shell(sid).status == "running"

    t0 = time.monotonic()
    msg = await kill.ainvoke({"shell_id": sid})
    elapsed = time.monotonic() - t0

    assert "Killed" in msg
    assert get_shell(sid).status == "killed"
    assert elapsed < 0.5, f"killpg slow: {elapsed:.2f}s"


async def test_killshell_idempotent_on_terminal():
    bash = BashTool()
    kill = KillShellTool()
    r = await bash.ainvoke({"command": "sleep 30", "run_in_background": True})
    sid = _extract_shell_id(r)
    await kill.ainvoke({"shell_id": sid})

    msg2 = await kill.ainvoke({"shell_id": sid})
    assert "already in terminal state" in msg2


async def test_killshell_unknown_id_returns_error():
    kill = KillShellTool()
    msg = await kill.ainvoke({"shell_id": "sh_doesnt_exist"})
    assert "not found" in msg


async def test_sigkill_escalation_for_sigterm_ignoring_child():
    """A Python child that SIG_IGNs SIGTERM forces the SIGKILL fallback."""
    bash = BashTool()
    r = await bash.ainvoke({
        "command": (
            "python3 -c \"import signal,time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)\""
        ),
        "run_in_background": True,
    })
    sid = _extract_shell_id(r)
    await asyncio.sleep(0.3)                            # let handler install

    t0 = time.monotonic()
    await kill_shell(sid)
    elapsed = time.monotonic() - t0

    entry = get_shell(sid)
    assert entry.status == "killed"
    assert entry.exit_code == -9                        # SIGKILL signal
    assert 1.5 < elapsed < 4.0, (
        f"SIGKILL escalation timing off: {elapsed:.2f}s "
        "(expected ~2s SIGTERM grace + fast SIGKILL)"
    )


async def test_bash_output_filter_regex():
    bash = BashTool()
    poll = BashOutputTool()
    r = await bash.ainvoke({
        "command": "for i in 1 2 3 4 5; do echo line-$i; done; echo target",
        "run_in_background": True,
    })
    sid = _extract_shell_id(r)
    await asyncio.sleep(0.3)

    msg = await poll.ainvoke({"bash_id": sid, "filter": "target"})
    assert "target" in msg
    assert "line-1" not in msg
