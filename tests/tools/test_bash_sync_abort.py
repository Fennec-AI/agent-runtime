"""Bash sync mode — abort-aware paths + process-group kills.

These are the regressions found in e2e scenario 9 + 51813fe (abort-aware)
+ 5d6740e (process-group kills). Each test asserts on TIMING because
the whole point is that aborts and kills are FAST now.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from agent_runtime import BashTool


pytestmark = pytest.mark.asyncio


async def test_natural_completion_unchanged(run_context, with_run_context):
    bash = BashTool()
    with with_run_context:
        r = await bash.ainvoke({"command": "echo hello"})
    assert "exit_code: 0" in r
    assert "hello" in r


async def test_timeout_kills_subprocess(run_context, with_run_context):
    bash = BashTool()
    t0 = time.monotonic()
    with with_run_context:
        r = await bash.ainvoke({"command": "sleep 30", "timeout": 300})
    elapsed = time.monotonic() - t0
    assert "timed out" in r
    assert elapsed < 3.0, f"timeout-then-kill took {elapsed:.2f}s"


async def test_fast_fail_when_already_aborted(run_context, with_run_context):
    """Pre-call abort → no subprocess spawned, instant return."""
    bash = BashTool()
    with with_run_context as ctx:
        ctx.abort.abort("already_done")
        t0 = time.monotonic()
        r = await bash.ainvoke({"command": "echo PROVE-I-RAN"})
        elapsed = time.monotonic() - t0
    assert "aborted before start" in r
    assert "already_done" in r
    assert "exit_code" not in r                         # subprocess never ran
    assert elapsed < 0.2, f"fast-fail wasn't fast: {elapsed:.2f}s"


async def test_abort_mid_execution_kills_subprocess_fast(run_context, with_run_context):
    """Process-group kill: sh-wraps-sleep dies as soon as abort fires."""
    bash = BashTool()

    async def fire_abort_at(delay: float):
        await asyncio.sleep(delay)
        run_context.abort.abort("mid_run")

    asyncio.create_task(fire_abort_at(0.2))
    t0 = time.monotonic()
    with with_run_context:
        # sh -c "sleep 30; echo never" — pre-5d6740e, SIGTERM on sh
        # left sleep running for full 30s. With killpg, dies in <0.3s.
        r = await bash.ainvoke({"command": "sleep 30; echo never"})
    elapsed = time.monotonic() - t0

    assert "aborted by parent" in r
    assert "mid_run" in r
    assert elapsed < 1.0, (
        f"sh-wraps-sleep took {elapsed:.2f}s to die — process-group "
        f"kill regressed"
    )


async def test_outer_task_cancel_terminates_subprocess(run_context, with_run_context):
    """asyncio.Task.cancel() inside `finally:` SIGTERMs the subprocess."""
    bash = BashTool()
    with with_run_context:
        task = asyncio.create_task(
            bash.ainvoke({"command": "sleep 30; echo never"})
        )
        await asyncio.sleep(0.2)                        # let subprocess start
        t0 = time.monotonic()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        elapsed = time.monotonic() - t0
    # The finally block awaits asyncio.shield(_terminate_proc) — should
    # be <1s now that killpg reaches sleep.
    assert elapsed < 1.0, f"outer cancel → finally took {elapsed:.2f}s"
