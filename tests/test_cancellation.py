"""AbortController + create_child_controller — cancellation primitive."""
from __future__ import annotations

import asyncio

import pytest

from agent_runtime import AbortController, create_child_controller


pytestmark = pytest.mark.asyncio


async def test_abort_flips_aborted_flag_and_sets_reason():
    a = AbortController()
    assert a.aborted is False
    assert a.reason is None

    a.abort("user_pressed_escape")

    assert a.aborted is True
    assert a.reason == "user_pressed_escape"


async def test_signal_wake_after_abort():
    """Coroutines parked on signal.wait() wake when abort fires."""
    a = AbortController()

    async def waiter():
        await a.signal.wait()
        return a.reason

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)                              # yield
    assert not task.done()

    a.abort("done")
    result = await asyncio.wait_for(task, timeout=1.0)
    assert result == "done"


async def test_child_aborts_when_parent_aborts():
    parent = AbortController()
    child = create_child_controller(parent)

    assert not child.aborted
    parent.abort("parent_reason")
    await asyncio.sleep(0.05)                           # propagation watcher

    assert child.aborted
    assert child.reason == "parent_reason"


async def test_child_created_after_parent_aborted_is_immediately_aborted():
    """Fast-path: parent already fired before child existed."""
    parent = AbortController()
    parent.abort("already_done")

    child = create_child_controller(parent)

    assert child.aborted
    assert child.reason == "already_done"


async def test_child_can_be_aborted_independently():
    """Aborting child doesn't ripple to parent."""
    parent = AbortController()
    child = create_child_controller(parent)

    child.abort("child_only")

    assert child.aborted
    assert not parent.aborted
