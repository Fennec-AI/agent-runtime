"""Hooks pipeline — registration, fan-out, error containment."""
from __future__ import annotations

import asyncio

import pytest

from agent_runtime import register_hook, clear_hooks, hook_count
from agent_runtime import hooks


pytestmark = pytest.mark.asyncio


async def test_register_returns_unregister_callable():
    """register_hook returns a callable that idempotently removes the hook."""
    called: list[str] = []
    unreg = register_hook("test_event", lambda: called.append("hi"))
    assert hook_count("test_event") == 1

    unreg()
    assert hook_count("test_event") == 0

    # Second call — idempotent, no exception
    unreg()
    assert hook_count("test_event") == 0


async def test_multiple_hooks_fire_in_registration_order():
    seen: list[int] = []
    register_hook("e", lambda: seen.append(1))
    register_hook("e", lambda: seen.append(2))
    register_hook("e", lambda: seen.append(3))

    await hooks.fire("e")

    assert seen == [1, 2, 3]


async def test_async_and_sync_hooks_compose():
    """Mixed sync + async hooks all fire under fire()."""
    seen: list[str] = []

    def sync_hook():
        seen.append("sync")

    async def async_hook():
        await asyncio.sleep(0)
        seen.append("async")

    register_hook("e", sync_hook)
    register_hook("e", async_hook)
    await hooks.fire("e")

    assert set(seen) == {"sync", "async"}


async def test_hook_exception_does_not_break_others():
    """A raising hook is caught + logged; subsequent hooks still fire."""
    seen: list[str] = []

    def good_one():
        seen.append("first")

    def bomb():
        raise RuntimeError("from hook")

    def good_two():
        seen.append("third")

    register_hook("e", good_one)
    register_hook("e", bomb)
    register_hook("e", good_two)

    await hooks.fire("e")                               # must NOT raise

    assert seen == ["first", "third"]


async def test_fire_unknown_event_is_noop():
    """Firing an event with no hooks is silent and fast."""
    await hooks.fire("nonexistent_event", foo=1, bar=2)


async def test_hook_payload_kwargs_pass_through():
    seen: dict = {}

    def capture(**kwargs):
        seen.update(kwargs)

    register_hook("e", capture)
    await hooks.fire("e", agent_id="a1", tool_name="Bash", status="success")

    assert seen == {"agent_id": "a1", "tool_name": "Bash", "status": "success"}


async def test_clear_hooks_removes_all_for_event():
    register_hook("a", lambda: None)
    register_hook("a", lambda: None)
    register_hook("b", lambda: None)

    clear_hooks("a")
    assert hook_count("a") == 0
    assert hook_count("b") == 1

    clear_hooks()                                       # all
    assert hook_count() == 0


async def test_hook_count_total_vs_per_event():
    register_hook("a", lambda: None)
    register_hook("a", lambda: None)
    register_hook("b", lambda: None)

    assert hook_count() == 3
    assert hook_count("a") == 2
    assert hook_count("b") == 1
    assert hook_count("nope") == 0
