"""Shared pytest fixtures + state cleanup.

Every test gets a clean slate: registries cleared, hooks cleared, any
pending shutdown cleanups drained. This matters because our runtime
uses module-level mutable state (AGENT_REGISTRY, SHELL_REGISTRY,
_cleanup_functions, _hooks) by design — between tests we have to
explicitly reset.

The `_state_reset` fixture is autouse so it applies to every test
without opt-in.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from agent_runtime import AGENT_REGISTRY
from agent_runtime.bash_shells import SHELL_REGISTRY
from agent_runtime.cleanup_registry import run_all_cleanups
from agent_runtime.hooks import clear_hooks


@pytest.fixture(autouse=True)
async def _state_reset() -> AsyncIterator[None]:
    """Drain registries + hooks + cleanups before AND after each test.

    Before: paranoia for tests that started before a registry was
    cleared (e.g. a previous test crashed).
    After: hygiene — leave the world the way we found it.
    """
    # Pre-test: snapshot empty
    AGENT_REGISTRY.clear()
    SHELL_REGISTRY.clear()
    clear_hooks()
    await run_all_cleanups()

    yield

    # Post-test: same thing
    await run_all_cleanups()
    AGENT_REGISTRY.clear()
    SHELL_REGISTRY.clear()
    clear_hooks()


@pytest.fixture
def run_context():
    """A bare RunContext that tools can be invoked under.

    Use when you need to set CURRENT_RUN_CONTEXT for a tool that reads
    it (Bash, SpawnAgent, SendMessage). The abort controller is fresh
    each test; trigger it via `run_context.abort.abort("...")`.
    """
    from agent_runtime.cancellation import AbortController
    from agent_runtime.types import AgentId, RunContext
    return RunContext(
        agent_id=AgentId("a_test"),
        messages=[],
        abort=AbortController(),
        tools=[],
        llm=None,
        depth=0,
    )


@pytest.fixture
def with_run_context(run_context):
    """Context manager that installs `run_context` into CURRENT_RUN_CONTEXT.

    Yields the run_context so you can mutate its abort flag. Example:

        async def test_x(with_run_context):
            with with_run_context as ctx:
                ctx.abort.abort("test")
                ...
    """
    from contextlib import contextmanager

    from agent_runtime.tool_executor import CURRENT_RUN_CONTEXT

    @contextmanager
    def _cm():
        token = CURRENT_RUN_CONTEXT.set(run_context)
        try:
            yield run_context
        finally:
            CURRENT_RUN_CONTEXT.reset(token)

    return _cm()
