"""Session.run_structured — structured output for the MAIN agent only.

The happy path (real coercion via with_structured_output) is validated
end-to-end against a live model in /tmp/e2e_test/structured_demo.py —
it can't be faithfully unit-tested because FakeMessagesListChatModel
doesn't honor with_structured_output's forced tool-calling.

What we CAN unit-test without a key:
  • the abort short-circuit returns None (query() exits at its abort
    check before any LLM call, so no real model is needed)
  • the method signature / typing contract exists
  • subagents have no structured-output machinery (scoping guarantee)
"""
from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel

from agent_runtime import Session


pytestmark = pytest.mark.asyncio


class _Answer(BaseModel):
    value: int


class _DummyLLM:
    """Minimal stand-in. With abort pre-set, query() returns at its abort
    check (§4) before binding tools or streaming (§5), so neither of these
    is actually called."""

    def bind_tools(self, _tools):
        return self

    async def astream(self, _messages):
        raise AssertionError("astream must NOT be called on an aborted run")
        yield  # pragma: no cover — makes this an async generator


async def test_run_structured_returns_none_when_aborted_before_run():
    session = Session(llm=_DummyLLM(), tools=[])
    try:
        session.cancel("pre_abort")                    # abort before running
        result = await session.run_structured("anything", _Answer)
        assert result is None
    finally:
        session.close()


async def test_run_structured_exists_with_expected_signature():
    sig = inspect.signature(Session.run_structured)
    params = list(sig.parameters)
    assert params == ["self", "prompt", "output_schema"]
    assert inspect.iscoroutinefunction(Session.run_structured)


async def test_subagents_have_no_structured_output_field():
    """Scoping guarantee: structured output lives only on Session, NOT on
    RunContext — so SpawnAgentTool's child contexts can't carry it and
    subagents are unaffected by construction."""
    from agent_runtime.types import RunContext
    field_names = {f for f in RunContext.__dataclass_fields__}
    assert "output_schema" not in field_names, (
        "structured output leaked into RunContext — it would then reach "
        "subagents, which must stay unaffected"
    )
