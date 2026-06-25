"""Session — the session-level facade.

Wraps query() with multi-turn state management. The MAIN agent uses
Session; subagents skip it and go straight to query() (one-shot
lifecycle, no multi-turn state to manage).
"""
from __future__ import annotations

import logging
from contextlib import aclosing
from typing import AsyncGenerator, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, BaseMessageChunk, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .cancellation import AbortController
from .cleanup_registry import run_all_cleanups
from .query import query
from .registry import AgentEntry, new_agent_id, register, unregister
from .permissions import AskCallback, PermissionMode
from .structured_output import (
    MAX_STRUCTURED_OUTPUT_RETRIES,
    STRUCTURED_OUTPUT_NUDGE,
    STRUCTURED_OUTPUT_TOOL_NAME,
    make_structured_output_tool,
)
from .types import AgentId, RunContext


logger = logging.getLogger(__name__)


# Bound TypeVar so run_structured() returns the EXACT model type the
# caller passed in (mirrors LangChain's with_structured_output typing).
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class Session:
    """Session facade. One instance per user session.

    Holds long-lived state (messages, abort, file cache) and exposes
    `submit_message(prompt)` as the single public entry point.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        custom_system_prompt: str | None = None,
        permission_mode: "PermissionMode" = "default",
        ask_callback: "AskCallback | None" = None,
    ) -> None:
        self.session_id: AgentId = new_agent_id()
        self.messages: list[BaseMessage] = []
        self.tools = tools
        self.llm = llm
        self.custom_system_prompt = custom_system_prompt
        self.abort = AbortController()

        # System prompt goes at the start of messages as a SystemMessage.
        system_prompt = self._build_system_prompt()
        if system_prompt:
            self.messages.append(SystemMessage(content=system_prompt))

        # Build the session's RunContext ONCE and reuse it across turns.
        # ctx.messages is the same list as self.messages, so appending the
        # next user turn to self.messages is visible to query() automatically.
        self.ctx = RunContext(
            agent_id=self.session_id,
            messages=self.messages,
            abort=self.abort,
            tools=self.tools,
            llm=self.llm,
            depth=0,
            parent_id=None,
            permission_mode=permission_mode,
            ask_callback=ask_callback,
        )

        # Register the main agent in the global registry so background
        # children can push notifications to its inbox.
        register(AgentEntry(
            agent_id=self.session_id,
            abort=self.abort,
            task=None,                              # main isn't itself a Task
            ctx=self.ctx,
            description="main session",
        ))

    # ── Public API ───────────────────────────────────────────────────────
    async def submit_message(
        self,
        prompt: str,
    ) -> AsyncGenerator[BaseMessageChunk | BaseMessage, None]:
        """Process one user message. Yields streaming chunks and complete
        messages until the loop exits.

        Typical usage:
            async for event in engine.submit_message("hello"):
                if isinstance(event, AIMessageChunk):
                    print(event.content, end="", flush=True)
                elif isinstance(event, AIMessage):
                    pass  # full message — already in self.messages
                elif isinstance(event, ToolMessage):
                    pass  # one tool result completed
        """
        self.messages.append(HumanMessage(content=prompt))

        # query() mutates ctx.messages directly; since ctx.messages IS
        # self.messages, the session history stays current automatically.
        async for event in query(self.ctx):
            yield event

    async def run_structured(
        self,
        prompt: str,
        output_schema: type[_ModelT],
    ) -> _ModelT | None:
        """Run one turn and return the final answer as a typed Pydantic model.

        Faithful to Claude Code's StructuredOutput (SyntheticOutputTool)
        mechanism: we add a tool whose input schema IS `output_schema`, and
        the model produces its typed answer by CALLING that tool once at the
        end — not by writing free text we then parse. Single pass; no extra
        coercion LLM call.

        The agent runs its normal loop — tools, subagents, inbox drains —
        and finishes by calling StructuredOutput. We capture the validated
        instance the moment that happens and stop. If the model ends a turn
        without calling it, we nudge and retry up to
        MAX_STRUCTURED_OUTPUT_RETRIES times. Returns None if aborted or if
        the model never produced valid structured output.

        SCOPING — MAIN-AGENT-ONLY by construction: the StructuredOutput tool
        is added only to THIS session's tool list (removed in `finally`).
        Subagents build their own tool lists from their SubagentDefinition
        and never see it, so spawned children are completely unaffected.

        Example:
            class Weather(BaseModel):
                city: str
                temp_c: float

            w = await session.run_structured("Weather in Paris?", Weather)
            assert isinstance(w, Weather)

        The streaming `submit_message` path is unchanged; use that for raw
        text + live events, this for a typed object.
        """
        capture: dict[str, _ModelT] = {}
        tool = make_structured_output_tool(output_schema, capture)

        # Add the tool to the (main-agent) tool list. ctx.tools IS self.tools,
        # so query() will bind it. Removed in finally so the session is clean
        # afterward and the tool never lingers into later turns.
        self.tools.append(tool)
        try:
            # First turn carries the real prompt plus an explicit instruction
            # to finish via the tool (reinforces the tool's own description).
            first = (
                f"{prompt}\n\nWhen you have finished, you MUST call the "
                f"{STRUCTURED_OUTPUT_TOOL_NAME} tool exactly once with your "
                f"final answer in the required structured format."
            )
            await self._drive_until_captured(first, capture)

            # Nudge-and-retry if the model ended without calling the tool.
            retries = 0
            while (
                "value" not in capture
                and not self.abort.aborted
                and retries < MAX_STRUCTURED_OUTPUT_RETRIES
            ):
                retries += 1
                await self._drive_until_captured(STRUCTURED_OUTPUT_NUDGE, capture)

            if self.abort.aborted and "value" not in capture:
                logger.info("run_structured: aborted before structured output")
                return None
            return capture.get("value")
        finally:
            if tool in self.tools:
                self.tools.remove(tool)

    async def _drive_until_captured(
        self,
        prompt: str,
        capture: dict,
    ) -> None:
        """Submit `prompt` and drive the loop until the StructuredOutput tool
        fires (capture populated) or the turn ends. Stops the moment the
        value is captured so we don't burn an extra model call."""
        async with aclosing(self.submit_message(prompt)) as stream:
            async for _event in stream:
                if "value" in capture:
                    break

    def cancel(self, reason: str = "user_cancel") -> None:
        """User pressed ESC or equivalent. Aborts everything reachable from
        this session's abort controller (including all sync subagents)."""
        self.abort.abort(reason)

    def close(self) -> None:
        """Synchronous close — removes the session from the global
        registry but does NOT run shutdown cleanups (no async context).

        Prefer `aclose()` in an async context so background agents get
        killed cleanly before the process exits.
        """
        unregister(self.session_id)

    async def aclose(self) -> None:
        """Async close — runs every registered cleanup (kills orphan
        background agents) then unregisters the session itself.

        Idempotent: safe to call multiple times. Call this from any
        async finally block (e.g., CLI shutdown, FastAPI lifespan).
        """
        try:
            await run_all_cleanups()
        finally:
            self.close()

    # ── Private helpers ──────────────────────────────────────────────────
    def _build_system_prompt(self) -> str:
        """Combine system prompt sources.

        Returns the custom prompt or a minimal default. Extension point
        for full assembly logic later (tool descriptions are auto-injected
        by bind_tools(); dynamic context like cwd/time, MCP server info,
        memory mechanics prompt would go here).
        """
        if self.custom_system_prompt:
            return self.custom_system_prompt
        return (
            "You are a helpful assistant. Use the available tools when "
            "appropriate to answer the user's questions."
        )
