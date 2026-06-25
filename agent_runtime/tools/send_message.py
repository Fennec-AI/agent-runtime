"""SendMessageTool — push a message into a running agent's inbox.

Mirrors Claude Code's `SendMessage` tool surface (simplified — we don't
have teammate-name addressing or broadcast targets yet, just agent_id).

Use case:
  The parent spawned a background subagent with run_in_background=True.
  Later the parent wants to clarify the prompt, supply new info, or
  redirect the work. SendMessage drops a message into the child's
  pendingMessages queue; the child drains it at the top of its next
  loop iteration and treats it as a user message.

Schema mirrors CC closely:
    to:       agent_id of recipient (CC also supports teammate names,
              "*" for broadcast, "uds:..." / "bridge:..." for remote peers
              — deferred for our scope)
    message:  plain text content
    summary:  optional 5-10 word preview (CC uses this for the UI)

Concurrency-safe — pure registry mutation, no I/O. Multiple SendMessage
calls in one turn fan out under the dispatcher's safe-batch path.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..registry import AGENT_REGISTRY
from ..types import AgentId


logger = logging.getLogger(__name__)


class SendMessageInput(BaseModel):
    """Arguments for SendMessage. Mirrors Claude Code's schema."""

    to: str = Field(
        description=(
            "Recipient — the agent_id of a running (typically background) "
            "agent. Get the id from the previous SpawnAgent call's response."
        ),
    )
    message: str = Field(
        description="Plain-text message content for the recipient.",
    )
    summary: str | None = Field(
        default=None,
        description=(
            "Optional 5-10 word preview of the message (shown in UI / logs)."
        ),
    )


class SendMessageTool(BaseTool):
    """Send a message to a running agent."""

    name: str = "SendMessage"
    description: str = (
        "Send a message into a running agent's inbox. The recipient drains "
        "its inbox at the top of its next loop iteration and treats the "
        "message as a user message — use this to redirect, clarify, or "
        "supply new info to a background subagent without re-spawning. "
        "Pass the agent_id (from the SpawnAgent response) as `to`."
    )
    args_schema: type[BaseModel] = SendMessageInput
    metadata: dict[str, Any] = {"concurrency_safe": True}

    async def _arun(
        self,
        to: str,
        message: str,
        summary: str | None = None,
    ) -> str:
        recipient_id = AgentId(to)
        entry = AGENT_REGISTRY.get(recipient_id)
        if entry is None:
            return (
                f"Recipient {to!r} not found. It may have already finished "
                f"and been cleaned up, or the id is invalid."
            )
        if entry.status != "running":
            return (
                f"Recipient {to!r} is in terminal state {entry.status!r}; "
                f"cannot receive new messages."
            )

        # Need parent id (the sender) so the recipient can attribute the
        # message in its transcript. Read it from the dispatcher's ContextVar.
        from ..tool_executor import CURRENT_RUN_CONTEXT
        sender_ctx = CURRENT_RUN_CONTEXT.get(None)
        sender_id = sender_ctx.agent_id if sender_ctx is not None else "unknown"

        # Format the inbox entry so the receiver knows it's a directed
        # message (not just any drain target). The receiver's query loop
        # treats this as a user message verbatim.
        summary_tag = f'\n<summary>{summary}</summary>' if summary else ""
        formatted = (
            f"<message from=\"{sender_id}\">"
            f"{summary_tag}"
            f"\n<content>{message}</content>"
            f"\n</message>"
        )

        entry.pending_messages.append(formatted)
        logger.info(
            "SendMessage: %s → %s (%d chars)",
            sender_id, recipient_id, len(message),
        )

        preview = summary if summary else (message[:60] + ("…" if len(message) > 60 else ""))
        return f"Message sent to {to!r}: {preview!r}"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("SendMessageTool is async-only; use ainvoke().")
