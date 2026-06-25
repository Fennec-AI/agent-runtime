"""Renderer Protocol — the shared shape every renderer conforms to.

A renderer transforms a stream of agent events (LangChain messages and
chunks) into a stream of output strings. The output format depends on the
renderer:
  - CLIRenderer    → ANSI-colored terminal text
  - SSERenderer    → "data: {...}\\n\\n" SSE-formatted strings (later)
  - WSRenderer     → JSON-serializable dicts the caller sends as ws.send_json()

The runtime never needs to know which renderer is in use; the renderer is
purely the transport adapter.
"""
from __future__ import annotations

from typing import AsyncGenerator, Protocol

from langchain_core.messages import BaseMessage, BaseMessageChunk


class Renderer(Protocol):
    """Transforms agent events into output strings.

    Implementations are themselves async generators — composes naturally
    into FastAPI's StreamingResponse, CLI loops, websockets, etc.
    """

    def render(
        self,
        events: AsyncGenerator[BaseMessageChunk | BaseMessage, None],
    ) -> AsyncGenerator[str, None]:
        """Iterate `events`, yield output strings.

        Args:
            events: The async generator returned by
                `Session.submit_message(prompt)`.

        Yields:
            Output strings appropriate for the target transport.
        """
        ...
