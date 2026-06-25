"""Renderers — turn agent events into something a transport can send.

A Renderer is a transformer: it consumes the stream of events from
Session.submit_message() and yields strings (or dicts, or whatever the
caller needs). Each transport (CLI, SSE, WebSocket) picks the renderer
whose output shape matches.

This separation keeps the runtime untouched by display concerns.
"""

from .base import Renderer
from .cli import CLIRenderer, print_subagent_event

__all__ = ["Renderer", "CLIRenderer", "print_subagent_event"]
