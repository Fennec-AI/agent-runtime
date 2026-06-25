"""Permission system — gate tool calls before they run.

Faithful (scoped) port of Claude Code's canUseTool / checkPermissions
flow. Before a tool runs, the dispatcher evaluates a permission decision:

    allow  → run the tool (optionally with a rewritten input)
    deny   → don't run; return an error result the model sees
    ask    → ask the embedder (a callback) — typically a y/n prompt

A permission MODE governs the policy (CC's four modes):

    default           — gated tools ask (or auto-allow if no callback wired)
    acceptEdits       — file-editing tools auto-allowed; others still ask
    plan              — read-only: gated (mutating) tools are denied
    bypassPermissions — everything allowed, no questions

Which tools are "gated" is declared on the tool itself via metadata:

    metadata = {"permission": "ask"}                 # Bash
    metadata = {"permission": "ask", "edits_files": True}  # Edit/Write

Tools without a "permission" key are ungated (always allowed) — Read,
Grep, Glob, TodoWrite, etc.

LIBRARY-FRIENDLY DEFAULT: in 'default' mode, a gated tool whose decision
is 'ask' is ALLOWED when no ask_callback is wired. The runtime is a
library first — it shouldn't deadlock waiting on a prompt that nobody is
listening for. The CLI wires an ask_callback to actually prompt; tests
and headless embedders that want hard gating use mode 'plan' or supply a
callback. (CC is interactive-first and prompts by default; this is the
one intentional divergence, documented.)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from langchain_core.tools import BaseTool


logger = logging.getLogger(__name__)


PermissionBehavior = Literal["allow", "deny", "ask"]
PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]

# (tool_name, args, reason) -> approved? The CLI implements this as a
# y/n prompt; a server might check a policy. Async so it can do I/O.
AskCallback = Callable[[str, dict, str], Awaitable[bool]]


VALID_MODES = ("default", "acceptEdits", "plan", "bypassPermissions")


@dataclass(slots=True)
class PermissionResult:
    """Outcome of a permission evaluation. After evaluate_permission, the
    behavior is only ever 'allow' or 'deny' ('ask' is resolved internally
    against the callback)."""
    behavior: PermissionBehavior
    updated_input: dict | None = None
    """If allow rewrote the tool input (rare), the new args. None = use
    the original."""
    message: str = ""
    """Human-readable reason, surfaced to the model on deny."""


def is_gated(tool: BaseTool) -> bool:
    """True if the tool opts into permission gating (metadata
    permission == 'ask')."""
    md = tool.metadata or {}
    return md.get("permission") == "ask"


def edits_files(tool: BaseTool) -> bool:
    """True if the tool mutates files (relevant to acceptEdits mode)."""
    md = tool.metadata or {}
    return bool(md.get("edits_files"))


async def evaluate_permission(
    tool: BaseTool,
    args: dict,
    mode: PermissionMode,
    ask_callback: AskCallback | None,
) -> PermissionResult:
    """Resolve a tool call to allow / deny for the given mode.

    'ask' is resolved here: if a callback is wired we await it; otherwise
    (library default) we allow. Plan mode denies gated tools outright.
    """
    if mode == "bypassPermissions":
        return PermissionResult("allow")

    # Ungated tools (reads, todo, etc.) always run.
    if not is_gated(tool):
        return PermissionResult("allow")

    # ── gated (mutating) tool from here ──────────────────────────────
    if mode == "plan":
        return PermissionResult(
            "deny",
            message=(
                f"{tool.name} is blocked in plan mode (read-only). Exit plan "
                f"mode to make changes."
            ),
        )

    if mode == "acceptEdits" and edits_files(tool):
        return PermissionResult("allow")

    # default mode (or acceptEdits for a non-edit gated tool like Bash):
    # the decision is 'ask'. Resolve via the callback if present.
    if ask_callback is None:
        # No one to ask — library-friendly default is to allow. Log so it's
        # visible that a gated tool ran without confirmation.
        logger.debug(
            "permission: %s gated but no ask_callback wired (mode=%s) — "
            "allowing", tool.name, mode,
        )
        return PermissionResult("allow")

    reason = f"{tool.name} wants to run"
    try:
        approved = await ask_callback(tool.name, args, reason)
    except Exception:  # noqa: BLE001 — a broken prompt must not crash the loop
        logger.warning("ask_callback raised; denying %s", tool.name, exc_info=True)
        return PermissionResult("deny", message="permission prompt failed")

    if approved:
        return PermissionResult("allow")
    return PermissionResult(
        "deny", message=f"User declined permission to run {tool.name}."
    )
