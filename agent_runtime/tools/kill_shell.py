"""KillShell — terminate a running background shell by ID.

Mirrors Claude Code's `KillShell` schema:

    shell_id: str  — the id returned by Bash(run_in_background=True)

Underneath, delegates to bash_shells.kill_shell(): SIGTERM, wait 2s,
then SIGKILL fallback. The entry stays in SHELL_REGISTRY after the
kill so a subsequent BashOutput can still retrieve the final tail and
exit code.

Idempotent — calling on an already-terminal shell returns a clear
"already finished" message rather than an error.

Concurrency-safe — registry mutation is lock-protected; multiple
KillShell calls on the same id collapse to one signal (the second
call sees status != 'running' and returns early).
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..bash_shells import SHELL_REGISTRY, ShellId, kill_shell


logger = logging.getLogger(__name__)


class KillShellInput(BaseModel):
    """Arguments for KillShell. Mirrors Claude Code's schema."""

    shell_id: str = Field(
        description=(
            "The shell_id returned by Bash(run_in_background=True). "
            "Get it from the spawn response."
        ),
    )


class KillShellTool(BaseTool):
    """Kill a running background shell."""

    name: str = "KillShell"
    description: str = (
        "Terminate a running background shell (started by "
        "Bash(run_in_background=True)) by its shell_id. Sends SIGTERM, "
        "waits 2 seconds, then escalates to SIGKILL if still running. "
        "The shell entry remains queryable via BashOutput after kill."
    )
    args_schema: type[BaseModel] = KillShellInput
    metadata: dict[str, Any] = {"concurrency_safe": True}

    async def _arun(self, shell_id: str) -> str:
        sid = ShellId(shell_id)
        entry = SHELL_REGISTRY.get(sid)
        if entry is None:
            return (
                f"Shell {shell_id!r} not found. The id may be invalid, "
                f"or the shell may have been evicted from the registry."
            )
        if entry.status != "running":
            return (
                f"Shell {shell_id!r} is already in terminal state "
                f"{entry.status!r} (exit_code={entry.exit_code}). "
                f"No signal sent."
            )

        signalled = await kill_shell(sid, reason="killed_by_KillShell")
        if not signalled:
            # Race: status flipped to terminal between our check and the call.
            return (
                f"Shell {shell_id!r} reached terminal state before the "
                f"kill could be sent (status={entry.status!r})."
            )
        return (
            f"Killed shell {shell_id!r} (status={entry.status!r}, "
            f"exit_code={entry.exit_code})."
        )

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("KillShellTool is async-only; use ainvoke().")
