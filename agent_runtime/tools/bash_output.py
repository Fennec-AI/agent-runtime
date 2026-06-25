"""BashOutput — poll a persistent background shell for new output.

Companion to Bash(run_in_background=True) and KillShell. Mirrors Claude
Code's `BashOutput` schema:

    bash_id: str                — shell to poll
    filter:  str | None = None  — optional regex; only matching lines
                                  are returned (applied per-line to
                                  stdout AND stderr)

Returns a structured plain-text report with:
  • shell_id, command, status, exit_code (if terminal)
  • new stdout since last BashOutput call (with bytes-dropped tally)
  • new stderr since last BashOutput call (with bytes-dropped tally)

The "new since last call" semantics come from the cursor each shell
entry maintains — every BashOutput call advances the cursor to the end
of the buffer, so the next call sees only fresh content.

Concurrency-safe — a pure poll, no writes outside the per-shell read
cursor (which is itself lock-protected).
"""
from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..bash_shells import SHELL_REGISTRY, read_new_output
from ..bash_shells import ShellId


logger = logging.getLogger(__name__)


class BashOutputInput(BaseModel):
    """Arguments for BashOutput. Mirrors Claude Code's schema."""

    bash_id: str = Field(
        description=(
            "The shell_id returned by Bash(run_in_background=True)."
        ),
    )
    filter: str | None = Field(
        default=None,
        description=(
            "Optional regex. When set, only lines matching the regex are "
            "returned (applied to BOTH stdout and stderr, line by line). "
            "Useful for fishing the interesting output out of a noisy log."
        ),
    )


class BashOutputTool(BaseTool):
    """Poll a background shell's new output by ID."""

    name: str = "BashOutput"
    description: str = (
        "Read new output from a background shell spawned with "
        "Bash(run_in_background=True). Returns only output appended "
        "since the previous BashOutput call (cursor-based). Includes "
        "status (running/exited/killed) and exit_code when terminal. "
        "Pass `filter` (regex) to keep only matching lines."
    )
    args_schema: type[BaseModel] = BashOutputInput
    metadata: dict[str, Any] = {"concurrency_safe": True}

    async def _arun(
        self,
        bash_id: str,
        filter: str | None = None,
    ) -> str:
        shell_id = ShellId(bash_id)
        entry = SHELL_REGISTRY.get(shell_id)
        if entry is None:
            return (
                f"Shell {bash_id!r} not found. The id may be invalid, "
                f"or the shell may have been evicted."
            )

        # Compile filter once (per call), defensively
        pattern: re.Pattern[str] | None = None
        if filter is not None:
            try:
                pattern = re.compile(filter)
            except re.error as e:
                return f"Invalid filter regex {filter!r}: {e}"

        stdout_result = await read_new_output(shell_id, "stdout")
        stderr_result = await read_new_output(shell_id, "stderr")
        # read_new_output returns None only if the entry vanished
        # between our get() and the call — unlikely but possible.
        if stdout_result is None or stderr_result is None:
            return f"Shell {bash_id!r} disappeared mid-read."

        stdout_bytes, stdout_dropped = stdout_result
        stderr_bytes, stderr_dropped = stderr_result
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")

        if pattern is not None:
            stdout_text = _filter_lines(stdout_text, pattern)
            stderr_text = _filter_lines(stderr_text, pattern)

        return _format_response(
            shell_id=str(shell_id),
            command=entry.command,
            status=entry.status,
            exit_code=entry.exit_code,
            stdout=stdout_text,
            stderr=stderr_text,
            stdout_dropped=stdout_dropped,
            stderr_dropped=stderr_dropped,
            filter_applied=filter,
        )

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("BashOutputTool is async-only; use ainvoke().")


# ─── Helpers ────────────────────────────────────────────────────────────

def _filter_lines(text: str, pattern: re.Pattern[str]) -> str:
    """Keep only lines that the pattern can find a match in."""
    if not text:
        return text
    return "\n".join(
        line for line in text.splitlines() if pattern.search(line)
    )


def _format_response(
    *,
    shell_id: str,
    command: str,
    status: str,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    stdout_dropped: int,
    stderr_dropped: int,
    filter_applied: str | None,
) -> str:
    """Build the structured plain-text response.

    Matches the readability convention of TaskOutput's response —
    key: value header lines, then `--- stream ---` sections.
    """
    lines: list[str] = [
        f"shell_id: {shell_id}",
        f"command: {command}",
        f"status: {status}",
    ]
    if exit_code is not None:
        lines.append(f"exit_code: {exit_code}")
    if filter_applied is not None:
        lines.append(f"filter: {filter_applied}")

    def _dropped_note(n: int) -> str:
        return f" ({n} bytes dropped total)" if n > 0 else ""

    if stdout:
        lines.append(f"--- stdout (new){_dropped_note(stdout_dropped)} ---")
        lines.append(stdout)
    if stderr:
        lines.append(f"--- stderr (new){_dropped_note(stderr_dropped)} ---")
        lines.append(stderr)
    if not stdout and not stderr:
        lines.append("(no new output)")
    return "\n".join(lines)
