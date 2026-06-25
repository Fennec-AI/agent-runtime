"""Bash — run a shell command.

Mirrors Claude Code's `Bash` tool. Required: command. Optional: timeout
(in milliseconds, matching Claude Code's unit), description,
run_in_background.

Two modes:
  • Synchronous (default): spawn subprocess, await communicate(),
    return exit_code + stdout/stderr.
  • Background (run_in_background=True): spawn subprocess but DON'T
    await it. Register it in SHELL_REGISTRY, fire off reader/waiter
    tasks, return the shell_id immediately. Model uses BashOutput to
    poll and KillShell to terminate.

NOT concurrency-safe — shell commands typically have side effects, and
two parallel commands might race on the filesystem, ports, env, etc.
Stays in the unsafe partition of the dispatcher.

# TODO — feature parity with Claude Code's BashTool, intentionally
#        deferred to keep this minimal:
#
#   [ ] Sandbox / permission classifier. Claude Code routes destructive
#       commands through a sandbox or permission prompt. We just run.
#
#   [ ] Working-directory persistence between calls. Each call gets its
#       own subprocess with cwd=Path.cwd(); `cd foo` won't carry over.
#       (Same as Claude Code, actually — they prefer absolute paths and
#       discourage `cd`.)
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..bash_shells import (
    ShellEntry,
    kill_shell,
    new_shell_id,
    register_shell,
    spawn_background_tasks,
)
from ..cancellation import AbortController
from ..cleanup_registry import register_cleanup
from ..tool_executor import CURRENT_RUN_CONTEXT


logger = logging.getLogger(__name__)


# Defaults match a reasonable interactive shell experience.
DEFAULT_TIMEOUT_MS = 120_000      # 2 minutes — same as Claude Code's default
MAX_TIMEOUT_MS = 600_000           # 10 minutes — Claude Code's hard cap
MAX_OUTPUT_CHARS = 30_000          # output cap; truncate with marker if longer


class BashInput(BaseModel):
    """Arguments for Bash."""
    command: str = Field(description="The command to execute.")
    timeout: int | None = Field(
        default=None, gt=0, le=MAX_TIMEOUT_MS,
        description=(
            f"Optional timeout in milliseconds (max {MAX_TIMEOUT_MS}, "
            f"default {DEFAULT_TIMEOUT_MS})."
        ),
    )
    description: str | None = Field(
        default=None,
        description=(
            "Clear, concise description of what this command does in active "
            'voice. Never use words like "complex" or "risk" — just describe '
            "what it does."
        ),
    )
    run_in_background: bool | None = Field(
        default=None,
        description=(
            "Set to true to spawn the command as a persistent background "
            "shell instead of waiting for it. Returns a shell_id "
            "immediately; use BashOutput(shell_id) to poll stdout/stderr "
            "and KillShell(shell_id) to terminate."
        ),
    )


class BashTool(BaseTool):
    """Execute a shell command in the current working directory."""

    name: str = "Bash"
    description: str = (
        "Execute a shell command and return stdout, stderr, and exit code. "
        f"Default timeout {DEFAULT_TIMEOUT_MS // 1000}s, max "
        f"{MAX_TIMEOUT_MS // 1000}s. Output capped at {MAX_OUTPUT_CHARS} "
        "characters per stream. Commands run in the runtime's current "
        "working directory; `cd` does not persist across calls."
    )
    args_schema: type[BaseModel] = BashInput
    # NOT concurrency-safe — shell commands typically have side effects.
    # NOT concurrency-safe (side effects), and gated: in default mode a
    # wired ask_callback prompts before running; plan mode blocks it.
    metadata: dict[str, Any] = {"concurrency_safe": False, "permission": "ask"}

    async def _arun(
        self,
        command: str,
        timeout: int | None = None,
        description: str | None = None,
        run_in_background: bool | None = None,
    ) -> str:
        if run_in_background:
            return await _spawn_background(command)

        timeout_seconds = (timeout or DEFAULT_TIMEOUT_MS) / 1000.0
        ctx = CURRENT_RUN_CONTEXT.get(None)

        # ── Fast-fail: abort already set ──────────────────────────────
        # Don't waste a subprocess fork on a command we'll just kill.
        if ctx is not None and ctx.abort.aborted:
            return (
                f"Error: command not executed (parent aborted before start: "
                f"{ctx.abort.reason or 'unknown'}).\nCommand: {command}"
            )

        logger.debug("Bash: %s (timeout=%ss)", command, timeout_seconds)

        # ── Spawn subprocess ──────────────────────────────────────────
        # start_new_session=True makes the subprocess the leader of a
        # NEW process group whose ID equals proc.pid. This lets us kill
        # the whole tree (sh + every descendant) via os.killpg, fixing
        # the "sh -c 'sleep 30' doesn't die on SIGTERM because sh just
        # waits for sleep" pathology — the 2s residue from 51813fe.
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd(),
                start_new_session=True,
            )
        except OSError as e:
            return f"Error spawning subprocess: {e}"

        # ── Race: communicate() vs abort signal vs timeout ────────────
        # The `finally` guarantees the subprocess can't outlive us no
        # matter how we exit — natural completion, timeout, abort, or
        # outer-task cancellation (asyncio.CancelledError).
        try:
            return await _await_result(
                proc=proc,
                command=command,
                timeout_seconds=timeout_seconds,
                ctx=ctx,
            )
        finally:
            if proc.returncode is None:
                # We're exiting and the proc is still alive — kill it.
                # asyncio.shield so a cancellation mid-terminate still
                # gives SIGTERM a chance to land before we're unwound.
                await asyncio.shield(_terminate_proc(proc))

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("BashTool is async-only; use ainvoke().")


def _truncate(text: str, label: str) -> str:
    """Truncate a long output stream, leaving a marker."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    kept = text[:MAX_OUTPUT_CHARS]
    dropped = len(text) - MAX_OUTPUT_CHARS
    return (
        f"{kept}\n\n[{label} truncated; {dropped} more chars omitted]"
    )


# ─── Sync-mode wait helpers (abort-aware) ───────────────────────────────

async def _await_result(
    proc: "asyncio.subprocess.Process",
    command: str,
    timeout_seconds: float,
    ctx: Any | None,
) -> str:
    """Wait for the subprocess to finish or abort/timeout to fire.

    Returns the formatted response string. Does NOT terminate the
    subprocess on its way out — the caller's `finally` does that, so
    every exit path (including unexpected exceptions) cleans up.
    """
    comm_task = asyncio.create_task(
        proc.communicate(), name="bash-sync-communicate",
    )
    waiters: list[asyncio.Task[Any]] = [comm_task]

    # Subscribe to abort if we have a RunContext. None when invoked
    # outside a tool dispatch (direct test harness).
    abort_task: asyncio.Task[bool] | None = None
    if ctx is not None:
        abort_task = asyncio.create_task(
            ctx.abort.signal.wait(), name="bash-sync-abort-watch",
        )
        waiters.append(abort_task)

    done, _pending = await asyncio.wait(
        waiters,
        timeout=timeout_seconds,
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel + drain any still-pending waiter so it doesn't linger.
    # We catch both CancelledError and Exception because comm_task may
    # have raised something we'd swallow by cancelling.
    for t in waiters:
        if not t.done():
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    # ── Determine outcome ─────────────────────────────────────────────
    if comm_task in done:
        # Natural completion (or comm_task raised; .result() re-raises).
        try:
            stdout_bytes, stderr_bytes = comm_task.result()
        except Exception as e:                          # noqa: BLE001
            return f"Error reading subprocess output: {e}"
        return _format_completed(stdout_bytes, stderr_bytes, proc.returncode)

    if abort_task is not None and abort_task in done:
        reason = (ctx.abort.reason if ctx is not None else None) or "unknown"
        return (
            f"Error: command aborted by parent (reason: {reason}).\n"
            f"Command: {command}"
        )

    # Neither completed nor aborted → timeout.
    return (
        f"Error: command timed out after {timeout_seconds:.1f}s "
        f"and was killed.\nCommand: {command}"
    )


async def _terminate_proc(proc: "asyncio.subprocess.Process") -> None:
    """SIGTERM the whole process group; SIGKILL if anyone survives 2s.

    Because we spawn with start_new_session=True, proc.pid is also the
    process group ID. `os.killpg(pgid, SIGTERM)` reaches sh AND every
    descendant (sh's `sleep 30; echo` would otherwise survive a plain
    proc.terminate(), since SIGTERM on sh doesn't propagate to its
    children automatically).

    Falls back to proc.terminate() if killpg can't find the group
    (ProcessLookupError after the leader already exited) or doesn't
    have permission. Idempotent — safe to call multiple times.
    """
    if proc.returncode is not None:
        return                                          # already gone

    pgid = _safe_pgid(proc.pid)
    _send_signal_group_or_proc(proc, pgid, signal.SIGTERM)

    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
        return
    except asyncio.TimeoutError:
        pass                                            # escalate

    _send_signal_group_or_proc(proc, pgid, signal.SIGKILL)
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        logger.warning(
            "bash sync subprocess did not exit after SIGKILL (pid=%s)",
            proc.pid,
        )


def _safe_pgid(pid: int) -> int | None:
    """Best-effort pgid lookup. Returns None if the process is gone."""
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, OSError):
        return None


def _send_signal_group_or_proc(
    proc: "asyncio.subprocess.Process",
    pgid: int | None,
    sig: signal.Signals,
) -> None:
    """Send signal to the process group; fall back to the lone proc."""
    if pgid is not None:
        try:
            os.killpg(pgid, sig)
            return
        except (ProcessLookupError, PermissionError):
            pass
    # Group send failed — try the single proc as a fallback.
    try:
        if sig == signal.SIGKILL:
            proc.kill()
        else:
            proc.terminate()                            # SIGTERM
    except ProcessLookupError:
        pass


def _format_completed(
    stdout_bytes: bytes,
    stderr_bytes: bytes,
    returncode: int | None,
) -> str:
    """Build the exit_code / stdout / stderr response for natural exit."""
    exit_code = returncode if returncode is not None else -1
    stdout = _truncate(
        stdout_bytes.decode("utf-8", errors="replace"), "stdout",
    )
    stderr = _truncate(
        stderr_bytes.decode("utf-8", errors="replace"), "stderr",
    )
    parts: list[str] = [f"exit_code: {exit_code}"]
    if stdout:
        parts.append(f"--- stdout ---\n{stdout}")
    if stderr:
        parts.append(f"--- stderr ---\n{stderr}")
    if not stdout and not stderr:
        parts.append("(no output)")
    return "\n".join(parts)


# ─── Background-spawn branch ────────────────────────────────────────────

async def _spawn_background(command: str) -> str:
    """Spawn a persistent background shell, register it, wire up reader
    + waiter tasks, register a shutdown cleanup hook, return shell_id.

    The reader/waiter tasks live on the event loop until the shell exits
    (naturally or via kill_shell). The cleanup hook ensures that if the
    Session aclose()s while this shell is still running, we SIGTERM it.

    start_new_session=True puts the subprocess in its own process group
    so kill_shell can SIGTERM the whole tree, not just sh.
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd(),
            start_new_session=True,
        )
    except OSError as e:
        return f"Error spawning background subprocess: {e}"

    shell_id = new_shell_id()
    entry = ShellEntry(
        shell_id=shell_id,
        command=command,
        proc=proc,
        started_at=time.time(),
    )
    register_shell(entry)
    spawn_background_tasks(entry)

    # Shutdown hook — kill this shell if the runtime Session.aclose()s
    # before the shell finishes. The hook auto-unregisters once the
    # shell exits naturally (see _drop_cleanup_on_exit below) so we
    # don't try to kill a corpse.
    cleanup_unregister = register_cleanup(
        lambda sid=shell_id: kill_shell(sid, reason="shutdown")
    )

    # Abort propagation — subscribe to the SPAWNING agent's abort signal
    # so Session.cancel() (and any abort that bubbles through the agent
    # hierarchy) kills our shell. Without this, the shell would outlive
    # its parent's lifecycle, matching the pre-fix bug found in e2e
    # scenario 9. SpawnAgent already wires child agents this way via
    # create_child_controller — this brings shells to parity.
    #
    # `ctx` is None only for shells spawned outside a tool dispatch
    # context (e.g., direct tests, harnesses); in that case we skip the
    # abort wiring — the shutdown hook is still in place.
    ctx = CURRENT_RUN_CONTEXT.get(None)
    abort_watcher: asyncio.Task[None] | None = None
    if ctx is not None:
        abort_watcher = asyncio.create_task(
            _kill_on_abort(entry, ctx.abort),
            name=f"shell-abort-watch-{shell_id}",
        )

    asyncio.create_task(
        _drop_cleanup_on_exit(entry, cleanup_unregister, abort_watcher),
        name=f"shell-cleanup-{shell_id}",
    )

    logger.info("Spawned background shell %s (pid=%s): %s",
                shell_id, proc.pid, command[:80])
    return (
        f"Spawned background shell {shell_id} (pid={proc.pid}).\n"
        f"Use BashOutput(bash_id={shell_id!r}) to poll output, "
        f"KillShell(shell_id={shell_id!r}) to terminate."
    )


async def _kill_on_abort(entry: ShellEntry, abort: AbortController) -> None:
    """Wait for the spawning agent's abort signal; when it fires, kill
    the shell. Cancelled by _drop_cleanup_on_exit when the shell exits
    naturally (so the watcher doesn't sit parked on a corpse)."""
    try:
        await abort.signal.wait()
    except asyncio.CancelledError:
        # Natural exit cancelled us — nothing to do.
        return
    if entry.status == "running":
        reason = f"parent_aborted:{abort.reason or 'unknown'}"
        try:
            await kill_shell(entry.shell_id, reason=reason)
        except Exception as e:                          # noqa: BLE001
            logger.warning(
                "kill_on_abort failed for shell %s: %s", entry.shell_id, e,
            )


async def _drop_cleanup_on_exit(
    entry: ShellEntry,
    cleanup_unregister: Any,
    abort_watcher: asyncio.Task[None] | None = None,
) -> None:
    """Once the shell ends, drop its shutdown cleanup hook AND cancel
    its abort watcher — nothing left to clean up in either case."""
    if entry.waiter is not None:
        try:
            await entry.waiter
        except Exception:
            pass
    cleanup_unregister()
    if abort_watcher is not None and not abort_watcher.done():
        abort_watcher.cancel()
