"""Registry of persistent background shells (Bash run_in_background=True).

Same shape as agent_runtime/registry.py (the agent registry) but for
shell subprocesses spawned by `Bash(run_in_background=True)`.

Per shell we hold:
  • The asyncio.subprocess.Process
  • Two bounded bytearray buffers (stdout, stderr) with read cursors,
    so `BashOutput` can return only "new since last read"
  • A lock that serializes buffer reads/writes
  • Three background tasks:
      - one stdout drainer (reads pipe → appends to buffer)
      - one stderr drainer (same)
      - one waiter (await proc.wait() → flips status to 'exited')

Buffer overflow: each stream caps at MAX_BUFFER_BYTES. If a noisy
command exceeds it, oldest bytes are dropped and `dropped_bytes` is
incremented. Read cursor is shifted down so it stays valid.

Status:
  • 'running' — proc still alive
  • 'exited'  — natural exit
  • 'killed'  — we sent it a signal via kill_shell()

Entries stay in the registry after exit so `BashOutput` can still
return the final tail. The shutdown cleanup hook (registered by
BashTool when spawning) only acts on shells still in 'running' state.
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
import signal
import time
from dataclasses import dataclass, field
from typing import Literal, NewType


logger = logging.getLogger(__name__)


ShellId = NewType("ShellId", str)


# Bounds — match a "reasonable interactive shell" expectation.
MAX_BUFFER_BYTES = 1_048_576       # 1 MB cap per stream
READ_CHUNK_SIZE = 4096             # how much to read at a time from each pipe


ShellStatus = Literal["running", "exited", "killed"]


@dataclass
class ShellEntry:
    """One persistent background shell."""

    shell_id: ShellId
    command: str
    proc: "asyncio.subprocess.Process | None"
    started_at: float

    # Bounded output buffers + cursors for "new since last read"
    stdout_buf: bytearray = field(default_factory=bytearray)
    stderr_buf: bytearray = field(default_factory=bytearray)
    stdout_cursor: int = 0
    stderr_cursor: int = 0
    stdout_dropped_bytes: int = 0
    stderr_dropped_bytes: int = 0

    # Lifecycle
    status: ShellStatus = "running"
    exit_code: int | None = None
    ended_at: float | None = None

    # Background tasks (set by spawn_background_tasks)
    reader_stdout: "asyncio.Task | None" = None
    reader_stderr: "asyncio.Task | None" = None
    waiter: "asyncio.Task | None" = None

    # Serializes buffer reads vs reader-task appends
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# Global registry — module-level by design (single-process runtime).
SHELL_REGISTRY: dict[ShellId, ShellEntry] = {}


# ─── Registry CRUD ─────────────────────────────────────────────────────

def new_shell_id() -> ShellId:
    """Generate a fresh shell_id ('sh_' + 16 hex), guaranteed unique
    against current registry contents."""
    while True:
        sid = ShellId(f"sh_{secrets.token_hex(8)}")
        if sid not in SHELL_REGISTRY:
            return sid


def register_shell(entry: ShellEntry) -> None:
    if entry.shell_id in SHELL_REGISTRY:
        raise ValueError(f"shell_id {entry.shell_id!r} already registered")
    SHELL_REGISTRY[entry.shell_id] = entry


def unregister_shell(shell_id: ShellId) -> None:
    SHELL_REGISTRY.pop(shell_id, None)


def get_shell(shell_id: ShellId) -> ShellEntry | None:
    return SHELL_REGISTRY.get(shell_id)


def list_running_shells() -> list[ShellEntry]:
    return [e for e in SHELL_REGISTRY.values() if e.status == "running"]


# ─── Kill ──────────────────────────────────────────────────────────────

async def kill_shell(shell_id: ShellId, *, reason: str = "killed") -> bool:
    """Terminate a running shell. Returns True if a signal was sent.

    Strategy: SIGTERM the whole process group; wait 2s; SIGKILL the
    group if anyone survives. The PROCESS GROUP matters — a command
    like `sh -c "sleep 30; echo X"` forks sh, which forks sleep; SIGTERM
    on sh alone doesn't kill sleep (sh just sits waiting). Killing the
    whole group reaches every descendant.

    Idempotent — calling on an already-terminal shell returns False.
    """
    entry = SHELL_REGISTRY.get(shell_id)
    if entry is None:
        return False
    if entry.status != "running":
        return False

    proc = entry.proc
    if proc is None or proc.returncode is not None:
        # subprocess already dead — just flip status
        entry.status = "killed"
        entry.ended_at = time.time()
        if entry.exit_code is None and proc is not None:
            entry.exit_code = proc.returncode
        return True

    pgid = _safe_pgid(proc.pid)

    # SIGTERM the group
    _send_signal_group_or_proc(proc, pgid, signal.SIGTERM)

    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        # Some descendant ignored SIGTERM — escalate to SIGKILL on the group.
        _send_signal_group_or_proc(proc, pgid, signal.SIGKILL)
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning(
                "shell %s did not exit after SIGKILL (pid=%s)",
                shell_id, proc.pid,
            )

    entry.status = "killed"
    entry.ended_at = time.time()
    if entry.exit_code is None:
        entry.exit_code = proc.returncode
    logger.info("Killed shell %s (%s, exit=%s)", shell_id, reason, entry.exit_code)
    return True


# ── Signal helpers ─────────────────────────────────────────────────────
# Duplicated (intentionally) from agent_runtime/tools/bash.py — both
# locations need them and we want bash_shells to stay independent of
# the tools/ layer. Same shape, same fallback ladder.

def _safe_pgid(pid: int) -> int | None:
    """Best-effort process-group lookup. None if the proc has exited."""
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, OSError):
        return None


def _send_signal_group_or_proc(
    proc: "asyncio.subprocess.Process",
    pgid: int | None,
    sig: signal.Signals,
) -> None:
    """Send `sig` to the process group; fall back to the lone proc.

    Group send fails (ProcessLookupError) if the leader has already
    exited but the proc reference is still around — common race after
    SIGTERM has done its job.
    """
    if pgid is not None:
        try:
            os.killpg(pgid, sig)
            return
        except (ProcessLookupError, PermissionError):
            pass
    try:
        if sig == signal.SIGKILL:
            proc.kill()
        else:
            proc.terminate()
    except ProcessLookupError:
        pass


# ─── Buffered read ─────────────────────────────────────────────────────

async def read_new_output(
    shell_id: ShellId,
    stream: Literal["stdout", "stderr"],
) -> tuple[bytes, int] | None:
    """Read all bytes appended since the last read for the given stream.

    Returns (new_bytes, total_dropped_bytes) or None if shell not found.
    `total_dropped_bytes` is cumulative for the shell's life — useful
    for callers that want to detect truncation between reads.
    """
    entry = SHELL_REGISTRY.get(shell_id)
    if entry is None:
        return None
    async with entry.lock:
        if stream == "stdout":
            new = bytes(entry.stdout_buf[entry.stdout_cursor:])
            entry.stdout_cursor = len(entry.stdout_buf)
            return new, entry.stdout_dropped_bytes
        else:
            new = bytes(entry.stderr_buf[entry.stderr_cursor:])
            entry.stderr_cursor = len(entry.stderr_buf)
            return new, entry.stderr_dropped_bytes


# ─── Internal: drain + wait tasks ──────────────────────────────────────

async def _drain_stream(
    entry: ShellEntry,
    stream: "asyncio.StreamReader",
    kind: Literal["stdout", "stderr"],
) -> None:
    """Read from a subprocess pipe until EOF, appending into the entry's
    buffer with overflow handling."""
    try:
        while True:
            chunk = await stream.read(READ_CHUNK_SIZE)
            if not chunk:
                break
            async with entry.lock:
                if kind == "stdout":
                    buf = entry.stdout_buf
                    buf.extend(chunk)
                    overflow = len(buf) - MAX_BUFFER_BYTES
                    if overflow > 0:
                        del buf[:overflow]
                        entry.stdout_dropped_bytes += overflow
                        entry.stdout_cursor = max(0, entry.stdout_cursor - overflow)
                else:
                    buf = entry.stderr_buf
                    buf.extend(chunk)
                    overflow = len(buf) - MAX_BUFFER_BYTES
                    if overflow > 0:
                        del buf[:overflow]
                        entry.stderr_dropped_bytes += overflow
                        entry.stderr_cursor = max(0, entry.stderr_cursor - overflow)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(
            "drain error on shell %s/%s: %s", entry.shell_id, kind, e,
        )


async def _wait_for_exit(entry: ShellEntry) -> None:
    """Await proc.wait() and flip the entry's status to 'exited'.

    Skips the status change if kill_shell() already set 'killed' — that
    branch sets status itself.
    """
    if entry.proc is None:
        return
    try:
        exit_code = await entry.proc.wait()
    except asyncio.CancelledError:
        raise
    if entry.status == "running":
        entry.status = "exited"
        entry.exit_code = exit_code
        entry.ended_at = time.time()
    logger.info(
        "Shell %s ended: status=%s exit_code=%s",
        entry.shell_id, entry.status, exit_code,
    )


def spawn_background_tasks(entry: ShellEntry) -> None:
    """Wire up the drain + waiter tasks for a freshly registered shell.

    Splits the task creation out of BashTool so the tool stays focused
    on subprocess setup; lifecycle plumbing lives here.
    """
    proc = entry.proc
    if proc is None:
        return
    assert proc.stdout is not None and proc.stderr is not None, \
        "expected PIPE'd stdout/stderr"

    entry.reader_stdout = asyncio.create_task(
        _drain_stream(entry, proc.stdout, "stdout"),
        name=f"shell-stdout-{entry.shell_id}",
    )
    entry.reader_stderr = asyncio.create_task(
        _drain_stream(entry, proc.stderr, "stderr"),
        name=f"shell-stderr-{entry.shell_id}",
    )
    entry.waiter = asyncio.create_task(
        _wait_for_exit(entry),
        name=f"shell-wait-{entry.shell_id}",
    )
