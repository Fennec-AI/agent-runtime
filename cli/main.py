"""CLI loop: prompt → stream agent response → repeat.

Usage:
    ANTHROPIC_API_KEY=... python -m cli

Press Ctrl+C during a streaming response to cancel it (the engine will
abort cleanly and you can issue another prompt). Press Ctrl+D or Ctrl+C
at the prompt to exit.

Configuration lives in cli/config.py — model name, max tokens, tool list.
"""
from __future__ import annotations

import asyncio
import os
import sys

from agent_runtime import Session, get_model, list_children
from agent_runtime.permissions import VALID_MODES as PERMISSION_VALID_MODES
from agent_runtime.tool_executor import CURRENT_SUBAGENT_EVENT_SINK

# Importing bootstrap triggers register_model / register_subagent calls
# in the composition root. `noqa: F401` — side-effect import.
import bootstrap  # noqa: F401
from cli.config import default_tools
from rendering import CLIRenderer, print_subagent_event


# ── ANSI for the prompt UI ──────────────────────────────────────────────
BOLD = "\033[1m"
CYAN = "\033[36m"
DIM = "\033[2m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RESET = "\033[0m"


# How long to wait after firing the abort signal before checking whether
# child agents actually unwound. Cooperative cancellation is fast in the
# common case (the abort check at the top of the loop sees the flag and
# returns immediately) but background subagents may be mid-stream and
# need a short moment.
_CANCEL_VERIFY_SECONDS = 0.3


async def chat_loop(engine: Session, renderer: CLIRenderer) -> None:
    """One prompt-and-stream cycle, repeated until user quits.

    Input modes:
      • TTY (interactive): one line per prompt — `input()` is line-buffered
        by design; users type ENTER to submit.
      • Piped stdin (`echo ... | python -m cli`, heredoc, automation):
        each prompt is a paragraph of one-or-more non-empty lines
        terminated by a blank line OR EOF. This lets test scripts and
        automation submit multi-line prompts without each line being
        treated as a separate turn.
    """
    is_tty = sys.stdin.isatty()
    read_prompt = _read_line_prompt if is_tty else _read_paragraph_prompt

    while True:
        try:
            prompt = await asyncio.to_thread(read_prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if prompt is None:                              # EOF on piped stdin
            print()
            return

        if not prompt.strip():
            continue
        if prompt.strip() in {"/exit", "/quit", "/q"}:
            return

        # In piped mode, echo the prompt so the transcript is self-
        # describing (TTY users already see what they typed).
        if not is_tty:
            preview = prompt if len(prompt) <= 200 else prompt[:200] + "…"
            print(f"\n{BOLD}{CYAN}you>{RESET} {preview}")

        print(f"\n{BOLD}assistant>{RESET} ", end="", flush=True)

        try:
            async for chunk in renderer.render(engine.submit_message(prompt)):
                print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            await _handle_cancel(engine)
            continue
        except Exception as e:
            print(f"\n{DIM}[error: {e}]{RESET}")
            continue

        print()                                         # final newline


# ── Input readers ─────────────────────────────────────────────────────────
# Two flavors: line-at-a-time for TTY (input() with prompt echo), and
# paragraph-at-a-time for piped stdin (multi-line prompts terminated by
# a blank line or EOF).
#
# Both return None ONLY on EOF — empty strings / whitespace get returned
# as-is so the caller's `if not prompt.strip(): continue` handles the
# "user just hit ENTER" case uniformly.

def _read_line_prompt() -> str:
    """Interactive TTY input: input() with the ANSI prompt label."""
    return input(f"\n{BOLD}{CYAN}you>{RESET} ")


def _read_paragraph_prompt() -> str | None:
    """Read a paragraph of non-empty lines from stdin, terminated by a
    blank line OR EOF. Returns None on EOF before any content was read,
    signalling the caller to exit the loop.

    Leading blank lines are skipped so prompts can be separated by
    multiple blanks without inserting empty turns.
    """
    lines: list[str] = []
    while True:
        line = sys.stdin.readline()
        if line == "":                                  # EOF
            return "\n".join(lines) if lines else None
        stripped = line.rstrip("\n").rstrip("\r")
        if stripped:
            lines.append(stripped)
        elif lines:
            return "\n".join(lines)                     # blank line ends paragraph
        # else: leading blank — skip


def _make_ask_callback():
    """Build an async (tool_name, args, reason) -> bool that prompts the
    user y/N before a gated tool (Bash / Edit / Write) runs. Used only in
    interactive TTY sessions."""
    async def ask(tool_name: str, args: dict, reason: str) -> bool:
        preview = str(args)
        if len(preview) > 100:
            preview = preview[:100] + "…"
        prompt = (
            f"\n{YELLOW}[permission] {BOLD}{tool_name}{RESET}{YELLOW}"
            f"({preview}) — allow? [y/N]{RESET} "
        )
        answer = await asyncio.to_thread(input, prompt)
        approved = answer.strip().lower() in ("y", "yes")
        if not approved:
            print(f"{YELLOW}[denied]{RESET}")
        return approved
    return ask


async def _handle_cancel(engine: Session) -> None:
    """User pressed Ctrl+C mid-turn. Fire abort, then verify that the
    main agent's children actually unwound, so the user can see whether
    cancellation propagated."""
    # Snapshot what was running BEFORE we fire the abort, so the message
    # is honest about scope.
    descendants_before = list_children(engine.session_id)
    running_before = [
        e for e in descendants_before
        if e.task is not None and not e.task.done()
    ]

    engine.cancel("user_pressed_ctrl_c")

    if not running_before:
        # Nothing in flight — only the parent's own LLM call was running.
        print(f"\n{YELLOW}[cancelled]{RESET}")
        return

    n = len(running_before)
    ids_preview = ", ".join(e.agent_id[:10] for e in running_before[:3])
    if n > 3:
        ids_preview += f", … (+{n - 3} more)"
    print(
        f"\n{YELLOW}[cancelling {n} background agent(s): {ids_preview}…]{RESET}",
        flush=True,
    )

    # Give children a moment to honor the abort flag and unregister.
    await asyncio.sleep(_CANCEL_VERIFY_SECONDS)

    still_running = [
        e for e in list_children(engine.session_id)
        if e.task is not None and not e.task.done()
    ]
    if not still_running:
        print(f"{GREEN}[✓ all {n} agent(s) cancelled cleanly]{RESET}")
    else:
        # Some agents ignored the cooperative signal. Tell the user;
        # they'll be force-killed when the session closes.
        remaining_ids = ", ".join(e.agent_id[:10] for e in still_running[:3])
        print(
            f"{YELLOW}[⚠ {len(still_running)} agent(s) still running after "
            f"{_CANCEL_VERIFY_SECONDS}s: {remaining_ids}…]{RESET}"
        )


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    # Permission mode from env (default / acceptEdits / plan / bypassPermissions).
    mode = os.environ.get("AGENT_PERMISSION_MODE", "default")
    if mode not in PERMISSION_VALID_MODES:
        print(f"unknown AGENT_PERMISSION_MODE={mode!r}; using 'default'", file=sys.stderr)
        mode = "default"

    # Prompt for gated tools only when interactive AND the mode can ask.
    # Piped/automated runs (no TTY) auto-allow so they don't deadlock.
    ask_callback = (
        _make_ask_callback()
        if sys.stdin.isatty() and mode in ("default", "acceptEdits")
        else None
    )

    engine = Session(
        llm=get_model("default"),
        tools=default_tools(),
        permission_mode=mode,
        ask_callback=ask_callback,
    )
    renderer = CLIRenderer()

    # Install the subagent event observer — surfaces each child's tool
    # calls and results to stdout with a "[<type> <short-id>] " prefix
    # so the user can see what spawned subagents are doing internally,
    # not just their final answer. Survives the chat_loop lifetime
    # because ContextVar set at this scope inherits into every task.
    CURRENT_SUBAGENT_EVENT_SINK.set(print_subagent_event)

    print(f"{DIM}agent_runtime CLI — Ctrl+C cancels current turn, Ctrl+D exits{RESET}")
    try:
        await chat_loop(engine, renderer)
    finally:
        # aclose runs every registered cleanup (kills orphan background
        # agents) before unregistering the session itself.
        await engine.aclose()


if __name__ == "__main__":
    asyncio.run(main())
