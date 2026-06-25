"""CLIRenderer — renders agent events to ANSI-colored terminal text.

The output is just strings; the caller is responsible for printing them
with `print(s, end="", flush=True)` so terminal streaming feels live.

Per-turn state (in-flight tool map) lives inside one render() call so the
renderer is safe to call once per turn without leaking state across turns.

Agent lifecycle visibility:
  - When the model calls SpawnAgentTool, we render a "🚀 Spawning…"
    banner that surfaces the subagent_type, run_in_background flag, and
    description. For parallel batches each spawn gets its own line.
  - When the ToolMessage comes back from a background spawn, we render
    the acknowledgement explicitly.
  - When a <task-notification> arrives in the parent's inbox (we get it
    as a yielded HumanMessage from query.py), we parse it and render a
    color-coded completion / failure / kill line.
"""
from __future__ import annotations

import re
from typing import AsyncGenerator

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


# ── ANSI palette ────────────────────────────────────────────────────────
GREY = "\033[90m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Name the model sees when calling SpawnAgentTool (matches the tool's
# `name` attribute). Hard-coded to avoid a runtime import dependency
# from the rendering layer.
_SPAWN_TOOL_NAME = "Agent"


class CLIRenderer:
    """Renders agent events as ANSI-colored terminal output strings.

    Attribution:
      - When the model emits multiple tool_calls in one turn, they're
        shown as a numbered batch ("Calling N tools").
      - Each ToolMessage result is matched back to its tool_call by
        tool_use_id so results show "[N] tool_name → preview" even when
        they arrive out of order (which is the norm under concurrent
        dispatch).
      - When subagents land (Step 5), an `agent_id` prefix can be added
        — slot wired but inert for now.
    """

    async def render(
        self,
        events: AsyncGenerator[BaseMessageChunk | BaseMessage, None],
    ) -> AsyncGenerator[str, None]:
        # Per-turn state. Maps tool_call_id → (index, tool_name, is_spawn,
        # is_background) so result messages can show which call they
        # correspond to and spawn ToolMessages can be rendered specially.
        in_flight: dict[str, _CallInfo] = {}

        async for event in events:
            # Streaming assistant text — yield each chunk inline.
            if isinstance(event, AIMessageChunk):
                text = self._extract_text(event.content)
                if text:
                    yield text

            # The complete assistant turn — useful for showing tool_calls
            # that the model just emitted.
            elif isinstance(event, AIMessage) and event.tool_calls:
                for line in self._render_tool_calls(event.tool_calls, in_flight):
                    yield line

            # A tool's result, attributed back to its tool_call by id.
            elif isinstance(event, ToolMessage):
                yield self._render_tool_result(event, in_flight)

            # A drained inbox notification — yielded by query.py when a
            # background child finishes (HumanMessage with the XML body).
            elif isinstance(event, HumanMessage) and _looks_like_notification(event.content):
                yield self._render_notification(event.content)

            # A compaction event — query() yields a SystemMessage with the
            # COMPACTION_MARKER prefix when auto-compaction fires.
            elif isinstance(event, SystemMessage) and _looks_like_compaction(event.content):
                yield self._render_compaction(event.content)

    # ── tool batch display ──────────────────────────────────────────────

    def _render_tool_calls(self, tool_calls, in_flight: dict) -> list[str]:
        """Build the list of output lines for a batch of tool calls.

        Returns a list (not a generator) so the caller can iterate cleanly
        from inside an `async def`. As a side effect, records each call in
        `in_flight` for later attribution by tool_use_id.

        Spawn calls (name="Agent") render specially: subagent_type and
        run_in_background flag get called out so the user can see what's
        actually happening.
        """
        n = len(tool_calls)
        agent_prefix = self._agent_prefix(None)         # slot for Step 5
        lines: list[str] = []

        # Count how many are spawns so we can choose a header.
        spawns = [tc for tc in tool_calls if tc["name"] == _SPAWN_TOOL_NAME]
        non_spawns = [tc for tc in tool_calls if tc["name"] != _SPAWN_TOOL_NAME]

        # Mixed batches are rare but possible — render generic for non-
        # spawns, special for spawns. Indexes are unified across the batch.
        # All-spawn batches get a distinct header ("🚀 Spawning N agents:").
        if spawns and not non_spawns and n > 1:
            lines.append(
                f"\n{agent_prefix}{MAGENTA}  ┌ 🚀 Spawning {n} agents:{RESET}\n"
            )
            for i, tc in enumerate(tool_calls, start=1):
                in_flight[tc["id"]] = _CallInfo(
                    idx=i, name=tc["name"], is_spawn=True,
                    is_background=bool(tc["args"].get("run_in_background")),
                    subagent_type=tc["args"].get("subagent_type") or "general-purpose",
                )
                lines.append(self._format_spawn_line(tc, idx=i, branch="│ "))
            return lines

        if n == 1:
            tc = tool_calls[0]
            is_spawn = tc["name"] == _SPAWN_TOOL_NAME
            in_flight[tc["id"]] = _CallInfo(
                idx=1, name=tc["name"], is_spawn=is_spawn,
                is_background=bool(tc["args"].get("run_in_background")) if is_spawn else False,
                subagent_type=(tc["args"].get("subagent_type") or "general-purpose") if is_spawn else None,
            )
            if is_spawn:
                lines.append(f"\n{agent_prefix}")
                lines.append(self._format_spawn_line(tc, idx=None, branch=""))
            else:
                lines.append(
                    f"\n{agent_prefix}{GREY}  → {tc['name']}({self._args(tc['args'])}){RESET}\n"
                )
            return lines

        # Mixed / non-spawn parallel batch — generic header.
        lines.append(f"\n{agent_prefix}{GREY}  ┌ Calling {n} tools:{RESET}\n")
        for i, tc in enumerate(tool_calls, start=1):
            is_spawn = tc["name"] == _SPAWN_TOOL_NAME
            in_flight[tc["id"]] = _CallInfo(
                idx=i, name=tc["name"], is_spawn=is_spawn,
                is_background=bool(tc["args"].get("run_in_background")) if is_spawn else False,
                subagent_type=(tc["args"].get("subagent_type") or "general-purpose") if is_spawn else None,
            )
            if is_spawn:
                lines.append(self._format_spawn_line(tc, idx=i, branch="│ "))
            else:
                lines.append(
                    f"{GREY}  │ [{i}] {tc['name']}({self._args(tc['args'])}){RESET}\n"
                )
        return lines

    def _format_spawn_line(self, tc, idx: int | None, branch: str) -> str:
        """One line announcing a spawn — subagent_type, mode, description."""
        args = tc["args"]
        subtype = args.get("subagent_type") or "general-purpose"
        bg = bool(args.get("run_in_background"))
        mode = f"{YELLOW}background{RESET}" if bg else f"{CYAN}sync{RESET}"
        desc = args.get("description") or "(no description)"
        idx_str = f"[{idx}] " if idx is not None else ""
        # Trim very long descriptions for the announce line; full prompt is
        # still in the AIMessage for anyone scrolling back.
        if len(desc) > 70:
            desc = desc[:70] + "…"
        prefix = f"{MAGENTA}  {branch}{RESET}" if branch else f"{MAGENTA}  🚀 {RESET}"
        return (
            f"{prefix}{idx_str}{BOLD}{subtype}{RESET} ({mode}): "
            f"{MAGENTA}{desc}{RESET}\n"
        )

    def _render_tool_result(self, event: ToolMessage, in_flight: dict) -> str:
        info = in_flight.get(event.tool_call_id)
        if info is None:
            info = _CallInfo(idx=0, name="?", is_spawn=False, is_background=False)
        color = RED if event.status == "error" else GREEN

        # Spawn results — render with the subagent_type (not the generic
        # "Agent" tool name) so the user sees "Explore → …" or "Plan → …".
        spawn_label = info.subagent_type or info.name

        # Background spawn ack: "Spawned background agent <id>…". Show
        # the agent_id so the user can correlate later notifications.
        if info.is_spawn and info.is_background and event.status != "error":
            agent_id = _extract_spawned_id(event.content)
            idx_str = f"[{info.idx}] " if self._batch_had_multiple(in_flight) else ""
            return (
                f"{MAGENTA}  ✓ {idx_str}{BOLD}{spawn_label}{RESET}{MAGENTA} → "
                f"{BOLD}{agent_id or 'unknown-id'}{RESET}{MAGENTA} "
                f"(awaiting notification){RESET}\n"
            )

        # Sync spawn that returned the child's final text.
        if info.is_spawn and not info.is_background and event.status != "error":
            preview = self._summarize(event.content)
            idx_str = f"[{info.idx}] " if self._batch_had_multiple(in_flight) else ""
            return (
                f"{MAGENTA}  ✓ {idx_str}{BOLD}{spawn_label}{RESET}{MAGENTA} → "
                f"{preview}{RESET}\n"
            )

        # TodoWrite — show the full checklist, not an 80-char preview, so
        # the user can watch the task list evolve.
        if info.name == "TodoWrite" and event.status != "error":
            return self._render_todos(event.content)

        # Regular tool result (or a spawn that errored).
        preview = self._summarize(event.content)
        idx_str = f"[{info.idx}] " if self._batch_had_multiple(in_flight) else ""
        return f"{color}  ✓ {idx_str}{info.name} → {preview}{RESET}\n"

    def _render_todos(self, content) -> str:
        """Render a TodoWrite result as a colored checklist. The tool already
        produced the text form; we just colorize the status glyphs."""
        text = content if isinstance(content, str) else str(content)
        out_lines = [f"{CYAN}  ✓ TodoWrite{RESET}"]
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("✔"):
                out_lines.append(f"{GREEN}    {stripped}{RESET}")
            elif stripped.startswith("▶"):
                out_lines.append(f"{YELLOW}    {stripped}{RESET}")
            elif stripped.startswith("☐"):
                out_lines.append(f"{GREY}    {stripped}{RESET}")
            elif stripped:
                out_lines.append(f"{GREY}    {stripped}{RESET}")
        return "\n".join(out_lines) + "\n"

    @staticmethod
    def _batch_had_multiple(in_flight: dict) -> bool:
        """True if any tool_call in this turn had an idx > 1 (i.e., the
        turn had a parallel batch — used to decide whether to print [N]
        prefixes on results)."""
        return any(info.idx > 1 for info in in_flight.values())

    # ── notification (drained inbox) display ────────────────────────────

    def _render_notification(self, content) -> str:
        """Render a <task-notification> XML body as a one-line lifecycle
        event. Always emits a leading newline so it stands out from the
        preceding assistant text or tool result."""
        text = content if isinstance(content, str) else str(content)
        task_id = _xml_tag(text, "task-id") or "unknown"
        status = _xml_tag(text, "status") or "?"
        summary = _xml_tag(text, "summary") or "(no summary)"
        usage_block = _xml_tag(text, "usage") or ""
        duration_ms = _xml_tag(usage_block, "duration-ms")
        duration_str = ""
        if duration_ms and duration_ms.isdigit():
            d = int(duration_ms)
            duration_str = f" ({d/1000:.1f}s)" if d >= 1000 else f" ({d}ms)"

        if status == "completed":
            icon, color = "📬 ✓", GREEN
        elif status == "failed":
            icon, color = "📬 ✗", RED
        elif status == "killed":
            icon, color = "📬 ⊘", YELLOW
        else:
            icon, color = "📬 ?", GREY

        return (
            f"\n{color}{icon} {BOLD}{task_id}{RESET}{color} "
            f"{summary}{duration_str}{RESET}\n"
        )

    def _render_compaction(self, content) -> str:
        """Render a compaction event as a dim, unmissable banner so the
        user can see the context was summarized."""
        text = content if isinstance(content, str) else str(content)
        # Strip the marker prefix for display.
        detail = text.replace("[compaction]", "", 1).strip()
        return (
            f"\n{YELLOW}🗜  context compacted{RESET}{GREY} — {detail}{RESET}\n"
        )

    # ── content helpers ─────────────────────────────────────────────────

    def _extract_text(self, content) -> str:
        """AIMessageChunk.content can be str or list of blocks; we only
        care about the plain-text portion for streaming display."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return ""

    def _args(self, args, max_len: int = 60) -> str:
        """One-line preview of tool arguments."""
        s = str(args).replace("\n", " ")
        return s if len(s) <= max_len else s[:max_len] + "…"

    def _summarize(self, content, max_len: int = 80) -> str:
        """One-line preview of a tool result."""
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        s = str(content).replace("\n", " ").strip()
        return s if len(s) <= max_len else s[:max_len] + "…"

    def _agent_prefix(self, agent_id) -> str:
        """Prefix to attribute output to a specific agent (for subagents).

        Returns empty string in the current event model — the parent's
        renderer only sees the parent's events; subagent internals don't
        cross the boundary. Kept as an extension point for when we add
        cross-agent event piping (e.g., live transcript of a background
        child).
        """
        if agent_id is None:
            return ""
        return f"{CYAN}[{agent_id[:8]}] {RESET}"


# ── Internal helpers ────────────────────────────────────────────────────

class _CallInfo:
    """Per-tool-call state kept across one render() pass. Tracks what
    each tool_use_id was so the result message can be rendered with the
    right styling (in particular: spawn vs. ordinary tool, background
    vs. sync, and which subagent_type was requested)."""
    __slots__ = ("idx", "name", "is_spawn", "is_background", "subagent_type")

    def __init__(
        self,
        idx: int,
        name: str,
        is_spawn: bool,
        is_background: bool,
        subagent_type: str | None = None,
    ):
        self.idx = idx
        self.name = name
        self.is_spawn = is_spawn
        self.is_background = is_background
        # Only meaningful when is_spawn is True. Used to render the
        # result line as "Explore → …" rather than the generic "Agent → …".
        self.subagent_type = subagent_type


# Matches the agent-id ("Spawned background agent abc123…") that
# SpawnAgentTool._arun returns when run_in_background=True.
_SPAWNED_ID_RE = re.compile(r"Spawned background agent (\S+)")


def _extract_spawned_id(content) -> str | None:
    """Pull the agent_id out of a 'Spawned background agent X …' string."""
    text = content if isinstance(content, str) else str(content)
    m = _SPAWNED_ID_RE.search(text)
    return m.group(1) if m else None


def _looks_like_compaction(content) -> bool:
    """True if a SystemMessage content carries the compaction marker."""
    text = content if isinstance(content, str) else str(content)
    return text.lstrip().startswith("[compaction]")


def _looks_like_notification(content) -> bool:
    """True if a HumanMessage's content looks like a task-notification.

    Drained-inbox messages are the only HumanMessages that flow out of
    query() — user-typed prompts come in from outside the loop — but
    we still check the prefix defensively to avoid mis-rendering future
    synthetic HumanMessages from other sources.
    """
    text = content if isinstance(content, str) else str(content)
    return text.lstrip().startswith("<task-notification>")


def _xml_tag(text: str, tag: str) -> str | None:
    """Return the text inside <tag>…</tag>, or None if not found.

    Deliberately a one-liner regex rather than a full XML parser:
    notification bodies are produced by our own formatter, so we know
    the shape. If we ever need nested/repeated/attributed tags, switch
    to ElementTree.
    """
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1) if m else None


# ── Subagent event observer ─────────────────────────────────────────────
#
# Function installed into CURRENT_SUBAGENT_EVENT_SINK so the CLI sees
# what each subagent is doing internally — its tool calls and tool
# results — rather than only the final answer. Prints to stdout directly
# (not through the parent's Renderer stream, since subagent events fire
# while the parent's renderer is paused inside tool.ainvoke).
#
# Format keeps it compact and attributable:
#
#     [Explore a3a577b…]   → Glob({'pattern': '**/*.py'})
#     [Explore a3a577b…]     ✓ → 47 matches: /src/main.py, …
#
# When multiple background agents run in parallel, the prefix lets the
# user disentangle interleaved output. Skips streaming chunks (would be
# too noisy with N parallel agents); shows tool calls + tool results.

def print_subagent_event(ctx, event) -> None:
    """Pretty-print a subagent's tool calls and results to stdout.

    Suitable for installing in CURRENT_SUBAGENT_EVENT_SINK. Defensively
    swallows any formatting error — observation must not break the
    child's actual work.
    """
    try:
        # Look up subagent_type from the registry — it's per-entry,
        # set when SpawnAgentTool._arun registered the child.
        from agent_runtime.registry import AGENT_REGISTRY
        entry = AGENT_REGISTRY.get(ctx.agent_id)
        label = entry.subagent_type if entry and entry.subagent_type else "subagent"
        short_id = ctx.agent_id[:8]
        prefix = f"{CYAN}[{label} {short_id}]{RESET}"

        # Strict type check: AIMessageChunk is a SUBCLASS of AIMessage,
        # so isinstance() would match streaming chunks too — and those
        # carry partial / accumulating tool_calls with empty args. We
        # only want to fire on the final accumulated AIMessage.
        if type(event) is AIMessage and event.tool_calls:
            for tc in event.tool_calls:
                args_str = str(tc.get("args", ""))
                if len(args_str) > 70:
                    args_str = args_str[:70] + "…"
                print(
                    f"  {prefix} {GREY}→ {tc.get('name', '?')}({args_str}){RESET}",
                    flush=True,
                )

        elif isinstance(event, ToolMessage):
            marker = "✗" if event.status == "error" else "✓"
            color = RED if event.status == "error" else GREEN
            content = event.content
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            preview = str(content).replace("\n", " ").strip()
            if len(preview) > 80:
                preview = preview[:80] + "…"
            print(f"  {prefix} {color}  {marker} → {preview}{RESET}", flush=True)

        # Skip AIMessageChunk (streaming text) and HumanMessage (drained
        # notifications inside the subagent — rare, would be from
        # grandchildren). Keeps output focused on the tool activity.
    except Exception:                               # noqa: BLE001
        # Never let a print bug bring down the agent loop.
        pass
