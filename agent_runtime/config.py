"""Runtime configuration — every tunable / magic number lives here.

Defaults are sensible for local development. Override at the process
boundary by:
  - Setting environment variables (read once at import; see Config.from_env)
  - Mutating CONFIG fields before any Session is constructed
  - Constructing your own Config instance and passing it explicitly (when
    we add per-engine config support — not yet wired)

Adding a new tunable? Add it as a field here, give it a clear default
and docstring, then reference `CONFIG.your_field` from the call site.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var. Unset → default; 1/true/yes/on → True."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Config:
    """All runtime tunables in one place."""

    # ── Concurrency ─────────────────────────────────────────────────────
    max_concurrent_tools: int = 10
    """Max in-flight concurrency-safe tools per assistant turn.
    Bounds API rate, file handles, and other shared resources."""

    # ── Loop limits ─────────────────────────────────────────────────────
    max_turns_default: int = 50
    """Default cap on iterations of the agent loop per submit_message call.
    Safety guard against runaway loops; override per-call via RunContext."""

    # ── Cancellation ────────────────────────────────────────────────────
    kill_grace_period_seconds: float = 5.0
    """How long to wait for an agent to exit cleanly after firing its
    AbortController before falling back to forceful task.cancel()."""

    # ── Tool execution ──────────────────────────────────────────────────
    tool_timeout_seconds: float | None = None
    """Optional per-tool wall-clock timeout. None = no timeout."""

    # ── Agent spawning ──────────────────────────────────────────────────
    max_agent_depth: int = 3
    """How deep the agent tree can go before SpawnAgentTool refuses to spawn.
    Depth 0 = main session; 1 = main's subagent; 2 = sub-subagent; etc.
    Bounds recursion explosions."""

    # ── Context compaction ──────────────────────────────────────────────
    # When the conversation approaches the model's context window, the loop
    # summarizes older history into a compact summary (full compaction) and,
    # optionally, mechanically clears stale tool results (microcompaction).
    # All the threshold math lives in agent_runtime/compaction/budget.py and
    # reads these constants. Ported verbatim from Claude Code.

    auto_compact_enabled: bool = True
    """Master switch for proactive auto-compaction. When True, the loop
    summarizes history before a turn once it crosses the threshold. Set
    False (env AGENT_AUTO_COMPACT=0) to disable entirely."""

    model_context_window_default: int = 200_000
    """Fallback context-window size (tokens) for models without a known
    window. 1M-context models ([1m] in the id) override this to 1_000_000
    (see budget.get_context_window_for_model)."""

    max_output_tokens_for_summary: int = 20_000
    """Tokens reserved at the top of the window for the summary OUTPUT, so
    a compaction call can't itself overflow. Sized on CC's p99.99 summary
    length (~17.4k)."""

    compact_max_output_tokens: int = 20_000
    """max_tokens passed to the summarization model call."""

    autocompact_buffer_tokens: int = 13_000
    """Headroom below the effective window where auto-compaction TRIGGERS.
    threshold = effective_window - this."""

    warning_threshold_buffer_tokens: int = 20_000
    """Headroom below effective window where a 'context getting full'
    warning state turns on (informational; headless runtime mostly ignores)."""

    error_threshold_buffer_tokens: int = 20_000
    """Headroom below effective window for the 'error' warning state."""

    manual_compact_buffer_tokens: int = 3_000
    """Headroom below effective window for the HARD blocking limit. If the
    live count crosses (effective - this) without compacting, the turn is
    refused — the prompt genuinely won't fit."""

    max_consecutive_autocompact_failures: int = 3
    """Circuit breaker: after this many failed compaction attempts in a
    row, stop trying (a persistently-failing summarize would otherwise
    burn a call every turn)."""

    microcompact_enabled: bool = False
    """Mechanical tool-result clearing (no LLM). Off by default, matching
    CC (it gates this behind a flag). When on, stale tool results older
    than the gap window get their content replaced with a placeholder."""

    microcompact_gap_minutes: int = 60
    """Microcompaction only fires when the last assistant turn is older
    than this (aligned with the 1h server cache TTL)."""

    microcompact_keep_recent: int = 5
    """Microcompaction always preserves the N most-recent compactable tool
    results; only older ones get cleared."""

    # ── Class methods ──────────────────────────────────────────────────
    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from environment variables, falling back to the
        dataclass defaults for any var that's unset.

        Env vars (all prefixed AGENT_):
            AGENT_MAX_CONCURRENT_TOOLS
            AGENT_MAX_TURNS
            AGENT_KILL_GRACE_SECONDS
            AGENT_TOOL_TIMEOUT_SECONDS
        """
        # `slots=True` makes class-level field access raise, so build an
        # instance with defaults and read from there.
        d = cls()
        return cls(
            max_concurrent_tools=int(
                os.environ.get("AGENT_MAX_CONCURRENT_TOOLS", d.max_concurrent_tools)
            ),
            max_turns_default=int(
                os.environ.get("AGENT_MAX_TURNS", d.max_turns_default)
            ),
            kill_grace_period_seconds=float(
                os.environ.get("AGENT_KILL_GRACE_SECONDS", d.kill_grace_period_seconds)
            ),
            tool_timeout_seconds=(
                float(os.environ["AGENT_TOOL_TIMEOUT_SECONDS"])
                if "AGENT_TOOL_TIMEOUT_SECONDS" in os.environ
                else None
            ),
            max_agent_depth=int(
                os.environ.get("AGENT_MAX_DEPTH", d.max_agent_depth)
            ),
            auto_compact_enabled=_env_bool(
                "AGENT_AUTO_COMPACT", d.auto_compact_enabled
            ),
            microcompact_enabled=_env_bool(
                "AGENT_MICROCOMPACT", d.microcompact_enabled
            ),
        )


# ── Module-level singleton ──────────────────────────────────────────────
# Read env vars at import time. Mutate fields directly if you need to
# change config after import (e.g., in tests).
CONFIG = Config.from_env()
