"""Context-window budget + threshold math.

Ported from Claude Code's tokenBudget.ts / contextAnalysis.ts (the parts
that gate compaction — NOT the `/context` UI grid or the per-turn spend
parser, which are separate features).

The chain of derivations (all constants live on CONFIG so they're tunable
in one obvious place):

    context_window      ← per-model (200k, or 1M for [1m] models)
    effective_window    = context_window - reserve_for_summary_output
    autocompact_thresh  = effective_window - AUTOCOMPACT_BUFFER
    blocking_limit      = effective_window - MANUAL_COMPACT_BUFFER

When the live token count crosses autocompact_thresh we summarize. If we
somehow blow past blocking_limit without compacting, we refuse the turn
(the prompt genuinely won't fit).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from ..config import CONFIG


# 1M-context models advertise it in their id, e.g. "claude-opus-4-8[1m]".
_ONE_M_MARKER = re.compile(r"\[1m\]", re.IGNORECASE)


def get_context_window_for_model(model_name: str | None) -> int:
    """Resolve the context-window size for a model name.

    1,000,000 for models whose id carries the [1m] marker (or 1M beta),
    else CONFIG.model_context_window_default (200k). An explicit env cap
    (CLAUDE_CODE_AUTO_COMPACT_WINDOW) wins if smaller — matches CC.
    """
    name = model_name or ""
    base = (
        1_000_000
        if _ONE_M_MARKER.search(name) and not _env_truthy("CLAUDE_CODE_DISABLE_1M_CONTEXT")
        else CONFIG.model_context_window_default
    )
    cap = os.environ.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW")
    if cap and cap.isdigit():
        base = min(base, int(cap))
    return base


@dataclass(slots=True, frozen=True)
class TokenWarningState:
    """Snapshot of where we sit relative to the thresholds."""
    token_count: int
    context_window: int
    effective_window: int
    autocompact_threshold: int
    blocking_limit: int
    percent_left: float
    is_above_warning: bool
    is_above_error: bool
    is_above_autocompact: bool
    is_at_blocking_limit: bool


def calculate_token_warning_state(
    token_count: int,
    model_name: str | None,
) -> TokenWarningState:
    """Compute all the threshold booleans for a given live token count.

    `is_above_autocompact` is the one auto-compaction gates on;
    `is_at_blocking_limit` is the hard refuse-the-turn guard.
    """
    context_window = get_context_window_for_model(model_name)
    reserve = min(CONFIG.compact_max_output_tokens, CONFIG.max_output_tokens_for_summary)
    effective = context_window - reserve

    autocompact_threshold = _resolve_autocompact_threshold(effective)
    blocking_limit = effective - CONFIG.manual_compact_buffer_tokens
    warning_at = effective - CONFIG.warning_threshold_buffer_tokens
    error_at = effective - CONFIG.error_threshold_buffer_tokens

    percent_left = max(0.0, min(100.0, (effective - token_count) / effective * 100.0))

    return TokenWarningState(
        token_count=token_count,
        context_window=context_window,
        effective_window=effective,
        autocompact_threshold=autocompact_threshold,
        blocking_limit=blocking_limit,
        percent_left=percent_left,
        is_above_warning=token_count >= warning_at,
        is_above_error=token_count >= error_at,
        is_above_autocompact=token_count >= autocompact_threshold,
        is_at_blocking_limit=token_count >= blocking_limit,
    )


def _resolve_autocompact_threshold(effective_window: int) -> int:
    """effective - AUTOCOMPACT_BUFFER, with an optional percentage env
    override (CLAUDE_AUTOCOMPACT_PCT_OVERRIDE, 1..100) that can only LOWER
    the threshold (never raise it past the buffer-based one)."""
    buffer_based = effective_window - CONFIG.autocompact_buffer_tokens
    pct_raw = os.environ.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE")
    if pct_raw:
        try:
            pct = int(pct_raw)
        except ValueError:
            pct = 0
        if 0 < pct <= 100:
            pct_based = (effective_window * pct) // 100
            return min(pct_based, buffer_based)
    return buffer_based


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
